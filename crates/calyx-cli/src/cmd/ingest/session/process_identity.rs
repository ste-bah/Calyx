use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(in crate::cmd::ingest) struct IngestProcessIdentity {
    pub(super) host_name: String,
    pub(super) boot_id: Option<String>,
    pub(super) process_id: u32,
    pub(super) process_start: u64,
    pub(super) process_start_kind: String,
    pub(super) executable: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum OwnerInspection {
    Alive,
    Dead(String),
    Unknown(String),
}

#[derive(Debug)]
enum IdentityReadError {
    NotFound(String),
    Unknown(String),
}

pub(super) fn current_process_identity() -> Result<IngestProcessIdentity, String> {
    let host_name = host_name()?;
    let boot_id = boot_id()?;
    let (process_start, process_start_kind, executable) = process_details(std::process::id())
        .map_err(|error| match error {
            IdentityReadError::NotFound(detail) | IdentityReadError::Unknown(detail) => detail,
        })?;
    Ok(IngestProcessIdentity {
        host_name,
        boot_id,
        process_id: std::process::id(),
        process_start,
        process_start_kind,
        executable,
    })
}

pub(super) fn inspect_owner(recorded: &IngestProcessIdentity) -> OwnerInspection {
    let local_host = match host_name() {
        Ok(value) => value,
        Err(error) => {
            return OwnerInspection::Unknown(format!(
                "cannot establish local host identity: {error}"
            ));
        }
    };
    if recorded.host_name != local_host {
        return OwnerInspection::Unknown(format!(
            "session owner host {} differs from local host {local_host}",
            recorded.host_name
        ));
    }

    let local_boot_id = match boot_id() {
        Ok(value) => value,
        Err(error) => {
            return OwnerInspection::Unknown(format!(
                "cannot establish local boot identity: {error}"
            ));
        }
    };
    if recorded.boot_id != local_boot_id {
        return OwnerInspection::Dead(format!(
            "session owner boot identity {:?} differs from current boot identity {local_boot_id:?}",
            recorded.boot_id
        ));
    }

    match process_details(recorded.process_id) {
        Ok((start, start_kind, executable)) => {
            if start != recorded.process_start
                || start_kind != recorded.process_start_kind
                || executable != recorded.executable
            {
                OwnerInspection::Dead(format!(
                    "pid {} was reused or changed identity: recorded start={} kind={} executable={:?}; current start={} kind={} executable={:?}",
                    recorded.process_id,
                    recorded.process_start,
                    recorded.process_start_kind,
                    recorded.executable,
                    start,
                    start_kind,
                    executable
                ))
            } else {
                OwnerInspection::Alive
            }
        }
        Err(IdentityReadError::NotFound(detail)) => OwnerInspection::Dead(detail),
        Err(IdentityReadError::Unknown(detail)) => OwnerInspection::Unknown(detail),
    }
}

pub(super) fn inspect_legacy_pid(process_id: u32) -> OwnerInspection {
    match process_details(process_id) {
        Err(IdentityReadError::NotFound(detail)) => OwnerInspection::Dead(detail),
        Ok(_) => OwnerInspection::Unknown(format!(
            "legacy session records pid {process_id} without process-start identity; a live or reused pid cannot be distinguished safely"
        )),
        Err(IdentityReadError::Unknown(detail)) => OwnerInspection::Unknown(detail),
    }
}

#[cfg(target_os = "linux")]
fn host_name() -> Result<String, String> {
    read_trimmed("/proc/sys/kernel/hostname", "Linux host name")
}

#[cfg(target_os = "linux")]
fn boot_id() -> Result<Option<String>, String> {
    read_trimmed("/proc/sys/kernel/random/boot_id", "Linux boot id").map(Some)
}

#[cfg(target_os = "linux")]
fn read_trimmed(path: &str, label: &str) -> Result<String, String> {
    let value = std::fs::read_to_string(path)
        .map_err(|error| format!("read {label} from {path}: {error}"))?;
    let value = value.trim();
    if value.is_empty() {
        return Err(format!("{label} at {path} is empty"));
    }
    Ok(value.to_string())
}

#[cfg(target_os = "linux")]
fn process_details(pid: u32) -> Result<(u64, String, String), IdentityReadError> {
    let stat_path = format!("/proc/{pid}/stat");
    let stat = std::fs::read(&stat_path).map_err(|error| classify_io(&stat_path, error))?;
    let end = stat
        .windows(2)
        .rposition(|window| window == b") ")
        .ok_or_else(|| {
            IdentityReadError::Unknown(format!(
                "parse process identity {stat_path}: malformed comm field"
            ))
        })?;
    let fields = stat[end + 2..]
        .split(|byte| *byte == b' ')
        .collect::<Vec<_>>();
    // proc_pid_stat(5): field 22 is starttime. This tail begins at field 3.
    let start = fields
        .get(19)
        .and_then(|raw| std::str::from_utf8(raw).ok())
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .filter(|value| *value != 0)
        .ok_or_else(|| {
            IdentityReadError::Unknown(format!(
                "parse process identity {stat_path}: missing or invalid field 22 starttime"
            ))
        })?;
    let exe_path = format!("/proc/{pid}/exe");
    let executable = std::fs::read_link(&exe_path)
        .map_err(|error| classify_io(&exe_path, error))?
        .display()
        .to_string();
    // Linux appends this marker when the on-disk executable is atomically
    // replaced while the exact same process remains alive. It is filesystem
    // state, not a process-identity change; boot id + pid + start ticks remain
    // the PID-reuse proof.
    let executable = executable
        .strip_suffix(" (deleted)")
        .unwrap_or(&executable)
        .to_string();
    Ok((start, "linux_boot_ticks".to_string(), executable))
}

#[cfg(target_os = "linux")]
fn classify_io(path: &str, error: std::io::Error) -> IdentityReadError {
    if error.kind() == std::io::ErrorKind::NotFound {
        IdentityReadError::NotFound(format!(
            "session owner process is absent: read {path}: {error}"
        ))
    } else {
        IdentityReadError::Unknown(format!(
            "cannot inspect session owner process at {path}: {error}"
        ))
    }
}

#[cfg(windows)]
fn host_name() -> Result<String, String> {
    use windows_sys::Win32::System::WindowsProgramming::GetComputerNameW;

    let mut buffer = [0u16; 256];
    let mut length = u32::try_from(buffer.len()).expect("host buffer length fits u32");
    // SAFETY: `buffer` is writable for `length` UTF-16 elements and `length`
    // remains valid for the call.
    let result = unsafe { GetComputerNameW(buffer.as_mut_ptr(), &mut length) };
    if result == 0 {
        return Err(format!(
            "GetComputerNameW failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    String::from_utf16(&buffer[..length as usize])
        .map_err(|error| format!("decode GetComputerNameW result: {error}"))
}

#[cfg(windows)]
fn boot_id() -> Result<Option<String>, String> {
    // Windows process creation FILETIME is an absolute timestamp, so it is a
    // stable PID-reuse discriminator without a separate boot-scoped value.
    Ok(None)
}

#[cfg(windows)]
fn process_details(pid: u32) -> Result<(u64, String, String), IdentityReadError> {
    use windows_sys::Win32::Foundation::{CloseHandle, ERROR_INVALID_PARAMETER, FILETIME};
    use windows_sys::Win32::System::Threading::{
        GetProcessTimes, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, QueryFullProcessImageNameW,
    };

    // SAFETY: the access mask and pid are plain values. The returned handle is
    // checked and closed on every subsequent path.
    let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
    if handle.is_null() {
        let error = std::io::Error::last_os_error();
        return if error.raw_os_error() == Some(ERROR_INVALID_PARAMETER as i32) {
            Err(IdentityReadError::NotFound(format!(
                "session owner process pid {pid} is absent: OpenProcess: {error}"
            )))
        } else {
            Err(IdentityReadError::Unknown(format!(
                "cannot inspect session owner pid {pid}: OpenProcess: {error}"
            )))
        };
    }

    let result = (|| {
        let mut creation = FILETIME::default();
        let mut exit = FILETIME::default();
        let mut kernel = FILETIME::default();
        let mut user = FILETIME::default();
        // SAFETY: `handle` is open and every FILETIME pointer is valid.
        if unsafe { GetProcessTimes(handle, &mut creation, &mut exit, &mut kernel, &mut user) } == 0
        {
            return Err(IdentityReadError::Unknown(format!(
                "cannot inspect session owner pid {pid}: GetProcessTimes: {}",
                std::io::Error::last_os_error()
            )));
        }
        let start = (u64::from(creation.dwHighDateTime) << 32) | u64::from(creation.dwLowDateTime);
        if start == 0 {
            return Err(IdentityReadError::Unknown(format!(
                "cannot inspect session owner pid {pid}: zero process creation FILETIME"
            )));
        }

        let mut path = vec![0u16; 32_768];
        let mut path_len = u32::try_from(path.len()).expect("process path buffer fits u32");
        // SAFETY: `handle` is open, `path` is writable for `path_len` UTF-16
        // elements, and `path_len` remains valid for the call.
        if unsafe { QueryFullProcessImageNameW(handle, 0, path.as_mut_ptr(), &mut path_len) } == 0 {
            return Err(IdentityReadError::Unknown(format!(
                "cannot inspect session owner pid {pid}: QueryFullProcessImageNameW: {}",
                std::io::Error::last_os_error()
            )));
        }
        let executable = String::from_utf16(&path[..path_len as usize]).map_err(|error| {
            IdentityReadError::Unknown(format!(
                "decode executable path for session owner pid {pid}: {error}"
            ))
        })?;
        Ok((start, "windows_filetime_100ns".to_string(), executable))
    })();
    // SAFETY: `handle` came from a successful OpenProcess and is closed once.
    unsafe { CloseHandle(handle) };
    result
}

#[cfg(all(unix, not(target_os = "linux")))]
fn host_name() -> Result<String, String> {
    Err("ingest process identity is unsupported on this Unix platform".to_string())
}

#[cfg(all(unix, not(target_os = "linux")))]
fn boot_id() -> Result<Option<String>, String> {
    Err("ingest boot identity is unsupported on this Unix platform".to_string())
}

#[cfg(all(unix, not(target_os = "linux")))]
fn process_details(_pid: u32) -> Result<(u64, String, String), IdentityReadError> {
    Err(IdentityReadError::Unknown(
        "ingest process identity is unsupported on this Unix platform".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_process_identity_is_live_and_start_bound() {
        let identity = current_process_identity().expect("current process identity");
        assert_eq!(identity.process_id, std::process::id());
        assert_ne!(identity.process_start, 0);
        assert!(!identity.host_name.is_empty());
        assert!(!identity.executable.is_empty());
        assert_eq!(inspect_owner(&identity), OwnerInspection::Alive);

        let mut reused = identity;
        reused.process_start = reused.process_start.saturating_add(1);
        assert!(matches!(inspect_owner(&reused), OwnerInspection::Dead(_)));
    }

    #[test]
    fn remote_host_is_unknown_and_never_declared_dead() {
        let mut identity = current_process_identity().expect("current process identity");
        identity.host_name.push_str("-remote");
        assert!(matches!(
            inspect_owner(&identity),
            OwnerInspection::Unknown(_)
        ));
    }
}
