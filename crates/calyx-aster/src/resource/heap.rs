//! Process resident-set probe over the operating system's physical source.

use calyx_core::{CalyxError, Result};

/// Stable code for resource probes that cannot run on this host.
pub const CALYX_RESOURCE_PROBE_UNAVAILABLE: &str = "CALYX_RESOURCE_PROBE_UNAVAILABLE";

pub(crate) fn probe_unavailable(message: impl Into<String>) -> CalyxError {
    CalyxError {
        code: CALYX_RESOURCE_PROBE_UNAVAILABLE,
        message: message.into(),
        remediation: "run resource_status on a Linux host with /proc mounted",
    }
}

/// Reads the resident set size of this process in bytes.
///
/// On Linux the source of truth is the kernel `VmRSS` line in
/// `/proc/self/status` (`proc_pid_status(5)`). On Windows it is the current
/// process `WorkingSetSize` returned by `K32GetProcessMemoryInfo`. Unsupported
/// hosts and failed kernel calls fail closed with
/// `CALYX_RESOURCE_PROBE_UNAVAILABLE`.
pub fn heap_rss_bytes() -> Result<u64> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/self/status")
            .map_err(|error| probe_unavailable(format!("read /proc/self/status: {error}")))?;
        parse_vm_rss_bytes(&text)
    }
    #[cfg(target_os = "windows")]
    {
        windows_working_set_bytes()
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        Err(probe_unavailable(
            "heap RSS probe has no authoritative implementation for this operating system",
        ))
    }
}

#[cfg(target_os = "windows")]
fn windows_working_set_bytes() -> Result<u64> {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let mut counters = PROCESS_MEMORY_COUNTERS {
        cb: u32::try_from(std::mem::size_of::<PROCESS_MEMORY_COUNTERS>())
            .expect("PROCESS_MEMORY_COUNTERS size fits u32"),
        ..PROCESS_MEMORY_COUNTERS::default()
    };
    // SAFETY: GetCurrentProcess returns the documented pseudo-handle for this
    // process, and `counters` is a correctly sized, writable structure.
    let succeeded =
        unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    if succeeded == 0 {
        return Err(probe_unavailable(format!(
            "K32GetProcessMemoryInfo(current process): {}",
            std::io::Error::last_os_error()
        )));
    }
    u64::try_from(counters.WorkingSetSize).map_err(|error| {
        probe_unavailable(format!(
            "convert Windows WorkingSetSize {} to u64: {error}",
            counters.WorkingSetSize
        ))
    })
}

/// Parses the `VmRSS:` line of a `/proc/<pid>/status` document into bytes.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn parse_vm_rss_bytes(status_text: &str) -> Result<u64> {
    for line in status_text.lines() {
        let Some(rest) = line.strip_prefix("VmRSS:") else {
            continue;
        };
        let mut fields = rest.split_whitespace();
        let value = fields
            .next()
            .ok_or_else(|| probe_unavailable("VmRSS line has no value field"))?;
        let unit = fields
            .next()
            .ok_or_else(|| probe_unavailable("VmRSS line has no unit field"))?;
        if unit != "kB" {
            return Err(probe_unavailable(format!(
                "VmRSS unit {unit:?} is not kB; refusing to guess a scale"
            )));
        }
        let kib = value
            .parse::<u64>()
            .map_err(|error| probe_unavailable(format!("parse VmRSS value {value:?}: {error}")))?;
        return Ok(kib.saturating_mul(1024));
    }
    Err(probe_unavailable(
        "VmRSS line not found in /proc/self/status",
    ))
}
