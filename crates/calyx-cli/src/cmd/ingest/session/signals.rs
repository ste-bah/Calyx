use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use signal_hook::consts::{SIGINT, SIGTERM};

static INGEST_SIGNAL: OnceLock<Result<Arc<AtomicUsize>, String>> = OnceLock::new();

pub(super) fn install() -> Result<(), String> {
    match INGEST_SIGNAL.get_or_init(|| {
        let signal = Arc::new(AtomicUsize::new(0));
        signal_hook::flag::register_usize(SIGINT, Arc::clone(&signal), SIGINT as usize)
            .map_err(|error| format!("install SIGINT ingest checkpoint handler: {error}"))?;
        signal_hook::flag::register_usize(SIGTERM, Arc::clone(&signal), SIGTERM as usize)
            .map_err(|error| format!("install SIGTERM ingest checkpoint handler: {error}"))?;
        Ok(signal)
    }) {
        Ok(_) => Ok(()),
        Err(error) => Err(error.clone()),
    }
}

pub(super) fn pending() -> Option<i32> {
    INGEST_SIGNAL
        .get()
        .and_then(|result| result.as_ref().ok())
        .map(|signal| signal.load(Ordering::SeqCst))
        .filter(|signal| *signal != 0)
        .and_then(|signal| i32::try_from(signal).ok())
}

pub(super) fn name(signal: i32) -> String {
    match signal {
        SIGINT => "SIGINT".to_string(),
        SIGTERM => "SIGTERM".to_string(),
        other => format!("signal-{other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handlers_install_and_expose_exact_signal_names() {
        install().expect("install ingest signal handlers");
        assert_eq!(name(SIGINT), "SIGINT");
        assert_eq!(name(SIGTERM), "SIGTERM");
        assert_eq!(pending(), None);
    }
}
