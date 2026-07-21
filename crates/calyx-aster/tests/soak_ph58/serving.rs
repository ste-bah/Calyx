use calyx_aster::cf::ColumnFamily;
use calyx_aster::vault::AsterVault;
use calyx_core::{Clock, Ts};
use serde_json::{Value, json};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Debug)]
pub(crate) struct SharedClock {
    now: Arc<AtomicU64>,
}

impl SharedClock {
    pub(crate) fn new(now: Ts) -> Self {
        Self {
            now: Arc::new(AtomicU64::new(now)),
        }
    }

    pub(crate) fn set(&self, now: Ts) {
        self.now.store(now, Ordering::Relaxed);
    }
}

impl Clock for SharedClock {
    fn now(&self) -> Ts {
        self.now.load(Ordering::Relaxed)
    }
}

pub(crate) fn assert_janitor_harness_contract(repo_root: &Path, harness: &str, filter: &str) {
    let path = repo_root.join("docs/audit/issue1546_test_target_migration.json");
    let migration: Value =
        serde_json::from_slice(&std::fs::read(path).expect("read integration migration map"))
            .expect("decode integration migration map");
    let contract = &migration["calyx-anneal:issue486_janitor_fsv"];
    assert_eq!(contract["harness"], harness);
    assert_eq!(contract["module_filter"], "issue486_janitor_fsv");
    assert!(filter.starts_with("issue486_janitor_fsv::"));
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct LiveReadbackCount {
    pub(crate) visible: usize,
    pub(crate) missing: usize,
}

impl LiveReadbackCount {
    pub(crate) fn to_json(self) -> Value {
        json!({
            "visible": self.visible,
            "missing": self.missing,
        })
    }
}

pub(crate) fn live_base_readback_count<C: Clock>(
    vault: &AsterVault<C>,
    start: u64,
    end: u64,
) -> LiveReadbackCount {
    let snapshot = vault.latest_seq();
    let mut visible = 0usize;
    let mut missing = 0usize;
    for id in start..end {
        let key = format!("key-{id:05}");
        match vault
            .read_cf_at(snapshot, ColumnFamily::Base, key.as_bytes())
            .expect("serving readback")
        {
            Some(_) => visible += 1,
            None => missing += 1,
        }
    }
    LiveReadbackCount { visible, missing }
}
