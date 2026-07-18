use std::sync::{Mutex, OnceLock};

use calyx_forge::{CudaOlapContext, ForgeError, Result};

static CONTEXT: OnceLock<Mutex<Option<CudaOlapContext>>> = OnceLock::new();

pub(crate) fn with_context<T>(operation: impl FnOnce(&CudaOlapContext) -> Result<T>) -> Result<T> {
    let mut guard = CONTEXT
        .get_or_init(|| Mutex::new(None))
        .lock()
        .map_err(|_| ForgeError::GpuError {
            detail: "Aster OLAP CUDA context lock is poisoned".to_string(),
            remediation: "restart the failed process before issuing another large OLAP operation"
                .to_string(),
        })?;
    if guard.is_none() {
        *guard = Some(CudaOlapContext::new(0)?);
    }
    operation(guard.as_ref().expect("OLAP context initialized"))
}
