use std::mem::size_of;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::sync::{Mutex, OnceLock};

use calyx_core::Result;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};

use super::telemetry::TELEMETRY;
use crate::error::{
    CALYX_INDEX_INVALID_PARAMS, CALYX_SEXTANT_GPU_SERVING_UNAVAILABLE, sextant_error,
};
use crate::index::diskann::cagra_dataset::{self, DatasetPayload};

enum DatasetStorage {
    I8(CudaSlice<i8>),
    F32(CudaSlice<f32>),
}

type SharedDevice = (Arc<CudaContext>, Arc<CudaStream>);
static DEVICE: OnceLock<Mutex<Option<SharedDevice>>> = OnceLock::new();
static UPLOAD: Mutex<()> = Mutex::new(());

pub(super) struct Asset {
    // Buffers drop while their stream/context remain live.
    dataset: DatasetStorage,
    global_ids: Option<CudaSlice<u64>>,
    stream: Arc<CudaStream>,
    _context: Arc<CudaContext>,
    rows: usize,
    dim: usize,
    base_bytes: u64,
    global_ids_digest: Option<[u8; 32]>,
}

#[derive(Clone, Copy)]
pub(super) struct RegionDeviceView {
    pub(super) dataset: u64,
    pub(super) dataset_dtype: i32,
    pub(super) global_ids: u64,
    pub(super) rows: i32,
    pub(super) dim: i32,
    pub(super) row_stride: i64,
    pub(super) column_stride: i64,
}

// CUDA state is only accessed through the cache's `Mutex<Asset>`.
unsafe impl Send for Asset {}

impl Asset {
    pub(super) fn pool_diagnostics() -> [u64; 4] {
        use cudarc::driver::sys::CUmemPool_attribute::*;

        let Some(device) = DEVICE.get() else {
            return [0; 4];
        };
        let Ok(device) = device.lock() else {
            return [0; 4];
        };
        let Some((context, _)) = device.as_ref() else {
            return [0; 4];
        };
        let Ok(pool) =
            (unsafe { cudarc::driver::result::device::get_mem_pool(context.cu_device()) })
        else {
            return [0; 4];
        };
        [
            pool_attribute(pool, CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT),
            pool_attribute(pool, CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH),
            pool_attribute(pool, CU_MEMPOOL_ATTR_USED_MEM_CURRENT),
            pool_attribute(pool, CU_MEMPOOL_ATTR_USED_MEM_HIGH),
        ]
    }

    pub(super) fn reclaim_unused() -> Result<()> {
        if DEVICE.get().is_none() {
            return Ok(());
        }
        let (context, stream) = shared_device()?;
        let _upload = UPLOAD
            .lock()
            .map_err(|_| unavailable("partitioned CUDA upload lock poisoned"))?;
        stream.synchronize().map_err(cuda_error("eviction sync"))?;
        let pool = unsafe { cudarc::driver::result::device::get_mem_pool(context.cu_device()) }
            .map_err(cuda_error("memory pool lookup"))?;
        unsafe { cudarc::driver::result::mem_pool::trim_to(pool, 0) }
            .map_err(cuda_error("memory pool trim"))
    }

    pub(super) fn load(path: &Path, global_ids: &[u64], digest: [u8; 32]) -> Result<Self> {
        let payload = cagra_dataset::load(path)?;
        let payload_rows = match &payload {
            DatasetPayload::I8(header, _) | DatasetPayload::F32(header, _) => header.rows,
        };
        if payload_rows != global_ids.len() {
            return Err(invalid(format!(
                "partitioned global id count {} != dataset rows {payload_rows}",
                global_ids.len()
            )));
        }
        let (context, stream) = shared_device()?;
        let _upload = UPLOAD
            .lock()
            .map_err(|_| unavailable("partitioned CUDA upload lock poisoned"))?;
        let (dataset, rows, dim, base_bytes) = match payload {
            DatasetPayload::I8(header, values) => {
                let bytes = u64::try_from(values.len()).unwrap_or(u64::MAX);
                let dataset = stream
                    .clone_htod(&values)
                    .map_err(cuda_error("i8 dataset upload"))?;
                TELEMETRY
                    .partitioned_i8_dataset_loads
                    .fetch_add(1, Ordering::Relaxed);
                (DatasetStorage::I8(dataset), header.rows, header.dim, bytes)
            }
            DatasetPayload::F32(header, values) => {
                let bytes = u64::try_from(values.len().saturating_mul(size_of::<f32>()))
                    .unwrap_or(u64::MAX);
                let dataset = stream
                    .clone_htod(&values)
                    .map_err(cuda_error("f32 dataset upload"))?;
                TELEMETRY
                    .partitioned_f32_dataset_loads
                    .fetch_add(1, Ordering::Relaxed);
                (DatasetStorage::F32(dataset), header.rows, header.dim, bytes)
            }
        };
        let global_ids_device = stream
            .clone_htod(global_ids)
            .map_err(cuda_error("global id upload"))?;
        stream
            .synchronize()
            .map_err(cuda_error("dataset and global id upload sync"))?;
        let global_id_bytes = u64::try_from(global_ids.len().saturating_mul(8)).unwrap_or(u64::MAX);
        TELEMETRY.h2d_bytes.fetch_add(
            base_bytes.saturating_add(global_id_bytes),
            Ordering::Relaxed,
        );
        Ok(Self {
            dataset,
            global_ids: Some(global_ids_device),
            stream,
            _context: context,
            rows,
            dim,
            base_bytes,
            global_ids_digest: Some(digest),
        })
    }

    pub(super) fn resident_bytes(&self) -> u64 {
        self.base_bytes
            .saturating_add(self.global_ids.as_ref().map_or(0, |ids| {
                u64::try_from(ids.len().saturating_mul(8)).unwrap_or(u64::MAX)
            }))
    }

    pub(super) fn global_ids_required(
        &self,
        global_ids: &[u64],
        digest: [u8; 32],
    ) -> Result<Option<u64>> {
        if global_ids.len() != self.rows {
            return Err(invalid(format!(
                "partitioned global id count {} != dataset rows {}",
                global_ids.len(),
                self.rows
            )));
        }
        match self.global_ids_digest {
            Some(existing) if existing != digest => Err(invalid(
                "partitioned global id generation changed for CUDA dataset",
            )),
            Some(_) => Ok(None),
            None => Ok(Some(self.base_bytes.saturating_add(
                u64::try_from(global_ids.len().saturating_mul(8)).unwrap_or(u64::MAX),
            ))),
        }
    }

    pub(super) fn region(
        &mut self,
        global_ids: &[u64],
        digest: [u8; 32],
    ) -> Result<RegionDeviceView> {
        if self.global_ids_required(global_ids, digest)?.is_some() {
            let _upload = UPLOAD
                .lock()
                .map_err(|_| unavailable("partitioned CUDA upload lock poisoned"))?;
            self.global_ids = Some(
                self.stream
                    .clone_htod(global_ids)
                    .map_err(cuda_error("global id upload"))?,
            );
            self.stream
                .synchronize()
                .map_err(cuda_error("global id sync"))?;
            self.global_ids_digest = Some(digest);
            TELEMETRY.h2d_bytes.fetch_add(
                u64::try_from(global_ids.len().saturating_mul(8)).unwrap_or(u64::MAX),
                Ordering::Relaxed,
            );
        }
        let (dataset, dataset_dtype) = match &self.dataset {
            DatasetStorage::I8(values) => (values.device_ptr(&self.stream).0, 1),
            DatasetStorage::F32(values) => (values.device_ptr(&self.stream).0, 0),
        };
        let global_ids = self
            .global_ids
            .as_ref()
            .expect("partitioned global ids")
            .device_ptr(&self.stream)
            .0;
        Ok(RegionDeviceView {
            dataset,
            dataset_dtype,
            global_ids,
            rows: i32::try_from(self.rows).map_err(|_| invalid("dataset rows exceed i32"))?,
            dim: i32::try_from(self.dim).map_err(|_| invalid("dataset dim exceeds i32"))?,
            row_stride: i64::try_from(self.dim).map_err(|_| invalid("dataset dim exceeds i64"))?,
            column_stride: 1,
        })
    }
}

fn pool_attribute(
    pool: cudarc::driver::sys::CUmemoryPool,
    attribute: cudarc::driver::sys::CUmemPool_attribute,
) -> u64 {
    let mut value = 0_u64;
    let _ = unsafe {
        cudarc::driver::result::mem_pool::get_attribute(
            pool,
            attribute,
            (&mut value as *mut u64).cast(),
        )
    };
    value
}

fn shared_device() -> Result<SharedDevice> {
    let device = DEVICE.get_or_init(|| Mutex::new(None));
    let mut device = device
        .lock()
        .map_err(|_| unavailable("partitioned CUDA device lock poisoned"))?;
    if device.is_none() {
        let context = CudaContext::new(0).map_err(cuda_error("context init"))?;
        let stream = context.new_stream().map_err(cuda_error("stream init"))?;
        *device = Some((context, stream));
    }
    Ok(device.as_ref().expect("shared device initialized").clone())
}

fn invalid(detail: impl Into<String>) -> calyx_core::CalyxError {
    sextant_error(CALYX_INDEX_INVALID_PARAMS, detail)
}

fn unavailable(detail: impl Into<String>) -> calyx_core::CalyxError {
    sextant_error(CALYX_SEXTANT_GPU_SERVING_UNAVAILABLE, detail)
}

fn cuda_error(
    stage: &'static str,
) -> impl FnOnce(cudarc::driver::DriverError) -> calyx_core::CalyxError {
    move |error| {
        TELEMETRY.failures.fetch_add(1, Ordering::Relaxed);
        sextant_error(
            CALYX_SEXTANT_GPU_SERVING_UNAVAILABLE,
            format!("partitioned CUDA dataset {stage}: {error}"),
        )
    }
}
