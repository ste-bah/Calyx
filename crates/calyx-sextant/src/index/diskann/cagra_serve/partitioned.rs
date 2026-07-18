use std::mem::size_of;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock};

use calyx_core::Result;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

use super::CagraServingMetric;
use super::partition_asset::RegionDeviceView;
use super::telemetry::TELEMETRY;
use crate::error::{
    CALYX_INDEX_DIM_MISMATCH, CALYX_INDEX_INVALID_PARAMS, CALYX_SEXTANT_GPU_SERVING_UNAVAILABLE,
    sextant_error,
};
use crate::index::distance::l2_normalize;

const CUBIN: &[u8] = include_bytes!(env!("SEXTANT_PARTITIONED_REGION_BATCH_CUBIN_PATH"));
const SMALL_K: usize = 32;
const SMALL_BLOCK_THREADS: u32 = 256;

struct Runner {
    small: CudaFunction,
    large: CudaFunction,
    merge: CudaFunction,
    stream: Arc<CudaStream>,
    _context: Arc<CudaContext>,
    dataset_addresses: Option<CudaSlice<u64>>,
    dataset_dtypes: Option<CudaSlice<i32>>,
    global_id_addresses: Option<CudaSlice<u64>>,
    rows: Option<CudaSlice<i32>>,
    row_strides: Option<CudaSlice<i64>>,
    column_strides: Option<CudaSlice<i64>>,
    query: Option<CudaSlice<f32>>,
    region_ids: Option<CudaSlice<u64>>,
    region_distances: Option<CudaSlice<f32>>,
    output_ids: Option<CudaSlice<u64>>,
    output_distances: Option<CudaSlice<f32>>,
}

static RUNNER: OnceLock<Mutex<Option<Runner>>> = OnceLock::new();

pub(super) fn search(
    regions: &[RegionDeviceView],
    query: &[f32],
    metric: CagraServingMetric,
    k: usize,
) -> Result<Vec<(u64, f32)>> {
    let runner = RUNNER.get_or_init(|| Mutex::new(None));
    let mut runner = runner
        .lock()
        .map_err(|_| unavailable("partitioned CUDA runner lock poisoned"))?;
    if runner.is_none() {
        *runner = Some(Runner::new()?);
    }
    runner
        .as_mut()
        .expect("partitioned runner initialized")
        .search(regions, query, metric, k)
}

impl Runner {
    fn new() -> Result<Self> {
        let context = CudaContext::new(0).map_err(cuda_error("context init"))?;
        let stream = context.new_stream().map_err(cuda_error("stream init"))?;
        let module = context
            .load_module(Ptx::from_binary(CUBIN.to_vec()))
            .map_err(cuda_error("CUBIN load"))?;
        Ok(Self {
            small: module
                .load_function("partitioned_region_exact_small")
                .map_err(cuda_error("small exact kernel load"))?,
            large: module
                .load_function("partitioned_region_exact_large")
                .map_err(cuda_error("large exact kernel load"))?,
            merge: module
                .load_function("partitioned_region_merge_topk")
                .map_err(cuda_error("merge kernel load"))?,
            stream,
            _context: context,
            dataset_addresses: None,
            dataset_dtypes: None,
            global_id_addresses: None,
            rows: None,
            row_strides: None,
            column_strides: None,
            query: None,
            region_ids: None,
            region_distances: None,
            output_ids: None,
            output_distances: None,
        })
    }

    fn search(
        &mut self,
        regions: &[RegionDeviceView],
        query: &[f32],
        metric: CagraServingMetric,
        k: usize,
    ) -> Result<Vec<(u64, f32)>> {
        let dim = regions[0].dim;
        if dim <= 0 || query.len() != dim as usize {
            return Err(sextant_error(
                CALYX_INDEX_DIM_MISMATCH,
                format!(
                    "partitioned CUDA query dim {} != region dim {dim}",
                    query.len()
                ),
            ));
        }
        if regions
            .iter()
            .any(|region| region.dim != dim || region.rows <= 0)
        {
            return Err(invalid(
                "partitioned CUDA regions have inconsistent metadata",
            ));
        }
        let region_count = regions.len();
        let pairs = region_count
            .checked_mul(k)
            .ok_or_else(|| invalid("partitioned CUDA output shape overflow"))?;
        self.ensure_buffers(region_count, query.len(), pairs, k)?;
        TELEMETRY
            .partitioned_scratch_bytes
            .store(self.scratch_bytes(), Ordering::Relaxed);
        let datasets = regions
            .iter()
            .map(|region| region.dataset)
            .collect::<Vec<_>>();
        let global_ids = regions
            .iter()
            .map(|region| region.global_ids)
            .collect::<Vec<_>>();
        let dataset_dtypes = regions
            .iter()
            .map(|region| region.dataset_dtype)
            .collect::<Vec<_>>();
        let rows = regions.iter().map(|region| region.rows).collect::<Vec<_>>();
        let row_strides = regions
            .iter()
            .map(|region| region.row_stride)
            .collect::<Vec<_>>();
        let column_strides = regions
            .iter()
            .map(|region| region.column_stride)
            .collect::<Vec<_>>();
        let query = match metric {
            CagraServingMetric::UnitL2 => l2_normalize(query),
            CagraServingMetric::RawL2 => query.to_vec(),
        };
        self.upload(
            &datasets,
            &dataset_dtypes,
            &global_ids,
            &rows,
            &row_strides,
            &column_strides,
            &query,
        )?;

        let region_count_i32 = i32::try_from(region_count)
            .map_err(|_| invalid("partitioned CUDA region count exceeds i32"))?;
        let metric_i32 = match metric {
            CagraServingMetric::RawL2 => 0_i32,
            CagraServingMetric::UnitL2 => 1_i32,
        };
        let k_i32 = i32::try_from(k).map_err(|_| invalid("partitioned CUDA k exceeds i32"))?;
        let exact = if k <= SMALL_K {
            &self.small
        } else {
            &self.large
        };
        let block = if k <= SMALL_K { SMALL_BLOCK_THREADS } else { 1 };
        let mut launch = self.stream.launch_builder(exact);
        unsafe {
            launch
                .arg(self.dataset_addresses.as_ref().expect("dataset addresses"))
                .arg(self.dataset_dtypes.as_ref().expect("dataset dtypes"))
                .arg(
                    self.global_id_addresses
                        .as_ref()
                        .expect("global id addresses"),
                )
                .arg(self.rows.as_ref().expect("region rows"))
                .arg(self.row_strides.as_ref().expect("row strides"))
                .arg(self.column_strides.as_ref().expect("column strides"))
                .arg(&region_count_i32)
                .arg(self.query.as_ref().expect("query"))
                .arg(&dim)
                .arg(&metric_i32)
                .arg(&k_i32)
                .arg(self.region_ids.as_mut().expect("region ids"))
                .arg(self.region_distances.as_mut().expect("region distances"))
                .launch(LaunchConfig {
                    grid_dim: (region_count as u32, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(cuda_error("region exact launch"))?;
        let mut merge = self.stream.launch_builder(&self.merge);
        unsafe {
            merge
                .arg(self.region_ids.as_ref().expect("region ids"))
                .arg(self.region_distances.as_ref().expect("region distances"))
                .arg(&region_count_i32)
                .arg(&k_i32)
                .arg(self.output_ids.as_mut().expect("output ids"))
                .arg(self.output_distances.as_mut().expect("output distances"))
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(cuda_error("region merge launch"))?;
        self.stream
            .synchronize()
            .map_err(cuda_error("partitioned search sync"))?;
        self.readback(k)
    }

    #[allow(clippy::too_many_arguments)]
    fn upload(
        &mut self,
        datasets: &[u64],
        dataset_dtypes: &[i32],
        global_ids: &[u64],
        rows: &[i32],
        row_strides: &[i64],
        column_strides: &[i64],
        query: &[f32],
    ) -> Result<()> {
        copy(
            &self.stream,
            datasets,
            &mut self.dataset_addresses,
            "dataset addresses",
        )?;
        copy(
            &self.stream,
            dataset_dtypes,
            &mut self.dataset_dtypes,
            "dataset dtypes",
        )?;
        copy(
            &self.stream,
            global_ids,
            &mut self.global_id_addresses,
            "global id addresses",
        )?;
        copy(&self.stream, rows, &mut self.rows, "region rows")?;
        copy(
            &self.stream,
            row_strides,
            &mut self.row_strides,
            "row strides",
        )?;
        copy(
            &self.stream,
            column_strides,
            &mut self.column_strides,
            "column strides",
        )?;
        copy(&self.stream, query, &mut self.query, "partitioned query")?;
        let bytes = size_of_val(datasets)
            + size_of_val(dataset_dtypes)
            + size_of_val(global_ids)
            + size_of_val(rows)
            + size_of_val(row_strides)
            + size_of_val(column_strides)
            + size_of_val(query);
        TELEMETRY.batches.fetch_add(1, Ordering::Relaxed);
        TELEMETRY.queries.fetch_add(1, Ordering::Relaxed);
        TELEMETRY.query_uploads.fetch_add(1, Ordering::Relaxed);
        TELEMETRY
            .h2d_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
        TELEMETRY
            .partitioned_exact_kernel_launches
            .fetch_add(1, Ordering::Relaxed);
        TELEMETRY
            .partitioned_merge_kernel_launches
            .fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn ensure_buffers(
        &mut self,
        regions: usize,
        query_values: usize,
        pairs: usize,
        k: usize,
    ) -> Result<()> {
        ensure(
            &self.stream,
            &mut self.dataset_addresses,
            regions,
            "dataset addresses",
        )?;
        ensure(
            &self.stream,
            &mut self.dataset_dtypes,
            regions,
            "dataset dtypes",
        )?;
        ensure(
            &self.stream,
            &mut self.global_id_addresses,
            regions,
            "global id addresses",
        )?;
        ensure(&self.stream, &mut self.rows, regions, "region rows")?;
        ensure(&self.stream, &mut self.row_strides, regions, "row strides")?;
        ensure(
            &self.stream,
            &mut self.column_strides,
            regions,
            "column strides",
        )?;
        ensure(&self.stream, &mut self.query, query_values, "query")?;
        ensure(&self.stream, &mut self.region_ids, pairs, "region ids")?;
        ensure(
            &self.stream,
            &mut self.region_distances,
            pairs,
            "region distances",
        )?;
        ensure(&self.stream, &mut self.output_ids, k, "output ids")?;
        ensure(
            &self.stream,
            &mut self.output_distances,
            k,
            "output distances",
        )
    }

    fn readback(&self, k: usize) -> Result<Vec<(u64, f32)>> {
        let ids = self
            .stream
            .clone_dtoh(&self.output_ids.as_ref().expect("output ids").slice(..k))
            .map_err(cuda_error("final id readback"))?;
        let distances = self
            .stream
            .clone_dtoh(
                &self
                    .output_distances
                    .as_ref()
                    .expect("output distances")
                    .slice(..k),
            )
            .map_err(cuda_error("final distance readback"))?;
        let out = ids
            .into_iter()
            .zip(distances)
            .filter(|(id, distance)| *id != u64::MAX && distance.is_finite())
            .collect::<Vec<_>>();
        TELEMETRY
            .final_readback_pairs
            .fetch_add(k as u64, Ordering::Relaxed);
        TELEMETRY.d2h_bytes.fetch_add(
            (k * (size_of::<u64>() + size_of::<f32>())) as u64,
            Ordering::Relaxed,
        );
        Ok(out)
    }

    fn scratch_bytes(&self) -> u64 {
        buffer_bytes(&self.dataset_addresses)
            .saturating_add(buffer_bytes(&self.dataset_dtypes))
            .saturating_add(buffer_bytes(&self.global_id_addresses))
            .saturating_add(buffer_bytes(&self.rows))
            .saturating_add(buffer_bytes(&self.row_strides))
            .saturating_add(buffer_bytes(&self.column_strides))
            .saturating_add(buffer_bytes(&self.query))
            .saturating_add(buffer_bytes(&self.region_ids))
            .saturating_add(buffer_bytes(&self.region_distances))
            .saturating_add(buffer_bytes(&self.output_ids))
            .saturating_add(buffer_bytes(&self.output_distances))
    }
}

fn buffer_bytes<T>(buffer: &Option<CudaSlice<T>>) -> u64 {
    buffer.as_ref().map_or(0, |slice| {
        u64::try_from(slice.len().saturating_mul(size_of::<T>())).unwrap_or(u64::MAX)
    })
}

fn copy<T>(
    stream: &Arc<CudaStream>,
    host: &[T],
    device: &mut Option<CudaSlice<T>>,
    name: &'static str,
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
{
    stream
        .memcpy_htod(host, device.as_mut().expect("device buffer"))
        .map_err(cuda_error(name))
}

fn ensure<T>(
    stream: &Arc<CudaStream>,
    buffer: &mut Option<CudaSlice<T>>,
    len: usize,
    name: &'static str,
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
{
    if buffer.as_ref().is_none_or(|current| current.len() < len) {
        *buffer = Some(stream.alloc_zeros(len).map_err(cuda_error(name))?);
    }
    Ok(())
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
        unavailable(format!("partitioned CUDA {stage}: {error}"))
    }
}
