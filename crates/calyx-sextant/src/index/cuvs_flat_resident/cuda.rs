use std::ffi::CStr;
use std::os::raw::c_void;
use std::ptr;
use std::sync::Arc;

use calyx_core::Result;
use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, ValidAsZeroBits, sys::CUdeviceptr,
};
use cuvs_sys as ffi;

use super::{CUVS_RESIDENT_FLAT_MAX_BATCH, CUVS_RESIDENT_FLAT_MAX_K, CuvsResidentFlatDiagnostics};
use crate::error::{
    CALYX_INDEX_DIM_MISMATCH, CALYX_INDEX_INVALID_PARAMS, CALYX_SEXTANT_GPU_SERVING_UNAVAILABLE,
    sextant_error,
};

#[path = "cuda/exact.rs"]
mod exact;
use exact::ExactKernels;

pub(super) struct ResidentFlat {
    // Drop index before its borrowed dataset and all CUDA ownership.
    index: BruteForceIndex,
    dataset: CudaSlice<f32>,
    exact: ExactKernels,
    resources: Resources,
    stream: Arc<CudaStream>,
    _context: Arc<CudaContext>,
    rows: usize,
    dim: usize,
    has_zero_row: bool,
    resident_bytes: u64,
    query: Option<CudaSlice<f32>>,
    ids: Option<CudaSlice<i64>>,
    distances: Option<CudaSlice<f32>>,
    filter: Option<CudaSlice<u32>>,
    row_distances: Option<CudaSlice<f32>>,
    diagnostics: CuvsResidentFlatDiagnostics,
}

// The owning cache serializes every use with `Mutex<ResidentFlat>`.
unsafe impl Send for ResidentFlat {}

impl ResidentFlat {
    pub(super) fn new(rows: usize, dim: usize, values: &[f32]) -> Result<Self> {
        let context = CudaContext::new(0).map_err(cuda_error("context init"))?;
        let stream = context.new_stream().map_err(cuda_error("stream init"))?;
        let resources = Resources::new(&stream)?;
        let dataset = stream
            .clone_htod(values)
            .map_err(cuda_error("dataset upload"))?;
        let index = BruteForceIndex::build(&resources, &stream, &dataset, rows, dim)?;
        let exact = ExactKernels::load(&context)?;
        let has_zero_row = values
            .chunks_exact(dim)
            .any(|row| row.iter().all(|value| *value == 0.0));
        let resident = values
            .len()
            .saturating_mul(size_of::<f32>())
            .saturating_add(
                CUVS_RESIDENT_FLAT_MAX_BATCH
                    .saturating_mul(dim)
                    .saturating_mul(size_of::<f32>()),
            )
            .saturating_add(
                CUVS_RESIDENT_FLAT_MAX_BATCH
                    .saturating_mul(CUVS_RESIDENT_FLAT_MAX_K)
                    .saturating_mul(size_of::<i64>() + size_of::<f32>()),
            )
            .saturating_add(rows.div_ceil(32).saturating_mul(size_of::<u32>()));
        let resident = resident.saturating_add(
            CUVS_RESIDENT_FLAT_MAX_BATCH
                .saturating_mul(rows)
                .saturating_mul(size_of::<f32>()),
        );
        let resident_bytes = u64::try_from(resident).unwrap_or(u64::MAX);
        Ok(Self {
            index,
            dataset,
            exact,
            resources,
            stream,
            _context: context,
            rows,
            dim,
            has_zero_row,
            resident_bytes,
            query: None,
            ids: None,
            distances: None,
            filter: None,
            row_distances: None,
            diagnostics: CuvsResidentFlatDiagnostics {
                backend: "cuvs-resident-flat-exact-v2",
                rows,
                dim,
                resident_bytes,
                h2d_bytes: size_of_val(values) as u64,
                ..CuvsResidentFlatDiagnostics::default()
            },
        })
    }

    pub(super) fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub(super) fn diagnostics(&self) -> CuvsResidentFlatDiagnostics {
        self.diagnostics.clone()
    }

    pub(super) fn search(
        &mut self,
        queries: &[f32],
        query_count: usize,
        k: usize,
        allowed_ids: Option<&[u32]>,
    ) -> Result<Vec<Vec<(u32, f32)>>> {
        if queries.len() != query_count.saturating_mul(self.dim) {
            return Err(sextant_error(
                CALYX_INDEX_DIM_MISMATCH,
                format!(
                    "resident flat query dim {} != {}",
                    queries.len() / query_count,
                    self.dim
                ),
            ));
        }
        let allowed = allowed_bitset(allowed_ids, self.rows)?;
        let available = allowed.as_ref().map_or(self.rows, |(_, count)| *count);
        let output_k = k.min(available);
        self.diagnostics.batches += 1;
        self.diagnostics.queries += query_count as u64;
        if output_k == 0 {
            return Ok(vec![Vec::new(); query_count]);
        }
        let pair_count = query_count
            .checked_mul(output_k)
            .ok_or_else(|| invalid("resident flat output shape overflow"))?;
        self.ensure_buffers(
            queries.len(),
            pair_count,
            self.rows.div_ceil(32),
            query_count.saturating_mul(self.rows),
        )?;
        self.stream
            .memcpy_htod(queries, self.query.as_mut().expect("query buffer"))
            .map_err(cuda_error("query upload"))?;
        self.diagnostics.query_uploads += 1;
        self.diagnostics.h2d_bytes += size_of_val(queries) as u64;
        let zero_query = queries
            .chunks_exact(self.dim)
            .any(|query| query.iter().all(|value| *value == 0.0));
        if allowed.is_some() || self.has_zero_row || zero_query {
            let words = allowed.map_or_else(|| all_allowed(self.rows), |(words, _)| words);
            self.stream
                .memcpy_htod(&words, self.filter.as_mut().expect("filter buffer"))
                .map_err(cuda_error("filter upload"))?;
            self.diagnostics.filter_uploads += 1;
            self.diagnostics.h2d_bytes += (words.len() * size_of::<u32>()) as u64;
            self.launch_exact(query_count, output_k)?;
            self.diagnostics.exact_filtered_kernel_launches += 2;
        } else {
            self.launch_cuvs(query_count, output_k)?;
            self.diagnostics.cuvs_kernel_launches += 1;
        }
        let ids = self
            .stream
            .clone_dtoh(&self.ids.as_ref().expect("id buffer").slice(..pair_count))
            .map_err(cuda_error("final id readback"))?;
        let distances = self
            .stream
            .clone_dtoh(
                &self
                    .distances
                    .as_ref()
                    .expect("distance buffer")
                    .slice(..pair_count),
            )
            .map_err(cuda_error("final distance readback"))?;
        self.diagnostics.final_readback_pairs += pair_count as u64;
        self.diagnostics.d2h_bytes += (pair_count * (size_of::<i64>() + size_of::<f32>())) as u64;
        validate_output(ids, distances, query_count, output_k, self.rows)
    }

    fn launch_cuvs(&mut self, query_count: usize, k: usize) -> Result<()> {
        let pairs = query_count * k;
        let mut query_shape = [query_count as i64, self.dim as i64];
        let mut output_shape = [query_count as i64, k as i64];
        let mut distance_shape = output_shape;
        let (query_ptr, _q) = self
            .query
            .as_ref()
            .expect("query buffer")
            .device_ptr(&self.stream);
        let (id_ptr, _i) = self
            .ids
            .as_mut()
            .expect("id buffer")
            .device_ptr_mut(&self.stream);
        let (distance_ptr, _d) = self
            .distances
            .as_mut()
            .expect("distance buffer")
            .device_ptr_mut(&self.stream);
        let mut query = device_tensor(query_ptr, &mut query_shape, dtype_f32());
        let mut ids = device_tensor(id_ptr, &mut output_shape, dtype_i64());
        let mut distances = device_tensor(distance_ptr, &mut distance_shape, dtype_f32());
        let _ = pairs;
        check(
            unsafe {
                ffi::cuvsBruteForceSearch(
                    self.resources.0,
                    self.index.0,
                    &mut query,
                    &mut ids,
                    &mut distances,
                    ffi::cuvsFilter {
                        addr: 0,
                        type_: ffi::cuvsFilterType::NO_FILTER,
                    },
                )
            },
            "exact search",
        )?;
        check(
            unsafe { ffi::cuvsStreamSync(self.resources.0) },
            "exact search sync",
        )
    }

    fn launch_exact(&mut self, query_count: usize, k: usize) -> Result<()> {
        self.exact.launch(
            &self.stream,
            &self.dataset,
            self.query.as_ref().expect("query buffer"),
            self.filter.as_ref().expect("filter buffer"),
            self.row_distances.as_mut().expect("row distance buffer"),
            self.ids.as_mut().expect("id buffer"),
            self.distances.as_mut().expect("distance buffer"),
            self.rows,
            self.dim,
            query_count,
            k,
        )
    }

    fn ensure_buffers(
        &mut self,
        query_values: usize,
        pairs: usize,
        filter_words: usize,
        row_distances: usize,
    ) -> Result<()> {
        ensure(&self.stream, &mut self.query, query_values, "query")?;
        ensure(&self.stream, &mut self.ids, pairs, "neighbor")?;
        ensure(&self.stream, &mut self.distances, pairs, "distance")?;
        ensure(&self.stream, &mut self.filter, filter_words, "filter")?;
        ensure(
            &self.stream,
            &mut self.row_distances,
            row_distances,
            "row distances",
        )
    }
}

struct Resources(ffi::cuvsResources_t);

impl Resources {
    fn new(stream: &Arc<CudaStream>) -> Result<Self> {
        let mut resources = 0;
        check(
            unsafe { ffi::cuvsResourcesCreate(&mut resources) },
            "create resources",
        )?;
        check(
            unsafe { ffi::cuvsStreamSet(resources, stream.cu_stream() as _) },
            "set stream",
        )?;
        Ok(Self(resources))
    }
}

impl Drop for Resources {
    fn drop(&mut self) {
        let _ = unsafe { ffi::cuvsResourcesDestroy(self.0) };
    }
}

struct BruteForceIndex(ffi::cuvsBruteForceIndex_t);

impl BruteForceIndex {
    fn build(
        resources: &Resources,
        stream: &Arc<CudaStream>,
        dataset: &CudaSlice<f32>,
        rows: usize,
        dim: usize,
    ) -> Result<Self> {
        let mut index = ptr::null_mut();
        check(
            unsafe { ffi::cuvsBruteForceIndexCreate(&mut index) },
            "create index",
        )?;
        if index.is_null() {
            return Err(unavailable("cuVS returned null resident flat index"));
        }
        let index = Self(index);
        let mut shape = [rows as i64, dim as i64];
        let (pointer, _guard) = dataset.device_ptr(stream);
        let mut tensor = device_tensor(pointer, &mut shape, dtype_f32());
        check(
            unsafe {
                ffi::cuvsBruteForceBuild(
                    resources.0,
                    &mut tensor,
                    ffi::cuvsDistanceType::CosineExpanded,
                    0.0,
                    index.0,
                )
            },
            "build index",
        )?;
        check(
            unsafe { ffi::cuvsStreamSync(resources.0) },
            "build index sync",
        )?;
        Ok(index)
    }
}

impl Drop for BruteForceIndex {
    fn drop(&mut self) {
        let _ = unsafe { ffi::cuvsBruteForceIndexDestroy(self.0) };
    }
}

fn allowed_bitset(ids: Option<&[u32]>, rows: usize) -> Result<Option<(Vec<u32>, usize)>> {
    let Some(ids) = ids else { return Ok(None) };
    let mut words = vec![0_u32; rows.div_ceil(32)];
    let mut count = 0usize;
    for &id in ids {
        let id = id as usize;
        if id >= rows {
            return Err(invalid(format!("flat filter id {id} exceeds {rows}")));
        }
        let mask = 1_u32 << (id % 32);
        if words[id / 32] & mask == 0 {
            words[id / 32] |= mask;
            count += 1;
        }
    }
    Ok(Some((words, count)))
}

fn all_allowed(rows: usize) -> Vec<u32> {
    let mut words = vec![u32::MAX; rows.div_ceil(32)];
    if !rows.is_multiple_of(32) {
        *words.last_mut().expect("nonempty bitset") = (1_u32 << (rows % 32)) - 1;
    }
    words
}

fn validate_output(
    ids: Vec<i64>,
    distances: Vec<f32>,
    query_count: usize,
    k: usize,
    rows: usize,
) -> Result<Vec<Vec<(u32, f32)>>> {
    let mut output = Vec::with_capacity(query_count);
    for query in 0..query_count {
        let start = query * k;
        let mut row = Vec::with_capacity(k);
        for (&id, &distance) in ids[start..start + k]
            .iter()
            .zip(&distances[start..start + k])
        {
            if id < 0 || id as usize >= rows || !distance.is_finite() {
                return Err(unavailable(format!(
                    "invalid flat result id={id} distance={distance}"
                )));
            }
            row.push((id as u32, distance.max(0.0)));
        }
        row.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        row.dedup_by_key(|(id, _)| *id);
        output.push(row);
    }
    Ok(output)
}

fn ensure<T>(
    stream: &Arc<CudaStream>,
    target: &mut Option<CudaSlice<T>>,
    len: usize,
    stage: &'static str,
) -> Result<()>
where
    T: cudarc::driver::DeviceRepr + ValidAsZeroBits,
{
    if target.as_ref().is_none_or(|buffer| buffer.len() < len) {
        *target = Some(stream.alloc_zeros(len).map_err(cuda_error(stage))?);
    }
    Ok(())
}

fn device_tensor(
    data: CUdeviceptr,
    shape: &mut [i64; 2],
    dtype: ffi::DLDataType,
) -> ffi::DLManagedTensor {
    ffi::DLManagedTensor {
        dl_tensor: ffi::DLTensor {
            data: data as usize as *mut c_void,
            device: ffi::DLDevice {
                device_type: ffi::DLDeviceType::kDLCUDA,
                device_id: 0,
            },
            ndim: 2,
            dtype,
            shape: shape.as_mut_ptr(),
            strides: ptr::null_mut(),
            byte_offset: 0,
        },
        manager_ctx: ptr::null_mut(),
        deleter: None,
    }
}

fn dtype_f32() -> ffi::DLDataType {
    ffi::DLDataType {
        code: ffi::DLDataTypeCode::kDLFloat as u8,
        bits: 32,
        lanes: 1,
    }
}

fn dtype_i64() -> ffi::DLDataType {
    ffi::DLDataType {
        code: ffi::DLDataTypeCode::kDLInt as u8,
        bits: 64,
        lanes: 1,
    }
}

fn check(status: ffi::cuvsError_t, stage: &'static str) -> Result<()> {
    if status == ffi::cuvsError_t::CUVS_SUCCESS {
        return Ok(());
    }
    let last = unsafe {
        let pointer = ffi::cuvsGetLastErrorText();
        if pointer.is_null() {
            "no cuVS error text".to_string()
        } else {
            CStr::from_ptr(pointer).to_string_lossy().into_owned()
        }
    };
    Err(unavailable(format!(
        "resident flat {stage}: {status:?}; {last}"
    )))
}

fn cuda_error(
    stage: &'static str,
) -> impl FnOnce(cudarc::driver::DriverError) -> calyx_core::CalyxError {
    move |error| unavailable(format!("resident flat {stage}: {error}"))
}

fn invalid(detail: impl Into<String>) -> calyx_core::CalyxError {
    sextant_error(CALYX_INDEX_INVALID_PARAMS, detail)
}

fn unavailable(detail: impl Into<String>) -> calyx_core::CalyxError {
    sextant_error(CALYX_SEXTANT_GPU_SERVING_UNAVAILABLE, detail)
}
