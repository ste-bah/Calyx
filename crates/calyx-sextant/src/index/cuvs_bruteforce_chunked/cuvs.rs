use std::ffi::CStr;
use std::os::raw::c_void;
use std::ptr;
use std::sync::Arc;

use calyx_core::Result;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut, sys::CUdeviceptr};
use cuvs_sys as ffi;

use super::invalid;
use crate::error::{CALYX_INDEX_IO, sextant_error};
use crate::index::CuvsDistanceMetric;

pub(super) struct Resources(ffi::cuvsResources_t);

impl Resources {
    pub(super) fn new() -> Result<Self> {
        let mut resources = 0;
        check(
            unsafe { ffi::cuvsResourcesCreate(&mut resources) },
            "create resources",
        )?;
        Ok(Self(resources))
    }
}

impl Drop for Resources {
    fn drop(&mut self) {
        let _ = unsafe { ffi::cuvsResourcesDestroy(self.0) };
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn search_chunk(
    resources: &Resources,
    stream: &Arc<CudaStream>,
    corpus: &CudaSlice<f32>,
    rows: usize,
    dim: usize,
    queries: &CudaSlice<f32>,
    query_count: usize,
    k: usize,
    metric: CuvsDistanceMetric,
    ids: &mut CudaSlice<i64>,
    distances: &mut CudaSlice<f32>,
) -> Result<()> {
    let index = BruteForceIndex::new()?;
    let mut dataset_shape = [rows as i64, dim as i64];
    let (dataset_ptr, _dataset_guard) = corpus.device_ptr(stream);
    let mut dataset_tensor = device_tensor(dataset_ptr, &mut dataset_shape, dtype_f32());
    check(
        unsafe {
            ffi::cuvsBruteForceBuild(
                resources.0,
                &mut dataset_tensor,
                ffi_metric(metric),
                0.0,
                index.0,
            )
        },
        "build chunk",
    )?;
    check(
        unsafe { ffi::cuvsStreamSync(resources.0) },
        "sync chunk build",
    )?;

    let mut query_shape = [query_count as i64, dim as i64];
    let mut output_shape = [query_count as i64, k as i64];
    let (query_ptr, _query_guard) = queries.device_ptr(stream);
    let (id_ptr, _id_guard) = ids.device_ptr_mut(stream);
    let (distance_ptr, _distance_guard) = distances.device_ptr_mut(stream);
    let mut query_tensor = device_tensor(query_ptr, &mut query_shape, dtype_f32());
    let mut id_tensor = device_tensor(id_ptr, &mut output_shape, dtype_i64());
    let mut distance_shape = output_shape;
    let mut distance_tensor = device_tensor(distance_ptr, &mut distance_shape, dtype_f32());
    let filter = ffi::cuvsFilter {
        addr: 0,
        type_: ffi::cuvsFilterType::NO_FILTER,
    };
    check(
        unsafe {
            ffi::cuvsBruteForceSearch(
                resources.0,
                index.0,
                &mut query_tensor,
                &mut id_tensor,
                &mut distance_tensor,
                filter,
            )
        },
        "search chunk",
    )?;
    check(
        unsafe { ffi::cuvsStreamSync(resources.0) },
        "sync chunk search",
    )
}

struct BruteForceIndex(ffi::cuvsBruteForceIndex_t);

impl BruteForceIndex {
    fn new() -> Result<Self> {
        let mut index = ptr::null_mut();
        check(
            unsafe { ffi::cuvsBruteForceIndexCreate(&mut index) },
            "create index",
        )?;
        if index.is_null() {
            return Err(invalid("cuVS returned a null brute-force index"));
        }
        Ok(Self(index))
    }
}

impl Drop for BruteForceIndex {
    fn drop(&mut self) {
        let _ = unsafe { ffi::cuvsBruteForceIndexDestroy(self.0) };
    }
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

fn ffi_metric(metric: CuvsDistanceMetric) -> ffi::cuvsDistanceType {
    match metric {
        CuvsDistanceMetric::Cosine => ffi::cuvsDistanceType::CosineExpanded,
        CuvsDistanceMetric::SquaredL2 => ffi::cuvsDistanceType::L2Expanded,
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
    Err(sextant_error(
        CALYX_INDEX_IO,
        format!("chunked cuVS exact {stage} failed: {status:?}; {last}"),
    ))
}
