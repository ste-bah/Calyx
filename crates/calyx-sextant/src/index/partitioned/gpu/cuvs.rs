use std::ffi::CStr;
use std::os::raw::c_void;
use std::ptr;
use std::sync::Arc;

use calyx_core::Result;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut, sys::CUdeviceptr};
use cuvs_sys as ffi;

use crate::error::{CALYX_INDEX_IO, sextant_error};

pub(super) struct Resources(pub(super) ffi::cuvsResources_t);

impl Resources {
    pub(super) fn new() -> Result<Self> {
        let mut resources = 0;
        check(
            unsafe { ffi::cuvsResourcesCreate(&mut resources) },
            "create resources",
        )?;
        Ok(Self(resources))
    }

    pub(super) fn sync(&self, stage: &'static str) -> Result<()> {
        check(unsafe { ffi::cuvsStreamSync(self.0) }, stage)
    }
}

impl Drop for Resources {
    fn drop(&mut self) {
        let _ = unsafe { ffi::cuvsResourcesDestroy(self.0) };
    }
}

#[allow(clippy::too_many_arguments)] // Mirrors the shape-explicit cuVS FFI call.
pub(super) fn fit_balanced_kmeans(
    resources: &Resources,
    stream: &Arc<CudaStream>,
    samples: &CudaSlice<f32>,
    rows: usize,
    dim: usize,
    centroids: &mut CudaSlice<f32>,
    clusters: usize,
    iterations: usize,
) -> Result<usize> {
    let params = KMeansParams::new()?;
    params.configure(clusters, iterations)?;
    let mut sample_shape = [to_i64(rows)?, to_i64(dim)?];
    let mut centroid_shape = [to_i64(clusters)?, to_i64(dim)?];
    let (sample_ptr, _sample_guard) = samples.device_ptr(stream);
    let (centroid_ptr, _centroid_guard) = centroids.device_ptr_mut(stream);
    let mut sample_tensor = device_tensor(sample_ptr, &mut sample_shape, dtype_f32());
    let mut centroid_tensor = device_tensor(centroid_ptr, &mut centroid_shape, dtype_f32());
    let mut inertia = 0.0;
    let mut completed = 0;
    check(
        unsafe {
            ffi::cuvsKMeansFit(
                resources.0,
                params.0,
                &mut sample_tensor,
                ptr::null_mut(),
                &mut centroid_tensor,
                &mut inertia,
                &mut completed,
            )
        },
        "fit k-means",
    )?;
    resources.sync("sync k-means")?;
    if !inertia.is_finite() || completed <= 0 {
        return Err(error(
            "fit k-means",
            format!("invalid inertia={inertia} iterations={completed}"),
        ));
    }
    usize::try_from(completed).map_err(|_| {
        error(
            "fit k-means",
            format!("invalid iteration count {completed}"),
        )
    })
}

struct KMeansParams(ffi::cuvsKMeansParams_t);

impl KMeansParams {
    fn new() -> Result<Self> {
        let mut params = ptr::null_mut();
        check(
            unsafe { ffi::cuvsKMeansParamsCreate(&mut params) },
            "create k-means params",
        )?;
        if params.is_null() {
            return Err(error("create k-means params", "returned null"));
        }
        Ok(Self(params))
    }

    fn configure(&self, clusters: usize, iterations: usize) -> Result<()> {
        unsafe {
            (*self.0).metric = ffi::cuvsDistanceType::L2Expanded;
            (*self.0).n_clusters = to_i32(clusters)?;
            (*self.0).hierarchical = true;
            (*self.0).hierarchical_n_iters = to_i32(iterations)?;
        }
        Ok(())
    }
}

impl Drop for KMeansParams {
    fn drop(&mut self) {
        let _ = unsafe { ffi::cuvsKMeansParamsDestroy(self.0) };
    }
}

pub(super) struct BruteForceIndex(ffi::cuvsBruteForceIndex_t);

impl BruteForceIndex {
    pub(super) fn build(
        resources: &Resources,
        stream: &Arc<CudaStream>,
        centroids: &CudaSlice<f32>,
        rows: usize,
        dim: usize,
    ) -> Result<Self> {
        let mut index = ptr::null_mut();
        check(
            unsafe { ffi::cuvsBruteForceIndexCreate(&mut index) },
            "create routing index",
        )?;
        if index.is_null() {
            return Err(error("create routing index", "returned null"));
        }
        let index = Self(index);
        let mut shape = [to_i64(rows)?, to_i64(dim)?];
        let (pointer, _guard) = centroids.device_ptr(stream);
        let mut tensor = device_tensor(pointer, &mut shape, dtype_f32());
        check(
            unsafe {
                ffi::cuvsBruteForceBuild(
                    resources.0,
                    &mut tensor,
                    ffi::cuvsDistanceType::L2Expanded,
                    0.0,
                    index.0,
                )
            },
            "build routing index",
        )?;
        resources.sync("sync routing index build")?;
        Ok(index)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn search(
        &self,
        resources: &Resources,
        stream: &Arc<CudaStream>,
        query_pointer: CUdeviceptr,
        query_rows: usize,
        dim: usize,
        probe: usize,
        ids: &mut CudaSlice<i64>,
        distances: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let mut query_shape = [to_i64(query_rows)?, to_i64(dim)?];
        let mut output_shape = [to_i64(query_rows)?, to_i64(probe)?];
        let mut distance_shape = output_shape;
        let (id_pointer, _id_guard) = ids.device_ptr_mut(stream);
        let (distance_pointer, _distance_guard) = distances.device_ptr_mut(stream);
        let mut queries = device_tensor(query_pointer, &mut query_shape, dtype_f32());
        let mut neighbors = device_tensor(id_pointer, &mut output_shape, dtype_i64());
        let mut scores = device_tensor(distance_pointer, &mut distance_shape, dtype_f32());
        let filter = ffi::cuvsFilter {
            addr: 0,
            type_: ffi::cuvsFilterType::NO_FILTER,
        };
        check(
            unsafe {
                ffi::cuvsBruteForceSearch(
                    resources.0,
                    self.0,
                    &mut queries,
                    &mut neighbors,
                    &mut scores,
                    filter,
                )
            },
            "route rows",
        )?;
        resources.sync("sync routed rows")
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

fn dtype_f32() -> ffi::DLDataType {
    dtype(ffi::DLDataTypeCode::kDLFloat, 32)
}

fn dtype_i64() -> ffi::DLDataType {
    dtype(ffi::DLDataTypeCode::kDLInt, 64)
}

fn dtype(code: ffi::DLDataTypeCode, bits: u8) -> ffi::DLDataType {
    ffi::DLDataType {
        code: code as u8,
        bits,
        lanes: 1,
    }
}

fn to_i32(value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| error("shape conversion", format!("{value} exceeds i32")))
}

fn to_i64(value: usize) -> Result<i64> {
    i64::try_from(value).map_err(|_| error("shape conversion", format!("{value} exceeds i64")))
}

fn check(status: ffi::cuvsError_t, stage: &'static str) -> Result<()> {
    if status == ffi::cuvsError_t::CUVS_SUCCESS {
        Ok(())
    } else {
        Err(error(stage, format!("status {status:?}")))
    }
}

fn error(stage: &str, detail: impl std::fmt::Display) -> calyx_core::CalyxError {
    let last = unsafe {
        let pointer = ffi::cuvsGetLastErrorText();
        if pointer.is_null() {
            "no cuVS error text".to_string()
        } else {
            CStr::from_ptr(pointer).to_string_lossy().into_owned()
        }
    };
    sextant_error(
        CALYX_INDEX_IO,
        format!("partition CUDA {stage}: {detail}; last_error={last}"),
    )
}
