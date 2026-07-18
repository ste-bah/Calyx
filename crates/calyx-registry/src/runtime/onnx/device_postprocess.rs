use calyx_core::{CalyxError, Result};
use ort::memory::AllocationDevice;
use ort::value::{DynTensorValueType, TensorElementType, ValueType};

use super::{OnnxProviderPolicy, PoolingPolicy, config_invalid};
use crate::frozen::NormPolicy;

pub(in crate::runtime::onnx) fn cuda_postprocess_context(
    policy: OnnxProviderPolicy,
    device_id: i32,
) -> Result<Option<calyx_forge::CudaContext>> {
    if policy != OnnxProviderPolicy::CudaFailLoud {
        return Ok(None);
    }
    let ordinal = u32::try_from(device_id).map_err(|_| CalyxError {
        code: "CALYX_ONNX_DEVICE_OUTPUT_INVALID_DEVICE",
        message: format!("ONNX CUDA device id {device_id} cannot initialize Forge CUDA context"),
        remediation: "set CALYX_ONNX_CUDA_DEVICE to a non-negative CUDA ordinal",
    })?;
    calyx_forge::init_cuda(ordinal, true)
        .map(Some)
        .map_err(forge_error)
}

pub(in crate::runtime::onnx) fn forge_error(error: calyx_forge::ForgeError) -> CalyxError {
    let remediation = match &error {
        calyx_forge::ForgeError::NumericalInvariant { .. } => "check lens runtime/normalize",
        calyx_forge::ForgeError::DeviceUnavailable { .. }
        | calyx_forge::ForgeError::GpuError { .. } => "restore lens service",
        calyx_forge::ForgeError::ShapeMismatch { .. } => "fix lens or slot shape",
        calyx_forge::ForgeError::VramBudget { .. }
        | calyx_forge::ForgeError::LensVramBudget { .. } => "split batch / raise budget / wait",
        _ => "fix ONNX model/tokenizer/config or register a supported lens spec",
    };
    CalyxError {
        code: error.code(),
        message: error.to_string(),
        remediation,
    }
}

pub(in crate::runtime::onnx) fn device_tensor<'a>(
    output: &'a ort::value::DynValue,
    label: &str,
) -> Result<ort::value::DynTensorRef<'a>> {
    let tensor = output
        .downcast_ref::<DynTensorValueType>()
        .map_err(|err| config_invalid(format!("{label} output is not an ONNX tensor: {err}")))?;
    if tensor.dtype().tensor_type() != Some(TensorElementType::Float32) {
        return Err(config_invalid(format!(
            "{label} output type {:?} is not Float32",
            tensor.dtype().tensor_type()
        )));
    }
    let memory = tensor.memory_info();
    if memory.allocation_device() != AllocationDevice::CUDA {
        return Err(CalyxError {
            code: "CALYX_ONNX_DEVICE_OUTPUT_ON_CPU",
            message: format!(
                "{label} output is allocated on {} device_id={} memory_type={:?}; CUDA postprocess requires device output",
                memory.allocation_device().as_str(),
                memory.device_id(),
                memory.memory_type()
            ),
            remediation: "bind ONNX outputs to CUDA device memory before CUDA postprocess",
        });
    }
    Ok(tensor)
}

pub(in crate::runtime::onnx) fn tensor_shape(
    tensor: &ort::value::DynTensorRef<'_>,
    label: &str,
) -> Result<Vec<i64>> {
    let ValueType::Tensor { shape, .. } = tensor.dtype() else {
        return Err(config_invalid(format!("{label} output is not a tensor")));
    };
    Ok(shape.iter().copied().collect())
}

pub(in crate::runtime::onnx) fn tensor_data_ptr(
    tensor: &ort::value::DynTensorRef<'_>,
    shape: &[i64],
    label: &str,
) -> Result<u64> {
    let ptr = tensor.data_ptr() as usize as u64;
    if ptr == 0 && tensor_elements(shape)? > 0 {
        return Err(CalyxError {
            code: "CALYX_ONNX_DEVICE_OUTPUT_NULL_PTR",
            message: format!("{label} output has non-empty shape {shape:?} but a null data ptr"),
            remediation: "verify ONNX CUDA output binding allocated a device tensor",
        });
    }
    Ok(ptr)
}

pub(in crate::runtime::onnx) fn pooling_to_cuda(
    policy: PoolingPolicy,
) -> calyx_forge::cuda::CudaPostprocessPooling {
    match policy {
        PoolingPolicy::Mean => calyx_forge::cuda::CudaPostprocessPooling::Mean,
        PoolingPolicy::Cls => calyx_forge::cuda::CudaPostprocessPooling::Cls,
        PoolingPolicy::LastToken => calyx_forge::cuda::CudaPostprocessPooling::LastToken,
    }
}

pub(in crate::runtime::onnx) const fn normalize_on_device(policy: NormPolicy) -> bool {
    matches!(policy, NormPolicy::L2 { .. } | NormPolicy::Unit { .. })
}

fn tensor_elements(shape: &[i64]) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, dim| {
        let dim = usize::try_from(*dim).map_err(|_| {
            CalyxError::lens_dim_mismatch(format!("negative ONNX output shape dim {dim}"))
        })?;
        acc.checked_mul(dim).ok_or_else(|| {
            CalyxError::lens_dim_mismatch(format!("ONNX output shape {shape:?} overflows usize"))
        })
    })
}
