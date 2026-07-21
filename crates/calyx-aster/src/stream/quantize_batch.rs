use calyx_core::{CalyxError, CxId};
use calyx_forge::quant::{QuantizedVec, Quantizer, TurboQuantCodec, new_seed};

use super::quantize_online::{
    QuantizeOnlineConfig, forge_to_calyx, input_nan_error, rotation_seed_entropy,
};

#[derive(Clone, Copy)]
pub(super) struct BatchRow<'a> {
    pub(super) raw: &'a [f32],
    pub(super) cx_id: CxId,
}

#[derive(Debug, Default)]
pub(super) struct BatchStats {
    pub(super) cpu_rows: usize,
    pub(super) cuda_rows: usize,
    pub(super) cuda_shape_groups: usize,
    pub(super) cuda_kernel_launches: u64,
    pub(super) cuda_h2d_bytes: u64,
    pub(super) cuda_d2h_bytes: u64,
}

pub(super) struct BatchOutput {
    pub(super) rows: Vec<QuantizedVec>,
    pub(super) stats: BatchStats,
}

#[derive(Default)]
pub(super) struct BatchQuantizer {
    #[cfg(feature = "cuda")]
    cuda: Option<calyx_forge::CudaQuantContext>,
}

impl BatchQuantizer {
    pub(super) fn encode(
        &mut self,
        config: &QuantizeOnlineConfig,
        rows: &[BatchRow<'_>],
    ) -> Result<BatchOutput, CalyxError> {
        if rows.is_empty() {
            return Ok(BatchOutput {
                rows: Vec::new(),
                stats: BatchStats::default(),
            });
        }
        validate(rows)?;
        let elements = rows.iter().try_fold(0_usize, |total, row| {
            total.checked_add(row.raw.len()).ok_or_else(|| CalyxError {
                code: "CALYX_FORGE_SHAPE_MISMATCH",
                message: "streaming quantization element count overflow".to_string(),
                remediation: "reduce the streaming microbatch size",
            })
        })?;
        let codecs = rows
            .iter()
            .map(|row| codec(config, *row))
            .collect::<Result<Vec<_>, _>>()?;
        if elements < cuda_crossover() {
            return cpu_encode(rows, &codecs, elements);
        }
        self.cuda_encode(rows, &codecs, elements)
    }

    #[cfg(feature = "cuda")]
    fn cuda_encode(
        &mut self,
        rows: &[BatchRow<'_>],
        codecs: &[TurboQuantCodec],
        elements: usize,
    ) -> Result<BatchOutput, CalyxError> {
        let quant = match self.cuda.as_ref() {
            Some(quant) => quant.clone(),
            None => {
                let context = calyx_forge::init_cuda(0, false).map_err(forge_to_calyx)?;
                let quant = calyx_forge::CudaQuantContext::new(context);
                self.cuda = Some(quant.clone());
                quant
            }
        };
        let batch = codecs
            .iter()
            .zip(rows)
            .map(|(codec, row)| calyx_forge::CudaTurboQuantRow::new(codec, row.raw))
            .collect::<Vec<_>>();
        quant.reset_stats();
        let encoded = quant
            .encode_turboquant_ragged(&batch)
            .map_err(forge_to_calyx)?;
        let cuda = quant.stats();
        let shape_groups = usize::try_from(cuda.kernel_launches / 6).unwrap_or(usize::MAX);
        eprintln!(
            "CALYX_ASTER_QUANT_BATCH provider=cuda rows={} elements={elements} shape_groups={shape_groups} kernel_launches={} h2d_bytes={} d2h_bytes={}",
            rows.len(),
            cuda.kernel_launches,
            cuda.h2d_bytes,
            cuda.d2h_bytes,
        );
        Ok(BatchOutput {
            rows: encoded,
            stats: BatchStats {
                cuda_rows: rows.len(),
                cuda_shape_groups: shape_groups,
                cuda_kernel_launches: cuda.kernel_launches,
                cuda_h2d_bytes: cuda.h2d_bytes,
                cuda_d2h_bytes: cuda.d2h_bytes,
                ..BatchStats::default()
            },
        })
    }

    #[cfg(not(feature = "cuda"))]
    fn cuda_encode(
        &mut self,
        _rows: &[BatchRow<'_>],
        _codecs: &[TurboQuantCodec],
        elements: usize,
    ) -> Result<BatchOutput, CalyxError> {
        Err(CalyxError {
            code: "CALYX_FORGE_DEVICE_UNAVAILABLE",
            message: format!(
                "streaming TurboQuant batch has {elements} elements but Aster CUDA is not compiled"
            ),
            remediation: "build the owning Calyx binary with its cuda feature",
        })
    }
}

fn validate(rows: &[BatchRow<'_>]) -> Result<(), CalyxError> {
    for (row, value) in rows.iter().enumerate() {
        if value.raw.is_empty() {
            return Err(input_nan_error(format!(
                "dense slot row {row} is empty; nothing to quantize"
            )));
        }
        if let Some(offset) = value
            .raw
            .iter()
            .position(|coefficient| !coefficient.is_finite())
        {
            return Err(input_nan_error(format!(
                "dense slot row {row} has a non-finite coefficient at index {offset}"
            )));
        }
    }
    Ok(())
}

fn codec(config: &QuantizeOnlineConfig, row: BatchRow<'_>) -> Result<TurboQuantCodec, CalyxError> {
    let entropy = rotation_seed_entropy(config.lens_id, row.cx_id);
    TurboQuantCodec::new(new_seed(row.raw.len(), &entropy), config.level).map_err(forge_to_calyx)
}

fn cpu_encode(
    rows: &[BatchRow<'_>],
    codecs: &[TurboQuantCodec],
    elements: usize,
) -> Result<BatchOutput, CalyxError> {
    let encoded = rows
        .iter()
        .zip(codecs)
        .map(|(row, codec)| codec.encode(row.raw).map_err(forge_to_calyx))
        .collect::<Result<Vec<_>, _>>()?;
    eprintln!(
        "CALYX_ASTER_QUANT_BATCH provider=cpu rows={} elements={elements} reason=below_measured_cuda_crossover",
        rows.len(),
    );
    Ok(BatchOutput {
        rows: encoded,
        stats: BatchStats {
            cpu_rows: rows.len(),
            ..BatchStats::default()
        },
    })
}

fn cuda_crossover() -> usize {
    #[cfg(feature = "cuda")]
    {
        calyx_forge::TURBOQUANT_CUDA_MIN_ELEMENTS
    }
    #[cfg(not(feature = "cuda"))]
    {
        32 * 1024
    }
}
