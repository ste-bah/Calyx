//! Binary outcome logistic-probe MI estimator.

mod calibration;
mod cuda;
mod train;

use calyx_core::{Anchor, CalyxError, Result};
use serde::{Deserialize, Serialize};

use crate::calibration::{
    DEFAULT_MIN_POWER_RECOVERY_RATIO, PowerCalibration, ensure_informative_binary_labels,
};
#[cfg(not(feature = "cuda"))]
use crate::cuda_strict::cuda_unavailable;
use crate::cuda_strict::strict_cuda_requested;
use crate::estimate::{EstimateReliability, EstimatorKind, MiEstimate, TrustTag, trust_for_anchor};
use crate::group_split::{GroupSplit, group_holdout_split, row_groups};
use crate::ksg::MIN_ASSAY_SAMPLES;
use crate::samples::validate_rectangular_finite;

use self::calibration::{logistic_power_calibration, logistic_power_calibration_cuda_strict};
use self::cuda::{
    LogisticCudaInputs, flatten_logistic_samples, logistic_summaries_cuda_strict_impl,
    split_buffers_for_cuda,
};
use self::train::{LogisticSummary, logistic_heldout_summary, mean, sample_sigma, seed_ci};

pub const DEFAULT_ASSAY_SEEDS: [u64; 5] = [20_260_612, 7, 101, 2_024, 99_999];
pub const DEFAULT_HOLDOUT_FRACTION: f32 = 0.2;
const LOGISTIC_STEPS: usize = 96;
const LOGISTIC_LR: f32 = 0.35;
const LOGISTIC_L2: f32 = 1.0e-4;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LogisticProbeReport {
    pub estimate: MiEstimate,
    pub accuracy: f32,
    pub selected_field: &'static str,
}

pub fn logistic_probe_mi(samples: &[Vec<f32>], labels: &[bool]) -> Result<LogisticProbeReport> {
    logistic_probe_mi_with_trust(samples, labels, TrustTag::Provisional)
}

pub fn logistic_probe_mi_calibrated(
    samples: &[Vec<f32>],
    labels: &[bool],
) -> Result<LogisticProbeReport> {
    ensure_informative_binary_labels(labels)?;
    let calibration = logistic_power_calibration(samples, labels, None, TrustTag::Provisional)?;
    let mut report = logistic_probe_mi_with_trust(samples, labels, TrustTag::Provisional)?;
    report.estimate = report.estimate.with_power_calibration(calibration);
    Ok(report)
}

pub fn logistic_probe_mi_with_anchor(
    samples: &[Vec<f32>],
    labels: &[bool],
    anchor: &Anchor,
) -> Result<LogisticProbeReport> {
    logistic_probe_mi_with_trust(samples, labels, trust_for_anchor(Some(anchor)))
}

pub fn logistic_probe_mi_cuda_strict(
    samples: &[Vec<f32>],
    labels: &[bool],
) -> Result<LogisticProbeReport> {
    logistic_probe_mi_with_trust_and_min_samples_cuda_strict(
        samples,
        labels,
        TrustTag::Provisional,
        MIN_ASSAY_SAMPLES,
    )
}

pub fn logistic_probe_mi_with_anchor_cuda_strict(
    samples: &[Vec<f32>],
    labels: &[bool],
    anchor: &Anchor,
) -> Result<LogisticProbeReport> {
    logistic_probe_mi_with_trust_and_min_samples_cuda_strict(
        samples,
        labels,
        trust_for_anchor(Some(anchor)),
        MIN_ASSAY_SAMPLES,
    )
}

pub fn logistic_probe_mi_calibrated_cuda_strict(
    samples: &[Vec<f32>],
    labels: &[bool],
) -> Result<LogisticProbeReport> {
    ensure_informative_binary_labels(labels)?;
    let calibration =
        logistic_power_calibration_cuda_strict(samples, labels, None, TrustTag::Provisional)?;
    let mut report = logistic_probe_mi_cuda_strict(samples, labels)?;
    report.estimate = report.estimate.with_power_calibration(calibration);
    Ok(report)
}

pub fn logistic_probe_mi_multiseed(
    samples: &[Vec<f32>],
    labels: &[bool],
    groups: Option<&[String]>,
) -> Result<LogisticProbeReport> {
    logistic_probe_mi_multiseed_with_trust(samples, labels, groups, TrustTag::Provisional)
}

pub fn logistic_probe_mi_multiseed_calibrated(
    samples: &[Vec<f32>],
    labels: &[bool],
    groups: Option<&[String]>,
) -> Result<LogisticProbeReport> {
    logistic_probe_mi_multiseed_calibrated_with_trust(
        samples,
        labels,
        groups,
        TrustTag::Provisional,
    )
}

pub fn logistic_probe_mi_multiseed_cuda_strict(
    samples: &[Vec<f32>],
    labels: &[bool],
    groups: Option<&[String]>,
) -> Result<LogisticProbeReport> {
    logistic_probe_mi_multiseed_with_trust_and_min_samples_cuda_strict(
        samples,
        labels,
        groups,
        TrustTag::Provisional,
        MIN_ASSAY_SAMPLES,
    )
}

pub fn logistic_probe_mi_multiseed_calibrated_cuda_strict(
    samples: &[Vec<f32>],
    labels: &[bool],
    groups: Option<&[String]>,
) -> Result<LogisticProbeReport> {
    logistic_probe_mi_multiseed_calibrated_with_trust_cuda_strict(
        samples,
        labels,
        groups,
        TrustTag::Provisional,
    )
}

pub fn logistic_probe_mi_multiseed_with_anchor(
    samples: &[Vec<f32>],
    labels: &[bool],
    groups: Option<&[String]>,
    anchor: &Anchor,
) -> Result<LogisticProbeReport> {
    logistic_probe_mi_multiseed_with_trust(samples, labels, groups, trust_for_anchor(Some(anchor)))
}

pub fn logistic_probe_mi_multiseed_calibrated_with_anchor(
    samples: &[Vec<f32>],
    labels: &[bool],
    groups: Option<&[String]>,
    anchor: &Anchor,
) -> Result<LogisticProbeReport> {
    logistic_probe_mi_multiseed_calibrated_with_trust(
        samples,
        labels,
        groups,
        trust_for_anchor(Some(anchor)),
    )
}

pub fn logistic_probe_mi_multiseed_with_anchor_cuda_strict(
    samples: &[Vec<f32>],
    labels: &[bool],
    groups: Option<&[String]>,
    anchor: &Anchor,
) -> Result<LogisticProbeReport> {
    logistic_probe_mi_multiseed_with_trust_and_min_samples_cuda_strict(
        samples,
        labels,
        groups,
        trust_for_anchor(Some(anchor)),
        MIN_ASSAY_SAMPLES,
    )
}

pub fn logistic_probe_mi_multiseed_calibrated_with_anchor_cuda_strict(
    samples: &[Vec<f32>],
    labels: &[bool],
    groups: Option<&[String]>,
    anchor: &Anchor,
) -> Result<LogisticProbeReport> {
    logistic_probe_mi_multiseed_calibrated_with_trust_cuda_strict(
        samples,
        labels,
        groups,
        trust_for_anchor(Some(anchor)),
    )
}

pub(crate) fn logistic_probe_mi_with_min_samples(
    samples: &[Vec<f32>],
    labels: &[bool],
    min_samples: usize,
) -> Result<LogisticProbeReport> {
    logistic_probe_mi_with_trust_and_min_samples(
        samples,
        labels,
        TrustTag::Provisional,
        min_samples,
    )
}

pub(crate) fn logistic_probe_mi_with_anchor_and_min_samples(
    samples: &[Vec<f32>],
    labels: &[bool],
    anchor: &Anchor,
    min_samples: usize,
) -> Result<LogisticProbeReport> {
    logistic_probe_mi_with_trust_and_min_samples(
        samples,
        labels,
        trust_for_anchor(Some(anchor)),
        min_samples,
    )
}

fn logistic_probe_mi_with_trust(
    samples: &[Vec<f32>],
    labels: &[bool],
    trust: TrustTag,
) -> Result<LogisticProbeReport> {
    logistic_probe_mi_with_trust_and_min_samples(samples, labels, trust, MIN_ASSAY_SAMPLES)
}

fn logistic_probe_mi_multiseed_with_trust(
    samples: &[Vec<f32>],
    labels: &[bool],
    groups: Option<&[String]>,
    trust: TrustTag,
) -> Result<LogisticProbeReport> {
    logistic_probe_mi_multiseed_with_trust_and_min_samples(
        samples,
        labels,
        groups,
        trust,
        MIN_ASSAY_SAMPLES,
    )
}

fn logistic_probe_mi_multiseed_with_trust_and_min_samples(
    samples: &[Vec<f32>],
    labels: &[bool],
    groups: Option<&[String]>,
    trust: TrustTag,
    min_samples: usize,
) -> Result<LogisticProbeReport> {
    if strict_cuda_requested() {
        return logistic_probe_mi_multiseed_with_trust_and_min_samples_cuda_strict(
            samples,
            labels,
            groups,
            trust,
            min_samples,
        );
    }
    if samples.len() != labels.len() || samples.len() < min_samples {
        return Err(CalyxError::assay_insufficient_samples(format!(
            "need at least {min_samples} labeled samples"
        )));
    }
    let dim = validate_rectangular_finite("logistic", samples)?;
    let owned_groups;
    let groups = match groups {
        Some(groups) => groups,
        None => {
            owned_groups = row_groups(labels.len());
            &owned_groups
        }
    };
    let mut seed_summaries = Vec::with_capacity(DEFAULT_ASSAY_SEEDS.len());
    for seed in DEFAULT_ASSAY_SEEDS {
        let split = group_holdout_split(labels, groups, DEFAULT_HOLDOUT_FRACTION, seed)?;
        seed_summaries.push(logistic_heldout_summary(samples, labels, dim, &split));
    }
    report_from_seed_summaries(seed_summaries, labels.len(), trust)
}

fn logistic_probe_mi_multiseed_with_trust_and_min_samples_cuda_strict(
    samples: &[Vec<f32>],
    labels: &[bool],
    groups: Option<&[String]>,
    trust: TrustTag,
    min_samples: usize,
) -> Result<LogisticProbeReport> {
    if samples.len() != labels.len() || samples.len() < min_samples {
        return Err(CalyxError::assay_insufficient_samples(format!(
            "need at least {min_samples} labeled samples"
        )));
    }
    let dim = validate_rectangular_finite("logistic", samples)?;
    let owned_groups;
    let groups = match groups {
        Some(groups) => groups,
        None => {
            owned_groups = row_groups(labels.len());
            &owned_groups
        }
    };
    let mut splits = Vec::with_capacity(DEFAULT_ASSAY_SEEDS.len());
    for seed in DEFAULT_ASSAY_SEEDS {
        splits.push(group_holdout_split(
            labels,
            groups,
            DEFAULT_HOLDOUT_FRACTION,
            seed,
        )?);
    }
    let flat = flatten_logistic_samples(samples, dim)?;
    let cuda_labels = labels
        .iter()
        .map(|label| i32::from(*label))
        .collect::<Vec<_>>();
    let (train_offsets, train_indices, test_offsets, test_indices) =
        split_buffers_for_cuda(&splits, labels.len())?;
    let summaries = logistic_summaries_cuda_strict_impl(LogisticCudaInputs {
        samples: &flat,
        labels: &cuda_labels,
        rows: labels.len(),
        dim,
        train_offsets: &train_offsets,
        train_indices: &train_indices,
        test_offsets: &test_offsets,
        test_indices: &test_indices,
    })?;
    if summaries.bits.len() != DEFAULT_ASSAY_SEEDS.len()
        || summaries.accuracy.len() != DEFAULT_ASSAY_SEEDS.len()
    {
        return Err(CalyxError::forge_numerical_invariant(format!(
            "logistic CUDA returned {} bits and {} accuracies for {} seeds",
            summaries.bits.len(),
            summaries.accuracy.len(),
            DEFAULT_ASSAY_SEEDS.len()
        )));
    }
    let seed_summaries = summaries
        .bits
        .iter()
        .zip(summaries.accuracy.iter())
        .map(|(&bits, &accuracy)| LogisticSummary { bits, accuracy })
        .collect::<Vec<_>>();
    report_from_seed_summaries(seed_summaries, labels.len(), trust)
}

fn report_from_seed_summaries(
    seed_summaries: Vec<LogisticSummary>,
    n_samples: usize,
    trust: TrustTag,
) -> Result<LogisticProbeReport> {
    let seed_bits = seed_summaries
        .iter()
        .map(|summary| summary.bits)
        .collect::<Vec<_>>();
    let bits = mean(&seed_bits);
    let seed_sigma = sample_sigma(&seed_bits);
    let (ci_low, ci_high) = seed_ci(bits, seed_sigma, seed_bits.len());
    let reliability =
        EstimateReliability::new(seed_bits.len(), seed_sigma, seed_sigma >= bits.abs())?;
    Ok(LogisticProbeReport {
        estimate: MiEstimate::new(
            bits,
            ci_low,
            ci_high,
            n_samples,
            EstimatorKind::LogisticProbe,
            trust,
        )
        .with_reliability(reliability),
        accuracy: mean(
            &seed_summaries
                .iter()
                .map(|summary| summary.accuracy)
                .collect::<Vec<_>>(),
        ),
        selected_field: "logistic_probe_multiseed_group_holdout",
    })
}

fn logistic_probe_mi_multiseed_calibrated_with_trust(
    samples: &[Vec<f32>],
    labels: &[bool],
    groups: Option<&[String]>,
    trust: TrustTag,
) -> Result<LogisticProbeReport> {
    if strict_cuda_requested() {
        return logistic_probe_mi_multiseed_calibrated_with_trust_cuda_strict(
            samples, labels, groups, trust,
        );
    }
    ensure_informative_binary_labels(labels)?;
    let calibration = logistic_power_calibration(samples, labels, groups, trust)?;
    let mut report = logistic_probe_mi_multiseed_with_trust(samples, labels, groups, trust)?;
    report.estimate = report.estimate.with_power_calibration(calibration);
    Ok(report)
}

fn logistic_probe_mi_multiseed_calibrated_with_trust_cuda_strict(
    samples: &[Vec<f32>],
    labels: &[bool],
    groups: Option<&[String]>,
    trust: TrustTag,
) -> Result<LogisticProbeReport> {
    ensure_informative_binary_labels(labels)?;
    let calibration = logistic_power_calibration_cuda_strict(samples, labels, groups, trust)?;
    let mut report = logistic_probe_mi_multiseed_with_trust_and_min_samples_cuda_strict(
        samples,
        labels,
        groups,
        trust,
        MIN_ASSAY_SAMPLES,
    )?;
    report.estimate = report.estimate.with_power_calibration(calibration);
    Ok(report)
}

fn logistic_probe_mi_with_trust_and_min_samples(
    samples: &[Vec<f32>],
    labels: &[bool],
    trust: TrustTag,
    min_samples: usize,
) -> Result<LogisticProbeReport> {
    logistic_probe_mi_multiseed_with_trust_and_min_samples(
        samples,
        labels,
        None,
        trust,
        min_samples,
    )
}

fn logistic_probe_mi_with_trust_and_min_samples_cuda_strict(
    samples: &[Vec<f32>],
    labels: &[bool],
    trust: TrustTag,
    min_samples: usize,
) -> Result<LogisticProbeReport> {
    logistic_probe_mi_multiseed_with_trust_and_min_samples_cuda_strict(
        samples,
        labels,
        None,
        trust,
        min_samples,
    )
}
