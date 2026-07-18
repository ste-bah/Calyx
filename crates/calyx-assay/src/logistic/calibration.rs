use super::*;

pub(super) fn logistic_power_calibration(
    samples: &[Vec<f32>],
    labels: &[bool],
    groups: Option<&[String]>,
    trust: TrustTag,
) -> Result<PowerCalibration> {
    let planted_bits = ensure_informative_binary_labels(labels)?;
    let dim = validate_rectangular_finite("logistic power calibration", samples)?;
    if dim == 0 {
        return Err(crate::calibration::underpowered(
            "power calibration requires at least one feature column",
        ));
    }
    let planted_column = dim - 1;
    let planted = plant_binary_signal(samples, labels, planted_column);
    let report = match groups {
        Some(groups) => {
            logistic_probe_mi_multiseed_with_trust(&planted, labels, Some(groups), trust)?
        }
        None => logistic_probe_mi_with_trust(&planted, labels, trust)?,
    };
    let calibration = PowerCalibration::new(
        planted_bits,
        report.estimate.bits,
        DEFAULT_MIN_POWER_RECOVERY_RATIO,
        labels.len(),
        dim,
        planted_column,
    )?;
    calibration.require_passed()?;
    Ok(calibration)
}

pub(super) fn logistic_power_calibration_cuda_strict(
    samples: &[Vec<f32>],
    labels: &[bool],
    groups: Option<&[String]>,
    trust: TrustTag,
) -> Result<PowerCalibration> {
    let planted_bits = ensure_informative_binary_labels(labels)?;
    let dim = validate_rectangular_finite("logistic power calibration", samples)?;
    if dim == 0 {
        return Err(crate::calibration::underpowered(
            "power calibration requires at least one feature column",
        ));
    }
    let planted_column = dim - 1;
    let planted = plant_binary_signal(samples, labels, planted_column);
    let report = match groups {
        Some(groups) => logistic_probe_mi_multiseed_with_trust_and_min_samples_cuda_strict(
            &planted,
            labels,
            Some(groups),
            trust,
            MIN_ASSAY_SAMPLES,
        )?,
        None => logistic_probe_mi_with_trust_and_min_samples_cuda_strict(
            &planted,
            labels,
            trust,
            MIN_ASSAY_SAMPLES,
        )?,
    };
    let calibration = PowerCalibration::new(
        planted_bits,
        report.estimate.bits,
        DEFAULT_MIN_POWER_RECOVERY_RATIO,
        labels.len(),
        dim,
        planted_column,
    )?;
    calibration.require_passed()?;
    Ok(calibration)
}

fn plant_binary_signal(samples: &[Vec<f32>], labels: &[bool], column: usize) -> Vec<Vec<f32>> {
    let mut planted = samples.to_vec();
    for (row, label) in planted.iter_mut().zip(labels) {
        row[column] = if *label { 1.0 } else { -1.0 };
    }
    planted
}
