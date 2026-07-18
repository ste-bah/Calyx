use calyx_core::Result;

use super::PartitionBuildParams;
use crate::error::{CALYX_INDEX_INVALID_PARAMS, sextant_error};

pub(super) fn validate(row_count: u64, dim: usize, params: &PartitionBuildParams) -> Result<()> {
    if row_count == 0 || dim == 0 || params.n_regions == 0 || params.final_assignment_probe == 0 {
        return Err(invalid(
            "partitioned vault requires nonzero source len, dim, n_regions, final_assignment_probe",
        ));
    }
    if params.final_assignment_cap == Some(0) {
        return Err(invalid("final_assignment_cap must be > 0 when set"));
    }
    if params.balance_cap == Some(0) {
        return Err(invalid("balance_cap must be > 0 when set"));
    }
    if !params.assignment_boundary_epsilon.is_finite()
        || params.assignment_boundary_epsilon < 0.0
        || params.assignment_max_replication == 0
        || !params.assignment_rng_factor.is_finite()
        || params.assignment_rng_factor <= 0.0
    {
        return Err(invalid(
            "assignment_boundary_epsilon must be finite and >= 0, assignment_max_replication >= 1, assignment_rng_factor finite and > 0",
        ));
    }
    if params.region_build_parallelism == 0 {
        return Err(invalid("region_build_parallelism must be > 0"));
    }
    Ok(())
}

fn invalid(message: &'static str) -> calyx_core::CalyxError {
    sextant_error(CALYX_INDEX_INVALID_PARAMS, message)
}
