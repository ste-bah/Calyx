use std::time::Instant;

use serde_json::json;

use super::test_support::{cpu_descent, deterministic_initial, deterministic_members};
use super::*;
use crate::Result;

#[test]
#[ignore = "requires an explicit CUDA acceptance host"]
fn issue1521_cuda_energy_crossover_and_realistic_workload_fsv() -> Result<()> {
    let _guard = crate::cuda::test_lock();
    let context = CudaEnergyContext::new(0)?;
    warm(&context)?;

    let sweep_dim = env_usize("CALYX_ENERGY_FSV_SWEEP_DIM", 256);
    let sweep_rows = env_rows(
        "CALYX_ENERGY_FSV_SWEEP_ROWS",
        &[128, 512, 1_024, 4_096, 16_384],
    );
    let mut sweep = Vec::new();
    for rows in sweep_rows {
        let members = deterministic_members(rows, sweep_dim);
        let refs = members.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let initial = deterministic_initial(sweep_dim);
        let cpu_started = Instant::now();
        let cpu = cpu_descent(&initial, &refs, 1.0, 20, 1.0e-4)?;
        let cpu_us = cpu_started.elapsed().as_micros();
        let gpu_started = Instant::now();
        let gpu = context.descend(&initial, &refs, 1.0, 20, 1.0e-4)?;
        let gpu_us = gpu_started.elapsed().as_micros();
        parity(&cpu, &gpu)?;
        sweep.push(json!({
            "rows": rows,
            "dim": sweep_dim,
            "elements": rows * sweep_dim,
            "cpu_us": cpu_us,
            "gpu_us": gpu_us,
            "speedup": cpu_us as f64 / gpu_us.max(1) as f64,
            "steps": gpu.steps_taken,
        }));
    }

    let rows = env_usize("CALYX_ENERGY_FSV_MEMBERS", 32_768);
    let dim = env_usize("CALYX_ENERGY_FSV_DIM", 384);
    let free_slots = env_usize("CALYX_ENERGY_FSV_FREE_SLOTS", 4);
    let members = deterministic_members(rows, dim);
    let refs = members.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let initials = (0..free_slots)
        .map(|slot| {
            let mut initial = deterministic_initial(dim);
            initial[slot % dim] += slot as f32 * 0.001;
            initial
        })
        .collect::<Vec<_>>();

    let cpu_started = Instant::now();
    let mut cpu_outputs = Vec::with_capacity(free_slots);
    for initial in &initials {
        cpu_outputs.push(cpu_descent(initial, &refs, 1.0, 20, 1.0e-4)?);
    }
    let cpu_us = cpu_started.elapsed().as_micros();

    let gpu_started = Instant::now();
    let mut gpu_outputs = Vec::with_capacity(free_slots);
    for initial in &initials {
        gpu_outputs.push(context.descend(initial, &refs, 1.0, 20, 1.0e-4)?);
    }
    let gpu_us = gpu_started.elapsed().as_micros();
    for (cpu, gpu) in cpu_outputs.iter().zip(&gpu_outputs) {
        parity(cpu, gpu)?;
    }
    let matrix_bytes = rows * dim * size_of::<f32>();
    let h2d = gpu_outputs
        .iter()
        .map(|output| output.stats.host_to_device_bytes)
        .sum::<u64>();
    let d2h = gpu_outputs
        .iter()
        .map(|output| output.stats.device_to_host_bytes)
        .sum::<u64>();
    let peak_device = gpu_outputs
        .iter()
        .map(|output| output.stats.peak_device_bytes)
        .max()
        .unwrap_or(0);
    let peak_pinned = gpu_outputs
        .iter()
        .map(|output| output.stats.peak_pinned_staging_bytes)
        .max()
        .unwrap_or(0);
    let speedup = cpu_us as f64 / gpu_us.max(1) as f64;
    assert!(speedup > 1.0, "realistic GPU speedup={speedup}");
    assert!(h2d < (matrix_bytes * free_slots * 2) as u64);
    assert!(d2h < matrix_bytes as u64);
    let report = json!({
        "format": "calyx-issue1521-energy-fsv-v1",
        "crossover_elements_configured": ENERGY_CUDA_MIN_ELEMENTS,
        "sweep": sweep,
        "realistic": {
            "members": rows,
            "dim": dim,
            "free_slots": free_slots,
            "matrix_bytes_per_slot": matrix_bytes,
            "cpu_us": cpu_us,
            "gpu_us": gpu_us,
            "speedup": speedup,
            "host_to_device_bytes": h2d,
            "device_to_host_bytes": d2h,
            "peak_device_bytes": peak_device,
            "peak_pinned_staging_bytes": peak_pinned,
            "steps": gpu_outputs.iter().map(|output| output.steps_taken).collect::<Vec<_>>(),
            "converged": gpu_outputs.iter().all(|output| output.converged),
        }
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("serialize FSV")
    );
    Ok(())
}

fn warm(context: &CudaEnergyContext) -> Result<()> {
    let members = deterministic_members(64, 32);
    let refs = members.iter().map(Vec::as_slice).collect::<Vec<_>>();
    context
        .descend(&deterministic_initial(32), &refs, 1.0, 2, 1.0e-4)
        .map(|_| ())
}

fn parity(cpu: &super::test_support::CpuDescent, gpu: &CudaEnergyDescent) -> Result<()> {
    if cpu.steps_taken != gpu.steps_taken || cpu.converged != gpu.converged {
        return Err(crate::ForgeError::NumericalInvariant {
            op: "energy.fsv_step_parity".to_string(),
            detail: format!(
                "CPU steps/converged={}/{}, GPU={}/{}",
                cpu.steps_taken, cpu.converged, gpu.steps_taken, gpu.converged
            ),
            remediation: "repair deterministic CUDA energy convergence parity".to_string(),
        });
    }
    let energy_delta = (cpu.final_energy - gpu.final_energy).abs();
    let vector_delta = cpu
        .vector
        .iter()
        .zip(&gpu.vector)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    if energy_delta > 2.0e-4 || vector_delta > 2.0e-4 {
        return Err(crate::ForgeError::NumericalInvariant {
            op: "energy.fsv_value_parity".to_string(),
            detail: format!("energy_delta={energy_delta} vector_delta={vector_delta}"),
            remediation: "repair CUDA energy reduction parity".to_string(),
        });
    }
    Ok(())
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_rows(name: &str, default: &[usize]) -> Vec<usize> {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|part| part.trim().parse().expect("valid row count"))
                .collect()
        })
        .unwrap_or_else(|| default.to_vec())
}
