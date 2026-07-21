use std::fs;
use std::path::Path;
use std::time::Instant;

use serde_json::{Value, json};

use super::test_support::{cpu_distances, cpu_mst, dense_slot};
use super::*;
use crate::{ForgeError, Result};

#[test]
#[ignore = "requires an explicit CUDA acceptance host"]
fn issue1517_cuda_skill_crossover_and_cap_fsv() -> Result<()> {
    let _guard = crate::cuda::test_lock();
    let context = CudaSkillContext::new(0)?;
    context.minimum_spanning_tree(32, &[dense_slot(32, 16, 3)], 3)?;

    let point_counts = env_values("CALYX_SKILL_FSV_POINTS", &[64, 128, 256, 512, 1_024]);
    let dimensions = env_values("CALYX_SKILL_FSV_DIMS", &[32, 128, 384]);
    let mut sweep = Vec::new();
    for dim in dimensions.iter().copied() {
        for points in point_counts.iter().copied() {
            sweep.push(measure(&context, points, dim, 11)?);
        }
    }

    let realistic_points = env_usize("CALYX_SKILL_FSV_CAP_POINTS", CUDA_SKILL_MAX_POINTS);
    let realistic_dim = env_usize("CALYX_SKILL_FSV_CAP_DIM", 384);
    let realistic = measure(&context, realistic_points, realistic_dim, 29)?;
    let speedup = realistic["speedup"].as_f64().unwrap_or(0.0);
    if speedup <= 1.0 {
        return Err(numerical(format!(
            "full-cap CUDA speedup must exceed 1.0, measured {speedup:.3}"
        )));
    }
    let crossovers = dimensions
        .iter()
        .map(|dim| {
            let first = sweep.iter().find_map(|row| {
                (row["dim"] == *dim as u64 && row["speedup"].as_f64().unwrap_or(0.0) > 1.0)
                    .then(|| row["points"].as_u64().unwrap_or(0))
            });
            json!({"dim": dim, "first_faster_points": first})
        })
        .collect::<Vec<_>>();
    let report = json!({
        "format": "calyx-issue1517-skill-cuda-fsv-v1",
        "configured_crossover_points": SKILL_CUDA_MIN_POINTS,
        "max_points": CUDA_SKILL_MAX_POINTS,
        "max_device_bytes": CUDA_SKILL_MAX_DEVICE_BYTES,
        "sweep": sweep,
        "measured_crossovers": crossovers,
        "realistic_cap": realistic,
    });
    write_fsv("issue1517_skill_cuda_fsv.json", &report)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("serialize issue1517 FSV")
    );
    Ok(())
}

fn measure(context: &CudaSkillContext, points: usize, dim: usize, salt: usize) -> Result<Value> {
    let slots = vec![dense_slot(points, dim, salt)];
    let cpu_started = Instant::now();
    let distances = cpu_distances(points, &slots)?;
    let cpu_edges = cpu_mst(points, &distances, 5);
    let cpu_us = cpu_started.elapsed().as_micros();

    let gpu_started = Instant::now();
    let gpu = context.minimum_spanning_tree(points, &slots, 5)?;
    let gpu_us = gpu_started.elapsed().as_micros();
    let max_weight_delta = edge_parity(&cpu_edges, &gpu.edges)?;
    let expected_h2d =
        points * dim * size_of::<f32>() + points * size_of::<i64>() + size_of::<i32>();
    let expected_d2h = size_of::<u32>() + (points - 1) * (size_of::<u32>() * 2 + size_of::<f64>());
    if gpu.stats.host_to_device_bytes != expected_h2d as u64
        || gpu.stats.device_to_host_bytes != expected_d2h as u64
        || gpu.stats.peak_device_bytes > CUDA_SKILL_MAX_DEVICE_BYTES as u64
        || gpu.stats.full_distance_readback
    {
        return Err(numerical(format!(
            "transfer/VRAM invariant failed for points={points} dim={dim}: {:?}",
            gpu.stats
        )));
    }
    Ok(json!({
        "points": points,
        "dim": dim,
        "cpu_us": cpu_us,
        "gpu_us": gpu_us,
        "speedup": cpu_us as f64 / gpu_us.max(1) as f64,
        "max_mst_weight_delta": max_weight_delta,
        "host_to_device_bytes": gpu.stats.host_to_device_bytes,
        "device_to_host_bytes": gpu.stats.device_to_host_bytes,
        "peak_device_bytes": gpu.stats.peak_device_bytes,
        "kernel_launches": gpu.stats.kernel_launches,
    }))
}

fn edge_parity(expected: &[(usize, usize, f64)], actual: &[CudaSkillEdge]) -> Result<f64> {
    if expected.len() != actual.len() {
        return Err(numerical(format!(
            "MST edge count differs: CPU={} CUDA={}",
            expected.len(),
            actual.len()
        )));
    }
    let mut max_delta = 0.0_f64;
    for (cpu, gpu) in expected.iter().zip(actual) {
        if (cpu.0, cpu.1) != (gpu.source, gpu.destination) {
            return Err(numerical(format!(
                "MST edge differs: CPU=({}, {}) CUDA=({}, {})",
                cpu.0, cpu.1, gpu.source, gpu.destination
            )));
        }
        max_delta = max_delta.max((cpu.2 - gpu.weight).abs());
    }
    if max_delta > 1.0e-7 {
        return Err(numerical(format!(
            "MST weight tolerance exceeded: delta={max_delta}"
        )));
    }
    Ok(max_delta)
}

fn write_fsv(name: &str, payload: &Value) -> Result<()> {
    let Some(root) = calyx_fsv::fsv_root("CALYX_FSV_ROOT") else {
        return Ok(());
    };
    fs::create_dir_all(&root).map_err(|error| fsv_error("mkdir", &root, error))?;
    let path = root.join(name);
    let bytes =
        serde_json::to_vec_pretty(payload).map_err(|error| fsv_error("serialize", &path, error))?;
    fs::write(&path, &bytes).map_err(|error| fsv_error("write", &path, error))?;
    let readback = fs::read(&path).map_err(|error| fsv_error("read", &path, error))?;
    if readback != bytes {
        return Err(fsv_error("readback", &path, "artifact bytes differ"));
    }
    Ok(())
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_values(name: &str, default: &[usize]) -> Vec<usize> {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|part| part.trim().parse().expect("valid positive integer"))
                .collect()
        })
        .unwrap_or_else(|| default.to_vec())
}

fn numerical(detail: String) -> ForgeError {
    ForgeError::NumericalInvariant {
        op: "skill.issue1517_fsv".to_string(),
        detail,
        remediation: "repair deterministic CUDA skill parity or routing".to_string(),
    }
}

fn fsv_error(op: &str, path: &Path, detail: impl ToString) -> ForgeError {
    ForgeError::CacheError {
        op: format!("skill_fsv_{op}"),
        path: path.display().to_string(),
        detail: detail.to_string(),
        remediation: "repair CALYX_FSV_ROOT and rerun the issue1517 acceptance".to_string(),
    }
}
