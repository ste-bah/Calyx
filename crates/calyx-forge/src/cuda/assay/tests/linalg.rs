use super::*;

#[test]
fn linalg_wrappers_match_cpu_oracles() -> Result<()> {
    let _guard = test_lock();
    let ctx = init_cuda(0, false)?;

    let n = 80usize;
    let d = 4usize;
    let columns = linalg_columns(n);
    let expected_corr = cpu_corr_matrix(&columns, n, d);
    let expected_precision = cpu_invert_symmetric(&expected_corr, d).expect("invertible corr");
    let actual = correlation_precision_host(&ctx, &columns, n, d)?;
    assert_eq!(actual.n_samples, n);
    assert_eq!(actual.n_variables, d);
    assert_close_f64_vec("corr", &actual.corr, &expected_corr, 1e-10);
    assert_close_f64_vec("precision", &actual.precision, &expected_precision, 1e-8);
    println!(
        "FORGE_LINALG_SOT corr_precision source=device_readback n={} d={} corr01={} precision01={}",
        actual.n_samples, actual.n_variables, actual.corr[1], actual.precision[1]
    );

    let (x, y) = granger_fixture(96);
    let batch = granger_lag_summaries_host(&ctx, &x, &y, &[1, 2, 0, 33])?;
    assert_eq!(batch.summaries.len(), 4);
    assert_eq!(batch.workspace_row_stride, 5);
    assert_eq!(batch.workspace_bytes, 1_920);
    for p in [1usize, 2usize] {
        let summary = batch
            .summaries
            .iter()
            .find(|summary| summary.lag == p)
            .expect("lag summary");
        let expected = cpu_granger_rss(&x, &y, p).expect("CPU Granger RSS");
        assert_eq!(summary.status, CUDA_GRANGER_STATUS_OK);
        assert_eq!(summary.n_used, x.len() - p);
        assert_eq!(summary.df_den, x.len() - p - (2 * p + 1));
        assert_close_f64("rss_r", summary.rss_restricted, expected.0, 1e-7);
        assert_close_f64("rss_u", summary.rss_unrestricted, expected.1, 1e-7);
    }
    assert_eq!(batch.summaries[2].status, CUDA_GRANGER_STATUS_INVALID_LAG);
    assert_eq!(batch.summaries[3].status, CUDA_GRANGER_STATUS_INVALID_LAG);
    println!(
        "FORGE_LINALG_SOT granger source=device_readback workspace_stride={} workspace_bytes={} summaries={:?}",
        batch.workspace_row_stride, batch.workspace_bytes, batch.summaries
    );

    let (max_x, max_y) = granger_fixture(128);
    let max_batch = granger_lag_summaries_host(&ctx, &max_x, &max_y, &[32])?;
    assert_eq!(max_batch.workspace_row_stride, 65);
    assert_eq!(max_batch.workspace_bytes, 68_640);
    let max_summary = &max_batch.summaries[0];
    let max_expected = cpu_granger_rss(&max_x, &max_y, 32).expect("CPU max-lag Granger RSS");
    assert_eq!(max_summary.status, CUDA_GRANGER_STATUS_OK);
    assert_close_f64(
        "max_lag_rss_r",
        max_summary.rss_restricted,
        max_expected.0,
        1e-7,
    );
    assert_close_f64(
        "max_lag_rss_u",
        max_summary.rss_unrestricted,
        max_expected.1,
        1e-7,
    );
    println!(
        "FORGE_LINALG_SOT granger_max_lag source=device_readback lag={} workspace_stride={} workspace_bytes={} rss_r={} rss_u={}",
        max_summary.lag,
        max_batch.workspace_row_stride,
        max_batch.workspace_bytes,
        max_summary.rss_restricted,
        max_summary.rss_unrestricted
    );
    Ok(())
}

#[test]
fn linalg_wrappers_fail_loud_on_edges() -> Result<()> {
    let _guard = test_lock();
    let ctx = init_cuda(0, false)?;

    let mut constant_columns = linalg_columns(32);
    for row in 0..32 {
        constant_columns[2 * 32 + row] = 7.0;
    }
    let constant = correlation_precision_host(&ctx, &constant_columns, 32, 4)
        .expect_err("constant correlation column must fail");
    assert!(matches!(constant, ForgeError::NumericalInvariant { .. }));
    println!("FORGE_LINALG_EDGE constant_corr_column before=column2_all_7 after={constant}");

    let malformed = correlation_precision_host(&ctx, &[1.0, 2.0, 3.0], 2, 2)
        .expect_err("malformed correlation buffer must fail");
    assert!(matches!(malformed, ForgeError::ShapeMismatch { .. }));
    println!("FORGE_LINALG_EDGE malformed_corr_buffer before=len3_n2_d2 after={malformed}");

    let (mut x, y) = granger_fixture(48);
    x[7] = f32::NAN;
    let nonfinite = granger_lag_summaries_host(&ctx, &x, &y, &[1])
        .expect_err("non-finite Granger input must fail before launch");
    assert!(matches!(nonfinite, ForgeError::NumericalInvariant { .. }));
    println!("FORGE_LINALG_EDGE nonfinite_granger before=x7_nan after={nonfinite}");

    let x_rank = (0..48).map(|idx| (idx % 3) as f32).collect::<Vec<_>>();
    let y_rank = vec![4.0_f32; 48];
    let rank_batch = granger_lag_summaries_host(&ctx, &x_rank, &y_rank, &[1])?;
    assert_eq!(
        rank_batch.summaries[0].status,
        CUDA_GRANGER_STATUS_RANK_DEFICIENT
    );
    println!(
        "FORGE_LINALG_EDGE rank_deficient_granger before=constant_y after_status={}",
        rank_batch.summaries[0].status
    );
    Ok(())
}

#[test]
#[ignore = "manual FSV writes issue1673 GPU workspace evidence"]
fn granger_workspace_fsv() -> Result<()> {
    let _guard = test_lock();
    let before = nvidia_smi_snapshot();
    let ctx = init_cuda(0, false)?;
    let (x, y) = granger_fixture(128);
    let lags = [1usize, 2, 4, 8, 16, 32];
    let batch = granger_lag_summaries_host(&ctx, &x, &y, &lags)?;
    assert_eq!(batch.workspace_row_stride, 65);
    assert_eq!(batch.workspace_bytes, 411_840);
    for summary in &batch.summaries {
        let expected = cpu_granger_rss(&x, &y, summary.lag).expect("CPU Granger RSS");
        assert_eq!(summary.status, CUDA_GRANGER_STATUS_OK);
        assert_close_f64("FSV rss_r", summary.rss_restricted, expected.0, 1e-7);
        assert_close_f64("FSV rss_u", summary.rss_unrestricted, expected.1, 1e-7);
    }
    let invalid = granger_lag_summaries_host(&ctx, &x, &y, &[0, 33])?;
    assert_eq!(invalid.workspace_row_stride, 1);
    assert_eq!(invalid.workspace_bytes, 64);
    assert!(
        invalid
            .summaries
            .iter()
            .all(|summary| summary.status == CUDA_GRANGER_STATUS_INVALID_LAG)
    );
    let after = nvidia_smi_snapshot();
    let root = std::env::var_os("CALYX_FORGE_ISSUE1673_FSV_DIR")
        .map(std::path::PathBuf::from)
        .expect("CALYX_FORGE_ISSUE1673_FSV_DIR is required for manual FSV");
    std::fs::create_dir_all(&root).expect("create issue1673 FSV directory");
    let root = std::fs::canonicalize(&root).expect("canonicalize issue1673 FSV directory");
    let path = root.join("issue1673-granger-workspace-fsv.json");
    let artifact = serde_json::json!({
        "artifact_kind": "issue1673.granger-cuda-workspace-fsv.v1",
        "source_of_truth": path.display().to_string(),
        "trigger": "cargo test -p calyx-forge --features cuda cuda::assay::tests::linalg::granger_workspace_fsv -- --ignored --nocapture",
        "gpu_before": before,
        "gpu_after": after,
        "minimum_sufficient_corpus": {
            "samples": x.len(),
            "lags": lags,
            "why_smaller_insufficient": "lag 32 is required to prove the maximum supported 65-column solver workspace and compiled launch path",
            "why_larger_wasteful": "larger samples do not increase the bounded normal-equation workspace"
        },
        "workspace": {
            "storage": "explicit_global_device_buffers",
            "row_stride": batch.workspace_row_stride,
            "bytes": batch.workspace_bytes,
            "former_per_thread_stack_bytes": 68640,
        },
        "happy_path": batch.summaries.iter().map(granger_summary_json).collect::<Vec<_>>(),
        "edge_case": {
            "lags": [0, 33],
            "row_stride": invalid.workspace_row_stride,
            "bytes": invalid.workspace_bytes,
            "summaries": invalid.summaries.iter().map(granger_summary_json).collect::<Vec<_>>()
        }
    });
    let encoded = serde_json::to_vec_pretty(&artifact).expect("encode issue1673 FSV");
    std::fs::write(&path, &encoded).expect("write issue1673 FSV");
    let readback = std::fs::read(&path).expect("read issue1673 FSV");
    let restored: serde_json::Value =
        serde_json::from_slice(&readback).expect("decode issue1673 FSV");
    assert_eq!(restored, artifact);
    println!(
        "ISSUE1673_GRANGER_WORKSPACE_FSV path={} bytes={}",
        path.display(),
        readback.len()
    );
    Ok(())
}

fn granger_summary_json(summary: &CudaGrangerLagSummary) -> serde_json::Value {
    serde_json::json!({
        "lag": summary.lag,
        "rss_restricted": summary.rss_restricted,
        "rss_unrestricted": summary.rss_unrestricted,
        "n_used": summary.n_used,
        "df_den": summary.df_den,
        "status": summary.status,
    })
}

fn nvidia_smi_snapshot() -> String {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ])
        .output()
        .expect("run nvidia-smi for issue1673 FSV");
    assert!(
        output.status.success(),
        "nvidia-smi failed: status={} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("nvidia-smi emitted UTF-8");
    assert_eq!(stdout.lines().filter(|line| !line.is_empty()).count(), 1);
    stdout.trim().to_string()
}
