#[cfg(feature = "cuda")]
use calyx_assay::{
    DcorPermConfig, HsicConfig, HsicPermConfig, MmdConfig, distance_correlation_test,
    distance_correlation_test_cuda_strict, gaussian_mmd_with_config,
    gaussian_mmd_with_config_cuda_strict, hsic_estimators_with_config,
    hsic_estimators_with_config_cuda_strict, hsic_permutation_test,
    hsic_permutation_test_cuda_strict, mmd_change_point, mmd_change_point_cuda_strict,
};
#[cfg(not(feature = "cuda"))]
use calyx_assay::{
    MmdConfig, distance_correlation_cuda_strict, gaussian_mmd_cuda_strict,
    mmd_change_point_cuda_strict,
};

#[cfg(feature = "cuda")]
fn assert_close(actual: f64, expected: f64, tol: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= tol,
        "{label}: actual={actual} expected={expected} tol={tol}"
    );
}

#[cfg(not(feature = "cuda"))]
#[test]
fn strict_cuda_entrypoints_fail_loud_without_cuda_feature() {
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let y = [2.0f32, 4.0, 6.0, 8.0];
    let err = distance_correlation_cuda_strict(&x, &y)
        .expect_err("strict dCor must fail loud without cuda feature");
    println!("ISSUE1502_NO_CUDA_DCOR_ERR {err}");
    assert_eq!(err.code, "CALYX_FORGE_DEVICE_UNAVAILABLE");
    assert!(
        err.message
            .contains("strict mode does not fall back to CPU")
    );

    let mmd_x = vec![vec![0.0, 0.0]; 4];
    let mmd_y = vec![vec![1.0, 1.0]; 4];
    let err = gaussian_mmd_cuda_strict(&mmd_x, &mmd_y)
        .expect_err("strict MMD must fail loud without cuda feature");
    println!("ISSUE1502_NO_CUDA_MMD_ERR {err}");
    assert_eq!(err.code, "CALYX_FORGE_DEVICE_UNAVAILABLE");

    let mut stream = mmd_x.clone();
    stream.extend(mmd_y.clone());
    let err = mmd_change_point_cuda_strict(&stream, 4, &MmdConfig::default())
        .expect_err("strict MMD change-point must fail loud without cuda feature");
    println!("ISSUE1502_NO_CUDA_MMD_CHANGE_ERR {err}");
    assert_eq!(err.code, "CALYX_FORGE_DEVICE_UNAVAILABLE");
}

#[cfg(feature = "cuda")]
#[test]
fn strict_cuda_matches_cpu_for_dcor_hsic_and_mmd() {
    let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y = [1.0f32, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0];
    let dcor_cfg = DcorPermConfig {
        permutations: 23,
        seed: 1502,
    };
    let dcor_cpu = distance_correlation_test(&x, &y, dcor_cfg).unwrap();
    let dcor_gpu = distance_correlation_test_cuda_strict(&x, &y, dcor_cfg).unwrap();
    println!("ISSUE1502_DCOR_CPU {dcor_cpu:?}");
    println!("ISSUE1502_DCOR_GPU {dcor_gpu:?}");
    assert_close(dcor_gpu.dcor as f64, dcor_cpu.dcor as f64, 2.0e-5, "dCor");
    assert_close(
        dcor_gpu.dcov2 as f64,
        dcor_cpu.dcov2 as f64,
        2.0e-5,
        "dCor dcov2",
    );
    assert_eq!(dcor_gpu.ge_count, dcor_cpu.ge_count);

    let hsic_cfg = HsicConfig {
        bandwidth_x: Some(1.25),
        bandwidth_y: Some(3.5),
    };
    let hsic_cpu = hsic_estimators_with_config(&x, &y, hsic_cfg).unwrap();
    let hsic_gpu = hsic_estimators_with_config_cuda_strict(&x, &y, hsic_cfg).unwrap();
    println!("ISSUE1502_HSIC_CPU {hsic_cpu:?}");
    println!("ISSUE1502_HSIC_GPU {hsic_gpu:?}");
    assert_close(
        hsic_gpu.hsic_biased as f64,
        hsic_cpu.hsic_biased as f64,
        2.0e-5,
        "HSIC biased",
    );
    assert_close(
        hsic_gpu.hsic_unbiased as f64,
        hsic_cpu.hsic_unbiased as f64,
        2.0e-5,
        "HSIC unbiased",
    );

    let hsic_perm_cfg = HsicPermConfig {
        kernel: hsic_cfg,
        permutations: 23,
        seed: 2501,
    };
    let hsic_perm_cpu = hsic_permutation_test(&x, &y, hsic_perm_cfg).unwrap();
    let hsic_perm_gpu = hsic_permutation_test_cuda_strict(&x, &y, hsic_perm_cfg).unwrap();
    println!("ISSUE1502_HSIC_PERM_CPU {hsic_perm_cpu:?}");
    println!("ISSUE1502_HSIC_PERM_GPU {hsic_perm_gpu:?}");
    assert_eq!(hsic_perm_gpu.ge_count, hsic_perm_cpu.ge_count);
    assert_close(
        hsic_perm_gpu.p_value as f64,
        hsic_perm_cpu.p_value as f64,
        1.0e-7,
        "HSIC p",
    );

    let mmd_x = vec![
        vec![0.0, 0.0],
        vec![0.2, 0.1],
        vec![0.1, -0.1],
        vec![0.3, 0.0],
    ];
    let mmd_y = vec![
        vec![1.0, 1.0],
        vec![1.2, 0.9],
        vec![0.9, 1.1],
        vec![1.1, 1.2],
    ];
    let mmd_cfg = MmdConfig {
        bandwidth: Some(0.75),
        permutations: 23,
        seed: 3501,
        alpha: 0.05,
    };
    let mmd_cpu = gaussian_mmd_with_config(&mmd_x, &mmd_y, &mmd_cfg).unwrap();
    let mmd_gpu = gaussian_mmd_with_config_cuda_strict(&mmd_x, &mmd_y, &mmd_cfg).unwrap();
    println!("ISSUE1502_MMD_CPU {mmd_cpu:?}");
    println!("ISSUE1502_MMD_GPU {mmd_gpu:?}");
    assert_close(mmd_gpu.mmd2, mmd_cpu.mmd2, 2.0e-10, "MMD2");
    assert_close(
        mmd_gpu.null_mean,
        mmd_cpu.null_mean,
        2.0e-10,
        "MMD null mean",
    );
    assert_close(
        mmd_gpu.critical_value,
        mmd_cpu.critical_value,
        2.0e-10,
        "MMD critical",
    );
    assert_close(mmd_gpu.p_value, mmd_cpu.p_value, 1.0e-12, "MMD p");
    assert_eq!(mmd_gpu.significant, mmd_cpu.significant);

    let mut stream = mmd_x.clone();
    stream.extend(mmd_y.clone());
    let change_cpu = mmd_change_point(&stream, 4, &mmd_cfg).unwrap();
    let change_gpu = mmd_change_point_cuda_strict(&stream, 4, &mmd_cfg).unwrap();
    println!("ISSUE1502_MMD_CHANGE_CPU {change_cpu:?}");
    println!("ISSUE1502_MMD_CHANGE_GPU {change_gpu:?}");
    assert_eq!(change_gpu.split_index, change_cpu.split_index);
    assert_close(
        change_gpu.report.mmd2,
        change_cpu.report.mmd2,
        2.0e-10,
        "MMD change mmd2",
    );
    assert_close(
        change_gpu.report.null_mean,
        change_cpu.report.null_mean,
        2.0e-10,
        "MMD change null mean",
    );
    assert_close(
        change_gpu.report.critical_value,
        change_cpu.report.critical_value,
        2.0e-10,
        "MMD change critical",
    );

    let edge_dcor =
        distance_correlation_test_cuda_strict(&[1.0f32, 2.0, 3.0], &[1.0f32, 2.0, 3.0], dcor_cfg)
            .expect_err("dCor n<4 must fail closed");
    let edge_hsic = hsic_estimators_with_config_cuda_strict(
        &[1.0f32, f32::NAN, 3.0, 4.0],
        &[1.0f32, 2.0, 3.0, 4.0],
        hsic_cfg,
    )
    .expect_err("HSIC non-finite input must fail closed");
    let edge_mmd = gaussian_mmd_with_config_cuda_strict(
        &[vec![0.0, 0.0], vec![0.1], vec![0.2, 0.2], vec![0.3, 0.3]],
        &mmd_y,
        &mmd_cfg,
    )
    .expect_err("MMD ragged input must fail closed");
    println!(
        "ISSUE1502_EDGE_BEFORE dcor={{x_len:3,y_len:3}} hsic={{nan_index:1}} mmd={{row_dims:[2,1,2,2]}}"
    );
    println!(
        "ISSUE1502_EDGE_AFTER dcor={} hsic={} mmd={}",
        edge_dcor.code, edge_hsic.code, edge_mmd.code
    );

    write_fsv_artifact(serde_json::json!({
        "artifact_kind": "issue1502.assay-cuda-strict-fsv.v1",
        "source_of_truth": "CALYX_ASSAY_ISSUE1502_FSV_DIR/issue1502-fsv-readback.json",
        "trigger": "cargo test -p calyx-assay --features cuda --test __calyx_integration_suite_issue1502_1504_cuda issue1502_assay_cuda_strict -- --nocapture",
        "happy_path": {
            "dcor": {"cpu": dcor_cpu, "gpu": dcor_gpu},
            "hsic": {"cpu": hsic_cpu, "gpu": hsic_gpu},
            "hsic_permutation": {"cpu": hsic_perm_cpu, "gpu": hsic_perm_gpu},
            "mmd": {"cpu": mmd_cpu, "gpu": mmd_gpu},
            "mmd_change_point": {"cpu": change_cpu, "gpu": change_gpu}
        },
        "edges": [
            {
                "name": "dcor_too_few",
                "before": {"x_len": 3, "y_len": 3, "min_required": 4},
                "after": {"code": edge_dcor.code, "message": edge_dcor.message}
            },
            {
                "name": "hsic_nonfinite",
                "before": {"x_len": 4, "y_len": 4, "nonfinite_index": 1},
                "after": {"code": edge_hsic.code, "message": edge_hsic.message}
            },
            {
                "name": "mmd_ragged",
                "before": {"side": "A", "row_dimensions": [2, 1, 2, 2]},
                "after": {"code": edge_mmd.code, "message": edge_mmd.message}
            }
        ]
    }));
}

#[cfg(feature = "cuda")]
fn write_fsv_artifact(value: serde_json::Value) {
    let Ok(root) = std::env::var("CALYX_ASSAY_ISSUE1502_FSV_DIR") else {
        return;
    };
    let root = std::path::PathBuf::from(root);
    std::fs::create_dir_all(&root).expect("create issue1502 FSV dir");
    let path = root.join("issue1502-fsv-readback.json");
    let bytes = serde_json::to_vec_pretty(&value).expect("serialize issue1502 FSV artifact");
    std::fs::write(&path, bytes).expect("write issue1502 FSV artifact");
    let readback = std::fs::read(&path).expect("read issue1502 FSV artifact");
    println!(
        "ISSUE1502_FSV_READBACK path={} bytes={}",
        path.display(),
        readback.len()
    );
    assert!(!readback.is_empty());
}
