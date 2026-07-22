//! Emits `cfg(sextant_cuvs)` when the cuVS GPU index paths are actually
//! compiled into this build (#1130): the `cuda` feature is enabled AND the
//! target OS ships libcuvs (Linux only — RAPIDS provides no native
//! Windows/macOS packages, #1016).
//!
//! Source code must gate cuVS usage on `cfg(sextant_cuvs)`, never on
//! `cfg(feature = "cuda")` alone: feature flags are target-independent, so on
//! a non-Linux target the feature can be "on" while the `cuvs-sys`/`cudarc`
//! dependencies (target-gated in Cargo.toml) do not exist.
//!
//! `CARGO_CFG_TARGET_OS` (not `cfg!`) is read because build scripts compile
//! for the host while this decision is about the target.

use std::path::{Path, PathBuf};
use std::process::Command;

const CUDA_PATH_DEFAULT: &str = "/usr/local/cuda-13.3";
/// Upstream develops on GB10 (sm_120). Our resident runs on a local RTX 4090
/// (sm_89); an sm_120 `-cubin` has no SASS for sm_89 and load fails with
/// CUDA_ERROR_NO_BINARY_FOR_GPU (the sextant kernels compile to `-cubin`, i.e.
/// SASS only, no PTX/JIT). `CALYX_CUDA_ARCH` overrides the compile target
/// without editing this upstream const on every import (mirrors the CUDA_PATH
/// override). Set `CALYX_CUDA_ARCH=sm_89` in `_build_cuda.sh` for the 4090.
const CUDA_ARCH: &str = "sm_120";

fn cuda_arch() -> String {
    std::env::var("CALYX_CUDA_ARCH")
        .ok()
        .map(|arch| arch.trim().to_string())
        .filter(|arch| !arch.is_empty())
        .unwrap_or_else(|| CUDA_ARCH.to_string())
}
const MERGE_SOURCE: &str = "src/index/kernels/chunked_exact_merge.cu";
const MERGE_CUBIN_ENV: &str = "SEXTANT_CHUNKED_EXACT_MERGE_CUBIN_PATH";
const MAXSIM_SOURCE: &str = "src/index/kernels/maxsim.cu";
const MAXSIM_CUBIN_ENV: &str = "SEXTANT_MAXSIM_CUBIN_PATH";
const PQ_SOURCE: &str = "src/index/kernels/diskann_pq.cu";
const PQ_CUBIN_ENV: &str = "SEXTANT_DISKANN_PQ_CUBIN_PATH";
const PARTITIONED_SOURCE: &str = "src/index/kernels/partitioned_region_batch.cu";
const PARTITIONED_CUBIN_ENV: &str = "SEXTANT_PARTITIONED_REGION_BATCH_CUBIN_PATH";
const SPARSE_BM25_SOURCE: &str = "src/index/kernels/sparse_bm25.cu";
const SPARSE_BM25_CUBIN_ENV: &str = "SEXTANT_SPARSE_BM25_CUBIN_PATH";

fn main() {
    println!("cargo::rustc-check-cfg=cfg(sextant_cuvs)");
    let cuda_feature = std::env::var_os("CARGO_FEATURE_CUDA").is_some();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS")
        .expect("CALYX_SEXTANT_BUILD: cargo did not set CARGO_CFG_TARGET_OS");
    if cuda_feature && target_os == "linux" {
        println!("cargo::rustc-cfg=sextant_cuvs");
        compile_cuda_kernel(
            MERGE_SOURCE,
            "sextant-chunked-exact-merge.cubin",
            MERGE_CUBIN_ENV,
        );
        compile_cuda_kernel(MAXSIM_SOURCE, "sextant-maxsim.cubin", MAXSIM_CUBIN_ENV);
        compile_cuda_kernel(PQ_SOURCE, "sextant-diskann-pq.cubin", PQ_CUBIN_ENV);
        compile_cuda_kernel(
            PARTITIONED_SOURCE,
            "sextant-partitioned-region-batch.cubin",
            PARTITIONED_CUBIN_ENV,
        );
        compile_cuda_kernel(
            SPARSE_BM25_SOURCE,
            "sextant-sparse-bm25.cubin",
            SPARSE_BM25_CUBIN_ENV,
        );
    }
}

fn compile_cuda_kernel(source: &str, output: &str, output_env: &str) {
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CALYX_CUDA_ARCH");
    println!("cargo:rerun-if-changed={source}");
    let cuda_arch = cuda_arch();
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let source = manifest.join(source);
    assert!(
        source.is_file(),
        "CUDA kernel missing: {}",
        source.display()
    );
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR")).join(output);
    let nvcc = std::env::var_os("CUDA_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(CUDA_PATH_DEFAULT))
        .join("bin/nvcc");
    assert!(nvcc.is_file(), "nvcc missing: {}", nvcc.display());
    let args = [
        format!("-arch={cuda_arch}"),
        "-O3".to_string(),
        "--ftz=false".to_string(),
        "--prec-div=true".to_string(),
        "--prec-sqrt=true".to_string(),
        "--fmad=false".to_string(),
        "-cubin".to_string(),
        "-o".to_string(),
        out.display().to_string(),
        source.display().to_string(),
    ];
    let output = Command::new(&nvcc)
        .args(&args)
        .output()
        .unwrap_or_else(|error| panic!("run {}: {error}", nvcc.display()));
    if !output.status.success() {
        panic!(
            "nvcc failed for {}: {}",
            source.display(),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    emit_kernel_path(output_env, &out);
}

fn emit_kernel_path(name: &str, path: &Path) {
    println!("cargo:rustc-env={name}={}", path.display());
}
