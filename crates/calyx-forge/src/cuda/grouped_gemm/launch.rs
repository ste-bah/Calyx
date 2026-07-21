use cudarc::cublas::{result::CublasError, sys};

use crate::Result;

use super::{ActiveGemmProblem, GemmProblem, cublas_error, to_i32};

pub(super) struct LaunchData {
    trans: Vec<sys::cublasOperation_t>,
    ms: Vec<i32>,
    ns: Vec<i32>,
    ks: Vec<i32>,
    alphas: Vec<f32>,
    betas: Vec<f32>,
    ldas: Vec<i32>,
    ldbs: Vec<i32>,
    ldcs: Vec<i32>,
    group_sizes: Vec<i32>,
    a_ptrs: Vec<u64>,
    b_ptrs: Vec<u64>,
    c_ptrs: Vec<u64>,
    active: Vec<GemmProblem>,
}

impl LaunchData {
    pub(super) fn new(
        active: &[ActiveGemmProblem],
        a_base: u64,
        b_base: u64,
        c_base: u64,
    ) -> Result<Self> {
        let mut data = Self::empty();
        let mut current_dims = None;
        for item in active {
            let problem = item.problem;
            let dims = (problem.m, problem.k, problem.n);
            if current_dims != Some(dims) {
                data.push_group(problem)?;
                current_dims = Some(dims);
            }
            *data.group_sizes.last_mut().expect("group exists") += 1;
            data.a_ptrs.push(offset_address(a_base, problem.a_offset));
            data.b_ptrs.push(offset_address(b_base, problem.b_offset));
            data.c_ptrs.push(offset_address(c_base, problem.c_offset));
            data.active.push(problem);
        }
        Ok(data)
    }

    pub(super) fn group_count(&self) -> usize {
        self.group_sizes.len()
    }

    pub(super) fn a_ptrs(&self) -> &[u64] {
        &self.a_ptrs
    }

    pub(super) fn b_ptrs(&self) -> &[u64] {
        &self.b_ptrs
    }

    pub(super) fn c_ptrs(&self) -> &[u64] {
        &self.c_ptrs
    }

    fn empty() -> Self {
        Self {
            trans: Vec::new(),
            ms: Vec::new(),
            ns: Vec::new(),
            ks: Vec::new(),
            alphas: Vec::new(),
            betas: Vec::new(),
            ldas: Vec::new(),
            ldbs: Vec::new(),
            ldcs: Vec::new(),
            group_sizes: Vec::new(),
            a_ptrs: Vec::new(),
            b_ptrs: Vec::new(),
            c_ptrs: Vec::new(),
            active: Vec::new(),
        }
    }

    fn push_group(&mut self, problem: GemmProblem) -> Result<()> {
        self.trans.push(sys::cublasOperation_t::CUBLAS_OP_N);
        self.ms.push(to_i32(problem.m, "m")?);
        self.ns.push(to_i32(problem.n, "n")?);
        self.ks.push(to_i32(problem.k, "k")?);
        self.alphas.push(1.0);
        self.betas.push(0.0);
        self.ldas.push(to_i32(problem.m, "lda")?);
        self.ldbs.push(to_i32(problem.k, "ldb")?);
        self.ldcs.push(to_i32(problem.m, "ldc")?);
        self.group_sizes.push(0);
        Ok(())
    }
}

pub(super) fn launch_grouped(
    handle: sys::cublasHandle_t,
    data: &LaunchData,
    a_array: *const *const f32,
    b_array: *const *const f32,
    c_array: *const *mut f32,
    group_count: i32,
) -> std::result::Result<(), CublasError> {
    unsafe {
        sys::cublasSgemmGroupedBatched(
            handle,
            data.trans.as_ptr(),
            data.trans.as_ptr(),
            data.ms.as_ptr(),
            data.ns.as_ptr(),
            data.ks.as_ptr(),
            data.alphas.as_ptr(),
            a_array,
            data.ldas.as_ptr(),
            b_array,
            data.ldbs.as_ptr(),
            data.betas.as_ptr(),
            c_array,
            data.ldcs.as_ptr(),
            group_count,
            data.group_sizes.as_ptr(),
        )
        .result()
    }
}

pub(super) fn launch_sequential(handle: sys::cublasHandle_t, data: &LaunchData) -> Result<()> {
    for (idx, problem) in data.active.iter().enumerate() {
        unsafe {
            sys::cublasSgemm_v2(
                handle,
                sys::cublasOperation_t::CUBLAS_OP_N,
                sys::cublasOperation_t::CUBLAS_OP_N,
                to_i32(problem.m, "m")?,
                to_i32(problem.n, "n")?,
                to_i32(problem.k, "k")?,
                &1.0,
                data.a_ptrs[idx] as *const f32,
                to_i32(problem.m, "lda")?,
                data.b_ptrs[idx] as *const f32,
                to_i32(problem.k, "ldb")?,
                &0.0,
                data.c_ptrs[idx] as *mut f32,
                to_i32(problem.m, "ldc")?,
            )
            .result()
        }
        .map_err(|err| cublas_error(format!("sequential cublasSgemm_v2 failed: {err}")))?;
    }
    Ok(())
}

fn offset_address(base: u64, offset: usize) -> u64 {
    base + byte_offset(offset)
}

fn byte_offset(offset: usize) -> u64 {
    (offset * std::mem::size_of::<f32>()) as u64
}
