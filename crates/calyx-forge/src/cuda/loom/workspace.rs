use std::mem::size_of;

use cudarc::driver::{CudaSlice, PinnedHostSlice, ValidAsZeroBits};

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LoomShape {
    pub(super) matrix_len: usize,
    gram_len: usize,
    pub(super) row_count: usize,
    pub(super) dim: usize,
    pub(super) agreement_count: usize,
    pub(super) vector_count: usize,
    pub(super) vector_len: usize,
    pub(super) device_bytes: usize,
}

impl LoomShape {
    pub(super) fn new(rows: usize, dim: usize, agreements: usize, vectors: usize) -> Result<Self> {
        let matrix_len = checked_mul(rows, dim, "Loom matrix length")?;
        let gram_len = if agreements > 0 {
            checked_mul(rows, rows, "Loom Gram length")?
        } else {
            0
        };
        let vector_len = checked_mul(vectors, dim, "Loom vector output length")?;
        let f32_count = checked_sum(&[
            matrix_len,
            if agreements > 0 { matrix_len } else { 0 },
            gram_len,
            agreements,
            vector_len,
        ])?;
        let agreement_indices = checked_mul(agreements, 2, "Loom agreement indices")?;
        let vector_metadata = checked_mul(vectors, 3, "Loom vector metadata")?;
        let u32_count = checked_sum(&[agreement_indices, vector_metadata, 1])?;
        let device_bytes = checked_sum(&[
            checked_mul(f32_count, size_of::<f32>(), "Loom f32 bytes")?,
            checked_mul(u32_count, size_of::<u32>(), "Loom u32 bytes")?,
        ])?;
        Ok(Self {
            matrix_len,
            gram_len,
            row_count: rows,
            dim,
            agreement_count: agreements,
            vector_count: vectors,
            vector_len,
            device_bytes,
        })
    }
}

pub(super) struct PairBuffers {
    pub(super) left_host: PinnedHostSlice<u32>,
    pub(super) right_host: PinnedHostSlice<u32>,
    pub(super) left_device: CudaSlice<u32>,
    pub(super) right_device: CudaSlice<u32>,
}

pub(super) struct VectorBuffers {
    pub(super) pairs: PairBuffers,
    pub(super) kinds_host: PinnedHostSlice<u32>,
    pub(super) kinds_device: CudaSlice<u32>,
    pub(super) output: CudaSlice<f32>,
}

pub(super) struct LoomWorkspace {
    pub(super) shape: LoomShape,
    pub(super) matrix_host: PinnedHostSlice<f32>,
    pub(super) matrix_device: CudaSlice<f32>,
    pub(super) normalized: Option<CudaSlice<f32>>,
    pub(super) gram: Option<CudaSlice<f32>>,
    pub(super) agreements: Option<PairBuffers>,
    pub(super) agreement_output: Option<CudaSlice<f32>>,
    pub(super) vectors: Option<VectorBuffers>,
    pub(super) flags: CudaSlice<u32>,
}

impl LoomWorkspace {
    pub(super) fn new(ctx: &CudaContext, shape: LoomShape) -> Result<Self> {
        let stream = ctx.inner().default_stream();
        let agreements = (shape.agreement_count > 0)
            .then(|| PairBuffers::new(ctx, shape.agreement_count))
            .transpose()?;
        let vectors = (shape.vector_count > 0)
            .then(|| VectorBuffers::new(ctx, shape.vector_count, shape.vector_len))
            .transpose()?;
        Ok(Self {
            shape,
            matrix_host: pinned_zeros(ctx, shape.matrix_len, "matrix host")?,
            matrix_device: alloc_zeros(ctx, shape.matrix_len, "matrix device")?,
            normalized: (shape.agreement_count > 0)
                .then(|| alloc_zeros(ctx, shape.matrix_len, "normalized rows"))
                .transpose()?,
            gram: (shape.agreement_count > 0)
                .then(|| alloc_zeros(ctx, shape.gram_len, "Gram matrix"))
                .transpose()?,
            agreements,
            agreement_output: (shape.agreement_count > 0)
                .then(|| alloc_zeros(ctx, shape.agreement_count, "agreement output"))
                .transpose()?,
            vectors,
            flags: stream
                .alloc_zeros(1)
                .map_err(|err| device_error(ctx, "flag allocation", err))?,
        })
    }

    pub(super) fn upload(
        &mut self,
        ctx: &CudaContext,
        matrix: &[f32],
        agreements: &[(usize, usize)],
        vectors: &[CudaLoomVectorRequest],
    ) -> Result<()> {
        self.matrix_host
            .as_mut_slice()
            .map_err(|err| device_error(ctx, "matrix host access", err))?
            .copy_from_slice(matrix);
        let stream = ctx.inner().default_stream();
        stream
            .memcpy_htod(&self.matrix_host, &mut self.matrix_device)
            .map_err(|err| device_error(ctx, "matrix upload", err))?;
        if let Some(buffers) = self.agreements.as_mut() {
            buffers.fill(ctx, agreements.iter().copied())?;
        }
        if let Some(buffers) = self.vectors.as_mut() {
            buffers.pairs.fill(
                ctx,
                vectors
                    .iter()
                    .map(|request| (request.left_row, request.right_row)),
            )?;
            let host = buffers
                .kinds_host
                .as_mut_slice()
                .map_err(|err| device_error(ctx, "vector kind host access", err))?;
            for (value, request) in host.iter_mut().zip(vectors) {
                *value = request.kind as u32;
            }
            stream
                .memcpy_htod(&buffers.kinds_host, &mut buffers.kinds_device)
                .map_err(|err| device_error(ctx, "vector kind upload", err))?;
        }
        Ok(())
    }

    pub(super) fn reset_outputs(&mut self, ctx: &CudaContext) -> Result<()> {
        let stream = ctx.inner().default_stream();
        stream
            .memset_zeros(&mut self.flags)
            .map_err(|err| device_error(ctx, "flag reset", err))?;
        for output in [
            self.normalized.as_mut(),
            self.gram.as_mut(),
            self.agreement_output.as_mut(),
        ]
        .into_iter()
        .flatten()
        {
            stream
                .memset_zeros(output)
                .map_err(|err| device_error(ctx, "output reset", err))?;
        }
        if let Some(vectors) = self.vectors.as_mut() {
            stream
                .memset_zeros(&mut vectors.output)
                .map_err(|err| device_error(ctx, "vector reset", err))?;
        }
        Ok(())
    }
}

impl PairBuffers {
    fn new(ctx: &CudaContext, len: usize) -> Result<Self> {
        Ok(Self {
            left_host: pinned_zeros(ctx, len, "left rows host")?,
            right_host: pinned_zeros(ctx, len, "right rows host")?,
            left_device: alloc_zeros(ctx, len, "left rows device")?,
            right_device: alloc_zeros(ctx, len, "right rows device")?,
        })
    }

    fn fill(
        &mut self,
        ctx: &CudaContext,
        pairs: impl Iterator<Item = (usize, usize)>,
    ) -> Result<()> {
        let left = self
            .left_host
            .as_mut_slice()
            .map_err(|err| device_error(ctx, "left host access", err))?;
        let right = self
            .right_host
            .as_mut_slice()
            .map_err(|err| device_error(ctx, "right host access", err))?;
        for ((left_out, right_out), (left_in, right_in)) in left.iter_mut().zip(right).zip(pairs) {
            *left_out = to_u32(left_in, "left row")?;
            *right_out = to_u32(right_in, "right row")?;
        }
        let stream = ctx.inner().default_stream();
        stream
            .memcpy_htod(&self.left_host, &mut self.left_device)
            .map_err(|err| device_error(ctx, "left rows upload", err))?;
        stream
            .memcpy_htod(&self.right_host, &mut self.right_device)
            .map_err(|err| device_error(ctx, "right rows upload", err))?;
        Ok(())
    }
}

impl VectorBuffers {
    fn new(ctx: &CudaContext, count: usize, output_len: usize) -> Result<Self> {
        Ok(Self {
            pairs: PairBuffers::new(ctx, count)?,
            kinds_host: pinned_zeros(ctx, count, "vector kinds host")?,
            kinds_device: alloc_zeros(ctx, count, "vector kinds device")?,
            output: alloc_zeros(ctx, output_len, "vector output")?,
        })
    }
}

fn pinned_zeros<T: cudarc::driver::DeviceRepr + ValidAsZeroBits>(
    ctx: &CudaContext,
    len: usize,
    label: &str,
) -> Result<PinnedHostSlice<T>> {
    let mut buffer = unsafe { ctx.inner().alloc_pinned::<T>(len) }
        .map_err(|err| device_error(ctx, label, err))?;
    let pointer = buffer
        .as_mut_ptr()
        .map_err(|err| device_error(ctx, label, err))?;
    unsafe { pointer.write_bytes(0, len) };
    Ok(buffer)
}

fn alloc_zeros<T: cudarc::driver::DeviceRepr + ValidAsZeroBits>(
    ctx: &CudaContext,
    len: usize,
    label: &str,
) -> Result<CudaSlice<T>> {
    ctx.inner()
        .default_stream()
        .alloc_zeros(len)
        .map_err(|err| device_error(ctx, label, err))
}
