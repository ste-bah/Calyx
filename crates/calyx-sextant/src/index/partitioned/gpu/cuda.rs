mod support;

use std::sync::Arc;
use std::time::Instant;

use calyx_core::Result;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, PinnedHostSlice};
use rayon::prelude::*;

use crate::index::SpannCentroidIndex;

use super::super::{PartitionBuildDiagnostics, VectorSource};
use super::cuvs::{BruteForceIndex, Resources, fit_balanced_kmeans};
use support::*;

const KMEANS_ITERS: usize = 12;
const DEFAULT_RESIDENT_MIB: usize = 512;
const DEFAULT_CHUNK_MIB: usize = 128;

pub(in crate::index::partitioned) struct PartitionGpu {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    resources: Resources,
    resident: Option<CudaSlice<f32>>,
    query_host: PinnedHostSlice<f32>,
    query_device: CudaSlice<f32>,
    row_count: usize,
    dim: usize,
    chunk_rows: usize,
    diagnostics: PartitionBuildDiagnostics,
}

impl PartitionGpu {
    pub(super) fn new(
        source: &dyn VectorSource,
        requested_chunk_rows: usize,
        sample_rows: usize,
        initial_centroids: usize,
    ) -> Result<Self> {
        let row_count =
            usize::try_from(source.len()).map_err(|_| invalid("source row count exceeds usize"))?;
        let dim = source.dim();
        let row_bytes = byte_len::<f32>(dim)?;
        let corpus_bytes = row_count
            .checked_mul(row_bytes)
            .ok_or_else(|| invalid("corpus byte size overflow"))?;
        let resident_limit = mib_bytes(
            "CALYX_PARTITION_GPU_RESIDENT_MIB",
            DEFAULT_RESIDENT_MIB,
            true,
        )?;
        let chunk_limit = mib_bytes("CALYX_PARTITION_GPU_CHUNK_MIB", DEFAULT_CHUNK_MIB, false)?;
        let resident_corpus = corpus_bytes <= resident_limit;
        let chunk_rows = requested_chunk_rows
            .max(1)
            .min((chunk_limit / row_bytes).max(1))
            .min(row_count.max(1));
        let query_values = chunk_rows
            .checked_mul(dim)
            .ok_or_else(|| invalid("query staging shape overflow"))?;

        let context = CudaContext::new(0).map_err(cuda_error("context init"))?;
        let stream = context.default_stream();
        let resources = Resources::new()?;
        let mut diagnostics = PartitionBuildDiagnostics {
            backend: "cuvs-balanced-kmeans-bruteforce-v1".to_string(),
            strict_gpu_required: true,
            row_count: source.len(),
            dim,
            sample_rows,
            initial_centroids,
            chunk_rows,
            resident_corpus,
            ..PartitionBuildDiagnostics::default()
        };
        let resident = if resident_corpus {
            let mut host = pinned_zeros(&context, row_count * dim, "resident pinned allocation")?;
            fill_range(source, 0, row_count, dim, &mut host)?;
            let device = stream
                .clone_htod(&host)
                .map_err(cuda_error("resident corpus upload"))?;
            stream
                .synchronize()
                .map_err(cuda_error("resident corpus upload sync"))?;
            diagnostics.corpus_uploads = 1;
            diagnostics.rows_uploaded = source.len();
            diagnostics.h2d_transfers = 1;
            diagnostics.h2d_bytes = to_u64(corpus_bytes)?;
            diagnostics.peak_device_bytes = to_u64(corpus_bytes)?;
            diagnostics.peak_pinned_host_bytes = to_u64(corpus_bytes)?;
            Some(device)
        } else {
            None
        };
        let query_host = pinned_zeros(&context, query_values, "query pinned allocation")?;
        let query_device = alloc_device(&stream, query_values, "query device allocation")?;
        diagnostics.peak_device_bytes = diagnostics
            .peak_device_bytes
            .max(to_u64(corpus_bytes_if(&resident))? + to_u64(byte_len::<f32>(query_values)?)?);
        diagnostics.peak_pinned_host_bytes = diagnostics
            .peak_pinned_host_bytes
            .max(to_u64(byte_len::<f32>(query_values)?)?);
        Ok(Self {
            context,
            stream,
            resources,
            resident,
            query_host,
            query_device,
            row_count,
            dim,
            chunk_rows,
            diagnostics,
        })
    }

    pub(in crate::index::partitioned) fn fit_centroids(
        &mut self,
        rows: &[(u32, Vec<f32>)],
        requested_clusters: usize,
        seed: u64,
    ) -> Result<SpannCentroidIndex> {
        let started = Instant::now();
        if rows.is_empty() || requested_clusters == 0 {
            return Err(invalid("GPU k-means requires rows and clusters"));
        }
        let dim = rows[0].1.len();
        validate_rows(rows, dim)?;
        let clusters = requested_clusters.min(rows.len());
        let sample_values = rows
            .len()
            .checked_mul(dim)
            .ok_or_else(|| invalid("k-means sample shape overflow"))?;
        let centroid_values = clusters
            .checked_mul(dim)
            .ok_or_else(|| invalid("k-means centroid shape overflow"))?;
        let mut sample_host =
            pinned_zeros(&self.context, sample_values, "sample pinned allocation")?;
        flatten_rows_seeded(rows, seed, &mut sample_host)?;
        self.track_pinned(byte_len::<f32>(sample_values)?)?;
        let samples = self
            .stream
            .clone_htod(&sample_host)
            .map_err(cuda_error("k-means sample upload"))?;
        self.stream
            .synchronize()
            .map_err(cuda_error("k-means sample upload sync"))?;
        self.record_h2d(byte_len::<f32>(sample_values)?, 1, 0)?;
        let mut centroids = alloc_device(&self.stream, centroid_values, "centroid allocation")?;
        let operation_values = sample_values
            .checked_add(centroid_values)
            .ok_or_else(|| invalid("k-means device workspace overflow"))?;
        self.track_device(byte_len::<f32>(operation_values)?)?;
        fit_balanced_kmeans(
            &self.resources,
            &self.stream,
            &samples,
            rows.len(),
            dim,
            &mut centroids,
            clusters,
            KMEANS_ITERS,
        )?;
        let flat = self
            .stream
            .clone_dtoh(&centroids)
            .map_err(cuda_error("centroid readback"))?;
        self.record_d2h(byte_len::<f32>(centroid_values)?, 1)?;
        if flat.iter().any(|value| !value.is_finite()) {
            return Err(invalid("cuVS k-means produced non-finite centroids"));
        }
        let centroid_rows = flat.chunks_exact(dim).map(<[f32]>::to_vec).collect();
        self.diagnostics.kmeans_calls += 1;
        self.diagnostics.centroid_training_us += started.elapsed().as_micros();
        SpannCentroidIndex::from_parts(
            dim as u32,
            centroid_rows,
            (0..clusters as u64).collect(),
            Vec::new(),
        )
    }

    pub(in crate::index::partitioned) fn route_all<F>(
        &mut self,
        centroids: &SpannCentroidIndex,
        source: &dyn VectorSource,
        probe: usize,
        mut sink: F,
    ) -> Result<()>
    where
        F: FnMut(u64, usize, &[i64], &[f32]) -> Result<()>,
    {
        self.validate_source(source)?;
        let probe = validate_probe(centroids, probe)?;
        let (centroid_device, index) = self.build_router(centroids)?;
        let output_values = self
            .chunk_rows
            .checked_mul(probe)
            .ok_or_else(|| invalid("routing output shape overflow"))?;
        let mut ids = alloc_device(&self.stream, output_values, "routing id allocation")?;
        let mut distances =
            alloc_device(&self.stream, output_values, "routing distance allocation")?;
        self.track_device(
            byte_len::<f32>(centroid_device.len())?
                + byte_len::<i64>(output_values)?
                + byte_len::<f32>(output_values)?,
        )?;
        for start in (0..self.row_count).step_by(self.chunk_rows) {
            let take = self.chunk_rows.min(self.row_count - start);
            self.search_range(&index, source, start, take, probe, &mut ids, &mut distances)?;
            let host_ids = self
                .stream
                .clone_dtoh(&ids)
                .map_err(cuda_error("routing id readback"))?;
            let host_distances = self
                .stream
                .clone_dtoh(&distances)
                .map_err(cuda_error("routing distance readback"))?;
            self.record_d2h(
                byte_len::<i64>(output_values)? + byte_len::<f32>(output_values)?,
                2,
            )?;
            let used = take * probe;
            sink(
                start as u64,
                take,
                &host_ids[..used],
                &host_distances[..used],
            )?;
        }
        self.diagnostics.routing_calls += 1;
        self.diagnostics.corpus_passes += 1;
        self.diagnostics.resident_reused_across_scans =
            self.resident.is_some() && self.diagnostics.corpus_passes > 1;
        Ok(())
    }

    pub(in crate::index::partitioned) fn route_members(
        &mut self,
        centroids: &SpannCentroidIndex,
        source: &dyn VectorSource,
        members: &[u64],
    ) -> Result<Vec<u32>> {
        self.validate_source(source)?;
        let (centroid_device, index) = self.build_router(centroids)?;
        let mut ids = alloc_device(&self.stream, self.chunk_rows, "member id allocation")?;
        let mut distances =
            alloc_device(&self.stream, self.chunk_rows, "member distance allocation")?;
        self.track_device(
            byte_len::<f32>(centroid_device.len())?
                + byte_len::<i64>(self.chunk_rows)?
                + byte_len::<f32>(self.chunk_rows)?,
        )?;
        let mut assignments = Vec::with_capacity(members.len());
        for chunk in members.chunks(self.chunk_rows) {
            self.stage_members(source, chunk)?;
            {
                let (query_pointer, _guard) = self.query_device.device_ptr(&self.stream);
                index.search(
                    &self.resources,
                    &self.stream,
                    query_pointer,
                    chunk.len(),
                    self.dim,
                    1,
                    &mut ids,
                    &mut distances,
                )?;
            }
            let host_ids = self
                .stream
                .clone_dtoh(&ids)
                .map_err(cuda_error("member id readback"))?;
            self.record_d2h(byte_len::<i64>(self.chunk_rows)?, 1)?;
            for &region in &host_ids[..chunk.len()] {
                assignments.push(
                    u32::try_from(region).map_err(|_| {
                        invalid(format!("routing returned invalid region {region}"))
                    })?,
                );
            }
        }
        self.diagnostics.routing_calls += 1;
        Ok(assignments)
    }

    pub(in crate::index::partitioned) fn diagnostics_mut(
        &mut self,
    ) -> &mut PartitionBuildDiagnostics {
        &mut self.diagnostics
    }

    fn build_router(
        &mut self,
        centroids: &SpannCentroidIndex,
    ) -> Result<(CudaSlice<f32>, BruteForceIndex)> {
        if centroids.dim() as usize != self.dim {
            return Err(invalid("routing centroid dimension mismatch"));
        }
        let values = centroids
            .centroid_count()
            .checked_mul(self.dim)
            .ok_or_else(|| invalid("routing centroid shape overflow"))?;
        let mut host = pinned_zeros(&self.context, values, "routing centroid pinned allocation")?;
        let slice = host
            .as_mut_slice()
            .map_err(cuda_error("routing centroid pinned access"))?;
        for (destination, centroid) in slice.chunks_exact_mut(self.dim).zip(centroids.centroids()) {
            destination.copy_from_slice(centroid);
        }
        let device = self
            .stream
            .clone_htod(&host)
            .map_err(cuda_error("routing centroid upload"))?;
        self.stream
            .synchronize()
            .map_err(cuda_error("routing centroid upload sync"))?;
        self.record_h2d(byte_len::<f32>(values)?, 1, 0)?;
        self.track_pinned(byte_len::<f32>(values)?)?;
        let index = BruteForceIndex::build(
            &self.resources,
            &self.stream,
            &device,
            centroids.centroid_count(),
            self.dim,
        )?;
        Ok((device, index))
    }

    #[allow(clippy::too_many_arguments)]
    fn search_range(
        &mut self,
        index: &BruteForceIndex,
        source: &dyn VectorSource,
        start: usize,
        take: usize,
        probe: usize,
        ids: &mut CudaSlice<i64>,
        distances: &mut CudaSlice<f32>,
    ) -> Result<()> {
        if let Some(corpus) = &self.resident {
            let (base, _guard) = corpus.device_ptr(&self.stream);
            let offset = start
                .checked_mul(self.dim)
                .and_then(|cells| cells.checked_mul(size_of::<f32>()))
                .ok_or_else(|| invalid("resident query offset overflow"))?;
            let pointer = base
                .checked_add(to_u64(offset)?)
                .ok_or_else(|| invalid("resident device pointer overflow"))?;
            index.search(
                &self.resources,
                &self.stream,
                pointer,
                take,
                self.dim,
                probe,
                ids,
                distances,
            )
        } else {
            self.stage_range(source, start, take)?;
            let (pointer, _guard) = self.query_device.device_ptr(&self.stream);
            index.search(
                &self.resources,
                &self.stream,
                pointer,
                take,
                self.dim,
                probe,
                ids,
                distances,
            )
        }
    }

    fn stage_range(&mut self, source: &dyn VectorSource, start: usize, take: usize) -> Result<()> {
        fill_range(source, start, take, self.dim, &mut self.query_host)?;
        self.upload_query(take)
    }

    fn stage_members(&mut self, source: &dyn VectorSource, members: &[u64]) -> Result<()> {
        let host = self
            .query_host
            .as_mut_slice()
            .map_err(cuda_error("member pinned access"))?;
        host[..members.len() * self.dim]
            .par_chunks_exact_mut(self.dim)
            .zip(members.par_iter())
            .try_for_each(|(destination, &row_id)| {
                source.row_into(row_id, destination);
                ensure_finite(destination, row_id)
            })?;
        self.upload_query(members.len())
    }

    fn upload_query(&mut self, rows: usize) -> Result<()> {
        self.stream
            .memcpy_htod(&self.query_host, &mut self.query_device)
            .map_err(cuda_error("query chunk upload"))?;
        self.stream
            .synchronize()
            .map_err(cuda_error("query chunk upload sync"))?;
        self.record_h2d(byte_len::<f32>(self.query_device.len())?, 1, rows)
    }

    fn validate_source(&self, source: &dyn VectorSource) -> Result<()> {
        if source.len() != self.row_count as u64 || source.dim() != self.dim {
            return Err(invalid("GPU routing source changed after session creation"));
        }
        Ok(())
    }

    fn record_h2d(&mut self, bytes: usize, transfers: usize, rows: usize) -> Result<()> {
        self.diagnostics.h2d_transfers += transfers;
        self.diagnostics.h2d_bytes = self
            .diagnostics
            .h2d_bytes
            .checked_add(to_u64(bytes)?)
            .ok_or_else(|| invalid("H2D byte counter overflow"))?;
        if rows > 0 {
            self.diagnostics.corpus_uploads += 1;
            self.diagnostics.rows_uploaded = self
                .diagnostics
                .rows_uploaded
                .checked_add(rows as u64)
                .ok_or_else(|| invalid("uploaded row counter overflow"))?;
        }
        Ok(())
    }

    fn record_d2h(&mut self, bytes: usize, transfers: usize) -> Result<()> {
        self.diagnostics.d2h_transfers += transfers;
        self.diagnostics.d2h_bytes = self
            .diagnostics
            .d2h_bytes
            .checked_add(to_u64(bytes)?)
            .ok_or_else(|| invalid("D2H byte counter overflow"))?;
        Ok(())
    }

    fn track_device(&mut self, operation_bytes: usize) -> Result<()> {
        let base = corpus_bytes_if(&self.resident) + byte_len::<f32>(self.query_device.len())?;
        self.diagnostics.peak_device_bytes = self.diagnostics.peak_device_bytes.max(to_u64(
            base.checked_add(operation_bytes)
                .ok_or_else(|| invalid("device peak overflow"))?,
        )?);
        Ok(())
    }

    fn track_pinned(&mut self, operation_bytes: usize) -> Result<()> {
        let base = byte_len::<f32>(self.query_host.len())?;
        self.diagnostics.peak_pinned_host_bytes =
            self.diagnostics.peak_pinned_host_bytes.max(to_u64(
                base.checked_add(operation_bytes)
                    .ok_or_else(|| invalid("pinned peak overflow"))?,
            )?);
        Ok(())
    }
}
