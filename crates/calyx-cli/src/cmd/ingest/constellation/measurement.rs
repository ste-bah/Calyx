use super::*;

pub(super) fn persisted_snapshot_for_lens(
    state: &VaultPanelState,
    lens_id: LensId,
) -> Option<&calyx_registry::RegistryLensSnapshot> {
    state
        .registry_snapshot
        .as_ref()?
        .lenses
        .iter()
        .find(|snapshot| snapshot.lens_id == lens_id)
}

pub(super) fn measure_persisted_lens_in_worker(
    snapshot: &calyx_registry::RegistryLensSnapshot,
    inputs: &[Input],
    runtime_batch_limit: Option<usize>,
) -> calyx_core::Result<Vec<SlotVector>> {
    let vectors = measure_lens_in_worker(snapshot, inputs, runtime_batch_limit)?;
    if vectors.len() != inputs.len() {
        return Err(CalyxError::lens_dim_mismatch(format!(
            "ingest lens worker for lens {} returned {} vectors for {} inputs",
            snapshot.lens_id,
            vectors.len(),
            inputs.len()
        )));
    }
    for vector in &vectors {
        snapshot.contract.verify_vector(snapshot.lens_id, vector)?;
    }
    Ok(vectors)
}

pub(super) fn measure_registry_lens_batch_with_limit(
    state: &VaultPanelState,
    lens_id: LensId,
    inputs: &[Input],
    runtime_batch_limit: Option<usize>,
) -> calyx_core::Result<Vec<SlotVector>> {
    calyx_registry::measure_registry_batch_with_runtime_limit(
        &state.registry,
        lens_id,
        inputs,
        runtime_batch_limit,
    )
}

pub(super) fn measure_applicable_lens_batch(
    state: &VaultPanelState,
    lens: ApplicableLens,
    modality: Modality,
    inputs: &[Input],
    runtime_batch_limit: Option<usize>,
) -> calyx_core::Result<Vec<SlotVector>> {
    let started = Instant::now();
    let spec = state.registry.lens_spec(lens.lens_id);
    let spec_name = spec
        .map(|spec| spec.name.as_str())
        .unwrap_or("missing_registry_snapshot");
    let runtime = spec
        .map(|spec| runtime_name(&spec.runtime))
        .unwrap_or("unregistered");
    ingest_runtime_log(format_args!(
        "phase=measure_lens_start lens_id={} slot={} name={} runtime={} modality={:?} placement={:?} batch_size={} runtime_batch_limit={:?}",
        lens.lens_id,
        lens.slot_id.get(),
        spec_name,
        runtime,
        modality,
        lens.placement,
        inputs.len(),
        runtime_batch_limit
    ));
    let result = if lens.placement == Placement::Gpu {
        if let Some(snapshot) = persisted_snapshot_for_lens(state, lens.lens_id) {
            ingest_runtime_log(format_args!(
                "phase=measure_lens_worker_start lens_id={} slot={} name={} inputs={} runtime_batch_limit={:?}",
                lens.lens_id,
                lens.slot_id.get(),
                spec_name,
                inputs.len(),
                runtime_batch_limit
            ));
            let result = measure_persisted_lens_in_worker(snapshot, inputs, runtime_batch_limit);
            if result.is_ok() {
                ingest_runtime_log(format_args!(
                    "phase=measure_lens_worker_ok lens_id={} slot={} name={}",
                    lens.lens_id,
                    lens.slot_id.get(),
                    spec_name
                ));
            }
            result
        } else {
            measure_registry_lens_batch_with_limit(state, lens.lens_id, inputs, runtime_batch_limit)
        }
    } else {
        measure_registry_lens_batch_with_limit(state, lens.lens_id, inputs, runtime_batch_limit)
    };
    match &result {
        Ok(vectors) => ingest_runtime_log(format_args!(
            "phase=measure_lens_ok lens_id={} slot={} name={} vectors={} elapsed_ms={}",
            lens.lens_id,
            lens.slot_id.get(),
            spec_name,
            vectors.len(),
            started.elapsed().as_millis()
        )),
        Err(error) => ingest_runtime_log(format_args!(
            "phase=measure_lens_err lens_id={} slot={} name={} code={} message={} elapsed_ms={}",
            lens.lens_id,
            lens.slot_id.get(),
            spec_name,
            error.code,
            error.message,
            started.elapsed().as_millis()
        )),
    }
    result
}

pub(super) fn group_applicable_lenses(
    state: &VaultPanelState,
    lenses: &[ApplicableLens],
) -> calyx_core::Result<Vec<ApplicableLensJob>> {
    let mut jobs: Vec<ApplicableLensJob> = Vec::new();
    let mut grouped_jobs: BTreeMap<MeasurementGroupKey, usize> = BTreeMap::new();
    for &lens in lenses {
        match state.registry.measurement_group_key(lens.lens_id)? {
            Some(key) => {
                if let Some(&job_index) = grouped_jobs.get(&key) {
                    jobs[job_index].lenses.push(lens);
                } else {
                    let job_index = jobs.len();
                    grouped_jobs.insert(key, job_index);
                    jobs.push(ApplicableLensJob {
                        lenses: vec![lens],
                        grouped: true,
                    });
                }
            }
            None => jobs.push(ApplicableLensJob {
                lenses: vec![lens],
                grouped: false,
            }),
        }
    }
    Ok(jobs)
}

pub(super) fn measure_applicable_lens_job(
    state: &VaultPanelState,
    job: &ApplicableLensJob,
    modality: Modality,
    inputs: &[Input],
    runtime_batch_limit: Option<usize>,
) -> calyx_core::Result<Vec<(LensId, Vec<SlotVector>)>> {
    if !job.grouped || job.lenses.len() == 1 {
        let lens = job.lenses[0];
        return measure_applicable_lens_batch(state, lens, modality, inputs, runtime_batch_limit)
            .map(|vectors| vec![(lens.lens_id, vectors)]);
    }
    let started = Instant::now();
    let lens_ids: Vec<LensId> = job.lenses.iter().map(|lens| lens.lens_id).collect();
    ingest_runtime_log(format_args!(
        "phase=measure_lens_group_start lenses={} lens_ids={:?} modality={:?} placement={:?} batch_size={} runtime_batch_limit={:?}",
        lens_ids.len(),
        lens_ids,
        modality,
        job.lenses[0].placement,
        inputs.len(),
        runtime_batch_limit
    ));
    let result = calyx_registry::measure_registry_group_with_runtime_limit(
        &state.registry,
        &lens_ids,
        inputs,
        runtime_batch_limit,
    );
    match &result {
        Ok(_) => ingest_runtime_log(format_args!(
            "phase=measure_lens_group_ok lenses={} lens_ids={:?} vectors_per_lens={} elapsed_ms={}",
            lens_ids.len(),
            lens_ids,
            inputs.len(),
            started.elapsed().as_millis()
        )),
        Err(error) => ingest_runtime_log(format_args!(
            "phase=measure_lens_group_err lenses={} lens_ids={:?} code={} message={} elapsed_ms={}",
            lens_ids.len(),
            lens_ids,
            error.code,
            error.message,
            started.elapsed().as_millis()
        )),
    }
    result.map(|mut measured| {
        lens_ids
            .into_iter()
            .map(|lens_id| {
                let vectors = measured
                    .remove(&lens_id)
                    .expect("registry validated every grouped lens result");
                (lens_id, vectors)
            })
            .collect()
    })
}
