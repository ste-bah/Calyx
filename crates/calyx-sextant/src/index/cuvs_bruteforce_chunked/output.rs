use calyx_core::Result;

use super::invalid;

pub(super) fn validate(
    ids: &[i64],
    distances: &[f32],
    corpus_rows: u64,
    query_count: usize,
    k: usize,
) -> Result<Vec<u64>> {
    let mut out = Vec::with_capacity(ids.len());
    for query in 0..query_count {
        let mut prior: Option<(f32, u64)> = None;
        for rank in 0..k {
            let pos = query * k + rank;
            let id = u64::try_from(ids[pos]).map_err(|_| invalid("negative final neighbor"))?;
            let distance = distances[pos];
            if id >= corpus_rows || !distance.is_finite() {
                return Err(invalid("invalid chunked cuVS exact output"));
            }
            if prior.is_some_and(|(prior_distance, prior_id)| {
                distance < prior_distance || (distance == prior_distance && id < prior_id)
            }) {
                return Err(invalid("non-canonical chunked cuVS exact ordering"));
            }
            prior = Some((distance, id));
            out.push(id);
        }
    }
    Ok(out)
}
