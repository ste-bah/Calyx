use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Instant;

use calyx_aster::retained_input::input_from_ref;
use calyx_core::{CalyxError, Constellation, CxId};
use calyx_sextant::{
    Hit, HitRerankEvidence, HitRerankMethod, RerankCandidateText, RerankRequest, RerankerClient,
};
use serde::Serialize;
use zeroize::Zeroizing;

use crate::error::CliResult;
use crate::metadata_exact::{MetadataMatch, metadata_match};

use super::SearchOutcome;

const PASSAGE_WORDS: usize = 180;
const PREFIX_BYTES: usize = 350;
const RERANK_HTTP_BATCH_SIZE: usize = 64;
const LEXICAL_COVERAGE_WEIGHT: f32 = 0.2;
const LEXICAL_BIGRAM_COVERAGE_WEIGHT: f32 = 0.3;
const METADATA_TIE_POLICY: &str = "non_dissent_then_earliest_same_component_type_then_fusion";

/// Candidate floor for explicit reranking. The fused top 64 is insufficient
/// for physical long-opinion recall; the cross encoder still sees bounded
/// passages and sends them in request-scoped batches.
pub const RERANK_CANDIDATE_FLOOR: usize = 128;

#[derive(Clone, Debug, Serialize)]
pub struct SearchRerankReport {
    pub method: HitRerankMethod,
    pub candidate_count: usize,
    pub exact_match_count: usize,
    pub case_exact_match_count: usize,
    pub docket_exact_match_count: usize,
    pub request_text_bytes: usize,
    pub request_count: usize,
    pub max_request_candidates: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lexical_coverage_weight: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lexical_bigram_coverage_weight: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata_tie_policy: Option<&'static str>,
    pub elapsed_ms: u128,
}

pub fn rerank_search_outcome(
    vault_dir: &Path,
    query: &str,
    k: usize,
    reranker: &RerankerClient,
    outcome: &mut SearchOutcome,
) -> CliResult<SearchRerankReport> {
    let started = Instant::now();
    let candidate_count = outcome.hits.len();
    if candidate_count == 0 {
        return Ok(SearchRerankReport {
            method: HitRerankMethod::CrossEncoderBoundedPassage,
            candidate_count: 0,
            exact_match_count: 0,
            case_exact_match_count: 0,
            docket_exact_match_count: 0,
            request_text_bytes: 0,
            request_count: 0,
            max_request_candidates: RERANK_HTTP_BATCH_SIZE,
            lexical_coverage_weight: Some(LEXICAL_COVERAGE_WEIGHT),
            lexical_bigram_coverage_weight: Some(LEXICAL_BIGRAM_COVERAGE_WEIGHT),
            metadata_tie_policy: None,
            elapsed_ms: started.elapsed().as_millis(),
        });
    }
    let metadata_matches = outcome
        .hits
        .iter()
        .map(|hit| {
            let doc = require_doc(&outcome.docs, hit.cx_id)?;
            Ok(metadata_match(query, &doc.metadata))
        })
        .collect::<CliResult<Vec<_>>>()?;
    let exact_match_count = metadata_matches
        .iter()
        .filter(|matched| matched.any())
        .count();
    let case_exact_match_count = metadata_matches
        .iter()
        .filter(|matched| matched.case_name)
        .count();
    let docket_exact_match_count = metadata_matches
        .iter()
        .filter(|matched| matched.docket)
        .count();

    if exact_match_count > 0 {
        let score_evidence = outcome
            .hits
            .iter()
            .zip(&metadata_matches)
            .map(|(hit, matched)| {
                let score = match (matched.case_name, matched.docket) {
                    (true, true) => 3.0,
                    (false, true) => 2.0,
                    (true, false) => 1.0,
                    (false, false) => hit.score,
                };
                let doc = require_doc(&outcome.docs, hit.cx_id)?;
                let opinion_type = matched
                    .any()
                    .then(|| metadata_value(doc, "opinion_type"))
                    .flatten();
                let date_filed = matched
                    .any()
                    .then(|| metadata_value(doc, "date_filed"))
                    .flatten();
                Ok(RerankScoreEvidence {
                    score,
                    cross_encoder_score: None,
                    lexical_coverage: None,
                    lexical_bigram_coverage: None,
                    metadata_dissent_component: matched.any().then(|| {
                        opinion_type
                            .as_deref()
                            .is_some_and(|value| value.starts_with("040"))
                    }),
                    metadata_opinion_type: opinion_type,
                    metadata_date_filed: date_filed,
                })
            })
            .collect::<CliResult<Vec<_>>>()?;
        apply_rerank(
            &mut outcome.hits,
            score_evidence,
            metadata_matches,
            vec![0; candidate_count],
            HitRerankMethod::MetadataExact,
            k,
        );
        return Ok(SearchRerankReport {
            method: HitRerankMethod::MetadataExact,
            candidate_count,
            exact_match_count,
            case_exact_match_count,
            docket_exact_match_count,
            request_text_bytes: 0,
            request_count: 0,
            max_request_candidates: RERANK_HTTP_BATCH_SIZE,
            lexical_coverage_weight: None,
            lexical_bigram_coverage_weight: None,
            metadata_tie_policy: Some(METADATA_TIE_POLICY),
            elapsed_ms: started.elapsed().as_millis(),
        });
    }

    let passages = outcome
        .hits
        .iter()
        .map(|hit| candidate_passage(vault_dir, query, require_doc(&outcome.docs, hit.cx_id)?))
        .collect::<CliResult<Vec<_>>>()?;
    let passage_bytes = passages
        .iter()
        .map(|passage| passage.text.len())
        .collect::<Vec<_>>();
    let lexical_coverages = passages
        .iter()
        .map(|passage| passage.lexical_coverage)
        .collect::<Vec<_>>();
    let lexical_bigram_coverages = passages
        .iter()
        .map(|passage| passage.lexical_bigram_coverage)
        .collect::<Vec<_>>();
    let request_text_bytes = passage_bytes.iter().sum();
    let mut candidates = passages
        .into_iter()
        .map(|passage| RerankCandidateText::new(passage.text));
    let mut cross_encoder_scores = Vec::with_capacity(candidate_count);
    let mut request_count = 0;
    loop {
        let request_candidates = candidates
            .by_ref()
            .take(RERANK_HTTP_BATCH_SIZE)
            .collect::<Vec<_>>();
        if request_candidates.is_empty() {
            break;
        }
        request_count += 1;
        let response = reranker.rerank(&RerankRequest::from_candidate_texts(
            query,
            request_candidates,
        ))?;
        cross_encoder_scores.extend(response.scores.into_iter().map(stable_score));
    }
    let score_evidence = cross_encoder_scores
        .into_iter()
        .zip(lexical_coverages)
        .zip(lexical_bigram_coverages)
        .map(
            |((cross_encoder, coverage), bigram_coverage)| RerankScoreEvidence {
                score: stable_score(
                    cross_encoder
                        + LEXICAL_COVERAGE_WEIGHT * coverage
                        + LEXICAL_BIGRAM_COVERAGE_WEIGHT * bigram_coverage,
                ),
                cross_encoder_score: Some(cross_encoder),
                lexical_coverage: Some(coverage),
                lexical_bigram_coverage: Some(bigram_coverage),
                metadata_opinion_type: None,
                metadata_date_filed: None,
                metadata_dissent_component: None,
            },
        )
        .collect();
    apply_rerank(
        &mut outcome.hits,
        score_evidence,
        metadata_matches,
        passage_bytes,
        HitRerankMethod::CrossEncoderBoundedPassage,
        k,
    );
    Ok(SearchRerankReport {
        method: HitRerankMethod::CrossEncoderBoundedPassage,
        candidate_count,
        exact_match_count,
        case_exact_match_count,
        docket_exact_match_count,
        request_text_bytes,
        request_count,
        max_request_candidates: RERANK_HTTP_BATCH_SIZE,
        lexical_coverage_weight: Some(LEXICAL_COVERAGE_WEIGHT),
        lexical_bigram_coverage_weight: Some(LEXICAL_BIGRAM_COVERAGE_WEIGHT),
        metadata_tie_policy: None,
        elapsed_ms: started.elapsed().as_millis(),
    })
}

fn require_doc(docs: &BTreeMap<CxId, Constellation>, cx_id: CxId) -> CliResult<&Constellation> {
    docs.get(&cx_id).ok_or_else(|| {
        CalyxError::stale_derived(format!(
            "rerank candidate {cx_id} has no hydrated constellation"
        ))
        .into()
    })
}

fn metadata_value(doc: &Constellation, key: &str) -> Option<String> {
    doc.metadata
        .get(key)
        .filter(|value| !value.trim().is_empty())
        .cloned()
}

struct CandidatePassage {
    text: String,
    lexical_coverage: f32,
    lexical_bigram_coverage: f32,
}

fn candidate_passage(
    vault_dir: &Path,
    query: &str,
    doc: &Constellation,
) -> CliResult<CandidatePassage> {
    let input = input_from_ref(vault_dir, doc.modality, &doc.input_ref)?;
    let bytes = Zeroizing::new(input.bytes);
    let text = std::str::from_utf8(&bytes).map_err(|error| {
        CalyxError::stale_derived(format!(
            "retained text for rerank candidate {} is not UTF-8: {error}",
            doc.cx_id
        ))
    })?;
    let text = select_passage(text, query);
    Ok(CandidatePassage {
        lexical_coverage: lexical_coverage(&text, query),
        lexical_bigram_coverage: lexical_bigram_coverage(&text, query),
        text,
    })
}

fn lexical_coverage(text: &str, query: &str) -> f32 {
    let query_terms = normalized_terms(query)
        .into_iter()
        .filter(|term| !stopword(term))
        .collect::<BTreeSet<_>>();
    if query_terms.is_empty() {
        return 0.0;
    }
    let passage_terms = normalized_terms(text).into_iter().collect::<BTreeSet<_>>();
    query_terms.intersection(&passage_terms).count() as f32 / query_terms.len() as f32
}

fn lexical_bigram_coverage(text: &str, query: &str) -> f32 {
    let query_terms = normalized_terms(query)
        .into_iter()
        .filter(|term| !stopword(term))
        .collect::<Vec<_>>();
    let query_bigrams = query_terms.windows(2).collect::<Vec<_>>();
    if query_bigrams.is_empty() {
        return 0.0;
    }
    let passage_terms = normalized_terms(text);
    let matched = query_bigrams
        .iter()
        .filter(|query_pair| {
            passage_terms.windows(2).any(|passage_pair| {
                passage_pair[0] == query_pair[0] && passage_pair[1] == query_pair[1]
            })
        })
        .count();
    matched as f32 / query_bigrams.len() as f32
}

#[derive(Clone)]
struct Word {
    normalized: String,
    start: usize,
    end: usize,
}

fn select_passage(text: &str, query: &str) -> String {
    let words = word_spans(text);
    if words.is_empty() {
        return text[..floor_char_boundary(text, PREFIX_BYTES.min(text.len()))].to_string();
    }
    let query_terms = normalized_terms(query)
        .into_iter()
        .filter(|term| !stopword(term))
        .collect::<Vec<_>>();
    let query_set = query_terms.iter().cloned().collect::<BTreeSet<_>>();
    let mut starts = BTreeSet::from([0]);
    for (index, word) in words.iter().enumerate() {
        if query_set.contains(&word.normalized) {
            starts.insert(index.saturating_sub(PASSAGE_WORDS / 2));
        }
    }

    let mut best_key = (0, 0, 0, usize::MAX);
    let mut best_window = (0, words.len().min(PASSAGE_WORDS));
    for start in starts {
        let end = (start + PASSAGE_WORDS).min(words.len());
        let window = &words[start..end];
        let unique = query_set
            .iter()
            .filter(|term| window.iter().any(|word| &word.normalized == *term))
            .count();
        let bigrams = query_terms
            .windows(2)
            .filter(|pair| {
                window
                    .windows(2)
                    .any(|words| words[0].normalized == pair[0] && words[1].normalized == pair[1])
            })
            .count();
        let total = window
            .iter()
            .filter(|word| query_set.contains(&word.normalized))
            .count();
        let score = (unique, bigrams, total, usize::MAX - start);
        if score > best_key {
            best_key = score;
            best_window = (start, end);
        }
    }

    let start = words[best_window.0].start;
    let end = words[best_window.1 - 1].end;
    let prefix_end = floor_char_boundary(text, PREFIX_BYTES.min(text.len()));
    if start <= prefix_end {
        return text[..end].to_string();
    }
    format!("{}\n[...]\n{}", &text[..prefix_end], &text[start..end])
}

fn word_spans(text: &str) -> Vec<Word> {
    let mut out = Vec::new();
    let mut start = None;
    for (index, ch) in text.char_indices() {
        if ch.is_ascii_alphanumeric() {
            start.get_or_insert(index);
        } else if let Some(word_start) = start.take() {
            out.push(Word {
                normalized: text[word_start..index].to_ascii_lowercase(),
                start: word_start,
                end: index,
            });
        }
    }
    if let Some(word_start) = start {
        out.push(Word {
            normalized: text[word_start..].to_ascii_lowercase(),
            start: word_start,
            end: text.len(),
        });
    }
    out
}

fn normalized_terms(text: &str) -> Vec<String> {
    word_spans(text)
        .into_iter()
        .map(|word| word.normalized)
        .collect()
}

fn stopword(term: &str) -> bool {
    matches!(
        term,
        "a" | "an"
            | "and"
            | "are"
            | "as"
            | "at"
            | "be"
            | "by"
            | "did"
            | "do"
            | "does"
            | "for"
            | "from"
            | "how"
            | "in"
            | "into"
            | "is"
            | "it"
            | "of"
            | "on"
            | "or"
            | "that"
            | "the"
            | "their"
            | "this"
            | "to"
            | "under"
            | "what"
            | "when"
            | "where"
            | "which"
            | "who"
            | "will"
            | "with"
    )
}

fn floor_char_boundary(text: &str, mut index: usize) -> usize {
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

struct RerankScoreEvidence {
    score: f32,
    cross_encoder_score: Option<f32>,
    lexical_coverage: Option<f32>,
    lexical_bigram_coverage: Option<f32>,
    metadata_opinion_type: Option<String>,
    metadata_date_filed: Option<String>,
    metadata_dissent_component: Option<bool>,
}

fn apply_rerank(
    hits: &mut Vec<Hit>,
    score_evidence: Vec<RerankScoreEvidence>,
    metadata_matches: Vec<MetadataMatch>,
    passage_bytes: Vec<usize>,
    method: HitRerankMethod,
    k: usize,
) {
    let mut ranked = hits
        .drain(..)
        .zip(score_evidence)
        .zip(metadata_matches)
        .zip(passage_bytes)
        .map(
            |(((mut hit, score_evidence), metadata_match), passage_bytes)| {
                let original_rank = hit.rank;
                let fusion_score = hit.score;
                hit.score = score_evidence.score;
                if hit.explain.is_none() {
                    hit = hit.with_explain("pipeline");
                }
                let explain = hit.explain.as_mut().expect("explain installed above");
                explain.strategy = format!("{}+{}", explain.strategy, method_name(method));
                explain.rerank = Some(HitRerankEvidence {
                    method,
                    fusion_score,
                    rerank_score: score_evidence.score,
                    cross_encoder_score: score_evidence.cross_encoder_score,
                    lexical_coverage: score_evidence.lexical_coverage,
                    lexical_bigram_coverage: score_evidence.lexical_bigram_coverage,
                    metadata_exact: metadata_match.any(),
                    metadata_case_exact: metadata_match.case_name,
                    metadata_docket_exact: metadata_match.docket,
                    metadata_opinion_type: score_evidence.metadata_opinion_type.clone(),
                    metadata_date_filed: score_evidence.metadata_date_filed.clone(),
                    metadata_dissent_component: score_evidence.metadata_dissent_component,
                    passage_bytes,
                });
                (hit, original_rank, score_evidence)
            },
        )
        .collect::<Vec<_>>();
    ranked.sort_by(
        |(left, left_rank, left_evidence), (right, right_rank, right_evidence)| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| metadata_tie_order(method, left_evidence, right_evidence))
                .then_with(|| left_rank.cmp(right_rank))
                .then_with(|| left.cx_id.cmp(&right.cx_id))
        },
    );
    hits.extend(ranked.into_iter().take(k).enumerate().map(
        |(index, (mut hit, _original_rank, _score_evidence))| {
            hit.rank = index + 1;
            hit
        },
    ));
}

fn metadata_tie_order(
    method: HitRerankMethod,
    left: &RerankScoreEvidence,
    right: &RerankScoreEvidence,
) -> std::cmp::Ordering {
    if method != HitRerankMethod::MetadataExact {
        return std::cmp::Ordering::Equal;
    }
    left.metadata_dissent_component
        .unwrap_or(false)
        .cmp(&right.metadata_dissent_component.unwrap_or(false))
        .then_with(|| {
            if left.metadata_opinion_type.is_none()
                || left.metadata_opinion_type != right.metadata_opinion_type
            {
                return std::cmp::Ordering::Equal;
            }
            match (
                left.metadata_date_filed.as_deref(),
                right.metadata_date_filed.as_deref(),
            ) {
                (Some(left), Some(right)) => left.cmp(right),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => std::cmp::Ordering::Equal,
            }
        })
}

fn stable_score(score: f32) -> f32 {
    // The live CUDA cross encoder varies by one millipoint on identical
    // requests. Public centi-scores are stable across that physical kernel
    // jitter; deterministic fused rank and CX ID remain the tie breakers.
    (score * 100.0).round() / 100.0
}

fn method_name(method: HitRerankMethod) -> &'static str {
    match method {
        HitRerankMethod::MetadataExact => "metadata_exact",
        HitRerankMethod::CrossEncoderBoundedPassage => "cross_encoder_bounded_passage",
    }
}
