//! `calyx materialize-citation-overlay` writes the REAL judge-authored
//! citation graph (#1465) into Calyx as a typed `cites` overlay collection
//! over EXISTING constellations.
//!
//! Unlike `materialize-bridge-corpus` and `materialize-evidence-substrate`
//! (which MINT fresh node CxIds from row content / stable keys), this command
//! resolves every citation endpoint to a pre-existing constellation CxId
//! through an `opinion_id -> cx_id` import map whose every row is first proven
//! against the accepted DB-native opinion-alias relation,
//! then writes depth-weighted `cites` edges between those constellations as an
//! accepted Aster Graph CF generation. Every row in the in-slice input must
//! resolve on both sides; missing aliases are a typed hard failure. Only the
//! separately declared frontier input may name an out-of-corpus endpoint.

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use calyx_core::{CalyxError, CxId, content_address};
use serde::Serialize;

use super::vault::home_dir;
use super::{Subcommand, value};
use crate::error::{CliError, CliResult};
use crate::output::print_json;

#[cfg(test)]
mod tests;
pub(crate) mod write;

pub(crate) const DEFAULT_COLLECTION: &str = "legal-citations-v1";
pub(crate) const PROVENANCE_DATASET: &str = "courtlistener-citation-map-2026-06-30";
pub(crate) const EDGE_TYPE: &str = "cites";
pub(crate) const FRONTIER_EDGE_TYPE: &str = "cites_outside_corpus";
pub(crate) const SCHEMA: &str = "legal_citation_overlay_v1";
pub(crate) const FRONTIER_SCHEMA: &str = "legal_citation_overlay_v2";
pub(crate) const NODE_TYPE: &str = "opinion";
pub(crate) const FRONTIER_NODE_TYPE: &str = "citation_frontier";
pub(crate) const FRONTIER_REASON: &str = "OUTSIDE_SEALED_CORPUS";
const DEPTH_CAP: u32 = 10;
const MAX_SKIP_SAMPLES: usize = 50;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MaterializeCitationOverlayArgs {
    pub vault: String,
    pub idmap: PathBuf,
    pub citations: PathBuf,
    pub frontier: Option<PathBuf>,
    pub frontier_authorities: Option<PathBuf>,
    pub collection: Option<String>,
    pub skip_report: Option<PathBuf>,
    pub report: Option<PathBuf>,
    pub home: Option<PathBuf>,
}

/// One resolved, deduplicated `cites` edge between two existing constellations.
#[derive(Clone, Debug)]
pub(crate) struct CitesEdge {
    pub src: CxId,
    pub dst: CxId,
    pub citing_opinion_id: String,
    pub cited_opinion_id: String,
    pub depth: u32,
    pub weight: f64,
    pub source_row_id: String,
    pub edge_type: &'static str,
    pub source_citations: Vec<SourceCitation>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct SourceCitation {
    pub citing_opinion_id: String,
    pub cited_opinion_id: String,
    pub depth: u32,
    pub source_row_id: String,
}

/// An opinion node that participates in at least one resolved edge.
#[derive(Clone, Debug)]
pub(crate) struct OpinionNode {
    pub cx_id: CxId,
    pub opinion_id: String,
    pub node_type: &'static str,
    pub authority_name: Option<String>,
    pub boundary_reason: Option<&'static str>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct SkipSample {
    pub row: usize,
    pub reason: &'static str,
    pub citing_opinion_id: String,
    pub cited_opinion_id: String,
}

#[derive(Debug, Default, Serialize)]
pub(crate) struct SkipReport {
    pub total_rows: usize,
    pub edges_built: usize,
    pub skipped_total: usize,
    pub skipped_unresolved_citing: usize,
    pub skipped_unresolved_cited: usize,
    pub skipped_unresolved_both: usize,
    pub skipped_invalid_depth: usize,
    pub skipped_duplicate_pair: usize,
    pub coalesced_duplicate_pair: usize,
    pub source_citation_rows_preserved: usize,
    pub idmap_entries: usize,
    pub physical_alias_rows_verified: usize,
    pub frontier_rows: usize,
    pub frontier_edges_built: usize,
    pub frontier_nodes_built: usize,
    pub frontier_duplicate_pair: usize,
    pub frontier_authority_names: usize,
    pub samples: Vec<SkipSample>,
}

pub(crate) struct CitationOverlayDraft {
    pub nodes: BTreeMap<CxId, OpinionNode>,
    pub edges: Vec<CitesEdge>,
    pub skip: SkipReport,
}

pub(crate) fn parse_materialize_citation_overlay(rest: &[String]) -> CliResult<Subcommand> {
    let vault = rest
        .first()
        .ok_or_else(|| CliError::usage("materialize-citation-overlay requires <vault>"))?
        .clone();
    let mut idmap = None;
    let mut citations = None;
    let mut frontier = None;
    let mut frontier_authorities = None;
    let mut collection = None;
    let mut skip_report = None;
    let mut report = None;
    let mut home = None;
    let mut idx = 1;
    while idx < rest.len() {
        match rest[idx].as_str() {
            "--idmap" => {
                idx += 1;
                idmap = Some(value(rest, idx, "--idmap")?.into());
            }
            "--citations" => {
                idx += 1;
                citations = Some(value(rest, idx, "--citations")?.into());
            }
            "--frontier" => {
                idx += 1;
                frontier = Some(value(rest, idx, "--frontier")?.into());
            }
            "--frontier-authorities" => {
                idx += 1;
                frontier_authorities = Some(value(rest, idx, "--frontier-authorities")?.into());
            }
            "--collection" => {
                idx += 1;
                collection = Some(value(rest, idx, "--collection")?.to_string());
            }
            "--skip-report" => {
                idx += 1;
                skip_report = Some(value(rest, idx, "--skip-report")?.into());
            }
            "--report" => {
                idx += 1;
                report = Some(value(rest, idx, "--report")?.into());
            }
            "--home" => {
                idx += 1;
                home = Some(value(rest, idx, "--home")?.into());
            }
            other => {
                return Err(CliError::usage(format!(
                    "unexpected materialize-citation-overlay flag {other}"
                )));
            }
        }
        idx += 1;
    }
    Ok(Subcommand::MaterializeCitationOverlay(
        MaterializeCitationOverlayArgs {
            vault,
            idmap: idmap.ok_or_else(|| {
                CliError::usage("materialize-citation-overlay requires --idmap <csv>")
            })?,
            citations: citations.ok_or_else(|| {
                CliError::usage("materialize-citation-overlay requires --citations <csv>")
            })?,
            frontier,
            frontier_authorities,
            collection,
            skip_report,
            report,
            home,
        },
    ))
}

pub(crate) fn run(command: Subcommand) -> CliResult {
    let Subcommand::MaterializeCitationOverlay(args) = command else {
        unreachable!("non-materialize-citation-overlay command routed here");
    };
    let home = args.home.clone().map_or_else(home_dir, Ok)?;
    let report = materialize_with_home(&home, args)?;
    print_json(&report)
}

fn materialize_with_home(
    home: &Path,
    args: MaterializeCitationOverlayArgs,
) -> CliResult<write::CitationOverlayReport> {
    preflight_report_paths(&args)?;
    let idmap = load_idmap(&args.idmap)?;
    let physical_alias_rows_verified =
        super::opinion_alias_overlay::verify_idmap_physical(home, &args.vault, &idmap)?;
    if args.frontier_authorities.is_some() && args.frontier.is_none() {
        return Err(CliError::usage(
            "--frontier-authorities requires --frontier <csv>",
        ));
    }
    let authorities = args
        .frontier_authorities
        .as_deref()
        .map(load_frontier_authorities)
        .transpose()?
        .unwrap_or_default();
    let mut draft = build_draft(
        &args.citations,
        args.frontier.as_deref(),
        &idmap,
        &authorities,
    )?;
    draft.skip.physical_alias_rows_verified = physical_alias_rows_verified;
    write::write_to_calyx(home, &args, draft)
}

fn preflight_report_paths(args: &MaterializeCitationOverlayArgs) -> CliResult {
    let outputs = [
        ("--skip-report", args.skip_report.as_deref()),
        ("--report", args.report.as_deref()),
    ];
    let mut identities = Vec::new();
    for (flag, path) in outputs {
        let Some(path) = path else { continue };
        let filename = path.file_name().ok_or_else(|| {
            CliError::usage(format!(
                "{flag} must identify an output file: {}",
                path.display()
            ))
        })?;
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent).map_err(|error| {
            CliError::io(format!(
                "create {flag} parent directory {} failed before graph mutation: {error}",
                parent.display()
            ))
        })?;
        let parent = parent.canonicalize().map_err(|error| {
            CliError::io(format!(
                "canonicalize {flag} parent directory {} failed before graph mutation: {error}",
                parent.display()
            ))
        })?;
        identities.push((flag, path, parent.join(filename)));
    }
    if identities.len() == 2 && identities[0].2 == identities[1].2 {
        return Err(CliError::usage(
            "--skip-report and --report must identify distinct output files",
        ));
    }
    for (flag, path, _) in identities {
        match fs::symlink_metadata(path) {
            Ok(_) => {
                return Err(CliError::usage(format!(
                    "{flag} destination {} already exists; citation overlay evidence is immutable",
                    path.display()
                )));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(CliError::io(format!(
                    "inspect {flag} destination {} failed before graph mutation: {error}",
                    path.display()
                )));
            }
        }
    }
    Ok(())
}

/// Parses the `opinion_id -> cx_id` import map (CSV header `opinion_id,cx_id`).
fn load_idmap(path: &Path) -> CliResult<BTreeMap<String, CxId>> {
    let file = fs::File::open(path)
        .map_err(|error| CliError::io(format!("open idmap {}: {error}", path.display())))?;
    let reader = BufReader::new(file);
    let mut map = BTreeMap::new();
    let mut header_seen = false;
    for (index, line) in reader.lines().enumerate() {
        let line = line.map_err(|error| CliError::io(format!("read idmap line: {error}")))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut fields = trimmed.split(',');
        let opinion_id = fields.next().unwrap_or("").trim();
        let cx_raw = fields.next().unwrap_or("").trim();
        if !header_seen {
            header_seen = true;
            if opinion_id.eq_ignore_ascii_case("opinion_id") {
                continue;
            }
        }
        if opinion_id.is_empty() || cx_raw.is_empty() {
            return Err(CliError::usage(format!(
                "idmap line {} requires opinion_id,cx_id",
                index + 1
            )));
        }
        let cx_id = cx_raw.parse::<CxId>().map_err(|error| {
            CliError::usage(format!(
                "idmap line {} has invalid cx_id: {error}",
                index + 1
            ))
        })?;
        if map.insert(opinion_id.to_string(), cx_id).is_some() {
            return Err(CliError::usage(format!(
                "idmap duplicates opinion_id {opinion_id}"
            )));
        }
    }
    if map.is_empty() {
        return Err(CliError::usage("idmap is empty"));
    }
    Ok(map)
}

struct CitationColumns {
    citing: usize,
    cited: usize,
    depth: usize,
    id: Option<usize>,
}

fn header_columns(header: &str) -> CliResult<CitationColumns> {
    let cols: Vec<String> = header
        .trim()
        .split(',')
        .map(|c| c.trim().to_ascii_lowercase())
        .collect();
    let find = |name: &str| cols.iter().position(|c| c == name);
    let citing = find("citing_opinion_id").ok_or_else(|| {
        CliError::usage("citations CSV header requires a citing_opinion_id column")
    })?;
    let cited = find("cited_opinion_id").ok_or_else(|| {
        CliError::usage("citations CSV header requires a cited_opinion_id column")
    })?;
    let depth = find("depth")
        .ok_or_else(|| CliError::usage("citations CSV header requires a depth column"))?;
    Ok(CitationColumns {
        citing,
        cited,
        depth,
        id: find("id"),
    })
}

/// Reads the in-slice citation CSV and resolves each row against the idmap,
/// building deduplicated `cites` edges and a counted skip report.
fn build_draft(
    citations: &Path,
    frontier: Option<&Path>,
    idmap: &BTreeMap<String, CxId>,
    frontier_authorities: &BTreeMap<String, String>,
) -> CliResult<CitationOverlayDraft> {
    let file = fs::File::open(citations).map_err(|error| {
        CliError::io(format!("open citations {}: {error}", citations.display()))
    })?;
    let mut lines = BufReader::new(file).lines();
    let header = lines
        .next()
        .transpose()
        .map_err(|error| CliError::io(format!("read citations header: {error}")))?
        .ok_or_else(|| CliError::usage("citations CSV is empty"))?;
    let cols = header_columns(&header)?;
    let mut nodes: BTreeMap<CxId, OpinionNode> = BTreeMap::new();
    let mut edges: Vec<CitesEdge> = Vec::new();
    let mut seen_pairs: BTreeMap<(CxId, CxId), usize> = BTreeMap::new();
    let mut skip = SkipReport {
        idmap_entries: idmap.len(),
        ..SkipReport::default()
    };
    for (offset, line) in lines.enumerate() {
        let row = offset + 2; // 1-based, header consumed
        let line =
            line.map_err(|error| CliError::io(format!("read citations row {row}: {error}")))?;
        if line.trim().is_empty() {
            continue;
        }
        skip.total_rows += 1;
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        let citing_opinion_id = fields.get(cols.citing).copied().unwrap_or("").to_string();
        let cited_opinion_id = fields.get(cols.cited).copied().unwrap_or("").to_string();
        let depth_raw = fields.get(cols.depth).copied().unwrap_or("");
        let source_row_id = cols
            .id
            .and_then(|i| fields.get(i).copied())
            .filter(|v| !v.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| format!("row{row}"));
        let src = idmap.get(&citing_opinion_id);
        let dst = idmap.get(&cited_opinion_id);
        let (src, dst) = match (src, dst) {
            (Some(src), Some(dst)) => (*src, *dst),
            _ => {
                let missing = match (src.is_none(), dst.is_none()) {
                    (true, true) => "citing and cited",
                    (true, false) => "citing",
                    (false, true) => "cited",
                    (false, false) => unreachable!(),
                };
                return Err(CliError::from(CalyxError {
                    code: "CALYX_CITATION_IN_SLICE_ENDPOINT_UNRESOLVED",
                    message: format!(
                        "in-slice citation row {row} has unresolved {missing} endpoint: {citing_opinion_id} -> {cited_opinion_id}"
                    ),
                    remediation: "materialize the complete source-opinion alias relation and rebuild the in-slice citation input; never convert an in-slice endpoint into a skip",
                }));
            }
        };
        let Ok(depth) = depth_raw.parse::<u32>() else {
            record_skip(
                &mut skip,
                "invalid_depth",
                row,
                &citing_opinion_id,
                &cited_opinion_id,
            );
            skip.skipped_invalid_depth += 1;
            continue;
        };
        if depth == 0 {
            record_skip(
                &mut skip,
                "invalid_depth",
                row,
                &citing_opinion_id,
                &cited_opinion_id,
            );
            skip.skipped_invalid_depth += 1;
            continue;
        }
        let source_citation = SourceCitation {
            citing_opinion_id: citing_opinion_id.clone(),
            cited_opinion_id: cited_opinion_id.clone(),
            depth,
            source_row_id: source_row_id.clone(),
        };
        if let Some(index) = seen_pairs.get(&(src, dst)).copied() {
            let edge = &mut edges[index];
            if edge.edge_type != EDGE_TYPE {
                return Err(CliError::runtime(
                    "in-slice citation pair collides with a non-citation edge",
                ));
            }
            edge.depth = edge.depth.max(depth);
            edge.weight = edge.weight.max(weight_for_depth(depth));
            edge.source_citations.push(source_citation);
            skip.coalesced_duplicate_pair += 1;
            continue;
        }
        let weight = weight_for_depth(depth);
        nodes.entry(src).or_insert_with(|| OpinionNode {
            cx_id: src,
            opinion_id: citing_opinion_id.clone(),
            node_type: NODE_TYPE,
            authority_name: None,
            boundary_reason: None,
        });
        nodes.entry(dst).or_insert_with(|| OpinionNode {
            cx_id: dst,
            opinion_id: cited_opinion_id.clone(),
            node_type: NODE_TYPE,
            authority_name: None,
            boundary_reason: None,
        });
        seen_pairs.insert((src, dst), edges.len());
        edges.push(CitesEdge {
            src,
            dst,
            citing_opinion_id,
            cited_opinion_id,
            depth,
            weight,
            source_row_id,
            edge_type: EDGE_TYPE,
            source_citations: vec![source_citation],
        });
    }
    if let Some(frontier) = frontier {
        append_frontier(
            frontier,
            idmap,
            frontier_authorities,
            &mut nodes,
            &mut edges,
            &mut seen_pairs,
            &mut skip,
        )?;
    }
    skip.edges_built = edges.len();
    skip.source_citation_rows_preserved = edges
        .iter()
        .filter(|edge| edge.edge_type == EDGE_TYPE)
        .map(|edge| edge.source_citations.len())
        .sum();
    skip.skipped_total = skip.skipped_unresolved_citing
        + skip.skipped_unresolved_cited
        + skip.skipped_unresolved_both
        + skip.skipped_invalid_depth;
    if edges.is_empty() {
        return Err(CliError::runtime(
            "no citation edges resolved; refusing to write an empty overlay collection",
        ));
    }
    Ok(CitationOverlayDraft { nodes, edges, skip })
}

fn load_frontier_authorities(path: &Path) -> CliResult<BTreeMap<String, String>> {
    let bytes = fs::read(path).map_err(|error| {
        CliError::io(format!(
            "read frontier authority JSON {}: {error}",
            path.display()
        ))
    })?;
    let map: BTreeMap<String, String> = serde_json::from_slice(&bytes).map_err(|error| {
        CliError::usage(format!(
            "decode frontier authority JSON {}: {error}",
            path.display()
        ))
    })?;
    if map
        .iter()
        .any(|(opinion_id, name)| opinion_id.trim().is_empty() || name.trim().is_empty())
    {
        return Err(CliError::usage(
            "frontier authority JSON requires nonempty opinion-id keys and authority names",
        ));
    }
    Ok(map)
}

fn append_frontier(
    path: &Path,
    idmap: &BTreeMap<String, CxId>,
    authorities: &BTreeMap<String, String>,
    nodes: &mut BTreeMap<CxId, OpinionNode>,
    edges: &mut Vec<CitesEdge>,
    seen_pairs: &mut BTreeMap<(CxId, CxId), usize>,
    report: &mut SkipReport,
) -> CliResult {
    let file = fs::File::open(path)
        .map_err(|error| CliError::io(format!("open frontier {}: {error}", path.display())))?;
    let mut lines = BufReader::new(file).lines();
    let header = lines
        .next()
        .transpose()
        .map_err(|error| CliError::io(format!("read frontier header: {error}")))?
        .ok_or_else(|| CliError::usage("frontier CSV is empty"))?;
    let columns = header
        .trim()
        .split(',')
        .map(|value| value.trim().to_ascii_lowercase())
        .collect::<Vec<_>>();
    let column = |name: &str| {
        columns
            .iter()
            .position(|value| value == name)
            .ok_or_else(|| CliError::usage(format!("frontier CSV header requires a {name} column")))
    };
    let citation_id_col = column("citation_id")?;
    let citing_col = column("citing_opinion_id")?;
    let cited_col = column("cited_opinion_id")?;
    let depth_col = column("depth")?;
    for (offset, line) in lines.enumerate() {
        let row = offset + 2;
        let line =
            line.map_err(|error| CliError::io(format!("read frontier row {row}: {error}")))?;
        if line.trim().is_empty() {
            continue;
        }
        report.frontier_rows += 1;
        let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
        let citation_id = fields.get(citation_id_col).copied().unwrap_or("");
        let citing_opinion_id = fields.get(citing_col).copied().unwrap_or("");
        let cited_opinion_id = fields.get(cited_col).copied().unwrap_or("");
        let depth_raw = fields.get(depth_col).copied().unwrap_or("");
        if citation_id.is_empty() || citing_opinion_id.is_empty() || cited_opinion_id.is_empty() {
            return Err(CliError::usage(format!(
                "frontier row {row} has a blank citation/citing/cited identity"
            )));
        }
        let src = *idmap.get(citing_opinion_id).ok_or_else(|| {
            CliError::from(calyx_core::CalyxError {
                code: "CALYX_CITATION_FRONTIER_SOURCE_UNRESOLVED",
                message: format!(
                    "frontier row {row} citing opinion {citing_opinion_id} has no sealed-corpus CxId"
                ),
                remediation: "rebuild the frontier from the same immutable corpus identity map",
            })
        })?;
        if idmap.contains_key(cited_opinion_id) {
            return Err(CliError::from(calyx_core::CalyxError {
                code: "CALYX_CITATION_FRONTIER_TARGET_IN_CORPUS",
                message: format!(
                    "frontier row {row} cited opinion {cited_opinion_id} resolves inside the sealed corpus"
                ),
                remediation: "rebuild the citation split; never label an in-corpus authority as frontier",
            }));
        }
        let depth = depth_raw.parse::<u32>().map_err(|error| {
            CliError::usage(format!(
                "frontier row {row} invalid depth {depth_raw}: {error}"
            ))
        })?;
        if depth == 0 {
            return Err(CliError::usage(format!(
                "frontier row {row} depth must be greater than zero"
            )));
        }
        let dst = frontier_node_id(cited_opinion_id);
        if idmap.values().any(|value| *value == dst) {
            return Err(CliError::runtime(format!(
                "frontier identity {dst} collides with a live corpus constellation"
            )));
        }
        let source_citation = SourceCitation {
            citing_opinion_id: citing_opinion_id.to_string(),
            cited_opinion_id: cited_opinion_id.to_string(),
            depth,
            source_row_id: citation_id.to_string(),
        };
        if let Some(index) = seen_pairs.get(&(src, dst)).copied() {
            let edge = &mut edges[index];
            if edge.edge_type != FRONTIER_EDGE_TYPE {
                return Err(CliError::runtime(
                    "frontier citation pair collides with an in-slice edge",
                ));
            }
            edge.depth = edge.depth.max(depth);
            edge.weight = edge.weight.max(weight_for_depth(depth));
            edge.source_citations.push(source_citation);
            report.frontier_duplicate_pair += 1;
            continue;
        }
        nodes.entry(src).or_insert_with(|| OpinionNode {
            cx_id: src,
            opinion_id: citing_opinion_id.to_string(),
            node_type: NODE_TYPE,
            authority_name: None,
            boundary_reason: None,
        });
        let named = authorities.get(cited_opinion_id).cloned();
        nodes.entry(dst).or_insert_with(|| OpinionNode {
            cx_id: dst,
            opinion_id: cited_opinion_id.to_string(),
            node_type: FRONTIER_NODE_TYPE,
            authority_name: Some(
                named.unwrap_or_else(|| format!("CourtListener opinion {cited_opinion_id}")),
            ),
            boundary_reason: Some(FRONTIER_REASON),
        });
        seen_pairs.insert((src, dst), edges.len());
        edges.push(CitesEdge {
            src,
            dst,
            citing_opinion_id: citing_opinion_id.to_string(),
            cited_opinion_id: cited_opinion_id.to_string(),
            depth,
            weight: weight_for_depth(depth),
            source_row_id: citation_id.to_string(),
            edge_type: FRONTIER_EDGE_TYPE,
            source_citations: vec![source_citation],
        });
        report.frontier_edges_built += 1;
    }
    report.frontier_nodes_built = nodes
        .values()
        .filter(|node| node.node_type == FRONTIER_NODE_TYPE)
        .count();
    report.frontier_authority_names = nodes
        .values()
        .filter(|node| {
            node.node_type == FRONTIER_NODE_TYPE
                && authorities.contains_key(node.opinion_id.as_str())
        })
        .count();
    Ok(())
}

fn weight_for_depth(depth: u32) -> f64 {
    f64::from(depth.min(DEPTH_CAP)) / f64::from(DEPTH_CAP)
}

pub(crate) fn frontier_node_id(opinion_id: &str) -> CxId {
    CxId::from_bytes(content_address([
        b"calyx-citation-frontier-v1".as_slice(),
        opinion_id.as_bytes(),
    ]))
}

fn record_skip(
    skip: &mut SkipReport,
    reason: &'static str,
    row: usize,
    citing_opinion_id: &str,
    cited_opinion_id: &str,
) {
    if skip.samples.len() < MAX_SKIP_SAMPLES {
        skip.samples.push(SkipSample {
            row,
            reason,
            citing_opinion_id: citing_opinion_id.to_string(),
            cited_opinion_id: cited_opinion_id.to_string(),
        });
    }
}
