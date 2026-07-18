//! DB-native court/operator summary attribution and arbitrary-Cx coverage (#1847).

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use calyx_aster::base_page_index::visit_indexed_base_rows_for_keys;
use calyx_aster::cf::base_key;
use calyx_aster::plain_graph::{PhysicalGraphCollectionLifecycle, PhysicalPlainGraph};
use calyx_aster::vault::encode::decode_constellation_base;
use calyx_core::{CalyxError, CxId, content_address};
use calyx_lodestar::AsterAssocNodeProps;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::opinion_alias_overlay::verify_idmap_physical;
use super::vault::{home_dir, resolve_vault_info};
use crate::durable_write::write_bytes_atomic_new;
use crate::error::{CliError, CliResult};
use crate::output::print_json;

mod write;

const DEFAULT_COLLECTION: &str = "legal-summary-attribution-v1";
const SCHEMA: &str = "legal_summary_attribution_v1";
const TARGET_NODE: &str = "physical_cx_summary_target";
const COURT_NODE: &str = "court_authored_parenthetical";
const OPERATOR_NODE: &str = "operator_authored_summary";
const COURT_EDGE: &str = "court_parenthetical_describes";
const OPERATOR_EDGE: &str = "operator_summary_describes";

#[derive(Clone, Debug)]
struct MaterializeArgs {
    vault: String,
    cx_set: PathBuf,
    alias_vault: String,
    aliases: PathBuf,
    parentheticals: PathBuf,
    operator_summaries: PathBuf,
    collection: String,
    report: PathBuf,
    home: Option<PathBuf>,
}

#[derive(Clone, Debug)]
struct CoverageArgs {
    vault: String,
    cx_set: PathBuf,
    collection: String,
    report: Option<PathBuf>,
    home: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub(super) struct Target {
    id: CxId,
    input_sha256: String,
    input_pointer: String,
    canonical_opinion_id: String,
    opinion_ids: BTreeSet<String>,
}

#[derive(Clone, Debug)]
pub(super) struct SummaryNode {
    id: CxId,
    node_type: &'static str,
    metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug)]
pub(super) struct SummaryEdge {
    src: CxId,
    dst: CxId,
    edge_type: &'static str,
    value: Value,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(super) struct Counts {
    scope_cx_count: usize,
    physical_base_rows_verified: usize,
    physical_alias_rows_verified: usize,
    scope_opinion_aliases: usize,
    parenthetical_source_rows: usize,
    court_parenthetical_rows_retained: usize,
    court_authored_targets: usize,
    operator_summary_rows: usize,
    operator_authored_targets: usize,
    missing_targets: usize,
    court_operator_overlap_targets: usize,
    all_targets_classified: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct CoverageRow {
    cx_id: String,
    physical_input_sha256: String,
    canonical_opinion_id: String,
    opinion_ids: Vec<String>,
    court_authored_rows: usize,
    operator_authored_rows: usize,
    coverage: &'static str,
}

pub(super) struct Draft {
    targets: BTreeMap<CxId, Target>,
    summary_nodes: BTreeMap<CxId, SummaryNode>,
    edges: BTreeMap<(CxId, &'static str, CxId), SummaryEdge>,
    counts: Counts,
    coverage: Vec<CoverageRow>,
}

#[derive(Clone, Debug)]
struct CourtRow {
    id: String,
    text: String,
    score: String,
    described_opinion_id: String,
    described_canonical_opinion_id: String,
    describing_opinion_id: String,
    describing_canonical_opinion_id: String,
    described_alias_cx_id: CxId,
    describing_alias_cx_id: Option<CxId>,
    group_id: String,
    target: CxId,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OperatorRow {
    source_row_id: String,
    cx_id: String,
    opinion_id: String,
    text: String,
    input_sha256: String,
    input_pointer: String,
    author: String,
    basis: String,
}

#[derive(Debug, Serialize)]
struct CoverageReadbackReport {
    status: &'static str,
    source_of_truth: &'static str,
    vault: String,
    vault_id: String,
    collection: String,
    scope_source: Value,
    counts: CoverageReadbackCounts,
    coverage: Vec<CoverageReadbackRow>,
    doctrine: &'static str,
}

#[derive(Debug, Serialize)]
struct CoverageReadbackCounts {
    requested_cx: usize,
    physical_target_nodes: usize,
    court_authored_targets: usize,
    operator_authored_targets: usize,
    missing_targets: usize,
    court_parenthetical_edges: usize,
    operator_summary_edges: usize,
    accepted_generations: usize,
}

#[derive(Debug, Serialize)]
struct CoverageReadbackRow {
    cx_id: String,
    physical_input_sha256: String,
    court_authored_rows: usize,
    operator_authored_rows: usize,
    coverage: &'static str,
}

pub(crate) fn try_run(args: &[String]) -> Option<CliResult> {
    match args.first().map(String::as_str) {
        Some("materialize-summary-attribution")
            if matches!(args.get(1).map(String::as_str), Some("--help" | "-h")) =>
        {
            Some(crate::usage::print_command_usage(
                "materialize-summary-attribution",
            ))
        }
        Some("materialize-summary-attribution") => {
            Some(parse_materialize(&args[1..]).and_then(materialize))
        }
        Some("summary-coverage")
            if matches!(args.get(1).map(String::as_str), Some("--help" | "-h")) =>
        {
            Some(crate::usage::print_command_usage("summary-coverage"))
        }
        Some("summary-coverage") => Some(parse_coverage(&args[1..]).and_then(coverage_readback)),
        _ => None,
    }
}

fn parse_materialize(rest: &[String]) -> CliResult<MaterializeArgs> {
    let vault = positional(rest, 0, "materialize-summary-attribution requires <vault>")?;
    let mut cx_set = None;
    let mut alias_vault = None;
    let mut aliases = None;
    let mut parentheticals = None;
    let mut operator_summaries = None;
    let mut collection = None;
    let mut report = None;
    let mut home = None;
    let mut index = 1;
    while index < rest.len() {
        let flag = &rest[index];
        let value = rest
            .get(index + 1)
            .ok_or_else(|| CliError::usage(format!("{flag} requires a value")))?;
        match flag.as_str() {
            "--cx-set" => cx_set = Some(value.into()),
            "--alias-vault" => alias_vault = Some(value.clone()),
            "--aliases" => aliases = Some(value.into()),
            "--parentheticals" => parentheticals = Some(value.into()),
            "--operator-summaries" => operator_summaries = Some(value.into()),
            "--collection" => collection = Some(value.clone()),
            "--report" => report = Some(value.into()),
            "--home" => home = Some(value.into()),
            other => {
                return Err(CliError::usage(format!(
                    "unexpected materialize-summary-attribution flag {other}"
                )));
            }
        }
        index += 2;
    }
    Ok(MaterializeArgs {
        alias_vault: alias_vault.unwrap_or_else(|| vault.clone()),
        vault,
        cx_set: required_path(cx_set, "--cx-set <json> is required")?,
        aliases: required_path(aliases, "--aliases <csv> is required")?,
        parentheticals: required_path(parentheticals, "--parentheticals <csv> is required")?,
        operator_summaries: required_path(
            operator_summaries,
            "--operator-summaries <jsonl> is required",
        )?,
        collection: collection.unwrap_or_else(|| DEFAULT_COLLECTION.to_string()),
        report: required_path(report, "--report <json> is required")?,
        home,
    })
}

fn parse_coverage(rest: &[String]) -> CliResult<CoverageArgs> {
    let vault = positional(rest, 0, "summary-coverage requires <vault>")?;
    let mut cx_set = None;
    let mut collection = None;
    let mut report = None;
    let mut home = None;
    let mut index = 1;
    while index < rest.len() {
        let flag = &rest[index];
        let value = rest
            .get(index + 1)
            .ok_or_else(|| CliError::usage(format!("{flag} requires a value")))?;
        match flag.as_str() {
            "--cx-set" => cx_set = Some(value.into()),
            "--collection" => collection = Some(value.clone()),
            "--report" => report = Some(value.into()),
            "--home" => home = Some(value.into()),
            other => {
                return Err(CliError::usage(format!(
                    "unexpected summary-coverage flag {other}"
                )));
            }
        }
        index += 2;
    }
    Ok(CoverageArgs {
        vault,
        cx_set: required_path(cx_set, "--cx-set <json> is required")?,
        collection: collection.unwrap_or_else(|| DEFAULT_COLLECTION.to_string()),
        report,
        home,
    })
}

fn materialize(args: MaterializeArgs) -> CliResult {
    require_new_output(&args.report, "summary attribution report")?;
    let home = args.home.clone().map_or_else(home_dir, Ok)?;
    let resolved = resolve_vault_info(&home, &args.vault)?;
    reject_existing_collection(&resolved.path, &args.collection)?;
    let requested = load_cx_set(&args.cx_set)?;
    let aliases = load_aliases(&args.aliases)?;
    let physical_alias_rows_verified = verify_idmap_physical(&home, &args.alias_vault, &aliases)?;
    let targets = load_physical_targets(&resolved.path, &requested, &aliases)?;
    let (parenthetical_source_rows, court_rows) =
        load_court_rows(&args.parentheticals, &aliases, &targets)?;
    let operator_rows = load_operator_rows(&args.operator_summaries, &aliases, &targets)?;
    let draft = build_draft(
        targets,
        court_rows,
        operator_rows,
        parenthetical_source_rows,
        physical_alias_rows_verified,
    )?;
    let report = write::write_to_calyx(&home, &args, draft)?;
    print_json(&report)
}

fn load_cx_set(path: &Path) -> CliResult<BTreeMap<CxId, BTreeSet<String>>> {
    let bytes = read_plain(path, "Cx-set JSON")?;
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|error| CliError::usage(format!("decode Cx-set JSON: {error}")))?;
    let rows = value
        .as_array()
        .ok_or_else(|| CliError::usage("Cx-set JSON must be an array"))?;
    let mut ids = BTreeMap::new();
    for (offset, row) in rows.iter().enumerate() {
        let text = row
            .as_str()
            .or_else(|| row.get("cx_id").and_then(Value::as_str));
        let text = text.ok_or_else(|| {
            CliError::usage(format!(
                "Cx-set row {} must be a Cx string or object with cx_id",
                offset + 1
            ))
        })?;
        let id = text.parse::<CxId>().map_err(|error| {
            CliError::usage(format!("Cx-set row {} invalid cx_id: {error}", offset + 1))
        })?;
        if ids.contains_key(&id) {
            return Err(CliError::usage(format!(
                "Cx-set row {} duplicates {id}",
                offset + 1
            )));
        }
        let mut opinion_ids = BTreeSet::new();
        if let Some(opinion_id) = row.get("opinion_id").and_then(Value::as_str) {
            if opinion_id.trim().is_empty() {
                return Err(CliError::usage(format!(
                    "Cx-set row {} has a blank opinion_id",
                    offset + 1
                )));
            }
            opinion_ids.insert(opinion_id.to_string());
        }
        ids.insert(id, opinion_ids);
    }
    if ids.is_empty() {
        return Err(CliError::usage("Cx-set must not be empty"));
    }
    Ok(ids)
}

fn load_aliases(path: &Path) -> CliResult<BTreeMap<String, CxId>> {
    let file = plain_file(path, "opinion alias CSV")?;
    let mut lines = BufReader::new(file).lines();
    let header = lines
        .next()
        .transpose()
        .map_err(|error| CliError::io(format!("read alias header: {error}")))?
        .ok_or_else(|| CliError::usage("opinion alias CSV is empty"))?;
    if header.trim_end_matches('\r')
        != "opinion_id,cx_id,canonical_opinion_id,content_sha256,is_canonical,source_url"
    {
        return Err(CliError::usage("opinion alias CSV has the wrong schema"));
    }
    let mut aliases = BTreeMap::new();
    for (offset, line) in lines.enumerate() {
        let row = offset + 2;
        let line = line.map_err(|error| CliError::io(format!("read alias row {row}: {error}")))?;
        let fields = line.trim_end_matches('\r').split(',').collect::<Vec<_>>();
        if fields.len() != 6 || fields[0].is_empty() {
            return Err(CliError::usage(format!(
                "alias row {row} has the wrong shape"
            )));
        }
        let cx = fields[1]
            .parse::<CxId>()
            .map_err(|error| CliError::usage(format!("alias row {row} invalid cx_id: {error}")))?;
        if aliases.insert(fields[0].to_string(), cx).is_some() {
            return Err(CliError::usage(format!(
                "alias row {row} duplicates opinion {}",
                fields[0]
            )));
        }
    }
    if aliases.is_empty() {
        return Err(CliError::usage("opinion alias CSV is empty"));
    }
    Ok(aliases)
}

fn load_physical_targets(
    vault: &Path,
    requested: &BTreeMap<CxId, BTreeSet<String>>,
    aliases: &BTreeMap<String, CxId>,
) -> CliResult<BTreeMap<CxId, Target>> {
    let expected = requested
        .keys()
        .map(|id| (base_key(*id), *id))
        .collect::<BTreeMap<_, _>>();
    let keys = expected.keys().cloned().collect::<Vec<_>>();
    let mut targets = BTreeMap::new();
    let stats = visit_indexed_base_rows_for_keys(vault, &keys, |key, value| {
        let id = *expected.get(key).ok_or_else(|| {
            contract_error(
                "CALYX_SUMMARY_BASE_KEY_MISMATCH",
                "physical Base read returned an unrequested key",
                "rebuild the Base page index before summary attribution",
            )
        })?;
        let bytes = value.ok_or_else(|| {
            contract_error(
                "CALYX_SUMMARY_BASE_ROW_MISSING",
                format!("physical Base row {id} is absent"),
                "restore the requested Cx before claiming summary coverage",
            )
        })?;
        let base = decode_constellation_base(&bytes)?;
        if base.cx_id != id {
            return Err(contract_error(
                "CALYX_SUMMARY_BASE_ID_MISMATCH",
                format!("physical Base row for {id} decodes as {}", base.cx_id),
                "quarantine and rebuild the Base page index",
            ));
        }
        let canonical_opinion_id = base
            .metadata
            .get("canonical_opinion_id")
            .or_else(|| base.metadata.get("opinion_id"))
            .cloned()
            .ok_or_else(|| {
                contract_error(
                    "CALYX_SUMMARY_BASE_METADATA_MISSING",
                    format!("physical Base row {id} omits canonical opinion identity"),
                    "restore the source-bound canonical Base metadata",
                )
            })?;
        let canonical_alias_cx = aliases.get(&canonical_opinion_id).ok_or_else(|| {
            contract_error(
                "CALYX_SUMMARY_SCOPE_ALIAS_MISSING",
                format!(
                    "physical Base row {id} canonical opinion {canonical_opinion_id} is absent from alias authority"
                ),
                "rebuild the scope from the accepted physical alias relation",
            )
        })?;
        let mut opinion_ids = requested.get(&id).cloned().unwrap_or_default();
        opinion_ids.insert(canonical_opinion_id.clone());
        for opinion_id in &opinion_ids {
            if aliases.get(opinion_id) != Some(canonical_alias_cx) {
                return Err(contract_error(
                    "CALYX_SUMMARY_BASE_ALIAS_MISMATCH",
                    format!(
                        "scoped Cx {id} opinion {opinion_id} does not share alias authority with canonical opinion {canonical_opinion_id}"
                    ),
                    "repair the cross-vault opinion identity join without equating vault-bound Cx IDs",
                ));
            }
        }
        let input_pointer = base.input_ref.pointer.clone().ok_or_else(|| {
            contract_error(
                "CALYX_SUMMARY_INPUT_POINTER_MISSING",
                format!("physical Base row {id} has no retained input pointer"),
                "restore the retained input bytes before authoring a summary",
            )
        })?;
        targets.insert(
            id,
            Target {
                id,
                input_sha256: hex(&base.input_ref.hash),
                input_pointer,
                canonical_opinion_id,
                opinion_ids,
            },
        );
        Ok(())
    })?;
    if stats.unique_keys != requested.len()
        || stats.live_rows != requested.len()
        || stats.missing_rows != 0
        || targets.len() != requested.len()
    {
        return Err(contract_error(
            "CALYX_SUMMARY_BASE_SCOPE_INCOMPLETE",
            format!(
                "requested={} unique_keys={} live_rows={} missing_rows={} decoded={}",
                requested.len(),
                stats.unique_keys,
                stats.live_rows,
                stats.missing_rows,
                targets.len()
            ),
            "rebuild the Base page index and retry the complete Cx set",
        ));
    }
    Ok(targets)
}

fn load_court_rows(
    path: &Path,
    aliases: &BTreeMap<String, CxId>,
    targets: &BTreeMap<CxId, Target>,
) -> CliResult<(usize, Vec<CourtRow>)> {
    let scope_identities =
        targets
            .values()
            .fold(BTreeMap::<String, CxId>::new(), |mut joined, target| {
                for opinion_id in &target.opinion_ids {
                    joined.insert(opinion_id.clone(), target.id);
                }
                joined
            });
    let file = plain_file(path, "parenthetical CSV")?;
    let mut lines = BufReader::new(file).lines();
    let header = lines
        .next()
        .transpose()
        .map_err(|error| CliError::io(format!("read parenthetical header: {error}")))?
        .ok_or_else(|| CliError::usage("parenthetical CSV is empty"))?;
    let expected = [
        "id",
        "text",
        "score",
        "described_opinion_id",
        "describing_opinion_id",
        "described_canonical_opinion_id",
        "describing_canonical_opinion_id",
        "group_id",
    ];
    if parse_csv_line(header.trim_end_matches('\r'))? != expected {
        return Err(CliError::usage("parenthetical CSV has the wrong schema"));
    }
    let mut source_rows = 0;
    let mut retained = Vec::new();
    let mut retained_ids = BTreeSet::new();
    for (offset, line) in lines.enumerate() {
        let row_number = offset + 2;
        let line = line.map_err(|error| {
            CliError::io(format!("read parenthetical row {row_number}: {error}"))
        })?;
        source_rows += 1;
        let fields = parse_csv_line(line.trim_end_matches('\r'))?;
        if fields.len() != expected.len() {
            return Err(CliError::usage(format!(
                "parenthetical row {row_number} has {} fields",
                fields.len()
            )));
        }
        if fields[0].is_empty() || fields[1].trim().is_empty() {
            return Err(CliError::usage(format!(
                "parenthetical row {row_number} omits id or text"
            )));
        }
        let score = fields[2].parse::<f64>().map_err(|error| {
            CliError::usage(format!(
                "parenthetical row {row_number} invalid score: {error}"
            ))
        })?;
        if !score.is_finite() {
            return Err(CliError::usage(format!(
                "parenthetical row {row_number} score is non-finite"
            )));
        }
        let described_alias_cx_id =
            resolve_pair(aliases, &fields[3], &fields[5], row_number, "described")?;
        let Some(target) =
            resolve_scope_pair(&scope_identities, &fields[3], &fields[5], row_number)?
        else {
            continue;
        };
        let described_alias_cx_id = described_alias_cx_id.ok_or_else(|| {
            contract_error(
                "CALYX_SUMMARY_PARENTHETICAL_ALIAS_GAP",
                format!("parenthetical row {row_number} target has no alias authority"),
                "retain a relation only after its scoped opinion identity resolves in the canonical alias vault",
            )
        })?;
        let describing_alias_cx_id =
            resolve_pair(aliases, &fields[4], &fields[6], row_number, "describing")?;
        if !retained_ids.insert(fields[0].clone()) {
            return Err(CliError::usage(format!(
                "parenthetical row {row_number} duplicates id {}",
                fields[0]
            )));
        }
        retained.push(CourtRow {
            id: fields[0].clone(),
            text: fields[1].clone(),
            score: fields[2].clone(),
            described_opinion_id: fields[3].clone(),
            describing_opinion_id: fields[4].clone(),
            described_canonical_opinion_id: fields[5].clone(),
            describing_canonical_opinion_id: fields[6].clone(),
            described_alias_cx_id,
            describing_alias_cx_id,
            group_id: fields[7].clone(),
            target,
        });
    }
    Ok((source_rows, retained))
}

fn resolve_scope_pair(
    scope: &BTreeMap<String, CxId>,
    source: &str,
    canonical: &str,
    row: usize,
) -> CliResult<Option<CxId>> {
    let source_id = scope.get(source).copied();
    let canonical_id = scope.get(canonical).copied();
    if let (Some(left), Some(right)) = (source_id, canonical_id)
        && left != right
    {
        return Err(contract_error(
            "CALYX_SUMMARY_SCOPE_IDENTITY_CONFLICT",
            format!(
                "parenthetical row {row} source/canonical identities map to scoped Cx {left} and {right}"
            ),
            "repair the scoped kernel identity set before reporting coverage",
        ));
    }
    Ok(source_id.or(canonical_id))
}

fn resolve_pair(
    aliases: &BTreeMap<String, CxId>,
    source: &str,
    canonical: &str,
    row: usize,
    role: &str,
) -> CliResult<Option<CxId>> {
    let source_id = (!source.is_empty())
        .then(|| aliases.get(source))
        .flatten()
        .copied();
    let canonical_id = (!canonical.is_empty())
        .then(|| aliases.get(canonical))
        .flatten()
        .copied();
    if let (Some(left), Some(right)) = (source_id, canonical_id)
        && left != right
    {
        return Err(contract_error(
            "CALYX_SUMMARY_PARENTHETICAL_ALIAS_CONFLICT",
            format!(
                "parenthetical row {row} {role} source/canonical aliases resolve to {left} and {right}"
            ),
            "repair the accepted source/canonical alias join before retaining the relation",
        ));
    }
    let resolved = source_id.or(canonical_id);
    if resolved.is_some()
        && ((!source.is_empty() && source_id.is_none())
            || (!canonical.is_empty() && canonical_id.is_none()))
    {
        return Err(contract_error(
            "CALYX_SUMMARY_PARENTHETICAL_ALIAS_GAP",
            format!("parenthetical row {row} has a partial {role} alias join"),
            "retain the relation only after every populated in-slice identity resolves physically",
        ));
    }
    Ok(resolved)
}

fn load_operator_rows(
    path: &Path,
    aliases: &BTreeMap<String, CxId>,
    targets: &BTreeMap<CxId, Target>,
) -> CliResult<Vec<OperatorRow>> {
    let file = plain_file(path, "operator summary JSONL")?;
    let mut rows = Vec::new();
    let mut ids = BTreeSet::new();
    let mut target_ids = BTreeSet::new();
    for (offset, line) in BufReader::new(file).lines().enumerate() {
        let row_number = offset + 1;
        let line = line.map_err(|error| {
            CliError::io(format!("read operator summary row {row_number}: {error}"))
        })?;
        let row: OperatorRow = serde_json::from_str(&line).map_err(|error| {
            CliError::usage(format!("decode operator summary row {row_number}: {error}"))
        })?;
        if !ids.insert(row.source_row_id.clone()) || row.source_row_id.trim().is_empty() {
            return Err(CliError::usage(format!(
                "operator summary row {row_number} has blank/duplicate source_row_id"
            )));
        }
        let cx = row.cx_id.parse::<CxId>().map_err(|error| {
            CliError::usage(format!(
                "operator summary row {row_number} invalid cx_id: {error}"
            ))
        })?;
        let target = targets.get(&cx).ok_or_else(|| {
            contract_error(
                "CALYX_SUMMARY_OPERATOR_OUTSIDE_SCOPE",
                format!("operator summary row {row_number} targets {cx} outside the Cx set"),
                "bind operator summaries only to requested physical Cx rows",
            )
        })?;
        let canonical_alias_cx = aliases.get(&target.canonical_opinion_id);
        if !target.opinion_ids.contains(&row.opinion_id)
            || aliases.get(&row.opinion_id) != canonical_alias_cx
            || canonical_alias_cx.is_none()
        {
            return Err(contract_error(
                "CALYX_SUMMARY_OPERATOR_ALIAS_MISMATCH",
                format!(
                    "operator summary row {row_number} opinion {} does not resolve to {cx}",
                    row.opinion_id
                ),
                "bind the operator row through the accepted physical alias relation",
            ));
        }
        if row.input_sha256 != target.input_sha256 || row.input_pointer != target.input_pointer {
            return Err(contract_error(
                "CALYX_SUMMARY_OPERATOR_PROVENANCE_MISMATCH",
                format!("operator summary row {row_number} differs from physical input for {cx}"),
                "re-author or rebind the summary to the exact retained input bytes",
            ));
        }
        if row.author != "operator"
            || row.basis != "retained_physical_opinion"
            || row.text.trim().is_empty()
        {
            return Err(contract_error(
                "CALYX_SUMMARY_OPERATOR_LABEL_INVALID",
                format!("operator summary row {row_number} lacks explicit attribution or text"),
                "label the row author=operator and basis=retained_physical_opinion; never present it as court-authored",
            ));
        }
        if !target_ids.insert(cx) {
            return Err(contract_error(
                "CALYX_SUMMARY_OPERATOR_TARGET_DUPLICATE",
                format!("multiple operator summaries target {cx}"),
                "retain one provenance-bound operator summary per physical Cx in this generation",
            ));
        }
        rows.push(row);
    }
    Ok(rows)
}

fn build_draft(
    targets: BTreeMap<CxId, Target>,
    court_rows: Vec<CourtRow>,
    operator_rows: Vec<OperatorRow>,
    parenthetical_source_rows: usize,
    physical_alias_rows_verified: usize,
) -> CliResult<Draft> {
    let mut nodes = BTreeMap::new();
    let mut edges = BTreeMap::new();
    for row in &court_rows {
        let node_id = summary_node_id(COURT_NODE, &row.id, &row.text);
        let metadata = BTreeMap::from([
            ("node_type".to_string(), COURT_NODE.to_string()),
            ("attribution".to_string(), "court_authored".to_string()),
            ("parenthetical_id".to_string(), row.id.clone()),
            ("text".to_string(), row.text.clone()),
            ("score".to_string(), row.score.clone()),
            (
                "described_opinion_id".to_string(),
                row.described_opinion_id.clone(),
            ),
            (
                "described_canonical_opinion_id".to_string(),
                row.described_canonical_opinion_id.clone(),
            ),
            (
                "describing_opinion_id".to_string(),
                row.describing_opinion_id.clone(),
            ),
            (
                "describing_canonical_opinion_id".to_string(),
                row.describing_canonical_opinion_id.clone(),
            ),
            ("scoped_target_cx_id".to_string(), row.target.to_string()),
            (
                "described_alias_authority_cx_id".to_string(),
                row.described_alias_cx_id.to_string(),
            ),
            (
                "describing_alias_authority_cx_id".to_string(),
                row.describing_alias_cx_id
                    .map(|id| id.to_string())
                    .unwrap_or_default(),
            ),
            ("group_id".to_string(), row.group_id.clone()),
        ]);
        insert_node(
            &mut nodes,
            SummaryNode {
                id: node_id,
                node_type: COURT_NODE,
                metadata,
            },
        )?;
        insert_edge(
            &mut edges,
            SummaryEdge {
                src: node_id,
                dst: row.target,
                edge_type: COURT_EDGE,
                value: json!({
                    "schema": SCHEMA,
                    "attribution": "court_authored",
                    "parenthetical_id": row.id,
                    "described_opinion_id": row.described_opinion_id,
                    "described_canonical_opinion_id": row.described_canonical_opinion_id,
                    "describing_opinion_id": row.describing_opinion_id,
                    "describing_canonical_opinion_id": row.describing_canonical_opinion_id,
                    "scoped_target_cx_id": row.target,
                    "described_alias_authority_cx_id": row.described_alias_cx_id,
                    "describing_alias_authority_cx_id": row.describing_alias_cx_id,
                    "group_id": row.group_id,
                    "text_sha256": sha256(row.text.as_bytes()),
                    "weight": 1.0,
                }),
            },
        )?;
    }
    for row in &operator_rows {
        let target = row.cx_id.parse::<CxId>().map_err(|error| {
            CliError::runtime(format!("validated operator cx no longer parses: {error}"))
        })?;
        let node_id = summary_node_id(OPERATOR_NODE, &row.source_row_id, &row.text);
        let metadata = BTreeMap::from([
            ("node_type".to_string(), OPERATOR_NODE.to_string()),
            ("attribution".to_string(), "operator_authored".to_string()),
            ("source_row_id".to_string(), row.source_row_id.clone()),
            ("text".to_string(), row.text.clone()),
            ("text_sha256".to_string(), sha256(row.text.as_bytes())),
            ("author".to_string(), row.author.clone()),
            ("basis".to_string(), row.basis.clone()),
            ("opinion_id".to_string(), row.opinion_id.clone()),
            ("physical_cx_id".to_string(), row.cx_id.clone()),
            (
                "physical_input_sha256".to_string(),
                row.input_sha256.clone(),
            ),
            (
                "physical_input_pointer".to_string(),
                row.input_pointer.clone(),
            ),
        ]);
        insert_node(
            &mut nodes,
            SummaryNode {
                id: node_id,
                node_type: OPERATOR_NODE,
                metadata,
            },
        )?;
        insert_edge(
            &mut edges,
            SummaryEdge {
                src: node_id,
                dst: target,
                edge_type: OPERATOR_EDGE,
                value: json!({
                    "schema": SCHEMA,
                    "attribution": "operator_authored",
                    "source_row_id": row.source_row_id,
                    "opinion_id": row.opinion_id,
                    "physical_cx_id": row.cx_id,
                    "physical_input_sha256": row.input_sha256,
                    "physical_input_pointer": row.input_pointer,
                    "text_sha256": sha256(row.text.as_bytes()),
                    "weight": 1.0,
                }),
            },
        )?;
    }
    let mut coverage = Vec::with_capacity(targets.len());
    let mut court_targets = 0;
    let mut operator_targets = 0;
    let mut missing_targets = 0;
    let mut overlap = 0;
    for target in targets.values() {
        let court = edges
            .values()
            .filter(|edge| edge.dst == target.id && edge.edge_type == COURT_EDGE)
            .count();
        let operator = edges
            .values()
            .filter(|edge| edge.dst == target.id && edge.edge_type == OPERATOR_EDGE)
            .count();
        if court > 0 {
            court_targets += 1;
        }
        if operator > 0 {
            operator_targets += 1;
        }
        if court > 0 && operator > 0 {
            overlap += 1;
        }
        let kind = if court > 0 {
            "court_authored"
        } else if operator > 0 {
            "operator_authored"
        } else {
            missing_targets += 1;
            "missing"
        };
        coverage.push(CoverageRow {
            cx_id: target.id.to_string(),
            physical_input_sha256: target.input_sha256.clone(),
            canonical_opinion_id: target.canonical_opinion_id.clone(),
            opinion_ids: target.opinion_ids.iter().cloned().collect(),
            court_authored_rows: court,
            operator_authored_rows: operator,
            coverage: kind,
        });
    }
    let scope_aliases = targets
        .values()
        .map(|target| target.opinion_ids.len())
        .sum();
    let counts = Counts {
        scope_cx_count: targets.len(),
        physical_base_rows_verified: targets.len(),
        physical_alias_rows_verified,
        scope_opinion_aliases: scope_aliases,
        parenthetical_source_rows,
        court_parenthetical_rows_retained: court_rows.len(),
        court_authored_targets: court_targets,
        operator_summary_rows: operator_rows.len(),
        operator_authored_targets: operator_targets,
        missing_targets,
        court_operator_overlap_targets: overlap,
        all_targets_classified: court_targets + operator_targets + missing_targets - overlap
            == targets.len(),
    };
    Ok(Draft {
        targets,
        summary_nodes: nodes,
        edges,
        counts,
        coverage,
    })
}

fn coverage_readback(args: CoverageArgs) -> CliResult {
    if let Some(path) = args.report.as_deref() {
        require_new_output(path, "summary coverage report")?;
    }
    let requested = load_cx_set(&args.cx_set)?;
    let home = args.home.clone().map_or_else(home_dir, Ok)?;
    let resolved = resolve_vault_info(&home, &args.vault)?;
    let lifecycle = PhysicalGraphCollectionLifecycle::open_latest(&resolved.path)?;
    let accepted = lifecycle
        .list_states()?
        .into_iter()
        .filter(|row| {
            row.state.collection == args.collection
                && row.state.status
                    == calyx_aster::plain_graph::GraphCollectionGenerationStatus::Accepted
        })
        .count();
    if accepted == 0 {
        return Err(contract_error(
            "CALYX_SUMMARY_COVERAGE_UNACCEPTED",
            format!("collection {} has no accepted generation", args.collection),
            "materialize and physically accept summary attribution before reporting coverage",
        ));
    }
    let physical = PhysicalPlainGraph::open_latest(&resolved.path, &args.collection)?;
    let nodes = physical
        .node_props()?
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    let edges = physical.edge_out_props()?;
    let mut rows = Vec::with_capacity(requested.len());
    let mut court_targets = 0;
    let mut operator_targets = 0;
    let mut missing = 0;
    let mut court_edges = 0;
    let mut operator_edges = 0;
    for id in requested.keys() {
        let bytes = nodes.get(id).ok_or_else(|| {
            contract_error(
                "CALYX_SUMMARY_COVERAGE_TARGET_MISSING",
                format!("requested Cx {id} has no physical summary target node"),
                "materialize this exact Cx set before promising reading-list coverage",
            )
        })?;
        let props: AsterAssocNodeProps = serde_json::from_slice(bytes)
            .map_err(|error| CliError::runtime(format!("decode summary target {id}: {error}")))?;
        if props.metadata.get("node_type").map(String::as_str) != Some(TARGET_NODE)
            || props.metadata.get("physical_cx_id") != Some(&id.to_string())
        {
            return Err(contract_error(
                "CALYX_SUMMARY_COVERAGE_TARGET_INVALID",
                format!("requested Cx {id} target node has the wrong type or identity"),
                "quarantine and rebuild the summary attribution collection",
            ));
        }
        let mut court = 0;
        let mut operator = 0;
        let id_text = id.to_string();
        for edge in edges.iter().filter(|edge| edge.dst == *id) {
            let (expected, expected_node_type, target_field) = match edge.edge_type.as_str() {
                COURT_EDGE => {
                    court += 1;
                    ("court_authored", COURT_NODE, "scoped_target_cx_id")
                }
                OPERATOR_EDGE => {
                    operator += 1;
                    ("operator_authored", OPERATOR_NODE, "physical_cx_id")
                }
                _ => continue,
            };
            let value: Value = serde_json::from_slice(&edge.value).map_err(|error| {
                CliError::runtime(format!("decode summary attribution edge: {error}"))
            })?;
            let source_bytes = nodes.get(&edge.src).ok_or_else(|| {
                contract_error(
                    "CALYX_SUMMARY_ATTRIBUTION_SOURCE_MISSING",
                    format!(
                        "edge {} -{}-> {} has no physical source node",
                        edge.src, edge.edge_type, edge.dst
                    ),
                    "quarantine and rebuild the typed summary attribution collection",
                )
            })?;
            let source: AsterAssocNodeProps =
                serde_json::from_slice(source_bytes).map_err(|error| {
                    CliError::runtime(format!("decode summary source node {}: {error}", edge.src))
                })?;
            let source_text_sha256 = source
                .metadata
                .get("text")
                .map(|text| sha256(text.as_bytes()));
            if value.get("attribution").and_then(Value::as_str) != Some(expected)
                || value.get(target_field).and_then(Value::as_str) != Some(id_text.as_str())
                || source.metadata.get("node_type").map(String::as_str) != Some(expected_node_type)
                || source.metadata.get("attribution").map(String::as_str) != Some(expected)
                || source_text_sha256.as_deref() != value.get("text_sha256").and_then(Value::as_str)
            {
                return Err(contract_error(
                    "CALYX_SUMMARY_ATTRIBUTION_TYPE_MISMATCH",
                    format!(
                        "edge {} -{}-> {} is not a source-bound {expected} attribution",
                        edge.src, edge.edge_type, edge.dst
                    ),
                    "quarantine and rebuild the typed summary attribution collection",
                ));
            }
        }
        court_edges += court;
        operator_edges += operator;
        let kind = if court > 0 {
            court_targets += 1;
            "court_authored"
        } else if operator > 0 {
            operator_targets += 1;
            "operator_authored"
        } else {
            missing += 1;
            "missing"
        };
        rows.push(CoverageReadbackRow {
            cx_id: id.to_string(),
            physical_input_sha256: props
                .metadata
                .get("physical_input_sha256")
                .cloned()
                .unwrap_or_default(),
            court_authored_rows: court,
            operator_authored_rows: operator,
            coverage: kind,
        });
    }
    let report = CoverageReadbackReport {
        status: "complete",
        source_of_truth: "physical accepted Aster Graph CF target/attribution edge bytes",
        vault: resolved.name,
        vault_id: resolved.vault_id.to_string(),
        collection: args.collection,
        scope_source: source_contract("requested_cx_set", &args.cx_set)?,
        counts: CoverageReadbackCounts {
            requested_cx: requested.len(),
            physical_target_nodes: rows.len(),
            court_authored_targets: court_targets,
            operator_authored_targets: operator_targets,
            missing_targets: missing,
            court_parenthetical_edges: court_edges,
            operator_summary_edges: operator_edges,
            accepted_generations: accepted,
        },
        coverage: rows,
        doctrine: "court-authored, operator-authored, and missing are distinct; typed constellation slots remain separate and untouched",
    };
    if let Some(path) = args.report.as_deref() {
        write_report(path, &report, "summary coverage report")?;
    }
    print_json(&report)
}

fn insert_node(nodes: &mut BTreeMap<CxId, SummaryNode>, node: SummaryNode) -> CliResult {
    if nodes.insert(node.id, node).is_some() {
        return Err(contract_error(
            "CALYX_SUMMARY_NODE_COLLISION",
            "summary node content address collided",
            "change the source identity or quarantine corrupt summary input",
        ));
    }
    Ok(())
}

fn insert_edge(
    edges: &mut BTreeMap<(CxId, &'static str, CxId), SummaryEdge>,
    edge: SummaryEdge,
) -> CliResult {
    let key = (edge.src, edge.edge_type, edge.dst);
    if edges.insert(key, edge).is_some() {
        return Err(contract_error(
            "CALYX_SUMMARY_EDGE_DUPLICATE",
            "summary attribution edge is duplicated",
            "deduplicate source rows without dropping distinct provenance",
        ));
    }
    Ok(())
}

fn summary_node_id(kind: &str, source_id: &str, text: &str) -> CxId {
    CxId::from_bytes(content_address([
        b"calyx-legal-summary-attribution-v1".as_slice(),
        kind.as_bytes(),
        source_id.as_bytes(),
        sha256(text.as_bytes()).as_bytes(),
    ]))
}

fn parse_csv_line(line: &str) -> CliResult<Vec<String>> {
    let chars = line.chars().collect::<Vec<_>>();
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut index = 0;
    let mut quoted = false;
    let mut closed_quote = false;
    while index < chars.len() {
        let ch = chars[index];
        if quoted {
            if ch == '"' {
                if chars.get(index + 1) == Some(&'"') {
                    field.push('"');
                    index += 2;
                    continue;
                }
                quoted = false;
                closed_quote = true;
            } else {
                field.push(ch);
            }
        } else if closed_quote {
            if ch != ',' {
                return Err(CliError::usage(
                    "CSV has characters between a closing quote and delimiter",
                ));
            }
            fields.push(std::mem::take(&mut field));
            closed_quote = false;
        } else if ch == ',' {
            fields.push(std::mem::take(&mut field));
        } else if ch == '"' {
            if !field.is_empty() {
                return Err(CliError::usage(
                    "CSV quote appears inside an unquoted field",
                ));
            }
            quoted = true;
        } else {
            field.push(ch);
        }
        index += 1;
    }
    if quoted {
        return Err(CliError::usage("CSV ends inside a quoted field"));
    }
    fields.push(field);
    Ok(fields)
}

fn positional(rest: &[String], index: usize, message: &'static str) -> CliResult<String> {
    rest.get(index)
        .filter(|value| !value.starts_with('-'))
        .cloned()
        .ok_or_else(|| CliError::usage(message))
}

fn required_path(path: Option<PathBuf>, message: &'static str) -> CliResult<PathBuf> {
    path.ok_or_else(|| CliError::usage(message))
}

fn reject_existing_collection(vault: &Path, collection: &str) -> CliResult {
    let lifecycle = PhysicalGraphCollectionLifecycle::open_latest(vault)?;
    if lifecycle
        .list_states()?
        .into_iter()
        .any(|row| row.state.collection == collection)
    {
        return Err(contract_error(
            "CALYX_SUMMARY_COLLECTION_EXISTS",
            format!("summary collection {collection} already has lifecycle state"),
            "use the accepted immutable generation or choose a new collection name",
        ));
    }
    Ok(())
}

fn require_new_output(path: &Path, label: &str) -> CliResult {
    if fs::symlink_metadata(path).is_ok() {
        return Err(contract_error(
            "CALYX_SUMMARY_REPORT_EXISTS",
            format!("{label} {} already exists", path.display()),
            "choose a new immutable report path",
        ));
    }
    if let Some(parent) = path.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .map_err(|error| CliError::io(format!("create report parent: {error}")))?;
    }
    Ok(())
}

pub(super) fn write_report(path: &Path, value: &impl Serialize, label: &str) -> CliResult {
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| CliError::runtime(format!("serialize {label}: {error}")))?;
    bytes.push(b'\n');
    write_bytes_atomic_new(path, &bytes, label)?;
    if fs::read(path).map_err(|error| CliError::io(format!("read back {label}: {error}")))? != bytes
    {
        return Err(CliError::runtime(format!("{label} readback mismatch")));
    }
    Ok(())
}

fn plain_file(path: &Path, label: &str) -> CliResult<fs::File> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| CliError::io(format!("inspect {label} {}: {error}", path.display())))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(CliError::usage(format!(
            "{label} {} is not a plain file",
            path.display()
        )));
    }
    fs::File::open(path)
        .map_err(|error| CliError::io(format!("open {label} {}: {error}", path.display())))
}

fn read_plain(path: &Path, label: &str) -> CliResult<Vec<u8>> {
    let mut file = plain_file(path, label)?;
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut file, &mut bytes)
        .map_err(|error| CliError::io(format!("read {label}: {error}")))?;
    Ok(bytes)
}

pub(super) fn source_contract(role: &str, path: &Path) -> CliResult<Value> {
    let bytes = read_plain(path, role)?;
    Ok(json!({
        "role": role,
        "path": path.display().to_string(),
        "bytes": bytes.len(),
        "sha256": sha256(&bytes),
    }))
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn contract_error(
    code: &'static str,
    message: impl Into<String>,
    remediation: &'static str,
) -> CliError {
    CliError::from(CalyxError {
        code,
        message: message.into(),
        remediation,
    })
}
