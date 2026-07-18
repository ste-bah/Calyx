use super::KernelBuildArgs;
use crate::cmd::{Subcommand, value};
use crate::error::{CliError, CliResult};

pub(crate) fn parse_kernel_build(rest: &[String]) -> CliResult<Subcommand> {
    let vault = rest
        .first()
        .ok_or_else(|| CliError::usage("kernel-build requires <vault>"))?
        .clone();
    let mut args = KernelBuildArgs {
        vault,
        ..KernelBuildArgs::default()
    };
    let mut idx = 1;
    while idx < rest.len() {
        match rest[idx].as_str() {
            "--held-out-fraction" => {
                idx += 1;
                args.held_out_fraction = parse_unit(
                    value(rest, idx, "--held-out-fraction")?,
                    "--held-out-fraction",
                )?;
            }
            "--top-k" => {
                idx += 1;
                let raw = value(rest, idx, "--top-k")?;
                args.top_k = raw
                    .parse::<usize>()
                    .map_err(|err| CliError::usage(format!("parse --top-k {raw}: {err}")))?;
                if args.top_k == 0 {
                    return Err(CliError::usage("--top-k must be >= 1"));
                }
            }
            "--min-recall" => {
                idx += 1;
                args.min_recall = parse_unit(value(rest, idx, "--min-recall")?, "--min-recall")?;
            }
            "--admission-queries" => {
                idx += 1;
                args.admission_queries = Some(value(rest, idx, "--admission-queries")?.into());
            }
            "--resident-addr" => {
                idx += 1;
                args.resident_addr = Some(super::super::search::parse_resident_addr(value(
                    rest,
                    idx,
                    "--resident-addr",
                )?)?);
            }
            other => {
                return Err(CliError::usage(format!(
                    "unexpected kernel-build flag {other}"
                )));
            }
        }
        idx += 1;
    }
    if args.admission_queries.is_some() != args.resident_addr.is_some() {
        return Err(CliError::usage(
            "--admission-queries and --resident-addr must be supplied together",
        ));
    }
    Ok(Subcommand::KernelBuild(args))
}

fn parse_unit(raw: &str, flag: &str) -> CliResult<f32> {
    let value = raw
        .parse::<f32>()
        .map_err(|err| CliError::usage(format!("parse {flag} {raw}: {err}")))?;
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(CliError::usage(format!(
            "{flag} must be finite and in [0,1]"
        )));
    }
    Ok(value)
}
