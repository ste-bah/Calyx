use std::env;

const DEFAULT_MAXSIM_CUDA_MIN_TOKENS: usize = 65_536;
#[cfg(feature = "cuda")]
const DEFAULT_MAXSIM_CUDA_CHUNK_ROWS: usize = 512;
#[cfg(feature = "cuda")]
const DEFAULT_MAXSIM_CUDA_CHUNK_TOKENS: usize = 131_072;

pub(super) fn maxsim_cuda_min_tokens() -> usize {
    env::var("CALYX_SEARCH_MAXSIM_CUDA_MIN_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAXSIM_CUDA_MIN_TOKENS)
}

#[cfg(feature = "cuda")]
pub(super) fn maxsim_cuda_chunk_rows() -> usize {
    env::var("CALYX_SEARCH_MAXSIM_CUDA_CHUNK_ROWS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_MAXSIM_CUDA_CHUNK_ROWS)
}

#[cfg(feature = "cuda")]
pub(super) fn maxsim_cuda_chunk_tokens() -> usize {
    env::var("CALYX_SEARCH_MAXSIM_CUDA_CHUNK_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_MAXSIM_CUDA_CHUNK_TOKENS)
}

pub(super) fn maxsim_cuda_strict() -> bool {
    env_truthy("CALYX_SEARCH_MAXSIM_CUDA_STRICT")
}

pub(super) fn maxsim_cuda_disabled() -> bool {
    env::var("CALYX_SEARCH_MAXSIM_CUDA")
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "0" | "false" | "FALSE" | "off" | "OFF"))
}

fn env_truthy(name: &str) -> bool {
    env::var(name).ok().is_some_and(|value| {
        matches!(
            value.as_str(),
            "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
        )
    })
}
