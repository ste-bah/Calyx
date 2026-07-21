use std::sync::Arc;

use calyx_core::{Input, Lens, LensId, Modality, Result, SlotShape, SlotVector};

use crate::frozen::{FrozenLensContract, LensDType, NormPolicy, sha256_digest};
use crate::lens::ensure_input_modality;

mod batch;
#[cfg(all(test, feature = "cuda"))]
mod benchmark;
mod cpu;
mod gdelt;
mod stylometry;

pub use batch::{
    AlgorithmicBatchProvider, AlgorithmicBatchStats, BYTE_FEATURES_CUDA_MIN_INPUT_BYTES,
    SPARSE_KEYWORDS_CUDA_MIN_TOKENS, TOKEN_HASH_CUDA_MIN_WORDS,
};
use cpu::{
    ast_style_features, byte_features, hash_part, one_hot_features, scalar_features,
    sparse_keywords, token_hash,
};

const BYTE_FEATURE_DIM: u32 = 16;

/// Deterministic, data-local feature encoders with no model weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlgorithmicEncoder {
    /// Byte and character-class features for text/code/structured inputs.
    ByteFeatures,
    /// Single scalar summary.
    Scalar,
    /// Hash-selected one-hot feature vector.
    OneHot { buckets: u32 },
    /// Small AST/code-style feature vector.
    AstStyle,
    /// Prose pacing statistics (sentence rhythm, dialogue share, figurative
    /// and abstract density, function-word profile) z-scored against frozen
    /// English-narrative reference statistics.
    Stylometry,
    /// Hashed whitespace terms in a sparse ambient space.
    SparseKeywords { dim: u32 },
    /// Hashed whitespace terms as per-token vectors for MaxSim.
    TokenHash { token_dim: u32 },
    /// Dense CAMEO/event-code features from GDELT text rows.
    GdeltCameo,
    /// Sparse actor/country/geography entity features from GDELT text rows.
    GdeltActorGeo { dim: u32 },
    /// Sparse source URL host/path features from GDELT text rows.
    GdeltSourceDomain { dim: u32 },
    /// Sparse event-code/geography interaction features from GDELT text rows.
    GdeltEventGeo { dim: u32 },
    /// Sparse directed actor-pair features from GDELT text rows.
    GdeltActorPair { dim: u32 },
    /// Sparse event-code/actor interaction features from GDELT text rows.
    GdeltEventActor { dim: u32 },
    /// Sparse Goldstein/tone bucket features from GDELT text rows.
    GdeltToneSignal { dim: u32 },
    /// Sparse source-domain/event-code interaction features from GDELT text rows.
    GdeltSourceEvent { dim: u32 },
    /// Sparse action-geo-only features from GDELT text rows.
    GdeltActionGeo { dim: u32 },
    /// Sparse actor-country/presence-only features from GDELT text rows.
    GdeltActorCountry { dim: u32 },
    /// Sparse source-host-only features from GDELT text rows.
    GdeltSourceHost { dim: u32 },
    /// Sparse SQLDATE/date bucket features from GDELT text rows.
    GdeltSqlDate { dim: u32 },
    /// Sparse event-code-only features from GDELT text rows.
    GdeltEventCode { dim: u32 },
}

impl AlgorithmicEncoder {
    /// Returns the primary output dimension.
    pub const fn dim(self) -> u32 {
        match self {
            Self::ByteFeatures => BYTE_FEATURE_DIM,
            Self::Scalar => 1,
            Self::OneHot { buckets } => {
                if buckets == 0 {
                    1
                } else {
                    buckets
                }
            }
            Self::AstStyle => 8,
            Self::Stylometry => stylometry::STYLO_DIM,
            Self::SparseKeywords { dim }
            | Self::GdeltActorGeo { dim }
            | Self::GdeltSourceDomain { dim }
            | Self::GdeltEventGeo { dim }
            | Self::GdeltActorPair { dim }
            | Self::GdeltEventActor { dim }
            | Self::GdeltToneSignal { dim }
            | Self::GdeltSourceEvent { dim }
            | Self::GdeltActionGeo { dim }
            | Self::GdeltActorCountry { dim }
            | Self::GdeltSourceHost { dim }
            | Self::GdeltSqlDate { dim }
            | Self::GdeltEventCode { dim } => {
                if dim == 0 {
                    1
                } else {
                    dim
                }
            }
            Self::TokenHash { token_dim } => {
                if token_dim == 0 {
                    1
                } else {
                    token_dim
                }
            }
            Self::GdeltCameo => 16,
        }
    }

    pub const fn shape(self) -> SlotShape {
        match self {
            Self::SparseKeywords { dim }
            | Self::GdeltActorGeo { dim }
            | Self::GdeltSourceDomain { dim }
            | Self::GdeltEventGeo { dim }
            | Self::GdeltActorPair { dim }
            | Self::GdeltEventActor { dim }
            | Self::GdeltToneSignal { dim }
            | Self::GdeltSourceEvent { dim }
            | Self::GdeltActionGeo { dim }
            | Self::GdeltActorCountry { dim }
            | Self::GdeltSourceHost { dim }
            | Self::GdeltSqlDate { dim }
            | Self::GdeltEventCode { dim } => SlotShape::Sparse(if dim == 0 { 1 } else { dim }),
            Self::TokenHash { token_dim } => SlotShape::Multi {
                token_dim: if token_dim == 0 { 1 } else { token_dim },
            },
            _ => SlotShape::Dense(self.dim()),
        }
    }
}

/// A frozen algorithmic lens.
#[derive(Clone, Debug)]
pub struct AlgorithmicLens {
    id: LensId,
    modality: Modality,
    encoder: AlgorithmicEncoder,
    contract: FrozenLensContract,
    batch: Arc<batch::BatchState>,
}

impl AlgorithmicLens {
    /// Creates an algorithmic byte-feature lens.
    pub fn byte_features(name: impl Into<String>, modality: Modality) -> Self {
        Self::new(name, modality, AlgorithmicEncoder::ByteFeatures)
    }

    pub fn scalar(name: impl Into<String>, modality: Modality) -> Self {
        Self::new(name, modality, AlgorithmicEncoder::Scalar)
    }

    pub fn one_hot(name: impl Into<String>, modality: Modality, buckets: u32) -> Self {
        Self::new(name, modality, AlgorithmicEncoder::OneHot { buckets })
    }

    pub fn ast_style(name: impl Into<String>, modality: Modality) -> Self {
        Self::new(name, modality, AlgorithmicEncoder::AstStyle)
    }

    pub fn stylometry(name: impl Into<String>, modality: Modality) -> Self {
        Self::new(name, modality, AlgorithmicEncoder::Stylometry)
    }

    pub fn sparse_keywords(name: impl Into<String>, modality: Modality, dim: u32) -> Self {
        Self::new(name, modality, AlgorithmicEncoder::SparseKeywords { dim })
    }

    pub fn token_hash(name: impl Into<String>, modality: Modality, token_dim: u32) -> Self {
        Self::new(name, modality, AlgorithmicEncoder::TokenHash { token_dim })
    }

    pub fn gdelt_cameo(name: impl Into<String>, modality: Modality) -> Self {
        Self::new(name, modality, AlgorithmicEncoder::GdeltCameo)
    }

    pub fn gdelt_actor_geo(name: impl Into<String>, modality: Modality, dim: u32) -> Self {
        Self::new(name, modality, AlgorithmicEncoder::GdeltActorGeo { dim })
    }

    pub fn gdelt_source_domain(name: impl Into<String>, modality: Modality, dim: u32) -> Self {
        Self::new(
            name,
            modality,
            AlgorithmicEncoder::GdeltSourceDomain { dim },
        )
    }

    pub fn gdelt_event_geo(name: impl Into<String>, modality: Modality, dim: u32) -> Self {
        Self::new(name, modality, AlgorithmicEncoder::GdeltEventGeo { dim })
    }

    pub fn gdelt_actor_pair(name: impl Into<String>, modality: Modality, dim: u32) -> Self {
        Self::new(name, modality, AlgorithmicEncoder::GdeltActorPair { dim })
    }

    pub fn gdelt_event_actor(name: impl Into<String>, modality: Modality, dim: u32) -> Self {
        Self::new(name, modality, AlgorithmicEncoder::GdeltEventActor { dim })
    }

    pub fn gdelt_tone_signal(name: impl Into<String>, modality: Modality, dim: u32) -> Self {
        Self::new(name, modality, AlgorithmicEncoder::GdeltToneSignal { dim })
    }

    pub fn gdelt_source_event(name: impl Into<String>, modality: Modality, dim: u32) -> Self {
        Self::new(name, modality, AlgorithmicEncoder::GdeltSourceEvent { dim })
    }

    pub fn gdelt_action_geo(name: impl Into<String>, modality: Modality, dim: u32) -> Self {
        Self::new(name, modality, AlgorithmicEncoder::GdeltActionGeo { dim })
    }

    pub fn gdelt_actor_country(name: impl Into<String>, modality: Modality, dim: u32) -> Self {
        Self::new(
            name,
            modality,
            AlgorithmicEncoder::GdeltActorCountry { dim },
        )
    }

    pub fn gdelt_source_host(name: impl Into<String>, modality: Modality, dim: u32) -> Self {
        Self::new(name, modality, AlgorithmicEncoder::GdeltSourceHost { dim })
    }

    pub fn gdelt_sql_date(name: impl Into<String>, modality: Modality, dim: u32) -> Self {
        Self::new(name, modality, AlgorithmicEncoder::GdeltSqlDate { dim })
    }

    pub fn gdelt_event_code(name: impl Into<String>, modality: Modality, dim: u32) -> Self {
        Self::new(name, modality, AlgorithmicEncoder::GdeltEventCode { dim })
    }

    /// Creates an algorithmic lens from an encoder.
    pub fn new(name: impl Into<String>, modality: Modality, encoder: AlgorithmicEncoder) -> Self {
        let name = name.into();
        let contract = algorithmic_contract(&name, modality, encoder);
        let id = contract.lens_id();
        Self {
            id,
            modality,
            encoder,
            contract,
            batch: Arc::new(batch::BatchState::default()),
        }
    }

    /// Returns the frozen contract that produced this lens id.
    pub fn contract(&self) -> &FrozenLensContract {
        &self.contract
    }

    /// Returns the most recent serializable batch provider/transfer evidence.
    pub fn last_batch_stats(&self) -> Option<AlgorithmicBatchStats> {
        self.batch.last_stats()
    }

    fn measure_cpu(&self, input: &Input) -> Result<SlotVector> {
        Ok(match self.encoder {
            AlgorithmicEncoder::ByteFeatures => SlotVector::Dense {
                dim: self.encoder.dim(),
                data: byte_features(&input.bytes),
            },
            AlgorithmicEncoder::Scalar => SlotVector::Dense {
                dim: self.encoder.dim(),
                data: scalar_features(&input.bytes),
            },
            AlgorithmicEncoder::OneHot { buckets } => SlotVector::Dense {
                dim: self.encoder.dim(),
                data: one_hot_features(&input.bytes, buckets),
            },
            AlgorithmicEncoder::AstStyle => SlotVector::Dense {
                dim: self.encoder.dim(),
                data: ast_style_features(&input.bytes),
            },
            AlgorithmicEncoder::Stylometry => SlotVector::Dense {
                dim: self.encoder.dim(),
                data: stylometry::stylometry_features(&input.bytes),
            },
            AlgorithmicEncoder::SparseKeywords { dim } => sparse_keywords(&input.bytes, dim)?,
            AlgorithmicEncoder::TokenHash { token_dim } => token_hash(&input.bytes, token_dim)?,
            AlgorithmicEncoder::GdeltCameo => SlotVector::Dense {
                dim: self.encoder.dim(),
                data: gdelt::cameo_features(&input.bytes),
            },
            AlgorithmicEncoder::GdeltActorGeo { dim } => gdelt::actor_geo(&input.bytes, dim)?,
            AlgorithmicEncoder::GdeltSourceDomain { dim } => {
                gdelt::source_domain(&input.bytes, dim)?
            }
            AlgorithmicEncoder::GdeltEventGeo { dim } => gdelt::event_geo(&input.bytes, dim)?,
            AlgorithmicEncoder::GdeltActorPair { dim } => gdelt::actor_pair(&input.bytes, dim)?,
            AlgorithmicEncoder::GdeltEventActor { dim } => gdelt::event_actor(&input.bytes, dim)?,
            AlgorithmicEncoder::GdeltToneSignal { dim } => gdelt::tone_signal(&input.bytes, dim)?,
            AlgorithmicEncoder::GdeltSourceEvent { dim } => gdelt::source_event(&input.bytes, dim)?,
            AlgorithmicEncoder::GdeltActionGeo { dim } => gdelt::action_geo(&input.bytes, dim)?,
            AlgorithmicEncoder::GdeltActorCountry { dim } => {
                gdelt::actor_country(&input.bytes, dim)?
            }
            AlgorithmicEncoder::GdeltSourceHost { dim } => {
                gdelt::source_host_lens(&input.bytes, dim)?
            }
            AlgorithmicEncoder::GdeltSqlDate { dim } => gdelt::sql_date(&input.bytes, dim)?,
            AlgorithmicEncoder::GdeltEventCode { dim } => gdelt::event_code(&input.bytes, dim)?,
        })
    }
}

impl Lens for AlgorithmicLens {
    fn id(&self) -> LensId {
        self.id
    }

    fn shape(&self) -> SlotShape {
        self.encoder.shape()
    }

    fn modality(&self) -> Modality {
        self.modality
    }

    fn measure(&self, input: &Input) -> Result<SlotVector> {
        ensure_input_modality(self, input)?;
        let output = self.measure_cpu(input)?;
        self.batch.record(batch::cpu_stats(
            self.encoder,
            std::slice::from_ref(input),
            1,
            "single-input CPU path",
        ));
        Ok(output)
    }

    fn measure_batch(&self, inputs: &[Input]) -> Result<Vec<SlotVector>> {
        batch::measure_batch(self, inputs)
    }
}

fn algorithmic_contract(
    name: &str,
    modality: Modality,
    encoder: AlgorithmicEncoder,
) -> FrozenLensContract {
    if encoder == AlgorithmicEncoder::ByteFeatures {
        return FrozenLensContract::algorithmic_byte_features(name, modality);
    }
    let encoder_text = format!("{encoder:?}:{}", encoder.dim());
    FrozenLensContract::new(
        name,
        sha256_digest(&[b"algorithmic-runtime-v2", encoder_text.as_bytes()]),
        sha256_digest(&[b"algorithmic-data-oblivious"]),
        encoder.shape(),
        modality,
        LensDType::F32,
        NormPolicy::None,
    )
}

#[cfg(test)]
mod tests;
