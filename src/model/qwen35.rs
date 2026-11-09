//! Packed Qwen3.5 text graph with static CPU/Metal projection execution.
//! No codec or ordinary model route.
//!
//! Model weights are immutable borrowed spans. Delta/history and attention KV
//! have distinct session owners. This is a correctness reference, not BF16 GPU
//! arithmetic emulation or a declaration of real-source quality qualification.
use crate::{
    cpu,
    mlx::affine::{Matrix, MetadataDtype},
};
use std::fmt;

mod binding;
mod execution;
mod forward;
mod generation;
pub use execution::{Cpu, Execution, MatrixStorage};
pub use generation::{GenerationMemory, GenerationSession, GenerationState, PrefixAction};
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use metal::{MetalExecution, MetalMatrix, MetalModel, MetalSession};
mod state;
pub use state::{Memory, Options, Outcome, Session, Work};
#[cfg(test)]
mod tests;

pub const CAPACITY_LIMIT: usize = 512;
pub const CHUNK_LIMIT: usize = 32;
const MEMORY_LIMIT: usize = 6 * 1024 * 1024 * 1024;
const METADATA_ALLOWANCE: usize = 64 * 1024 * 1024;

#[derive(Debug)]
pub enum Error {
    Model(super::Error),
    Weights(crate::weights::Error),
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(crate::metal::Error),
    #[cfg(all(feature = "metal", target_os = "macos"))]
    ExecutionPoisoned,
    MemoryBudget {
        required: usize,
        limit: usize,
    },
    Cancelled,
    Observation(&'static str),
}
impl From<cpu::Error> for Error {
    fn from(e: cpu::Error) -> Self {
        Self::Model(e.into())
    }
}
impl From<super::Error> for Error {
    fn from(e: super::Error) -> Self {
        Self::Model(e)
    }
}
impl From<crate::weights::Error> for Error {
    fn from(e: crate::weights::Error) -> Self {
        Self::Weights(e)
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(e) => e.fmt(f),
            Self::Weights(e) => e.fmt(f),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(e) => e.fmt(f),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::ExecutionPoisoned => f.write_str(
                "Qwen3.5 GPU workspace requires explicit repair; committed prefix remains valid",
            ),
            Self::MemoryBudget { required, limit } => {
                write!(
                    f,
                    "Qwen3.5 admission requires {required} bytes, limit {limit}"
                )
            }
            Self::Cancelled => f.write_str("Qwen3.5 chunk cancelled before publication"),
            Self::Observation(reason) => write!(f, "Qwen3.5 observation rejected: {reason}"),
        }
    }
}
impl std::error::Error for Error {}
type Result<T> = std::result::Result<T, Error>;
fn invalid(message: &'static str) -> Error {
    super::Error::InvalidConfig(message).into()
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or(cpu::Error::SizeOverflow.into())
}
fn mul(a: usize, b: usize) -> Result<usize> {
    Ok(cpu::product(a, b)?)
}
fn reserved<T>(n: usize) -> Result<Vec<T>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(n)
        .map_err(|_| super::Error::Allocation)?;
    Ok(result)
}
fn zeros(n: usize) -> Result<Vec<f32>> {
    Ok(super::zeros(n)?)
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Config {
    pub hidden: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub layers: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotated_dims: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub max_positions: usize,
    pub rope_theta: f32,
    pub rms_epsilon: f32,
}
impl Config {
    fn validate(self) -> Result<()> {
        for n in [
            self.hidden,
            self.intermediate,
            self.vocab,
            self.layers,
            self.query_heads,
            self.kv_heads,
            self.head_dim,
            self.key_heads,
            self.value_heads,
            self.key_dim,
            self.value_dim,
            self.max_positions,
        ] {
            cpu::nonzero(n, "Qwen3.5 geometry")?;
        }
        if self.layers > 32
            || !self.layers.is_multiple_of(4)
            || !self.query_heads.is_multiple_of(self.kv_heads)
            || !self.value_heads.is_multiple_of(self.key_heads)
            || self.rotated_dims == 0
            || !self.rotated_dims.is_multiple_of(2)
            || self.rotated_dims > self.head_dim
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.
            || !self.rms_epsilon.is_finite()
            || self.rms_epsilon <= 0.
            || self.max_positions as u128 > 1u128 << 53
        {
            return Err(invalid("Qwen3.5 text geometry/position convention"));
        }
        self.query_width()?;
        self.kv_width()?;
        self.key_width()?;
        self.value_width()?;
        self.conv_width()?;
        Ok(())
    }
    fn query_width(self) -> Result<usize> {
        mul(self.query_heads, self.head_dim)
    }
    fn kv_width(self) -> Result<usize> {
        mul(self.kv_heads, self.head_dim)
    }
    fn key_width(self) -> Result<usize> {
        mul(self.key_heads, self.key_dim)
    }
    fn value_width(self) -> Result<usize> {
        mul(self.value_heads, self.value_dim)
    }
    fn conv_width(self) -> Result<usize> {
        add(mul(2, self.key_width()?)?, self.value_width()?)
    }
    fn delta_count(self) -> usize {
        self.layers - self.layers / 4
    }
    fn attention_count(self) -> usize {
        self.layers / 4
    }
}

pub struct DeltaWeights<'a, M = Matrix<'a>> {
    pub qkv: M,
    pub z: M,
    pub a: M,
    pub b: M,
    pub out: M,
    pub convolution: &'a [f32],
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub norm: &'a [f32],
}
pub struct AttentionWeights<'a, M = Matrix<'a>> {
    pub q_gate: M,
    pub k: M,
    pub v: M,
    pub out: M,
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
}
pub enum MixingWeights<'a, M = Matrix<'a>> {
    Delta(DeltaWeights<'a, M>),
    Attention(AttentionWeights<'a, M>),
}
pub struct LayerWeights<'a, M = Matrix<'a>> {
    pub input_norm: &'a [f32],
    pub post_norm: &'a [f32],
    pub gate: M,
    pub up: M,
    pub down: M,
    pub mixing: MixingWeights<'a, M>,
}
pub struct Weights<'a, M = Matrix<'a>> {
    pub embedding: M,
    pub final_norm: &'a [f32],
    pub layers: Vec<LayerWeights<'a, M>>,
}
struct Layer<'a, M> {
    weights: LayerWeights<'a, M>,
    owner: usize,
}
pub struct Model<'a, M: MatrixStorage = Matrix<'a>> {
    config: Config,
    embedding: M,
    output_head: Option<M>,
    final_norm: &'a [f32],
    layers: Vec<Layer<'a, M>>,
    learned_bytes: usize,
    owned_bytes: usize,
}
fn matrix(m: &impl MatrixStorage, rows: usize, columns: usize) -> Result<usize> {
    let l = m.layout();
    if l.rows() != rows
        || l.columns() != columns
        || l.bits() != 4
        || l.group_size() != 64
        || l.metadata_dtype() != MetadataDtype::BF16
    {
        return Err(invalid("Qwen3.5 original Q4/G64/BF16 matrix geometry"));
    }
    Ok(l.total_bytes())
}
fn vector(v: &[f32], n: usize) -> Result<usize> {
    cpu::length("Qwen3.5 vector", v.len(), n)?;
    cpu::finite("Qwen3.5 vector", v)?;
    mul(n, 4)
}
impl<'a, M: MatrixStorage> Model<'a, M> {
    /// Borrow admitted numerical spans. Any caller ownership beyond those spans
    /// needs a separate allowance. The CPU-store factory accounts its capacities.
    pub fn new(config: Config, weights: Weights<'a, M>) -> Result<Self> {
        Self::new_with_output(config, weights, None)
    }
    /// A separate original output projection, with the same vocabulary/hidden
    /// dimensions as the embedding. Use `new` for tied storage, counted once.
    /// The source-qualified9B factory resolves its independently owned table.
    pub fn new_untied(config: Config, weights: Weights<'a, M>, head: M) -> Result<Self> {
        Self::new_with_output(config, weights, Some(head))
    }
    fn new_with_output(
        config: Config,
        weights: Weights<'a, M>,
        output_head: Option<M>,
    ) -> Result<Self> {
        let c = config;
        c.validate()?;
        if weights.layers.len() != c.layers {
            return Err(super::Error::LayerCount {
                expected: c.layers,
                actual: weights.layers.len(),
            }
            .into());
        }
        let mut learned = add(
            matrix(&weights.embedding, c.vocab, c.hidden)?,
            vector(weights.final_norm, c.hidden)?,
        )?;
        if let Some(head) = &output_head {
            learned = add(learned, matrix(head, c.vocab, c.hidden)?)?;
        }
        let mut layers = reserved(c.layers)?;
        for (i, w) in weights.layers.into_iter().enumerate() {
            for n in [
                vector(w.input_norm, c.hidden)?,
                vector(w.post_norm, c.hidden)?,
                matrix(&w.gate, c.intermediate, c.hidden)?,
                matrix(&w.up, c.intermediate, c.hidden)?,
                matrix(&w.down, c.hidden, c.intermediate)?,
            ] {
                learned = add(learned, n)?;
            }
            let owner = match &w.mixing {
                MixingWeights::Delta(d) if i % 4 != 3 => {
                    for n in [
                        matrix(&d.qkv, c.conv_width()?, c.hidden)?,
                        matrix(&d.z, c.value_width()?, c.hidden)?,
                        matrix(&d.a, c.value_heads, c.hidden)?,
                        matrix(&d.b, c.value_heads, c.hidden)?,
                        matrix(&d.out, c.hidden, c.value_width()?)?,
                        vector(d.convolution, mul(c.conv_width()?, 4)?)?,
                        vector(d.a_log, c.value_heads)?,
                        vector(d.dt_bias, c.value_heads)?,
                        vector(d.norm, c.value_dim)?,
                    ] {
                        learned = add(learned, n)?;
                    }
                    i - i / 4
                }
                MixingWeights::Attention(a) if i % 4 == 3 => {
                    for n in [
                        matrix(&a.q_gate, mul(2, c.query_width()?)?, c.hidden)?,
                        matrix(&a.k, c.kv_width()?, c.hidden)?,
                        matrix(&a.v, c.kv_width()?, c.hidden)?,
                        matrix(&a.out, c.hidden, c.query_width()?)?,
                        vector(a.q_norm, c.head_dim)?,
                        vector(a.k_norm, c.head_dim)?,
                    ] {
                        learned = add(learned, n)?;
                    }
                    i / 4
                }
                _ => return Err(invalid("Qwen3.5 three-delta/one-attention layer order")),
            };
            layers.push(Layer { weights: w, owner });
        }
        let owned_bytes = add(
            std::mem::size_of::<Self>(),
            mul(layers.capacity(), std::mem::size_of::<Layer<'_, M>>())?,
        )?;
        Ok(Self {
            config: c,
            embedding: weights.embedding,
            output_head,
            final_norm: weights.final_norm,
            layers,
            learned_bytes: learned,
            owned_bytes,
        })
    }
    fn head(&self) -> &M {
        self.output_head.as_ref().unwrap_or(&self.embedding)
    }
    pub fn config(&self) -> Config {
        self.config
    }
    pub fn model_bytes(&self) -> Result<usize> {
        add(self.learned_bytes, self.owned_bytes)
    }
}

impl<'a> Model<'a> {
    pub fn session(&self, options: Options) -> Result<Session<'_, 'a>> {
        Session::new(self, options)
    }
    /// Authenticated scalar storage; no tokenizer or ordinary managed route.
    pub fn from_cpu_store(store: &'a crate::mlx::qwen35::payload::PayloadStore) -> Result<Self> {
        Self::bind_source(
            store.plan(),
            |name| Ok(store.matrix(name)?),
            |name| Ok(store.vector(name)?),
            |name| Ok(store.convolution(name)?),
            store.report().retained_tensor_capacity_bytes,
        )
    }
}

/// Actual candidate-stage slices. `head` is used for strided cache/attention
/// rows; no observer runs after the first publication swap.
pub struct Observation<'a> {
    pub token: usize,
    pub position: usize,
    pub layer: Option<usize>,
    pub stage: &'static str,
    pub head: Option<usize>,
    pub values: &'a [f32],
}
