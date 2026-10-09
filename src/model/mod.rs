//! Immutable FP32 decoder weights and scalar full-prefix/cached execution.
//!
//! Numerical execution uses preallocated caller-owned scratch. Diagnostic
//! traces allocate separately; tokenization and generation state live elsewhere.
//! Projection products/reductions, normalization and softmax use FP64 arithmetic
//! with FP32 storage; rotary combination, activations and residuals remain FP32.

use crate::cpu::{self, AttentionShape, finite, length, nonzero, product};
use std::fmt;

mod cache;
mod trace;
pub(crate) use cache::KvCache;
pub use trace::{LayerTrace, Trace};

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub max_positions: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

impl Config {
    pub fn validate(&self) -> Result<(), Error> {
        for (name, value) in [
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_layers", self.num_layers),
            ("query_heads", self.query_heads),
            ("kv_heads", self.kv_heads),
            ("head_dim", self.head_dim),
            ("max_positions", self.max_positions),
        ] {
            nonzero(value, name)?;
        }
        if self.hidden_size != product(self.query_heads, self.head_dim)?
            || !self.query_heads.is_multiple_of(self.kv_heads)
            || !self.head_dim.is_multiple_of(2)
        {
            return Err(Error::InvalidConfig(
                "hidden/head dimensions or GQA mapping",
            ));
        }
        if !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
        {
            return Err(Error::InvalidConfig(
                "epsilon and theta must be finite and positive",
            ));
        }
        if self.vocab_size - 1 > u32::MAX as usize || self.max_positions as u128 > 1_u128 << 53 {
            return Err(Error::InvalidConfig(
                "token IDs or positions exceed supported range",
            ));
        }
        let h = self.hidden_size;
        let kv = product(self.kv_heads, self.head_dim)?;
        for count in [
            product(self.vocab_size, h)?,
            product(h, h)?,
            product(kv, h)?,
            product(self.intermediate_size, h)?,
        ] {
            allocation_bytes(count)?;
        }
        Ok(())
    }
}

/// Q/K/V row-major projection with required Qwen bias.
pub struct Projection {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

pub struct LayerWeights {
    pub input_norm: Vec<f32>,
    pub post_attention_norm: Vec<f32>,
    pub q: Projection,
    pub k: Projection,
    pub v: Projection,
    pub o: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
}

/// The embedding table is also the output projection; no duplicate is retained.
pub struct Weights {
    pub embedding: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub final_norm: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Kernel(cpu::Error),
    InvalidConfig(&'static str),
    LayerCount {
        expected: usize,
        actual: usize,
    },
    Capacity {
        offset: usize,
        tokens: usize,
        limit: usize,
    },
    ScratchCapacity {
        tokens: usize,
        limit: usize,
    },
    ScratchMismatch,
    Allocation,
}

impl From<cpu::Error> for Error {
    fn from(error: cpu::Error) -> Self {
        Self::Kernel(error)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Kernel(error) => error.fmt(f),
            Self::InvalidConfig(reason) => write!(f, "unsupported decoder configuration: {reason}"),
            Self::LayerCount { expected, actual } => {
                write!(f, "expected {expected} layers, got {actual}")
            }
            Self::Capacity {
                offset,
                tokens,
                limit,
            } => write!(
                f,
                "{tokens} tokens at offset {offset} exceed capacity {limit}"
            ),
            Self::ScratchCapacity { tokens, limit } => {
                write!(f, "{tokens} tokens exceed scratch capacity {limit}")
            }
            Self::ScratchMismatch => {
                write!(f, "scratch belongs to a different decoder configuration")
            }
            Self::Allocation => write!(f, "unable to allocate decoder buffers"),
        }
    }
}
impl std::error::Error for Error {}

pub(super) fn allocation_bytes(count: usize) -> Result<usize, Error> {
    let bytes = product(count, std::mem::size_of::<f32>())?;
    if bytes > isize::MAX as usize {
        return Err(cpu::Error::SizeOverflow.into());
    }
    Ok(bytes)
}

pub(crate) fn total_bytes(counts: &[usize]) -> Result<usize, Error> {
    counts.iter().try_fold(0_usize, |total, &count| {
        total
            .checked_add(allocation_bytes(count)?)
            .ok_or(cpu::Error::SizeOverflow.into())
    })
}

pub(crate) fn zeros(count: usize) -> Result<Vec<f32>, Error> {
    allocation_bytes(count)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| Error::Allocation)?;
    values.resize(count, 0.0);
    Ok(values)
}

pub struct Model {
    pub(crate) config: Config,
    pub(crate) weights: Weights,
}

impl Model {
    /// Validate and take ownership of weights once, without copying them.
    pub fn new(config: Config, weights: Weights) -> Result<Self, Error> {
        config.validate()?;
        if weights.layers.len() != config.num_layers {
            return Err(Error::LayerCount {
                expected: config.num_layers,
                actual: weights.layers.len(),
            });
        }
        let check = |name, values: &[f32], count| -> Result<(), Error> {
            length(name, values.len(), count)?;
            finite(name, values)?;
            Ok(())
        };
        let h = config.hidden_size;
        let kv = product(config.kv_heads, config.head_dim)?;
        check(
            "embedding",
            &weights.embedding,
            product(config.vocab_size, h)?,
        )?;
        check("final norm", &weights.final_norm, h)?;
        for layer in &weights.layers {
            check("input norm", &layer.input_norm, h)?;
            check("post attention norm", &layer.post_attention_norm, h)?;
            for (projection, rows) in [(&layer.q, h), (&layer.k, kv), (&layer.v, kv)] {
                check("QKV weight", &projection.weight, product(rows, h)?)?;
                check("QKV bias", &projection.bias, rows)?;
            }
            check("o weight", &layer.o, product(h, h)?)?;
            for values in [&layer.gate, &layer.up, &layer.down] {
                check("MLP weight", values, product(config.intermediate_size, h)?)?;
            }
        }
        Ok(Self { config, weights })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Owned numerical weight allocations, counted once when a session borrows
    /// this model. Cold Rust metadata is covered by the session admission allowance.
    pub fn allocated_weights_bytes(&self) -> Result<usize, Error> {
        let mut bytes = total_bytes(&[
            self.weights.embedding.capacity(),
            self.weights.final_norm.capacity(),
        ])?;
        for layer in &self.weights.layers {
            bytes = bytes
                .checked_add(total_bytes(&[
                    layer.input_norm.capacity(),
                    layer.post_attention_norm.capacity(),
                    layer.q.weight.capacity(),
                    layer.q.bias.capacity(),
                    layer.k.weight.capacity(),
                    layer.k.bias.capacity(),
                    layer.v.weight.capacity(),
                    layer.v.bias.capacity(),
                    layer.o.capacity(),
                    layer.gate.capacity(),
                    layer.up.capacity(),
                    layer.down.capacity(),
                ])?)
                .ok_or(cpu::Error::SizeOverflow)?;
        }
        Ok(bytes)
    }

    pub(crate) fn allocated_metadata_bytes(&self) -> Result<usize, Error> {
        product(
            self.weights.layers.capacity(),
            std::mem::size_of::<LayerWeights>(),
        )
        .map_err(Into::into)
    }

    pub fn scratch(&self, capacity: usize) -> Result<Scratch, Error> {
        Scratch::new(&self.config, capacity)
    }

    /// Recompute the full prefix and return only its final distribution.
    ///
    /// Logits must be exactly `[vocab_size]`. Shape/token/capacity checks precede
    /// any scratch mutation. On arithmetic error discard output and scratch;
    /// the immutable model stays valid and a later call recomputes from scratch.
    /// Steady-state numerical execution allocates no buffers.
    pub fn forward(
        &self,
        tokens: &[u32],
        offset: usize,
        scratch: &mut Scratch,
        logits: &mut [f32],
    ) -> Result<(), Error> {
        self.admit(tokens, offset, scratch)?;
        length("logits", logits.len(), self.config.vocab_size)?;
        self.run(tokens, offset, scratch, None, None, 0)?;
        let start = (tokens.len() - 1) * self.config.hidden_size;
        cpu::matvec_f64(
            &scratch.norm[start..start + self.config.hidden_size],
            &self.weights.embedding,
            None,
            logits,
            self.config.hidden_size,
            self.config.vocab_size,
        )?;
        Ok(())
    }

    /// Allocate an explicit full intermediate trace and all-position logits.
    pub fn forward_diagnostic(
        &self,
        tokens: &[u32],
        offset: usize,
        scratch: &mut Scratch,
    ) -> Result<Trace, Error> {
        self.admit(tokens, offset, scratch)?;
        let mut trace = Trace::new(&self.config, tokens.len())?;
        self.run(tokens, offset, scratch, Some(&mut trace), None, 0)?;
        cpu::matmul_f64(
            &scratch.norm[..tokens.len() * self.config.hidden_size],
            &self.weights.embedding,
            None,
            &mut trace.logits,
            tokens.len(),
            self.config.hidden_size,
            self.config.vocab_size,
        )?;
        Ok(trace)
    }

    /// Private session path: stage a chunk at cached_len; the caller commits its
    /// length/logits only after this returns successfully. Scratch/cache originate
    /// from the same borrowed model and cannot be replaced through the public API.
    pub(crate) fn forward_cached(
        &self,
        tokens: &[u32],
        rotary_offset: usize,
        scratch: &mut Scratch,
        cache: &mut KvCache,
        cached_len: usize,
        logits: &mut [f32],
    ) -> Result<(), Error> {
        self.admit(tokens, rotary_offset, scratch)?;
        length("logits", logits.len(), self.config.vocab_size)?;
        let end = cached_len
            .checked_add(tokens.len())
            .ok_or(cpu::Error::SizeOverflow)?;
        if end > cache.capacity {
            return Err(Error::Capacity {
                offset: cached_len,
                tokens: tokens.len(),
                limit: cache.capacity,
            });
        }
        self.run(
            tokens,
            rotary_offset,
            scratch,
            None,
            Some(cache),
            cached_len,
        )?;
        let start = (tokens.len() - 1) * self.config.hidden_size;
        cpu::matvec_f64(
            &scratch.norm[start..start + self.config.hidden_size],
            &self.weights.embedding,
            None,
            logits,
            self.config.hidden_size,
            self.config.vocab_size,
        )?;
        Ok(())
    }

    fn admit(&self, tokens: &[u32], offset: usize, scratch: &Scratch) -> Result<(), Error> {
        nonzero(tokens.len(), "tokens")?;
        if scratch.config != self.config {
            return Err(Error::ScratchMismatch);
        }
        let end = offset
            .checked_add(tokens.len())
            .ok_or(cpu::Error::SizeOverflow)?;
        if end > self.config.max_positions {
            return Err(Error::Capacity {
                offset,
                tokens: tokens.len(),
                limit: self.config.max_positions,
            });
        }
        if tokens.len() > scratch.capacity {
            return Err(Error::ScratchCapacity {
                tokens: tokens.len(),
                limit: scratch.capacity,
            });
        }
        for (position, &token) in tokens.iter().enumerate() {
            if usize::try_from(token).map_or(true, |id| id >= self.config.vocab_size) {
                return Err(cpu::Error::Token {
                    position,
                    token,
                    vocab_size: self.config.vocab_size,
                }
                .into());
            }
        }
        Ok(())
    }

    fn run(
        &self,
        tokens: &[u32],
        offset: usize,
        s: &mut Scratch,
        mut trace: Option<&mut Trace>,
        mut cache: Option<&mut KvCache>,
        cached_len: usize,
    ) -> Result<(), Error> {
        let c = &self.config;
        let n = tokens.len();
        let nh = n * c.hidden_size;
        let nkv = n * c.kv_heads * c.head_dim;
        let nm = n * c.intermediate_size;
        let ns = match &cache {
            Some(cache) => cache.capacity,
            None => c.query_heads * n * n,
        };
        let activation = &mut s.activation[..nh];
        let norm = &mut s.norm[..nh];
        let q = &mut s.q[..nh];
        let k = &mut s.k[..nkv];
        let v = &mut s.v[..nkv];
        let q_rot = &mut s.q_rot[..nh];
        let k_rot = &mut s.k_rot[..nkv];
        let context = &mut s.context[..nh];
        let projection = &mut s.projection[..nh];
        let gate = &mut s.gate[..nm];
        let up = &mut s.up[..nm];
        let gated = &mut s.gated[..nm];
        let scores = &mut s.scores[..ns];
        let probabilities = &mut s.probabilities[..ns];
        cpu::embedding(
            tokens,
            &self.weights.embedding,
            activation,
            c.vocab_size,
            c.hidden_size,
        )?;
        if let Some(trace) = trace.as_deref_mut() {
            trace.embedding.copy_from_slice(activation);
        }
        for (index, layer) in self.weights.layers.iter().enumerate() {
            let mut t = trace.as_deref_mut().map(|trace| &mut trace.layers[index]);
            cpu::rmsnorm(
                activation,
                &layer.input_norm,
                norm,
                n,
                c.hidden_size,
                c.rms_norm_eps,
            )?;
            cpu::matmul_f64(
                norm,
                &layer.q.weight,
                Some(&layer.q.bias),
                q,
                n,
                c.hidden_size,
                c.hidden_size,
            )?;
            cpu::matmul_f64(
                norm,
                &layer.k.weight,
                Some(&layer.k.bias),
                k,
                n,
                c.hidden_size,
                c.kv_heads * c.head_dim,
            )?;
            cpu::matmul_f64(
                norm,
                &layer.v.weight,
                Some(&layer.v.bias),
                v,
                n,
                c.hidden_size,
                c.kv_heads * c.head_dim,
            )?;
            cpu::rotary(q, q_rot, n, c.query_heads, c.head_dim, offset, c.rope_theta)?;
            cpu::rotary(k, k_rot, n, c.kv_heads, c.head_dim, offset, c.rope_theta)?;
            if let Some(cache) = cache.as_deref_mut() {
                let capacity = cache.capacity;
                let kv = &mut cache.layers[index];
                kv.append(k_rot, v, c, capacity, cached_len);
                kv.attend(
                    q_rot,
                    context,
                    scores,
                    probabilities,
                    &cpu::CachedAttentionShape {
                        tokens: n,
                        query_heads: c.query_heads,
                        kv_heads: c.kv_heads,
                        head_dim: c.head_dim,
                        capacity,
                        offset: cached_len,
                    },
                )?;
            } else {
                cpu::attention(
                    q_rot,
                    k_rot,
                    v,
                    context,
                    scores,
                    probabilities,
                    &AttentionShape {
                        tokens: n,
                        query_heads: c.query_heads,
                        kv_heads: c.kv_heads,
                        head_dim: c.head_dim,
                    },
                )?;
            }
            cpu::matmul_f64(
                context,
                &layer.o,
                None,
                projection,
                n,
                c.hidden_size,
                c.hidden_size,
            )?;
            if let Some(t) = t.as_deref_mut() {
                t.input_norm.copy_from_slice(norm);
                t.q_projection.copy_from_slice(q);
                t.k_projection.copy_from_slice(k);
                t.v_projection.copy_from_slice(v);
                t.q_rotary.copy_from_slice(q_rot);
                t.k_rotary.copy_from_slice(k_rot);
                t.attention_scores.copy_from_slice(scores);
                t.attention_probabilities.copy_from_slice(probabilities);
                t.attention_context.copy_from_slice(context);
                t.attention_projection.copy_from_slice(projection);
            }
            cpu::residual_add(activation, projection)?;
            if let Some(t) = t.as_deref_mut() {
                t.attention_residual.copy_from_slice(activation);
            }
            cpu::rmsnorm(
                activation,
                &layer.post_attention_norm,
                norm,
                n,
                c.hidden_size,
                c.rms_norm_eps,
            )?;
            cpu::matmul_f64(
                norm,
                &layer.gate,
                None,
                gate,
                n,
                c.hidden_size,
                c.intermediate_size,
            )?;
            cpu::matmul_f64(
                norm,
                &layer.up,
                None,
                up,
                n,
                c.hidden_size,
                c.intermediate_size,
            )?;
            cpu::gated_silu(gate, up, gated)?;
            cpu::matmul_f64(
                gated,
                &layer.down,
                None,
                projection,
                n,
                c.intermediate_size,
                c.hidden_size,
            )?;
            if let Some(t) = t.as_deref_mut() {
                t.post_attention_norm.copy_from_slice(norm);
                t.gate_projection.copy_from_slice(gate);
                t.up_projection.copy_from_slice(up);
                cpu::silu(gate, &mut t.gate_silu)?;
                t.gated.copy_from_slice(gated);
                t.down_projection.copy_from_slice(projection);
            }
            cpu::residual_add(activation, projection)?;
            if let Some(t) = t {
                t.output.copy_from_slice(activation);
            }
        }
        cpu::rmsnorm(
            activation,
            &self.weights.final_norm,
            norm,
            n,
            c.hidden_size,
            c.rms_norm_eps,
        )?;
        if let Some(trace) = trace {
            trace.final_norm.copy_from_slice(norm);
        }
        Ok(())
    }
}

/// Reusable full-prefix scratch, bounded by a model-compatible token capacity.
/// Buffers are contiguous FP32 arrays with the kernel layouts; no KV is retained.
pub struct Scratch {
    config: Config,
    capacity: usize,
    allocated_bytes: usize,
    activation: Vec<f32>,
    norm: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    q_rot: Vec<f32>,
    k_rot: Vec<f32>,
    context: Vec<f32>,
    projection: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    gated: Vec<f32>,
    scores: Vec<f32>,
    probabilities: Vec<f32>,
}

impl Scratch {
    fn new(c: &Config, capacity: usize) -> Result<Self, Error> {
        let ns = product(product(c.query_heads, capacity)?, capacity)?;
        Self::with_scores(c, capacity, ns)
    }

    pub(crate) fn for_session(c: &Config, chunk: usize, capacity: usize) -> Result<Self, Error> {
        Self::with_scores(c, chunk, capacity)
    }

    pub(crate) fn planned_bytes(
        c: &Config,
        capacity: usize,
        scores: usize,
    ) -> Result<usize, Error> {
        let nh = product(capacity, c.hidden_size)?;
        let nkv = product(product(capacity, c.kv_heads)?, c.head_dim)?;
        let nm = product(capacity, c.intermediate_size)?;
        total_bytes(&[
            nh, nh, nh, nkv, nkv, nh, nkv, nh, nh, nm, nm, nm, scores, scores,
        ])
    }

    fn with_scores(c: &Config, capacity: usize, ns: usize) -> Result<Self, Error> {
        nonzero(capacity, "scratch capacity")?;
        if capacity > c.max_positions {
            return Err(Error::Capacity {
                offset: 0,
                tokens: capacity,
                limit: c.max_positions,
            });
        }
        let nh = product(capacity, c.hidden_size)?;
        let nkv = product(product(capacity, c.kv_heads)?, c.head_dim)?;
        let nm = product(capacity, c.intermediate_size)?;
        Self::planned_bytes(c, capacity, ns)?;
        let mut s = Self {
            config: c.clone(),
            capacity,
            allocated_bytes: 0,
            activation: zeros(nh)?,
            norm: zeros(nh)?,
            q: zeros(nh)?,
            k: zeros(nkv)?,
            v: zeros(nkv)?,
            q_rot: zeros(nh)?,
            k_rot: zeros(nkv)?,
            context: zeros(nh)?,
            projection: zeros(nh)?,
            gate: zeros(nm)?,
            up: zeros(nm)?,
            gated: zeros(nm)?,
            scores: zeros(ns)?,
            probabilities: zeros(ns)?,
        };
        s.allocated_bytes = total_bytes(&[
            s.activation.capacity(),
            s.norm.capacity(),
            s.q.capacity(),
            s.k.capacity(),
            s.v.capacity(),
            s.q_rot.capacity(),
            s.k_rot.capacity(),
            s.context.capacity(),
            s.projection.capacity(),
            s.gate.capacity(),
            s.up.capacity(),
            s.gated.capacity(),
            s.scores.capacity(),
            s.probabilities.capacity(),
        ])?;
        Ok(s)
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }
}
