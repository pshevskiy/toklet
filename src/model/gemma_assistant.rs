//! Scalar Gemma4 E2B assistant, with fixed source geometry and borrowed target KV.
//! BF16 coefficients are kept packed. Projections accumulate in FP64, with an
//! FP32 store at the operation boundary. The target remains the greedy authority.

use crate::{
    cpu::{self, AttentionKvSegment, ProportionalRotaryShape, SpanAttentionShape},
    mlx::gemma4::assistant::{
        CANDIDATES, CENTROIDS, ORDERING_ALLOWANCE_BYTES, OrderedHead, SOURCE_ALLOWANCE_BYTES,
        Source, VOCAB,
    },
    weights::Error,
};
use std::{mem::size_of, path::Path};

const HIDDEN: usize = 256;
pub const TARGET_HIDDEN: usize = 1536;
const FF: usize = 2048;
const HEADS: usize = 4;
const WINDOW: usize = 512;
const MAX_POSITIONS: usize = 1025;
const EPSILON: f32 = 1e-6;
const LAYERS: usize = 4;
const QUERY_GAIN_VALUES: usize = 3 * 256 + 512;
const NORM_VALUES: usize = (4 * HIDDEN + 1) * LAYERS + QUERY_GAIN_VALUES + HIDDEN;
const MAX_GAIN_STAGING_BYTES: usize = 512 * size_of::<u16>();
const MATRIX_VALUES: usize = 155_462_664 / 2 - NORM_VALUES;
/// Maximum requested cold bytes for matrix/gain data, ordering and bounded
/// source inspection, plus the small inline owner. No process-footprint claim.
pub const LOAD_BYTES: usize = MATRIX_VALUES * 2
    + QUERY_GAIN_VALUES * size_of::<f32>()
    + MAX_GAIN_STAGING_BYTES
    + ORDERING_ALLOWANCE_BYTES
    + SOURCE_ALLOWANCE_BYTES
    + size_of::<Assistant>();

fn bad(reason: impl Into<String>) -> Error {
    Error::Invalid(reason.into())
}

struct Matrix {
    columns: usize,
    values: Vec<u16>,
}

fn words(source: &mut Source, name: &str, shape: &[usize]) -> Result<Vec<u16>, Error> {
    let descriptor = source
        .plan()
        .tensors()
        .get(name)
        .ok_or_else(|| bad("assistant tensor is missing"))?;
    if descriptor.shape != shape {
        return Err(bad("assistant tensor shape differs"));
    }
    let count = descriptor.range.len() / 2;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| Error::Allocation)?;
    if values.capacity() != count {
        return Err(bad("assistant tensor allocation differs"));
    }
    source.visit_bf16(name, |bytes| {
        for word in bytes.chunks_exact(2) {
            let bits = u16::from_le_bytes([word[0], word[1]]);
            if bits & 0x7f80 == 0x7f80 {
                return Err(bad("non-finite BF16 assistant coefficient"));
            }
            values.push(bits);
        }
        Ok(())
    })?;
    if values.len() != count {
        return Err(bad("assistant tensor delivery length differs"));
    }
    Ok(values)
}

fn gain<const N: usize>(source: &mut Source, name: &str) -> Result<[f32; N], Error> {
    let words = words(source, name, &[N])?;
    Ok(std::array::from_fn(|i| widen(words[i])))
}

fn widen(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

impl Matrix {
    fn read(source: &mut Source, name: &str, rows: usize, columns: usize) -> Result<Self, Error> {
        Ok(Self {
            columns,
            values: words(source, name, &[rows, columns])?,
        })
    }

    fn row(&self, index: usize, x: &[f32]) -> f32 {
        let start = index * self.columns;
        self.values[start..start + self.columns]
            .iter()
            .zip(x)
            .map(|(&w, &x)| f64::from(widen(w)) * f64::from(x))
            .sum::<f64>() as f32
    }

    fn project(&self, x: &[f32], y: &mut [f32]) -> Result<(), cpu::Error> {
        cpu::length("assistant projection input", x.len(), self.columns)?;
        cpu::length(
            "assistant projection output",
            y.len(),
            self.values.len() / self.columns,
        )?;
        for (row, y) in y.iter_mut().enumerate() {
            *y = self.row(row, x);
        }
        cpu::finite("assistant projection output", y)
    }
}

struct Layer {
    input: [f32; HIDDEN],
    q: Matrix,
    q_gain: Vec<f32>,
    o: Matrix,
    post_attention: [f32; HIDDEN],
    pre_ff: [f32; HIDDEN],
    gate: Matrix,
    up: Matrix,
    down: Matrix,
    post_ff: [f32; HIDDEN],
    scalar: f32,
}

impl Layer {
    fn read(source: &mut Source, index: usize, width: usize) -> Result<Self, Error> {
        let name = |suffix: &str| format!("model.layers.{index}.{suffix}");
        let q_gain = words(source, &name("self_attn.q_norm.weight"), &[width])?;
        let mut q_gain_f32 = Vec::new();
        q_gain_f32
            .try_reserve_exact(width)
            .map_err(|_| Error::Allocation)?;
        q_gain_f32.extend(q_gain.into_iter().map(widen));
        if q_gain_f32.capacity() != width {
            return Err(bad("assistant gain allocation differs"));
        }
        Ok(Self {
            input: gain(source, &name("input_layernorm.weight"))?,
            q: Matrix::read(
                source,
                &name("self_attn.q_proj.weight"),
                HEADS * width,
                HIDDEN,
            )?,
            q_gain: q_gain_f32,
            o: Matrix::read(
                source,
                &name("self_attn.o_proj.weight"),
                HIDDEN,
                HEADS * width,
            )?,
            post_attention: gain(source, &name("post_attention_layernorm.weight"))?,
            pre_ff: gain(source, &name("pre_feedforward_layernorm.weight"))?,
            gate: Matrix::read(source, &name("mlp.gate_proj.weight"), FF, HIDDEN)?,
            up: Matrix::read(source, &name("mlp.up_proj.weight"), FF, HIDDEN)?,
            down: Matrix::read(source, &name("mlp.down_proj.weight"), HIDDEN, FF)?,
            post_ff: gain(source, &name("post_feedforward_layernorm.weight"))?,
            scalar: gain::<1>(source, &name("layer_scalar"))?[0],
        })
    }
}

pub struct Assistant {
    pre: Matrix,
    layers: [Layer; LAYERS],
    final_gain: [f32; HIDDEN],
    post: Matrix,
    centroids: Matrix,
    embedding: Matrix,
    ordering: OrderedHead,
}

impl Assistant {
    pub fn open(directory: &Path, budget: usize) -> Result<Self, Error> {
        if budget < LOAD_BYTES {
            return Err(Error::Budget {
                required: LOAD_BYTES,
                limit: budget,
            });
        }
        let mut source = Source::open(directory, SOURCE_ALLOWANCE_BYTES)?;
        Ok(Self {
            pre: Matrix::read(
                &mut source,
                "pre_projection.weight",
                HIDDEN,
                2 * TARGET_HIDDEN,
            )?,
            layers: [
                Layer::read(&mut source, 0, 256)?,
                Layer::read(&mut source, 1, 256)?,
                Layer::read(&mut source, 2, 256)?,
                Layer::read(&mut source, 3, 512)?,
            ],
            final_gain: gain(&mut source, "model.norm.weight")?,
            post: Matrix::read(&mut source, "post_projection.weight", TARGET_HIDDEN, HIDDEN)?,
            centroids: Matrix::read(
                &mut source,
                "masked_embedding.centroids.weight",
                CENTROIDS,
                HIDDEN,
            )?,
            embedding: Matrix::read(&mut source, "model.embed_tokens.weight", VOCAB, HIDDEN)?,
            ordering: source.ordered_head(ORDERING_ALLOWANCE_BYTES)?,
        })
    }

    /// One proposal at the fixed absolute target position. `input` is scaled
    /// target token embedding followed by the target/previous proposal hidden.
    /// Failure leaves the target immutable; the workspace contents are private.
    pub fn propose<'a>(
        &self,
        input: &[f32; 2 * TARGET_HIDDEN],
        kv: &TargetKv<'_>,
        scratch: &'a mut Scratch,
    ) -> Result<Proposal<'a>, Error> {
        let cpu_error = |err: cpu::Error| bad(err.to_string());
        let position = kv.position;
        if position >= MAX_POSITIONS || position >= scratch.positions {
            return Err(bad(
                "assistant committed position exceeds admitted capacity",
            ));
        }
        cpu::finite("assistant input", input).map_err(cpu_error)?;
        self.pre
            .project(input, &mut scratch.hidden)
            .map_err(cpu_error)?;
        for (index, layer) in self.layers.iter().enumerate() {
            layer.forward(scratch, kv, index == 3).map_err(cpu_error)?;
        }
        cpu::rmsnorm(
            &scratch.hidden,
            &self.final_gain,
            &mut scratch.normal,
            1,
            HIDDEN,
            EPSILON,
        )
        .map_err(cpu_error)?;
        self.post
            .project(&scratch.normal, &mut scratch.projected)
            .map_err(cpu_error)?;
        self.centroids
            .project(&scratch.normal, &mut scratch.centroid_scores)
            .map_err(cpu_error)?;
        self.ordering
            .candidate_ids(&scratch.centroid_scores, &mut scratch.ids)?;
        let mut winner = u32::MAX;
        let mut best = f32::NEG_INFINITY;
        for (&id, score) in scratch.ids.iter().zip(scratch.logits.iter_mut()) {
            *score = self.embedding.row(id as usize, &scratch.normal);
            if !score.is_finite() {
                return Err(bad("non-finite assistant candidate logit"));
            }
            if *score > best || (*score == best && id < winner) {
                best = *score;
                winner = id;
            }
        }
        Ok(Proposal {
            winner,
            hidden: &scratch.projected,
            ids: &scratch.ids,
            logits: &scratch.logits,
        })
    }
}

impl Layer {
    fn forward(&self, s: &mut Scratch, kv: &TargetKv<'_>, full: bool) -> Result<(), cpu::Error> {
        let position = kv.position;
        let (segments, width, start, theta) = if full {
            (&kv.full, 512, 0, 1e6)
        } else {
            (&kv.local, 256, (position + 1).saturating_sub(WINDOW), 1e4)
        };
        let q = HEADS * width;
        let keys = position + 1 - start;
        let scores = HEADS * keys;
        cpu::rmsnorm(&s.hidden, &self.input, &mut s.normal, 1, HIDDEN, EPSILON)?;
        self.q.project(&s.normal, &mut s.query[..q])?;
        cpu::rmsnorm(
            &s.query[..q],
            &self.q_gain,
            &mut s.query_norm[..q],
            HEADS,
            width,
            EPSILON,
        )?;
        cpu::proportional_rotary(
            &s.query_norm[..q],
            &mut s.query[..q],
            &ProportionalRotaryShape {
                tokens: 1,
                heads: HEADS,
                head_dim: width,
                rotated_dims: if full { width / 4 } else { width },
                offset: position,
                base: theta,
                factor: 1.,
            },
        )?;
        cpu::span_attention(
            &s.query[..q],
            segments,
            &mut s.context[..q],
            &mut s.scores[..scores],
            &mut s.probabilities[..scores],
            &SpanAttentionShape {
                queries: 1,
                keys,
                query_heads: HEADS,
                kv_heads: 1,
                head_dim: width,
                query_start: position,
                key_start: start,
                window: (!full).then_some(WINDOW),
                scale: 1.,
            },
        )?;
        self.o.project(&s.context[..q], &mut s.delta)?;
        cpu::rmsnorm(
            &s.delta,
            &self.post_attention,
            &mut s.normal,
            1,
            HIDDEN,
            EPSILON,
        )?;
        cpu::residual_add(&mut s.hidden, &s.normal)?;
        cpu::rmsnorm(&s.hidden, &self.pre_ff, &mut s.normal, 1, HIDDEN, EPSILON)?;
        self.gate.project(&s.normal, &mut s.gate)?;
        self.up.project(&s.normal, &mut s.up)?;
        cpu::gated_gelu_tanh(&s.gate, &s.up, &mut s.product)?;
        self.down.project(&s.product, &mut s.delta)?;
        cpu::rmsnorm(&s.delta, &self.post_ff, &mut s.normal, 1, HIDDEN, EPSILON)?;
        cpu::residual_add(&mut s.hidden, &s.normal)?;
        cpu::scale_in_place(&mut s.hidden, self.scalar)
    }
}

pub struct TargetKv<'a> {
    pub position: usize,
    pub local: [AttentionKvSegment<'a>; 3],
    pub full: [AttentionKvSegment<'a>; 3],
}

pub struct Proposal<'a> {
    pub winner: u32,
    pub hidden: &'a [f32; TARGET_HIDDEN],
    pub ids: &'a [u32; CANDIDATES],
    pub logits: &'a [f32; CANDIDATES],
}

pub struct Scratch {
    positions: usize,
    hidden: [f32; HIDDEN],
    normal: [f32; HIDDEN],
    delta: [f32; HIDDEN],
    query: [f32; HEADS * 512],
    query_norm: [f32; HEADS * 512],
    context: [f32; HEADS * 512],
    gate: [f32; FF],
    up: [f32; FF],
    product: [f32; FF],
    scores: Vec<f32>,
    probabilities: Vec<f32>,
    projected: [f32; TARGET_HIDDEN],
    centroid_scores: [f32; CENTROIDS],
    ids: [u32; CANDIDATES],
    logits: [f32; CANDIDATES],
}

impl Scratch {
    pub fn planned_bytes(positions: usize) -> Result<usize, Error> {
        if !(1..=MAX_POSITIONS).contains(&positions) {
            return Err(bad("assistant scratch positions exceed contract"));
        }
        Ok(size_of::<Self>() + 2 * HEADS * positions * size_of::<f32>())
    }

    pub fn new(positions: usize, budget: usize) -> Result<Self, Error> {
        let required = Self::planned_bytes(positions)?;
        if required > budget {
            return Err(Error::Budget {
                required,
                limit: budget,
            });
        }
        let zeros = || {
            let count = HEADS * positions;
            let mut values = Vec::new();
            values
                .try_reserve_exact(count)
                .map_err(|_| Error::Allocation)?;
            if values.capacity() != count {
                return Err(bad("assistant scratch allocation differs"));
            }
            values.resize(count, 0.);
            Ok(values)
        };
        Ok(Self {
            positions,
            hidden: [0.; HIDDEN],
            normal: [0.; HIDDEN],
            delta: [0.; HIDDEN],
            query: [0.; HEADS * 512],
            query_norm: [0.; HEADS * 512],
            context: [0.; HEADS * 512],
            gate: [0.; FF],
            up: [0.; FF],
            product: [0.; FF],
            scores: zeros()?,
            probabilities: zeros()?,
            projected: [0.; TARGET_HIDDEN],
            centroid_scores: [0.; CENTROIDS],
            ids: [0; CANDIDATES],
            logits: [0.; CANDIDATES],
        })
    }
}

#[cfg(test)]
mod tests;
