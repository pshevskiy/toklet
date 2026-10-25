//! Bounded dense FP32 Gemma text graph for the frozen synthetic witness.
//!
//! This module admits exactly H8/V17/four alternating local/global layers,
//! Q2/KV1, D8/16, PLE2, MLP12/24 and positions 0..31. It is not a checkpoint
//! loader, quantized fallback or app capability. Mathematical and source CPU
//! parity are separate from the unresolved source GPU and long-rotary gaps.

use super::shared_kv::{KvKind, LayerKvGeometry, SharedKvPlan};
use super::window_cache::{WindowKvCache, reset_batch};
use super::{Error, allocation_bytes, total_bytes, zeros};
use crate::cpu::{self, DenseDecoderWeights, finite, length};

mod forward;
#[cfg(test)]
pub(crate) mod tests;

pub const MAX_POSITIONS: usize = 32;
pub const VOCAB_SIZE: usize = 17;
const HIDDEN: usize = 8;
const LAYERS: usize = 4;
const HEADS: usize = 2;
const PLE: usize = 2;
const WINDOW: usize = 4;
const EPSILON: f32 = 1e-6;

pub struct KvWeights<'a> {
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub k_norm: &'a [f32],
}

pub struct LayerWeights<'a> {
    pub input_norm: &'a [f32],
    pub q: &'a [f32],
    pub q_norm: &'a [f32],
    pub o: &'a [f32],
    /// Present only for layers 0 and 1. Consumers own no K/V projections.
    pub kv: Option<KvWeights<'a>>,
    pub finish: DenseDecoderWeights<'a>,
}

/// Immutable borrowed row-major FP32 weights, with one tied embedding/head.
/// Norm scales are direct, not `1 + weight`. No maps or tensor-name resolution
/// occur in forward execution. Callers retain backing storage for this model.
pub struct Weights<'a> {
    pub embedding: &'a [f32],
    pub per_layer_embedding: &'a [f32],
    pub per_layer_projection: &'a [f32],
    pub per_layer_norm: &'a [f32],
    pub layers: [LayerWeights<'a>; LAYERS],
    pub final_norm: &'a [f32],
}

pub struct Model<'a> {
    weights: Weights<'a>,
    plan: SharedKvPlan,
    unit_norm: [f32; 16],
    weight_bytes: usize,
}

fn dimension(layer: usize) -> usize {
    if layer.is_multiple_of(2) { 8 } else { 16 }
}

fn intermediate(layer: usize) -> usize {
    if layer < 2 { 12 } else { 24 }
}

fn check(values: &[f32], expected: usize, name: &'static str) -> Result<(), Error> {
    length(name, values.len(), expected)?;
    finite(name, values)?;
    Ok(())
}

impl<'a> Model<'a> {
    pub fn new(weights: Weights<'a>) -> Result<Self, Error> {
        check(weights.embedding, VOCAB_SIZE * HIDDEN, "Gemma embedding")?;
        check(
            weights.per_layer_embedding,
            VOCAB_SIZE * LAYERS * PLE,
            "Gemma PLE table",
        )?;
        check(
            weights.per_layer_projection,
            LAYERS * PLE * HIDDEN,
            "Gemma PLE projection",
        )?;
        check(weights.per_layer_norm, PLE, "Gemma PLE norm")?;
        check(weights.final_norm, HIDDEN, "Gemma final norm")?;
        let mut elements = weights.embedding.len()
            + weights.per_layer_embedding.len()
            + weights.per_layer_projection.len()
            + PLE
            + HIDDEN;
        for (index, layer) in weights.layers.iter().enumerate() {
            let d = dimension(index);
            let ff = intermediate(index);
            for (values, size, name) in [
                (layer.input_norm, HIDDEN, "Gemma input norm"),
                (layer.q, HEADS * d * HIDDEN, "Gemma Q projection"),
                (layer.q_norm, d, "Gemma Q norm"),
                (layer.o, HIDDEN * HEADS * d, "Gemma O projection"),
                (
                    layer.finish.post_attention_norm,
                    HIDDEN,
                    "Gemma post attention norm",
                ),
                (layer.finish.pre_ff_norm, HIDDEN, "Gemma pre FF norm"),
                (layer.finish.post_ff_norm, HIDDEN, "Gemma post FF norm"),
                (layer.finish.ff_gate, ff * HIDDEN, "Gemma FF gate"),
                (layer.finish.ff_up, ff * HIDDEN, "Gemma FF up"),
                (layer.finish.ff_down, HIDDEN * ff, "Gemma FF down"),
            ] {
                check(values, size, name)?;
                elements += values.len();
            }
            match (&layer.kv, index < 2) {
                (Some(kv), true) => {
                    for (values, size, name) in [
                        (kv.k, d * HIDDEN, "Gemma K projection"),
                        (kv.v, d * HIDDEN, "Gemma V projection"),
                        (kv.k_norm, d, "Gemma K norm"),
                    ] {
                        check(values, size, name)?;
                        elements += values.len();
                    }
                }
                (None, false) => {}
                _ => return Err(Error::InvalidConfig("Gemma reference KV ownership")),
            }
            let ple = layer
                .finish
                .per_layer
                .as_ref()
                .ok_or(Error::InvalidConfig("Gemma reference requires PLE gate"))?;
            if ple.width != PLE {
                return Err(Error::InvalidConfig("Gemma reference PLE width"));
            }
            for (values, size, name) in [
                (ple.gate, PLE * HIDDEN, "Gemma PLE gate"),
                (ple.projection, HIDDEN * PLE, "Gemma PLE output"),
                (ple.post_norm, HIDDEN, "Gemma PLE post norm"),
            ] {
                check(values, size, name)?;
                elements += values.len();
            }
            if layer.finish.scalar.is_none_or(|scalar| !scalar.is_finite()) {
                return Err(Error::InvalidConfig(
                    "Gemma reference requires finite scalar",
                ));
            }
            elements += 1;
        }
        let geometry = std::array::from_fn::<_, LAYERS, _>(|index| LayerKvGeometry {
            kind: if index.is_multiple_of(2) {
                KvKind::Sliding
            } else {
                KvKind::Full
            },
            heads: 1,
            head_dim: dimension(index),
        });
        Ok(Self {
            weights,
            plan: SharedKvPlan::new(&geometry, 2, WINDOW, MAX_POSITIONS, MAX_POSITIONS)?,
            unit_norm: [1.; 16],
            weight_bytes: allocation_bytes(elements)?,
        })
    }

    /// Logical borrowed weights plus this model and its route allocation. This
    /// conservative count does not infer a process/device allowance or fit.
    pub fn model_bytes(&self) -> usize {
        self.weight_bytes + std::mem::size_of::<Self>() + self.plan.allocated_metadata_bytes()
            - std::mem::size_of::<SharedKvPlan>()
    }

    pub fn planned_session_bytes(&self) -> Result<usize, Error> {
        Ok(
            std::mem::size_of::<Session<'_, '_>>() + self.plan.planned_cache_bytes()?
                - std::mem::size_of::<Vec<WindowKvCache>>()
                + total_bytes(&Scratch::counts())?
                + allocation_bytes(MAX_POSITIONS * (8 + 16) * 2)?,
        )
    }

    /// Check the complete logical model+session requirement before allocating
    /// caches, scratch or staging. Exact actual Vec capacities are rechecked.
    pub fn new_session(&self, budget: usize) -> Result<Session<'_, 'a>, Error> {
        let planned = self
            .model_bytes()
            .checked_add(self.planned_session_bytes()?)
            .ok_or(cpu::Error::SizeOverflow)?;
        if budget < planned {
            return Err(Error::InvalidConfig("Gemma reference memory budget"));
        }
        let result = Session {
            model: self,
            caches: self.plan.new_caches()?,
            staging: [
                Staging {
                    k: zeros(MAX_POSITIONS * 8)?,
                    v: zeros(MAX_POSITIONS * 8)?,
                },
                Staging {
                    k: zeros(MAX_POSITIONS * 16)?,
                    v: zeros(MAX_POSITIONS * 16)?,
                },
            ],
            scratch: Scratch::new()?,
            history: [0; MAX_POSITIONS],
            committed: 0,
            published_rows: 0,
            #[cfg(test)]
            fault: None,
        };
        if self
            .model_bytes()
            .checked_add(result.allocated_bytes()?)
            .ok_or(cpu::Error::SizeOverflow)?
            > budget
        {
            return Err(Error::InvalidConfig("Gemma actual reference memory budget"));
        }
        Ok(result)
    }
}

struct Staging {
    k: Vec<f32>,
    v: Vec<f32>,
}

struct Scratch {
    hidden: Vec<f32>,
    next: Vec<f32>,
    norm: Vec<f32>,
    attention: Vec<f32>,
    q: Vec<f32>,
    q_norm: Vec<f32>,
    context: Vec<f32>,
    kv: Vec<f32>,
    kv_norm: Vec<f32>,
    ple_lookup: Vec<f32>,
    ple_projection: Vec<f32>,
    ple_scaled: Vec<f32>,
    ple: Vec<f32>,
    layer_ple: Vec<f32>,
    dense_work: Vec<f32>,
    dense_norm: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    product: Vec<f32>,
    scores: Vec<f32>,
    probabilities: Vec<f32>,
    logits: Vec<f32>,
    published: Vec<f32>,
}

impl Scratch {
    fn counts() -> [usize; 23] {
        let h = MAX_POSITIONS * HIDDEN;
        let q = MAX_POSITIONS * HEADS * 16;
        let kv = MAX_POSITIONS * 16;
        let ple = MAX_POSITIONS * LAYERS * PLE;
        let ff = MAX_POSITIONS * 24;
        let scores = HEADS * MAX_POSITIONS * MAX_POSITIONS;
        [
            h,
            h,
            h,
            h,
            q,
            q,
            q,
            kv,
            kv,
            ple,
            ple,
            ple,
            ple,
            MAX_POSITIONS * PLE,
            h,
            h,
            ff,
            ff,
            ff,
            scores,
            scores,
            MAX_POSITIONS * VOCAB_SIZE,
            MAX_POSITIONS * VOCAB_SIZE,
        ]
    }
    fn new() -> Result<Self, Error> {
        let [
            h,
            n,
            norm,
            a,
            q,
            qn,
            c,
            kv,
            kvn,
            lookup,
            projection,
            scaled,
            ple,
            lp,
            dw,
            dn,
            g,
            u,
            p,
            s,
            pr,
            l,
            pub_l,
        ] = Self::counts();
        Ok(Self {
            hidden: zeros(h)?,
            next: zeros(n)?,
            norm: zeros(norm)?,
            attention: zeros(a)?,
            q: zeros(q)?,
            q_norm: zeros(qn)?,
            context: zeros(c)?,
            kv: zeros(kv)?,
            kv_norm: zeros(kvn)?,
            ple_lookup: zeros(lookup)?,
            ple_projection: zeros(projection)?,
            ple_scaled: zeros(scaled)?,
            ple: zeros(ple)?,
            layer_ple: zeros(lp)?,
            dense_work: zeros(dw)?,
            dense_norm: zeros(dn)?,
            gate: zeros(g)?,
            up: zeros(u)?,
            product: zeros(p)?,
            scores: zeros(s)?,
            probabilities: zeros(pr)?,
            logits: zeros(l)?,
            published: zeros(pub_l)?,
        })
    }
    fn capacities(&self) -> [usize; 23] {
        [
            &self.hidden,
            &self.next,
            &self.norm,
            &self.attention,
            &self.q,
            &self.q_norm,
            &self.context,
            &self.kv,
            &self.kv_norm,
            &self.ple_lookup,
            &self.ple_projection,
            &self.ple_scaled,
            &self.ple,
            &self.layer_ple,
            &self.dense_work,
            &self.dense_norm,
            &self.gate,
            &self.up,
            &self.product,
            &self.scores,
            &self.probabilities,
            &self.logits,
            &self.published,
        ]
        .map(Vec::capacity)
    }
}

/// A session borrows exactly one immutable model. Callers cannot pass a different
/// same-shaped model into forward. Committed history contains only input tokens;
/// sampling, pending-token history and prefix reuse are not implemented here.
pub struct Session<'model, 'weights> {
    model: &'model Model<'weights>,
    caches: Vec<WindowKvCache>,
    staging: [Staging; 2],
    scratch: Scratch,
    history: [u32; MAX_POSITIONS],
    committed: usize,
    published_rows: usize,
    #[cfg(test)]
    fault: Option<usize>,
}

impl Session<'_, '_> {
    pub fn tokens(&self) -> &[u32] {
        &self.history[..self.committed]
    }
    /// All rows from the last successful chunk, `[rows,17]`, or empty after reset.
    pub fn logits(&self) -> &[f32] {
        &self.scratch.published[..self.published_rows * VOCAB_SIZE]
    }
    pub fn allocated_bytes(&self) -> Result<usize, Error> {
        Ok(
            std::mem::size_of::<Self>() + self.model.plan.allocated_cache_bytes(&self.caches)?
                - std::mem::size_of::<Vec<WindowKvCache>>()
                + total_bytes(&self.scratch.capacities())?
                + total_bytes(&[
                    self.staging[0].k.capacity(),
                    self.staging[0].v.capacity(),
                    self.staging[1].k.capacity(),
                    self.staging[1].v.capacity(),
                ])?,
        )
    }
    /// Discard the committed prefix without allocating new cache/scratch owners.
    pub fn reset(&mut self) -> Result<(), Error> {
        reset_batch(&mut self.caches)?;
        self.committed = 0;
        self.published_rows = 0;
        self.history.fill(0);
        Ok(())
    }
    /// Append a nonempty chunk within 32 total positions. Failure preserves all
    /// committed KV/history and the prior public logits; scratch is discardable.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<&[f32], Error> {
        forward::run(self, tokens)?;
        Ok(self.logits())
    }
}
