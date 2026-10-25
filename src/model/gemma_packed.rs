//! Borrowed packed Gemma graph for a frozen witness and a named short E2B adapter.
//!
//! `Model::new` admits the short synthetic profile; `Model::from_e2b` borrows
//! an admitted E2B payload, with32 executable positions and no app capability.
//! Packed4/group64 BF16 coefficient values reconstruct into
//! FP32; activations/KV remain FP32 and reductions use the scalar FP64 oracle.
//! Original-BF16 source-graph, GPU, long-position and device gates stay separate.

use super::shared_kv::{KvKind, LayerKvGeometry, SharedKvPlan};
use super::window_cache::{WindowKvCache, reset_batch};
use super::{Error, allocation_bytes, total_bytes, zeros};
use crate::cpu::{self, finite, length};
use crate::mlx::affine::{Matrix, MetadataDtype};

mod e2b;
mod forward;
mod generation;
pub use generation::GenerationSession;
#[cfg(test)]
mod tests;

pub const MAX_POSITIONS: usize = 32;
pub const VOCAB_SIZE: usize = 17;
const EPSILON: f32 = 1e-6;
const MAX_PRODUCERS: usize = 15;

pub struct KvWeights<'a> {
    pub k: Matrix<'a>,
    pub v: Matrix<'a>,
    pub k_norm: &'a [f32],
}
pub struct LayerWeights<'a> {
    pub input_norm: &'a [f32],
    pub q: Matrix<'a>,
    pub q_norm: &'a [f32],
    pub o: Matrix<'a>,
    /// Only the producer prefix owns K/V projections.
    pub kv: Option<KvWeights<'a>>,
    pub post_attention_norm: &'a [f32],
    pub pre_ff_norm: &'a [f32],
    pub post_ff_norm: &'a [f32],
    pub ff_gate: Matrix<'a>,
    pub ff_up: Matrix<'a>,
    pub ff_down: Matrix<'a>,
    pub ple_gate: Matrix<'a>,
    pub ple_projection: Matrix<'a>,
    pub post_ple_norm: &'a [f32],
    pub scalar: f32,
}
/// Immutable borrowed byte-backed matrices and direct learned FP32 norms.
/// The embedding is the tied output head and is counted only once. Tensor names
/// and byte spans are resolved before construction, never in numerical loops.
pub struct Weights<'a> {
    pub embedding: Matrix<'a>,
    pub per_layer_embedding: Matrix<'a>,
    pub per_layer_projection: Matrix<'a>,
    pub per_layer_norm: &'a [f32],
    pub layers: Vec<LayerWeights<'a>>,
    pub final_norm: &'a [f32],
}
struct LayerLayout {
    kind: KvKind,
    head_dim: usize,
    rotated_dims: usize,
    intermediate: usize,
    theta: f32,
}
// Geometry is private. A later separately qualified checkpoint adapter can
// construct this same execution layout; callers cannot grant arbitrary shapes.
struct Layout {
    hidden: usize,
    vocab: usize,
    ple: usize,
    heads: usize,
    producers: usize,
    positions: usize,
    window: usize,
    layers: Vec<LayerLayout>,
}
impl Layout {
    fn witness() -> Self {
        Self {
            hidden: 64,
            vocab: VOCAB_SIZE,
            ple: 64,
            heads: 2,
            producers: 2,
            positions: MAX_POSITIONS,
            window: 4,
            layers: (0_usize..4)
                .map(|i| {
                    let local = i.is_multiple_of(2);
                    LayerLayout {
                        kind: if local { KvKind::Sliding } else { KvKind::Full },
                        head_dim: if local { 64 } else { 128 },
                        rotated_dims: if local { 64 } else { 32 },
                        intermediate: if i < 2 { 64 } else { 128 },
                        theta: if local { 1e4 } else { 1e6 },
                    }
                })
                .collect(),
        }
    }
    fn max_dim(&self) -> usize {
        self.layers.iter().map(|l| l.head_dim).max().unwrap()
    }
    fn max_ff(&self) -> usize {
        self.layers.iter().map(|l| l.intermediate).max().unwrap()
    }
}
pub struct Model<'a> {
    weights: Weights<'a>,
    layout: Layout,
    plan: SharedKvPlan,
    unit_norm: Vec<f32>,
    bytes: usize,
    tensor_bytes: usize,
    metadata_allowance_bytes: usize,
}
fn check(values: &[f32], expected: usize, name: &'static str) -> Result<(), Error> {
    length(name, values.len(), expected)?;
    finite(name, values)?;
    Ok(())
}
fn matrix(value: &Matrix<'_>, rows: usize, columns: usize) -> Result<usize, Error> {
    let l = value.layout();
    if l.rows() != rows
        || l.columns() != columns
        || l.bits() != 4
        || l.group_size() != 64
        || l.metadata_dtype() != MetadataDtype::BF16
    {
        return Err(Error::InvalidConfig("Gemma packed matrix profile"));
    }
    Ok(l.total_bytes())
}
fn add_bytes(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(cpu::Error::SizeOverflow.into())
}
impl<'a> Model<'a> {
    pub fn new(weights: Weights<'a>) -> Result<Self, Error> {
        Self::with_layout(weights, Layout::witness())
    }
    fn with_layout(weights: Weights<'a>, l: Layout) -> Result<Self, Error> {
        Self::with_storage(weights, l, None)
    }
    fn with_storage(
        weights: Weights<'a>,
        l: Layout,
        retained: Option<&crate::mlx::gemma4::payload::LoadReport>,
    ) -> Result<Self, Error> {
        // A bounded proposal frame must never spill to the heap during forward.
        // Public admission selects only the frozen witness or named E2B geometry.
        if l.producers == 0
            || l.producers > MAX_PRODUCERS
            || l.producers > l.layers.len()
            || l.positions > MAX_POSITIONS
        {
            return Err(Error::InvalidConfig("Gemma packed producer/position bound"));
        }
        if weights.layers.len() != l.layers.len() {
            return Err(Error::LayerCount {
                expected: l.layers.len(),
                actual: weights.layers.len(),
            });
        }
        let mut bytes = 0;
        for (m, r, c) in [
            (&weights.embedding, l.vocab, l.hidden),
            (
                &weights.per_layer_embedding,
                l.vocab,
                l.layers.len() * l.ple,
            ),
            (
                &weights.per_layer_projection,
                l.layers.len() * l.ple,
                l.hidden,
            ),
        ] {
            bytes = add_bytes(bytes, matrix(m, r, c)?)?;
        }
        for (v, n) in [
            (weights.per_layer_norm, l.ple),
            (weights.final_norm, l.hidden),
        ] {
            check(v, n, "Gemma packed norm")?;
            bytes = add_bytes(bytes, allocation_bytes(v.len())?)?;
        }
        for (i, (w, g)) in weights.layers.iter().zip(&l.layers).enumerate() {
            for (m, r, c) in [
                (&w.q, l.heads * g.head_dim, l.hidden),
                (&w.o, l.hidden, l.heads * g.head_dim),
                (&w.ff_gate, g.intermediate, l.hidden),
                (&w.ff_up, g.intermediate, l.hidden),
                (&w.ff_down, l.hidden, g.intermediate),
                (&w.ple_gate, l.ple, l.hidden),
                (&w.ple_projection, l.hidden, l.ple),
            ] {
                bytes = add_bytes(bytes, matrix(m, r, c)?)?;
            }
            for (v, n) in [
                (w.input_norm, l.hidden),
                (w.q_norm, g.head_dim),
                (w.post_attention_norm, l.hidden),
                (w.pre_ff_norm, l.hidden),
                (w.post_ff_norm, l.hidden),
                (w.post_ple_norm, l.hidden),
            ] {
                check(v, n, "Gemma packed layer norm")?;
                bytes = add_bytes(bytes, allocation_bytes(v.len())?)?;
            }
            match (&w.kv, i < l.producers) {
                (Some(kv), true) => {
                    bytes = add_bytes(bytes, matrix(&kv.k, g.head_dim, l.hidden)?)?;
                    bytes = add_bytes(bytes, matrix(&kv.v, g.head_dim, l.hidden)?)?;
                    check(kv.k_norm, g.head_dim, "Gemma packed K norm")?;
                    bytes = add_bytes(bytes, allocation_bytes(kv.k_norm.len())?)?;
                }
                (None, false) => {}
                _ => return Err(Error::InvalidConfig("Gemma packed KV ownership")),
            }
            if !w.scalar.is_finite() {
                return Err(Error::InvalidConfig("Gemma packed scalar"));
            }
        }
        let tensor_bytes = retained.map_or(bytes, |report| report.tensor_capacity_bytes);
        let metadata_allowance_bytes =
            retained.map_or(0, |report| report.metadata_reservation_bytes);
        if let Some(report) = retained {
            // Replace the borrowed logical tensor sum with its owner's actual
            // full capacity. This includes retained source scalar vectors and
            // padding, exactly once; copied LayerWeights scalars are metadata.
            if report.tensor_capacity_bytes < bytes {
                return Err(Error::InvalidConfig("Gemma retained backing capacity"));
            }
            bytes = add_bytes(
                report.tensor_capacity_bytes,
                report.metadata_reservation_bytes,
            )?;
        }
        let geometry: Vec<_> = l
            .layers
            .iter()
            .map(|g| LayerKvGeometry {
                kind: g.kind,
                heads: 1,
                head_dim: g.head_dim,
            })
            .collect();
        let plan = SharedKvPlan::new(
            &geometry,
            l.layers.len() - l.producers,
            l.window,
            l.positions,
            // Physical local rings may exceed the short executable prefix.
            // Forward still rejects any request beyond l.positions before work.
            l.positions.max(l.window),
        )?;
        let mut unit_norm = zeros(l.max_dim())?;
        unit_norm.fill(1.);
        // Scalar fields live in LayerWeights and are counted by its capacity.
        for n in [
            std::mem::size_of::<Self>(),
            cpu::product(
                weights.layers.capacity(),
                std::mem::size_of::<LayerWeights<'_>>(),
            )?,
            cpu::product(l.layers.capacity(), std::mem::size_of::<LayerLayout>())?,
            allocation_bytes(unit_norm.capacity())?,
            plan.allocated_metadata_bytes() - std::mem::size_of::<SharedKvPlan>(),
        ] {
            bytes = add_bytes(bytes, n)?;
        }
        Ok(Self {
            weights,
            layout: l,
            plan,
            unit_norm,
            bytes,
            tensor_bytes,
            metadata_allowance_bytes,
        })
    }
    /// Packed/norm backing plus model metadata and actual owned Vec capacities.
    /// The E2B constructor includes its entire PayloadStore tensor capacity and
    /// retained metadata reservation. Other caller padding, tokenizer and process
    /// memory are outside this count.
    pub fn model_bytes(&self) -> usize {
        self.bytes
    }
    pub fn planned_session_bytes(&self) -> Result<usize, Error> {
        let l = &self.layout;
        let mut stages = 0;
        for g in &l.layers[..l.producers] {
            stages = add_bytes(
                stages,
                cpu::product(cpu::product(l.positions, g.head_dim)?, 2)?,
            )?;
        }
        let mut bytes = std::mem::size_of::<Session<'_, '_>>();
        for n in [
            self.plan.planned_cache_bytes()? - std::mem::size_of::<Vec<WindowKvCache>>(),
            total_bytes(&Scratch::counts(l)?)?,
            allocation_bytes(stages)?,
            cpu::product(l.producers, std::mem::size_of::<Staging>())?,
        ] {
            bytes = add_bytes(bytes, n)?;
        }
        Ok(bytes)
    }
    /// Admit model+session before allocating any numerical session buffers.
    pub fn new_session(&self, budget: usize) -> Result<Session<'_, 'a>, Error> {
        if budget < add_bytes(self.model_bytes(), self.planned_session_bytes()?)? {
            return Err(Error::InvalidConfig("Gemma packed memory budget"));
        }
        let l = &self.layout;
        let mut staging = Vec::new();
        staging
            .try_reserve_exact(l.producers)
            .map_err(|_| Error::Allocation)?;
        for g in &l.layers[..l.producers] {
            let n = cpu::product(l.positions, g.head_dim)?;
            staging.push(Staging {
                k: zeros(n)?,
                v: zeros(n)?,
            });
        }
        let result = Session {
            model: self,
            caches: self.plan.new_caches()?,
            staging,
            scratch: Scratch::new(l)?,
            history: [0; MAX_POSITIONS],
            committed: 0,
            published_rows: 0,
            #[cfg(test)]
            fault: None,
        };
        if add_bytes(self.model_bytes(), result.allocated_bytes()?)? > budget {
            return Err(Error::InvalidConfig("Gemma packed actual memory budget"));
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
    fn counts(l: &Layout) -> Result<[usize; 23], Error> {
        let h = cpu::product(l.positions, l.hidden)?;
        let q = cpu::product(cpu::product(l.positions, l.heads)?, l.max_dim())?;
        let kv = cpu::product(l.positions, l.max_dim())?;
        let ple = cpu::product(cpu::product(l.positions, l.layers.len())?, l.ple)?;
        let ff = cpu::product(l.positions, l.max_ff().max(l.ple))?;
        let scores = cpu::product(cpu::product(l.heads, l.positions)?, l.positions)?;
        Ok([
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
            cpu::product(l.positions, l.ple)?,
            h,
            h,
            ff,
            ff,
            ff,
            scores,
            scores,
            cpu::product(l.positions, l.vocab)?,
            cpu::product(l.positions, l.vocab)?,
        ])
    }
    fn new(l: &Layout) -> Result<Self, Error> {
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
        ] = Self::counts(l)?;
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
    staging: Vec<Staging>,
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
    /// All rows from the last successful chunk, `[rows,vocab]`, or empty after reset.
    pub fn logits(&self) -> &[f32] {
        &self.scratch.published[..self.published_rows * self.model.layout.vocab]
    }
    pub fn allocated_bytes(&self) -> Result<usize, Error> {
        let mut bytes = std::mem::size_of::<Self>();
        for n in [
            self.model.plan.allocated_cache_bytes(&self.caches)?
                - std::mem::size_of::<Vec<WindowKvCache>>(),
            total_bytes(&self.scratch.capacities())?,
            cpu::product(self.staging.capacity(), std::mem::size_of::<Staging>())?,
        ] {
            bytes = add_bytes(bytes, n)?;
        }
        for stage in &self.staging {
            bytes = add_bytes(
                bytes,
                total_bytes(&[stage.k.capacity(), stage.v.capacity()])?,
            )?;
        }
        Ok(bytes)
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
