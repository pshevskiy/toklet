//! Borrowed packed Gemma graph for a frozen witness and a named short E2B adapter.
//!
//! `Model::new` admits the short synthetic profile; an explicit scalar witness
//! constructor admits up to1025 positions. `Model::from_e2b` borrows
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
pub(crate) mod generation;
pub(crate) mod operations;
pub use generation::GenerationSession;
pub use operations::{BackendMemory, MatrixSource, Operations, Scalar};
#[cfg(test)]
mod tests;

pub const MAX_POSITIONS: usize = 32;
/// Explicit synthetic scalar context bound, independent of the32-row call bound.
pub const MAX_WITNESS_CONTEXT: usize = 1025;
pub const VOCAB_SIZE: usize = 17;
const EPSILON: f32 = 1e-6;
const MAX_PRODUCERS: usize = 15;

pub struct KvWeights<'a, M = Matrix<'a>> {
    pub k: M,
    pub v: M,
    pub k_norm: &'a [f32],
}
pub struct LayerWeights<'a, M = Matrix<'a>> {
    pub input_norm: &'a [f32],
    pub q: M,
    pub q_norm: &'a [f32],
    pub o: M,
    /// Only the producer prefix owns K/V projections.
    pub kv: Option<KvWeights<'a, M>>,
    pub post_attention_norm: &'a [f32],
    pub pre_ff_norm: &'a [f32],
    pub post_ff_norm: &'a [f32],
    pub ff_gate: M,
    pub ff_up: M,
    pub ff_down: M,
    pub ple_gate: M,
    pub ple_projection: M,
    pub post_ple_norm: &'a [f32],
    pub scalar: f32,
}
/// Immutable borrowed byte-backed matrices and direct learned FP32 norms.
/// The embedding is the tied output head and is counted only once. Tensor names
/// and byte spans are resolved before construction, never in numerical loops.
pub struct Weights<'a, M = Matrix<'a>> {
    pub embedding: M,
    pub per_layer_embedding: M,
    pub per_layer_projection: M,
    pub per_layer_norm: &'a [f32],
    pub layers: Vec<LayerWeights<'a, M>>,
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
pub struct Model<'a, M = Matrix<'a>> {
    weights: Weights<'a, M>,
    layout: Layout,
    plan: SharedKvPlan,
    unit_norm: Vec<f32>,
    bytes: usize,
    tensor_bytes: usize,
    metadata_allowance_bytes: usize,
    content_identity: Option<crate::mlx::gemma4::payload::ContentIdentity>,
}
fn check(values: &[f32], expected: usize, name: &'static str) -> Result<(), Error> {
    length(name, values.len(), expected)?;
    finite(name, values)?;
    Ok(())
}
fn matrix(value: &impl MatrixSource, rows: usize, columns: usize) -> Result<usize, Error> {
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
fn token_history(capacity: usize) -> Result<Vec<u32>, Error> {
    // Both f32 numerical elements and u32 token IDs occupy four bytes.
    allocation_bytes(capacity)?;
    let mut history = Vec::new();
    history
        .try_reserve_exact(capacity)
        .map_err(|_| Error::Allocation)?;
    history.resize(capacity, 0);
    Ok(history)
}
impl<'a> Model<'a> {
    pub fn new(weights: Weights<'a>) -> Result<Self, Error> {
        Self::with_layout(weights, Layout::witness())
    }
    /// Synthetic scalar diagnostic only; does not admit E2B or raise GPU limits.
    /// Sessions retain at most32 rows of per-call scratch independently of C.
    pub fn new_witness_with_context(weights: Weights<'a>, capacity: usize) -> Result<Self, Error> {
        if !(1..=MAX_WITNESS_CONTEXT).contains(&capacity) {
            return Err(Error::InvalidConfig("Gemma scalar witness context bound"));
        }
        let mut layout = Layout::witness();
        layout.positions = capacity;
        Self::with_layout(weights, layout)
    }
    fn with_layout(weights: Weights<'a>, l: Layout) -> Result<Self, Error> {
        Self::with_storage(weights, l, None)
    }
    pub fn planned_session_bytes(&self) -> Result<usize, Error> {
        self.planned_backend_session_bytes::<Scalar>()
    }
    pub fn new_session(&self, budget: usize) -> Result<Session<'_, 'a>, Error> {
        self.session_with_backend(Scalar, budget)
    }
    /// Plan bounded per-call scratch independently of the unchanged context.
    pub fn planned_session_bytes_with_chunk(&self, chunk_rows: usize) -> Result<usize, Error> {
        self.planned_backend_session_bytes_for_chunk::<Scalar>(chunk_rows)
    }
    /// Keep the same total context/cache bound, admitting at most chunk_rows per
    /// forward. All private/published logits and producer proposals share this bound.
    pub fn new_session_with_chunk(
        &self,
        chunk_rows: usize,
        budget: usize,
    ) -> Result<Session<'_, 'a>, Error> {
        self.session_with_backend_for_chunk(Scalar, chunk_rows, budget)
    }
    /// Plan actual context/cache/history and independent per-call scratch.
    pub fn planned_session_bytes_with_capacity(
        &self,
        capacity: usize,
        chunk_rows: usize,
    ) -> Result<usize, Error> {
        self.planned_backend_session_bytes_for_capacity::<Scalar>(capacity, chunk_rows)
    }
    pub fn new_session_with_capacity(
        &self,
        capacity: usize,
        chunk_rows: usize,
        budget: usize,
    ) -> Result<Session<'_, 'a>, Error> {
        self.session_with_backend_for_capacity(Scalar, capacity, chunk_rows, budget)
    }
}
/// Known backing is supplied only by the two admitted artifact owners. Includes
/// their real capacities once; no source loader scratch is retained as live data.
#[derive(Clone, Copy)]
pub(crate) struct Backing {
    pub tensor_bytes: usize,
    pub metadata_bytes: usize,
    pub metadata_allowance_bytes: usize,
}
impl Backing {
    fn cpu(report: &crate::mlx::gemma4::payload::LoadReport) -> Self {
        Self {
            tensor_bytes: report.tensor_capacity_bytes,
            metadata_bytes: 0,
            metadata_allowance_bytes: report.metadata_reservation_bytes,
        }
    }
}
impl<'a, M: MatrixSource> Model<'a, M> {
    fn with_storage(
        weights: Weights<'a, M>,
        l: Layout,
        retained: Option<Backing>,
    ) -> Result<Self, Error> {
        // A bounded proposal frame must never spill to the heap during forward.
        // Public admission selects only the frozen witness or named E2B geometry.
        if l.producers == 0
            || l.producers > MAX_PRODUCERS
            || l.producers > l.layers.len()
            || l.positions == 0
            || l.positions > MAX_WITNESS_CONTEXT
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
        let tensor_bytes = retained.map_or(bytes, |report| report.tensor_bytes);
        let metadata_allowance_bytes = retained.map_or(0, |report| report.metadata_allowance_bytes);
        if let Some(report) = retained {
            // Replace the borrowed logical tensor sum with its owner's actual
            // full capacity. This includes retained source scalar vectors and
            // padding, exactly once; copied LayerWeights scalars are metadata.
            if report.tensor_bytes < bytes {
                return Err(Error::InvalidConfig("Gemma retained backing capacity"));
            }
            bytes = add_bytes(
                add_bytes(report.tensor_bytes, report.metadata_bytes)?,
                report.metadata_allowance_bytes,
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
                std::mem::size_of::<LayerWeights<'_, M>>(),
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
            content_identity: None,
        })
    }
    // Cold adapter check; numerical forward does not search these handles.
    #[cfg(all(
        feature = "metal",
        any(target_os = "macos", target_os = "ios"),
        target_arch = "aarch64"
    ))]
    pub(crate) fn matrix_handles(&self) -> impl Iterator<Item = &M> {
        [
            &self.weights.embedding,
            &self.weights.per_layer_embedding,
            &self.weights.per_layer_projection,
        ]
        .into_iter()
        .chain(self.weights.layers.iter().flat_map(|w| {
            [
                &w.q,
                &w.o,
                &w.ff_gate,
                &w.ff_up,
                &w.ff_down,
                &w.ple_gate,
                &w.ple_projection,
            ]
            .into_iter()
            .chain(w.kv.iter().flat_map(|kv| [&kv.k, &kv.v]))
        }))
    }
    /// Exact storage content evidence, only for a model built from an admitted
    /// E2B PayloadStore. This grants no provenance/tokenizer/app capability.
    pub fn content_identity(&self) -> Option<&crate::mlx::gemma4::payload::ContentIdentity> {
        self.content_identity.as_ref()
    }
    /// Packed/norm backing plus model metadata and actual owned Vec capacities.
    /// The E2B constructor includes its entire PayloadStore tensor capacity and
    /// retained metadata reservation. Other caller padding, tokenizer and process
    /// memory are outside this count.
    pub fn model_bytes(&self) -> usize {
        self.bytes
    }
    pub(crate) fn planned_backend_session_bytes<B>(&self) -> Result<usize, Error> {
        self.planned_backend_session_bytes_for_chunk::<B>(self.layout.positions.min(MAX_POSITIONS))
    }
    pub(crate) fn planned_backend_session_bytes_for_chunk<B>(
        &self,
        chunk_rows: usize,
    ) -> Result<usize, Error> {
        self.planned_backend_session_bytes_for_capacity::<B>(self.layout.positions, chunk_rows)
    }
    pub(crate) fn planned_backend_session_bytes_for_capacity<B>(
        &self,
        capacity: usize,
        chunk_rows: usize,
    ) -> Result<usize, Error> {
        let l = &self.layout;
        let scratch = Scratch::counts(l, capacity, chunk_rows)?;
        let mut stages = 0;
        for g in &l.layers[..l.producers] {
            stages = add_bytes(
                stages,
                cpu::product(cpu::product(chunk_rows, g.head_dim)?, 2)?,
            )?;
        }
        let mut bytes = std::mem::size_of::<Session<'_, '_, M, B>>();
        for n in [
            self.plan.planned_narrowed_cache_bytes(capacity)?
                - std::mem::size_of::<Vec<WindowKvCache>>(),
            SharedKvPlan::planned_metadata_bytes(l.layers.len())?
                - std::mem::size_of::<SharedKvPlan>(),
            allocation_bytes(capacity)?,
            total_bytes(&scratch)?,
            allocation_bytes(stages)?,
            cpu::product(l.producers, std::mem::size_of::<Staging>())?,
        ] {
            bytes = add_bytes(bytes, n)?;
        }
        Ok(bytes)
    }
    /// Admit model+session before allocating any numerical session buffers.
    pub(crate) fn session_with_backend<B: Operations<M>>(
        &self,
        operations: B,
        budget: usize,
    ) -> Result<Session<'_, 'a, M, B>, B::Error> {
        self.session_with_backend_for_chunk(
            operations,
            self.layout.positions.min(MAX_POSITIONS),
            budget,
        )
    }
    pub(crate) fn session_with_backend_for_chunk<B: Operations<M>>(
        &self,
        operations: B,
        chunk_rows: usize,
        budget: usize,
    ) -> Result<Session<'_, 'a, M, B>, B::Error> {
        self.session_with_backend_for_capacity(
            operations,
            self.layout.positions,
            chunk_rows,
            budget,
        )
    }
    pub(crate) fn session_with_backend_for_capacity<B: Operations<M>>(
        &self,
        operations: B,
        capacity: usize,
        chunk_rows: usize,
        budget: usize,
    ) -> Result<Session<'_, 'a, M, B>, B::Error> {
        let numerical =
            self.planned_backend_session_bytes_for_capacity::<B>(capacity, chunk_rows)?;
        operations.ready()?;
        let backend = operations.memory()?.total()?;
        if budget < add_bytes(add_bytes(self.model_bytes(), numerical)?, backend)? {
            return Err(Error::InvalidConfig("Gemma packed memory budget").into());
        }
        let l = &self.layout;
        let plan = self.plan.narrowed(capacity)?;
        let mut staging = Vec::new();
        staging
            .try_reserve_exact(l.producers)
            .map_err(|_| Error::Allocation)?;
        for g in &l.layers[..l.producers] {
            let n = cpu::product(chunk_rows, g.head_dim)?;
            staging.push(Staging {
                k: zeros(n)?,
                v: zeros(n)?,
            });
        }
        let result = Session {
            model: self,
            operations,
            capacity,
            chunk_rows,
            caches: plan.new_caches()?,
            plan,
            staging,
            scratch: Scratch::new(l, capacity, chunk_rows)?,
            history: token_history(capacity)?,
            committed: 0,
            published_rows: 0,
            #[cfg(test)]
            fault: None,
        };
        if add_bytes(self.model_bytes(), result.allocated_bytes()?)? > budget {
            return Err(Error::InvalidConfig("Gemma packed actual memory budget").into());
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
    fn counts(l: &Layout, capacity: usize, chunk_rows: usize) -> Result<[usize; 23], Error> {
        cpu::nonzero(capacity, "Gemma context capacity")?;
        cpu::nonzero(chunk_rows, "Gemma chunk capacity")?;
        if capacity > l.positions {
            return Err(Error::InvalidConfig(
                "Gemma capacity exceeds executable model positions",
            ));
        }
        if chunk_rows > capacity || chunk_rows > MAX_POSITIONS {
            return Err(Error::InvalidConfig("Gemma packed chunk capacity"));
        }
        let h = cpu::product(chunk_rows, l.hidden)?;
        let q = cpu::product(cpu::product(chunk_rows, l.heads)?, l.max_dim())?;
        let kv = cpu::product(chunk_rows, l.max_dim())?;
        let ple = cpu::product(cpu::product(chunk_rows, l.layers.len())?, l.ple)?;
        let ff = cpu::product(chunk_rows, l.max_ff().max(l.ple))?;
        let scores = cpu::product(cpu::product(l.heads, chunk_rows)?, capacity)?;
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
            cpu::product(chunk_rows, l.ple)?,
            h,
            h,
            ff,
            ff,
            ff,
            scores,
            scores,
            cpu::product(chunk_rows, l.vocab)?,
            cpu::product(chunk_rows, l.vocab)?,
        ])
    }
    fn new(l: &Layout, capacity: usize, chunk_rows: usize) -> Result<Self, Error> {
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
        ] = Self::counts(l, capacity, chunk_rows)?;
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
pub struct Session<'model, 'weights, M = Matrix<'weights>, B = Scalar> {
    model: &'model Model<'weights, M>,
    operations: B,
    capacity: usize,
    chunk_rows: usize,
    plan: SharedKvPlan,
    caches: Vec<WindowKvCache>,
    staging: Vec<Staging>,
    scratch: Scratch,
    history: Vec<u32>,
    committed: usize,
    published_rows: usize,
    #[cfg(test)]
    fault: Option<usize>,
}

impl<M: MatrixSource, B: Operations<M>> Session<'_, '_, M, B> {
    /// Maximum rows admitted per forward; independent of total context positions.
    pub fn chunk_capacity(&self) -> usize {
        self.chunk_rows
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn tokens(&self) -> &[u32] {
        &self.history[..self.committed]
    }
    /// All rows from the last successful chunk, `[rows,vocab]`, or empty after reset.
    pub fn logits(&self) -> &[f32] {
        &self.scratch.published[..self.published_rows * self.model.layout.vocab]
    }
    pub fn allocated_bytes(&self) -> Result<usize, Error> {
        let mut bytes = add_bytes(
            std::mem::size_of::<Self>(),
            self.operations.memory()?.total()?,
        )?;
        for n in [
            self.plan.allocated_cache_bytes(&self.caches)?
                - std::mem::size_of::<Vec<WindowKvCache>>(),
            self.plan.allocated_metadata_bytes() - std::mem::size_of::<SharedKvPlan>(),
            allocation_bytes(self.history.capacity())?,
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
    pub fn reset(&mut self) -> Result<(), B::Error> {
        self.operations.ready()?;
        reset_batch(&mut self.caches)?;
        self.committed = 0;
        self.published_rows = 0;
        self.history.fill(0);
        Ok(())
    }
    /// Append within the admitted context and chunk bounds. Failure preserves all
    /// committed KV/history and the prior public logits; scratch is discardable.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<&[f32], B::Error> {
        forward::run(self, tokens)?;
        Ok(self.logits())
    }
}
