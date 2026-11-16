//! Explicit 13 -> 14 -> 15 resident qualification, one transaction owner.
//! No ordinary model/session/CLI route selects this cfg(test) slice.
use super::*;
use crate::metal::gemma_affine::resident::bindings::ChainLayerSlots;
use project::bf16_tiled::{Bf16MatrixView, StridedBf16Pipelines};
use std::time::Instant;
mod kv;
mod layout;
mod reference;
mod timing;
use kv::CacheStorage;
use layout::{LayerLayout, LayerStorage, sentinel, tail, validate_storage};
const LAYERS: [usize; 3] = [13, 14, 15];
const POSITIONS: usize = 512;
const MAX_RMS_REASONS: usize = 32 * 15;
static NEXT_SESSION: AtomicU64 = AtomicU64::new(1);
fn identity() -> Result<u64> {
    NEXT_SESSION
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |x| x.checked_add(1))
        .map_err(|_| Error::Options("chain owner identity exhausted"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub(super) enum OutputRows {
    All,
    Last,
}
#[derive(Clone, Copy)]
pub(super) struct Options {
    pub capacity: usize,
    pub max_rows: usize,
}
impl Options {
    fn validate(self) -> Result<()> {
        if !(1..=POSITIONS).contains(&self.capacity)
            || !(1..=32.min(self.capacity)).contains(&self.max_rows)
        {
            return Err(Error::Options(
                "chain capacity/chunk outside admitted bounds",
            ));
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub(super) struct NodeFrame {
    pub rows: usize,
    pub query_start: usize,
    pub key_start: usize,
    pub keys: usize,
    pub proposal_rows: usize,
    pub proposal_start: usize,
}
#[derive(Clone, Copy)]
struct GraphFrame {
    owner: u64,
    epoch: u64,
    offset: usize,
    end: usize,
    rows: usize,
    mode: OutputRows,
    cache_owners: [u64; 2],
    cache_generations: [u64; 2],
    history: [u32; POSITIONS],
    nodes: [NodeFrame; 3],
}
#[derive(Clone, Copy)]
enum LayerKind {
    Producer { cache: usize },
    Consumer { donor: usize },
}
struct ResidentLayerPlan<'w> {
    slots: ChainLayerSlots,
    kind: LayerKind,
    layout: LayerLayout,
    matrix_slots: [Option<usize>; 9],
    norm_slots: [Option<usize>; 7],
    matrices: [Option<Bf16MatrixView<'w>>; 9],
    admissions: [Bf16Tile<'w>; 6],
    norm_regions: [Option<Region>; 8],
    scalar: f32,
}
pub(super) struct ResidentGraph<'w> {
    source: &'w Bf16Weights,
    context: &'w Bf16Context,
    pipelines: &'w Pipelines,
    table: &'w gelu::CertifiedTable,
    options: Options,
    plans: [ResidentLayerPlan<'w>; 3],
    norms: [Buffer<u32>; 3],
    rope: [Buffer<u32>; 2],
    strided: StridedBf16Pipelines,
    precision_rms: rms::precision::Pipelines,
}
fn norm_words(d: usize, producer: bool) -> usize {
    5 * H
        + if producer {
            2 * d + 1 + 9 * GUARD
        } else {
            d + 7 * GUARD
        }
}
impl<'w> ResidentGraph<'w> {
    pub fn planned_bytes(options: Options) -> Result<usize> {
        options.validate()?;
        let words = norm_words(256, true)
            + norm_words(512, true)
            + norm_words(256, false)
            + options.capacity * (256 + 128)
            + 2 * GUARD;
        words
            .checked_mul(4)
            .and_then(|n| n.checked_add(size_of::<Self>()))
            .and_then(|n| n.checked_add(StridedBf16Pipelines::policy_bytes()))
            .and_then(|n| n.checked_add(rms::precision::POLICY_BYTES))
            .ok_or(Error::Options("chain graph byte overflow"))
    }
    pub fn new(
        source: &'w Bf16Weights,
        context: &'w Bf16Context,
        pipelines: &'w Pipelines,
        table: &'w gelu::CertifiedTable,
        options: Options,
        available: usize,
    ) -> Result<Self> {
        check_budget(Self::planned_bytes(options)?, available)?;
        if !std::ptr::eq(&*source.device, &*context.0.gpu.device)
            || !std::ptr::eq(&*pipelines.device, &*context.0.gpu.device)
        {
            return Err(Error::Options("chain foreign shared owner device"));
        }
        table.validate_device(&context.0.gpu)?;
        pipelines.gelu.validate_table(table)?;
        let (a, na) = Self::layer(source, context, 13, options)?;
        let (b, nb) = Self::layer(source, context, 14, options)?;
        let (c, nc) = Self::layer(source, context, 15, options)?;
        let make_rope = |d, rotated, base| -> Result<Buffer<u32>> {
            let mut buffer = Buffer::new(&context.0.gpu, options.capacity * rotated + GUARD)?;
            buffer.with_mut(|s| -> Result<()> {
                s.fill(SENTINEL);
                for offset in (0..options.capacity).step_by(32) {
                    let rows = (options.capacity - offset).min(32);
                    rotary::prepare_coefficients(
                        &cpu::ProportionalRotaryShape {
                            tokens: rows,
                            heads: 1,
                            head_dim: d,
                            rotated_dims: rotated,
                            offset,
                            base,
                            factor: 1.,
                        },
                        &mut s[offset * rotated..(offset + rows) * rotated],
                    )?;
                }
                Ok(())
            })?;
            Ok(buffer)
        };
        let rope = [make_rope(256, 256, 10000.)?, make_rope(512, 128, 1000000.)?];
        let strided = StridedBf16Pipelines::new(&context.0, StridedBf16Pipelines::policy_bytes())?;
        let precision_rms = rms::precision::Pipelines::new(
            &context.0.gpu,
            &pipelines.rms,
            rms::precision::POLICY_BYTES,
        )?;
        let graph = Self {
            source,
            context,
            pipelines,
            table,
            options,
            plans: [a, b, c],
            norms: [na, nb, nc],
            rope,
            strided,
            precision_rms,
        };
        check_budget(graph.admitted_bytes(), available)?;
        Ok(graph)
    }
    fn layer(
        source: &'w Bf16Weights,
        context: &Bf16Context,
        index: usize,
        options: Options,
    ) -> Result<(ResidentLayerPlan<'w>, Buffer<u32>)> {
        let slots = source.bindings.chain_layer_slots(index)?;
        let producer = index != 15;
        let d = if index == 14 { 512 } else { 256 };
        let ff = if producer { 6144 } else { 12288 };
        if !LAYERS.contains(&index)
            || slots.geometry.head_dim != d
            || slots.geometry.rotated_dims != if d == 512 { 128 } else { 256 }
            || slots.geometry.intermediate != ff
            || slots.geometry.producer != if producer { index } else { 13 }
            || slots.kv.is_some() != producer
            || slots.geometry.kind
                != if d == 512 {
                    crate::model::shared_kv::KvKind::Full
                } else {
                    crate::model::shared_kv::KvKind::Sliding
                }
        {
            return Err(Error::Options("chain original source geometry differs"));
        }
        let m = slots.matrices;
        let mut matrix_slots = [
            Some(m[0]),
            None,
            None,
            Some(m[1]),
            Some(m[2]),
            Some(m[3]),
            Some(m[4]),
            Some(m[5]),
            Some(m[6]),
        ];
        let n = slots.norms;
        let mut norm_slots = [
            Some(n[0]),
            Some(n[1]),
            None,
            Some(n[2]),
            Some(n[3]),
            Some(n[4]),
            Some(n[5]),
        ];
        if let Some(kv) = slots.kv {
            matrix_slots[1] = Some(kv.k);
            matrix_slots[2] = Some(kv.v);
            norm_slots[2] = Some(kv.norm);
        }
        let expected = [
            (8 * d, H),
            (d, H),
            (d, H),
            (H, 8 * d),
            (ff, H),
            (ff, H),
            (H, ff),
            (P, H),
            (H, P),
        ];
        let mut matrices = [None; 9];
        for i in 0..9 {
            if let Some(slot) = matrix_slots[i] {
                let matrix = source
                    .matrices
                    .get(slot)
                    .and_then(Option::as_ref)
                    .ok_or(Error::Options("chain original matrix role absent"))?;
                if (matrix.rows, matrix.columns) != expected[i] {
                    return Err(Error::Options("chain original matrix shape differs"));
                }
                matrices[i] = Some(matrix.borrowed_tiled(context)?);
            }
        }
        let tile = |i: usize| -> Result<Bf16Tile<'w>> {
            Ok(matrices[i]
                .ok_or(Error::Options("chain admission matrix absent"))?
                .admission_tile())
        };
        let q = if producer {
            strongest_admission_tile(&[tile(0)?, tile(1)?, tile(2)?])?
        } else {
            tile(0)?
        };
        let ff_group = strongest_admission_tile(&[tile(4)?, tile(5)?])?;
        let admissions = [q, tile(3)?, ff_group, tile(6)?, tile(7)?, tile(8)?];
        let mut norm_regions = [None; 8];
        let mut cursor = GUARD;
        let widths = [H, d, d, H, H, H, H, 1];
        for (i, &len) in widths.iter().enumerate() {
            if i < 7 && norm_slots[i].is_some() || i == 7 && producer {
                norm_regions[i] = Some(Region { start: cursor, len });
                cursor += len + GUARD;
            }
        }
        if cursor != norm_words(d, producer) {
            return Err(Error::Options("chain normalization plan size differs"));
        }
        let mut norms = Buffer::new(&context.0.gpu, cursor)?;
        for i in 0..7 {
            if let Some(slot) = norm_slots[i] {
                let values = source
                    .vectors
                    .get(slot)
                    .ok_or(Error::Options("chain normalization source absent"))?;
                if values.len() != widths[i] || values.iter().any(|x| !x.is_finite()) {
                    return Err(Error::Options("chain normalization source shape differs"));
                }
            }
        }
        norms.with_mut(|out| {
            out.fill(SENTINEL);
            for i in 0..7 {
                if let Some(slot) = norm_slots[i] {
                    let region = norm_regions[i].unwrap();
                    for (dst, value) in out[region.start..region.end()]
                        .iter_mut()
                        .zip(&source.vectors[slot])
                    {
                        *dst = value.to_bits();
                    }
                }
            }
            if let Some(unit) = norm_regions[7] {
                out[unit.start] = 1_f32.to_bits();
            }
        });
        let scalar = source
            .vectors
            .get(slots.scalar)
            .filter(|v| v.len() == 1 && v[0].is_finite())
            .ok_or(Error::Options("chain learned scalar source differs"))?[0];
        let layout = LayerLayout::new(options.max_rows, options.capacity, d, ff, producer)?;
        let kind = match index {
            13 => LayerKind::Producer { cache: 0 },
            14 => LayerKind::Producer { cache: 1 },
            15 => LayerKind::Consumer { donor: 0 },
            _ => unreachable!(),
        };
        Ok((
            ResidentLayerPlan {
                slots,
                kind,
                layout,
                matrix_slots,
                norm_slots,
                matrices,
                admissions,
                norm_regions,
                scalar,
            },
            norms,
        ))
    }
    pub fn owned_bytes(&self) -> usize {
        size_of::<Self>()
            + self
                .norms
                .iter()
                .chain(&self.rope)
                .map(|b| b.len * 4)
                .sum::<usize>()
    }
    pub fn admitted_bytes(&self) -> usize {
        self.owned_bytes() + StridedBf16Pipelines::policy_bytes() + rms::precision::POLICY_BYTES
    }
    fn node_frames(&self, offset: usize, rows: usize, mode: OutputRows) -> [NodeFrame; 3] {
        let end = offset + rows;
        std::array::from_fn(|node| {
            let last = node == 2 && mode == OutputRows::Last;
            let window = if node == 1 { None } else { Some(512) };
            let start = window_rows::view_start(offset, window);
            NodeFrame {
                rows: if last { 1 } else { rows },
                query_start: if last { end - 1 } else { offset },
                key_start: start,
                keys: end - start,
                proposal_rows: rows,
                proposal_start: offset,
            }
        })
    }
    fn layer_recipe<'a>(
        &'a self,
        storage: &'a [LayerStorage; 3],
        caches: &'a [CacheStorage; 2],
        frame: &GraphFrame,
        node: usize,
        force_reference: bool,
    ) -> Result<recipe::Recipe<'a>> {
        let plan = &self.plans[node];
        let f = frame.nodes[node];
        let layer = &storage[node];
        plan.layout.check_frame(f)?;
        let (cache, donor) = match plan.kind {
            LayerKind::Producer { cache } => (cache, node),
            LayerKind::Consumer { donor } => (donor, donor),
        };
        let donor_frame = frame.nodes[donor];
        let donor_layout = self.plans[donor].layout;
        let k = donor_layout.range(&storage[donor].arena, State::KProposal, donor_frame)?;
        let v = donor_layout.range(&storage[donor].arena, State::VProposal, donor_frame)?;
        let segments = caches[cache].segments(f, k, v)?;
        let mut norms = [None; 8];
        for (i, r) in plan.norm_regions.iter().enumerate() {
            if let Some(r) = r {
                norms[i] = Some(r.range(&self.norms[node], r.len)?);
            }
        }
        let rope = if plan.layout.d == 512 { 1 } else { 0 };
        let rotated = plan.slots.geometry.rotated_dims;
        let operation = recipe::Recipe {
            context: self.context,
            pipelines: self.pipelines,
            table: self.table,
            projections: plan.matrices.map(|m| {
                m.map(|matrix| recipe::Projection::Tiled {
                    matrix,
                    pipeline: &self.strided,
                })
            }),
            admissions: plan.admissions,
            states: plan.layout.states(&layer.arena, f)?,
            norms,
            norm_kind: recipe::NormKind::Precision {
                pipelines: &self.precision_rms,
                reasons: plan.layout.rms_reasons(&layer.rms_reasons, f)?,
            },
            status: BufferRange::new(&layer.status, 0, STATUS_WORDS)?,
            coefficients: BufferRange::new(
                &self.rope[rope],
                f.query_start * rotated,
                f.rows * rotated,
            )?,
            flags: [
                BufferRange::new(&layer.flags[0], 0, f.rows * plan.layout.ff)?,
                BufferRange::new(&layer.flags[1], 0, f.rows * P)?,
            ],
            intervals: [
                BufferRange::new(&layer.intervals[0], 0, 8 * f.rows * plan.layout.ff)?,
                BufferRange::new(&layer.intervals[1], 0, 8 * f.rows * P)?,
            ],
            segments,
            shape: recipe::Shape {
                rows: f.rows,
                d: plan.layout.d,
                ff: plan.layout.ff,
                rotated,
                query_start: f.query_start,
                key_start: f.key_start,
                keys: f.keys,
                window: caches[cache].window,
                theta: if plan.layout.d == 512 {
                    1000000.
                } else {
                    10000.
                },
                producer: plan.layout.producer,
            },
            scalar: plan.scalar,
            dispatches: Cell::new(0),
            partial_count_unknown: Cell::new(false),
            force_gelu_reference: force_reference,
        };
        Ok(operation)
    }
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub(super) struct Metrics {
    pub epoch: u64,
    pub command_identity: usize,
    // Native properties of this already completed candidate command only.
    // Caller work may share that command; later reference commands are excluded.
    pub candidate_gpu_start_seconds: Option<f64>,
    pub candidate_gpu_end_seconds: Option<f64>,
    pub candidate_completed_gpu_seconds: Option<f64>,
    pub candidate_gpu_timing_completed: bool,
    pub candidate_gpu_timing_epoch: Option<u64>,
    pub candidate_gpu_timing_command_identity: Option<usize>,
    pub candidate_gpu_timing_unavailable_reason: Option<&'static str>,
    pub candidate_gpu_timing_scope: Option<&'static str>,
    pub encoding_complete: bool,
    pub encoded_dispatches: Option<usize>,
    pub blit_copies: usize,
    pub candidate_commands: usize,
    pub candidate_waits: usize,
    pub replay_count: usize,
    pub replay_node: Option<usize>,
    pub replay_slot: Option<usize>,
    pub reference_matrix_operations: usize,
    pub reference_commands: Option<usize>,
    pub reference_waits: Option<usize>,
    pub reference_input_bytes: Option<usize>,
    pub reference_readback_bytes: Option<usize>,
    pub input_upload_bytes: usize,
    pub device_checkpoint_copy_bytes: usize,
    pub checkpoint_readback_bytes: usize,
    pub reference_upload_bytes: usize,
    pub new_kv_copy_bytes: usize,
    pub output_copy_bytes: usize,
    pub diagnostic_bytes: usize,
    pub validation_span_bytes: usize,
    pub owner_bytes: usize,
    pub shared_graph_bytes: usize,
    pub shared_weights_bytes: usize,
    pub shared_table_bytes: usize,
    pub shared_pipeline_policy_bytes: usize,
    pub encode_seconds: f64,
    pub wait_seconds: f64,
    pub complete_seconds: f64,
    pub publication_seconds: f64,
}
impl Metrics {
    fn record_recipe_encoding(
        &mut self,
        result: &Result<()>,
        count: usize,
        partial_count_unknown: bool,
    ) {
        // An encoder can fail after dispatching without updating its recorder.
        self.encoded_dispatches = if result.is_err() || partial_count_unknown {
            None
        } else {
            self.encoded_dispatches.map(|n| n + count)
        };
    }
}
pub(super) struct ResidentSession<'g, 'w> {
    graph: &'g ResidentGraph<'w>,
    storage: [LayerStorage; 3],
    caches: [CacheStorage; 2],
    entry: Buffer<u32>,
    entry_regions: [Region; 4],
    reference: reference::Runner,
    history: [u32; POSITIONS],
    committed: usize,
    published: Vec<f32>,
    published_rows: usize,
    owner: u64,
    epoch: u64,
    busy: bool,
    poisoned: bool,
    metrics: Metrics,
    diagnostic_bytes: Cell<usize>,
    force_reference_node: Option<usize>,
}
impl<'g, 'w> ResidentSession<'g, 'w> {
    fn entry_plan(rows: usize) -> ([Region; 4], usize) {
        let mut cursor = GUARD;
        let regions = [H, P, P, P].map(|width| {
            let r = Region {
                start: cursor,
                len: rows * width,
            };
            cursor = r.end() + GUARD;
            r
        });
        (regions, cursor)
    }
    pub fn planned_bytes(graph: &ResidentGraph<'_>) -> Result<usize> {
        let mut bytes = size_of::<Self>();
        for p in &graph.plans {
            bytes = add(
                bytes,
                LayerStorage::planned_bytes(p.layout)? - size_of::<LayerStorage>(),
            )?;
        }
        bytes = add(
            bytes,
            CacheStorage::planned_bytes(256, 512)? - size_of::<CacheStorage>(),
        )?;
        bytes = add(
            bytes,
            CacheStorage::planned_bytes(512, graph.options.capacity)? - size_of::<CacheStorage>(),
        )?;
        bytes = add(bytes, Self::entry_plan(graph.options.max_rows).1 * 4)?;
        bytes = add(bytes, graph.options.max_rows * H * 4)?;
        add(bytes, reference::Runner::planned_bytes(graph)?)
    }
    pub fn new(graph: &'g ResidentGraph<'w>, available: usize) -> Result<Self> {
        check_budget(Self::planned_bytes(graph)?, available)?;
        let gpu = &graph.context.0.gpu;
        let storage = [
            LayerStorage::new(gpu, graph.plans[0].layout)?,
            LayerStorage::new(gpu, graph.plans[1].layout)?,
            LayerStorage::new(gpu, graph.plans[2].layout)?,
        ];
        let caches = [
            CacheStorage::new(gpu, 256, 512, Some(512), identity()?)?,
            CacheStorage::new(gpu, 512, graph.options.capacity, None, identity()?)?,
        ];
        let (entry_regions, words) = Self::entry_plan(graph.options.max_rows);
        let mut entry = Buffer::new(gpu, words)?;
        entry.with_mut(|s| s.fill(SENTINEL));
        let reference = reference::Runner::new(graph)?;
        let published = reference::zero(graph.options.max_rows * H)?;
        let mut session = Self {
            graph,
            storage,
            caches,
            entry,
            entry_regions,
            reference,
            history: [0; POSITIONS],
            committed: 0,
            published,
            published_rows: 0,
            owner: identity()?,
            epoch: 0,
            busy: false,
            poisoned: false,
            metrics: Metrics {
                candidate_gpu_timing_unavailable_reason: Some(timing::NOT_OBSERVED),
                ..Metrics::default()
            },
            diagnostic_bytes: Cell::new(0),
            force_reference_node: None,
        };
        for storage in &mut session.storage {
            storage.reset();
        }
        check_budget(session.owned_bytes(), available)?;
        Ok(session)
    }
    pub fn owned_bytes(&self) -> usize {
        size_of::<Self>()
            + self
                .storage
                .iter()
                .map(|s| s.bytes() - size_of::<LayerStorage>())
                .sum::<usize>()
            + self
                .caches
                .iter()
                .map(|c| c.bytes() - size_of::<CacheStorage>())
                .sum::<usize>()
            + self.entry.len * 4
            + self.reference.allocation_bytes()
            + self.published.capacity() * 4
    }
    pub fn committed(&self) -> usize {
        self.committed
    }
    pub fn prefix(&self) -> &[u32] {
        &self.history[..self.committed]
    }
    pub fn output(&self) -> Option<&[f32]> {
        (!self.busy && !self.poisoned && self.published_rows != 0)
            .then_some(&self.published[..self.published_rows * H])
    }
    fn observed_metrics(&self) -> Metrics {
        let mut metrics = self.metrics;
        metrics.diagnostic_bytes = self.diagnostic_bytes.get();
        metrics
    }
    pub fn metrics(&self) -> Result<Metrics> {
        if self.poisoned || self.busy {
            Err(Error::Options(
                "chain diagnostics unavailable after failure/pending work",
            ))
        } else {
            Ok(self.observed_metrics())
        }
    }
    pub fn reset(&mut self) -> Result<()> {
        if self.busy || self.poisoned {
            return Err(Error::Options("chain cannot reset pending/poisoned owner"));
        }
        let generations = [
            self.caches[0].generation.checked_add(1),
            self.caches[1].generation.checked_add(1),
        ];
        if generations.iter().any(Option::is_none) {
            return Err(Error::Options("chain reset generation exhausted"));
        }
        for (c, g) in self.caches.iter_mut().zip(generations) {
            c.offset = 0;
            c.generation = g.unwrap();
        }
        self.committed = 0;
        self.history.fill(0);
        self.published_rows = 0;
        Ok(())
    }
    /// Qualification imports an independently authenticated causal prefix;
    /// this does not create a public cache-resume/model route.
    pub fn seed_prefix(&mut self, prefix: &[u32], kv: [(&[f32], &[f32]); 2]) -> Result<()> {
        if self.busy
            || self.poisoned
            || self.committed != 0
            || prefix.is_empty()
            || prefix.len() > self.graph.options.capacity
            || prefix.iter().any(|&x| x >= 262144)
        {
            return Err(Error::Options("chain prefix import differs"));
        }
        let generations = [
            self.caches[0].generation.checked_add(1),
            self.caches[1].generation.checked_add(1),
        ];
        if generations.iter().any(Option::is_none) {
            return Err(Error::Options("chain prefix import generation exhausted"));
        }
        for (i, (k, v)) in kv.iter().enumerate() {
            let n = prefix.len() * self.caches[i].d;
            if k.len() != n || v.len() != n || k.iter().chain(v.iter()).any(|x| !x.is_finite()) {
                return Err(Error::Options("chain imported KV extent/nonfinite differs"));
            }
        }
        for (i, (k, v)) in kv.into_iter().enumerate() {
            let cache = &mut self.caches[i];
            cache.k.with_mut(|s| {
                for (dst, x) in s.iter_mut().zip(k) {
                    *dst = x.to_bits();
                }
            });
            cache.v.with_mut(|s| {
                for (dst, x) in s.iter_mut().zip(v) {
                    *dst = x.to_bits();
                }
            });
            cache.offset = prefix.len();
            cache.generation = generations[i].unwrap();
        }
        self.history[..prefix.len()].copy_from_slice(prefix);
        self.committed = prefix.len();
        Ok(())
    }
    fn frame(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        prefix: &[u32],
        mode: OutputRows,
        cancel: &AtomicBool,
    ) -> Result<GraphFrame> {
        if self.busy
            || self.poisoned
            || cancel.load(Ordering::Acquire)
            || prefix.len() <= self.committed
            || prefix.len() > self.graph.options.capacity
            || prefix[..self.committed] != self.history[..self.committed]
            || prefix.iter().any(|&x| x >= 262144)
        {
            return Err(Error::Options(
                "chain unavailable/canceled or exact prefix differs",
            ));
        }
        let rows = prefix.len() - self.committed;
        if rows > self.graph.options.max_rows {
            return Err(Error::Options("chain new rows exceed admission"));
        }
        let epoch = self
            .epoch
            .checked_add(1)
            .ok_or(Error::Options("chain epoch exhausted"))?;
        let mut history = [0; POSITIONS];
        history[..prefix.len()].copy_from_slice(prefix);
        for cache in &self.caches {
            if cache.offset != self.committed || cache.generation.checked_add(1).is_none() {
                return Err(Error::Options(
                    "chain committed cache prefix/generation differs",
                ));
            }
        }
        self.validate_storage()?;
        crate::metal::memory::validate_encode_ranges(
            command,
            &self.graph.context.0.gpu.device,
            &[],
            0,
        )?;
        let frame = GraphFrame {
            owner: self.owner,
            epoch,
            offset: self.committed,
            end: prefix.len(),
            rows,
            mode,
            cache_owners: self.caches.each_ref().map(|c| c.owner),
            cache_generations: self.caches.each_ref().map(|c| c.generation),
            history,
            nodes: self.graph.node_frames(self.committed, rows, mode),
        };
        for node in 0..3 {
            self.graph
                .layer_recipe(&self.storage, &self.caches, &frame, node, false)?
                .preflight(command)?;
        }
        Ok(frame)
    }
    fn validate_storage(&self) -> Result<()> {
        let gpu = &self.graph.context.0.gpu;
        for (s, p) in self.storage.iter().zip(&self.graph.plans) {
            s.validate(gpu, p.layout)?;
        }
        self.caches[0].validate(gpu, 256, 512, Some(512))?;
        self.caches[1].validate(gpu, 512, self.graph.options.capacity, None)?;
        validate_storage(
            gpu,
            &self.entry,
            Self::entry_plan(self.graph.options.max_rows).1,
        )?;
        if self.published.len() != self.graph.options.max_rows * H {
            return Err(Error::Options("chain published storage differs"));
        }
        Ok(())
    }
    fn stage(
        &mut self,
        frame: &GraphFrame,
        host_upload: bool,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
    ) {
        self.busy = true;
        self.poisoned = true;
        self.epoch = frame.epoch;
        self.diagnostic_bytes.set(0);
        self.entry.with_mut(|s| s.fill(SENTINEL));
        for node in &mut self.storage {
            node.reset();
        }
        self.metrics = Metrics {
            epoch: frame.epoch,
            command_identity: std::ptr::from_ref(command).cast::<()>() as usize,
            candidate_gpu_timing_unavailable_reason: Some(timing::NOT_OBSERVED),
            encoded_dispatches: Some(0),
            reference_commands: Some(0),
            reference_waits: Some(0),
            reference_input_bytes: Some(0),
            reference_readback_bytes: Some(0),
            input_upload_bytes: if host_upload {
                frame.rows * (H + 3 * P) * 4
            } else {
                0
            },
            owner_bytes: self.owned_bytes(),
            shared_graph_bytes: self.graph.owned_bytes(),
            shared_weights_bytes: self.graph.source.report.retained_bytes,
            shared_table_bytes: self.graph.table.owned_bytes(),
            shared_pipeline_policy_bytes: PIPELINE_POLICY_BYTES
                + StridedBf16Pipelines::policy_bytes()
                + rms::precision::POLICY_BYTES,
            ..Metrics::default()
        };
    }
    pub fn begin<'s>(
        &'s mut self,
        prefix: &[u32],
        hidden: &[f32],
        ple: [&[f32]; 3],
        mode: OutputRows,
        cancel: &AtomicBool,
    ) -> Result<GraphTransaction<'s, 'g, 'w>> {
        let command = self.graph.context.0.gpu.command()?;
        let frame = self.frame(&command, prefix, mode, cancel)?;
        let inputs = [hidden, ple[0], ple[1], ple[2]];
        for (i, input) in inputs.iter().enumerate() {
            if input.len() != frame.rows * if i == 0 { H } else { P }
                || input.iter().any(|x| !x.is_finite())
            {
                return Err(Error::Options(
                    "chain host checkpoint extent/nonfinite differs",
                ));
            }
        }
        self.stage(&frame, true, &command);
        self.entry.with_mut(|words| {
            for (i, input) in inputs.into_iter().enumerate() {
                let region = self.entry_regions[i];
                for (dst, value) in words[region.start..region.start + input.len()]
                    .iter_mut()
                    .zip(input)
                {
                    *dst = value.to_bits();
                }
            }
        });
        Ok(GraphTransaction {
            session: self,
            command,
            frame,
            encoding_started: false,
            encoded: false,
            validated: false,
            published: false,
            resident_sources: None,
            status_snapshot: [[0; STATUS_WORDS]; 3],
            rms_snapshot: [[0; MAX_RMS_REASONS]; 3],
        })
    }
    pub fn begin_resident<'s>(
        &'s mut self,
        command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        prefix: &[u32],
        inputs: [BufferRange<'s>; 4],
        mode: OutputRows,
        cancel: &AtomicBool,
    ) -> Result<GraphTransaction<'s, 'g, 'w>> {
        let frame = self.frame(command, prefix, mode, cancel)?;
        for (i, &input) in inputs.iter().enumerate() {
            let width = if i == 0 { H } else { P };
            if input.len != frame.rows * width
                || input
                    .offset
                    .checked_add(input.len)
                    .and_then(|n| n.checked_mul(4))
                    .is_none_or(|end| end > input.raw.length())
                || input.raw.storageMode() != MTLStorageMode::Shared
                || input.raw.hazardTrackingMode() != MTLHazardTrackingMode::Tracked
            {
                return Err(Error::Options("chain resident checkpoint range differs"));
            }
            let check = |b: &Buffer<u32>| -> Result<()> {
                crate::metal::memory::validate_encode_ranges(
                    command,
                    &self.graph.context.0.gpu.device,
                    &[input, BufferRange::new(b, 0, b.len)?],
                    1,
                )
            };
            check(&self.entry)?;
            for storage in &self.storage {
                for b in storage.buffers() {
                    check(b)?;
                }
            }
            for cache in &self.caches {
                check(&cache.k)?;
                check(&cache.v)?;
            }
        }
        self.stage(&frame, false, command);
        let mut tx = GraphTransaction {
            session: self,
            command: command.clone(),
            frame,
            encoding_started: false,
            encoded: false,
            validated: false,
            published: false,
            resident_sources: Some(inputs),
            status_snapshot: [[0; STATUS_WORDS]; 3],
            rms_snapshot: [[0; MAX_RMS_REASONS]; 3],
        };
        tx.copy_external_checkpoint()?;
        Ok(tx)
    }
    fn validate_frame(&self, frame: &GraphFrame) -> Result<()> {
        self.validate_storage()?;
        if !self.busy
            || frame.owner != self.owner
            || frame.epoch != self.epoch
            || frame.offset != self.committed
            || frame.rows == 0
            || frame.rows > self.graph.options.max_rows
            || frame.offset.checked_add(frame.rows) != Some(frame.end)
            || frame.end > self.graph.options.capacity
            || frame.history[..frame.offset] != self.history[..frame.offset]
            || frame.history[..frame.end].iter().any(|&x| x >= 262144)
            || frame.nodes != self.graph.node_frames(frame.offset, frame.rows, frame.mode)
        {
            return Err(Error::Options("chain stale/foreign frame"));
        }
        for (i, cache) in self.caches.iter().enumerate() {
            if cache.owner != frame.cache_owners[i]
                || cache.generation != frame.cache_generations[i]
                || cache.offset != frame.offset
            {
                return Err(Error::Options("chain cache frame identity differs"));
            }
            cache.runs(frame)?;
        }
        Ok(())
    }
    fn guards(
        &mut self,
        frame: &GraphFrame,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<()> {
        self.graph.table.check_completed_guards(command)?;
        let mut bytes = 0;
        let mut valid = true;
        self.entry.with_slice(|s| {
            valid &= sentinel(&s[..GUARD], &mut bytes);
            for (i, r) in self.entry_regions.iter().enumerate() {
                let n = frame.rows * if i == 0 { H } else { P };
                valid &= sentinel(&s[r.start + n..r.end() + GUARD], &mut bytes);
            }
        });
        for node in 0..3 {
            valid &= self.storage[node].guards(
                self.graph.plans[node].layout,
                frame.nodes[node],
                &mut bytes,
            );
            self.graph.norms[node].with_slice(|s| {
                valid &= sentinel(&s[..GUARD], &mut bytes);
                for r in self.graph.plans[node].norm_regions.iter().flatten() {
                    valid &= sentinel(&s[r.end()..r.end() + GUARD], &mut bytes);
                }
            });
            for slot in self.graph.plans[node].matrix_slots.iter().flatten() {
                let m = self.graph.source.matrices[*slot].as_ref().unwrap();
                for (i, tile) in m.tiles.iter().enumerate() {
                    valid &= tail(
                        tile,
                        (m.rows - i * 8192).min(8192) * m.columns / 2,
                        &mut bytes,
                    );
                }
            }
        }
        for (i, r) in self.graph.rope.iter().enumerate() {
            valid &= tail(
                r,
                self.graph.options.capacity * if i == 0 { 256 } else { 128 },
                &mut bytes,
            );
        }
        for c in &self.caches {
            valid &= c.guards(&mut bytes);
        }
        self.metrics.validation_span_bytes += bytes;
        if !valid {
            return Err(Error::Options("chain guard/inactive capacity changed"));
        }
        Ok(())
    }
    fn finite(&mut self, frame: &GraphFrame) -> Result<()> {
        let mut bytes = 0;
        let mut finite = true;
        for node in 0..3 {
            finite &= self.storage[node].finite(
                self.graph.plans[node].layout,
                frame.nodes[node],
                &mut bytes,
            );
        }
        self.metrics.validation_span_bytes += bytes;
        if !finite {
            return Err(Error::Nonfinite);
        }
        Ok(())
    }
    pub fn evaluate(
        &mut self,
        prefix: &[u32],
        hidden: &[f32],
        ple: [&[f32]; 3],
        mode: OutputRows,
        output: &mut [f32],
        cancel: &AtomicBool,
    ) -> Result<Metrics> {
        let rows = prefix
            .len()
            .checked_sub(self.committed)
            .ok_or(Error::Options("chain shortened prefix"))?;
        if output.len()
            != if mode == OutputRows::Last {
                H
            } else {
                rows * H
            }
        {
            return Err(Error::Options("chain caller output extent differs"));
        }
        let mut tx = self.begin(prefix, hidden, ple, mode, cancel)?;
        tx.encode_all(cancel)?;
        tx.submit_and_wait()?;
        tx.complete(cancel)?.publish(Some(output), cancel)
    }
}

#[derive(Clone, Copy, Debug)]
enum ChainReplayReason {
    Original(ReplayReason),
    Rms { slot: usize },
}
impl ChainReplayReason {
    fn slot(self) -> usize {
        match self {
            Self::Original(
                ReplayReason::Admission { slot }
                | ReplayReason::Gelu { slot }
                | ReplayReason::Residual { slot },
            )
            | Self::Rms { slot } => slot,
        }
    }
}
fn chain_audit_status_words(words: &[u32; STATUS_WORDS], consumer: bool) -> Result<()> {
    let mut original = *words;
    for (slot, word) in original.iter_mut().enumerate() {
        if consumer && (5..=9).contains(&slot) && *word != 0 {
            return Err(Error::Options("consumer reserved K/V status differs"));
        }
        if rms::precision::STATUS_SLOTS.contains(&slot) {
            if *word & !3 != 0 {
                return Err(Error::Options("chain RMS status contract differs"));
            }
            *word &= !2;
        }
    }
    audit_status_words(&original, consumer)
}
fn chain_node_decision(words: &[u32; STATUS_WORDS]) -> Result<Option<ChainReplayReason>> {
    let first_rms = words.iter().enumerate().find_map(|(slot, &word)| {
        (rms::precision::STATUS_SLOTS.contains(&slot) && word & 2 != 0).then_some(slot)
    });
    let mut prefix = *words;
    for &slot in &rms::precision::STATUS_SLOTS {
        prefix[slot] &= !2;
    }
    if let Some(slot) = first_rms {
        prefix[slot + 1..].fill(0);
    }
    // Preserve the original ordered classifier for all existing operations and
    // fatal1 at the same RMS site. Descendants of uncertainty are speculative.
    if let Some(reason) = numerical_status_decision(&prefix)? {
        return Ok(Some(ChainReplayReason::Original(reason)));
    }
    Ok(first_rms.map(|slot| ChainReplayReason::Rms { slot }))
}
fn chain_status_decision(
    statuses: &[[u32; STATUS_WORDS]; 3],
) -> Result<Option<(usize, ChainReplayReason)>> {
    for (node, words) in statuses.iter().enumerate() {
        chain_audit_status_words(words, node == 2)?;
    }
    for (node, words) in statuses.iter().enumerate() {
        if let Some(reason) = chain_node_decision(words)? {
            return Ok(Some((node, reason)));
        }
    }
    Ok(None)
}

pub(super) struct GraphTransaction<'s, 'g, 'w> {
    session: &'s mut ResidentSession<'g, 'w>,
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    frame: GraphFrame,
    encoding_started: bool,
    encoded: bool,
    validated: bool,
    published: bool,
    resident_sources: Option<[BufferRange<'s>; 4]>,
    status_snapshot: [[u32; STATUS_WORDS]; 3],
    rms_snapshot: [[u32; MAX_RMS_REASONS]; 3],
}
pub(super) struct CompletedGraph<'s, 'g, 'w>(GraphTransaction<'s, 'g, 'w>);
impl GraphTransaction<'_, '_, '_> {
    fn copy_external_checkpoint(&mut self) -> Result<()> {
        let sources = self
            .resident_sources
            .ok_or(Error::Options("resident sources absent"))?;
        let encoder = self
            .command
            .blitCommandEncoder()
            .ok_or_else(|| Error::Native("chain checkpoint blit encoder absent".into()))?;
        // SAFETY: begin preflight checked all exact same-device shared tracked
        // source spans and disjoint owned entry destinations, retained command
        // and checked byte indices. Source Rust borrows remain in this guard.
        unsafe {
            for (i, source) in sources.into_iter().enumerate() {
                encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    source.raw,
                    source.offset * 4,
                    &self.session.entry.raw,
                    self.session.entry_regions[i].start * 4,
                    source.len * 4,
                );
            }
        }
        encoder.endEncoding();
        self.session.metrics.blit_copies += 4;
        self.session.metrics.device_checkpoint_copy_bytes += self.frame.rows * (H + 3 * P) * 4;
        Ok(())
    }
    fn copy_node_inputs(&mut self, node: usize) -> Result<()> {
        let f = self.frame.nodes[node];
        let plan = &self.session.graph.plans[node];
        let (raw, offset) = if node == 0 {
            (
                &*self.session.entry.raw,
                self.session.entry_regions[0].start,
            )
        } else {
            let previous = &self.session.graph.plans[node - 1];
            let r = previous.layout.region(State::Output);
            let dropped = if node == 2 && self.frame.mode == OutputRows::Last {
                self.frame.rows - 1
            } else {
                0
            };
            (
                &*self.session.storage[node - 1].arena.raw,
                r.start + dropped * H,
            )
        };
        let ple = self.session.entry_regions[node + 1];
        let dropped = if node == 2 && self.frame.mode == OutputRows::Last {
            self.frame.rows - 1
        } else {
            0
        };
        let encoder = self
            .command
            .blitCommandEncoder()
            .ok_or_else(|| Error::Native("chain node input blit encoder absent".into()))?;
        // SAFETY: cold disjoint node owners and checked frame/ranges; every
        // producer row stays live in its own arena through terminal validation.
        // Tracked same-command dependencies order previous writes and these
        // snapshots. Last drops query/hidden/PLE rows only, never donor proposals.
        unsafe {
            encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                raw,
                offset * 4,
                &self.session.storage[node].arena.raw,
                plan.layout.region(State::Hidden).start * 4,
                f.rows * H * 4,
            );
            encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &self.session.entry.raw,
                (ple.start + dropped * P) * 4,
                &self.session.storage[node].arena.raw,
                plan.layout.region(State::PleInput).start * 4,
                f.rows * P * 4,
            );
        }
        encoder.endEncoding();
        self.session.metrics.blit_copies += 2;
        self.session.metrics.device_checkpoint_copy_bytes += f.rows * (H + P) * 4;
        Ok(())
    }
    pub fn encode_all(&mut self, cancel: &AtomicBool) -> Result<()> {
        if self.encoding_started || cancel.load(Ordering::Acquire) {
            return Err(Error::Options("chain already encoded or canceled"));
        }
        self.session.validate_frame(&self.frame)?;
        for node in 0..3 {
            self.session
                .graph
                .layer_recipe(
                    &self.session.storage,
                    &self.session.caches,
                    &self.frame,
                    node,
                    false,
                )?
                .preflight(&self.command)?;
        }
        self.encoding_started = true;
        let start = Instant::now();
        for node in 0..3 {
            self.copy_node_inputs(node)?;
            let operation = self.session.graph.layer_recipe(
                &self.session.storage,
                &self.session.caches,
                &self.frame,
                node,
                self.session.force_reference_node == Some(node),
            )?;
            let result = operation.encode(&self.command);
            let count = operation.dispatches.get();
            let unknown = operation.partial_count_unknown.get();
            self.session
                .metrics
                .record_recipe_encoding(&result, count, unknown);
            result?;
        }
        self.session.metrics.encode_seconds = start.elapsed().as_secs_f64();
        self.session.metrics.encoding_complete = true;
        self.encoded = true;
        Ok(())
    }
    pub fn submit_and_wait(&mut self) -> Result<()> {
        if !self.encoded
            || !matches!(
                self.command.status(),
                MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued
            )
        {
            return Err(Error::Options(
                "chain command not encoded/already submitted",
            ));
        }
        let start = Instant::now();
        self.command.commit();
        self.command.waitUntilCompleted();
        self.session.metrics.candidate_commands += 1;
        self.session.metrics.candidate_waits += 1;
        self.session.metrics.wait_seconds += start.elapsed().as_secs_f64();
        if self.command.status() != MTLCommandBufferStatus::Completed
            || self.command.error().is_some()
        {
            return Err(Error::Completion);
        }
        // These two properties observe the same retained command after its
        // existing successful wait/status check. No new work or synchronization.
        let gpu_start = self.command.GPUStartTime();
        let gpu_end = self.command.GPUEndTime();
        timing::record(
            &mut self.session.metrics,
            self.frame.epoch,
            std::ptr::from_ref(&*self.command).cast::<()>() as usize,
            gpu_start,
            gpu_end,
        );
        Ok(())
    }
    pub fn with_completed_state<T>(
        &self,
        node: usize,
        state: State,
        read: impl FnOnce(&[u32]) -> T,
    ) -> Result<T> {
        if node >= 3
            || self.command.status() != MTLCommandBufferStatus::Completed
            || self.command.error().is_some()
        {
            return Err(Error::Completion);
        }
        let layout = self.session.graph.plans[node].layout;
        let r = layout.region(state);
        let n = layout.active(state, self.frame.nodes[node]);
        self.session
            .diagnostic_bytes
            .set(self.session.diagnostic_bytes.get() + n * 4);
        Ok(self.session.storage[node]
            .arena
            .with_slice(|s| read(&s[r.start..r.start + n])))
    }
}
impl<'s, 'g, 'w> GraphTransaction<'s, 'g, 'w> {
    pub fn complete(mut self, cancel: &AtomicBool) -> Result<CompletedGraph<'s, 'g, 'w>> {
        if !self.encoded
            || self.command.status() != MTLCommandBufferStatus::Completed
            || self.command.error().is_some()
        {
            return Err(Error::Completion);
        }
        let start = Instant::now();
        self.session.validate_frame(&self.frame)?;
        self.session.guards(&self.frame, &self.command)?;
        // Global contracts/proof integrity precede ANY early replay decision.
        for node in 0..3 {
            self.session.storage[node].status.with_slice(|s| {
                self.status_snapshot[node].copy_from_slice(&s[..STATUS_WORDS]);
            });
            self.session.metrics.validation_span_bytes += self.session.storage[node].snapshot_rms(
                self.session.graph.plans[node].layout,
                self.frame.nodes[node],
                &mut self.rms_snapshot[node],
            )?;
        }
        let replay = chain_status_decision(&self.status_snapshot)?;
        if let Some((node, reason)) = replay {
            self.session.metrics.replay_count = 1;
            self.session.metrics.replay_node = Some(node);
            self.session.metrics.replay_slot = Some(reason.slot());
            let session = &mut *self.session;
            let result = session.reference.run(
                session.graph,
                &session.entry,
                &session.entry_regions,
                &session.caches,
                &self.frame,
            );
            session.metrics.reference_matrix_operations = session.reference.matrix_operations;
            session.metrics.reference_commands = session
                .reference
                .counts_known
                .then_some(session.reference.commands);
            session.metrics.reference_waits = session
                .reference
                .counts_known
                .then_some(session.reference.waits);
            session.metrics.reference_input_bytes = session
                .reference
                .counts_known
                .then_some(session.reference.input_bytes);
            session.metrics.reference_readback_bytes = session
                .reference
                .counts_known
                .then_some(session.reference.readback_bytes);
            session.metrics.checkpoint_readback_bytes = session.reference.checkpoint_bytes;
            result?;
            session.metrics.reference_upload_bytes =
                session
                    .reference
                    .upload(session.graph, &mut session.storage, &self.frame);
            session.guards(&self.frame, &self.command)?;
        }
        self.session.finite(&self.frame)?;
        self.validated = true;
        self.session.poisoned = false;
        self.session.metrics.complete_seconds = start.elapsed().as_secs_f64();
        if cancel.load(Ordering::Acquire) {
            return Err(Error::Options("chain canceled before publication"));
        }
        Ok(CompletedGraph(self))
    }
}
impl CompletedGraph<'_, '_, '_> {
    pub fn metrics(&self) -> Metrics {
        self.0.session.observed_metrics()
    }
    pub fn with_state<T>(
        &self,
        node: usize,
        state: State,
        read: impl FnOnce(&[u32]) -> T,
    ) -> Result<T> {
        self.0.with_completed_state(node, state, read)
    }
    pub fn publish(
        mut self,
        mut output: Option<&mut [f32]>,
        cancel: &AtomicBool,
    ) -> Result<Metrics> {
        let e = &mut self.0;
        let start = Instant::now();
        e.session.poisoned = true;
        e.session.validate_frame(&e.frame)?;
        e.session.guards(&e.frame, &e.command)?;
        for node in 0..3 {
            let unchanged = e.session.storage[node]
                .status
                .with_slice(|s| s[..STATUS_WORDS] == e.status_snapshot[node]);
            if !unchanged {
                return Err(Error::Options("chain status changed after validation"));
            }
            chain_audit_status_words(&e.status_snapshot[node], node == 2)?;
            let mut current = [0; MAX_RMS_REASONS];
            e.session.metrics.validation_span_bytes += e.session.storage[node].snapshot_rms(
                e.session.graph.plans[node].layout,
                e.frame.nodes[node],
                &mut current,
            )?;
            if current != e.rms_snapshot[node] {
                return Err(Error::Options("chain RMS reasons changed after validation"));
            }
        }
        e.session.finite(&e.frame)?;
        let runs = [
            e.session.caches[0].runs(&e.frame)?,
            e.session.caches[1].runs(&e.frame)?,
        ];
        let generations = [
            e.session.caches[0].generation.checked_add(1),
            e.session.caches[1].generation.checked_add(1),
        ];
        if generations.iter().any(Option::is_none) {
            return Err(Error::Options("chain publication generation exhausted"));
        }
        let rows = e.frame.nodes[2].rows;
        let count = rows * H;
        if output.as_ref().is_some_and(|out| out.len() != count) {
            e.session.poisoned = false;
            return Err(Error::Options("chain caller output extent differs"));
        }
        if cancel.load(Ordering::Acquire) {
            e.session.poisoned = false;
            return Err(Error::Options("chain canceled before publication"));
        }
        // Every fallible operation has completed. Exclusive session borrowing
        // excludes reset/other commits; cancellation is observed next boundary.
        for (node, runs) in runs.iter().enumerate() {
            let d = e.session.caches[node].d;
            let layout = e.session.graph.plans[node].layout;
            let k = layout.region(State::KProposal);
            let v = layout.region(State::VProposal);
            let cache = &mut e.session.caches[node];
            e.session.storage[node].arena.with_slice(|s| {
                cache.k.with_mut(|dst| {
                    for run in *runs {
                        let n = run.rows * d;
                        let start = k.start + run.source_row * d;
                        dst[run.slot * d..run.slot * d + n].copy_from_slice(&s[start..start + n]);
                    }
                });
                cache.v.with_mut(|dst| {
                    for run in *runs {
                        let n = run.rows * d;
                        let start = v.start + run.source_row * d;
                        dst[run.slot * d..run.slot * d + n].copy_from_slice(&s[start..start + n]);
                    }
                });
            });
        }
        let out = e.session.graph.plans[2].layout.region(State::Output);
        e.session.storage[2].arena.with_slice(|s| {
            for (dst, &word) in e.session.published[..count]
                .iter_mut()
                .zip(&s[out.start..out.start + count])
            {
                *dst = f32::from_bits(word);
            }
            if let Some(output) = output.as_deref_mut() {
                output.copy_from_slice(&e.session.published[..count]);
            }
        });
        for (cache, generation) in e.session.caches.iter_mut().zip(generations) {
            cache.generation = generation.unwrap();
            cache.offset = e.frame.end;
        }
        e.session.history = e.frame.history;
        e.session.committed = e.frame.end;
        e.session.published_rows = rows;
        e.session.metrics.new_kv_copy_bytes = e.frame.rows * (256 + 512) * 8;
        e.session.metrics.output_copy_bytes = count * 4 * (1 + usize::from(output.is_some()));
        e.session.metrics.publication_seconds = start.elapsed().as_secs_f64();
        e.session.poisoned = false;
        e.published = true;
        Ok(e.session.observed_metrics())
    }
}
impl Drop for GraphTransaction<'_, '_, '_> {
    fn drop(&mut self) {
        // Sole graph owner drains while every session/input/source borrow lives.
        // Retaining an unsubmitted caller command alone cannot prevent later
        // submission after resources have been reset; queued commands must drain.
        match self.command.status() {
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued => {
                self.command.commit();
                self.command.waitUntilCompleted();
                self.session.metrics.candidate_commands += 1;
                self.session.metrics.candidate_waits += 1;
            }
            MTLCommandBufferStatus::Committed | MTLCommandBufferStatus::Scheduled => {
                self.command.waitUntilCompleted();
                self.session.metrics.candidate_waits += 1;
            }
            _ => (),
        }
        let terminal = self.command.status() == MTLCommandBufferStatus::Completed
            && self.command.error().is_none();
        debug_assert!(!self.published || self.validated);
        self.session.busy = false;
        if !terminal || self.encoding_started && !self.validated {
            self.session.poisoned = true;
        } else if !self.encoding_started {
            self.session.poisoned = false;
        }
    }
}

#[cfg(test)]
mod tests;
