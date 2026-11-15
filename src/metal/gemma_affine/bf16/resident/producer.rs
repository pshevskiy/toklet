//! Explicit first producer block on original authenticated resident BF16 weights.
//! No model/session route selects this private qualification experiment.
use super::super::super::resident::bindings::ProducerSlots;
use super::*;
use crate::{
    cpu,
    metal::{
        gemma_attention as attention, gemma_gelu as gelu, gemma_residual as residual,
        gemma_rms as rms, gemma_rotary as rotary, memory::BufferRange,
    },
    model::window_rows,
};
use project::bf16_encode::{AdmissionDecision, AdmissionPipeline, Bf16Tile, admission_decision};
use std::{
    cell::Cell,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
};
mod kv;
mod layout;
mod reference;
pub(super) use kv::ProducerKv;
pub(super) use layout::State;
use layout::{Layout, Region, STATES};
static NEXT_BLOCK: AtomicU64 = AtomicU64::new(1);
const STATUS_WORDS: usize = 35;
const PIPELINE_POLICY_BYTES: usize = 6 * 64 * 1024 * 1024;
const H: usize = 1536;
const P: usize = 256;
const F: usize = 6144;
#[derive(Clone, Copy)]
struct Frame {
    owner: u64,
    generation: u64,
    offset: usize,
    rows: usize,
    end: usize,
    history: [u32; 4],
    epoch: u64,
}
pub(super) fn strongest_admission_tile<'a>(tiles: &[Bf16Tile<'a>]) -> Result<Bf16Tile<'a>> {
    let selected = *tiles
        .iter()
        .max_by_key(|tile| tile.maximum_bits())
        .ok_or(Error::Options("producer admission group empty"))?;
    if tiles.iter().any(|tile| {
        selected.max_input_bits() > tile.max_input_bits()
            || selected.l1_input_bits() > tile.l1_input_bits()
    }) {
        return Err(Error::Options("producer grouped admission proof differs"));
    }
    Ok(selected)
}
/// One shared pipeline bundle may serve both qualification layers. Its compiler
/// policy is separate from numerical allocations and is not physical residency.
pub(super) struct Pipelines {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    rms: rms::Pipelines,
    rotary: rotary::Pipelines,
    attention: attention::Pipelines,
    gelu: gelu::Pipelines,
    residual: residual::Pipelines,
    admission: AdmissionPipeline,
}
impl Pipelines {
    /// Admit one immutable table as a caller-owned shared resource; count its
    /// bytes once across layers. This leaves normal graph selection unchanged.
    pub fn admit_table(
        context: &Bf16Context,
        approved: &gelu::ApprovedBytes<'_>,
        available: usize,
    ) -> Result<gelu::CertifiedTable> {
        check_budget(gelu::CertifiedTable::planned_owned_bytes(), available)?;
        gelu::CertifiedTable::from_approved(&context.0.gpu, approved, available)
    }

    pub fn new(c: &Bf16Context, available: usize) -> Result<Self> {
        check_budget(PIPELINE_POLICY_BYTES, available)?;
        let g = &c.0.gpu;
        Ok(Self {
            device: g.device.clone(),
            rms: rms::Pipelines::new(g)?,
            rotary: rotary::Pipelines::new(g)?,
            attention: attention::Pipelines::new(g)?,
            gelu: gelu::Pipelines::new(g)?,
            residual: residual::Pipelines::new(g)?,
            admission: AdmissionPipeline::new(g)?,
        })
    }
}
pub(super) struct ProducerBlock<'a> {
    source: &'a Bf16Weights,
    context: &'a Bf16Context,
    pipelines: &'a Pipelines,
    table: &'a gelu::CertifiedTable,
    slots: ProducerSlots,
    tiles: [Bf16Tile<'a>; 9],
    admission_tiles: [Bf16Tile<'a>; 6],
    layer: usize,
    scalar: f32,
    identity: u64,
    layout: Layout,
    arena: Buffer<u32>,
    norms: Buffer<u32>,
    norm_regions: [Region; 8],
    coefficients: Buffer<u32>,
    status: Buffer<u32>,
    gelu_flags: [Buffer<u32>; 2],
    gelu_intervals: [Buffer<u32>; 2],
    reference: reference::Runner,
    epoch: u64,
    busy: bool,
    poisoned: bool,
    metrics: Cell<Metrics>,
    #[cfg(test)]
    fault: Option<Fault>,
}
enum Checkpoint<'b> {
    Host {
        hidden: &'b [f32],
        ple: &'b [f32],
    },
    Resident {
        hidden: BufferRange<'b>,
        ple: BufferRange<'b>,
    },
}
impl<'a> ProducerBlock<'a> {
    pub fn planned_bytes(layer: usize) -> Result<usize> {
        let d = match layer {
            0 => 256,
            4 => 512,
            _ => return Err(Error::Options("producer layer outside scope")),
        };
        let l = Layout::new(d)?;
        let norm_words = 5 * H + 2 * d + 1 + 8 * GUARD;
        let gpu_words = l.words
            + norm_words
            + 4 * (if layer == 0 { 256 } else { 128 })
            + GUARD
            + STATUS_WORDS
            + GUARD
            + 4 * (F + P) * 9
            + 4 * GUARD;
        let cpu_words = l.regions.iter().map(|r| r.len).sum::<usize>() + 8 * d;
        let projection_words = 2 * (4 * F + GUARD) + 1 + GUARD;
        (gpu_words + cpu_words + projection_words)
            .checked_mul(4)
            .and_then(|b| b.checked_add(size_of::<Self>()))
            .ok_or(Error::Options("producer owned byte overflow"))
    }
    pub fn new(
        source: &'a Bf16Weights,
        context: &'a Bf16Context,
        pipelines: &'a Pipelines,
        table: &'a gelu::CertifiedTable,
        layer: usize,
        available: usize,
    ) -> Result<Self> {
        check_budget(Self::planned_bytes(layer)?, available)?;
        if !std::ptr::eq(&*source.device, &*context.0.gpu.device) {
            return Err(Error::Options("producer foreign weights/context"));
        }
        if !std::ptr::eq(&*pipelines.device, &*context.0.gpu.device) {
            return Err(Error::Options("producer foreign pipeline device"));
        }
        table.validate_device(&context.0.gpu)?;
        let slots = source.bindings.producer_slots(layer)?;
        let d = slots.geometry.head_dim;
        if slots.geometry.kind
            != if layer == 0 {
                crate::model::shared_kv::KvKind::Sliding
            } else {
                crate::model::shared_kv::KvKind::Full
            }
        {
            return Err(Error::Options("producer original attention kind differs"));
        }
        if (layer == 0 && (d != 256 || slots.geometry.rotated_dims != 256))
            || (layer == 4 && (d != 512 || slots.geometry.rotated_dims != 128))
        {
            return Err(Error::Options("producer original geometry differs"));
        }
        let expected = [
            (8 * d, H),
            (d, H),
            (d, H),
            (H, 8 * d),
            (F, H),
            (F, H),
            (H, F),
            (P, H),
            (H, P),
        ];
        let matrix = |i: usize| -> Result<&'a Matrix> {
            let m = source
                .matrices
                .get(slots.matrices[i])
                .and_then(Option::as_ref)
                .ok_or(Error::Options("producer matrix role absent"))?;
            if (m.rows, m.columns) != expected[i] {
                return Err(Error::Options("producer matrix role shape differs"));
            }
            Ok(m)
        };
        let tiles = [
            matrix(0)?.single_tile(context)?,
            matrix(1)?.single_tile(context)?,
            matrix(2)?.single_tile(context)?,
            matrix(3)?.single_tile(context)?,
            matrix(4)?.single_tile(context)?,
            matrix(5)?.single_tile(context)?,
            matrix(6)?.single_tile(context)?,
            matrix(7)?.single_tile(context)?,
            matrix(8)?.single_tile(context)?,
        ];
        // All members' columns are checked above. Choose largest actual scanned
        // maxW: both unchanged max/L1 proofs are monotone in maxW. Choosing only
        // rounded max_input_bits could tie while L1 thresholds still differ.
        let qkv = strongest_admission_tile(&tiles[..3])?;
        let ff = strongest_admission_tile(&tiles[4..6])?;
        let admission_tiles = [qkv, tiles[3], ff, tiles[6], tiles[7], tiles[8]];
        let mut cursor = 0;
        let widths = [H, d, d, H, H, H, H, 1];
        let norm_regions = widths.map(|len| {
            let r = Region { start: cursor, len };
            cursor += len + GUARD;
            r
        });
        let g = &context.0.gpu;
        for (i, &slot) in slots.norms.iter().enumerate() {
            if source
                .vectors
                .get(slot)
                .is_none_or(|v| v.len() != widths[i] || v.iter().any(|x| !x.is_finite()))
            {
                return Err(Error::Options("producer normalization role differs"));
            }
        }
        let scalar = source
            .vectors
            .get(slots.scalar)
            .filter(|v| v.len() == 1 && v[0].is_finite())
            .ok_or(Error::Options("producer learned scalar role differs"))?[0];
        let layout = Layout::new(d)?;
        for words in [
            layout.words,
            cursor,
            4 * slots.geometry.rotated_dims + GUARD,
            STATUS_WORDS + GUARD,
            4 * F + GUARD,
            4 * P + GUARD,
            32 * F + GUARD,
            32 * P + GUARD,
            4 * F + GUARD,
            1 + GUARD,
        ] {
            validate_buffer_bytes(words * 4, g.device.maxBufferLength())?;
        }
        let mut norms = Buffer::new(g, cursor)?;
        norms.with_mut(|s| {
            s.fill(SENTINEL);
            for (i, &slot) in slots.norms.iter().enumerate() {
                for (out, value) in s[norm_regions[i].start..norm_regions[i].end()]
                    .iter_mut()
                    .zip(&source.vectors[slot])
                {
                    *out = value.to_bits();
                }
            }
            s[norm_regions[7].start] = 1_f32.to_bits();
        });
        let identity = NEXT_BLOCK
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1))
            .map_err(|_| Error::Options("producer block identity exhausted"))?;
        let reference = reference::Runner::new(&layout, g)?;
        let owner = Self {
            source,
            context,
            pipelines,
            table,
            slots,
            tiles,
            admission_tiles,
            layer,
            scalar,
            identity,
            arena: Buffer::new(g, layout.words)?,
            norms,
            norm_regions,
            coefficients: Buffer::new(g, 4 * slots.geometry.rotated_dims + GUARD)?,
            status: Buffer::new(g, STATUS_WORDS + GUARD)?,
            gelu_flags: [
                Buffer::new(g, 4 * F + GUARD)?,
                Buffer::new(g, 4 * P + GUARD)?,
            ],
            gelu_intervals: [
                Buffer::new(g, 32 * F + GUARD)?,
                Buffer::new(g, 32 * P + GUARD)?,
            ],
            reference,
            layout,
            epoch: 0,
            busy: false,
            poisoned: false,
            metrics: Cell::new(Metrics::default()),
            #[cfg(test)]
            fault: None,
        };
        check_budget(owner.owned_bytes(), available)?;
        owner.metrics.set(Metrics {
            owner_bytes: owner.owned_bytes(),
            shared_table_identity: table as *const _ as usize,
            shared_table_bytes: table.owned_bytes(),
            shared_pipeline_identity: pipelines as *const _ as usize,
            shared_pipeline_policy_bytes: PIPELINE_POLICY_BYTES,
            shared_weight_identity: source as *const _ as usize,
            shared_weight_retained_bytes: source.report.retained_bytes,
            reference_commands: Some(0),
            reference_waits: Some(0),
            ..Metrics::default()
        });
        Ok(owner)
    }
    fn r(&self, s: State, f: &Frame) -> Result<BufferRange<'_>> {
        self.layout.range(&self.arena, s, f)
    }
    fn s(&self, i: usize) -> Result<BufferRange<'_>> {
        BufferRange::new(&self.status, i, 1)
    }
    fn rope(&self, f: &Frame, heads: usize) -> cpu::ProportionalRotaryShape {
        cpu::ProportionalRotaryShape {
            tokens: f.rows,
            heads,
            head_dim: self.layout.d,
            rotated_dims: self.slots.geometry.rotated_dims,
            offset: f.offset,
            base: if self.layer == 0 { 10000. } else { 1000000. },
            factor: 1.,
        }
    }
    fn rms(
        &self,
        c: &ProtocolObject<dyn MTLCommandBuffer>,
        f: &Frame,
        states: [State; 2],
        norm: usize,
        status: usize,
        shape: [usize; 2],
    ) -> Result<()> {
        let [input, out] = states;
        let [rows, width] = shape;
        self.record(
            self.pipelines.rms.encode(
                c,
                &rms::DeviceBuffers {
                    input: self.r(input, f)?,
                    weights: self.norm_regions[norm]
                        .range(&self.norms, self.norm_regions[norm].len)?,
                    output: self.r(out, f)?,
                    status: self.s(status)?,
                },
                rows,
                width,
                1e-6,
                norm != 7,
            ),
            1,
        )
    }
    fn project(
        &self,
        c: &ProtocolObject<dyn MTLCommandBuffer>,
        f: &Frame,
        m: usize,
        input: State,
        out: State,
        status: usize,
    ) -> Result<()> {
        self.record(
            project::encode_bf16(
                &self.context.0,
                c,
                self.tiles[m],
                self.r(input, f)?,
                self.r(out, f)?,
                self.s(status)?,
                f.rows,
            ),
            1,
        )
    }
    fn admit(
        &self,
        c: &ProtocolObject<dyn MTLCommandBuffer>,
        f: &Frame,
        group: usize,
        input: State,
        status: usize,
    ) -> Result<()> {
        self.record(
            self.pipelines.admission.encode_with_l1(
                c,
                self.admission_tiles[group],
                self.r(input, f)?,
                self.s(status)?,
                f.rows,
            ),
            1,
        )
    }
    fn rotate(
        &self,
        c: &ProtocolObject<dyn MTLCommandBuffer>,
        f: &Frame,
        input: State,
        out: State,
        heads: usize,
        status: usize,
    ) -> Result<()> {
        self.record(
            self.pipelines.rotary.encode(
                c,
                &rotary::DeviceBuffers {
                    input: self.r(input, f)?,
                    output: self.r(out, f)?,
                    status: self.s(status)?,
                    coefficients: BufferRange::new(
                        &self.coefficients,
                        0,
                        f.rows * self.slots.geometry.rotated_dims,
                    )?,
                },
                &self.rope(f, heads),
            ),
            1,
        )
    }
    fn add(
        &self,
        c: &ProtocolObject<dyn MTLCommandBuffer>,
        f: &Frame,
        a: State,
        b: State,
        out: State,
        status: usize,
    ) -> Result<()> {
        self.record(
            self.pipelines.residual.encode_add(
                c,
                self.r(a, f)?,
                self.r(b, f)?,
                self.r(out, f)?,
                self.s(status)?,
            ),
            1,
        )
    }
    fn gelu(
        &self,
        c: &ProtocolObject<dyn MTLCommandBuffer>,
        f: &Frame,
        index: usize,
        states: [State; 4],
        status: usize,
    ) -> Result<()> {
        let [gate, up, activation, product] = states;
        let width = if index == 0 { F } else { P };
        let n = f.rows * width;
        let buffers = gelu::DeviceBuffers {
            gate: self.r(gate, f)?,
            up: self.r(up, f)?,
            activated: self.r(activation, f)?,
            product: self.r(product, f)?,
            status: self.s(status)?,
            lane_flags: BufferRange::new(&self.gelu_flags[index], 0, n)?,
            intervals: BufferRange::new(&self.gelu_intervals[index], 0, 8 * n)?,
        };
        #[cfg(test)]
        if self.fault == Some(Fault::GeluReplay) {
            return self.record(
                self.pipelines.gelu.encode_forced_reference(
                    c,
                    self.table,
                    &buffers,
                    gelu::Shape {
                        rows: f.rows,
                        width,
                    },
                ),
                1,
            );
        }
        self.record(
            self.pipelines.gelu.encode(
                c,
                self.table,
                &buffers,
                gelu::Shape {
                    rows: f.rows,
                    width,
                },
            ),
            1,
        )
    }
    /// Stage the retained checkpoint, then encode35 dependent dispatches into
    /// the caller's retained NotEnqueued command. No submit/wait occurs here.
    /// Returned guard exclusively borrows block/cache through terminal completion.
    /// Dropping it commits/drains unfinished speculative work before releasing
    /// borrows; only explicit validated publication changes the committed prefix.
    /// Any encode error invalidates both owners; caller must discard that command.
    pub fn encode<'b>(
        &'b mut self,
        cache: &'b mut ProducerKv,
        command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        prefix: &[u32],
        hidden: &'b [f32],
        ple: &'b [f32],
        cancel: &AtomicBool,
    ) -> Result<Encoded<'b, 'a>> {
        autoreleasepool(move |_| {
            self.encode_checkpoint(
                cache,
                command,
                prefix,
                Checkpoint::Host { hidden, ple },
                cancel,
            )
        })
    }
    /// Resident import snapshots two finite/checkpoint-capable upstream ranges
    /// with GPU blits on this same command. Source borrows remain retained by
    /// the guard; there is no CPU staging/readback or host wait. Whole-graph
    /// callers must resolve upstream statuses before downstream local replay.
    pub fn encode_resident<'b>(
        &'b mut self,
        cache: &'b mut ProducerKv,
        command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        prefix: &[u32],
        hidden: BufferRange<'b>,
        ple: BufferRange<'b>,
        cancel: &AtomicBool,
    ) -> Result<Encoded<'b, 'a>> {
        autoreleasepool(move |_| {
            self.encode_checkpoint(
                cache,
                command,
                prefix,
                Checkpoint::Resident { hidden, ple },
                cancel,
            )
        })
    }
    fn encode_checkpoint<'b>(
        &'b mut self,
        cache: &'b mut ProducerKv,
        command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        prefix: &[u32],
        input: Checkpoint<'b>,
        cancel: &AtomicBool,
    ) -> Result<Encoded<'b, 'a>> {
        if self.busy || self.poisoned || cancel.load(Ordering::Acquire) {
            return Err(Error::Options("producer unavailable or canceled"));
        }
        self.epoch
            .checked_add(1)
            .ok_or(Error::Options("producer epoch exhausted"))?;
        let frame = cache.check(self, prefix)?;
        let (hn, pn, resident_sources) = match &input {
            Checkpoint::Host { hidden, ple } => {
                if hidden.iter().chain(ple.iter()).any(|v| !v.is_finite()) {
                    return Err(Error::Options("producer checkpoint nonfinite value"));
                }
                (hidden.len(), ple.len(), None)
            }
            Checkpoint::Resident { hidden, ple } => {
                fn buffer(b: &Buffer<u32>) -> Result<BufferRange<'_>> {
                    BufferRange::new(b, 0, b.len)
                }
                crate::metal::memory::validate_encode_ranges(
                    command,
                    &self.context.0.gpu.device,
                    &[
                        *hidden,
                        *ple,
                        buffer(&self.arena)?,
                        buffer(&self.status)?,
                        buffer(&self.coefficients)?,
                        buffer(&self.gelu_flags[0])?,
                        buffer(&self.gelu_flags[1])?,
                        buffer(&self.gelu_intervals[0])?,
                        buffer(&self.gelu_intervals[1])?,
                    ],
                    2,
                )?;
                (hidden.len, ple.len, Some([*hidden, *ple]))
            }
        };
        if hn != frame.rows * H || pn != frame.rows * P {
            return Err(Error::Options("producer checkpoint extent differs"));
        }
        crate::metal::memory::validate_encode_ranges(command, &self.context.0.gpu.device, &[], 0)?;
        self.metrics.set(Metrics {
            owner_bytes: self.owned_bytes(),
            shared_table_identity: self.table as *const _ as usize,
            shared_table_bytes: self.table.owned_bytes(),
            kv_owner_bytes: size_of::<ProducerKv>() + (cache.k.len + cache.v.len) * 4,
            shared_pipeline_identity: self.pipelines as *const _ as usize,
            shared_pipeline_policy_bytes: PIPELINE_POLICY_BYTES,
            shared_weight_identity: self.source as *const _ as usize,
            shared_weight_retained_bytes: self.source.report.retained_bytes,
            scratch_reset_bytes: (self.arena.len
                + self.status.len
                + self.coefficients.len
                + self.gelu_flags.iter().map(|b| b.len).sum::<usize>()
                + self.gelu_intervals.iter().map(|b| b.len).sum::<usize>())
                * 4,
            input_upload_bytes: if resident_sources.is_none() {
                (hn + pn) * 4
            } else {
                0
            },
            device_input_copy_bytes: if resident_sources.is_some() {
                (hn + pn) * 4
            } else {
                0
            },
            epoch: frame.epoch,
            command_identity: std::ptr::from_ref(&**command).cast::<()>() as usize,
            coefficient_upload_bytes: frame.rows * self.slots.geometry.rotated_dims * 4,
            reference_commands: Some(0),
            reference_waits: Some(0),
            ..Metrics::default()
        });
        self.busy = true;
        self.poisoned = true;
        cache.busy = true;
        self.epoch = frame.epoch;
        self.arena.with_mut(|s| {
            s.fill(SENTINEL);
            if let Checkpoint::Host { hidden, ple } = &input {
                for (state, values) in [(State::Hidden, *hidden), (State::PleInput, *ple)] {
                    let r = self.layout.region(state);
                    for (out, value) in s[r.start..r.start + values.len()].iter_mut().zip(values) {
                        *out = value.to_bits();
                    }
                }
            }
        });
        self.status.with_mut(|s| {
            s.fill(SENTINEL);
            s[..STATUS_WORDS].fill(0);
        });
        for b in self.gelu_flags.iter_mut().chain(&mut self.gelu_intervals) {
            b.with_mut(|s| s.fill(SENTINEL));
        }
        let shape = self.rope(&frame, 8);
        self.coefficients.with_mut(|s| {
            s.fill(SENTINEL);
            rotary::prepare_coefficients(&shape, &mut s[..frame.rows * shape.rotated_dims])
        })?;
        if let Some(sources) = resident_sources {
            let blit = command
                .blitCommandEncoder()
                .ok_or_else(|| Error::Native("producer checkpoint blit encoder absent".into()))?;
            // SAFETY: preflighted same-device retained/tracked initialized u32
            // ranges; all source ranges are disjoint from CPU-reset destinations.
            // The caller's command orders upstream writes and these snapshots.
            unsafe {
                for (source, state) in sources.into_iter().zip([State::Hidden, State::PleInput]) {
                    let dst = self.layout.region(state);
                    blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        source.raw,
                        source.offset * 4,
                        &self.arena.raw,
                        dst.start * 4,
                        source.len * 4,
                    );
                }
            }
            blit.endEncoding();
            let mut m = self.metrics.get();
            m.encoded_blit_copies = 2;
            self.metrics.set(m);
        }
        self.encode_steps(command, cache, &frame)?;
        let mut metrics = self.metrics.get();
        metrics.encoding_complete = true;
        metrics.speculative_dispatches = metrics.encoded_dispatches;
        metrics.speculative_kv_bytes = 2 * frame.rows * self.layout.d * 4;
        self.metrics.set(metrics);
        Ok(Encoded {
            block: self,
            cache,
            command: command.clone(),
            frame,
            validated: false,
            replayed: false,
            validated_status: [0; STATUS_WORDS],
            _resident_sources: resident_sources,
        })
    }
    fn encode_steps(
        &self,
        c: &ProtocolObject<dyn MTLCommandBuffer>,
        cache: &ProducerKv,
        f: &Frame,
    ) -> Result<()> {
        use State::*;
        let d = self.layout.d;
        self.rms(c, f, [Hidden, InputNorm], 0, 0, [f.rows, H])?;
        self.admit(c, f, 0, InputNorm, 1)?;
        self.project(c, f, 0, InputNorm, QRaw, 2)?;
        self.rms(c, f, [QRaw, QNorm], 1, 3, [f.rows * 8, d])?;
        self.rotate(c, f, QNorm, QRotary, 8, 4)?;
        self.project(c, f, 1, InputNorm, KRaw, 5)?;
        self.rms(c, f, [KRaw, KNorm], 2, 6, [f.rows, d])?;
        self.rotate(c, f, KNorm, KProposal, 1, 7)?;
        self.project(c, f, 2, InputNorm, VRaw, 8)?;
        self.rms(c, f, [VRaw, VProposal], 7, 9, [f.rows, d])?;
        let old = window_rows::old_spans(cache.capacity, f.offset, cache.window);
        let segment = |i: usize| -> Result<attention::DeviceSegment<'_>> {
            let s = old[i];
            Ok(attention::DeviceSegment {
                k: BufferRange::new(&cache.k, s.slot * d, s.rows * d)?,
                v: BufferRange::new(&cache.v, s.slot * d, s.rows * d)?,
                keys: s.rows,
                head_stride: s.rows * d,
                key_start: s.position,
            })
        };
        self.record(
            self.pipelines.attention.encode(
                c,
                &attention::DeviceBuffers {
                    q: self.r(QRotary, f)?,
                    segments: [
                        segment(0)?,
                        segment(1)?,
                        attention::DeviceSegment {
                            k: self.r(KProposal, f)?,
                            v: self.r(VProposal, f)?,
                            keys: f.rows,
                            head_stride: f.rows * d,
                            key_start: f.offset,
                        },
                    ],
                    scores: self.r(Scores, f)?,
                    probabilities: self.r(Probabilities, f)?,
                    output: self.r(Context, f)?,
                    status: self.s(10)?,
                },
                &cpu::SpanAttentionShape {
                    queries: f.rows,
                    keys: f.end,
                    query_heads: 8,
                    kv_heads: 1,
                    head_dim: d,
                    query_start: f.offset,
                    key_start: 0,
                    window: cache.window,
                    scale: 1.,
                },
            ),
            4,
        )?;
        // Slots11..13 are reserved zero: existing four attention encoders share
        // one fatal-only status, so their graph-order interpretation is identical.
        self.admit(c, f, 1, Context, 14)?;
        self.project(c, f, 3, Context, AttentionRaw, 15)?;
        self.rms(c, f, [AttentionRaw, AttentionNorm], 3, 16, [f.rows, H])?;
        self.add(c, f, AttentionNorm, Hidden, AttentionResidual, 17)?;
        self.rms(c, f, [AttentionResidual, PreFf], 4, 18, [f.rows, H])?;
        self.admit(c, f, 2, PreFf, 19)?;
        self.project(c, f, 4, PreFf, FfGate, 20)?;
        self.project(c, f, 5, PreFf, FfUp, 21)?;
        self.gelu(c, f, 0, [FfGate, FfUp, FfActivation, FfProduct], 22)?;
        self.admit(c, f, 3, FfProduct, 23)?;
        self.project(c, f, 6, FfProduct, FfDown, 24)?;
        self.rms(c, f, [FfDown, FfNorm], 5, 25, [f.rows, H])?;
        self.add(c, f, FfNorm, AttentionResidual, FfResidual, 26)?;
        self.admit(c, f, 4, FfResidual, 27)?;
        self.project(c, f, 7, FfResidual, PleGate, 28)?;
        self.gelu(c, f, 1, [PleGate, PleInput, PleActivation, PleProduct], 29)?;
        self.admit(c, f, 5, PleProduct, 30)?;
        self.project(c, f, 8, PleProduct, PleProjection, 31)?;
        self.rms(c, f, [PleProjection, PleNorm], 6, 32, [f.rows, H])?;
        self.add(c, f, FfResidual, PleNorm, PleResidual, 33)?;
        self.record(
            self.pipelines.residual.encode_scale(
                c,
                self.r(PleResidual, f)?,
                self.r(Output, f)?,
                self.s(34)?,
                self.scalar,
            ),
            1,
        )
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(super) enum ReplayReason {
    Admission { slot: usize },
    Gelu { slot: usize },
    Residual { slot: usize },
}
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub(super) struct Metrics {
    pub epoch: u64,
    pub command_identity: usize,
    pub encoded_dispatches: usize,
    pub encoded_blit_copies: usize,
    pub device_input_copy_bytes: usize,
    pub speculative_dispatches: usize,
    pub encoding_complete: bool,
    pub gpu_completed_dispatches: Option<usize>,
    pub submitted_commands: Option<usize>,
    pub completed_commands: usize,
    pub waits: Option<usize>,
    pub input_upload_bytes: usize,
    pub scratch_reset_bytes: usize,
    pub reference_checkpoint_copy_bytes: usize,
    pub coefficient_upload_bytes: usize,
    pub readback_copy_bytes: usize,
    pub diagnostic_exposed_bytes: usize,
    pub speculative_kv_bytes: usize,
    pub new_kv_copy_bytes: usize,
    pub owner_bytes: usize,
    pub shared_table_identity: usize,
    pub shared_table_bytes: usize,
    pub kv_owner_bytes: usize,
    pub shared_pipeline_identity: usize,
    pub shared_pipeline_policy_bytes: usize,
    pub shared_weight_identity: usize,
    pub shared_weight_retained_bytes: usize,
    pub replay_reason: Option<ReplayReason>,
    pub replay_count: usize,
    pub reference_commands: Option<usize>,
    pub reference_waits: Option<usize>,
    pub reference_input_copy_bytes: usize,
    pub reference_readback_copy_bytes: usize,
    pub reference_state_upload_bytes: usize,
}
impl ProducerBlock<'_> {
    pub fn metrics(&self) -> Metrics {
        self.metrics.get()
    }
    pub fn owned_bytes(&self) -> usize {
        size_of::<Self>()
            + (self.arena.len
                + self.norms.len
                + self.coefficients.len
                + self.status.len
                + self.gelu_flags.iter().map(|b| b.len).sum::<usize>()
                + self.gelu_intervals.iter().map(|b| b.len).sum::<usize>())
                * 4
            + self.reference.allocation_bytes()
    }
    fn record(&self, result: Result<()>, dispatches: usize) -> Result<()> {
        if result.is_ok() {
            let mut m = self.metrics.get();
            m.encoded_dispatches += dispatches;
            self.metrics.set(m);
        }
        result
    }
    fn guards(
        &self,
        f: &Frame,
        cache: &ProducerKv,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<()> {
        self.table.check_completed_guards(command)?;
        let tail = |b: &Buffer<u32>, n| b.with_slice(|s| s[n..].iter().all(|&w| w == SENTINEL));
        if !self.layout.guards(&self.arena, f)
            || !cache.guards()
            || !tail(&self.status, STATUS_WORDS)
            || !tail(
                &self.coefficients,
                f.rows * self.slots.geometry.rotated_dims,
            )
            || !self.norms.with_slice(|s| {
                self.norm_regions
                    .iter()
                    .all(|r| s[r.end()..r.end() + GUARD].iter().all(|&w| w == SENTINEL))
            })
            || ![F, P].iter().enumerate().all(|(i, &width)| {
                tail(&self.gelu_flags[i], f.rows * width)
                    && tail(&self.gelu_intervals[i], 8 * f.rows * width)
            })
        {
            return Err(Error::Options("producer guard/inactive capacity changed"));
        }
        for &slot in &self.slots.matrices {
            let matrix = self.source.matrices[slot].as_ref().unwrap();
            if !tail(&matrix.tiles[0], matrix.rows * matrix.columns / 2) {
                return Err(Error::Options("producer source weight guard changed"));
            }
        }
        Ok(())
    }
    fn status_decision(&self) -> Result<Option<ReplayReason>> {
        self.status
            .with_slice(|s| status_words_decision(&s[..STATUS_WORDS]))
    }
    pub fn apply(
        &mut self,
        cache: &mut ProducerKv,
        prefix: &[u32],
        hidden: &[f32],
        ple: &[f32],
        output: &mut [f32],
        cancel: &AtomicBool,
    ) -> Result<Metrics> {
        let frame = cache.check(self, prefix)?;
        if output.len() != frame.rows * H {
            return Err(Error::Options("producer caller output extent differs"));
        }
        let command = self.context.0.gpu.command()?;
        let mut encoded = self.encode(cache, &command, prefix, hidden, ple, cancel)?;
        encoded.submit_and_wait()?;
        encoded.complete(cancel)?.publish(Some(output), cancel)
    }
}
/// Contract/proof flags are global failures even when an earlier operation
/// requests replay. Numeric errors retain graph order: speculative descendants
/// of the first replay request are discarded with the candidate.
fn status_words_decision(words: &[u32]) -> Result<Option<ReplayReason>> {
    if words.len() != STATUS_WORDS {
        return Err(Error::Options("producer status extent differs"));
    }
    for (slot, &word) in words.iter().enumerate() {
        let allowed = if [1, 14, 19, 23, 27, 30, 17, 26, 33, 34].contains(&slot) {
            3
        } else if [22, 29].contains(&slot) {
            if word & 16 != 0 {
                return Err(Error::Options("producer GELU interval proof failed"));
            }
            31
        } else if [11, 12, 13].contains(&slot) {
            0
        } else {
            1
        };
        if word & !allowed != 0 {
            return Err(Error::Options("producer status contract differs"));
        }
    }
    for (slot, &word) in words.iter().enumerate() {
        if [1, 14, 19, 23, 27, 30].contains(&slot) {
            if admission_decision(word)? == AdmissionDecision::ReferenceReplay {
                return Ok(Some(ReplayReason::Admission { slot }));
            }
        } else if [22, 29].contains(&slot) {
            match gelu::CompletionStatus::parse(word)? {
                gelu::CompletionStatus::CandidateReady => (),
                gelu::CompletionStatus::NeedsReference => {
                    return Ok(Some(ReplayReason::Gelu { slot }));
                }
                gelu::CompletionStatus::Fatal(_) => return Err(Error::Nonfinite),
            }
        } else if [17, 26, 33, 34].contains(&slot) {
            if residual::decision(word)? == residual::Decision::ReferenceReplay {
                return Ok(Some(ReplayReason::Residual { slot }));
            }
        } else if word != 0 {
            return Err(Error::Nonfinite);
        }
    }
    Ok(None)
}
pub(super) struct Encoded<'b, 'a> {
    block: &'b mut ProducerBlock<'a>,
    cache: &'b mut ProducerKv,
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    frame: Frame,
    validated: bool,
    replayed: bool,
    validated_status: [u32; STATUS_WORDS],
    _resident_sources: Option<[BufferRange<'b>; 2]>,
}
pub(super) struct Completed<'b, 'a>(Encoded<'b, 'a>);
impl Encoded<'_, '_> {
    /// Borrowed GPU output can feed another encoder on this same caller command.
    /// The guard must stay alive through completion; no CPU view is exposed here.
    pub fn hidden(&self) -> Result<BufferRange<'_>> {
        self.block.r(State::Output, &self.frame)
    }
    pub fn submit_and_wait(&mut self) -> Result<()> {
        if !matches!(
            self.command.status(),
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued
        ) {
            return Err(Error::Options("producer command already submitted"));
        }
        let mut m = self.block.metrics.get();
        m.submitted_commands = Some(1);
        m.waits = Some(1);
        self.block.metrics.set(m);
        self.command.commit();
        self.command.waitUntilCompleted();
        if self.command.status() == MTLCommandBufferStatus::Completed
            && self.command.error().is_none()
        {
            let mut m = self.block.metrics.get();
            m.completed_commands = 1;
            m.gpu_completed_dispatches = Some(m.encoded_dispatches);
            self.block.metrics.set(m);
        }
        Ok(())
    }
    /// Explicit raw diagnostics are allowed only after terminal success and
    /// before any numerical/status gates or CPU fallback overwrites speculation.
    pub fn with_completed_arena<T>(&self, f: impl FnOnce(&[u32]) -> T) -> Result<T> {
        if self.command.status() != MTLCommandBufferStatus::Completed
            || self.command.error().is_some()
        {
            return Err(Error::Completion);
        }
        let mut m = self.block.metrics.get();
        m.diagnostic_exposed_bytes += self.block.arena.len * 4;
        self.block.metrics.set(m);
        Ok(self.block.arena.with_slice(f))
    }
}
impl<'b, 'a> Encoded<'b, 'a> {
    pub fn complete(mut self, cancel: &AtomicBool) -> Result<Completed<'b, 'a>> {
        if self.command.status() != MTLCommandBufferStatus::Completed
            || self.command.error().is_some()
        {
            return Err(Error::Completion);
        }
        let mut metrics = self.block.metrics.get();
        metrics.completed_commands = 1;
        metrics.gpu_completed_dispatches = Some(metrics.encoded_dispatches);
        self.block.metrics.set(metrics);
        self.block.guards(&self.frame, self.cache, &self.command)?;
        self.cache.validate_frame(self.block, &self.frame)?;
        if let Some(reason) = self.block.status_decision()? {
            let mut m = self.block.metrics.get();
            m.replay_reason = Some(reason);
            m.replay_count = 1;
            m.reference_checkpoint_copy_bytes =
                (self.frame.rows * (H + P) + 2 * self.frame.offset * self.block.layout.d) * 4;
            m.readback_copy_bytes += m.reference_checkpoint_copy_bytes;
            self.block.metrics.set(m);
            let b = &mut *self.block;
            let result = b.reference.run(reference::Input {
                source: b.source,
                context: b.context,
                slots: b.slots,
                layer: b.layer,
                arena: &b.arena,
                layout: &b.layout,
                cache: self.cache,
                frame: &self.frame,
            });
            let mut m = b.metrics.get();
            m.reference_commands = result.as_ref().ok().map(|_| b.reference.commands);
            m.reference_waits = m.reference_commands;
            m.reference_input_copy_bytes = b.reference.input_bytes;
            m.reference_readback_copy_bytes = b.reference.readback_bytes;
            m.readback_copy_bytes += b.reference.readback_bytes;
            m.completed_commands += b.reference.commands;
            if result.is_ok() {
                m.encoded_dispatches += b.reference.commands;
                m.gpu_completed_dispatches =
                    m.gpu_completed_dispatches.map(|n| n + b.reference.commands);
                m.submitted_commands = m.submitted_commands.map(|n| n + b.reference.commands);
                m.waits = m.waits.map(|n| n + b.reference.commands);
            }
            b.metrics.set(m);
            result?;
            b.reference.upload(&b.layout, &self.frame, &mut b.arena);
            let mut m = b.metrics.get();
            m.reference_state_upload_bytes = STATES
                .iter()
                .map(|&s| b.layout.active(s, &self.frame) * 4)
                .sum();
            b.metrics.set(m);
            self.replayed = true;
        }
        if !self.block.arena.with_slice(|s| {
            STATES.iter().all(|&state| {
                let r = self.block.layout.region(state);
                s[r.start..r.start + self.block.layout.active(state, &self.frame)]
                    .iter()
                    .all(|&w| f32::from_bits(w).is_finite())
            })
        }) {
            return Err(Error::Nonfinite);
        }
        self.cache.runs(&self.frame)?;
        self.block
            .status
            .with_slice(|s| self.validated_status.copy_from_slice(&s[..STATUS_WORDS]));
        self.validated = true;
        self.block.poisoned = false;
        if cancel.load(Ordering::Acquire) {
            return Err(Error::Options("producer canceled before publication"));
        }
        Ok(Completed(self))
    }
}
impl Completed<'_, '_> {
    pub fn metrics(&self) -> Metrics {
        self.0.block.metrics.get()
    }
    pub fn with_state<T>(&self, state: State, f: impl FnOnce(&[u32]) -> T) -> T {
        let b = &*self.0.block;
        let r = b.layout.region(state);
        let n = b.layout.active(state, &self.0.frame);
        let mut m = b.metrics.get();
        m.diagnostic_exposed_bytes += n * 4;
        b.metrics.set(m);
        b.arena.with_slice(|s| f(&s[r.start..r.start + n]))
    }
    /// Validates everything before the first committed write. Copy runs and
    /// output lengths are checked first; cancellation is not checked mid-copy.
    /// Metadata updates only after both K/V payloads and optional output copy.
    pub fn publish(
        mut self,
        mut output: Option<&mut [f32]>,
        cancel: &AtomicBool,
    ) -> Result<Metrics> {
        let e = &mut self.0;
        e.block.poisoned = true;
        e.block.guards(&e.frame, e.cache, &e.command)?;
        if !e
            .block
            .status
            .with_slice(|s| s[..STATUS_WORDS] == e.validated_status)
        {
            return Err(Error::Options("producer status changed after validation"));
        }
        if !e.block.arena.with_slice(|s| {
            [State::Output, State::KProposal, State::VProposal]
                .iter()
                .all(|&state| {
                    let r = e.block.layout.region(state);
                    s[r.start..r.start + e.block.layout.active(state, &e.frame)]
                        .iter()
                        .all(|&w| f32::from_bits(w).is_finite())
                })
        }) {
            return Err(Error::Nonfinite);
        }
        e.cache.validate_frame(e.block, &e.frame)?;
        let runs = e.cache.runs(&e.frame)?;
        e.block.poisoned = false;
        if output.as_ref().is_some_and(|s| s.len() != e.frame.rows * H) {
            return Err(Error::Options("producer caller output extent differs"));
        }
        if cancel.load(Ordering::Acquire) {
            return Err(Error::Options("producer canceled before publication"));
        }
        let next_generation = e
            .cache
            .generation
            .checked_add(1)
            .ok_or(Error::Options("producer generation exhausted"))?;
        let d = e.block.layout.d;
        let kr = e.block.layout.region(State::KProposal);
        let vr = e.block.layout.region(State::VProposal);
        let out = e.block.layout.region(State::Output);
        let f = e.frame;
        // All indices/lengths were validated above; only infallible slice copies
        // occur from this point until metadata is published. Exclusive borrows
        // exclude cancellation/reset or partial owner reuse in the copy phase.
        e.block.arena.with_slice(|s| {
            e.cache.k.with_mut(|dst| {
                for run in runs {
                    let n = run.rows * d;
                    let start = kr.start + run.source_row * d;
                    dst[run.slot * d..run.slot * d + n].copy_from_slice(&s[start..start + n]);
                }
            });
            e.cache.v.with_mut(|dst| {
                for run in runs {
                    let n = run.rows * d;
                    let start = vr.start + run.source_row * d;
                    dst[run.slot * d..run.slot * d + n].copy_from_slice(&s[start..start + n]);
                }
            });
            if let Some(dst) = output.as_deref_mut() {
                for (out, &bits) in dst.iter_mut().zip(&s[out.start..out.start + f.rows * H]) {
                    *out = f32::from_bits(bits);
                }
            }
        });
        e.cache.history = f.history;
        e.cache.offset = f.end;
        e.cache.generation = next_generation;
        let mut m = e.block.metrics.get();
        m.new_kv_copy_bytes = 2 * f.rows * d * 4;
        if output.is_some() {
            m.readback_copy_bytes += f.rows * H * 4;
        }
        e.block.metrics.set(m);
        Ok(m)
    }
}
impl Drop for Encoded<'_, '_> {
    fn drop(&mut self) {
        // Metal has no discard/cancel operation for an encoded command. Merely
        // releasing our retained reference is insufficient: the caller retains
        // it too, and enqueue() reserves a queue position without submitting.
        // Drain while all native resources and resident source borrows are held.
        // Submission authority must not race between caller and this guard.
        match self.command.status() {
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued => {
                let _ = self.submit_and_wait();
            }
            MTLCommandBufferStatus::Committed | MTLCommandBufferStatus::Scheduled => {
                self.command.waitUntilCompleted();
            }
            _ => (),
        }
        if self.command.status() == MTLCommandBufferStatus::Completed
            && self.command.error().is_none()
        {
            let mut m = self.block.metrics.get();
            m.completed_commands = m.completed_commands.max(1);
            m.gpu_completed_dispatches = Some(m.encoded_dispatches);
            self.block.metrics.set(m);
        }
        self.block.busy = false;
        self.cache.busy = false;
        // Failed command/guard/numerics retain sticky poison. A validated frame
        // may be discarded or canceled without changing the valid KV prefix.
        if !self.validated {
            self.block.poisoned = true;
        }
    }
}
#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fault {
    GeluReplay,
}
#[cfg(test)]
mod tests;
