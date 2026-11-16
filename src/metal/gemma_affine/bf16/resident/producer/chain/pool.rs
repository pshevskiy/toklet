//! One interlayer pool; every numerical range remains disjoint within a node.
//! Retired states are replaced only by the same GPU audit that validated them.
use super::*;
use crate::metal::gemma_residual::pool_audit;
const SPANS: usize = 32;
const STATE_SPANS: usize = 28;
const CHUNK: usize = 1024;
const ABI: u32 = 1;

#[derive(Clone, Copy)]
pub(super) struct View<'a> {
    pub entry: &'a Buffer<u32>,
    pub entry_regions: &'a [Region; 4],
    pub layouts: [LayerLayout; 3],
    pub frame: &'a GraphFrame,
}

fn slot(state: State) -> Option<usize> {
    if matches!(
        state,
        State::Hidden | State::Output | State::KProposal | State::VProposal
    ) {
        return None;
    }
    let i = state as usize;
    Some(i - usize::from(i > 0) - usize::from(i > 12) - usize::from(i > 19) - usize::from(i > 21))
}
#[derive(Clone, Copy)]
pub(super) struct Plan {
    spans: [Region; SPANS],
    pub words: usize,
    pub chunks: usize,
    rows: usize,
    keys: usize,
}
impl Plan {
    pub fn new(layouts: [LayerLayout; 3]) -> Result<Self> {
        let rows = layouts[0].rows;
        let keys = layouts[0].keys;
        for (node, l) in layouts.iter().enumerate() {
            if l.rows != rows
                || l.keys != keys
                || l.d != if node == 1 { 512 } else { 256 }
                || l.ff != if node == 2 { 12288 } else { 6144 }
                || l.producer != (node != 2)
            {
                return Err(Error::Options("pool requires exact13/14/15 cold roles"));
            }
            LayerLayout::new(l.rows, l.keys, l.d, l.ff, l.producer)?;
        }
        let mut capacities = [0; SPANS];
        for state in STATES {
            if let Some(slot) = slot(state) {
                capacities[slot] = layouts.iter().map(|l| l.region(state).len).max().unwrap();
            }
        }
        capacities[28] = rows * 12288;
        capacities[29] = rows * P;
        capacities[30] = 8 * rows * 12288;
        capacities[31] = 8 * rows * P;
        let mut words = 0usize;
        let mut chunks = 0usize;
        let mut spans = [Region { start: 0, len: 0 }; SPANS];
        for (span, capacity) in spans.iter_mut().zip(capacities) {
            *span = Region {
                start: words,
                len: capacity,
            };
            let count = capacity
                .checked_add(GUARD)
                .ok_or(Error::Options("pool span overflow"))?;
            words = words
                .checked_add(count)
                .ok_or(Error::Options("pool word overflow"))?;
            chunks = chunks
                .checked_add(count.div_ceil(CHUNK))
                .ok_or(Error::Options("pool chunk overflow"))?;
        }
        if words > 6_594_784 || chunks > 6472 || chunks == 0 {
            return Err(Error::Options("pool exceeds fixed audit envelope"));
        }
        let plan = Self {
            spans,
            words,
            chunks,
            rows,
            keys,
        };
        plan.coverage()?;
        Ok(plan)
    }
    fn coverage(self) -> Result<()> {
        let mut cursor = 0;
        let mut chunks = 0;
        for span in self.spans {
            if span.start != cursor {
                return Err(Error::Options("pool span coverage differs"));
            }
            cursor = span
                .end()
                .checked_add(GUARD)
                .ok_or(Error::Options("pool coverage overflow"))?;
            chunks += (span.len + GUARD).div_ceil(CHUNK);
        }
        if cursor != self.words || chunks != self.chunks {
            return Err(Error::Options("pool physical coverage differs"));
        }
        Ok(())
    }
    fn active(self, layout: LayerLayout, frame: NodeFrame) -> Result<[u32; SPANS]> {
        layout.check_frame(frame)?;
        if layout.rows != self.rows || layout.keys != self.keys {
            return Err(Error::Options("pool frame cold capacity differs"));
        }
        let mut active = [0usize; SPANS];
        for state in STATES {
            if let Some(slot) = slot(state) {
                active[slot] = layout.active(state, frame);
            }
        }
        active[28] = frame.rows * layout.ff;
        active[29] = frame.rows * P;
        active[30] = 8 * active[28];
        active[31] = 8 * active[29];
        let mut out = [0u32; SPANS];
        for ((out, n), span) in out.iter_mut().zip(active).zip(self.spans) {
            if n > span.len {
                return Err(Error::Options("pool active span exceeds capacity"));
            }
            *out = u32::try_from(n).map_err(|_| Error::Options("pool active count truncation"))?;
        }
        Ok(out)
    }
    fn span_words(self, slot: usize) -> [u32; 4] {
        let r = self.spans[slot];
        [
            r.start as u32,
            r.len as u32,
            u32::from(slot < STATE_SPANS),
            ABI,
        ]
    }
    fn work(self, mut visit: impl FnMut(usize, [u32; 4])) {
        let mut record = 0;
        for (slot, span) in self.spans.iter().enumerate() {
            let count = span.len + GUARD;
            for begin in (0..count).step_by(CHUNK) {
                visit(
                    record,
                    [
                        slot as u32,
                        begin as u32,
                        (count - begin).min(CHUNK) as u32,
                        ABI,
                    ],
                );
                record += 1;
            }
        }
        debug_assert_eq!(record, self.chunks);
    }
    fn range(
        self,
        buffer: &Buffer<u32>,
        state: State,
        layout: LayerLayout,
        frame: NodeFrame,
    ) -> Result<Option<BufferRange<'_>>> {
        slot(state)
            .map(|slot| self.spans[slot].range(buffer, layout.active(state, frame)))
            .transpose()
    }
}

struct Audit {
    spans: Buffer<u32>,
    work: Buffer<u32>,
    active: [Buffer<u32>; 3],
    receipt: [Buffer<u32>; 3],
    snapshot: [Vec<u32>; 3],
    active_snapshot: [[u32; SPANS]; 3],
    snapshotted: bool,
}
impl Audit {
    fn planned_payload(plan: Plan) -> usize {
        (SPANS * 4
            + 2 * GUARD
            + plan.chunks * 4
            + 2 * GUARD
            + 3 * (SPANS + 2 * GUARD)
            + 3 * (plan.chunks + 2 * GUARD)
            + 3 * plan.chunks)
            * 4
    }
    fn new(gpu: &Gpu, plan: Plan) -> Result<Self> {
        let make = |n| Buffer::new(gpu, n + 2 * GUARD);
        let mut spans = make(SPANS * 4)?;
        let mut work = make(plan.chunks * 4)?;
        spans.with_mut(|out| {
            out.fill(SENTINEL);
            for slot in 0..SPANS {
                out[GUARD + slot * 4..GUARD + slot * 4 + 4].copy_from_slice(&plan.span_words(slot));
            }
        });
        work.with_mut(|out| {
            out.fill(SENTINEL);
            plan.work(|i, w| out[GUARD + i * 4..GUARD + i * 4 + 4].copy_from_slice(&w));
        });
        let zero = || -> Result<Vec<u32>> {
            let mut v = Vec::new();
            v.try_reserve_exact(plan.chunks)
                .map_err(|_| Error::Native("pool receipt snapshot reservation failed".into()))?;
            v.resize(plan.chunks, SENTINEL);
            Ok(v)
        };
        Ok(Self {
            spans,
            work,
            active: [make(SPANS)?, make(SPANS)?, make(SPANS)?],
            receipt: [make(plan.chunks)?, make(plan.chunks)?, make(plan.chunks)?],
            snapshot: [zero()?, zero()?, zero()?],
            active_snapshot: [[0; SPANS]; 3],
            snapshotted: false,
        })
    }
    fn bytes(&self) -> usize {
        self.spans.len * 4
            + self.work.len * 4
            + self
                .active
                .iter()
                .chain(&self.receipt)
                .map(|b| b.len * 4)
                .sum::<usize>()
            + self
                .snapshot
                .iter()
                .map(|s| s.capacity() * 4)
                .sum::<usize>()
    }
    fn validate(&self, gpu: &Gpu, plan: Plan) -> Result<()> {
        validate_storage(gpu, &self.spans, SPANS * 4 + 2 * GUARD)?;
        validate_storage(gpu, &self.work, plan.chunks * 4 + 2 * GUARD)?;
        for b in &self.active {
            validate_storage(gpu, b, SPANS + 2 * GUARD)?;
        }
        for (b, s) in self.receipt.iter().zip(&self.snapshot) {
            validate_storage(gpu, b, plan.chunks + 2 * GUARD)?;
            if s.len() != plan.chunks {
                return Err(Error::Options("pool snapshot extent differs"));
            }
        }
        Ok(())
    }
    fn stage(&mut self, plan: Plan, layouts: [LayerLayout; 3], frame: &GraphFrame) {
        self.snapshotted = false;
        for (node, &layout) in layouts.iter().enumerate() {
            let active = plan
                .active(layout, frame.nodes[node])
                .expect("pool frame preflight");
            self.active_snapshot[node] = active;
            self.active[node].with_mut(|s| {
                s.fill(SENTINEL);
                s[GUARD..GUARD + SPANS].copy_from_slice(&active);
            });
            self.receipt[node].with_mut(|s| s.fill(SENTINEL));
        }
    }
    fn ranges<'a>(
        &'a self,
        plan: Plan,
        pool: &'a Buffer<u32>,
        node: usize,
    ) -> Result<[BufferRange<'a>; 5]> {
        Ok([
            BufferRange::new(&self.spans, GUARD, SPANS * 4)?,
            BufferRange::new(&self.work, GUARD, plan.chunks * 4)?,
            BufferRange::new(&self.active[node], GUARD, SPANS)?,
            BufferRange::new(pool, 0, plan.words)?,
            BufferRange::new(&self.receipt[node], GUARD, plan.chunks)?,
        ])
    }
    fn descriptors(&self, plan: Plan, bytes: &mut usize) -> bool {
        let mut valid = true;
        self.spans.with_slice(|s| {
            valid &= sentinel(&s[..GUARD], bytes) & sentinel(&s[GUARD + SPANS * 4..], bytes);
            for i in 0..SPANS {
                *bytes += 16;
                valid &= s[GUARD + i * 4..GUARD + i * 4 + 4] == plan.span_words(i);
            }
        });
        self.work.with_slice(|s| {
            valid &= sentinel(&s[..GUARD], bytes) & sentinel(&s[GUARD + plan.chunks * 4..], bytes);
            plan.work(|i, w| {
                *bytes += 16;
                valid &= s[GUARD + i * 4..GUARD + i * 4 + 4] == w;
            });
        });
        valid
    }
    fn integrity(&self, plan: Plan, bytes: &mut usize) -> Result<()> {
        let mut valid = self.descriptors(plan, bytes);
        for node in 0..3 {
            self.active[node].with_slice(|s| {
                valid &= sentinel(&s[..GUARD], bytes) & sentinel(&s[GUARD + SPANS..], bytes);
                *bytes += SPANS * 4;
                valid &= s[GUARD..GUARD + SPANS] == self.active_snapshot[node];
            });
            self.receipt[node].with_slice(|s| {
                valid &= sentinel(&s[..GUARD], bytes) & sentinel(&s[GUARD + plan.chunks..], bytes);
                *bytes += plan.chunks * 4;
                for &word in &s[GUARD..GUARD + plan.chunks] {
                    valid &= word & !2 == 0;
                }
            });
        }
        if valid {
            Ok(())
        } else {
            Err(Error::Options(
                "pool audit guard/descriptor/receipt integrity differs",
            ))
        }
    }
    fn snapshot(&mut self, plan: Plan) {
        for (b, snapshot) in self.receipt.iter().zip(&mut self.snapshot) {
            b.with_slice(|s| snapshot.copy_from_slice(&s[GUARD..GUARD + plan.chunks]));
        }
        self.snapshotted = true;
    }
    fn unchanged(&self, plan: Plan) -> bool {
        self.snapshotted
            && self
                .receipt
                .iter()
                .zip(&self.snapshot)
                .all(|(b, snapshot)| {
                    b.with_slice(|s| s[GUARD..GUARD + plan.chunks] == snapshot[..])
                })
    }
    fn candidate_finite(&self, plan: Plan) -> bool {
        self.snapshotted
            && self.snapshot.iter().all(|s| {
                s[..plan.chunks]
                    .iter()
                    .all(|w| w & pool_audit::NONFINITE == 0)
            })
    }
}

struct Persistent {
    state: Buffer<u32>,
    regions: [Option<Region>; 3],
    status: Buffer<u32>,
    rms_reasons: Buffer<u32>,
}
impl Persistent {
    fn plan(layout: LayerLayout) -> ([Option<Region>; 3], usize) {
        let mut words = 0;
        let regions = std::array::from_fn(|i| {
            if i != 0 && !layout.producer {
                return None;
            }
            let len = layout.rows * if i == 0 { H } else { layout.d };
            let r = Region { start: words, len };
            words = r.end() + GUARD;
            Some(r)
        });
        (regions, words)
    }
    fn planned_payload(layout: LayerLayout) -> usize {
        (Self::plan(layout).1 + STATUS_WORDS + GUARD + layout.rms_plan().1) * 4
    }
    fn new(gpu: &Gpu, layout: LayerLayout) -> Result<Self> {
        let (regions, words) = Self::plan(layout);
        Ok(Self {
            state: Buffer::new(gpu, words)?,
            regions,
            status: Buffer::new(gpu, STATUS_WORDS + GUARD)?,
            rms_reasons: Buffer::new(gpu, layout.rms_plan().1)?,
        })
    }
    fn region(&self, state: State) -> Option<Region> {
        let i = match state {
            State::Output => 0,
            State::KProposal => 1,
            State::VProposal => 2,
            _ => return None,
        };
        self.regions[i]
    }
    fn active(state: State, layout: LayerLayout, frame: NodeFrame) -> usize {
        if state == State::Output {
            frame.rows * H
        } else {
            frame.proposal_rows * layout.d
        }
    }
    fn range(
        &self,
        state: State,
        layout: LayerLayout,
        frame: NodeFrame,
    ) -> Result<BufferRange<'_>> {
        self.region(state)
            .ok_or(Error::Options("inactive persistent pool state"))?
            .range(&self.state, Self::active(state, layout, frame))
    }
    fn bytes(&self) -> usize {
        (self.state.len + self.status.len + self.rms_reasons.len) * 4
    }
    fn validate(&self, gpu: &Gpu, layout: LayerLayout) -> Result<()> {
        let (regions, words) = Self::plan(layout);
        for (a, b) in self.regions.iter().zip(regions) {
            if a.map(|r| (r.start, r.len)) != b.map(|r| (r.start, r.len)) {
                return Err(Error::Options("pool persistent layout differs"));
            }
        }
        validate_storage(gpu, &self.state, words)?;
        validate_storage(gpu, &self.status, STATUS_WORDS + GUARD)?;
        validate_storage(gpu, &self.rms_reasons, layout.rms_plan().1)
    }
    fn reset(&mut self) {
        self.state.with_mut(|s| s.fill(SENTINEL));
        self.status.with_mut(|s| {
            s.fill(SENTINEL);
            s[..STATUS_WORDS].fill(0);
        });
        self.rms_reasons.with_mut(|s| s.fill(SENTINEL));
    }
    fn guards(&self, layout: LayerLayout, frame: NodeFrame, bytes: &mut usize) -> bool {
        let mut valid = self.state.with_slice(|s| {
            let mut valid = true;
            for state in [State::Output, State::KProposal, State::VProposal] {
                if let Some(r) = self.region(state) {
                    valid &= sentinel(
                        &s[r.start + Self::active(state, layout, frame)..r.end() + GUARD],
                        bytes,
                    );
                }
            }
            valid
        });
        valid &= tail(&self.status, STATUS_WORDS, bytes);
        self.rms_reasons.with_slice(|s| {
            valid &= sentinel(&s[..GUARD], bytes);
            for (slot, region) in layout.rms_plan().0.into_iter().enumerate() {
                if let Some(region) = region {
                    let active = frame.rows * if slot == 1 { 8 } else { 1 };
                    valid &= sentinel(&s[region.start + active..region.end() + GUARD], bytes);
                }
            }
        });
        valid
    }
    fn snapshot_rms(
        &self,
        layout: LayerLayout,
        frame: NodeFrame,
        out: &mut [u32; MAX_RMS_REASONS],
    ) -> Result<usize> {
        out.fill(0);
        self.rms_reasons.with_slice(|s| {
            let mut cursor = 0;
            for (slot, region) in layout.rms_plan().0.into_iter().enumerate() {
                if let Some(region) = region {
                    let active = frame.rows * if slot == 1 { 8 } else { 1 };
                    let values = &s[region.start..region.start + active];
                    if values
                        .iter()
                        .any(|&v| v & !rms::precision::KNOWN_REASON_BITS != 0)
                    {
                        return Err(Error::Options("chain RMS reason contract differs"));
                    }
                    out[cursor..cursor + active].copy_from_slice(values);
                    cursor += active;
                }
            }
            Ok(cursor * 4)
        })
    }
}
struct Diagnostic {
    buffers: [Buffer<u32>; 2],
    regions: [[Option<Region>; 32]; 2],
}
impl Diagnostic {
    fn plan(layout: LayerLayout) -> ([Option<Region>; 32], usize) {
        let mut words = 0;
        let regions = STATES.map(|state| {
            slot(state)?;
            let len = layout.region(state).len;
            let r = Region { start: words, len };
            words = r.end() + GUARD;
            Some(r)
        });
        (regions, words)
    }
    fn planned_payload(layouts: [LayerLayout; 3]) -> usize {
        (Self::plan(layouts[0]).1 + Self::plan(layouts[1]).1) * 4
    }
    fn new(gpu: &Gpu, layouts: [LayerLayout; 3]) -> Result<Self> {
        let (a, na) = Self::plan(layouts[0]);
        let (b, nb) = Self::plan(layouts[1]);
        Ok(Self {
            buffers: [Buffer::new(gpu, na)?, Buffer::new(gpu, nb)?],
            regions: [a, b],
        })
    }
    fn bytes(&self) -> usize {
        self.buffers.iter().map(|b| b.len * 4).sum()
    }
    fn reset(&mut self) {
        for b in &mut self.buffers {
            b.with_mut(|s| s.fill(SENTINEL));
        }
    }
    fn validate(&self, gpu: &Gpu, layouts: [LayerLayout; 3]) -> Result<()> {
        for (node, &layout) in layouts.iter().take(2).enumerate() {
            let (r, n) = Self::plan(layout);
            if self.regions[node]
                .iter()
                .zip(r)
                .any(|(a, b)| a.map(|v| (v.start, v.len)) != b.map(|v| (v.start, v.len)))
            {
                return Err(Error::Options("pool diagnostic snapshot plan differs"));
            }
            validate_storage(gpu, &self.buffers[node], n)?;
        }
        Ok(())
    }
    fn guards(&self, layouts: [LayerLayout; 3], frame: &GraphFrame, bytes: &mut usize) -> bool {
        let mut valid = true;
        for (node, &layout) in layouts.iter().take(2).enumerate() {
            self.buffers[node].with_slice(|s| {
                for state in STATES {
                    if let Some(r) = self.regions[node][state as usize] {
                        let n = layout.active(state, frame.nodes[node]);
                        valid &= sentinel(&s[r.start + n..r.end() + GUARD], bytes);
                    }
                }
            });
        }
        valid
    }
    fn copy_retired(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        plan: Plan,
        pool: &Buffer<u32>,
        layouts: [LayerLayout; 3],
        frame: &GraphFrame,
        node: usize,
    ) -> Result<(usize, usize)> {
        if node >= 2 {
            return Ok((0, 0));
        }
        let mut copies = 0;
        let mut bytes = 0;
        let encoder = command
            .blitCommandEncoder()
            .ok_or_else(|| Error::Native("pool diagnostic blit encoder absent".into()))?;
        for state in STATES {
            if let (Some(slot), Some(dst)) = (slot(state), self.regions[node][state as usize]) {
                let n = layouts[node].active(state, frame.nodes[node]);
                if n == 0 {
                    continue;
                }
                let src = plan.spans[slot];
                // SAFETY: every range was checked before any encoder, disjoint
                // tracked same-device owned diagnostic destinations, retained
                // command, and the copy occurs before audit/fill retires data.
                unsafe {
                    encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        &pool.raw,
                        src.start * 4,
                        &self.buffers[node].raw,
                        dst.start * 4,
                        n * 4,
                    );
                }
                copies += 1;
                bytes += n * 4;
            }
        }
        encoder.endEncoding();
        Ok((copies, bytes))
    }
}
pub(super) struct Pooled {
    plan: Plan,
    data: Buffer<u32>,
    persistent: [Persistent; 3],
    audit: Audit,
    diagnostic: Option<Diagnostic>,
    replayed: bool,
}
impl Pooled {
    fn planned_payload(layouts: [LayerLayout; 3], diagnostic: bool) -> Result<usize> {
        let p = Plan::new(layouts)?;
        Ok(p.words * 4
            + layouts
                .into_iter()
                .map(Persistent::planned_payload)
                .sum::<usize>()
            + Audit::planned_payload(p)
            + if diagnostic {
                Diagnostic::planned_payload(layouts)
            } else {
                0
            })
    }
    fn new(gpu: &Gpu, layouts: [LayerLayout; 3], diagnostic: bool) -> Result<Self> {
        let plan = Plan::new(layouts)?;
        Ok(Self {
            plan,
            data: Buffer::new(gpu, plan.words)?,
            persistent: [
                Persistent::new(gpu, layouts[0])?,
                Persistent::new(gpu, layouts[1])?,
                Persistent::new(gpu, layouts[2])?,
            ],
            audit: Audit::new(gpu, plan)?,
            diagnostic: if diagnostic {
                Some(Diagnostic::new(gpu, layouts)?)
            } else {
                None
            },
            replayed: false,
        })
    }
    fn bytes(&self) -> usize {
        self.data.len * 4
            + self.persistent.iter().map(Persistent::bytes).sum::<usize>()
            + self.audit.bytes()
            + self.diagnostic.as_ref().map_or(0, Diagnostic::bytes)
    }
    fn validate(&self, gpu: &Gpu, layouts: [LayerLayout; 3]) -> Result<()> {
        let expected = Plan::new(layouts)?;
        if expected.words != self.plan.words
            || expected.chunks != self.plan.chunks
            || expected
                .spans
                .iter()
                .zip(self.plan.spans)
                .any(|(a, b)| a.start != b.start || a.len != b.len)
        {
            return Err(Error::Options("pool cold plan differs"));
        }
        self.plan.coverage()?;
        validate_storage(gpu, &self.data, self.plan.words)?;
        for (node, p) in self.persistent.iter().enumerate() {
            p.validate(gpu, layouts[node])?;
        }
        self.audit.validate(gpu, self.plan)?;
        if let Some(d) = &self.diagnostic {
            d.validate(gpu, layouts)?;
        }
        Ok(())
    }
    fn reset(&mut self, layouts: [LayerLayout; 3], frame: &GraphFrame) {
        self.replayed = false;
        self.data.with_mut(|s| s.fill(SENTINEL));
        for p in &mut self.persistent {
            p.reset();
        }
        self.audit.stage(self.plan, layouts, frame);
        if let Some(d) = &mut self.diagnostic {
            d.reset();
        }
    }
}

/// Explicit qualification-only fault at the retirement boundary, after all
/// numerical writers and before diagnostic copy/audit/fill. No CPU/GPU race.
#[derive(Clone, Copy)]
pub(super) struct RetirementFault {
    pub node: usize,
    pub state: State,
    pub inactive_guard: bool,
    pub byte: u8,
}

pub(super) enum Storage {
    Separate(Box<[LayerStorage; 3]>),
    Pooled(Box<Pooled>),
}
impl Storage {
    pub fn planned_payload(
        layouts: [LayerLayout; 3],
        pooled: bool,
        diagnostic: bool,
    ) -> Result<usize> {
        if pooled {
            add(
                size_of::<Pooled>(),
                Pooled::planned_payload(layouts, diagnostic)?,
            )
        } else {
            let buffers = layouts
                .into_iter()
                .map(|l| LayerStorage::planned_bytes(l).map(|b| b - size_of::<LayerStorage>()))
                .try_fold(0, |a, b| add(a, b?))?;
            add(size_of::<[LayerStorage; 3]>(), buffers)
        }
    }
    pub fn new(
        gpu: &Gpu,
        layouts: [LayerLayout; 3],
        pooled: bool,
        diagnostic: bool,
    ) -> Result<Self> {
        Ok(if pooled {
            Self::Pooled(Box::new(Pooled::new(gpu, layouts, diagnostic)?))
        } else {
            Self::Separate(Box::new([
                LayerStorage::new(gpu, layouts[0])?,
                LayerStorage::new(gpu, layouts[1])?,
                LayerStorage::new(gpu, layouts[2])?,
            ]))
        })
    }
    pub fn pooled(&self) -> bool {
        matches!(self, Self::Pooled(_))
    }
    pub fn payload_bytes(&self) -> usize {
        match self {
            Self::Separate(s) => {
                size_of::<[LayerStorage; 3]>()
                    + s.iter()
                        .map(|s| s.bytes() - size_of::<LayerStorage>())
                        .sum::<usize>()
            }
            Self::Pooled(p) => size_of::<Pooled>() + p.bytes(),
        }
    }
    pub fn separate(&self) -> Result<&[LayerStorage; 3]> {
        match self {
            Self::Separate(s) => Ok(s),
            _ => Err(Error::Options(
                "explicit unpooled diagnostic owner required",
            )),
        }
    }
    pub fn separate_mut(&mut self) -> Result<&mut [LayerStorage; 3]> {
        match self {
            Self::Separate(s) => Ok(s),
            _ => Err(Error::Options(
                "explicit unpooled diagnostic owner required",
            )),
        }
    }
    pub fn validate(&self, gpu: &Gpu, layouts: [LayerLayout; 3]) -> Result<()> {
        match self {
            Self::Separate(s) => {
                for (s, l) in s.iter().zip(layouts) {
                    s.validate(gpu, l)?;
                }
                Ok(())
            }
            Self::Pooled(p) => p.validate(gpu, layouts),
        }
    }
    pub fn initialize(&mut self) {
        match self {
            Self::Separate(s) => {
                for s in s.iter_mut() {
                    s.reset();
                }
            }
            Self::Pooled(p) => {
                p.data.with_mut(|s| s.fill(SENTINEL));
                for n in &mut p.persistent {
                    n.reset();
                }
                for b in p.audit.active.iter_mut().chain(&mut p.audit.receipt) {
                    b.with_mut(|s| s.fill(SENTINEL));
                }
                if let Some(d) = &mut p.diagnostic {
                    d.reset();
                }
            }
        }
    }
    pub fn reset(&mut self, layouts: [LayerLayout; 3], frame: &GraphFrame) -> usize {
        match self {
            Self::Separate(s) => {
                let bytes = s.iter().flat_map(|s| s.buffers()).map(|b| b.len * 4).sum();
                for s in s.iter_mut() {
                    s.reset();
                }
                bytes
            }
            Self::Pooled(p) => {
                let bytes = p.data.len * 4
                    + p.persistent.iter().map(Persistent::bytes).sum::<usize>()
                    + p.audit
                        .active
                        .iter()
                        .chain(&p.audit.receipt)
                        .map(|b| b.len * 4)
                        .sum::<usize>()
                    + p.diagnostic.as_ref().map_or(0, Diagnostic::bytes);
                p.reset(layouts, frame);
                bytes
            }
        }
    }
    pub fn capture_audit<T>(
        &self,
        read: impl FnOnce(usize, usize, &[u32], &[u32], [&[u32]; 3], [&[u32]; 3]) -> T,
    ) -> Option<T> {
        let Self::Pooled(p) = self else { return None };
        Some(p.audit.spans.with_slice(|spans| {
            p.audit.work.with_slice(|work| {
                p.audit.active[0].with_slice(|a| {
                    p.audit.active[1].with_slice(|b| {
                        p.audit.active[2].with_slice(|c| {
                            p.audit.receipt[0].with_slice(|x| {
                                p.audit.receipt[1].with_slice(|y| {
                                    p.audit.receipt[2].with_slice(|z| {
                                        read(
                                            p.plan.words,
                                            p.plan.chunks,
                                            spans,
                                            work,
                                            [a, b, c],
                                            [x, y, z],
                                        )
                                    })
                                })
                            })
                        })
                    })
                })
            })
        }))
    }
    pub fn mutate_control(&mut self, kind: usize) -> Result<()> {
        let Self::Pooled(p) = self else {
            return Err(Error::Options("pool fault requires pooled owner"));
        };
        match kind {
            0 => p.audit.receipt[0].with_mut(|s| s[GUARD] = SENTINEL),
            1 => p.audit.receipt[2].with_mut(|s| s[GUARD] = 8),
            2 => p.audit.spans.with_mut(|s| s[GUARD] ^= 1),
            3 => p.audit.work.with_mut(|s| s[GUARD + 2] ^= 1),
            4 => p.audit.active[1].with_mut(|s| s[GUARD] ^= 1),
            5 => p.audit.receipt[1].with_mut(|s| s[GUARD] ^= pool_audit::NONFINITE),
            6 => p.data.with_mut(|s| s[p.plan.spans[0].end()] = 0),
            7 => {
                let start = p.persistent[0].regions[1].unwrap().start;
                p.persistent[0]
                    .state
                    .with_mut(|s| s[start] = f32::NAN.to_bits());
            }
            10 => p
                .data
                .with_mut(|s| s[p.plan.spans[0].start] = f32::NAN.to_bits()),
            _ => return Err(Error::Options("unknown bounded pool fault")),
        }
        Ok(())
    }
    pub fn pool_words(&self) -> usize {
        match self {
            Self::Separate(_) => 0,
            Self::Pooled(p) => p.plan.words,
        }
    }
    pub fn status(&self, node: usize) -> &Buffer<u32> {
        match self {
            Self::Separate(s) => &s[node].status,
            Self::Pooled(p) => &p.persistent[node].status,
        }
    }
    pub fn status_mut(&mut self, node: usize) -> &mut Buffer<u32> {
        match self {
            Self::Separate(s) => &mut s[node].status,
            Self::Pooled(p) => &mut p.persistent[node].status,
        }
    }
    pub fn reasons(&self, node: usize) -> &Buffer<u32> {
        match self {
            Self::Separate(s) => &s[node].rms_reasons,
            Self::Pooled(p) => &p.persistent[node].rms_reasons,
        }
    }
    pub fn reasons_mut(&mut self, node: usize) -> &mut Buffer<u32> {
        match self {
            Self::Separate(s) => &mut s[node].rms_reasons,
            Self::Pooled(p) => &mut p.persistent[node].rms_reasons,
        }
    }
    fn hidden<'a>(&'a self, view: View<'a>, node: usize) -> Result<(&'a Buffer<u32>, Region)> {
        let View {
            entry,
            entry_regions,
            frame,
            ..
        } = view;
        let rows = frame.nodes[node].rows;
        if node == 0 {
            let r = entry_regions[0];
            let n = rows * H;
            if n > r.len {
                return Err(Error::Options("pool entry hidden extent differs"));
            }
            return Ok((
                entry,
                Region {
                    start: r.start,
                    len: n,
                },
            ));
        }
        let Self::Pooled(p) = self else {
            return Err(Error::Options("pool hidden view used on separate owner"));
        };
        let previous = &p.persistent[node - 1];
        let r = previous
            .region(State::Output)
            .ok_or(Error::Options("pool previous boundary output absent"))?;
        let dropped = if node == 2 && frame.mode == OutputRows::Last {
            frame.rows - 1
        } else {
            0
        };
        let start = r
            .start
            .checked_add(dropped * H)
            .ok_or(Error::Options("pool hidden offset overflow"))?;
        if start.checked_add(rows * H).is_none_or(|end| end > r.end()) {
            return Err(Error::Options("pool previous boundary range differs"));
        }
        Ok((
            &previous.state,
            Region {
                start,
                len: rows * H,
            },
        ))
    }
    pub fn range<'a>(
        &'a self,
        view: View<'a>,
        node: usize,
        state: State,
    ) -> Result<Option<BufferRange<'a>>> {
        let View { layouts, frame, .. } = view;
        let layout = layouts[node];
        let f = frame.nodes[node];
        if layout.active(state, f) == 0 {
            return Ok(None);
        }
        match self {
            Self::Separate(s) => Ok(Some(layout.range(&s[node].arena, state, f)?)),
            Self::Pooled(p) => {
                if state == State::Hidden {
                    let (b, r) = self.hidden(view, node)?;
                    return Ok(Some(r.range(b, r.len)?));
                }
                if slot(state).is_some() {
                    p.plan.range(&p.data, state, layout, f)
                } else {
                    Ok(Some(p.persistent[node].range(state, layout, f)?))
                }
            }
        }
    }
    pub fn states<'a>(
        &'a self,
        view: View<'a>,
        node: usize,
    ) -> Result<[Option<BufferRange<'a>>; 32]> {
        if let Self::Separate(s) = self {
            return view.layouts[node].states(&s[node].arena, view.frame.nodes[node]);
        }
        let mut out = [None; 32];
        for state in STATES {
            out[state as usize] = self.range(view, node, state)?;
        }
        Ok(out)
    }
    pub fn flags(
        &self,
        layout: LayerLayout,
        frame: NodeFrame,
        node: usize,
    ) -> Result<[BufferRange<'_>; 2]> {
        match self {
            Self::Separate(s) => Ok([
                BufferRange::new(&s[node].flags[0], 0, frame.rows * layout.ff)?,
                BufferRange::new(&s[node].flags[1], 0, frame.rows * P)?,
            ]),
            Self::Pooled(p) => Ok([
                p.plan.spans[28].range(&p.data, frame.rows * layout.ff)?,
                p.plan.spans[29].range(&p.data, frame.rows * P)?,
            ]),
        }
    }
    pub fn intervals(
        &self,
        layout: LayerLayout,
        frame: NodeFrame,
        node: usize,
    ) -> Result<[BufferRange<'_>; 2]> {
        match self {
            Self::Separate(s) => Ok([
                BufferRange::new(&s[node].intervals[0], 0, 8 * frame.rows * layout.ff)?,
                BufferRange::new(&s[node].intervals[1], 0, 8 * frame.rows * P)?,
            ]),
            Self::Pooled(p) => Ok([
                p.plan.spans[30].range(&p.data, 8 * frame.rows * layout.ff)?,
                p.plan.spans[31].range(&p.data, 8 * frame.rows * P)?,
            ]),
        }
    }
    pub fn visit_buffers(&self, mut visit: impl FnMut(&Buffer<u32>) -> Result<()>) -> Result<()> {
        match self {
            Self::Separate(s) => {
                for s in s.iter() {
                    for b in s.buffers() {
                        visit(b)?;
                    }
                }
            }
            Self::Pooled(p) => {
                visit(&p.data)?;
                for n in &p.persistent {
                    for b in [&n.state, &n.status, &n.rms_reasons] {
                        visit(b)?;
                    }
                }
                for b in [&p.audit.spans, &p.audit.work] {
                    visit(b)?;
                }
                for b in p.audit.active.iter().chain(&p.audit.receipt) {
                    visit(b)?;
                }
                if let Some(d) = &p.diagnostic {
                    for b in &d.buffers {
                        visit(b)?;
                    }
                }
            }
        }
        Ok(())
    }
    pub fn preflight(
        &self,
        graph: &ResidentGraph<'_>,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        frame: &GraphFrame,
    ) -> Result<usize> {
        let mut read_bytes = 0;
        if let Self::Pooled(p) = self {
            let pipeline = graph
                .pipelines
                .residual
                .pool_audit
                .as_ref()
                .ok_or(Error::Options("pool audit pipeline absent"))?;
            let layouts = graph.plans.each_ref().map(|p| p.layout);
            p.plan.coverage()?;

            if !p.audit.descriptors(p.plan, &mut read_bytes) {
                return Err(Error::Options("pool descriptors changed before encode"));
            }
            for (node, &layout) in layouts.iter().enumerate() {
                p.plan.active(layout, frame.nodes[node])?;
                if let Some(d) = &p.diagnostic
                    && node < 2
                {
                    for state in STATES {
                        if let (Some(slot), Some(dst)) =
                            (slot(state), d.regions[node][state as usize])
                        {
                            let n = layout.active(state, frame.nodes[node]);
                            if n == 0 {
                                continue;
                            }
                            let source = p.plan.spans[slot].range(&p.data, n)?;
                            let target = dst.range(&d.buffers[node], n)?;
                            crate::metal::memory::validate_encode_ranges(
                                command,
                                &graph.context.0.gpu.device,
                                &[source, target],
                                1,
                            )?;
                        }
                    }
                }
                let [spans, work, active, pool, receipt] = p.audit.ranges(p.plan, &p.data, node)?;
                pipeline.preflight(
                    command,
                    spans,
                    work,
                    active,
                    pool,
                    receipt,
                    pool_audit::Shape {
                        pool_words: p.plan.words,
                        chunks: p.plan.chunks,
                        fill_after: node < 2,
                    },
                )?;
            }
        }
        Ok(read_bytes)
    }
    pub fn preflight_fault(
        &self,
        graph: &ResidentGraph<'_>,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        fault: Option<RetirementFault>,
    ) -> Result<()> {
        let Some(fault) = fault else { return Ok(()) };
        let Self::Pooled(p) = self else {
            return Err(Error::Options("retirement fault requires pool"));
        };
        if fault.node >= 3 || !matches!(fault.byte, 0 | 255) {
            return Err(Error::Options("retirement fault outside bounded controls"));
        }
        let span = p.plan.spans
            [slot(fault.state).ok_or(Error::Options("fault must name pooled state"))?];
        let offset = span.start + if fault.inactive_guard { span.len } else { 0 };
        crate::metal::memory::validate_encode_ranges(
            command,
            &graph.context.0.gpu.device,
            &[BufferRange::new(&p.data, offset, 1)?],
            0,
        )
    }
    pub fn encode_fault(
        &self,
        graph: &ResidentGraph<'_>,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        fault: Option<RetirementFault>,
        node: usize,
    ) -> Result<()> {
        let Some(fault) = fault.filter(|fault| fault.node == node) else {
            return Ok(());
        };
        self.preflight_fault(graph, command, Some(fault))?;
        let Self::Pooled(p) = self else {
            unreachable!()
        };
        let span = p.plan.spans[slot(fault.state).unwrap()];
        let offset = span.start + if fault.inactive_guard { span.len } else { 0 };
        let encoder = command
            .blitCommandEncoder()
            .ok_or_else(|| Error::Native("retirement fault blit unavailable".into()))?;
        // SAFETY: one checked same-device shared tracked word, caller command
        // remains retained, and the next ordered audit observes this write.
        encoder.fillBuffer_range_value(
            &p.data.raw,
            objc2_foundation::NSRange::new(offset * 4, 4),
            fault.byte,
        );
        encoder.endEncoding();
        Ok(())
    }
    pub fn copy_retired(
        &self,
        graph: &ResidentGraph<'_>,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        frame: &GraphFrame,
        node: usize,
    ) -> Result<(usize, usize)> {
        let Self::Pooled(p) = self else {
            return Ok((0, 0));
        };
        let layouts = graph.plans.each_ref().map(|p| p.layout);
        if let Some(d) = &p.diagnostic {
            d.copy_retired(command, p.plan, &p.data, layouts, frame, node)
        } else {
            Ok((0, 0))
        }
    }
    pub fn encode_retirement(
        &self,
        graph: &ResidentGraph<'_>,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        node: usize,
    ) -> Result<()> {
        let Self::Pooled(p) = self else { return Ok(()) };
        let pipeline = graph
            .pipelines
            .residual
            .pool_audit
            .as_ref()
            .ok_or(Error::Options("pool audit pipeline absent"))?;
        let [spans, work, active, pool, receipt] = p.audit.ranges(p.plan, &p.data, node)?;
        pipeline.encode(
            command,
            spans,
            work,
            active,
            pool,
            receipt,
            pool_audit::Shape {
                pool_words: p.plan.words,
                chunks: p.plan.chunks,
                fill_after: node < 2,
            },
        )?;
        Ok(())
    }
    pub fn audit_integrity(
        &self,
        layouts: [LayerLayout; 3],
        frame: &GraphFrame,
        bytes: &mut usize,
    ) -> Result<()> {
        if let Self::Pooled(p) = self {
            p.audit.integrity(p.plan, bytes)?;
            if let Some(d) = &p.diagnostic
                && !d.guards(layouts, frame, bytes)
            {
                return Err(Error::Options("pool diagnostic snapshot guard changed"));
            }
        }
        Ok(())
    }
    pub fn guards_node(
        &self,
        layout: LayerLayout,
        frame: NodeFrame,
        node: usize,
        bytes: &mut usize,
    ) -> bool {
        match self {
            Self::Separate(s) => s[node].guards(layout, frame, bytes),
            Self::Pooled(p) => {
                let mut valid = p.persistent[node].guards(layout, frame, bytes);
                if node == 2 {
                    let active = p.plan.active(layout, frame).expect("pool frame preflight");
                    p.data.with_slice(|s| {
                        for (slot, r) in p.plan.spans.iter().enumerate() {
                            valid &= sentinel(
                                &s[r.start + active[slot] as usize..r.end() + GUARD],
                                bytes,
                            );
                        }
                    });
                }
                valid
            }
        }
    }
    pub fn snapshot_rms(
        &self,
        layout: LayerLayout,
        frame: NodeFrame,
        node: usize,
        out: &mut [u32; MAX_RMS_REASONS],
    ) -> Result<usize> {
        match self {
            Self::Separate(s) => s[node].snapshot_rms(layout, frame, out),
            Self::Pooled(p) => p.persistent[node].snapshot_rms(layout, frame, out),
        }
    }
    pub fn snapshot_audits(&mut self) -> usize {
        if let Self::Pooled(p) = self {
            p.audit.snapshot(p.plan);
            3 * p.plan.chunks * 4
        } else {
            0
        }
    }
    pub fn unchanged_audits(&self) -> bool {
        match self {
            Self::Separate(_) => true,
            Self::Pooled(p) => p.audit.unchanged(p.plan),
        }
    }
    pub fn live_state(
        &self,
        layout: LayerLayout,
        frame: NodeFrame,
        node: usize,
        state: State,
    ) -> Result<(&Buffer<u32>, Region)> {
        if !matches!(state, State::Output | State::KProposal | State::VProposal) {
            return Err(Error::Options("publication state is not persistent"));
        }
        let n = layout.active(state, frame);
        let (buffer, r) = match self {
            Self::Separate(s) => (&s[node].arena, layout.region(state)),
            Self::Pooled(p) => (
                &p.persistent[node].state,
                p.persistent[node]
                    .region(state)
                    .ok_or(Error::Options("publication persistent state absent"))?,
            ),
        };
        r.range(buffer, n)?;
        Ok((
            buffer,
            Region {
                start: r.start,
                len: n,
            },
        ))
    }
    pub fn with_state<T>(
        &self,
        view: View<'_>,
        node: usize,
        state: State,
        read: impl FnOnce(&[u32]) -> T,
    ) -> Result<T> {
        let View { layouts, frame, .. } = view;
        let layout = layouts[node];
        let f = frame.nodes[node];
        let n = layout.active(state, f);
        if n == 0 {
            return Ok(read(&[]));
        }
        match self {
            Self::Separate(s) => {
                let r = layout.region(state);
                Ok(s[node].arena.with_slice(|s| read(&s[r.start..r.start + n])))
            }
            Self::Pooled(p) => {
                if state == State::Hidden {
                    let (b, r) = self.hidden(view, node)?;
                    return Ok(b.with_slice(|s| read(&s[r.start..r.end()])));
                }
                if let Some(slot) = slot(state) {
                    if node == 2 {
                        let r = p.plan.spans[slot];
                        return Ok(p.data.with_slice(|s| read(&s[r.start..r.start + n])));
                    }
                    let diagnostic = p.diagnostic.as_ref().ok_or(Error::Options(
                        "retired state requires explicit pool diagnostics",
                    ))?;
                    let r = diagnostic.regions[node][state as usize]
                        .ok_or(Error::Options("retired diagnostic state absent"))?;
                    Ok(diagnostic.buffers[node].with_slice(|s| read(&s[r.start..r.start + n])))
                } else {
                    let r = p.persistent[node]
                        .region(state)
                        .ok_or(Error::Options("persistent pool state absent"))?;
                    Ok(p.persistent[node]
                        .state
                        .with_slice(|s| read(&s[r.start..r.start + n])))
                }
            }
        }
    }
    pub fn finite(&self, view: View<'_>, bytes: &mut usize) -> Result<bool> {
        let View { layouts, frame, .. } = view;
        if let Self::Separate(s) = self {
            return Ok(s.iter().enumerate().fold(true, |ok, (node, s)| {
                s.finite(layouts[node], frame.nodes[node], bytes) & ok
            }));
        }
        let Self::Pooled(p) = self else {
            unreachable!()
        };
        let mut finite = p.replayed || p.audit.candidate_finite(p.plan);
        if !p.replayed {
            *bytes += 3 * p.plan.chunks * 4;
        }
        for (node, &layout) in layouts.iter().enumerate() {
            for state in STATES {
                if slot(state).is_some() && node != 2 {
                    continue;
                }
                let n = layout.active(state, frame.nodes[node]);
                *bytes += n * 4;
                finite &= self.with_state(view, node, state, |s| {
                    s.iter().all(|&w| f32::from_bits(w).is_finite())
                })?;
            }
        }
        Ok(finite)
    }
    pub fn upload_reference(
        &mut self,
        reference: &reference::Runner,
        graph: &ResidentGraph<'_>,
        frame: &GraphFrame,
    ) -> (usize, usize, usize) {
        match self {
            Self::Separate(s) => (reference.upload(graph, s.as_mut(), frame), 0, 0),
            Self::Pooled(p) => {
                let layouts = graph.plans.each_ref().map(|p| p.layout);
                let mut upload = 0;
                let mut diagnostic = 0;
                p.data.with_mut(|out| {
                    out.fill(SENTINEL);
                    for state in STATES {
                        if let Some(slot) = slot(state) {
                            let r = p.plan.spans[slot];
                            let n = layouts[2].active(state, frame.nodes[2]);
                            for (dst, x) in out[r.start..r.start + n]
                                .iter_mut()
                                .zip(&reference.states[2][state as usize][..n])
                            {
                                *dst = x.to_bits();
                            }
                            upload += n * 4;
                        }
                    }
                });
                for (node, &layout) in layouts.iter().enumerate() {
                    let regions = p.persistent[node].regions;
                    p.persistent[node].state.with_mut(|out| {
                        for (state, region) in [State::Output, State::KProposal, State::VProposal]
                            .into_iter()
                            .zip(regions)
                        {
                            if let Some(r) = region {
                                let n = layout.active(state, frame.nodes[node]);
                                for (dst, x) in out[r.start..r.start + n]
                                    .iter_mut()
                                    .zip(&reference.states[node][state as usize][..n])
                                {
                                    *dst = x.to_bits();
                                }
                                upload += n * 4;
                            }
                        }
                    });
                }
                if let Some(d) = &mut p.diagnostic {
                    for (node, &layout) in layouts.iter().take(2).enumerate() {
                        let regions = d.regions[node];
                        d.buffers[node].with_mut(|out| {
                            for state in STATES {
                                if let Some(r) = regions[state as usize] {
                                    let n = layout.active(state, frame.nodes[node]);
                                    for (dst, x) in out[r.start..r.start + n]
                                        .iter_mut()
                                        .zip(&reference.states[node][state as usize][..n])
                                    {
                                        *dst = x.to_bits();
                                    }
                                    diagnostic += n * 4;
                                }
                            }
                        });
                    }
                }
                p.replayed = true;
                (upload, p.data.len * 4, diagnostic)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn layouts(rows: usize, keys: usize) -> [LayerLayout; 3] {
        [
            LayerLayout::new(rows, keys, 256, 6144, true).unwrap(),
            LayerLayout::new(rows, keys, 512, 6144, true).unwrap(),
            LayerLayout::new(rows, keys, 256, 12288, false).unwrap(),
        ]
    }
    #[test]
    fn pool_plan_partitions_all_words_and_keeps_every_node_disjoint() {
        let mut seen = [false; STATE_SPANS];
        for state in STATES {
            if let Some(i) = slot(state) {
                assert!(i < STATE_SPANS && !seen[i]);
                seen[i] = true;
            }
        }
        assert!(seen.into_iter().all(|x| x));
        for rows in [1, 4, 32] {
            let layouts = layouts(rows, 512);
            let plan = Plan::new(layouts).unwrap();
            let mut end = [0; SPANS];
            let mut records = 0;
            plan.work(|i, w| {
                assert_eq!(i, records);
                records += 1;
                let [slot, begin, count, version] = w;
                assert_eq!(version, ABI);
                assert_eq!(begin as usize, end[slot as usize]);
                assert!((1..=CHUNK as u32).contains(&count));
                end[slot as usize] += count as usize;
            });
            assert_eq!(records, plan.chunks);
            for (end, span) in end.into_iter().zip(plan.spans) {
                assert_eq!(end, span.len + GUARD);
            }
            let frames = [NodeFrame {
                rows,
                query_start: 0,
                key_start: 0,
                keys: rows,
                proposal_rows: rows,
                proposal_start: 0,
            }; 3];
            for (node, layout) in layouts.into_iter().enumerate() {
                let mut frame = frames[node];
                if node == 2 {
                    frame.rows = 1;
                    frame.query_start = rows - 1;
                }
                let active = plan.active(layout, frame).unwrap();
                for (n, span) in active.into_iter().zip(plan.spans) {
                    assert!(n as usize <= span.len);
                }
            }
            if rows == 32 {
                assert_eq!(plan.words, 6_594_784);
                assert_eq!(plan.chunks, 6472);
            }
        }
    }
    #[test]
    fn pool_rejects_noncanonical_roles_and_capacity_mismatch() {
        let mut bad = layouts(4, 512);
        bad[2] = LayerLayout::new(4, 512, 512, 12288, false).unwrap();
        assert!(Plan::new(bad).is_err());
        bad = layouts(4, 512);
        bad[1] = LayerLayout::new(3, 512, 512, 6144, true).unwrap();
        assert!(Plan::new(bad).is_err());
        bad = layouts(4, 512);
        bad[0] = LayerLayout::new(4, 511, 256, 6144, true).unwrap();
        assert!(Plan::new(bad).is_err());
    }
}
