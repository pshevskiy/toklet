//! One concrete operator schedule shared by the qualified pilot and chain.
//! Ranges borrow storage only for encoding; no guard, submission or publication.
use super::*;

#[derive(Clone, Copy)]
pub(super) enum Projection<'a> {
    Single(Bf16Tile<'a>),
    #[cfg(test)]
    Tiled {
        matrix: project::bf16_tiled::Bf16MatrixView<'a>,
        pipeline: &'a project::bf16_tiled::StridedBf16Pipelines,
    },
}
#[derive(Clone, Copy)]
pub(super) enum NormKind<'a> {
    Original(&'a rms::Pipelines),
    #[cfg(test)]
    Precision {
        pipelines: &'a rms::precision::Pipelines,
        reasons: [Option<BufferRange<'a>>; 8],
    },
}
#[derive(Clone, Copy)]
pub(super) struct Shape {
    pub rows: usize,
    pub d: usize,
    pub ff: usize,
    pub rotated: usize,
    pub query_start: usize,
    pub key_start: usize,
    pub keys: usize,
    pub window: Option<usize>,
    pub theta: f32,
    pub producer: bool,
}
pub(super) struct Recipe<'a> {
    pub context: &'a Bf16Context,
    pub pipelines: &'a Pipelines,
    pub table: &'a gelu::CertifiedTable,
    pub projections: [Option<Projection<'a>>; 9],
    pub admissions: [Bf16Tile<'a>; 6],
    pub states: [Option<BufferRange<'a>>; 32],
    pub norms: [Option<BufferRange<'a>>; 8],
    pub norm_kind: NormKind<'a>,
    pub status: BufferRange<'a>,
    pub coefficients: BufferRange<'a>,
    pub flags: [BufferRange<'a>; 2],
    pub intervals: [BufferRange<'a>; 2],
    pub segments: [attention::DeviceSegment<'a>; 3],
    pub shape: Shape,
    pub scalar: f32,
    pub dispatches: Cell<usize>,
    pub partial_count_unknown: Cell<bool>,
    #[cfg(test)]
    pub force_gelu_reference: bool,
}
pub(super) fn subrange(r: BufferRange<'_>, first: usize, len: usize) -> Result<BufferRange<'_>> {
    if first.checked_add(len).is_none_or(|end| end > r.len) {
        return Err(Error::Options("layer subrange exceeds admitted extent"));
    }
    Ok(BufferRange {
        raw: r.raw,
        offset: r
            .offset
            .checked_add(first)
            .ok_or(Error::Options("layer range offset overflow"))?,
        len,
    })
}
pub(super) fn values(
    state: State,
    rows: usize,
    keys: usize,
    d: usize,
    ff: usize,
    producer: bool,
) -> usize {
    match state {
        State::QRaw | State::QNorm | State::QRotary | State::Context => rows * 8 * d,
        State::KRaw | State::KNorm | State::KProposal | State::VRaw | State::VProposal => {
            if producer {
                rows * d
            } else {
                0
            }
        }
        State::FfGate | State::FfUp | State::FfActivation | State::FfProduct => rows * ff,
        State::PleInput | State::PleGate | State::PleActivation | State::PleProduct => rows * P,
        State::Scores | State::Probabilities => 8 * rows * keys,
        _ => rows * H,
    }
}
impl Recipe<'_> {
    fn r(&self, state: State) -> Result<BufferRange<'_>> {
        self.states[state as usize].ok_or(Error::Options("inactive layer state was requested"))
    }
    fn s(&self, slot: usize) -> Result<BufferRange<'_>> {
        subrange(self.status, slot, 1)
    }
    fn norm(&self, slot: usize) -> Result<BufferRange<'_>> {
        self.norms[slot].ok_or(Error::Options("inactive layer normalization was requested"))
    }
    fn record(&self, result: Result<()>, count: usize) -> Result<()> {
        if result.is_ok() {
            self.dispatches.set(self.dispatches.get() + count);
        }
        result
    }
    pub fn preflight(&self, command: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<()> {
        let s = self.shape;
        if !(1..=32).contains(&s.rows)
            || ![256, 512].contains(&s.d)
            || ![6144, 12288].contains(&s.ff)
            || !(1..=1025).contains(&s.keys)
            || ![128, 256].contains(&s.rotated)
            || s.rotated > s.d
            || s.rotated != if s.d == 512 { 128 } else { 256 }
            || s.theta != if s.d == 512 { 1000000. } else { 10000. }
            || s.window == Some(0)
            || s.key_start > s.query_start
            || s.query_start.checked_add(s.rows).is_none_or(|end| {
                s.key_start
                    .checked_add(s.keys)
                    .is_none_or(|limit| end > limit)
            })
            || !self.scalar.is_finite()
            || self.status.len != STATUS_WORDS
            || self.coefficients.len != s.rows * s.rotated
        {
            return Err(Error::Options("layer recipe geometry differs"));
        }
        let first = self.r(State::Hidden)?;
        let mut ranges = [first; 64];
        let mut n = 0;
        for state in [State::Hidden, State::PleInput] {
            ranges[n] = self.r(state)?;
            n += 1;
        }
        let widths = [H, s.d, s.d, H, H, H, H, 1];
        for (i, &width) in widths.iter().enumerate() {
            let active = s.producer || (i != 2 && i != 7);
            match (active, self.norms[i]) {
                (true, Some(range)) if range.len == width => {
                    ranges[n] = range;
                    n += 1;
                }
                (false, None) => (),
                _ => return Err(Error::Options("layer normalization extent differs")),
            }
        }
        ranges[n] = self.coefficients;
        n += 1;
        for (i, segment) in self.segments.iter().enumerate() {
            if i == 2 && s.producer {
                for (actual, state) in
                    [(segment.k, State::KProposal), (segment.v, State::VProposal)]
                {
                    let expected = self.r(state)?;
                    if !std::ptr::eq(actual.raw, expected.raw)
                        || actual.offset != expected.offset
                        || actual.len != expected.len
                    {
                        return Err(Error::Options("producer proposal range differs"));
                    }
                }
            } else {
                for range in [segment.k, segment.v] {
                    ranges[n] = range;
                    n += 1;
                }
            }
        }
        let read_only = n;
        for state in STATES {
            let expected = values(state, s.rows, s.keys, s.d, s.ff, s.producer);
            match (expected, self.states[state as usize]) {
                (0, None) => (),
                (len, Some(range)) if len != 0 && range.len == len => {
                    if state != State::Hidden && state != State::PleInput {
                        ranges[n] = range;
                        n += 1;
                    }
                }
                _ => return Err(Error::Options("layer state extent differs")),
            }
        }
        for (index, width) in [s.ff, P].into_iter().enumerate() {
            let len = s.rows * width;
            if self.flags[index].len != len || self.intervals[index].len != 8 * len {
                return Err(Error::Options("layer GELU workspace extent differs"));
            }
            ranges[n] = self.flags[index];
            n += 1;
            ranges[n] = self.intervals[index];
            n += 1;
        }
        ranges[n] = self.status;
        n += 1;
        #[cfg(test)]
        if let NormKind::Precision { reasons, .. } = self.norm_kind {
            for (slot, range) in reasons.into_iter().enumerate() {
                let active = s.producer || (slot != 2 && slot != 7);
                match (active, range) {
                    (true, Some(range)) if range.len == s.rows * if slot == 1 { 8 } else { 1 } => {
                        ranges[n] = range;
                        n += 1;
                    }
                    (false, None) => (),
                    _ => return Err(Error::Options("layer RMS reason extent differs")),
                }
            }
        }
        for (i, p) in self.projections.iter().enumerate() {
            if p.is_some() != (s.producer || (i != 1 && i != 2)) {
                return Err(Error::Options("layer projection role differs"));
            }
        }
        self.table.validate_device(&self.context.0.gpu)?;
        self.pipelines.gelu.validate_table(self.table)?;
        crate::metal::memory::validate_encode_ranges(
            command,
            &self.context.0.gpu.device,
            &ranges[..n],
            read_only,
        )
    }
    fn rms(
        &self,
        c: &ProtocolObject<dyn MTLCommandBuffer>,
        states: [State; 2],
        norm: usize,
        status: usize,
        shape: [usize; 2],
    ) -> Result<()> {
        let buffers = rms::DeviceBuffers {
            input: self.r(states[0])?,
            weights: self.norm(norm)?,
            output: self.r(states[1])?,
            status: self.s(status)?,
        };
        let result = match self.norm_kind {
            NormKind::Original(pipelines) => {
                pipelines.encode(c, &buffers, shape[0], shape[1], 1e-6, norm != 7)
            }
            #[cfg(test)]
            NormKind::Precision { pipelines, reasons } => pipelines.encode(
                c,
                &buffers,
                reasons[norm].ok_or(Error::Options("inactive RMS reason range"))?,
                shape[0],
                shape[1],
                norm != 7,
            ),
        };
        self.record(result, 1)
    }

    fn project(
        &self,
        c: &ProtocolObject<dyn MTLCommandBuffer>,
        m: usize,
        input: State,
        out: State,
        status: usize,
    ) -> Result<()> {
        let projection =
            self.projections[m].ok_or(Error::Options("inactive layer matrix was requested"))?;
        let (result, count) = match projection {
            Projection::Single(tile) => (
                project::encode_bf16(
                    &self.context.0,
                    c,
                    tile,
                    self.r(input)?,
                    self.r(out)?,
                    self.s(status)?,
                    self.shape.rows,
                ),
                1,
            ),
            #[cfg(test)]
            Projection::Tiled { matrix, pipeline } => {
                let result = pipeline.encode(
                    c,
                    matrix,
                    project::bf16_tiled::DeviceBuffers {
                        input: self.r(input)?,
                        output: self.r(out)?,
                        status: self.s(status)?,
                    },
                    self.shape.rows,
                );
                if result.is_err() {
                    self.partial_count_unknown.set(true);
                }
                (result, matrix.tile_count())
            }
        };
        self.record(result, count)
    }
    fn admit(
        &self,
        c: &ProtocolObject<dyn MTLCommandBuffer>,
        group: usize,
        input: State,
        status: usize,
    ) -> Result<()> {
        self.record(
            self.pipelines.admission.encode_with_l1(
                c,
                self.admissions[group],
                self.r(input)?,
                self.s(status)?,
                self.shape.rows,
            ),
            1,
        )
    }
    fn rotate(
        &self,
        c: &ProtocolObject<dyn MTLCommandBuffer>,
        input: State,
        out: State,
        heads: usize,
        status: usize,
    ) -> Result<()> {
        let s = self.shape;
        self.record(
            self.pipelines.rotary.encode(
                c,
                &rotary::DeviceBuffers {
                    input: self.r(input)?,
                    output: self.r(out)?,
                    status: self.s(status)?,
                    coefficients: self.coefficients,
                },
                &cpu::ProportionalRotaryShape {
                    tokens: s.rows,
                    heads,
                    head_dim: s.d,
                    rotated_dims: s.rotated,
                    offset: s.query_start,
                    base: s.theta,
                    factor: 1.,
                },
            ),
            1,
        )
    }
    fn add(
        &self,
        c: &ProtocolObject<dyn MTLCommandBuffer>,
        a: State,
        b: State,
        out: State,
        status: usize,
    ) -> Result<()> {
        self.record(
            self.pipelines.residual.encode_add(
                c,
                self.r(a)?,
                self.r(b)?,
                self.r(out)?,
                self.s(status)?,
            ),
            1,
        )
    }
    fn gelu(
        &self,
        c: &ProtocolObject<dyn MTLCommandBuffer>,
        index: usize,
        states: [State; 4],
        status: usize,
    ) -> Result<()> {
        let width = if index == 0 { self.shape.ff } else { P };
        let buffers = gelu::DeviceBuffers {
            gate: self.r(states[0])?,
            up: self.r(states[1])?,
            activated: self.r(states[2])?,
            product: self.r(states[3])?,
            status: self.s(status)?,
            lane_flags: self.flags[index],
            intervals: self.intervals[index],
        };
        let shape = gelu::Shape {
            rows: self.shape.rows,
            width,
        };
        #[cfg(test)]
        if self.force_gelu_reference {
            return self.record(
                self.pipelines
                    .gelu
                    .encode_forced_reference(c, self.table, &buffers, shape),
                1,
            );
        }
        self.record(
            self.pipelines.gelu.encode(c, self.table, &buffers, shape),
            1,
        )
    }
    pub fn encode(&self, c: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<()> {
        use State::*;
        self.preflight(c)?;
        let s = self.shape;
        self.rms(c, [Hidden, InputNorm], 0, 0, [s.rows, H])?;
        self.admit(c, 0, InputNorm, 1)?;
        self.project(c, 0, InputNorm, QRaw, 2)?;
        self.rms(c, [QRaw, QNorm], 1, 3, [s.rows * 8, s.d])?;
        self.rotate(c, QNorm, QRotary, 8, 4)?;
        if s.producer {
            self.project(c, 1, InputNorm, KRaw, 5)?;
            self.rms(c, [KRaw, KNorm], 2, 6, [s.rows, s.d])?;
            self.rotate(c, KNorm, KProposal, 1, 7)?;
            self.project(c, 2, InputNorm, VRaw, 8)?;
            self.rms(c, [VRaw, VProposal], 7, 9, [s.rows, s.d])?;
        }
        let segments = self.segments.each_ref().map(|x| attention::DeviceSegment {
            k: x.k,
            v: x.v,
            keys: x.keys,
            head_stride: x.head_stride,
            key_start: x.key_start,
        });
        self.record(
            self.pipelines.attention.encode(
                c,
                &attention::DeviceBuffers {
                    q: self.r(QRotary)?,
                    segments,
                    scores: self.r(Scores)?,
                    probabilities: self.r(Probabilities)?,
                    output: self.r(Context)?,
                    status: self.s(10)?,
                },
                &cpu::SpanAttentionShape {
                    queries: s.rows,
                    keys: s.keys,
                    query_heads: 8,
                    kv_heads: 1,
                    head_dim: s.d,
                    query_start: s.query_start,
                    key_start: s.key_start,
                    window: s.window,
                    scale: 1.,
                },
            ),
            4,
        )?;
        self.admit(c, 1, Context, 14)?;
        self.project(c, 3, Context, AttentionRaw, 15)?;
        self.rms(c, [AttentionRaw, AttentionNorm], 3, 16, [s.rows, H])?;
        self.add(c, AttentionNorm, Hidden, AttentionResidual, 17)?;
        self.rms(c, [AttentionResidual, PreFf], 4, 18, [s.rows, H])?;
        self.admit(c, 2, PreFf, 19)?;
        self.project(c, 4, PreFf, FfGate, 20)?;
        self.project(c, 5, PreFf, FfUp, 21)?;
        self.gelu(c, 0, [FfGate, FfUp, FfActivation, FfProduct], 22)?;
        self.admit(c, 3, FfProduct, 23)?;
        self.project(c, 6, FfProduct, FfDown, 24)?;
        self.rms(c, [FfDown, FfNorm], 5, 25, [s.rows, H])?;
        self.add(c, FfNorm, AttentionResidual, FfResidual, 26)?;
        self.admit(c, 4, FfResidual, 27)?;
        self.project(c, 7, FfResidual, PleGate, 28)?;
        self.gelu(c, 1, [PleGate, PleInput, PleActivation, PleProduct], 29)?;
        self.admit(c, 5, PleProduct, 30)?;
        self.project(c, 8, PleProduct, PleProjection, 31)?;
        self.rms(c, [PleProjection, PleNorm], 6, 32, [s.rows, H])?;
        self.add(c, FfResidual, PleNorm, PleResidual, 33)?;
        self.record(
            self.pipelines.residual.encode_scale(
                c,
                self.r(PleResidual)?,
                self.r(Output)?,
                self.s(34)?,
                self.scalar,
            ),
            1,
        )
    }
}
