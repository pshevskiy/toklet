use super::*;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub(in crate::metal::gemma_affine::bf16::resident) enum State {
    Hidden,
    InputNorm,
    AttentionRaw,
    AttentionNorm,
    AttentionResidual,
    PreFf,
    FfDown,
    FfNorm,
    FfResidual,
    PleProjection,
    PleNorm,
    PleResidual,
    Output,
    QRaw,
    QNorm,
    QRotary,
    Context,
    KRaw,
    KNorm,
    KProposal,
    VRaw,
    VProposal,
    FfGate,
    FfUp,
    FfActivation,
    FfProduct,
    PleInput,
    PleGate,
    PleActivation,
    PleProduct,
    Scores,
    Probabilities,
}
pub(super) const STATES: [State; 32] = [
    State::Hidden,
    State::InputNorm,
    State::AttentionRaw,
    State::AttentionNorm,
    State::AttentionResidual,
    State::PreFf,
    State::FfDown,
    State::FfNorm,
    State::FfResidual,
    State::PleProjection,
    State::PleNorm,
    State::PleResidual,
    State::Output,
    State::QRaw,
    State::QNorm,
    State::QRotary,
    State::Context,
    State::KRaw,
    State::KNorm,
    State::KProposal,
    State::VRaw,
    State::VProposal,
    State::FfGate,
    State::FfUp,
    State::FfActivation,
    State::FfProduct,
    State::PleInput,
    State::PleGate,
    State::PleActivation,
    State::PleProduct,
    State::Scores,
    State::Probabilities,
];
#[derive(Clone, Copy)]
pub(super) struct Region {
    pub start: usize,
    pub len: usize,
}
impl Region {
    pub fn end(self) -> usize {
        self.start + self.len
    }
    pub fn range(self, b: &Buffer<u32>, active: usize) -> Result<BufferRange<'_>> {
        if active > self.len {
            return Err(Error::Options("producer active region exceeds capacity"));
        }
        BufferRange::new(b, self.start, active)
    }
}
pub(super) struct Layout {
    pub regions: [Region; 32],
    pub words: usize,
    pub d: usize,
}
impl Layout {
    pub fn new(d: usize) -> Result<Self> {
        if d != 256 && d != 512 {
            return Err(Error::Options("producer head dimension outside scope"));
        }
        let mut cursor = 0;
        let regions = STATES.map(|s| {
            let n = Self::values(s, 4, 4, d);
            let r = Region {
                start: cursor,
                len: n,
            };
            cursor += n + GUARD;
            r
        });
        Ok(Self {
            regions,
            words: cursor,
            d,
        })
    }
    pub fn values(s: State, t: usize, k: usize, d: usize) -> usize {
        match s {
            State::QRaw | State::QNorm | State::QRotary | State::Context => t * 8 * d,
            State::KRaw | State::KNorm | State::KProposal | State::VRaw | State::VProposal => t * d,
            State::FfGate | State::FfUp | State::FfActivation | State::FfProduct => t * 6144,
            State::PleInput | State::PleGate | State::PleActivation | State::PleProduct => t * 256,
            State::Scores | State::Probabilities => 8 * t * k,
            _ => t * 1536,
        }
    }
    pub fn active(&self, s: State, f: &Frame) -> usize {
        Self::values(s, f.rows, f.end, self.d)
    }
    pub fn region(&self, s: State) -> Region {
        self.regions[s as usize]
    }
    pub fn range<'a>(&self, b: &'a Buffer<u32>, s: State, f: &Frame) -> Result<BufferRange<'a>> {
        self.region(s).range(b, self.active(s, f))
    }
    pub fn guards(&self, b: &Buffer<u32>, f: &Frame) -> bool {
        b.with_slice(|words| {
            STATES.iter().all(|&s| {
                let r = self.region(s);
                words[r.start + self.active(s, f)..r.end() + GUARD]
                    .iter()
                    .all(|&w| w == SENTINEL)
            })
        })
    }
}
