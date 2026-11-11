//! Single-sequence BF16 execution for the pinned 24-layer LFM3 candidate.
use std::mem::size_of;

use half::bf16;

use super::{
    Error,
    lfm3_rotary::{Profile, Rotary, SOURCE_MAX_POSITIONS},
    lfm3_state::{ATTENTION_LAYERS, CONV_LAYERS, State},
};
use crate::{
    cpu::{self, nonzero, product},
    weights::lfm3::{PARAMETERS, Weights},
};

mod forward;
mod generation;
pub use generation::GenerationSession;

pub const HIDDEN: usize = 2048;
pub const INTERMEDIATE: usize = 7680;
pub const QUERY_HEADS: usize = 16;
pub const KV_HEADS: usize = 4;
pub const HEAD_DIM: usize = 128;
pub const VOCAB: usize = 128_000;
pub const LAYERS: usize = 24;
const KV_WIDTH: usize = KV_HEADS * HEAD_DIM;
const CONV_OWNERS: usize = CONV_LAYERS.len();
const KV_OWNERS: usize = ATTENTION_LAYERS.len();
const HISTORY_WORDS: usize = 6 * HIDDEN;

/// The single active graph session is included in the model admission plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Memory {
    pub weights_bytes: usize,
    pub positional_bytes: usize,
    pub state_bytes: usize,
    pub scratch_bytes: usize,
    pub history_bytes: usize,
    pub published_logits_bytes: usize,
    pub owner_bytes: usize,
    pub total_bytes: usize,
}

pub struct Model {
    weights: Weights,
    full_rotary: Rotary,
    sliding_rotary: Rotary,
    capacity: usize,
    chunk: usize,
    memory: Memory,
}

/// Observations are borrowed from the uncommitted step. A callback error
/// discards that step and leaves all published outputs and state intact.
pub struct Observation<'a> {
    pub layer: Option<usize>,
    pub stage: &'static str,
    pub values: &'a [bf16],
}

pub struct Session<'a> {
    model: &'a Model,
    state: State,
    scratch: Scratch,
    history: Vec<u32>,
    logits: Vec<bf16>,
}

fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(cpu::Error::SizeOverflow.into())
}

pub(crate) fn reserve<T>(count: usize) -> Result<Vec<T>, Error> {
    let bytes = product(count, size_of::<T>())?;
    if bytes > isize::MAX as usize {
        return Err(cpu::Error::SizeOverflow.into());
    }
    let mut v = Vec::new();
    v.try_reserve_exact(count).map_err(|_| Error::Allocation)?;
    if v.capacity() != count {
        return Err(Error::InvalidConfig("LFM3 buffer capacity exceeded plan"));
    }
    Ok(v)
}

fn filled<T: Copy>(count: usize, initial: T) -> Result<Vec<T>, Error> {
    let mut v = reserve(count)?;
    v.resize(count, initial);
    Ok(v)
}

impl Model {
    pub(crate) fn plan(capacity: usize, chunk: usize) -> Result<Memory, Error> {
        nonzero(capacity, "LFM3 context capacity")?;
        nonzero(chunk, "LFM3 chunk capacity")?;
        if capacity > SOURCE_MAX_POSITIONS || chunk > capacity {
            return Err(Error::InvalidConfig("LFM3 context/chunk capacity"));
        }
        let mut plan = Memory {
            weights_bytes: product(PARAMETERS, size_of::<bf16>())?,
            positional_bytes: product(2, Rotary::planned_bytes(capacity)?)?,
            state_bytes: State::planned_bytes(capacity)?,
            scratch_bytes: Scratch::planned_bytes(capacity, chunk)?,
            history_bytes: product(capacity, size_of::<u32>())?,
            published_logits_bytes: product(VOCAB, size_of::<bf16>())?,
            owner_bytes: size_of::<Self>() + size_of::<Session<'_>>()
                - 2 * size_of::<Rotary>()
                - size_of::<State>(),
            total_bytes: 0,
        };
        plan.total_bytes = [
            plan.weights_bytes,
            plan.positional_bytes,
            plan.state_bytes,
            plan.scratch_bytes,
            plan.history_bytes,
            plan.published_logits_bytes,
            plan.owner_bytes,
        ]
        .into_iter()
        .try_fold(0, add)?;
        Ok(plan)
    }

    pub(crate) fn from_verified(
        weights: Weights,
        capacity: usize,
        chunk: usize,
        memory: Memory,
    ) -> Result<Self, Error> {
        let rotary_budget = memory.positional_bytes / 2;
        Ok(Self {
            weights,
            full_rotary: Rotary::new(Profile::FullAttention, capacity, rotary_budget)?,
            sliding_rotary: Rotary::new(Profile::SlidingAttention, capacity, rotary_budget)?,
            capacity,
            chunk,
            memory,
        })
    }

    pub fn memory(&self) -> Memory {
        self.memory
    }

    /// Borrow the admitted model exclusively while the session exists.
    pub fn session(&mut self) -> Result<Session<'_>, Error> {
        let capacity = self.capacity;
        let chunk = self.chunk;
        Ok(Session {
            model: self,
            state: State::new(capacity)?,
            scratch: Scratch::new(capacity, chunk)?,
            history: filled(capacity, 0)?,
            logits: filled(VOCAB, bf16::ZERO)?,
        })
    }
}

impl Session<'_> {
    pub fn history(&self) -> &[u32] {
        &self.history[..self.state.offset()]
    }

    pub fn logits(&self) -> Option<&[bf16]> {
        (self.state.offset() != 0).then_some(&self.logits)
    }

    pub fn generation(&self) -> u64 {
        self.state.generation()
    }

    pub fn reset(&mut self) -> Result<(), Error> {
        self.state.reset()
    }

    /// Commit a raw token chunk and publish its last-token tied-head logits.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<&[bf16], Error> {
        self.step::<false, _>(tokens, |_| Ok(()))
    }

    pub fn forward_observed(
        &mut self,
        tokens: &[u32],
        observer: impl FnMut(Observation<'_>) -> Result<(), Error>,
    ) -> Result<&[bf16], Error> {
        self.step::<true, _>(tokens, observer)
    }
}

struct Scratch {
    // Six token-major hidden buffers, three token-major FFN buffers.
    hidden: [Vec<bf16>; 6],
    ffn: [Vec<bf16>; 3],
    bcx: Vec<bf16>,
    kv: [Vec<bf16>; 2],
    gates: [Vec<bf16>; 2],
    histories: [Vec<bf16>; CONV_OWNERS],
    keys: [Vec<bf16>; KV_OWNERS],
    values: [Vec<bf16>; KV_OWNERS],
    scores: Vec<f32>,
    logits: Vec<bf16>,
}

fn bf16_buffers<const N: usize>(count: usize) -> Result<[Vec<bf16>; N], Error> {
    let mut buffers = std::array::from_fn(|_| Vec::new());
    for buffer in &mut buffers {
        *buffer = filled(count, bf16::ZERO)?;
    }
    Ok(buffers)
}

impl Scratch {
    fn planned_bytes(capacity: usize, chunk: usize) -> Result<usize, Error> {
        let per_token =
            9 * HIDDEN + 3 * INTERMEDIATE + (2 + 2 * KV_OWNERS) * KV_WIDTH + 2 * QUERY_HEADS;
        let words = add(product(chunk, per_token)?, CONV_OWNERS * HISTORY_WORDS)?;
        add(
            product(add(words, VOCAB)?, size_of::<bf16>())?,
            product(capacity, size_of::<f32>())?,
        )
    }

    fn new(capacity: usize, chunk: usize) -> Result<Self, Error> {
        Ok(Self {
            hidden: bf16_buffers(product(chunk, HIDDEN)?)?,
            ffn: bf16_buffers(product(chunk, INTERMEDIATE)?)?,
            bcx: filled(product(chunk, 3 * HIDDEN)?, bf16::ZERO)?,
            kv: bf16_buffers(product(chunk, KV_WIDTH)?)?,
            gates: bf16_buffers(product(chunk, QUERY_HEADS)?)?,
            histories: bf16_buffers(HISTORY_WORDS)?,
            keys: bf16_buffers(product(chunk, KV_WIDTH)?)?,
            values: bf16_buffers(product(chunk, KV_WIDTH)?)?,
            scores: filled(capacity, 0.0)?,
            logits: filled(VOCAB, bf16::ZERO)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planned_scratch_covers_each_owner_and_buffer() {
        for (capacity, chunk) in [(1, 1), (32, 3), (2048, 16)] {
            let memory = Model::plan(capacity, chunk).unwrap();
            let scratch = Scratch::new(capacity, chunk).unwrap();
            let ordinary = scratch
                .hidden
                .iter()
                .chain(&scratch.ffn)
                .chain(&scratch.kv)
                .chain(&scratch.gates)
                .chain(&scratch.histories)
                .chain(&scratch.keys)
                .chain(&scratch.values)
                .chain([&scratch.bcx, &scratch.logits])
                .map(Vec::capacity)
                .sum::<usize>();
            assert_eq!(
                memory.scratch_bytes,
                ordinary * 2 + scratch.scores.capacity() * 4
            );
            assert_eq!(memory.weights_bytes, 3_457_603_072);
            assert_eq!(
                memory.state_bytes,
                State::new(capacity).unwrap().allocated_bytes()
            );
            assert_eq!(
                memory.total_bytes,
                memory.weights_bytes
                    + memory.positional_bytes
                    + memory.state_bytes
                    + memory.scratch_bytes
                    + memory.history_bytes
                    + memory.published_logits_bytes
                    + memory.owner_bytes
            );
        }
    }

    #[test]
    fn shape_arithmetic_rejects_outside_the_supported_capacity() {
        for (capacity, chunk) in [(0, 1), (1, 0), (1, 2), (SOURCE_MAX_POSITIONS + 1, 1)] {
            assert!(Model::plan(capacity, chunk).is_err());
        }
    }
}
