//! Concrete transactional BF16 state for the pinned 24-layer LFM3 topology.
//! No weights, model graph, sampled history or token/session policy lives here.
use super::{Error, short_conv7_bf16 as convolution, window_cache_bf16 as attention};
use crate::cpu::{self, length, nonzero, product};

pub const CONV_LAYERS: [usize; 13] = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23];
pub const ATTENTION_LAYERS: [usize; 11] = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22];
pub const ATTENTION_WINDOWS: [Option<usize>; 11] = [
    None,
    Some(256),
    None,
    Some(256),
    None,
    Some(256),
    None,
    Some(256),
    None,
    Some(256),
    None,
];
const CHANNELS: usize = 2048;
const HEADS: usize = 4;
const DIMS: usize = 128;
const SOURCE_MAX_POSITIONS: usize = 262144;

pub struct State {
    histories: Vec<convolution::History>,
    caches: Vec<attention::WindowKvCache>,
    capacity: usize,
    offset: usize,
    generation: u64,
    allocated: usize,
}

fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(cpu::Error::SizeOverflow.into())
}

impl State {
    /// Owner objects and unique BF16 buffers only. Weights, candidate/input
    /// staging, numerical scratch and application/codec costs remain separate.
    /// Local owners retain all256 physical rows even at smaller admission caps.
    pub fn planned_bytes(capacity: usize) -> Result<usize, Error> {
        nonzero(capacity, "LFM3 BF16 context capacity")?;
        if capacity > SOURCE_MAX_POSITIONS {
            return Err(Error::InvalidConfig(
                "LFM3 state capacity exceeds source positions",
            ));
        }
        let history = convolution::History::planned_bytes(CHANNELS, capacity)?;
        let mut total = add(
            std::mem::size_of::<Self>(),
            product(history, CONV_LAYERS.len())?,
        )?;
        for window in ATTENTION_WINDOWS {
            let retained = window.unwrap_or(capacity);
            let positions = capacity.max(retained);
            total = add(
                total,
                attention::WindowKvCache::planned_bytes(HEADS, DIMS, retained, positions, window)?,
            )?;
        }
        Ok(total)
    }

    /// Construct only after the caller admits the complete working set.
    /// This allocation is not evidence of whole-model memory fit or adoption.
    pub fn new(capacity: usize) -> Result<Self, Error> {
        let planned = Self::planned_bytes(capacity)?;
        let mut histories = Vec::new();
        let mut caches = Vec::new();
        histories
            .try_reserve_exact(CONV_LAYERS.len())
            .map_err(|_| Error::Allocation)?;
        caches
            .try_reserve_exact(ATTENTION_LAYERS.len())
            .map_err(|_| Error::Allocation)?;
        for _ in CONV_LAYERS {
            histories.push(convolution::History::new(CHANNELS, capacity)?);
        }
        for window in ATTENTION_WINDOWS {
            let retained = window.unwrap_or(capacity);
            caches.push(attention::WindowKvCache::new(
                HEADS,
                DIMS,
                retained,
                capacity.max(retained),
                window,
            )?);
        }
        let mut allocated = add(
            std::mem::size_of::<Self>(),
            product(
                histories.capacity(),
                std::mem::size_of::<convolution::History>(),
            )?,
        )?;
        allocated = add(
            allocated,
            product(
                caches.capacity(),
                std::mem::size_of::<attention::WindowKvCache>(),
            )?,
        )?;
        for owner in &histories {
            allocated = add(
                allocated,
                owner
                    .allocated_bytes()?
                    .checked_sub(std::mem::size_of::<convolution::History>())
                    .ok_or(Error::InvalidConfig("LFM3 history memory accounting"))?,
            )?;
        }
        for owner in &caches {
            allocated = add(
                allocated,
                owner
                    .allocated_bytes()
                    .checked_sub(std::mem::size_of::<attention::WindowKvCache>())
                    .ok_or(Error::InvalidConfig("LFM3 KV memory accounting"))?,
            )?;
        }
        if allocated > planned {
            return Err(Error::InvalidConfig("LFM3 state allocation exceeded plan"));
        }
        Ok(Self {
            histories,
            caches,
            capacity,
            offset: 0,
            generation: 0,
            allocated,
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn offset(&self) -> usize {
        self.offset
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn allocated_bytes(&self) -> usize {
        self.allocated
    }
    /// Immutable owners allow preparation/views but prevent partial public
    /// publication; only the complete state commit/reset below mutates them.
    pub fn histories(&self) -> &[convolution::History] {
        &self.histories
    }
    pub fn caches(&self) -> &[attention::WindowKvCache] {
        &self.caches
    }

    fn admit(&self, tokens: usize) -> Result<usize, Error> {
        nonzero(tokens, "LFM3 BF16 state tokens")?;
        let after = self
            .offset
            .checked_add(tokens)
            .ok_or(cpu::Error::SizeOverflow)?;
        if after > self.capacity {
            return Err(Error::Capacity {
                offset: self.offset,
                tokens,
                limit: self.capacity,
            });
        }
        self.generation
            .checked_add(1)
            .ok_or(cpu::Error::SizeOverflow)?;
        Ok(after)
    }

    /// Validate ALL13 history and11 KV proposals before changing any owner.
    /// Caller invokes this after its whole numerical step succeeds. A failure
    /// leaves the exact committed prefix, generation and physical data intact.
    /// No allocation or fallible operation remains after the validation phase.
    pub fn commit(
        &mut self,
        histories: &[convolution::Prepared<'_>],
        caches: &[attention::PreparedKv<'_>],
        tokens: usize,
    ) -> Result<(), Error> {
        let after = self.admit(tokens)?;
        length(
            "LFM3 history proposals",
            histories.len(),
            self.histories.len(),
        )?;
        length("LFM3 KV proposals", caches.len(), self.caches.len())?;
        for (owner, frame) in self.histories.iter().zip(histories) {
            owner.validate_step(frame, self.offset, tokens, self.generation)?;
        }
        for (owner, frame) in self.caches.iter().zip(caches) {
            owner.validate_step(frame, self.offset, tokens, self.generation)?;
        }
        for (owner, frame) in self.histories.iter_mut().zip(histories) {
            owner.commit_validated(frame);
        }
        for (owner, frame) in self.caches.iter_mut().zip(caches) {
            owner.commit_validated(frame);
        }
        self.offset = after;
        self.generation += 1;
        Ok(())
    }

    /// Reset/replay is required for a changed or shortened prefix. Validate all
    /// generations before any reset, zero history and invalidate KV logically.
    /// Existing proposals become stale; physical allocations remain resident.
    pub fn reset(&mut self) -> Result<(), Error> {
        self.generation
            .checked_add(1)
            .ok_or(cpu::Error::SizeOverflow)?;
        for owner in &self.histories {
            owner.validate_reset()?;
        }
        for owner in &self.caches {
            owner.validate_reset()?;
        }
        for owner in &mut self.histories {
            owner.reset_validated();
        }
        for owner in &mut self.caches {
            owner.reset_validated();
        }
        self.offset = 0;
        self.generation += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
