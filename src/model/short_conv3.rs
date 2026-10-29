//! Concrete transactional two-row FP32 convolution history.
//!
//! The unchanged scalar leaf computes oldest-to-current taps in FP64 and stores
//! FP32, not a future Half profile. Inputs are already `u=B*x`; there is no SiLU.
//! Coefficients remain model-owned `[channels,3]`. Candidate `[2,channels]` is
//! caller-owned per history; work `[2,channels]` can be reused across owners.
//! Output `[tokens,channels]` and input are also caller-owned. None are allocated
//! here. A prepared frame borrows its final candidate until commit or discard.
//!
//! Only the current committed prefix is retained. Arbitrary shortening/change
//! cannot rewind two rows: the caller must reset and replay from position zero.
//! Token history, model identity, pending tokens and logits belong to the future
//! hybrid session, not this leaf. This batch is NOT a whole-model KV transaction.

use super::{Error, allocation_bytes, zeros};
use crate::cpu::{self, causal_depthwise_conv_step, finite, length, nonzero, product};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

pub struct History {
    committed: Vec<f32>,
    channels: usize,
    max_positions: usize,
    offset: usize,
    generation: u64,
    owner: u64,
}

/// Opaque non-owning final candidate, authenticated against its actual history
/// instance and generation. It borrows neither the work buffer nor the owner.
/// Dropping a frame aborts without mutation; it grants no graph-wide success.
pub struct Prepared<'a> {
    candidate: &'a [f32],
    owner: u64,
    generation: u64,
    offset: usize,
    tokens: usize,
}

impl History {
    /// Committed bank and owner object only. Per-owner candidate, one reusable
    /// work bank and all input/output/model/collection storage are separate.
    pub fn planned_bytes(channels: usize, max_positions: usize) -> Result<usize, Error> {
        nonzero(channels, "convolution channels")?;
        nonzero(max_positions, "convolution maximum positions")?;
        allocation_bytes(product(2, channels)?)?
            .checked_add(std::mem::size_of::<Self>())
            .ok_or(cpu::Error::SizeOverflow.into())
    }
    pub fn new(channels: usize, max_positions: usize) -> Result<Self, Error> {
        Self::planned_bytes(channels, max_positions)?;
        let committed = zeros(product(2, channels)?)?;
        allocation_bytes(committed.capacity())?
            .checked_add(std::mem::size_of::<Self>())
            .ok_or(cpu::Error::SizeOverflow)?;
        let owner = NEXT_OWNER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| Error::InvalidConfig("convolution owner identity exhausted"))?;
        Ok(Self {
            committed,
            channels,
            max_positions,
            offset: 0,
            generation: 0,
            owner,
        })
    }
    pub fn allocated_bytes(&self) -> Result<usize, Error> {
        allocation_bytes(self.committed.capacity())?
            .checked_add(std::mem::size_of::<Self>())
            .ok_or(cpu::Error::SizeOverflow.into())
    }
    pub fn channels(&self) -> usize {
        self.channels
    }
    pub fn offset(&self) -> usize {
        self.offset
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn committed(&self) -> &[f32] {
        &self.committed
    }
    /// Exact number of FP32 elements required by EACH candidate and work bank.
    pub fn staging_elements(&self) -> usize {
        self.committed.len()
    }

    fn admit(&self, tokens: usize) -> Result<usize, Error> {
        nonzero(tokens, "convolution new tokens")?;
        let after = self
            .offset
            .checked_add(tokens)
            .ok_or(cpu::Error::SizeOverflow)?;
        if after > self.max_positions {
            return Err(Error::Capacity {
                offset: self.offset,
                tokens,
                limit: self.max_positions,
            });
        }
        self.generation
            .checked_add(1)
            .ok_or(cpu::Error::SizeOverflow)?;
        let elements = product(tokens, self.channels)?;
        allocation_bytes(elements)?;
        Ok(elements)
    }

    /// Validate the ENTIRE chunk and all operands before any scratch write.
    /// Zero tokens reject; callers make no prepare/commit call for empty input.
    /// On numerical cast overflow output/candidate/work may be partial; discard
    /// them. No frame is returned and committed history/offset remain unchanged.
    pub fn prepare_chunk<'a>(
        &self,
        input: &[f32],
        weights: &[f32],
        candidate: &'a mut [f32],
        work: &mut [f32],
        output: &mut [f32],
        tokens: usize,
    ) -> Result<Prepared<'a>, Error> {
        let elements = self.admit(tokens)?;
        for (name, actual, expected) in [
            ("convolution chunk input", input.len(), elements),
            (
                "convolution coefficients",
                weights.len(),
                product(3, self.channels)?,
            ),
            (
                "convolution candidate",
                candidate.len(),
                self.committed.len(),
            ),
            ("convolution work", work.len(), self.committed.len()),
            ("convolution chunk output", output.len(), elements),
        ] {
            length(name, actual, expected)?;
        }
        finite("convolution chunk input", input)?;
        finite("convolution coefficients", weights)?;
        let c = self.channels;
        causal_depthwise_conv_step(
            &input[..c],
            weights,
            &self.committed,
            candidate,
            &mut output[..c],
            c,
            3,
        )?;
        for token in 1..tokens {
            let row = token * c..(token + 1) * c;
            if token % 2 == 1 {
                causal_depthwise_conv_step(
                    &input[row.clone()],
                    weights,
                    candidate,
                    work,
                    &mut output[row],
                    c,
                    3,
                )?;
            } else {
                causal_depthwise_conv_step(
                    &input[row.clone()],
                    weights,
                    work,
                    candidate,
                    &mut output[row],
                    c,
                    3,
                )?;
            }
        }
        if tokens.is_multiple_of(2) {
            candidate.copy_from_slice(work);
        }
        Ok(Prepared {
            candidate,
            owner: self.owner,
            generation: self.generation,
            offset: self.offset,
            tokens,
        })
    }
    fn validate(&self, frame: &Prepared<'_>) -> Result<(), Error> {
        if frame.owner != self.owner
            || frame.generation != self.generation
            || frame.offset != self.offset
        {
            return Err(Error::InvalidConfig(
                "stale or foreign convolution proposal",
            ));
        }
        self.admit(frame.tokens)?;
        length(
            "convolution final candidate",
            frame.candidate.len(),
            self.committed.len(),
        )?;
        Ok(())
    }
    pub fn commit(&mut self, frame: &Prepared<'_>) -> Result<(), Error> {
        commit_batch(std::slice::from_mut(self), std::slice::from_ref(frame))
    }
    pub fn reset(&mut self) -> Result<(), Error> {
        reset_batch(std::slice::from_mut(self))
    }
}

/// Validate the complete ordered owner/frame list and common step before writes.
/// An empty owner/frame pair is a no-op; an empty numerical chunk is rejected.
/// Frames borrow immutable finite candidates, so copies below cannot fail. This
/// publishes only these convolution histories; future hybrid composition must
/// prevalidate its KV and history publication together before either mutates.
pub fn commit_batch(histories: &mut [History], frames: &[Prepared<'_>]) -> Result<(), Error> {
    length("convolution commit frames", frames.len(), histories.len())?;
    if histories.is_empty() {
        return Ok(());
    }
    let first = &frames[0];
    for (index, (history, frame)) in histories.iter().zip(frames).enumerate() {
        if histories[..index].iter().any(|h| h.owner == history.owner)
            || frames[..index].iter().any(|f| f.owner == frame.owner)
            || frame.offset != first.offset
            || frame.tokens != first.tokens
        {
            return Err(Error::InvalidConfig("duplicate or mixed convolution step"));
        }
        history.validate(frame)?;
    }
    for (history, frame) in histories.iter_mut().zip(frames) {
        history.committed.copy_from_slice(frame.candidate);
        history.offset += frame.tokens;
        history.generation += 1;
    }
    Ok(())
}

/// Reset to a zero causal prefix, invalidate old frames and retain allocation.
/// Generation exhaustion is checked across ALL owners before any mutation.
pub fn reset_batch(histories: &mut [History]) -> Result<(), Error> {
    for history in histories.iter() {
        history
            .generation
            .checked_add(1)
            .ok_or(Error::InvalidConfig(
                "convolution reset generation exhausted",
            ))?;
    }
    for history in histories {
        history.committed.fill(0.0);
        history.offset = 0;
        history.generation += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
