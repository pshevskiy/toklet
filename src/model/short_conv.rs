//! Transactional typed convolution history with concrete numerical boundaries.
//!
//! The unchanged scalar leaf computes oldest-to-current taps in FP64 and stores
//! FP32. A separate concrete BF16 boundary uses the qualified LFM3 CPU chunk
//! leaf. Inputs are already raw u=B*x; neither profile inserts SiLU.
//! Coefficients remain model-owned `[channels,KERNEL]`. Candidate `[KERNEL-1,channels]` is
//! caller-owned per history; work of the same shape can be reused across owners.
//! Output `[tokens,channels]` and input are also caller-owned. None are allocated
//! here. A prepared frame borrows its final candidate until commit or discard.
//!
//! Only the current committed prefix is retained. Arbitrary shortening/change
//! cannot rewind the retained rows: reset and replay from position zero.
//! Token history, model identity, pending tokens and logits belong to the future
//! hybrid session, not this leaf. This batch is NOT a whole-model KV transaction.

use super::state_element::{allocation_bytes, zeros};
use super::{Error, StateElement};
use crate::capacity::KvStorage;
use crate::cpu::{self, causal_depthwise_conv_step, finite, length, nonzero, product};
use half::bf16;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

pub struct History<const KERNEL: usize, T: StateElement = f32> {
    committed: Vec<T>,
    channels: usize,
    max_positions: usize,
    offset: usize,
    generation: u64,
    owner: u64,
}

/// Opaque non-owning final candidate, authenticated against its actual history
/// instance and generation. It borrows neither the work buffer nor the owner.
/// Dropping a frame aborts without mutation; it grants no graph-wide success.
pub struct Prepared<'a, T = f32> {
    candidate: &'a [T],
    owner: u64,
    generation: u64,
    offset: usize,
    tokens: usize,
}

impl Prepared<'_, bf16> {
    pub(super) fn candidate(&self) -> &[bf16] {
        self.candidate
    }
}

impl<const KERNEL: usize, T: StateElement> History<KERNEL, T> {
    /// Committed typed bank and owner object only. Per-owner candidates, FP32
    /// work banks and all input/output/model/collection storage are separate.
    pub fn planned_bytes(channels: usize, max_positions: usize) -> Result<usize, Error> {
        if !matches!(KERNEL, 3 | 7) {
            return Err(Error::InvalidConfig(
                "convolution history kernel must be three or seven",
            ));
        }
        if T::STORAGE == KvStorage::Bf16 && KERNEL != 7 {
            return Err(Error::InvalidConfig(
                "BF16 convolution history requires seven taps",
            ));
        }
        nonzero(channels, "convolution channels")?;
        nonzero(max_positions, "convolution maximum positions")?;
        allocation_bytes::<T>(product(KERNEL - 1, channels)?)?
            .checked_add(std::mem::size_of::<Self>())
            .ok_or(cpu::Error::SizeOverflow.into())
    }
    pub fn new(channels: usize, max_positions: usize) -> Result<Self, Error> {
        Self::planned_bytes(channels, max_positions)?;
        let committed = zeros::<T>(product(KERNEL - 1, channels)?)?;
        allocation_bytes::<T>(committed.capacity())?
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
        allocation_bytes::<T>(self.committed.capacity())?
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
    #[cfg(test)]
    pub(super) fn set_generation_for_test(&mut self, generation: u64) {
        self.generation = generation;
    }

    pub fn committed(&self) -> &[T] {
        &self.committed
    }
    /// Exact typed candidate elements. FP32 also needs a same-sized work bank;
    /// the BF16 chunk leaf consumes immutable history and needs no work bank.
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
        allocation_bytes::<T>(elements)?;
        Ok(elements)
    }

    fn validate(&self, frame: &Prepared<'_, T>) -> Result<(), Error> {
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
    pub(super) fn validate_step(
        &self,
        frame: &Prepared<'_, T>,
        offset: usize,
        tokens: usize,
        generation: u64,
    ) -> Result<(), Error> {
        if self.offset != offset || self.generation != generation || frame.tokens != tokens {
            return Err(Error::InvalidConfig("mixed convolution/session step"));
        }
        self.validate(frame)
    }
    /// Call only after all participating owner/session validations, without any
    /// intervening mutation. The concrete hybrid boundary owns that sequence.
    pub(super) fn commit_validated(&mut self, frame: &Prepared<'_, T>) {
        self.committed.copy_from_slice(frame.candidate);
        self.offset += frame.tokens;
        self.generation += 1;
    }
    pub(super) fn validate_reset(&self) -> Result<(), Error> {
        self.generation.checked_add(1).ok_or(Error::InvalidConfig(
            "convolution reset generation exhausted",
        ))?;
        Ok(())
    }
    pub(super) fn reset_validated(&mut self) {
        self.committed.fill(T::ZERO);
        self.offset = 0;
        self.generation += 1;
    }

    pub fn commit(&mut self, frame: &Prepared<'_, T>) -> Result<(), Error> {
        commit_batch_impl(std::slice::from_mut(self), std::slice::from_ref(frame))
    }
    pub fn reset(&mut self) -> Result<(), Error> {
        reset_batch_impl(std::slice::from_mut(self))
    }
}

impl<const KERNEL: usize> History<KERNEL, f32> {
    /// Validate the ENTIRE chunk and all operands before any scratch write.
    /// Zero tokens reject; callers make no prepare/commit call for empty input.
    /// On numerical cast overflow output/candidate/work may be partial; discard
    /// them. No frame is returned and committed history/offset remain unchanged.
    fn prepare_chunk_impl<'a>(
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
                product(KERNEL, self.channels)?,
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
            KERNEL,
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
                    KERNEL,
                )?;
            } else {
                causal_depthwise_conv_step(
                    &input[row.clone()],
                    weights,
                    work,
                    candidate,
                    &mut output[row],
                    c,
                    KERNEL,
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
}

// Concrete entrypoints keep the shared implementation in the engine crate.
// This preserves optimization of its scalar validation helpers across the public
// boundary; profile selection remains outside numerical loops.
impl History<3> {
    /// Prepare one qualified 3-tap FP32 chunk without publishing history.
    /// Scratch may be partial on numerical failure; discard it and replay the
    /// valid committed prefix. All dimensions/finite inputs precede writes.
    pub fn prepare_chunk<'a>(
        &self,
        input: &[f32],
        weights: &[f32],
        candidate: &'a mut [f32],
        work: &mut [f32],
        output: &mut [f32],
        tokens: usize,
    ) -> Result<Prepared<'a>, Error> {
        self.prepare_chunk_impl(input, weights, candidate, work, output, tokens)
    }
}

impl History<7> {
    /// Prepare one qualified 7-tap FP32 chunk without publishing history.
    /// Scratch may be partial on numerical failure; discard it and replay the
    /// valid committed prefix. All dimensions/finite inputs precede writes.
    pub fn prepare_chunk<'a>(
        &self,
        input: &[f32],
        weights: &[f32],
        candidate: &'a mut [f32],
        work: &mut [f32],
        output: &mut [f32],
        tokens: usize,
    ) -> Result<Prepared<'a>, Error> {
        self.prepare_chunk_impl(input, weights, candidate, work, output, tokens)
    }
}

impl History<7, bf16> {
    /// Prepare a canonical BF16 chunk and final six raw B*x rows, without owner
    /// mutation. Numerical failure may partially write output; the candidate
    /// is written only after numerical success. No work bank or allocation.
    pub fn prepare_chunk<'a>(
        &self,
        input: &[bf16],
        weights: &[bf16],
        candidate: &'a mut [bf16],
        output: &mut [bf16],
        tokens: usize,
    ) -> Result<Prepared<'a, bf16>, Error> {
        let elements = self.admit(tokens)?;
        length("BF16 convolution input", input.len(), elements)?;
        length("BF16 convolution output", output.len(), elements)?;
        length(
            "BF16 convolution candidate",
            candidate.len(),
            self.committed.len(),
        )?;
        let channels = self.channels;
        cpu::lfm3_conv7_bf16(input, weights, &self.committed, output, tokens, channels)?;
        if tokens >= 6 {
            candidate.copy_from_slice(&input[(tokens - 6) * channels..]);
        } else {
            let kept = (6 - tokens) * channels;
            candidate[..kept].copy_from_slice(&self.committed[tokens * channels..]);
            candidate[kept..].copy_from_slice(input);
        }
        Ok(Prepared {
            candidate,
            owner: self.owner,
            generation: self.generation,
            offset: self.offset,
            tokens,
        })
    }
}

/// Validate the complete ordered owner/frame list and common step before writes.
/// An empty owner/frame pair is a no-op; an empty numerical chunk is rejected.
/// Frames borrow immutable finite candidates, so copies below cannot fail. This
/// publishes only these convolution histories; future hybrid composition must
/// prevalidate its KV and history publication together before either mutates.
pub fn commit_batch<const KERNEL: usize>(
    histories: &mut [History<KERNEL>],
    frames: &[Prepared<'_>],
) -> Result<(), Error> {
    commit_batch_impl(histories, frames)
}

pub(super) fn commit_batch_impl<const KERNEL: usize, T: StateElement>(
    histories: &mut [History<KERNEL, T>],
    frames: &[Prepared<'_, T>],
) -> Result<(), Error> {
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
        history.commit_validated(frame);
    }
    Ok(())
}

/// Reset to a zero causal prefix, invalidate old frames and retain allocation.
/// Generation exhaustion is checked across ALL owners before any mutation.
pub fn reset_batch<const KERNEL: usize>(histories: &mut [History<KERNEL>]) -> Result<(), Error> {
    reset_batch_impl(histories)
}

pub(super) fn reset_batch_impl<const KERNEL: usize, T: StateElement>(
    histories: &mut [History<KERNEL, T>],
) -> Result<(), Error> {
    for history in histories.iter() {
        history
            .generation
            .checked_add(1)
            .ok_or(Error::InvalidConfig(
                "convolution reset generation exhausted",
            ))?;
    }
    for history in histories {
        history.reset_validated();
    }
    Ok(())
}

#[cfg(test)]
#[path = "short_conv3/tests.rs"]
mod tests;
