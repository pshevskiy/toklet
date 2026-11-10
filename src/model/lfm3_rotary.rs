//! Immutable source-profile LFM3 BF16 rotary data, shared across attention layers.
//!
//! Only the pinned 128-dimensional full/sliding profiles are represented here.
//! This does not construct or qualify a complete LFM3 model. The independent
//! mathematical proportional rotary reference remains unchanged.

use super::{Error, allocation_bytes};
use crate::cpu::{self, length, nonzero, product};
use half::bf16;

pub const HEAD_DIM: usize = 128;
pub const SOURCE_MAX_POSITIONS: usize = 262_144;
const PAIRS: usize = HEAD_DIM / 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    FullAttention,
    SlidingAttention,
}

/// Model-owned frequencies and [capacity,64,cosine/sine] BF16 coefficients.
/// Each pair joins i to i+64, including inactive zero-frequency pairs.
/// Tables are shared across heads/layers; sessions own their input/output/KV.
pub struct Rotary {
    profile: Profile,
    frequencies: [f32; PAIRS],
    coefficients: Vec<[bf16; 2]>,
}

impl Rotary {
    /// Object (including inline frequencies) plus requested coefficient storage.
    /// Allocator bookkeeping and caller input/output/session storage are separate.
    pub fn planned_bytes(capacity: usize) -> Result<usize, Error> {
        nonzero(capacity, "LFM3 rotary capacity")?;
        if capacity > SOURCE_MAX_POSITIONS {
            return Err(Error::InvalidConfig("LFM3 rotary source position limit"));
        }
        // [bf16;2] and f32 both occupy four bytes.
        allocation_bytes(product(capacity, PAIRS)?)?
            .checked_add(std::mem::size_of::<Self>())
            .ok_or(cpu::Error::SizeOverflow.into())
    }

    /// Admit this owner's bytes before allocation. available_bytes must be the
    /// remaining model budget, after accounting weights and every other owner.
    /// No whole-model memory-fit claim follows from admitting this one table.
    pub fn new(profile: Profile, capacity: usize, available_bytes: usize) -> Result<Self, Error> {
        if Self::planned_bytes(capacity)? > available_bytes {
            return Err(Error::InvalidConfig("LFM3 rotary admitted memory"));
        }
        let mut coefficients = Vec::new();
        coefficients
            .try_reserve_exact(product(capacity, PAIRS)?)
            .map_err(|_| Error::Allocation)?;
        let mut table = Self {
            profile,
            frequencies: [0.; PAIRS],
            coefficients,
        };
        if table.allocated_bytes()? > available_bytes {
            return Err(Error::InvalidConfig("LFM3 rotary allocated memory"));
        }
        let (theta, active_pairs) = match profile {
            Profile::FullAttention => (1_000_000_f64, 32),
            Profile::SlidingAttention => (10_000_f64, 64),
        };
        for (pair, frequency) in table.frequencies.iter_mut().take(active_pairs).enumerate() {
            // Match the declared child recipe: FP32 exponent, FP64 power
            // rounded to FP32, then FP32 reciprocal. The zero tail stays +0.
            let exponent = (2 * pair) as f32 / HEAD_DIM as f32;
            let power = theta.powf(f64::from(exponent)) as f32;
            *frequency = 1. / power;
        }
        for position in 0..capacity {
            for &frequency in &table.frequencies {
                let angle = position as f32 * frequency;
                let (sine, cosine) = angle.sin_cos();
                table
                    .coefficients
                    .push([bf16::from_f32(cosine), bf16::from_f32(sine)]);
            }
        }
        Ok(table)
    }

    pub fn allocated_bytes(&self) -> Result<usize, Error> {
        allocation_bytes(self.coefficients.capacity())?
            .checked_add(std::mem::size_of::<Self>())
            .ok_or(cpu::Error::SizeOverflow.into())
    }

    pub fn capacity(&self) -> usize {
        self.coefficients.len() / PAIRS
    }

    pub fn profile(&self) -> Profile {
        self.profile
    }

    pub fn frequencies(&self) -> &[f32; PAIRS] {
        &self.frequencies
    }

    /// Read-only diagnostic/backend view; no copy or per-head expansion.
    pub fn coefficient_row(&self, position: usize) -> Option<&[[bf16; 2]]> {
        if position >= self.capacity() {
            return None;
        }
        Some(&self.coefficients[position * PAIRS..(position + 1) * PAIRS])
    }

    /// Rotate caller-owned contiguous [tokens,heads,128] BF16 slices.
    ///
    /// Absolute positions and all sizes/finite inputs validate before writes.
    /// Empty calls reject; the caller skips them. Each BF16 input/coefficient
    /// widens exactly to FP32. Separate FP32 products/additions precede one
    /// nearest-even BF16 store. Even zero-frequency pairs execute the source
    /// arithmetic: signed zeros are not unconditionally copied.
    ///
    /// Numerical/store overflow may leave partial output; discard it on error.
    /// This immutable owner publishes no cache state. No allocations, dynamic
    /// dispatch, trig calls or parameter selection occur in steady-state apply.
    pub fn apply(
        &self,
        input: &[bf16],
        output: &mut [bf16],
        tokens: usize,
        heads: usize,
        offset: usize,
    ) -> Result<(), Error> {
        nonzero(tokens, "LFM3 rotary tokens")?;
        nonzero(heads, "LFM3 rotary heads")?;
        let after = offset.checked_add(tokens).ok_or(cpu::Error::SizeOverflow)?;
        if after > self.capacity() {
            return Err(Error::Capacity {
                offset,
                tokens,
                limit: self.capacity(),
            });
        }
        let stride = product(heads, HEAD_DIM)?;
        let count = product(tokens, stride)?;
        length("LFM3 rotary input", input.len(), count)?;
        length("LFM3 rotary output", output.len(), count)?;
        if let Some(index) = input.iter().position(|value| !value.is_finite()) {
            return Err(cpu::Error::NonFinite {
                tensor: "LFM3 rotary input",
                index,
            }
            .into());
        }
        for (token, (source, target)) in input
            .chunks_exact(stride)
            .zip(output.chunks_exact_mut(stride))
            .enumerate()
        {
            let first = (offset + token) * PAIRS;
            let coefficients = &self.coefficients[first..first + PAIRS];
            for (head, (source, target)) in source
                .chunks_exact(HEAD_DIM)
                .zip(target.chunks_exact_mut(HEAD_DIM))
                .enumerate()
            {
                for (pair, &[cosine, sine]) in coefficients.iter().enumerate() {
                    let (a, b) = (source[pair].to_f32(), source[pair + PAIRS].to_f32());
                    let (cosine, sine) = (cosine.to_f32(), sine.to_f32());
                    let first = bf16::from_f32(a * cosine + (-b) * sine);
                    let second = bf16::from_f32(b * cosine + a * sine);
                    if !first.is_finite() || !second.is_finite() {
                        return Err(cpu::Error::NonFinite {
                            tensor: "LFM3 rotary output",
                            index: token * stride
                                + head * HEAD_DIM
                                + pair
                                + if first.is_finite() { PAIRS } else { 0 },
                        }
                        .into());
                    }
                    target[pair] = first;
                    target[pair + PAIRS] = second;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
