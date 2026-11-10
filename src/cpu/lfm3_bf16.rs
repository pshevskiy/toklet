//! Canonical LFM3 parent BF16 normalization and unary activation boundaries.
//!
//! The source profile fixes epsilon 1e-5 and widths 128/2,048. Norm means follow
//! the pinned Torch 2.9.1 M1 CPU reduction order; this is a numerical order,
//! independent of the Rust target's hardware SIMD width. Original FP64
//! mathematical norms and stable FP32 activations remain separate references.

use super::{Error, length, nonzero, product};
use half::bf16;

const EPSILON: f32 = 1e-5;

fn finite(tensor: &'static str, values: &[bf16]) -> Result<(), Error> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        Err(Error::NonFinite { tensor, index })
    } else {
        Ok(())
    }
}

// For the two declared widths, the source's cascade uses sixteen logical
// accumulators, groups of at most 256 inputs, and at most two nonzero levels.
// Four logical lanes then combine four partials each, in source order.
// These are numerical grouping rules, not device/cache tuning assumptions.
// Products round to FP32 before accumulation; do not replace with mul_add.
fn squared_sum(input: &[bf16]) -> f32 {
    let mut outer = [0_f32; 16];
    for block in input.chunks(256) {
        let mut inner = [0_f32; 16];
        for row in block.chunks_exact(16) {
            for (sum, value) in inner.iter_mut().zip(row) {
                let value = value.to_f32();
                *sum += value * value;
            }
        }
        for (total, value) in outer.iter_mut().zip(inner) {
            *total += value;
        }
    }
    let mut result = 0_f32;
    for lane in 0..4 {
        let value = ((outer[lane] + outer[lane + 4]) + outer[lane + 8]) + outer[lane + 12];
        result += value;
    }
    result
}

/// Normalize contiguous BF16 rows [rows,width] with shared BF16 [width] gains.
///
/// Only the source widths 128/2,048 and epsilon 1e-5 are implemented. Squares,
/// sum/mean, reciprocal square root and normalization use FP32. Normalized
/// values round to BF16 BEFORE gain multiplication; a second BF16 store follows.
/// No +1 gain or residual/activation fusion is implied.
///
/// All geometry, lengths and finite inputs/gains validate before any write.
/// Nonfinite FP32 reduction or BF16 output rejects; earlier rows/elements may
/// be written and must be discarded. This explicitly rejects source-overflow
/// inputs outside the qualified finite-reduction domain. Caller buffers and
/// gains remain independently owned; the operation allocates nothing.
pub fn lfm3_rmsnorm_bf16(
    input: &[bf16],
    gain: &[bf16],
    output: &mut [bf16],
    rows: usize,
    width: usize,
) -> Result<(), Error> {
    nonzero(rows, "LFM3 BF16 norm rows")?;
    if !matches!(width, 128 | 2048) {
        return Err(Error::InvalidParameter("LFM3 BF16 norm width"));
    }
    let count = product(rows, width)?;
    length("LFM3 BF16 norm input", input.len(), count)?;
    length("LFM3 BF16 norm output", output.len(), count)?;
    length("LFM3 BF16 norm gain", gain.len(), width)?;
    finite("LFM3 BF16 norm input", input)?;
    finite("LFM3 BF16 norm gain", gain)?;
    for (row, (source, target)) in input
        .chunks_exact(width)
        .zip(output.chunks_exact_mut(width))
        .enumerate()
    {
        let variance = squared_sum(source) / width as f32;
        if !variance.is_finite() {
            return Err(Error::NonFinite {
                tensor: "LFM3 BF16 norm variance",
                index: row,
            });
        }
        let inverse = 1. / (variance + EPSILON).sqrt();
        for (column, ((value, gain), out)) in source.iter().zip(gain).zip(target).enumerate() {
            let normalized = bf16::from_f32(value.to_f32() * inverse);
            let value = bf16::from_f32(normalized.to_f32() * gain.to_f32());
            if !value.is_finite() {
                return Err(Error::NonFinite {
                    tensor: "LFM3 BF16 norm output",
                    index: row * width + column,
                });
            }
            *out = value;
        }
    }
    Ok(())
}

fn sigmoid(value: f32) -> f32 {
    // Preserve the source FP32 exponential/division, including its saturation.
    // The exponential may overflow to +inf; the final sigmoid remains finite.
    1. / (1. + (-value).exp())
}

/// Canonical BF16 sigmoid: widen, source FP32 work, one nearest-even BF16 store.
/// Length/nonfinite/empty errors reject before writes. No allocation or state.
pub fn lfm3_sigmoid_bf16(input: &[bf16], output: &mut [bf16]) -> Result<(), Error> {
    nonzero(input.len(), "LFM3 BF16 sigmoid length")?;
    length("LFM3 BF16 sigmoid output", output.len(), input.len())?;
    finite("LFM3 BF16 sigmoid input", input)?;
    for (value, out) in input.iter().zip(output) {
        *out = bf16::from_f32(sigmoid(value.to_f32()));
    }
    Ok(())
}

/// Canonical BF16 SiLU: FP32 x*sigmoid(x), then one nearest-even BF16 store.
/// An MLP's subsequent BF16 up-product is a separate rounding boundary.
/// Length/nonfinite/empty errors reject before writes. Allocates no buffers.
pub fn lfm3_silu_bf16(input: &[bf16], output: &mut [bf16]) -> Result<(), Error> {
    nonzero(input.len(), "LFM3 BF16 SiLU length")?;
    length("LFM3 BF16 SiLU output", output.len(), input.len())?;
    finite("LFM3 BF16 SiLU input", input)?;
    for (value, out) in input.iter().zip(output) {
        let value = value.to_f32();
        *out = bf16::from_f32(value * sigmoid(value));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
