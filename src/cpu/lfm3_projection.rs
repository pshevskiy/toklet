//! Bias-free BF16 projection with the pinned LFM3 source CPU reduction profile.
use super::{Error, length, lfm3_bf16::finite, nonzero, product};
use half::bf16;

/// Contiguous input [tokens,in], weights [out,in] and output [tokens,out].
///
/// BF16 operands widen exactly; source-order FP32 fused multiply/add partials
/// reduce before one nearest-even BF16 output store. The 32 partials and their
/// tree describe the pinned no-BFDOT Torch CPU numerical profile, not a native
/// SIMD-width assumption or hardware tuning. No bias or graph/state operation
/// is fused. All geometry/finite operands validate before writes; arithmetic
/// overflow leaves discardable partial output. Caller storage is borrowed and
/// no allocation, indirect numerical call or expanded weight copy is required.
pub fn lfm3_linear_bf16(
    input: &[bf16],
    weight: &[bf16],
    output: &mut [bf16],
    tokens: usize,
    in_features: usize,
    out_features: usize,
) -> Result<(), Error> {
    nonzero(tokens, "LFM3 BF16 projection tokens")?;
    nonzero(in_features, "LFM3 BF16 projection input width")?;
    nonzero(out_features, "LFM3 BF16 projection output width")?;
    length(
        "LFM3 BF16 projection input",
        input.len(),
        product(tokens, in_features)?,
    )?;
    length(
        "LFM3 BF16 projection weight",
        weight.len(),
        product(out_features, in_features)?,
    )?;
    length(
        "LFM3 BF16 projection output",
        output.len(),
        product(tokens, out_features)?,
    )?;
    finite("LFM3 BF16 projection input", input)?;
    finite("LFM3 BF16 projection weight", weight)?;
    for (token, (x, target)) in input
        .chunks_exact(in_features)
        .zip(output.chunks_exact_mut(out_features))
        .enumerate()
    {
        for (row, (w, out)) in weight.chunks_exact(in_features).zip(target).enumerate() {
            let value = bf16::from_f32(dot(x, w));
            if !value.is_finite() {
                return Err(Error::NonFinite {
                    tensor: "LFM3 BF16 projection output",
                    index: token * out_features + row,
                });
            }
            *out = value;
        }
    }
    Ok(())
}

// The public entry validates both complete operands before calling this
// reduction. Finite BF16 widens exactly by placing its bits in the high half
// of an FP32 word; no rounding or NaN quieting is needed in this admitted domain.
// Keep the public finite checks: half::bf16::to_f32 has additional NaN behavior
// which this private finite-only conversion deliberately does not implement.
fn widen_finite(value: bf16) -> f32 {
    f32::from_bits(u32::from(value.to_bits()) << 16)
}

pub(super) fn dot(x: &[bf16], w: &[bf16]) -> f32 {
    let mut partial = [0_f32; 32];
    let aligned = x.len() / 32 * 32;
    for base in (0..aligned).step_by(32) {
        for (lane, sum) in partial.iter_mut().enumerate() {
            *sum = widen_finite(x[base + lane]).mul_add(widen_finite(w[base + lane]), *sum);
        }
    }
    for offset in [4, 2, 1] {
        for register in 0..offset {
            for lane in 0..4 {
                partial[register * 4 + lane] += partial[(register + offset) * 4 + lane];
            }
        }
    }
    let mut sum = (partial[0] + partial[1]) + (partial[2] + partial[3]);
    let vector_end = x.len() / 8 * 8;
    let mut tail = [0_f32; 4];
    for base in (aligned..vector_end).step_by(8) {
        for (lane, acc) in tail.iter_mut().enumerate() {
            *acc = widen_finite(x[base + lane]).mul_add(widen_finite(w[base + lane]), *acc);
            *acc = widen_finite(x[base + lane + 4]).mul_add(widen_finite(w[base + lane + 4]), *acc);
        }
    }
    sum += (tail[0] + tail[1]) + (tail[2] + tail[3]);
    for index in vector_end..x.len() {
        sum = widen_finite(x[index]).mul_add(widen_finite(w[index]), sum);
    }
    sum
}

#[cfg(test)]
mod tests;
