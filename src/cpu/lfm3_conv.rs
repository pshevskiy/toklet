//! Canonical LFM3 BF16 seven-tap CPU convolution over immutable history.

use super::{Error, length, lfm3_bf16::finite, nonzero, product};
use half::bf16;

/// Source BF16 depthwise convolution with seven oldest-to-current taps.
///
/// Input/output are contiguous [tokens,channels], coefficients [channels,7],
/// and immutable history [6,channels] oldest-to-newest. Input already contains
/// the BF16 B*x product. Output rounds to BF16 before the separate C product.
///
/// FP32 products use the pinned Torch CPU BLAS four-partial numerical order:
/// partial zero receives taps 0,4,5,6; partials one/two/three receive taps 1/2/3,
/// then combine in order 0+1+2+3. This is a source order, not a SIMD-width or
/// device assumption. No bias, activation, gate, mask or state commit is fused.
///
/// All geometry, lengths and finite operands validate before writes. Numerical
/// FP32/BF16 overflow rejects with discardable partial output. History stays
/// immutable, all storage is caller-owned, and no allocation is performed.
pub fn lfm3_conv7_bf16(
    input: &[bf16],
    weight: &[bf16],
    history: &[bf16],
    output: &mut [bf16],
    tokens: usize,
    channels: usize,
) -> Result<(), Error> {
    nonzero(tokens, "LFM3 BF16 convolution tokens")?;
    nonzero(channels, "LFM3 BF16 convolution channels")?;
    let elements = product(tokens, channels)?;
    length("LFM3 BF16 convolution input", input.len(), elements)?;
    length("LFM3 BF16 convolution output", output.len(), elements)?;
    length(
        "LFM3 BF16 convolution weight",
        weight.len(),
        product(channels, 7)?,
    )?;
    length(
        "LFM3 BF16 convolution history",
        history.len(),
        product(channels, 6)?,
    )?;
    finite("LFM3 BF16 convolution input", input)?;
    finite("LFM3 BF16 convolution weight", weight)?;
    finite("LFM3 BF16 convolution history", history)?;
    for (token, target) in output.chunks_exact_mut(channels).enumerate() {
        for (channel, (taps, out)) in weight.chunks_exact(7).zip(target).enumerate() {
            let mut partial = [0_f32; 4];
            for (tap, weight) in taps.iter().enumerate() {
                let preceding = 6 - tap;
                let value = if token < preceding {
                    history[(token + tap) * channels + channel]
                } else {
                    input[(token - preceding) * channels + channel]
                };
                let term = value.to_f32() * weight.to_f32();
                partial[if tap < 4 { tap } else { 0 }] += term;
            }
            let sum = ((partial[0] + partial[1]) + partial[2]) + partial[3];
            let value = bf16::from_f32(sum);
            if !value.is_finite() {
                return Err(Error::NonFinite {
                    tensor: "LFM3 BF16 convolution output",
                    index: token * channels + channel,
                });
            }
            *out = value;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
