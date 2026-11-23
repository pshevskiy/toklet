use super::{AttentionKvSegment, Error};
use std::ops::RangeInclusive;

/// Accumulate contiguous output features while visiting the same chronological
/// visible keys in each binary64 sum. Upstream attention validates every span.
pub(super) fn context_row(
    segments: &[AttentionKvSegment<'_>; 3],
    probabilities: &[f32],
    visible: RangeInclusive<usize>,
    kv_head: usize,
    output_base: usize,
    context: &mut [f32],
) -> Result<(), Error> {
    const TILE: usize = 32;
    let head_dim = context.len();
    let first = *visible.start();
    let end = *visible.end() + 1;
    for (tile, output) in context.chunks_mut(TILE).enumerate() {
        let first_feature = tile * TILE;
        let mut sums = [0_f64; TILE];
        let active = &mut sums[..output.len()];
        let mut logical = 0;
        for segment in segments {
            if segment.keys == 0 {
                continue;
            }
            let begin = first.saturating_sub(logical).min(segment.keys);
            let end = end.saturating_sub(logical).min(segment.keys);
            let head_base = kv_head * segment.head_stride;
            for key in begin..end {
                let weight = f64::from(probabilities[logical + key]);
                let start = head_base + key * head_dim + first_feature;
                let values = &segment.v[start..start + active.len()];
                for (sum, &value) in active.iter_mut().zip(values) {
                    *sum += weight * f64::from(value);
                }
            }
            logical += segment.keys;
        }
        for (feature, (&sum, target)) in active.iter().zip(output).enumerate() {
            let value = sum as f32;
            if !value.is_finite() {
                return Err(Error::NonFinite {
                    tensor: "attention context",
                    index: output_base + first_feature + feature,
                });
            }
            *target = value;
        }
    }
    Ok(())
}
