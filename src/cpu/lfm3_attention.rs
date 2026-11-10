//! BF16 attention for the pinned LFM3 eager CPU profile.

use half::bf16;

use super::{
    Error, KvSegment, SpanAttentionShape, length, lfm3_bf16::finite, lfm3_projection, nonzero,
    product, span_attention::validate_segments,
};

const QUERY_HEADS: usize = 16;
const KV_HEADS: usize = 4;
const WIDTH: usize = 128;
const WINDOW: usize = 256;
pub const LFM3_ATTENTION_SCALE: f32 = f32::from_bits(0x3db5_04f3);

/// Compute LFM3 causal or 256-token window GQA over borrowed BF16 KV spans.
///
/// Q and context are [query, 16, 128]. K/V are per-head contiguous rows in
/// logical span order. The caller supplies one FP32 scratch row of `s.keys`
/// elements. The source rounds QK, scale, additive mask, probabilities and
/// context separately to BF16. Softmax and dot accumulators use FP32. Only
/// finite input values are accepted, including keys and values outside a
/// query's visible interval. Stride padding is never read.
///
/// The operation validates all shapes and input finiteness before writes,
/// allocates nothing and never publishes cache state. On arithmetic overflow,
/// discard scratch and any partial context before retrying.
pub fn lfm3_attention_bf16(
    q: &[bf16],
    segments: &[KvSegment<'_, bf16>; 3],
    context: &mut [bf16],
    scratch: &mut [f32],
    s: &SpanAttentionShape,
) -> Result<(), Error> {
    validate(q, segments, context.len(), scratch.len(), s)?;
    // Both source BMM products have the same scalar operation count. The
    // small path precedes the two stride-dependent GEMM reduction paths.
    if s.queries * s.keys <= 3 {
        run::<true>(q, segments, context, scratch, s)
    } else {
        run::<false>(q, segments, context, scratch, s)
    }
}

fn validate(
    q: &[bf16],
    segments: &[KvSegment<'_, bf16>; 3],
    context_len: usize,
    scratch_len: usize,
    s: &SpanAttentionShape,
) -> Result<(), Error> {
    nonzero(s.queries, "LFM3 attention queries")?;
    nonzero(s.keys, "LFM3 attention keys")?;
    if (s.query_heads, s.kv_heads, s.head_dim) != (QUERY_HEADS, KV_HEADS, WIDTH)
        || !matches!(s.window, None | Some(WINDOW))
        || s.scale != LFM3_ATTENTION_SCALE
    {
        return Err(Error::InvalidParameter("LFM3 attention heads/window/scale"));
    }
    let end = s
        .query_start
        .checked_add(s.queries)
        .ok_or(Error::SizeOverflow)?;
    let key_end = s.key_start.checked_add(s.keys).ok_or(Error::SizeOverflow)?;
    let first_key = s.window.map_or(0, |w| s.query_start.saturating_sub(w - 1));
    if end != key_end || s.key_start != first_key {
        return Err(Error::InvalidParameter("LFM3 attention query/key range"));
    }
    let count = product(s.queries, QUERY_HEADS * WIDTH)?;
    length("LFM3 attention query", q.len(), count)?;
    length("LFM3 attention context", context_len, count)?;
    length("LFM3 attention scratch", scratch_len, s.keys)?;
    product(product(QUERY_HEADS, s.queries)?, s.keys)?;
    validate_segments(segments, s)?;
    finite("LFM3 attention query", q)?;
    for segment in segments.iter().filter(|span| span.keys != 0) {
        let rows = segment.keys * WIDTH;
        for head in 0..KV_HEADS {
            let start = head * segment.head_stride;
            finite("LFM3 attention key", &segment.k[start..start + rows])?;
            finite("LFM3 attention value", &segment.v[start..start + rows])?;
        }
    }
    Ok(())
}

fn qk_dot<const SMALL: bool>(q: &[bf16], key: &[bf16]) -> f32 {
    if SMALL {
        q.iter()
            .zip(key)
            .fold(0.0, |sum, (q, k)| q.to_f32().mul_add(k.to_f32(), sum))
    } else {
        lfm3_projection::dot(q, key)
    }
}

fn scale_score(score: bf16) -> bf16 {
    bf16::from_f32(score.to_f32() * LFM3_ATTENTION_SCALE)
}

fn mask_score(score: bf16, visible: bool) -> f32 {
    let mask = if visible { 0.0 } else { bf16::MIN.to_f32() };
    bf16::from_f32(score.to_f32() + mask).to_f32()
}

// Scalar adaptation of the nonpositive range of SLEEF xexpf (AdvSIMD/FMA).
// Copyright Naoki Shibata and contributors 2010–2024; ARM Ltd. 2010–2024.
// Boost Software License 1.0, see lfm3_attention/SLEEF-LICENSE.txt.
// Source revision 5a1d179df9cf652951b59010a2d2075372d67f68.
fn source_exp(delta: f32) -> f32 {
    debug_assert!(delta <= 0.0);
    if delta < -104.0 {
        return 0.0;
    }
    let exponent = (delta * std::f32::consts::LOG2_E).round_ties_even() as i32;
    let multiple = exponent as f32;
    let remainder = multiple.mul_add(-0.693_145_75, delta);
    let remainder = multiple.mul_add(-1.428_606_8e-6, remainder);

    let polynomial = 1.985_276_2e-4_f32.mul_add(remainder, 0.001_393_043_6);
    let polynomial = polynomial.mul_add(remainder, 0.008_333_361);
    let polynomial = polynomial.mul_add(remainder, 0.041_666_485);
    let polynomial = polynomial.mul_add(remainder, 0.166_666_67);
    let polynomial = polynomial.mul_add(remainder, 0.5);
    let value = 1.0 + (remainder * remainder).mul_add(polynomial, remainder);
    // Apply two normal powers of two in source order to preserve underflow.
    let lower = exponent >> 1;
    let power_of_two = |n: i32| f32::from_bits(((n + 127) as u32) << 23);
    value * power_of_two(lower) * power_of_two(exponent - lower)
}

fn softmax(row: &mut [f32]) {
    let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for score in row.iter_mut() {
        *score = source_exp(*score - maximum);
    }

    // Source FP32 softmax reduces four logical lanes. Short rows use a
    // scalar fold, and remaining values go to the corresponding low lanes.
    let sum = if row.len() < 4 {
        row.iter().copied().sum()
    } else {
        let mut groups = row.chunks_exact(4);
        let mut lanes = [0.0; 4];
        for group in groups.by_ref() {
            for (lane, value) in lanes.iter_mut().zip(group) {
                *lane += value;
            }
        }
        for (lane, value) in lanes.iter_mut().zip(groups.remainder()) {
            *lane += value;
        }
        (lanes[0] + lanes[2]) + (lanes[1] + lanes[3])
    };
    let reciprocal = 1.0 / sum;
    for exponent in row {
        *exponent *= reciprocal;
    }
}

fn mix_values<const SMALL: bool>(
    probabilities: &[f32],
    segments: &[KvSegment<'_, bf16>; 3],
    head: usize,
) -> [f32; WIDTH] {
    let mut partial = [[0.0; WIDTH]; 4];
    let aligned = probabilities.len() / 4 * 4;
    let mut logical = 0;
    for span in segments.iter().filter(|span| span.keys != 0) {
        let base = head * span.head_stride;
        let values = &span.v[base..base + span.keys * WIDTH];
        for value in values.chunks_exact(WIDTH) {
            let lane = if !SMALL && logical < aligned {
                logical % 4
            } else {
                0
            };
            let probability = probabilities[logical];
            for (sum, feature) in partial[lane].iter_mut().zip(value) {
                if SMALL {
                    *sum = probability.mul_add(feature.to_f32(), *sum);
                } else {
                    *sum += probability * feature.to_f32();
                }
            }
            logical += 1;
        }
    }
    let [mut output, b, c, d] = partial;
    if !SMALL {
        for (((sum, b), c), d) in output.iter_mut().zip(b).zip(c).zip(d) {
            *sum = ((*sum + b) + c) + d;
        }
    }
    output
}

fn run<const SMALL: bool>(
    q: &[bf16],
    segments: &[KvSegment<'_, bf16>; 3],
    context: &mut [bf16],
    scratch: &mut [f32],
    s: &SpanAttentionShape,
) -> Result<(), Error> {
    for head in 0..QUERY_HEADS {
        let kv_head = head / (QUERY_HEADS / KV_HEADS);
        for token in 0..s.queries {
            let begin = (token * QUERY_HEADS + head) * WIDTH;
            let query = &q[begin..begin + WIDTH];
            let position = s.query_start + token;
            let first = s.window.map_or(0, |w| position.saturating_sub(w - 1));
            let mut logical = 0;
            for span in segments.iter().filter(|span| span.keys != 0) {
                let base = kv_head * span.head_stride;
                let keys = &span.k[base..base + span.keys * WIDTH];
                for key in keys.chunks_exact(WIDTH) {
                    let score = bf16::from_f32(qk_dot::<SMALL>(query, key));
                    if !score.is_finite() {
                        return Err(Error::NonFinite {
                            tensor: "LFM3 attention QK",
                            index: (head * s.queries + token) * s.keys + logical,
                        });
                    }
                    let key_position = s.key_start + logical;
                    let scaled = scale_score(score);
                    scratch[logical] =
                        mask_score(scaled, (first..=position).contains(&key_position));
                    logical += 1;
                }
            }
            softmax(scratch);
            for value in scratch.iter_mut() {
                *value = bf16::from_f32(*value).to_f32();
            }
            let mixed = mix_values::<SMALL>(scratch, segments, kv_head);
            for (feature, (dst, value)) in context[begin..begin + WIDTH]
                .iter_mut()
                .zip(mixed)
                .enumerate()
            {
                let word = bf16::from_f32(value);
                if !word.is_finite() {
                    return Err(Error::NonFinite {
                        tensor: "LFM3 attention context",
                        index: begin + feature,
                    });
                }
                *dst = word;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
