use super::{Error, finite, length, nonzero, product, softmax};

/// Apply Qwen split-half rotary pairing to `[tokens, heads, head_dim]`.
///
/// Pair feature `j` with `j + head_dim/2`, using frequency
/// `theta^(-2*j/head_dim)` and unscaled absolute position `offset + token`.
/// Frequencies/angles/trigonometry use FP64, then FP32 sine/cosine and FP32
/// rotation arithmetic. Head dimension must be positive and even; theta must
/// be finite and positive. Positions above 2^53 are rejected rather than rounded.
pub fn rotary(
    input: &[f32],
    output: &mut [f32],
    tokens: usize,
    heads: usize,
    head_dim: usize,
    offset: usize,
    theta: f32,
) -> Result<(), Error> {
    nonzero(tokens, "tokens")?;
    nonzero(heads, "heads")?;
    nonzero(head_dim, "head_dim")?;
    if !head_dim.is_multiple_of(2) {
        return Err(Error::InvalidParameter("rotary head_dim must be even"));
    }
    if !theta.is_finite() || theta <= 0.0 {
        return Err(Error::InvalidParameter("theta must be finite and positive"));
    }
    let last = offset.checked_add(tokens - 1).ok_or(Error::SizeOverflow)?;
    if last as u128 > 1_u128 << 53 {
        return Err(Error::InvalidParameter(
            "rotary positions must be at most 2^53",
        ));
    }
    let token_stride = product(heads, head_dim)?;
    let count = product(tokens, token_stride)?;
    length("rotary input", input.len(), count)?;
    length("rotary output", output.len(), count)?;
    finite("rotary input", input)?;
    let half = head_dim / 2;
    // Hoist invariant frequencies outside token/head loops, without allocating.
    for feature in 0..half {
        let frequency = f64::from(theta).powf(-(2.0 * feature as f64) / head_dim as f64);
        for token in 0..tokens {
            let angle = (offset + token) as f64 * frequency;
            let (sin, cos) = angle.sin_cos();
            let (sin, cos) = (sin as f32, cos as f32);
            for head in 0..heads {
                let first = token * token_stride + head * head_dim + feature;
                let second = first + half;
                let x = input[first] * cos - input[second] * sin;
                let y = input[second] * cos + input[first] * sin;
                if !x.is_finite() || !y.is_finite() {
                    return Err(Error::NonFinite {
                        tensor: "rotary output",
                        index: if !x.is_finite() { first } else { second },
                    });
                }
                output[first] = x;
                output[second] = y;
            }
        }
    }
    Ok(())
}

/// Full-prefix attention dimensions, with no expanded copies of KV heads.
pub struct AttentionShape {
    pub tokens: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

/// Scalar causal scaled dot-product attention over a complete prefix.
///
/// Q/context: `[tokens, query_heads, head_dim]`; K/V:
/// `[tokens, kv_heads, head_dim]`; scores/probabilities:
/// `[query_heads, query_tokens, key_tokens]`. Scores retain unmasked values for
/// diagnostics. Query head `h` reads KV head `h / (query_heads / kv_heads)`.
/// Only keys at or before the query position contribute; masked probabilities
/// are exactly zero. Dot products and context sums use FP32, softmax FP64.
///
/// All dimensions, lengths and finite inputs are checked before writes. No
/// buffers are allocated. Discard partial buffers on arithmetic failure.
pub fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    context: &mut [f32],
    scores: &mut [f32],
    probabilities: &mut [f32],
    shape: &AttentionShape,
) -> Result<(), Error> {
    let AttentionShape {
        tokens,
        query_heads,
        kv_heads,
        head_dim,
    } = *shape;
    nonzero(tokens, "tokens")?;
    nonzero(query_heads, "query_heads")?;
    nonzero(kv_heads, "kv_heads")?;
    nonzero(head_dim, "head_dim")?;
    if !query_heads.is_multiple_of(kv_heads) {
        return Err(Error::InvalidParameter(
            "query_heads must be divisible by kv_heads",
        ));
    }
    let q_stride = product(query_heads, head_dim)?;
    let kv_stride = product(kv_heads, head_dim)?;
    let q_count = product(tokens, q_stride)?;
    let kv_count = product(tokens, kv_stride)?;
    let score_count = product(product(query_heads, tokens)?, tokens)?;
    length("q", q.len(), q_count)?;
    length("k", k.len(), kv_count)?;
    length("v", v.len(), kv_count)?;
    length("attention context", context.len(), q_count)?;
    length("attention scores", scores.len(), score_count)?;
    length("attention probabilities", probabilities.len(), score_count)?;
    finite("q", q)?;
    finite("k", k)?;
    finite("v", v)?;
    let group = query_heads / kv_heads;
    let denominator = (head_dim as f32).sqrt();
    for head in 0..query_heads {
        let kv_head = head / group;
        for query in 0..tokens {
            let q_start = query * q_stride + head * head_dim;
            let row_start = (head * tokens + query) * tokens;
            for key in 0..tokens {
                let k_start = key * kv_stride + kv_head * head_dim;
                let mut dot = 0.0;
                for feature in 0..head_dim {
                    dot += q[q_start + feature] * k[k_start + feature];
                }
                let score = dot / denominator;
                if !score.is_finite() {
                    return Err(Error::NonFinite {
                        tensor: "attention scores",
                        index: row_start + key,
                    });
                }
                scores[row_start + key] = score;
            }
        }
    }
    for head in 0..query_heads {
        let kv_head = head / group;
        for query in 0..tokens {
            let row_start = (head * tokens + query) * tokens;
            let included_end = row_start + query + 1;
            softmax(
                &scores[row_start..included_end],
                None,
                &mut probabilities[row_start..included_end],
                1,
                query + 1,
            )?;
            probabilities[included_end..row_start + tokens].fill(0.0);
            let context_start = query * q_stride + head * head_dim;
            for feature in 0..head_dim {
                let mut value = 0.0;
                for key in 0..=query {
                    value += probabilities[row_start + key]
                        * v[key * kv_stride + kv_head * head_dim + feature];
                }
                if !value.is_finite() {
                    return Err(Error::NonFinite {
                        tensor: "attention context",
                        index: context_start + feature,
                    });
                }
                context[context_start + feature] = value;
            }
        }
    }
    Ok(())
}
