use super::{Error, finite, length, nonzero, product};

/// Query layout `[queries,query_heads,head_dim]`; separate K/V layouts
/// `[kv_heads,capacity,head_dim]` with the first `keys` rows valid per head.
/// Logical key/query origins are absolute; physical ring order is not admitted.
pub struct WindowAttentionShape {
    pub queries: usize,
    pub keys: usize,
    pub capacity: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub query_start: usize,
    pub key_start: usize,
    pub window: Option<usize>,
    pub scale: f32,
}

/// Scalar causal/sliding GQA with immutable inputs and caller-owned scratch.
///
/// Context has the query layout. Scores/probabilities are `[query_heads,queries,
/// keys]`; scores retain all valid-key dot products, including masked keys, for
/// diagnostics. Even an excluded key's score overflow rejects. Query heads
/// map contiguously to KV heads. An included key satisfies `key <= query` and,
/// for window W, `query-key < W`. Scale is explicit, without implicit sqrt(D).
///
/// Dot products/scaling accumulate FP64 then store FP32 scores. Stable softmax
/// uses FP64 and stores FP32 probabilities; context accumulates those FP32
/// probabilities times FP32 V in FP64 then rounds to FP32. Masked probabilities
/// are exactly zero. Unused KV capacity is neither validated nor read.
///
/// Dimensions, position/size arithmetic, all valid inputs and every query's
/// nonempty visible range are checked before writes. Arithmetic overflow can
/// leave partial scratch/context; discard all outputs on error and replay from
/// unchanged inputs. No allocation, cache mutation or extra alignment.
pub fn window_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    context: &mut [f32],
    scores: &mut [f32],
    probabilities: &mut [f32],
    shape: &WindowAttentionShape,
) -> Result<(), Error> {
    let s = shape;
    for (value, name) in [
        (s.queries, "queries"),
        (s.keys, "keys"),
        (s.query_heads, "query_heads"),
        (s.kv_heads, "kv_heads"),
        (s.head_dim, "head_dim"),
    ] {
        nonzero(value, name)?;
    }
    if s.keys > s.capacity || !s.query_heads.is_multiple_of(s.kv_heads) {
        return Err(Error::InvalidParameter("attention capacity/head mapping"));
    }
    if s.window == Some(0) || !s.scale.is_finite() || s.scale <= 0.0 {
        return Err(Error::InvalidParameter("attention window/scale"));
    }
    let key_end = s
        .key_start
        .checked_add(s.keys - 1)
        .ok_or(Error::SizeOverflow)?;
    s.query_start
        .checked_add(s.queries - 1)
        .ok_or(Error::SizeOverflow)?;
    let query_stride = product(s.query_heads, s.head_dim)?;
    let query_count = product(s.queries, query_stride)?;
    let kv_stride = product(s.capacity, s.head_dim)?;
    let kv_count = product(s.kv_heads, kv_stride)?;
    let used_kv = product(s.keys, s.head_dim)?;
    let scratch_count = product(product(s.query_heads, s.queries)?, s.keys)?;
    for (name, actual, expected) in [
        ("attention query", q.len(), query_count),
        ("attention key", k.len(), kv_count),
        ("attention value", v.len(), kv_count),
        ("attention context", context.len(), query_count),
        ("attention scores", scores.len(), scratch_count),
        (
            "attention probabilities",
            probabilities.len(),
            scratch_count,
        ),
    ] {
        length(name, actual, expected)?;
    }
    finite("attention query", q)?;
    for head in 0..s.kv_heads {
        let start = head * kv_stride;
        for (name, values) in [("attention key", k), ("attention value", v)] {
            if let Some(i) = values[start..start + used_kv]
                .iter()
                .position(|x| !x.is_finite())
            {
                return Err(Error::NonFinite {
                    tensor: name,
                    index: start + i,
                });
            }
        }
    }
    for query in 0..s.queries {
        let position = s.query_start + query;
        let first = s.window.map_or(s.key_start, |w| {
            s.key_start.max(position.saturating_sub(w - 1))
        });
        if first > key_end.min(position) {
            return Err(Error::FullyMasked { row: query });
        }
    }
    let group = s.query_heads / s.kv_heads;
    for head in 0..s.query_heads {
        let kv_base = (head / group) * kv_stride;
        for query in 0..s.queries {
            let q_base = query * query_stride + head * s.head_dim;
            let row = (head * s.queries + query) * s.keys;
            for key in 0..s.keys {
                let k_base = kv_base + key * s.head_dim;
                let mut sum = 0.0_f64;
                for feature in 0..s.head_dim {
                    sum += f64::from(q[q_base + feature]) * f64::from(k[k_base + feature]);
                }
                let score = (sum * f64::from(s.scale)) as f32;
                if !score.is_finite() {
                    return Err(Error::NonFinite {
                        tensor: "attention scores",
                        index: row + key,
                    });
                }
                scores[row + key] = score;
            }
            let position = s.query_start + query;
            let first = s.window.map_or(s.key_start, |w| {
                s.key_start.max(position.saturating_sub(w - 1))
            }) - s.key_start;
            let last = key_end.min(position) - s.key_start;
            let included = &scores[row + first..=row + last];
            let maximum = included.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator: f64 = included
                .iter()
                .map(|&x| (f64::from(x) - f64::from(maximum)).exp())
                .sum();
            probabilities[row..row + s.keys].fill(0.0);
            for key in first..=last {
                probabilities[row + key] = ((f64::from(scores[row + key]) - f64::from(maximum))
                    .exp()
                    / denominator) as f32;
            }
            for feature in 0..s.head_dim {
                let mut sum = 0.0_f64;
                for key in first..=last {
                    sum += f64::from(probabilities[row + key])
                        * f64::from(v[kv_base + key * s.head_dim + feature]);
                }
                let value = sum as f32;
                if !value.is_finite() {
                    return Err(Error::NonFinite {
                        tensor: "attention context",
                        index: q_base + feature,
                    });
                }
                context[q_base + feature] = value;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
