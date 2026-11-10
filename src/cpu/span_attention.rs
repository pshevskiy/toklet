use super::{Error, finite, length, nonzero, product};

/// One logical contiguous key interval, with separately borrowed typed K/V rows.
/// For KV head h, data begins at `h * head_stride` elements, then `[keys,D]`.
/// Nonempty backing slices have exactly `(kv_heads-1)*head_stride + keys*D`
/// elements; initialized gaps between heads are not read. Logical intervals
/// concatenate in array order. Empty segments require empty K/V; their origin
/// and stride are ignored. No alignment beyond T is assumed. This descriptor
/// chooses no numerical arithmetic or conversion.
#[derive(Clone, Copy)]
pub struct KvSegment<'a, T> {
    pub k: &'a [T],
    pub v: &'a [T],
    pub keys: usize,
    pub head_stride: usize,
    pub key_start: usize,
}

/// Existing FP32 numerical API; BF16 storage does not change its arithmetic.
pub type AttentionKvSegment<'a> = KvSegment<'a, f32>;

pub struct SpanAttentionShape {
    pub queries: usize,
    pub keys: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub query_start: usize,
    pub key_start: usize,
    pub window: Option<usize>,
    pub scale: f32,
}

/// Causal/window GQA over at most two borrowed old-ring spans and a new chunk.
/// Q/context are `[queries,query_heads,D]`; scores/probabilities are
/// `[query_heads,queries,keys]` in logical segment order. No KV linearization,
/// allocation, cache mutation, dynamic dispatch or per-scalar variant selection.
///
/// Precision and reduction order match `window_attention`: FP64 dot/scale to
/// FP32 scores, stable FP64 softmax to FP32 probabilities, then a single FP64
/// context sum over logical keys to FP32. Masked probabilities are exact zero;
/// even a masked key's diagnostic score overflow rejects. Geometry, all used
/// finite inputs and every visible range are validated before output writes.
/// Arithmetic failure may leave partial outputs: discard them and replay from
/// unchanged inputs. Empty sequences make no numerical calls.
pub fn span_attention(
    q: &[f32],
    segments: &[AttentionKvSegment<'_>; 3],
    context: &mut [f32],
    scores: &mut [f32],
    probabilities: &mut [f32],
    s: &SpanAttentionShape,
) -> Result<(), Error> {
    for (value, name) in [
        (s.queries, "queries"),
        (s.keys, "keys"),
        (s.query_heads, "query_heads"),
        (s.kv_heads, "kv_heads"),
        (s.head_dim, "head_dim"),
    ] {
        nonzero(value, name)?;
    }
    if !s.query_heads.is_multiple_of(s.kv_heads)
        || s.window == Some(0)
        || !s.scale.is_finite()
        || s.scale <= 0.
    {
        return Err(Error::InvalidParameter(
            "segmented attention heads/window/scale",
        ));
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
    let scratch_count = product(product(s.query_heads, s.queries)?, s.keys)?;
    for (name, actual, expected) in [
        ("attention query", q.len(), query_count),
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
    validate_segments(segments, s)?;
    finite("attention query", q)?;
    for segment in segments {
        if segment.keys == 0 {
            continue;
        }
        let rows = segment.keys * s.head_dim;
        for head in 0..s.kv_heads {
            let start = head * segment.head_stride;
            for (name, values) in [
                ("attention segment key", segment.k),
                ("attention segment value", segment.v),
            ] {
                if let Some(i) = values[start..start + rows]
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
        let kv_head = head / group;
        for query in 0..s.queries {
            let q_base = query * query_stride + head * s.head_dim;
            let row = (head * s.queries + query) * s.keys;
            let mut logical = 0;
            for segment in segments {
                if segment.keys == 0 {
                    continue;
                }
                let base = kv_head * segment.head_stride;
                for key in 0..segment.keys {
                    let k_base = base + key * s.head_dim;
                    let mut sum = 0_f64;
                    for feature in 0..s.head_dim {
                        sum +=
                            f64::from(q[q_base + feature]) * f64::from(segment.k[k_base + feature]);
                    }
                    let score = (sum * f64::from(s.scale)) as f32;
                    if !score.is_finite() {
                        return Err(Error::NonFinite {
                            tensor: "attention scores",
                            index: row + logical + key,
                        });
                    }
                    scores[row + logical + key] = score;
                }
                logical += segment.keys;
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
            probabilities[row..row + s.keys].fill(0.);
            for key in first..=last {
                probabilities[row + key] = ((f64::from(scores[row + key]) - f64::from(maximum))
                    .exp()
                    / denominator) as f32;
            }
            for feature in 0..s.head_dim {
                let mut sum = 0_f64;
                let mut logical = 0;
                for segment in segments {
                    if segment.keys == 0 {
                        continue;
                    }
                    let begin = first.saturating_sub(logical).min(segment.keys);
                    let end = (last + 1).saturating_sub(logical).min(segment.keys);
                    let base = kv_head * segment.head_stride;
                    for key in begin..end {
                        sum += f64::from(probabilities[row + logical + key])
                            * f64::from(segment.v[base + key * s.head_dim + feature]);
                    }
                    logical += segment.keys;
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

pub(super) fn validate_segments<T>(
    segments: &[KvSegment<'_, T>; 3],
    s: &SpanAttentionShape,
) -> Result<(), Error> {
    let mut keys = 0_usize;
    for segment in segments {
        let rows = product(segment.keys, s.head_dim)?;
        let expected = if segment.keys == 0 {
            0
        } else {
            if segment.head_stride < rows {
                return Err(Error::InvalidParameter("attention segment heads overlap"));
            }
            let start = s.key_start.checked_add(keys).ok_or(Error::SizeOverflow)?;
            if segment.key_start != start {
                return Err(Error::InvalidParameter("attention segment logical origin"));
            }
            product(s.kv_heads - 1, segment.head_stride)?
                .checked_add(rows)
                .ok_or(Error::SizeOverflow)?
        };
        length("attention segment key", segment.k.len(), expected)?;
        length("attention segment value", segment.v.len(), expected)?;
        keys = keys.checked_add(segment.keys).ok_or(Error::SizeOverflow)?;
    }
    length("attention segment keys", keys, s.keys)?;
    Ok(())
}

#[cfg(test)]
mod tests;
