use super::*;
use crate::cpu::{
    DenseDecoderScratch, DenseDecoderShape, PerLayerInputShape, ProportionalRotaryShape,
    SpanAttentionShape,
};
use crate::model::window_cache::{KvView, PreparedKv};

fn queries(
    index: usize,
    model: &Model<'_>,
    s: &mut Scratch,
    count: usize,
    offset: usize,
) -> Result<(), Error> {
    let layer = &model.weights.layers[index];
    let d = dimension(index);
    let q = count * HEADS * d;
    cpu::rmsnorm(
        &s.hidden[..count * HIDDEN],
        layer.input_norm,
        &mut s.norm[..count * HIDDEN],
        count,
        HIDDEN,
        EPSILON,
    )?;
    cpu::matmul_f64(
        &s.norm[..count * HIDDEN],
        layer.q,
        None,
        &mut s.q[..q],
        count,
        HIDDEN,
        HEADS * d,
    )?;
    cpu::rmsnorm(
        &s.q[..q],
        layer.q_norm,
        &mut s.q_norm[..q],
        count * HEADS,
        d,
        EPSILON,
    )?;
    cpu::proportional_rotary(
        &s.q_norm[..q],
        &mut s.q[..q],
        &rotary(index, count, HEADS, offset),
    )?;
    Ok(())
}

fn rotary(index: usize, tokens: usize, heads: usize, offset: usize) -> ProportionalRotaryShape {
    let local = index.is_multiple_of(2);
    ProportionalRotaryShape {
        tokens,
        heads,
        head_dim: dimension(index),
        rotated_dims: if local { 8 } else { 4 },
        offset,
        base: if local { 10000. } else { 1e6 },
        factor: 1.,
    }
}

fn finish(
    index: usize,
    model: &Model<'_>,
    s: &mut Scratch,
    view: KvView<'_, '_>,
    count: usize,
    offset: usize,
) -> Result<(), Error> {
    let d = dimension(index);
    let q = count * HEADS * d;
    let keys = view.end() - view.start();
    let scores = HEADS * count * keys;
    cpu::span_attention(
        &s.q[..q],
        &view.segments(),
        &mut s.context[..q],
        &mut s.scores[..scores],
        &mut s.probabilities[..scores],
        &SpanAttentionShape {
            queries: count,
            keys,
            query_heads: HEADS,
            kv_heads: 1,
            head_dim: d,
            query_start: offset,
            key_start: view.start(),
            window: if index.is_multiple_of(2) {
                Some(WINDOW)
            } else {
                None
            },
            scale: 1.,
        },
    )?;
    cpu::matmul_f64(
        &s.context[..q],
        model.weights.layers[index].o,
        None,
        &mut s.attention[..count * HIDDEN],
        count,
        HEADS * d,
        HIDDEN,
    )?;
    for token in 0..count {
        let start = (token * LAYERS + index) * PLE;
        s.layer_ple[token * PLE..(token + 1) * PLE].copy_from_slice(&s.ple[start..start + PLE]);
    }
    let ff = count * intermediate(index);
    let hidden = count * HIDDEN;
    cpu::finish_dense_decoder(
        &s.hidden[..hidden],
        &s.attention[..hidden],
        Some(&s.layer_ple[..count * PLE]),
        &model.weights.layers[index].finish,
        &mut DenseDecoderScratch {
            hidden_work: &mut s.dense_work[..hidden],
            hidden_norm: &mut s.dense_norm[..hidden],
            gate: &mut s.gate[..ff],
            up: &mut s.up[..ff],
            product: &mut s.product[..ff],
        },
        &mut s.next[..hidden],
        &DenseDecoderShape {
            tokens: count,
            hidden: HIDDEN,
            intermediate: intermediate(index),
            epsilon: EPSILON,
        },
    )?;
    std::mem::swap(&mut s.hidden, &mut s.next);
    Ok(())
}

fn producer<'a>(
    index: usize,
    model: &Model<'_>,
    cache: &WindowKvCache,
    stage: &'a mut Staging,
    s: &mut Scratch,
    count: usize,
    offset: usize,
) -> Result<PreparedKv<'a>, Error> {
    queries(index, model, s, count, offset)?;
    let d = dimension(index);
    let kv = count * d;
    let weights = model.weights.layers[index]
        .kv
        .as_ref()
        .ok_or(Error::InvalidConfig("Gemma producer projection"))?;
    cpu::matmul_f64(
        &s.norm[..count * HIDDEN],
        weights.k,
        None,
        &mut s.kv[..kv],
        count,
        HIDDEN,
        d,
    )?;
    cpu::rmsnorm(
        &s.kv[..kv],
        weights.k_norm,
        &mut s.kv_norm[..kv],
        count,
        d,
        EPSILON,
    )?;
    cpu::proportional_rotary(
        &s.kv_norm[..kv],
        &mut stage.k[..kv],
        &rotary(index, count, 1, offset),
    )?;
    cpu::matmul_f64(
        &s.norm[..count * HIDDEN],
        weights.v,
        None,
        &mut s.kv[..kv],
        count,
        HIDDEN,
        d,
    )?;
    cpu::rmsnorm(
        &s.kv[..kv],
        &model.unit_norm[..d],
        &mut stage.v[..kv],
        count,
        d,
        EPSILON,
    )?;
    let frame = cache.prepare(&stage.k[..kv], &stage.v[..kv], count)?;
    finish(index, model, s, cache.view(&frame)?, count, offset)?;
    Ok(frame)
}

pub(super) fn run(session: &mut Session<'_, '_>, tokens: &[u32]) -> Result<(), Error> {
    cpu::nonzero(tokens.len(), "Gemma chunk")?;
    let count = tokens.len();
    let offset = session.committed;
    let after = offset.checked_add(count).ok_or(cpu::Error::SizeOverflow)?;
    if after > MAX_POSITIONS {
        return Err(Error::Capacity {
            offset,
            tokens: count,
            limit: MAX_POSITIONS,
        });
    }
    for (position, &token) in tokens.iter().enumerate() {
        if token as usize >= VOCAB_SIZE {
            return Err(cpu::Error::Token {
                position,
                token,
                vocab_size: VOCAB_SIZE,
            }
            .into());
        }
    }
    if session.caches.iter().any(|cache| {
        cache.offset() != offset || cache.generation() != session.caches[0].generation()
    }) {
        return Err(Error::InvalidConfig("Gemma committed cache prefix"));
    }
    let model = session.model;
    let s = &mut session.scratch;
    let h = count * HIDDEN;
    let p = count * LAYERS * PLE;
    cpu::embedding(
        tokens,
        model.weights.embedding,
        &mut s.next[..h],
        VOCAB_SIZE,
        HIDDEN,
    )?;
    cpu::prepare_per_layer_projection_input(&s.next[..h], &mut s.hidden[..h], count, HIDDEN)?;
    cpu::embedding(
        tokens,
        model.weights.per_layer_embedding,
        &mut s.ple_lookup[..p],
        VOCAB_SIZE,
        LAYERS * PLE,
    )?;
    cpu::matmul_f64(
        &s.hidden[..h],
        model.weights.per_layer_projection,
        None,
        &mut s.ple_projection[..p],
        count,
        HIDDEN,
        LAYERS * PLE,
    )?;
    cpu::finish_per_layer_inputs(
        &s.ple_projection[..p],
        Some(&s.ple_lookup[..p]),
        model.weights.per_layer_norm,
        &mut s.ple_scaled[..p],
        &mut s.ple[..p],
        &PerLayerInputShape {
            tokens: count,
            hidden: HIDDEN,
            layers: LAYERS,
            width: PLE,
            epsilon: EPSILON,
        },
    )?;
    // Exact two-producer geometry keeps proposals on the stack. Borrow each
    // persistent staging owner separately; neither can be overwritten while a
    // later shared consumer or the all-owner commit still holds its frame.
    let [local, global] = &mut session.staging;
    let first = producer(0, model, &session.caches[0], local, s, count, offset)?;
    #[cfg(test)]
    if session.fault == Some(0) {
        return Err(Error::InvalidConfig("injected Gemma layer failure"));
    }
    let second = producer(1, model, &session.caches[1], global, s, count, offset)?;
    #[cfg(test)]
    if session.fault == Some(1) {
        return Err(Error::InvalidConfig("injected Gemma layer failure"));
    }
    let frames = [first, second];
    for index in 2..LAYERS {
        queries(index, model, s, count, offset)?;
        finish(
            index,
            model,
            s,
            model.plan.view(index, &session.caches, &frames)?,
            count,
            offset,
        )?;
        #[cfg(test)]
        if session.fault == Some(index) {
            return Err(Error::InvalidConfig("injected Gemma layer failure"));
        }
    }
    cpu::rmsnorm(
        &s.hidden[..h],
        model.weights.final_norm,
        &mut s.norm[..h],
        count,
        HIDDEN,
        EPSILON,
    )?;
    cpu::matmul_f64(
        &s.norm[..h],
        model.weights.embedding,
        None,
        &mut s.logits[..count * VOCAB_SIZE],
        count,
        HIDDEN,
        VOCAB_SIZE,
    )?;
    cpu::logit_softcap(&mut s.logits[..count * VOCAB_SIZE], 30.)?;
    #[cfg(test)]
    if session.fault == Some(LAYERS) {
        return Err(Error::InvalidConfig("injected Gemma final failure"));
    }
    // No fallible operation follows publication: every producer is validated
    // before any cache writes, then history and public logits advance together.
    model.plan.commit(&mut session.caches, &frames)?;
    session.history[offset..after].copy_from_slice(tokens);
    session.committed = after;
    session.published_rows = count;
    std::mem::swap(&mut s.logits, &mut s.published);
    Ok(())
}
