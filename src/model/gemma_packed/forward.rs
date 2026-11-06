use super::*;
use crate::cpu::{PerLayerInputShape, ProportionalRotaryShape, SpanAttentionShape};
use crate::model::window_cache::{KvView, PreparedKv};
use smallvec::SmallVec;

fn queries<M: MatrixSource, B: Operations<M>>(
    ops: &mut B,
    index: usize,
    model: &Model<'_, M>,
    s: &mut Scratch,
    count: usize,
    offset: usize,
) -> Result<(), B::Error> {
    let layer = &model.weights.layers[index];
    let l = &model.layout;
    let d = l.layers[index].head_dim;
    let q = count * l.heads * d;
    cpu::rmsnorm(
        &s.hidden[..count * l.hidden],
        layer.input_norm,
        &mut s.norm[..count * l.hidden],
        count,
        l.hidden,
        EPSILON,
    )?;
    ops.project(&s.norm[..count * l.hidden], &layer.q, &mut s.q[..q], count)?;
    cpu::rmsnorm(
        &s.q[..q],
        layer.q_norm,
        &mut s.q_norm[..q],
        count * l.heads,
        d,
        EPSILON,
    )?;
    cpu::proportional_rotary(
        &s.q_norm[..q],
        &mut s.q[..q],
        &rotary(&l.layers[index], count, l.heads, offset),
    )?;
    Ok(())
}

fn rotary(g: &LayerLayout, tokens: usize, heads: usize, offset: usize) -> ProportionalRotaryShape {
    ProportionalRotaryShape {
        tokens,
        heads,
        head_dim: g.head_dim,
        rotated_dims: g.rotated_dims,
        offset,
        base: g.theta,
        factor: 1.,
    }
}

fn finish<M: MatrixSource, B: Operations<M>>(
    ops: &mut B,
    index: usize,
    model: &Model<'_, M>,
    s: &mut Scratch,
    view: KvView<'_, '_>,
    count: usize,
    offset: usize,
) -> Result<(), B::Error> {
    let l = &model.layout;
    let d = l.layers[index].head_dim;
    let q = count * l.heads * d;
    let keys = view.end() - view.start();
    let scores = l.heads * count * keys;
    cpu::span_attention(
        &s.q[..q],
        &view.segments(),
        &mut s.context[..q],
        &mut s.scores[..scores],
        &mut s.probabilities[..scores],
        &SpanAttentionShape {
            queries: count,
            keys,
            query_heads: l.heads,
            kv_heads: 1,
            head_dim: d,
            query_start: offset,
            key_start: view.start(),
            window: if matches!(l.layers[index].kind, KvKind::Sliding) {
                Some(l.window)
            } else {
                None
            },
            scale: 1.,
        },
    )?;
    ops.project(
        &s.context[..q],
        &model.weights.layers[index].o,
        &mut s.attention[..count * l.hidden],
        count,
    )?;
    for token in 0..count {
        let start = (token * l.layers.len() + index) * l.ple;
        s.layer_ple[token * l.ple..(token + 1) * l.ple]
            .copy_from_slice(&s.ple[start..start + l.ple]);
    }
    let w = &model.weights.layers[index];
    let ff = count * l.layers[index].intermediate;
    let h = count * l.hidden;
    let p = count * l.ple;
    // Keep attention and the post-FF residual live while each branch reuses
    // private work buffers. No whole packed matrix is expanded or retained.
    cpu::rmsnorm(
        &s.attention[..h],
        w.post_attention_norm,
        &mut s.dense_work[..h],
        count,
        l.hidden,
        EPSILON,
    )?;
    cpu::residual_add(&mut s.dense_work[..h], &s.hidden[..h])?;
    cpu::rmsnorm(
        &s.dense_work[..h],
        w.pre_ff_norm,
        &mut s.dense_norm[..h],
        count,
        l.hidden,
        EPSILON,
    )?;
    ops.project(&s.dense_norm[..h], &w.ff_gate, &mut s.gate[..ff], count)?;
    ops.project(&s.dense_norm[..h], &w.ff_up, &mut s.up[..ff], count)?;
    cpu::gated_gelu_tanh(&s.gate[..ff], &s.up[..ff], &mut s.product[..ff])?;
    ops.project(&s.product[..ff], &w.ff_down, &mut s.dense_norm[..h], count)?;
    cpu::rmsnorm(
        &s.dense_norm[..h],
        w.post_ff_norm,
        &mut s.next[..h],
        count,
        l.hidden,
        EPSILON,
    )?;
    cpu::residual_add(&mut s.next[..h], &s.dense_work[..h])?;
    ops.project(&s.next[..h], &w.ple_gate, &mut s.gate[..p], count)?;
    cpu::gated_gelu_tanh(&s.gate[..p], &s.layer_ple[..p], &mut s.product[..p])?;
    ops.project(
        &s.product[..p],
        &w.ple_projection,
        &mut s.dense_work[..h],
        count,
    )?;
    cpu::rmsnorm(
        &s.dense_work[..h],
        w.post_ple_norm,
        &mut s.dense_norm[..h],
        count,
        l.hidden,
        EPSILON,
    )?;
    cpu::residual_add(&mut s.next[..h], &s.dense_norm[..h])?;
    for (index, value) in s.next[..h].iter_mut().enumerate() {
        *value *= w.scalar;
        if !value.is_finite() {
            return Err(cpu::Error::NonFinite {
                tensor: "Gemma packed scalar output",
                index,
            }
            .into());
        }
    }
    std::mem::swap(&mut s.hidden, &mut s.next);
    Ok(())
}

fn producer<'a, M: MatrixSource, B: Operations<M>>(
    ops: &mut B,
    index: usize,
    model: &Model<'_, M>,
    cache: &WindowKvCache,
    stage: &'a mut Staging,
    s: &mut Scratch,
    span: (usize, usize),
) -> Result<PreparedKv<'a>, B::Error> {
    let (count, offset) = span;
    queries(ops, index, model, s, count, offset)?;
    let l = &model.layout;
    let d = l.layers[index].head_dim;
    let kv = count * d;
    let weights = model.weights.layers[index]
        .kv
        .as_ref()
        .ok_or(Error::InvalidConfig("Gemma producer projection"))?;
    ops.project(
        &s.norm[..count * l.hidden],
        &weights.k,
        &mut s.kv[..kv],
        count,
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
        &rotary(&l.layers[index], count, 1, offset),
    )?;
    ops.project(
        &s.norm[..count * l.hidden],
        &weights.v,
        &mut s.kv[..kv],
        count,
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
    finish(ops, index, model, s, cache.view(&frame)?, count, offset)?;
    Ok(frame)
}

#[derive(Clone, Copy)]
pub(super) enum OutputRows {
    All,
    #[cfg(test)]
    Last,
}

pub(super) fn run<M: MatrixSource, B: Operations<M>>(
    session: &mut Session<'_, '_, M, B>,
    tokens: &[u32],
    output: OutputRows,
) -> Result<(), B::Error> {
    cpu::nonzero(tokens.len(), "Gemma chunk")?;
    if tokens.len() > session.chunk_rows {
        return Err(Error::InvalidConfig("Gemma chunk exceeds scratch capacity").into());
    }
    let l = &session.model.layout;
    let count = tokens.len();
    let offset = session.committed;
    let after = offset.checked_add(count).ok_or(cpu::Error::SizeOverflow)?;
    if after > session.capacity {
        return Err(Error::Capacity {
            offset,
            tokens: count,
            limit: session.capacity,
        }
        .into());
    }
    for (position, &token) in tokens.iter().enumerate() {
        if token as usize >= l.vocab {
            return Err(cpu::Error::Token {
                position,
                token,
                vocab_size: l.vocab,
            }
            .into());
        }
    }
    if session.caches.iter().any(|cache| {
        cache.offset() != offset || cache.generation() != session.caches[0].generation()
    }) {
        return Err(Error::InvalidConfig("Gemma committed cache prefix").into());
    }
    if session.staging.len() != l.producers
        || session.caches.len() != l.producers
        || l.producers > MAX_PRODUCERS
    {
        return Err(Error::InvalidConfig("Gemma packed proposal frame bound").into());
    }
    let ops = &mut session.operations;
    ops.ready()?;
    let model = session.model;
    let s = &mut session.scratch;
    let h = count * l.hidden;
    let p = count * l.layers.len() * l.ple;
    ops.gather(&model.weights.embedding, tokens, &mut s.next[..h])?;
    cpu::prepare_per_layer_projection_input(&s.next[..h], &mut s.hidden[..h], count, l.hidden)?;
    ops.gather(
        &model.weights.per_layer_embedding,
        tokens,
        &mut s.ple_lookup[..p],
    )?;
    ops.project(
        &s.hidden[..h],
        &model.weights.per_layer_projection,
        &mut s.ple_projection[..p],
        count,
    )?;
    cpu::finish_per_layer_inputs(
        &s.ple_projection[..p],
        Some(&s.ple_lookup[..p]),
        model.weights.per_layer_norm,
        &mut s.ple_scaled[..p],
        &mut s.ple[..p],
        &PerLayerInputShape {
            tokens: count,
            hidden: l.hidden,
            layers: l.layers.len(),
            width: l.ple,
            epsilon: EPSILON,
        },
    )?;
    // Borrow each persistent stage owner once. Every producer proposal remains
    // live until all shared consumers and the final head succeed. Admission caps
    // this prefix at the inline capacity, so this container never spills.
    let mut frames: SmallVec<[PreparedKv<'_>; MAX_PRODUCERS]> = SmallVec::new();
    for (index, stage) in session.staging.iter_mut().enumerate() {
        frames.push(producer(
            ops,
            index,
            model,
            &session.caches[index],
            stage,
            s,
            (count, offset),
        )?);
        #[cfg(test)]
        if session.fault == Some(index) {
            return Err(Error::InvalidConfig("injected Gemma layer failure").into());
        }
    }
    debug_assert!(!frames.spilled());
    let (tail_count, tail_offset) = match output {
        OutputRows::All => (count, offset),
        #[cfg(test)]
        OutputRows::Last => {
            // Producers staged every token. Only the final hidden and its full
            // token-major PLE row advance through the consumer-only tail.
            if count > 1 {
                s.hidden.copy_within(h - l.hidden..h, 0);
                s.ple.copy_within(p - l.layers.len() * l.ple..p, 0);
            }
            (1, after - 1)
        }
    };
    let h = tail_count * l.hidden;
    for index in l.producers..l.layers.len() {
        queries(ops, index, model, s, tail_count, tail_offset)?;
        finish(
            ops,
            index,
            model,
            s,
            session.plan.view(index, &session.caches, &frames)?,
            tail_count,
            tail_offset,
        )?;
        #[cfg(test)]
        if session.fault == Some(index) {
            return Err(Error::InvalidConfig("injected Gemma layer failure").into());
        }
    }
    cpu::rmsnorm(
        &s.hidden[..h],
        model.weights.final_norm,
        &mut s.norm[..h],
        tail_count,
        l.hidden,
        EPSILON,
    )?;
    #[cfg(test)]
    if session.fault == Some(l.layers.len()) {
        return Err(Error::InvalidConfig("injected Gemma final norm failure").into());
    }
    ops.project(
        &s.norm[..h],
        &model.weights.embedding,
        &mut s.logits[..tail_count * l.vocab],
        tail_count,
    )?;
    #[cfg(test)]
    if session.fault == Some(l.layers.len() + 1) {
        return Err(Error::InvalidConfig("injected Gemma head failure").into());
    }
    cpu::logit_softcap(&mut s.logits[..tail_count * l.vocab], 30.)?;
    #[cfg(test)]
    if session.fault == Some(l.layers.len() + 2) {
        return Err(Error::InvalidConfig("injected Gemma final failure").into());
    }
    // No fallible operation follows publication: every producer is validated
    // before any cache writes, then history and public logits advance together.
    session.plan.commit(&mut session.caches, &frames)?;
    session.history[offset..after].copy_from_slice(tokens);
    session.committed = after;
    session.published_rows = tail_count;
    std::mem::swap(&mut s.logits, &mut s.published);
    Ok(())
}
