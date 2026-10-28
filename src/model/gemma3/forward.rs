use super::*;
use crate::cpu::{ProportionalRotaryShape, SpanAttentionShape};
use crate::model::window_cache::{PreparedKv, WindowKvCache};
use smallvec::SmallVec;

/// Diagnostic borrowing is operation-granular and statically eliminated from
/// ordinary forward. Callers own any retained copies and their separate budget.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) struct Observation<'a> {
    pub layer: Option<usize>,
    pub stage: &'static str,
    pub rows: usize,
    pub columns: usize,
    pub token_offset: usize,
    pub heads: usize,
    pub key_start: Option<usize>,
    pub values: &'a [f32],
}
fn emit<const TRACE: bool, F: for<'a> FnMut(Observation<'a>)>(
    observe: &mut F,
    item: Observation<'_>,
) {
    if TRACE {
        observe(item);
    }
}

struct Step<'a, F> {
    count: usize,
    offset: usize,
    observe: &'a mut F,
}
fn layer<
    'a,
    const TRACE: bool,
    M: MatrixSource,
    B: Operations<M>,
    F: for<'b> FnMut(Observation<'b>),
>(
    ops: &mut B,
    model: &Model<M>,
    index: usize,
    cache: &WindowKvCache,
    stage: &'a mut Staging,
    scratch: &mut Scratch,
    step: Step<'_, F>,
) -> Result<PreparedKv<'a>, B::Error> {
    let Step {
        count,
        offset,
        observe,
    } = step;
    let c = &model.config;
    let w = &model.layers[index];
    let s = scratch;
    let h = count * c.hidden_size;
    let q = count * c.query_heads * c.head_dim;
    let kv = count * c.head_dim;
    let ff = count * c.intermediate_size;
    let eps = c.rms_norm_eps;
    {
        let mut ordinary = |name, width, values: &[f32]| {
            emit::<TRACE, _>(
                observe,
                Observation {
                    layer: Some(index),
                    stage: name,
                    rows: count,
                    columns: width,
                    token_offset: offset,
                    heads: 1,
                    key_start: None,
                    values,
                },
            );
        };
        cpu::rmsnorm(
            &s.hidden[..h],
            &w.norms.input,
            &mut s.norm[..h],
            count,
            c.hidden_size,
            eps,
        )?;
        ordinary("input_norm", c.hidden_size, &s.norm[..h]);
        ops.project(&s.norm[..h], &w.q, &mut s.q[..q], count)?;
        ordinary("q_projection", c.query_heads * c.head_dim, &s.q[..q]);
        ops.project(&s.norm[..h], &w.k, &mut s.kv[..kv], count)?;
        ordinary("k_projection", c.head_dim, &s.kv[..kv]);
        ops.project(&s.norm[..h], &w.v, &mut stage.v[..kv], count)?;
        ordinary("v_projection", c.head_dim, &stage.v[..kv]);
        cpu::rmsnorm(
            &s.q[..q],
            &w.norms.q,
            &mut s.q_norm[..q],
            count * c.query_heads,
            c.head_dim,
            eps,
        )?;
        ordinary("q_norm", c.query_heads * c.head_dim, &s.q_norm[..q]);
        cpu::rmsnorm(
            &s.kv[..kv],
            &w.norms.k,
            &mut s.kv_norm[..kv],
            count,
            c.head_dim,
            eps,
        )?;
        ordinary("k_norm", c.head_dim, &s.kv_norm[..kv]);
        let rotary = |heads| ProportionalRotaryShape {
            tokens: count,
            heads,
            head_dim: c.head_dim,
            rotated_dims: c.head_dim,
            offset,
            base: w.spec.theta,
            factor: 1.0,
        };
        cpu::proportional_rotary(&s.q_norm[..q], &mut s.q[..q], &rotary(c.query_heads))?;
        ordinary("q_rotary", c.query_heads * c.head_dim, &s.q[..q]);
        cpu::proportional_rotary(&s.kv_norm[..kv], &mut stage.k[..kv], &rotary(1))?;
        ordinary("k_rotary", c.head_dim, &stage.k[..kv]);
    }
    let frame = cache.prepare(&stage.k[..kv], &stage.v[..kv], count)?;
    let view = cache.view(&frame)?;
    let keys = view.end() - view.start();
    let scores = c.query_heads * count * keys;
    cpu::span_attention(
        &s.q[..q],
        &view.segments(),
        &mut s.context[..q],
        &mut s.scores[..scores],
        &mut s.probabilities[..scores],
        &SpanAttentionShape {
            queries: count,
            keys,
            query_heads: c.query_heads,
            kv_heads: 1,
            head_dim: c.head_dim,
            query_start: offset,
            key_start: view.start(),
            window: match w.spec.kind {
                KvKind::Full => None,
                KvKind::Sliding => Some(model.window),
            },
            scale: (1.0 / (c.head_dim as f64).sqrt()) as f32,
        },
    )?;
    // The actual absolute-key interval is required for oracle slicing.
    for (name, values) in [
        ("attention_scores", &s.scores[..scores]),
        ("attention_probabilities", &s.probabilities[..scores]),
    ] {
        emit::<TRACE, _>(
            observe,
            Observation {
                layer: Some(index),
                stage: name,
                rows: count,
                columns: keys,
                token_offset: offset,
                heads: c.query_heads,
                key_start: Some(view.start()),
                values,
            },
        );
    }
    let mut ordinary = |name, width, values: &[f32]| {
        emit::<TRACE, _>(
            observe,
            Observation {
                layer: Some(index),
                stage: name,
                rows: count,
                columns: width,
                token_offset: offset,
                heads: 1,
                key_start: None,
                values,
            },
        );
    };
    ordinary(
        "attention_context",
        c.query_heads * c.head_dim,
        &s.context[..q],
    );
    ops.project(&s.context[..q], &w.o, &mut s.attention[..h], count)?;
    ordinary("attention_projection", c.hidden_size, &s.attention[..h]);
    cpu::rmsnorm(
        &s.attention[..h],
        &w.norms.post_attention,
        &mut s.next[..h],
        count,
        c.hidden_size,
        eps,
    )?;
    ordinary("post_attention_norm", c.hidden_size, &s.next[..h]);
    cpu::residual_add(&mut s.next[..h], &s.hidden[..h])?;
    ordinary("attention_residual", c.hidden_size, &s.next[..h]);
    cpu::rmsnorm(
        &s.next[..h],
        &w.norms.pre_ff,
        &mut s.norm[..h],
        count,
        c.hidden_size,
        eps,
    )?;
    ordinary("pre_ff_norm", c.hidden_size, &s.norm[..h]);
    ops.project(&s.norm[..h], &w.gate, &mut s.gate[..ff], count)?;
    ordinary("gate_projection", c.intermediate_size, &s.gate[..ff]);
    ops.project(&s.norm[..h], &w.up, &mut s.up[..ff], count)?;
    ordinary("up_projection", c.intermediate_size, &s.up[..ff]);
    if TRACE {
        cpu::gelu_tanh(&s.gate[..ff], &mut s.gated[..ff])?;
        ordinary("gate_gelu", c.intermediate_size, &s.gated[..ff]);
    }
    cpu::gated_gelu_tanh(&s.gate[..ff], &s.up[..ff], &mut s.gated[..ff])?;
    ordinary("gated", c.intermediate_size, &s.gated[..ff]);
    ops.project(&s.gated[..ff], &w.down, &mut s.attention[..h], count)?;
    ordinary("down_projection", c.hidden_size, &s.attention[..h]);
    cpu::rmsnorm(
        &s.attention[..h],
        &w.norms.post_ff,
        &mut s.residual[..h],
        count,
        c.hidden_size,
        eps,
    )?;
    ordinary("post_ff_norm", c.hidden_size, &s.residual[..h]);
    cpu::residual_add(&mut s.residual[..h], &s.next[..h])?;
    ordinary("output", c.hidden_size, &s.residual[..h]);
    std::mem::swap(&mut s.hidden, &mut s.residual);
    Ok(frame)
}

pub(super) fn run<M: MatrixSource, B: Operations<M>>(
    session: &mut Session<'_, M, B>,
    tokens: &[u32],
) -> Result<(), B::Error> {
    execute::<false, _, _, _>(session, tokens, &mut |_| {})
}
pub(super) fn execute<
    const TRACE: bool,
    M: MatrixSource,
    B: Operations<M>,
    F: for<'a> FnMut(Observation<'a>),
>(
    session: &mut Session<'_, M, B>,
    tokens: &[u32],
    observe: &mut F,
) -> Result<(), B::Error> {
    cpu::nonzero(tokens.len(), "Gemma3 new chunk")?;
    let m = session.model;
    let c = &m.config;
    let count = tokens.len();
    let offset = session.committed;
    if count > session.chunk {
        return Err(Error::ScratchCapacity {
            tokens: count,
            limit: session.chunk,
        }
        .into());
    }
    let after = offset.checked_add(count).ok_or(cpu::Error::SizeOverflow)?;
    if after > c.max_positions {
        return Err(Error::Capacity {
            offset,
            tokens: count,
            limit: c.max_positions,
        }
        .into());
    }
    for (position, &token) in tokens.iter().enumerate() {
        if token as usize >= c.vocab_size {
            return Err(cpu::Error::Token {
                position,
                token,
                vocab_size: c.vocab_size,
            }
            .into());
        }
    }
    if session.caches.len() != m.layers.len()
        || session.staging.len() != m.layers.len()
        || m.layers.len() > MAX_LAYERS
        || session
            .caches
            .iter()
            .any(|v| v.offset() != offset || v.generation() != session.caches[0].generation())
    {
        return Err(Error::InvalidConfig("Gemma3 committed cache/staging prefix").into());
    }
    session.operations.ready()?;
    let s = &mut session.scratch;
    let h = count * c.hidden_size;
    session
        .operations
        .gather(&m.embedding, tokens, &mut s.next[..h])?;
    emit::<TRACE, _>(
        observe,
        Observation {
            layer: None,
            stage: "embedding_lookup",
            rows: count,
            columns: c.hidden_size,
            token_offset: offset,
            heads: 1,
            key_start: None,
            values: &s.next[..h],
        },
    );
    for (out, &input) in s.hidden[..h].iter_mut().zip(&s.next[..h]) {
        *out = input * m.embedding_scale;
    }
    finite("Gemma3 scaled embedding", &s.hidden[..h])?;
    emit::<TRACE, _>(
        observe,
        Observation {
            layer: None,
            stage: "embedding",
            rows: count,
            columns: c.hidden_size,
            token_offset: offset,
            heads: 1,
            key_start: None,
            values: &s.hidden[..h],
        },
    );
    let mut frames: SmallVec<[PreparedKv<'_>; MAX_LAYERS]> = SmallVec::new();
    for (index, stage) in session.staging.iter_mut().enumerate() {
        frames.push(layer::<TRACE, _, _, _>(
            &mut session.operations,
            m,
            index,
            &session.caches[index],
            stage,
            s,
            Step {
                count,
                offset,
                observe,
            },
        )?);
        #[cfg(test)]
        if session.fault == Some(index) {
            return Err(Error::InvalidConfig("injected Gemma3 layer failure").into());
        }
    }
    debug_assert!(!frames.spilled());
    cpu::rmsnorm(
        &s.hidden[..h],
        &m.final_norm,
        &mut s.norm[..h],
        count,
        c.hidden_size,
        c.rms_norm_eps,
    )?;
    emit::<TRACE, _>(
        observe,
        Observation {
            layer: None,
            stage: "final_norm",
            rows: count,
            columns: c.hidden_size,
            token_offset: offset,
            heads: 1,
            key_start: None,
            values: &s.norm[..h],
        },
    );
    #[cfg(test)]
    if session.fault == Some(m.layers.len()) {
        return Err(Error::InvalidConfig("injected Gemma3 final norm failure").into());
    }
    session.operations.project(
        &s.norm[..h],
        &m.embedding,
        &mut s.logits[..count * c.vocab_size],
        count,
    )?;
    finite("Gemma3 final logits", &s.logits[..count * c.vocab_size])?;
    emit::<TRACE, _>(
        observe,
        Observation {
            layer: None,
            stage: "logits",
            rows: count,
            columns: c.vocab_size,
            token_offset: offset,
            heads: 1,
            key_start: None,
            values: &s.logits[..count * c.vocab_size],
        },
    );
    session.operations.ready()?;
    #[cfg(test)]
    if session.fault == Some(m.layers.len() + 1) {
        return Err(Error::InvalidConfig("injected Gemma3 head failure").into());
    }
    m.plan.commit(&mut session.caches, &frames)?;
    session.history[offset..after].copy_from_slice(tokens);
    s.published[..count * c.vocab_size].copy_from_slice(&s.logits[..count * c.vocab_size]);
    session.committed = after;
    session.published_rows = count;
    Ok(())
}
