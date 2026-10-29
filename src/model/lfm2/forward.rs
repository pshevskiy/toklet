use super::*;
use crate::cpu::SpanAttentionShape;
use crate::model::{short_conv3::Prepared, window_cache::PreparedKv};
use smallvec::SmallVec;

/// Borrowed diagnostic slices only. Fused B/C/x rows retain stride 3H; scores
/// retain [head, query, actual key] and absolute key_start. State is observed by
/// the caller before/after the actual publication, never fabricated per row.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) struct Observation<'a> {
    pub layer: Option<usize>,
    pub stage: &'static str,
    pub rows: usize,
    pub columns: usize,
    pub row_stride: usize,
    pub token_offset: usize,
    pub heads: usize,
    pub key_start: Option<usize>,
    pub values: &'a [f32],
}
struct Trace<'a, F> {
    observe: &'a mut F,
    count: usize,
    offset: usize,
    layer: Option<usize>,
}
impl<F: for<'b> FnMut(Observation<'b>) -> Result<(), Error>> Trace<'_, F> {
    fn ordinary<const TRACE: bool>(
        &mut self,
        name: &'static str,
        width: usize,
        values: &[f32],
    ) -> Result<(), Error> {
        self.strided::<TRACE>(name, width, width, values)
    }
    fn strided<const TRACE: bool>(
        &mut self,
        name: &'static str,
        width: usize,
        stride: usize,
        values: &[f32],
    ) -> Result<(), Error> {
        if TRACE {
            (self.observe)(Observation {
                layer: self.layer,
                stage: name,
                rows: self.count,
                columns: width,
                row_stride: stride,
                token_offset: self.offset,
                heads: 1,
                key_start: None,
                values,
            })?;
        }
        Ok(())
    }
    fn attention<const TRACE: bool>(
        &mut self,
        name: &'static str,
        heads: usize,
        start: usize,
        keys: usize,
        values: &[f32],
    ) -> Result<(), Error> {
        if TRACE {
            (self.observe)(Observation {
                layer: self.layer,
                stage: name,
                rows: self.count,
                columns: keys,
                row_stride: keys,
                token_offset: self.offset,
                heads,
                key_start: Some(start),
                values,
            })?;
        }
        Ok(())
    }
}

/// All destinations are disjoint session fields. All validation finishes before
/// the first physical owner copy; no public fallible per-family commits chained.
pub(super) struct Publication<'a> {
    pub caches: &'a mut [WindowKvCache],
    pub histories: &'a mut [History],
    pub tokens: &'a mut [u32],
    pub published: &'a mut [f32],
    pub committed: &'a mut usize,
    pub generation: &'a mut u64,
    pub rows: &'a mut usize,
    pub capacity: usize,
    pub chunk: usize,
    pub vocab: usize,
}
impl Publication<'_> {
    pub(super) fn commit(
        &mut self,
        kv: &[PreparedKv<'_>],
        history: &[Prepared<'_>],
        tokens: &[u32],
        logits: &[f32],
    ) -> Result<(), Error> {
        let count = tokens.len();
        let offset = *self.committed;
        cpu::nonzero(count, "LFM2 publication chunk")?;
        if count > self.chunk {
            return Err(Error::InvalidConfig("LFM2 publication chunk capacity"));
        }
        let after = offset.checked_add(count).ok_or(cpu::Error::SizeOverflow)?;
        if after > self.capacity {
            return Err(Error::Capacity {
                offset,
                tokens: count,
                limit: self.capacity,
            });
        }
        let generation = self
            .generation
            .checked_add(1)
            .ok_or(cpu::Error::SizeOverflow)?;
        for (name, actual, expected) in [
            ("LFM2 KV owners", self.caches.len(), KV_OWNERS),
            ("LFM2 KV frames", kv.len(), KV_OWNERS),
            ("LFM2 history owners", self.histories.len(), CONV_OWNERS),
            ("LFM2 history frames", history.len(), CONV_OWNERS),
            ("LFM2 token capacity", self.tokens.len(), self.capacity),
            (
                "LFM2 published capacity",
                self.published.len(),
                product(self.chunk, self.vocab)?,
            ),
            ("LFM2 logits", logits.len(), product(count, self.vocab)?),
        ] {
            length(name, actual, expected)?;
        }
        for (position, &token) in tokens.iter().enumerate() {
            if token as usize >= self.vocab {
                return Err(cpu::Error::Token {
                    position,
                    token,
                    vocab_size: self.vocab,
                }
                .into());
            }
        }
        finite("LFM2 candidate logits", logits)?;
        for (owner, frame) in self.caches.iter().zip(kv) {
            owner.validate_step(frame, offset, count, *self.generation)?;
        }
        for (owner, frame) in self.histories.iter().zip(history) {
            owner.validate_step(frame, offset, count, *self.generation)?;
        }
        let token_dest = &mut self.tokens[offset..after];
        let logit_dest = &mut self.published[..logits.len()];
        // Infallible publication tail. Frames authenticate actual owner IDs.
        for (owner, frame) in self.caches.iter_mut().zip(kv) {
            owner.commit_validated(frame);
        }
        for (owner, frame) in self.histories.iter_mut().zip(history) {
            owner.commit_validated(frame);
        }
        token_dest.copy_from_slice(tokens);
        logit_dest.copy_from_slice(logits);
        *self.committed = after;
        *self.generation = generation;
        *self.rows = count;
        Ok(())
    }
}

pub(super) fn execute<const TRACE: bool, F: for<'a> FnMut(Observation<'a>) -> Result<(), Error>>(
    session: &mut Session<'_, '_>,
    tokens: &[u32],
    observe: &mut F,
) -> Result<(), Error> {
    let count = tokens.len();
    let offset = session.committed;
    cpu::nonzero(count, "LFM2 chunk")?;
    if count > session.chunk {
        return Err(Error::InvalidConfig("LFM2 scalar chunk capacity"));
    }
    let after = offset.checked_add(count).ok_or(cpu::Error::SizeOverflow)?;
    let c = &session.model.config;
    if after > c.max_positions {
        return Err(Error::Capacity {
            offset,
            tokens: count,
            limit: c.max_positions,
        });
    }
    session
        .generation
        .checked_add(1)
        .ok_or(cpu::Error::SizeOverflow)?;
    session.check_inventory()?;
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
    let model = session.model;
    let ops = &mut session.operations;
    let s = &mut session.scratch;
    let h = count * c.hidden_size;
    let q = count * c.query_heads * c.head_dim;
    let kv = count * c.kv_heads * c.head_dim;
    let ff = count * c.intermediate_size;
    let eps = c.rms_norm_eps;
    let mut trace = Trace {
        observe,
        count,
        offset,
        layer: None,
    };
    ops.gather(&model.embedding, tokens, &mut s.hidden[..h])?;
    trace.ordinary::<TRACE>("embedding_lookup", c.hidden_size, &s.hidden[..h])?;
    let mut kv_frames: SmallVec<[PreparedKv<'_>; KV_OWNERS]> = SmallVec::new();
    let mut history_frames: SmallVec<[Prepared<'_>; CONV_OWNERS]> = SmallVec::new();
    // Cold-validated routes consume each disjoint staging owner exactly once.
    let mut kv_stages = session.kv_stage.iter_mut();
    let mut history_stages = session.conv_stage.iter_mut();
    for (index, layer) in model.layers.iter().enumerate() {
        trace.layer = Some(index);
        cpu::rmsnorm(
            &s.hidden[..h],
            &layer.operator_norm,
            &mut s.norm[..h],
            count,
            c.hidden_size,
            eps,
        )?;
        trace.ordinary::<TRACE>("operator_norm", c.hidden_size, &s.norm[..h])?;
        match layer.kind {
            Kind::Convolution(owner) => {
                let w = &model.convolution[owner];
                ops.project(&s.norm[..h], &w.bcx, &mut s.bcx[..3 * h], count)?;
                trace.ordinary::<TRACE>("bcx_projection", 3 * c.hidden_size, &s.bcx[..3 * h])?;
                let strided = (count - 1) * 3 * c.hidden_size + c.hidden_size;
                for (part, name) in ["b", "c", "x"].into_iter().enumerate() {
                    let start = part * c.hidden_size;
                    trace.strided::<TRACE>(
                        name,
                        c.hidden_size,
                        3 * c.hidden_size,
                        &s.bcx[start..start + strided],
                    )?;
                }
                for row in 0..count {
                    for col in 0..c.hidden_size {
                        s.u[row * c.hidden_size + col] = s.bcx[row * 3 * c.hidden_size + col]
                            * s.bcx[row * 3 * c.hidden_size + 2 * c.hidden_size + col];
                    }
                }
                finite("LFM2 raw B*x", &s.u[..h])?;
                trace.ordinary::<TRACE>("u", c.hidden_size, &s.u[..h])?;
                let stage = history_stages
                    .next()
                    .ok_or(Error::InvalidConfig("LFM2 history staging"))?;
                let frame = session.histories[owner].prepare_chunk(
                    &s.u[..h],
                    &w.coefficients,
                    stage,
                    &mut s.work,
                    &mut s.conv[..h],
                    count,
                )?;
                trace.ordinary::<TRACE>("convolution", c.hidden_size, &s.conv[..h])?;
                for row in 0..count {
                    for col in 0..c.hidden_size {
                        s.context[row * c.hidden_size + col] = s.bcx
                            [row * 3 * c.hidden_size + c.hidden_size + col]
                            * s.conv[row * c.hidden_size + col];
                    }
                }
                finite("LFM2 C*z", &s.context[..h])?;
                trace.ordinary::<TRACE>("gated_conv", c.hidden_size, &s.context[..h])?;
                ops.project(&s.context[..h], &w.output, &mut s.next[..h], count)?;
                history_frames.push(frame);
            }
            Kind::Attention(owner) => {
                let w = &model.attention[owner];
                ops.project(&s.norm[..h], &w.q, &mut s.q[..q], count)?;
                ops.project(&s.norm[..h], &w.k, &mut s.k[..kv], count)?;
                ops.project(&s.norm[..h], &w.v, &mut s.v[..kv], count)?;
                trace.ordinary::<TRACE>("q_projection", c.query_heads * c.head_dim, &s.q[..q])?;
                trace.ordinary::<TRACE>("k_projection", c.kv_heads * c.head_dim, &s.k[..kv])?;
                trace.ordinary::<TRACE>("v_projection", c.kv_heads * c.head_dim, &s.v[..kv])?;
                cpu::rmsnorm(
                    &s.q[..q],
                    &w.q_norm,
                    &mut s.qn[..q],
                    count * c.query_heads,
                    c.head_dim,
                    eps,
                )?;
                cpu::rmsnorm(
                    &s.k[..kv],
                    &w.k_norm,
                    &mut s.kn[..kv],
                    count * c.kv_heads,
                    c.head_dim,
                    eps,
                )?;
                trace.ordinary::<TRACE>("q_norm", c.query_heads * c.head_dim, &s.qn[..q])?;
                trace.ordinary::<TRACE>("k_norm", c.kv_heads * c.head_dim, &s.kn[..kv])?;
                cpu::rotary(
                    &s.qn[..q],
                    &mut s.q[..q],
                    count,
                    c.query_heads,
                    c.head_dim,
                    offset,
                    c.rope_theta,
                )?;
                cpu::rotary(
                    &s.kn[..kv],
                    &mut s.k[..kv],
                    count,
                    c.kv_heads,
                    c.head_dim,
                    offset,
                    c.rope_theta,
                )?;
                trace.ordinary::<TRACE>("q_rotary", c.query_heads * c.head_dim, &s.q[..q])?;
                trace.ordinary::<TRACE>("k_rotary", c.kv_heads * c.head_dim, &s.k[..kv])?;
                let stage = kv_stages
                    .next()
                    .ok_or(Error::InvalidConfig("LFM2 KV staging"))?;
                for token in 0..count {
                    for head in 0..c.kv_heads {
                        let source = (token * c.kv_heads + head) * c.head_dim;
                        let dest = (head * count + token) * c.head_dim;
                        stage.k[dest..dest + c.head_dim]
                            .copy_from_slice(&s.k[source..source + c.head_dim]);
                        stage.v[dest..dest + c.head_dim]
                            .copy_from_slice(&s.v[source..source + c.head_dim]);
                    }
                }
                let frame = session.caches[owner].prepare(&stage.k[..kv], &stage.v[..kv], count)?;
                let view = session.caches[owner].view(&frame)?;
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
                        kv_heads: c.kv_heads,
                        head_dim: c.head_dim,
                        query_start: offset,
                        key_start: view.start(),
                        window: None,
                        scale: 1.0 / (c.head_dim as f32).sqrt(),
                    },
                )?;
                trace.attention::<TRACE>(
                    "attention_scores",
                    c.query_heads,
                    view.start(),
                    keys,
                    &s.scores[..scores],
                )?;
                trace.attention::<TRACE>(
                    "attention_probabilities",
                    c.query_heads,
                    view.start(),
                    keys,
                    &s.probabilities[..scores],
                )?;
                trace.ordinary::<TRACE>(
                    "attention_context",
                    c.query_heads * c.head_dim,
                    &s.context[..q],
                )?;
                ops.project(&s.context[..q], &w.o, &mut s.next[..h], count)?;
                kv_frames.push(frame);
            }
        }
        trace.ordinary::<TRACE>("operator_projection", c.hidden_size, &s.next[..h])?;
        cpu::residual_add(&mut s.next[..h], &s.hidden[..h])?;
        trace.ordinary::<TRACE>("operator_residual", c.hidden_size, &s.next[..h])?;
        cpu::rmsnorm(
            &s.next[..h],
            &layer.ffn_norm,
            &mut s.norm[..h],
            count,
            c.hidden_size,
            eps,
        )?;
        trace.ordinary::<TRACE>("ffn_norm", c.hidden_size, &s.norm[..h])?;
        ops.project(&s.norm[..h], &layer.gate, &mut s.gate[..ff], count)?;
        ops.project(&s.norm[..h], &layer.up, &mut s.up[..ff], count)?;
        trace.ordinary::<TRACE>("gate_projection", c.intermediate_size, &s.gate[..ff])?;
        trace.ordinary::<TRACE>("up_projection", c.intermediate_size, &s.up[..ff])?;
        if TRACE {
            cpu::silu(&s.gate[..ff], &mut s.gated[..ff])?;
            trace.ordinary::<TRACE>("gate_silu", c.intermediate_size, &s.gated[..ff])?;
        }
        cpu::gated_silu(&s.gate[..ff], &s.up[..ff], &mut s.gated[..ff])?;
        trace.ordinary::<TRACE>("ffn_gated", c.intermediate_size, &s.gated[..ff])?;
        ops.project(&s.gated[..ff], &layer.down, &mut s.projected[..h], count)?;
        trace.ordinary::<TRACE>("down_projection", c.hidden_size, &s.projected[..h])?;
        cpu::residual_add(&mut s.projected[..h], &s.next[..h])?;
        trace.ordinary::<TRACE>("output", c.hidden_size, &s.projected[..h])?;
        std::mem::swap(&mut s.hidden, &mut s.projected);
        #[cfg(test)]
        if session.fault == Some(index) {
            return Err(Error::InvalidConfig("injected LFM2 layer failure"));
        }
    }
    trace.layer = None;
    cpu::rmsnorm(
        &s.hidden[..h],
        &model.final_norm,
        &mut s.norm[..h],
        count,
        c.hidden_size,
        eps,
    )?;
    trace.ordinary::<TRACE>("final_norm", c.hidden_size, &s.norm[..h])?;
    #[cfg(test)]
    if session.fault == Some(LAYERS) {
        return Err(Error::InvalidConfig("injected LFM2 final failure"));
    }
    ops.project(
        &s.norm[..h],
        &model.embedding,
        &mut s.logits[..count * c.vocab_size],
        count,
    )?;
    finite("LFM2 logits", &s.logits[..count * c.vocab_size])?;
    trace.ordinary::<TRACE>("logits", c.vocab_size, &s.logits[..count * c.vocab_size])?;
    <Scalar as Operations<Matrix<'_>>>::ready(ops)?;
    #[cfg(test)]
    if session.fault == Some(LAYERS + 1) {
        return Err(Error::InvalidConfig("injected LFM2 head failure"));
    }
    Publication {
        caches: &mut session.caches,
        histories: &mut session.histories,
        tokens: &mut session.tokens,
        published: &mut s.published,
        committed: &mut session.committed,
        generation: &mut session.generation,
        rows: &mut session.published_rows,
        capacity: c.max_positions,
        chunk: session.chunk,
        vocab: c.vocab_size,
    }
    .commit(
        &kv_frames,
        &history_frames,
        tokens,
        &s.logits[..count * c.vocab_size],
    )
}
