use half::bf16;
use smallvec::SmallVec;

use super::{
    CONV_OWNERS, Error, HEAD_DIM, HIDDEN, INTERMEDIATE, KV_HEADS, KV_OWNERS, KV_WIDTH, Observation,
    QUERY_HEADS, Session, VOCAB,
};
use crate::{
    cpu::{
        self, LFM3_ATTENTION_SCALE, lfm3_add_bf16 as add, lfm3_attention_bf16 as attention,
        lfm3_head_gate_bf16 as head_gate, lfm3_linear_bf16 as linear, lfm3_mul_bf16 as mul,
        lfm3_rmsnorm_bf16 as norm, lfm3_sigmoid_bf16 as sigmoid, lfm3_silu_bf16 as silu,
    },
    model::{
        lfm3_state::ATTENTION_WINDOWS, short_conv7_bf16::Prepared, window_cache_bf16::PreparedKv,
    },
    weights::lfm3::Operator,
};

fn head_major(input: &[bf16], output: &mut [bf16], tokens: usize) {
    for (head, destination) in output.chunks_exact_mut(tokens * HEAD_DIM).enumerate() {
        let start = head * HEAD_DIM;
        for (row, out) in input
            .chunks_exact(KV_WIDTH)
            .zip(destination.chunks_exact_mut(HEAD_DIM))
        {
            out.copy_from_slice(&row[start..start + HEAD_DIM]);
        }
    }
}

impl Session<'_> {
    pub(super) fn step<const TRACE: bool, F>(
        &mut self,
        tokens: &[u32],
        mut observer: F,
    ) -> Result<&[bf16], Error>
    where
        F: FnMut(Observation<'_>) -> Result<(), Error>,
    {
        let count = tokens.len();
        let offset = self.state.offset();
        let end = self.state.admit(count)?;
        if count > self.model.chunk {
            return Err(Error::ScratchCapacity {
                tokens: count,
                limit: self.model.chunk,
            });
        }
        for (position, &token) in tokens.iter().enumerate() {
            if token as usize >= VOCAB {
                return Err(cpu::Error::Token {
                    position: offset + position,
                    token,
                    vocab_size: VOCAB,
                }
                .into());
            }
        }
        let mut report = |layer, stage, values: &[bf16]| {
            if TRACE {
                observer(Observation {
                    layer,
                    stage,
                    values,
                })?;
            }
            Ok::<_, Error>(())
        };
        let width = count * HIDDEN;
        let kv_words = count * KV_WIDTH;
        let weights = &self.model.weights;
        let embedding = weights.get(weights.embedding);
        let [current, normalized, output, post, residual, query] =
            self.scratch.hidden.each_mut().map(|v| &mut v[..width]);
        let [gate, activated, up] = self
            .scratch
            .ffn
            .each_mut()
            .map(|v| &mut v[..count * INTERMEDIATE]);
        let [kv, kv_norm] = self.scratch.kv.each_mut().map(|v| &mut v[..kv_words]);
        let [gate_raw, gate_sigmoid] = self
            .scratch
            .gates
            .each_mut()
            .map(|v| &mut v[..count * QUERY_HEADS]);
        let bcx = &mut self.scratch.bcx[..count * 3 * HIDDEN];
        let mut histories = self.scratch.histories.iter_mut();
        let mut keys = self.scratch.keys.iter_mut();
        let mut values = self.scratch.values.iter_mut();
        let mut conv_frames = SmallVec::<[Prepared<'_>; CONV_OWNERS]>::new();
        let mut kv_frames = SmallVec::<[PreparedKv<'_>; KV_OWNERS]>::new();

        for (&token, row) in tokens.iter().zip(current.chunks_exact_mut(HIDDEN)) {
            let first = token as usize * HIDDEN;
            row.copy_from_slice(&embedding[first..first + HIDDEN]);
        }
        for (index, layer) in weights.layers.iter().enumerate() {
            let layer_index = Some(index);
            norm(
                current,
                weights.get(layer.operator_norm),
                normalized,
                count,
                HIDDEN,
            )?;
            match layer.operator {
                Operator::Convolution {
                    bcx: project,
                    coefficients,
                    output: project_out,
                } => {
                    linear(
                        normalized,
                        weights.get(project),
                        bcx,
                        count,
                        HIDDEN,
                        3 * HIDDEN,
                    )?;
                    for (row, bx) in bcx
                        .chunks_exact(3 * HIDDEN)
                        .zip(post.chunks_exact_mut(HIDDEN))
                    {
                        mul(&row[..HIDDEN], &row[2 * HIDDEN..], bx)?;
                    }
                    let frame = self.state.histories()[conv_frames.len()].prepare_chunk(
                        post,
                        weights.get(coefficients),
                        histories.next().unwrap(),
                        output,
                        count,
                    )?;
                    report(layer_index, "history", frame.candidate())?;
                    conv_frames.push(frame);
                    for ((row, conv), gated) in bcx
                        .chunks_exact(3 * HIDDEN)
                        .zip(output.chunks_exact(HIDDEN))
                        .zip(post.chunks_exact_mut(HIDDEN))
                    {
                        mul(&row[HIDDEN..2 * HIDDEN], conv, gated)?;
                    }
                    linear(
                        post,
                        weights.get(project_out),
                        output,
                        count,
                        HIDDEN,
                        HIDDEN,
                    )?;
                }
                Operator::Attention {
                    q,
                    k,
                    v,
                    gate,
                    q_norm,
                    k_norm,
                    output: project_out,
                } => {
                    let slot = kv_frames.len();
                    let rotary = if ATTENTION_WINDOWS[slot].is_some() {
                        &self.model.sliding_rotary
                    } else {
                        &self.model.full_rotary
                    };
                    linear(normalized, weights.get(q), output, count, HIDDEN, HIDDEN)?;
                    norm(
                        output,
                        weights.get(q_norm),
                        post,
                        count * QUERY_HEADS,
                        HEAD_DIM,
                    )?;
                    rotary.apply(post, query, count, QUERY_HEADS, offset)?;

                    linear(normalized, weights.get(k), kv, count, HIDDEN, KV_WIDTH)?;
                    norm(kv, weights.get(k_norm), kv_norm, count * KV_HEADS, HEAD_DIM)?;
                    rotary.apply(kv_norm, kv, count, KV_HEADS, offset)?;
                    let new_k = &mut keys.next().unwrap()[..kv_words];
                    head_major(kv, new_k, count);
                    linear(normalized, weights.get(v), kv, count, HIDDEN, KV_WIDTH)?;
                    let new_v = &mut values.next().unwrap()[..kv_words];
                    head_major(kv, new_v, count);
                    report(layer_index, "key_proposal", new_k)?;
                    report(layer_index, "value_proposal", new_v)?;
                    let cache = &self.state.caches()[slot];
                    let frame = cache.prepare(new_k, new_v, count)?;
                    let view = cache.view(&frame)?;
                    let shape = view.attention_shape(QUERY_HEADS, LFM3_ATTENTION_SCALE);
                    attention(
                        query,
                        &view.segments(),
                        post,
                        &mut self.scratch.scores[..shape.keys],
                        &shape,
                    )?;
                    kv_frames.push(frame);
                    linear(
                        normalized,
                        weights.get(gate),
                        gate_raw,
                        count,
                        HIDDEN,
                        QUERY_HEADS,
                    )?;
                    sigmoid(gate_raw, gate_sigmoid)?;
                    head_gate(post, gate_sigmoid, query, count)?;
                    linear(
                        query,
                        weights.get(project_out),
                        output,
                        count,
                        HIDDEN,
                        HIDDEN,
                    )?;
                }
            }

            norm(
                output,
                weights.get(layer.post_operator_norm),
                post,
                count,
                HIDDEN,
            )?;
            add(current, post, residual)?;
            norm(
                residual,
                weights.get(layer.ffn_norm),
                normalized,
                count,
                HIDDEN,
            )?;
            linear(
                normalized,
                weights.get(layer.gate),
                gate,
                count,
                HIDDEN,
                INTERMEDIATE,
            )?;
            silu(gate, activated)?;
            linear(
                normalized,
                weights.get(layer.up),
                up,
                count,
                HIDDEN,
                INTERMEDIATE,
            )?;
            mul(activated, up, gate)?;
            linear(
                gate,
                weights.get(layer.down),
                output,
                count,
                INTERMEDIATE,
                HIDDEN,
            )?;
            norm(
                output,
                weights.get(layer.post_ffn_norm),
                post,
                count,
                HIDDEN,
            )?;
            add(residual, post, current)?;
            report(layer_index, "layer_output", current)?;
        }
        norm(
            current,
            weights.get(weights.final_norm),
            normalized,
            count,
            HIDDEN,
        )?;
        report(None, "final_norm", normalized)?;
        linear(
            &normalized[width - HIDDEN..],
            embedding,
            &mut self.scratch.logits,
            1,
            HIDDEN,
            VOCAB,
        )?;
        report(None, "logits", &self.scratch.logits)?;

        // No fallible operations after the joint validation and state copy.
        self.state.commit(&conv_frames, &kv_frames, count)?;
        self.history[offset..end].copy_from_slice(tokens);
        std::mem::swap(&mut self.logits, &mut self.scratch.logits);
        Ok(&self.logits)
    }
}
