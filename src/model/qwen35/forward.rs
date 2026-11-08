use super::*;
use state::{AttentionScratch, DeltaScratch, DeltaState, Kv, Scratch, check_cancel};
use std::sync::atomic::AtomicBool;

pub(super) struct Trace<'a, F> {
    pub observe: &'a mut F,
    pub token: usize,
    pub position: usize,
    pub layer: Option<usize>,
    pub cancel: &'a AtomicBool,
}
impl<F: for<'b> FnMut(Observation<'b>) -> Result<()>> Trace<'_, F> {
    fn emit<const TRACE: bool>(
        &mut self,
        stage: &'static str,
        head: Option<usize>,
        values: &[f32],
    ) -> Result<()> {
        if TRACE {
            (self.observe)(Observation {
                token: self.token,
                position: self.position,
                layer: self.layer,
                stage,
                head,
                values,
            })?;
            check_cancel(self.cancel)?;
        }
        Ok(())
    }
}
fn project(m: &Matrix<'_>, input: &[f32], output: &mut [f32], work: &mut Work) -> Result<()> {
    m.matvec(input, output).map_err(|e| invalid(e.0))?;
    work.projections_completed += 1;
    Ok(())
}
fn partial_rotary(
    input: &[f32],
    output: &mut [f32],
    heads: usize,
    c: Config,
    position: usize,
) -> Result<()> {
    // The Qwen prefix has its own pair distance and frequency denominator.
    // Gemma proportional rotary uses the full dimension and is not equivalent.
    output.copy_from_slice(input);
    for head in 0..heads {
        let start = head * c.head_dim;
        let end = start + c.rotated_dims;
        cpu::rotary(
            &input[start..end],
            &mut output[start..end],
            1,
            1,
            c.rotated_dims,
            position,
            c.rope_theta,
        )?;
    }
    Ok(())
}
fn sigmoid(x: f32) -> f32 {
    let x = f64::from(x);
    let ex = (-x.abs()).exp();
    (if x >= 0. {
        1. / (1. + ex)
    } else {
        ex / (1. + ex)
    }) as f32
}

#[allow(clippy::too_many_arguments)]
fn delta<const TRACE: bool, F: for<'b> FnMut(Observation<'b>) -> Result<()>>(
    c: Config,
    w: &DeltaWeights<'_>,
    state: &mut DeltaState,
    s: &mut DeltaScratch,
    input: &[f32],
    output: &mut [f32],
    local: usize,
    work: &mut Work,
    t: &mut Trace<'_, F>,
) -> Result<()> {
    project(&w.qkv, input, &mut s.qkv, work)?;
    t.emit::<TRACE>("delta_qkv", None, &s.qkv)?;
    project(&w.z, input, &mut s.z, work)?;
    t.emit::<TRACE>("delta_z", None, &s.z)?;
    project(&w.a, input, &mut s.a, work)?;
    t.emit::<TRACE>("delta_a", None, &s.a)?;
    project(&w.b, input, &mut s.b, work)?;
    t.emit::<TRACE>("delta_b", None, &s.b)?;
    let (history, next_history) = state.history.next(local);
    cpu::causal_depthwise_conv_step(
        &s.qkv,
        w.convolution,
        history,
        next_history,
        &mut s.conv,
        c.conv_width()?,
        4,
    )?;
    work.convolution_steps_completed += 1;
    cpu::silu(&s.conv, &mut s.activated)?;
    t.emit::<TRACE>("delta_convolved", None, &s.activated)?;
    let width = c.key_width()?;
    let q = &s.activated[..width];
    let k = &s.activated[width..2 * width];
    let v = &s.activated[2 * width..];
    cpu::gated_delta_normalize(q, k, &mut s.q, &mut s.k, c.key_heads, c.key_dim, 1e-6)?;
    t.emit::<TRACE>("delta_q", None, &s.q)?;
    t.emit::<TRACE>("delta_k", None, &s.k)?;
    t.emit::<TRACE>("delta_v", None, v)?;
    cpu::gated_delta_gates(&s.a, &s.b, w.a_log, w.dt_bias, &mut s.decay, &mut s.beta)?;
    t.emit::<TRACE>("delta_decay", None, &s.decay)?;
    t.emit::<TRACE>("delta_beta", None, &s.beta)?;
    let (previous, next) = state.rnn.next(local);
    cpu::gated_delta_step(
        &s.q,
        &s.k,
        v,
        &s.decay,
        &s.beta,
        previous,
        next,
        &mut s.output,
        &cpu::GatedDeltaShape {
            key_heads: c.key_heads,
            value_heads: c.value_heads,
            key_dim: c.key_dim,
            value_dim: c.value_dim,
        },
    )?;
    work.recurrence_steps_completed += 1;
    t.emit::<TRACE>("delta_state", None, next)?;
    t.emit::<TRACE>("delta_history", None, next_history)?;
    t.emit::<TRACE>("delta_output", None, &s.output)?;
    cpu::rmsnorm(
        &s.output,
        w.norm,
        &mut s.norm,
        c.value_heads,
        c.value_dim,
        c.rms_epsilon,
    )?;
    t.emit::<TRACE>("delta_norm", None, &s.norm)?;
    cpu::gated_silu(&s.z, &s.norm, &mut s.gated)?;
    t.emit::<TRACE>("delta_gated", None, &s.gated)?;
    project(&w.out, &s.gated, output, work)
}

#[allow(clippy::too_many_arguments)]
fn attention<const TRACE: bool, F: for<'b> FnMut(Observation<'b>) -> Result<()>>(
    c: Config,
    w: &AttentionWeights<'_>,
    kv: &mut Kv,
    s: &mut AttentionScratch,
    input: &[f32],
    output: &mut [f32],
    position: usize,
    absolute: usize,
    capacity: usize,
    work: &mut Work,
    t: &mut Trace<'_, F>,
) -> Result<()> {
    project(&w.q_gate, input, &mut s.q_gate, work)?;
    t.emit::<TRACE>("attention_q_gate", None, &s.q_gate)?;
    for head in 0..c.query_heads {
        let target = head * c.head_dim;
        let source = 2 * target;
        s.q[target..target + c.head_dim].copy_from_slice(&s.q_gate[source..source + c.head_dim]);
        s.gate[target..target + c.head_dim]
            .copy_from_slice(&s.q_gate[source + c.head_dim..source + 2 * c.head_dim]);
    }
    project(&w.k, input, &mut s.k, work)?;
    project(&w.v, input, &mut s.v, work)?;
    t.emit::<TRACE>("attention_q", None, &s.q)?;
    t.emit::<TRACE>("attention_k", None, &s.k)?;
    t.emit::<TRACE>("attention_v", None, &s.v)?;
    cpu::rmsnorm(
        &s.q,
        w.q_norm,
        &mut s.q_norm,
        c.query_heads,
        c.head_dim,
        c.rms_epsilon,
    )?;
    cpu::rmsnorm(
        &s.k,
        w.k_norm,
        &mut s.k_norm,
        c.kv_heads,
        c.head_dim,
        c.rms_epsilon,
    )?;
    t.emit::<TRACE>("attention_q_norm", None, &s.q_norm)?;
    t.emit::<TRACE>("attention_k_norm", None, &s.k_norm)?;
    partial_rotary(&s.q_norm, &mut s.q_rot, c.query_heads, c, absolute)?;
    partial_rotary(&s.k_norm, &mut s.k_rot, c.kv_heads, c, absolute)?;
    t.emit::<TRACE>("attention_q_rot", None, &s.q_rot)?;
    t.emit::<TRACE>("attention_k_rot", None, &s.k_rot)?;
    let stride = capacity * c.head_dim;
    for head in 0..c.kv_heads {
        let target = head * stride + position * c.head_dim;
        let source = head * c.head_dim;
        kv.k[target..target + c.head_dim].copy_from_slice(&s.k_rot[source..source + c.head_dim]);
        kv.v[target..target + c.head_dim].copy_from_slice(&s.v[source..source + c.head_dim]);
        t.emit::<TRACE>(
            "attention_k_cache",
            Some(head),
            &kv.k[head * stride..head * stride + (position + 1) * c.head_dim],
        )?;
        t.emit::<TRACE>(
            "attention_v_cache",
            Some(head),
            &kv.v[head * stride..head * stride + (position + 1) * c.head_dim],
        )?;
    }
    let scale = (c.head_dim as f64).sqrt().recip() as f32;
    for head in 0..c.query_heads {
        let key_head = head / (c.query_heads / c.kv_heads);
        let source = key_head * stride;
        let query = head * c.head_dim;
        cpu::attention_cached_scaled(
            &s.q_rot[query..query + c.head_dim],
            &kv.k[source..source + stride],
            &kv.v[source..source + stride],
            &mut s.context[query..query + c.head_dim],
            &mut s.scores,
            &mut s.probabilities,
            &cpu::CachedAttentionShape {
                tokens: 1,
                query_heads: 1,
                kv_heads: 1,
                head_dim: c.head_dim,
                capacity,
                offset: position,
            },
            scale,
        )?;
        work.attention_heads_completed += 1;
        t.emit::<TRACE>("attention_scores", Some(head), &s.scores[..position + 1])?;
        t.emit::<TRACE>(
            "attention_probabilities",
            Some(head),
            &s.probabilities[..position + 1],
        )?;
    }
    t.emit::<TRACE>("attention_context", None, &s.context)?;
    for ((out, &value), &gate) in s.gated.iter_mut().zip(&s.context).zip(&s.gate) {
        *out = value * sigmoid(gate);
    }
    cpu::finite("Qwen3.5 gated attention", &s.gated)?;
    t.emit::<TRACE>("attention_gated", None, &s.gated)?;
    project(&w.out, &s.gated, output, work)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn frame<const TRACE: bool, F: for<'b> FnMut(Observation<'b>) -> Result<()>>(
    model: &Model<'_>,
    states: &mut [DeltaState],
    kv: &mut [Kv],
    s: &mut Scratch,
    token: u32,
    local: usize,
    position: usize,
    absolute: usize,
    capacity: usize,
    logits: &mut [f32],
    work: &mut Work,
    t: &mut Trace<'_, F>,
) -> Result<()> {
    let c = model.config;
    model
        .embedding
        .copy_row(token as usize, &mut s.x)
        .map_err(|e| invalid(e.0))?;
    work.gathers_completed += 1;
    t.emit::<TRACE>("embedding", None, &s.x)?;
    for (layer_index, layer) in model.layers.iter().enumerate() {
        check_cancel(t.cancel)?;
        t.layer = Some(layer_index);
        let w = &layer.weights;
        cpu::rmsnorm(
            &s.x,
            w.input_norm,
            &mut s.input_norm,
            1,
            c.hidden,
            c.rms_epsilon,
        )?;
        t.emit::<TRACE>("input_norm", None, &s.input_norm)?;
        match &w.mixing {
            MixingWeights::Delta(d) => delta::<TRACE, F>(
                c,
                d,
                &mut states[layer.owner],
                &mut s.delta,
                &s.input_norm,
                &mut s.operator,
                local,
                work,
                t,
            )?,
            MixingWeights::Attention(a) => attention::<TRACE, F>(
                c,
                a,
                &mut kv[layer.owner],
                &mut s.attention,
                &s.input_norm,
                &mut s.operator,
                position,
                absolute,
                capacity,
                work,
                t,
            )?,
        }
        t.emit::<TRACE>("operator_output", None, &s.operator)?;
        cpu::residual_add(&mut s.x, &s.operator)?;
        t.emit::<TRACE>("operator_residual", None, &s.x)?;
        cpu::rmsnorm(
            &s.x,
            w.post_norm,
            &mut s.post_norm,
            1,
            c.hidden,
            c.rms_epsilon,
        )?;
        t.emit::<TRACE>("post_norm", None, &s.post_norm)?;
        project(&w.gate, &s.post_norm, &mut s.gate, work)?;
        t.emit::<TRACE>("mlp_gate", None, &s.gate)?;
        project(&w.up, &s.post_norm, &mut s.up, work)?;
        t.emit::<TRACE>("mlp_up", None, &s.up)?;
        cpu::gated_silu(&s.gate, &s.up, &mut s.activated)?;
        t.emit::<TRACE>("mlp_activated", None, &s.activated)?;
        project(&w.down, &s.activated, &mut s.output, work)?;
        t.emit::<TRACE>("mlp_output", None, &s.output)?;
        cpu::residual_add(&mut s.x, &s.output)?;
        t.emit::<TRACE>("layer_output", None, &s.x)?;
    }
    t.layer = None;
    cpu::rmsnorm(
        &s.x,
        model.final_norm,
        &mut s.final_norm,
        1,
        c.hidden,
        c.rms_epsilon,
    )?;
    t.emit::<TRACE>("final_norm", None, &s.final_norm)?;
    project(&model.embedding, &s.final_norm, logits, work)?;
    t.emit::<TRACE>("logits", None, logits)
}
