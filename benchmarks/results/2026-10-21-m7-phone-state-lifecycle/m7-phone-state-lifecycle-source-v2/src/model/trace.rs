use super::{Config, Error, product, total_bytes, zeros};

/// Diagnostic arrays use the same layouts and names as the frozen tiny oracle.
pub struct LayerTrace {
    pub input_norm: Vec<f32>,
    pub q_projection: Vec<f32>,
    pub k_projection: Vec<f32>,
    pub v_projection: Vec<f32>,
    pub q_rotary: Vec<f32>,
    pub k_rotary: Vec<f32>,
    pub attention_scores: Vec<f32>,
    pub attention_probabilities: Vec<f32>,
    pub attention_context: Vec<f32>,
    pub attention_projection: Vec<f32>,
    pub attention_residual: Vec<f32>,
    pub post_attention_norm: Vec<f32>,
    pub gate_projection: Vec<f32>,
    pub up_projection: Vec<f32>,
    pub gate_silu: Vec<f32>,
    pub gated: Vec<f32>,
    pub down_projection: Vec<f32>,
    pub output: Vec<f32>,
}

pub struct Trace {
    pub embedding: Vec<f32>,
    pub layers: Vec<LayerTrace>,
    pub final_norm: Vec<f32>,
    pub logits: Vec<f32>,
}

impl Trace {
    pub(super) fn new(c: &Config, tokens: usize) -> Result<Self, Error> {
        let nh = product(tokens, c.hidden_size)?;
        let nkv = product(product(tokens, c.kv_heads)?, c.head_dim)?;
        let nm = product(tokens, c.intermediate_size)?;
        let ns = product(product(c.query_heads, tokens)?, tokens)?;
        let logits = product(tokens, c.vocab_size)?;
        let layer_bytes = total_bytes(&[
            nh, nh, nkv, nkv, nh, nkv, ns, ns, nh, nh, nh, nh, nm, nm, nm, nm, nh, nh,
        ])?;
        let base_bytes = total_bytes(&[nh, nh, logits])?;
        product(layer_bytes, c.num_layers)?
            .checked_add(base_bytes)
            .ok_or(crate::cpu::Error::SizeOverflow)?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(c.num_layers)
            .map_err(|_| Error::Allocation)?;
        for _ in 0..c.num_layers {
            layers.push(LayerTrace {
                input_norm: zeros(nh)?,
                q_projection: zeros(nh)?,
                k_projection: zeros(nkv)?,
                v_projection: zeros(nkv)?,
                q_rotary: zeros(nh)?,
                k_rotary: zeros(nkv)?,
                attention_scores: zeros(ns)?,
                attention_probabilities: zeros(ns)?,
                attention_context: zeros(nh)?,
                attention_projection: zeros(nh)?,
                attention_residual: zeros(nh)?,
                post_attention_norm: zeros(nh)?,
                gate_projection: zeros(nm)?,
                up_projection: zeros(nm)?,
                gate_silu: zeros(nm)?,
                gated: zeros(nm)?,
                down_projection: zeros(nh)?,
                output: zeros(nh)?,
            });
        }
        Ok(Self {
            embedding: zeros(nh)?,
            layers,
            final_norm: zeros(nh)?,
            logits: zeros(logits)?,
        })
    }
}
