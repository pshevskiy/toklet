use super::*;
use serde::Deserialize;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Raw {
    architectures: Vec<String>,
    model_type: String,
    hidden_act: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    num_local_experts: usize,
    num_experts_per_tok: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    embedding_multiplier: f32,
    attention_multiplier: f32,
    residual_multiplier: f32,
    logits_scaling: f32,
    tie_word_embeddings: bool,
    attention_bias: bool,
    attention_dropout: f64,
    rope_scaling: Option<serde_json::Value>,
    output_router_logits: bool,
    router_aux_loss_coef: f64,
    torch_dtype: String,
    bos_token_id: u32,
    eos_token_id: u32,
    pad_token_id: u32,
    #[serde(default)]
    initializer_range: Option<f64>,
    #[serde(default)]
    transformers_version: Option<String>,
    #[serde(default)]
    use_cache: Option<bool>,
}
pub(super) fn parse(bytes: &[u8]) -> Result<Config, Error> {
    let r: Raw = serde_json::from_slice(bytes)?;
    if r.architectures != ["GraniteMoeForCausalLM"]
        || r.model_type != "granitemoe"
        || r.hidden_act != "silu"
        || !r.tie_word_embeddings
        || r.attention_bias
        || r.attention_dropout != 0.
        || r.rope_scaling.is_some()
        || r.output_router_logits
        || r.router_aux_loss_coef != 0.
        || r.torch_dtype != "bfloat16"
        || r.bos_token_id != 0
        || r.eos_token_id != 0
        || r.pad_token_id != 0
        || r.num_local_experts == 0
        || r.num_local_experts > 32
        || r.num_experts_per_tok == 0
        || r.num_experts_per_tok > r.num_local_experts
        || r.num_hidden_layers == 0
        || r.num_hidden_layers > 128
        || r.num_attention_heads == 0
        || !r.hidden_size.is_multiple_of(r.num_attention_heads)
        || r.initializer_range
            .is_some_and(|x| !x.is_finite() || x < 0.)
        || r.transformers_version
            .as_ref()
            .is_some_and(|s| s.len() > 128)
        || r.use_cache == Some(false)
    {
        return Err(invalid("unsupported Granite MoE source semantics"));
    }
    let dense = crate::model::Config {
        vocab_size: r.vocab_size,
        hidden_size: r.hidden_size,
        intermediate_size: r.intermediate_size,
        num_layers: r.num_hidden_layers,
        query_heads: r.num_attention_heads,
        kv_heads: r.num_key_value_heads,
        head_dim: r.hidden_size / r.num_attention_heads,
        max_positions: r.max_position_embeddings,
        rms_norm_eps: r.rms_norm_eps,
        rope_theta: r.rope_theta,
    };
    dense.validate()?;
    let scales = crate::model::DenseScales {
        embedding: r.embedding_multiplier,
        attention: Some(r.attention_multiplier),
        residual: r.residual_multiplier,
        logits_divisor: r.logits_scaling,
    };
    scales.validate()?;
    for x in [scales.embedding, scales.residual] {
        if !half::f16::from_f32(x).is_normal() {
            return Err(invalid("Granite Half coefficients must remain normal"));
        }
    }
    Ok(Config {
        dense,
        experts: r.num_local_experts,
        selected: r.num_experts_per_tok,
        scales,
    })
}
