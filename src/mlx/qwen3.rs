//! Qwen3 dense config admission. MoE, scaled RoPE and local attention are separate graphs.
use crate::{
    model::Config,
    weights::{Error, LoadOptions, MAX_CONTEXT_LIMIT},
};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct DenseConfig {
    model_type: String,
    architectures: Vec<String>,
    hidden_act: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    tie_word_embeddings: bool,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    #[serde(default)]
    attention_dropout: f64,
    #[serde(default)]
    rope_scaling: Option<Value>,
    #[serde(default)]
    sliding_window: Option<Value>,
    #[serde(default)]
    use_sliding_window: bool,
    #[serde(default)]
    max_window_layers: Option<usize>,
    #[serde(default, alias = "dtype")]
    torch_dtype: Option<String>,
    #[serde(flatten)]
    metadata: BTreeMap<String, Value>,
}

pub(super) fn config(value: Value, options: LoadOptions) -> Result<(Config, bool), Error> {
    let c: DenseConfig = serde_json::from_value(value)?;
    let invalid = |s: &str| Error::Invalid(s.into());
    if c.model_type != "qwen3"
        || c.architectures != ["Qwen3ForCausalLM"]
        || c.hidden_act != "silu"
        || c.attention_bias
        || c.mlp_bias
        || c.attention_dropout != 0.0
        || c.rope_scaling.is_some()
        || c.sliding_window.is_some()
        || c.use_sliding_window
        || c.max_window_layers.is_some_and(|v| v > c.num_hidden_layers)
    {
        return Err(invalid(
            "Qwen3 dense requires bias-free SiLU, per-head Q/K RMSNorm, ordinary split-half RoPE and full attention",
        ));
    }
    if c.torch_dtype
        .as_deref()
        .is_some_and(|v| !["float32", "float16", "bfloat16"].contains(&v))
    {
        return Err(invalid("unsupported Qwen3 declared dtype"));
    }
    for name in c.metadata.keys() {
        if ![
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "initializer_range",
            "transformers_version",
            "use_cache",
            "_name_or_path",
        ]
        .contains(&name.as_str())
        {
            return Err(Error::Invalid(format!(
                "unsupported Qwen3 dense config field: {name}"
            )));
        }
    }
    if options.context_limit == 0
        || options.context_limit > MAX_CONTEXT_LIMIT
        || options.context_limit > c.max_position_embeddings
    {
        return Err(invalid(
            "Qwen3 context must fit model positions and 32768-token ceiling",
        ));
    }
    let config = Config {
        vocab_size: c.vocab_size,
        hidden_size: c.hidden_size,
        intermediate_size: c.intermediate_size,
        num_layers: c.num_hidden_layers,
        query_heads: c.num_attention_heads,
        kv_heads: c.num_key_value_heads,
        head_dim: c.head_dim,
        max_positions: options.context_limit,
        rms_norm_eps: c.rms_norm_eps,
        rope_theta: c.rope_theta,
    };
    config.validate()?;
    Ok((config, c.tie_word_embeddings))
}
