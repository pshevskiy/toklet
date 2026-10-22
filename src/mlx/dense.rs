//! Explicit graph admission for the supported MLX dense decoders.
use crate::{
    model::Config,
    weights::{self, Error, LoadOptions},
};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Architecture {
    Qwen2,
    Llama,
}

#[derive(Deserialize)]
struct LlamaConfig {
    model_type: String,
    architectures: Vec<String>,
    hidden_act: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
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
    rope_traditional: bool,
    #[serde(default)]
    rope_interleaved: bool,
    #[serde(default)]
    pretraining_tp: Option<usize>,
    #[serde(default)]
    is_llama_config: Option<bool>,
    #[serde(default, rename = "transformers.js_config")]
    transformers_js_config: Option<Value>,
    #[serde(default)]
    rope_scaling: Option<Value>,
    #[serde(default)]
    use_sliding_window: bool,
    #[serde(default)]
    sliding_window: Option<Value>,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default, alias = "dtype")]
    torch_dtype: Option<String>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}
fn invalid(reason: impl Into<String>) -> Error {
    Error::Invalid(reason.into())
}

pub(crate) fn config(
    value: Value,
    options: LoadOptions,
) -> Result<(Config, Architecture, bool), Error> {
    match value.get("model_type").and_then(Value::as_str) {
        Some("qwen2") => Ok((
            weights::qwen_config(value, options)?,
            Architecture::Qwen2,
            true,
        )),
        Some("llama") => {
            let c: LlamaConfig = serde_json::from_value(value)?;
            if c.model_type != "llama"
                || c.architectures != ["LlamaForCausalLM"]
                || c.hidden_act != "silu"
                || c.attention_bias
                || c.mlp_bias
                || c.attention_dropout != 0.0
                || c.rope_traditional
                || c.rope_interleaved
                || c.pretraining_tp.is_some_and(|v| v != 1)
                || c.is_llama_config.is_some_and(|v| !v)
                || c.transformers_js_config.as_ref().is_some_and(|v| {
                    // Export metadata only: the native session always owns Half KV.
                    v != &serde_json::json!({"kv_cache_dtype":{"q4f16":"float16","fp16":"float16"}})
                })
                || c.rope_scaling.is_some()
                || c.use_sliding_window
                || c.sliding_window.is_some()
            {
                return Err(invalid(
                    "only bias-free Llama with SiLU, ordinary split-half RoPE and full causal attention is supported",
                ));
            }
            if c.torch_dtype
                .as_deref()
                .is_some_and(|v| !["float32", "float16", "bfloat16"].contains(&v))
            {
                return Err(invalid("unsupported declared torch dtype"));
            }
            for name in c.extra.keys() {
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
                    return Err(invalid(format!("unsupported Llama config field: {name}")));
                }
            }
            if c.num_attention_heads == 0 || !c.hidden_size.is_multiple_of(c.num_attention_heads) {
                return Err(invalid(
                    "Llama hidden_size must be divisible by nonzero num_attention_heads",
                ));
            }
            let head_dim = c.hidden_size / c.num_attention_heads;
            if c.head_dim.is_some_and(|v| v != head_dim) {
                return Err(invalid(
                    "Llama head_dim disagrees with hidden_size/num_attention_heads",
                ));
            }
            if options.context_limit == 0
                || options.context_limit > weights::MAX_CONTEXT_LIMIT
                || options.context_limit > c.max_position_embeddings
            {
                return Err(invalid(
                    "Llama context limit must fit model positions and 32768-token ceiling",
                ));
            }
            let config = Config {
                vocab_size: c.vocab_size,
                hidden_size: c.hidden_size,
                intermediate_size: c.intermediate_size,
                num_layers: c.num_hidden_layers,
                query_heads: c.num_attention_heads,
                kv_heads: c.num_key_value_heads,
                head_dim,
                max_positions: options.context_limit,
                rms_norm_eps: c.rms_norm_eps,
                rope_theta: c.rope_theta,
            };
            config.validate()?;
            Ok((config, Architecture::Llama, c.tie_word_embeddings))
        }
        _ => Err(invalid("unsupported native MLX decoder architecture")),
    }
}
