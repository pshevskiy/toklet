//! Explicit graph admission for the supported MLX dense decoders.
use crate::{
    model::{Config, PositionPlan},
    weights::{self, Error, LoadOptions},
};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
mod granite;
mod smollm3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Architecture {
    Qwen2,
    Qwen3,
    Llama,
    Mistral,
    Phi3,
    SmolLm3,
    Granite,
}
impl Architecture {
    pub(crate) const fn has_projection_bias(self) -> bool {
        matches!(self, Self::Qwen2)
    }
}

/// Validated source semantics, before physical tensors or numerical allocation.
pub(crate) struct ParsedConfig {
    pub config: Config,
    pub architecture: Architecture,
    pub tied_output: bool,
    pub position_plan: PositionPlan,
    /// Source declaration; not a sliding cache implementation.
    pub declared_sliding_window: Option<usize>,
    /// Source mask: true applies the positional plan; false leaves Q/K unchanged.
    pub rotary_layers: Option<Vec<bool>>,
    pub scales: crate::model::DenseScales,
}

#[derive(Deserialize)]
struct BiasFreeConfig {
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

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Llama3Parameters {
    #[serde(default)]
    rope_type: Option<String>,
    #[serde(default, rename = "type")]
    legacy_type: Option<String>,
    factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_max_position_embeddings: usize,
}

fn positions(value: Option<Value>, architecture: Architecture) -> Result<PositionPlan, Error> {
    let Some(value) = value else {
        return Ok(PositionPlan::Ordinary);
    };
    if architecture != Architecture::Llama {
        return Err(invalid(
            "scaled RoPE is implemented only for the Llama adapter",
        ));
    }
    let object = value
        .as_object()
        .ok_or_else(|| invalid("RoPE scaling must be an object"))?;
    for key in ["rope_type", "type"] {
        if object.get(key).is_some_and(|v| v.as_str().is_none()) {
            return Err(invalid("RoPE type aliases must be strings"));
        }
    }
    let p: Llama3Parameters = serde_json::from_value(value)?;
    if p.rope_type
        .as_ref()
        .zip(p.legacy_type.as_ref())
        .is_some_and(|(a, b)| a != b)
    {
        return Err(invalid("conflicting RoPE type aliases"));
    }
    if p.rope_type.or(p.legacy_type).as_deref() != Some("llama3") {
        return Err(invalid("only explicit llama3 RoPE scaling is implemented"));
    }
    Ok(PositionPlan::Llama3(crate::cpu::Llama3Scaling::new(
        p.factor,
        p.low_freq_factor,
        p.high_freq_factor,
        p.original_max_position_embeddings,
    )?))
}
fn invalid(reason: impl Into<String>) -> Error {
    Error::Invalid(reason.into())
}

pub(crate) fn config(mut value: Value, options: LoadOptions) -> Result<ParsedConfig, Error> {
    match value.get("model_type").and_then(Value::as_str) {
        Some("qwen3") => {
            let (config, tied) = super::qwen3::config(value, options)?;
            Ok(ParsedConfig {
                config,
                architecture: Architecture::Qwen3,
                tied_output: tied,
                position_plan: PositionPlan::Ordinary,
                declared_sliding_window: None,
                rotary_layers: None,
                scales: Default::default(),
            })
        }
        Some("qwen2") => Ok(ParsedConfig {
            config: weights::qwen_config(value, options)?,
            architecture: Architecture::Qwen2,
            tied_output: true,
            position_plan: PositionPlan::Ordinary,
            declared_sliding_window: None,
            rotary_layers: None,
            scales: Default::default(),
        }),
        Some("phi3") => super::phi3::config(value, options),
        Some("llama" | "mistral" | "smollm3" | "granite") => {
            let scales = if value["model_type"] == "granite" {
                granite::prepare(&mut value)?
            } else {
                Default::default()
            };
            let rotary_layers = if value["model_type"] == "smollm3" {
                Some(smollm3::prepare(&mut value)?)
            } else {
                None
            };
            // Mistral defaults can imply sliding attention when the field is
            // absent. Only an explicit null opts into this full-attention graph.
            if value.get("model_type").and_then(Value::as_str) == Some("mistral")
                && (value.get("sliding_window") != Some(&Value::Null)
                    || [
                        "is_llama_config",
                        "transformers.js_config",
                        "pretraining_tp",
                    ]
                    .iter()
                    .any(|name| value.get(*name).is_some()))
            {
                return Err(invalid(
                    "Mistral requires explicit sliding_window:null and no Llama-only export metadata",
                ));
            }
            let c: BiasFreeConfig = serde_json::from_value(value)?;
            let (architecture, class) = if c.model_type == "llama" {
                (Architecture::Llama, "LlamaForCausalLM")
            } else if c.model_type == "smollm3" {
                (Architecture::SmolLm3, "SmolLM3ForCausalLM")
            } else if c.model_type == "granite" {
                (Architecture::Granite, "GraniteForCausalLM")
            } else {
                (Architecture::Mistral, "MistralForCausalLM")
            };
            if c.architectures != [class]
                || c.hidden_act != "silu"
                || c.attention_bias
                || c.mlp_bias
                || c.attention_dropout != 0.0
                || c.rope_traditional
                || c.rope_interleaved
                || c.pretraining_tp
                    .is_some_and(|v| v != 1 && !(architecture == Architecture::SmolLm3 && v == 2))
                || c.is_llama_config.is_some_and(|v| !v)
                || c.transformers_js_config.as_ref().is_some_and(|v| {
                    // Export metadata only: the native session always owns Half KV.
                    v != &serde_json::json!({"kv_cache_dtype":{"q4f16":"float16","fp16":"float16"}})
                })
                || c.use_sliding_window
                || c.sliding_window.is_some()
            {
                return Err(invalid(
                    "dense source requires bias-free SiLU, split-half RoPE and full causal attention",
                ));
            }
            let position_plan = positions(c.rope_scaling, architecture)?;
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
                    return Err(invalid(format!(
                        "unsupported bias-free decoder config field: {name}"
                    )));
                }
            }
            if c.num_attention_heads == 0 || !c.hidden_size.is_multiple_of(c.num_attention_heads) {
                return Err(invalid(
                    "bias-free decoder hidden_size must be divisible by nonzero num_attention_heads",
                ));
            }
            let head_dim = c.hidden_size / c.num_attention_heads;
            if c.head_dim.is_some_and(|v| v != head_dim) {
                return Err(invalid(
                    "bias-free decoder head_dim disagrees with hidden_size/num_attention_heads",
                ));
            }
            if options.context_limit == 0
                || options.context_limit > weights::MAX_CONTEXT_LIMIT
                || options.context_limit > c.max_position_embeddings
            {
                return Err(invalid(
                    "bias-free decoder context limit must fit model positions and 32768-token ceiling",
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
            Ok(ParsedConfig {
                config,
                architecture,
                tied_output: c.tie_word_embeddings,
                position_plan,
                declared_sliding_window: None,
                rotary_layers,
                scales,
            })
        }
        _ => Err(invalid("unsupported native MLX decoder architecture")),
    }
}
