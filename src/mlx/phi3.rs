//! Strict ordinary-RoPE Phi3 admission in a bounded full-attention prefix.
use super::dense::{Architecture, ParsedConfig};
use crate::{
    model::{Config, PositionPlan},
    weights::{Error, LoadOptions},
};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;

pub(crate) const IMPLEMENTED_POSITIONS: usize = 512;

fn one() -> f64 {
    1.0
}

#[derive(Deserialize)]
struct Phi3Config {
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
    original_max_position_embeddings: usize,
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
    embd_pdrop: f64,
    #[serde(default)]
    resid_pdrop: f64,
    #[serde(default)]
    rope_traditional: bool,
    #[serde(default)]
    rope_interleaved: bool,
    #[serde(default)]
    rope_scaling: Option<Value>,
    #[serde(default = "one")]
    partial_rotary_factor: f64,
    #[serde(default)]
    sliding_window: Option<usize>,
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

pub(crate) fn config(value: Value, options: LoadOptions) -> Result<ParsedConfig, Error> {
    let c: Phi3Config = serde_json::from_value(value)?;
    if c.model_type != "phi3"
        || c.architectures != ["Phi3ForCausalLM"]
        || c.hidden_act != "silu"
        || c.attention_bias
        || c.mlp_bias
        || c.attention_dropout != 0.0
        || c.embd_pdrop != 0.0
        || c.resid_pdrop != 0.0
        || c.rope_traditional
        || c.rope_interleaved
        || c.rope_scaling.is_some()
        || c.partial_rotary_factor != 1.0
    {
        return Err(invalid(
            "Phi3 requires bias-free SiLU, zero dropout and ordinary full-head split-half RoPE; LongRoPE is unimplemented",
        ));
    }
    if c.torch_dtype
        .as_deref()
        .is_some_and(|v| !["float32", "float16", "bfloat16"].contains(&v))
    {
        return Err(invalid("unsupported Phi3 declared torch dtype"));
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
            "auto_map",
        ]
        .contains(&name.as_str())
        {
            return Err(invalid(format!("unsupported Phi3 config field: {name}")));
        }
    }
    // auto_map is descriptive export metadata; it is never imported/executed.
    if c.extra.get("auto_map").is_some_and(|v| {
        v.as_object()
            .is_none_or(|m| m.values().any(|entry| entry.as_str().is_none()))
    }) {
        return Err(invalid("Phi3 auto_map must be descriptive string metadata"));
    }
    if c.num_attention_heads == 0 || !c.hidden_size.is_multiple_of(c.num_attention_heads) {
        return Err(invalid(
            "Phi3 hidden size must divide into nonzero query heads",
        ));
    }
    let head_dim = c.hidden_size / c.num_attention_heads;
    if c.head_dim.is_some_and(|v| v != head_dim)
        || c.original_max_position_embeddings == 0
        || c.sliding_window == Some(0)
        || options.context_limit == 0
        || options.context_limit > IMPLEMENTED_POSITIONS
        || options.context_limit > c.max_position_embeddings
        || c.sliding_window.is_some_and(|w| options.context_limit > w)
    {
        return Err(invalid(
            "Phi3 requires consistent head geometry and context within declared positions, sliding window and initial512 implementation",
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
        architecture: Architecture::Phi3,
        tied_output: c.tie_word_embeddings,
        position_plan: PositionPlan::Ordinary,
        declared_sliding_window: c.sliding_window,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn document() -> Value {
        json!({
            "model_type":"phi3", "architectures":["Phi3ForCausalLM"],
            "hidden_act":"silu", "hidden_size":3072, "intermediate_size":8192,
            "num_hidden_layers":32, "num_attention_heads":32, "num_key_value_heads":32,
            "vocab_size":32064, "max_position_embeddings":4096,
            "original_max_position_embeddings":4096, "rms_norm_eps":1e-5,
            "rope_theta":10000, "tie_word_embeddings":false, "sliding_window":2047,
            "rope_scaling":null, "torch_dtype":"bfloat16",
            "auto_map":{"AutoConfig":"configuration_phi3.Phi3Config"}
        })
    }
    fn request(positions: usize) -> LoadOptions {
        LoadOptions {
            context_limit: positions,
            ..LoadOptions::default()
        }
    }

    #[test]
    fn preserves_requested_context_and_source_window_without_clamping() {
        for positions in [1, 31, 511, 512] {
            let parsed = config(document(), request(positions)).unwrap();
            assert_eq!(parsed.config.max_positions, positions);
            assert_eq!(parsed.declared_sliding_window, Some(2047));
            assert_eq!(parsed.config.head_dim, 96);
            assert_eq!(parsed.architecture, Architecture::Phi3);
            assert_eq!(parsed.position_plan, PositionPlan::Ordinary);
            assert!(!parsed.tied_output);
        }
        for positions in [0, 513, 2048, 4096, usize::MAX] {
            assert!(config(document(), request(positions)).is_err());
        }
        let mut d = document();
        d["sliding_window"] = json!(31);
        assert_eq!(
            config(d.clone(), request(31)).unwrap().config.max_positions,
            31
        );
        assert!(config(d, request(32)).is_err());
        let mut d = document();
        d["max_position_embeddings"] = json!(31);
        assert!(config(d, request(32)).is_err());
    }

    #[test]
    fn validates_window_types_and_upstream_full_attention_default() {
        for value in [
            json!(0),
            json!(-1),
            json!(1.0),
            json!("2047"),
            json!(true),
            json!([]),
            json!({}),
        ] {
            let mut d = document();
            d["sliding_window"] = value;
            assert!(config(d, request(1)).is_err());
        }
        for absent in [false, true] {
            let mut d = document();
            d["sliding_window"] = Value::Null;
            if absent {
                d.as_object_mut().unwrap().remove("sliding_window");
            }
            assert_eq!(
                config(d, request(512)).unwrap().declared_sliding_window,
                None
            );
        }
    }

    #[test]
    fn rejects_unimplemented_semantics_and_conflicting_dtype_aliases() {
        for (key, value) in [
            ("rope_scaling", json!({"type":"longrope"})),
            ("partial_rotary_factor", json!(0.5)),
            ("attention_bias", json!(true)),
            ("mlp_bias", json!(true)),
            ("attention_dropout", json!(0.1)),
            ("embd_pdrop", json!(0.1)),
            ("resid_pdrop", json!(0.1)),
            ("rope_interleaved", json!(true)),
            ("rope_traditional", json!(true)),
            ("head_dim", json!(128)),
            ("num_attention_heads", json!(0)),
            ("num_key_value_heads", json!(3)),
            ("original_max_position_embeddings", json!(0)),
            ("hidden_act", json!("gelu")),
            ("qk_layernorm", json!(true)),
            ("auto_map", json!({"AutoConfig":[]})),
            ("dtype", json!("float16")),
        ] {
            let mut d = document();
            d[key] = value;
            assert!(config(d, request(32)).is_err(), "{key}");
        }
        let mut d = document();
        d["tie_word_embeddings"] = json!(true);
        assert!(config(d, request(32)).unwrap().tied_output);
    }
}
