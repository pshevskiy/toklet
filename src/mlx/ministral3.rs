//! Selected Ministral 3 text semantics from a complete multimodal config wrapper.
//!
//! This borrowed metadata check authenticates no source bytes, tensor payload,
//! tokenizer or package. Its marker is not ordinary loader, graph, Metal, codec
//! or adoption admission. Vision fields are checked as selected metadata only;
//! no vision operation is implemented or qualified here.

use crate::weights::Error;
use serde_json::{Map, Value};

/// Fixed selected text geometry; query width is independent of hidden width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layers: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub query_width: usize,
    pub kv_width: usize,
    pub vocab_size: usize,
    pub original_max_positions: usize,
    pub source_max_positions: usize,
}

/// Evidence only of the selected text configuration's semantics.
///
/// The source storage declaration is BF16. This says nothing about the separate
/// prospective Half primitive or any full-model execution/storage precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Profile {
    geometry: Geometry,
}

impl Profile {
    pub const fn geometry(self) -> Geometry {
        self.geometry
    }
}

fn invalid(reason: impl Into<String>) -> Error {
    Error::Invalid(format!("Ministral 3 selected metadata: {}", reason.into()))
}

fn object<'a>(value: &'a Value, name: &str) -> Result<&'a Map<String, Value>, Error> {
    value
        .as_object()
        .ok_or_else(|| invalid(format!("{name} must be an object")))
}

fn whitelist(object: &Map<String, Value>, allowed: &[&str], name: &str) -> Result<(), Error> {
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(invalid(format!("unsupported {name} field {key}")));
        }
    }
    Ok(())
}

fn string(object: &Map<String, Value>, name: &str, expected: &str) -> Result<(), Error> {
    if object.get(name).and_then(Value::as_str) != Some(expected) {
        return Err(invalid(format!("{name} must be {expected}")));
    }
    Ok(())
}

fn integer(object: &Map<String, Value>, name: &str, expected: u64) -> Result<(), Error> {
    if object.get(name).and_then(Value::as_u64) != Some(expected) {
        return Err(invalid(format!("{name} must be integer {expected}")));
    }
    Ok(())
}

fn flag(
    object: &Map<String, Value>,
    name: &str,
    expected: bool,
    required: bool,
) -> Result<(), Error> {
    match object.get(name) {
        None if !required => Ok(()),
        Some(value) if value.as_bool() == Some(expected) => Ok(()),
        _ => Err(invalid(format!("{name} must be {expected}"))),
    }
}

// Compare decimal values before any binary floating-point conversion. The
// bounded normalization accepts equivalent notation, but never a nearby value
// that merely rounds to the selected FP32/FP64 number. JSON syntax is already
// established by serde_json; this function only normalizes its number spelling.
fn decimal(raw: &str) -> Option<(bool, String, i32)> {
    if raw.len() > 96 {
        return None;
    }
    let (negative, raw) = raw
        .strip_prefix('-')
        .map_or((false, raw), |rest| (true, rest));
    let (mantissa, exponent) = raw
        .split_once(['e', 'E'])
        .map_or(Some((raw, 0)), |(mantissa, exponent)| {
            Some((mantissa, exponent.parse::<i32>().ok()?))
        })?;
    let fractional = mantissa
        .split_once('.')
        .map_or(0, |(_, fraction)| fraction.len());
    let mut digits: String = mantissa.chars().filter(|&c| c != '.').collect();
    if digits.is_empty() || !digits.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let mut exponent = exponent.checked_sub(i32::try_from(fractional).ok()?)?;
    let significant = digits.trim_start_matches('0');
    if significant.is_empty() {
        return Some((false, "0".into(), 0));
    }
    digits = significant.to_owned();
    while digits.ends_with('0') {
        digits.pop();
        exponent = exponent.checked_add(1)?;
    }
    Some((negative, digits, exponent))
}

fn number(object: &Map<String, Value>, name: &str, expected: &str) -> Result<(), Error> {
    let actual = object
        .get(name)
        .and_then(Value::as_number)
        .and_then(|value| decimal(&value.to_string()));
    if actual.is_none() || actual != decimal(expected) {
        return Err(invalid(format!("{name} must equal decimal {expected}")));
    }
    Ok(())
}

fn optional_number(object: &Map<String, Value>, name: &str, expected: &str) -> Result<(), Error> {
    if object.contains_key(name) {
        number(object, name, expected)?;
    }
    Ok(())
}

fn dtype(object: &Map<String, Value>, required: bool) -> Result<(), Error> {
    let mut declared = false;
    for name in ["dtype", "torch_dtype"] {
        if object.contains_key(name) {
            string(object, name, "bfloat16")?;
            declared = true;
        }
    }
    if required && !declared {
        return Err(invalid("wrapper must declare BF16 dtype"));
    }
    Ok(())
}

fn metadata(object: &Map<String, Value>) -> Result<(), Error> {
    for name in ["transformers_version", "_name_or_path"] {
        if let Some(value) = object.get(name)
            && !value
                .as_str()
                .is_some_and(|text| !text.is_empty() && text.len() <= 1024)
        {
            return Err(invalid(format!("{name} must be a bounded export string")));
        }
    }
    if let Some(value) = object.get("unsloth_fixed")
        && value.as_bool().is_none()
    {
        return Err(invalid("unsloth_fixed must be a Boolean export flag"));
    }
    optional_number(object, "initializer_range", "0.02")
}

fn tokens(parent: &Map<String, Value>, text: &Map<String, Value>) -> Result<(), Error> {
    // Optional declarations are checked, never invented as a codec contract.
    for (name, expected) in [
        ("bos_token_id", 1),
        ("eos_token_id", 2),
        ("pad_token_id", 11),
    ] {
        for object in [parent, text] {
            if object.contains_key(name) {
                integer(object, name, expected)?;
            }
        }
    }
    Ok(())
}

fn yarn(value: &Value) -> Result<(), Error> {
    let rope = object(value, "text rope_parameters")?;
    whitelist(
        rope,
        &[
            "rope_type",
            "type",
            "rope_theta",
            "factor",
            "beta_fast",
            "beta_slow",
            "mscale",
            "mscale_all_dim",
            "original_max_position_embeddings",
            "llama_4_scaling_beta",
            "truncate",
        ],
        "text rope_parameters",
    )?;
    if !rope.contains_key("rope_type") && !rope.contains_key("type") {
        return Err(invalid("explicit yarn type is required"));
    }
    for name in ["rope_type", "type"] {
        if rope.contains_key(name) {
            string(rope, name, "yarn")?;
        }
    }
    for (name, expected) in [
        ("rope_theta", "1000000"),
        ("factor", "16"),
        ("beta_fast", "32"),
        ("beta_slow", "1"),
        ("mscale", "1"),
        ("mscale_all_dim", "1"),
        ("llama_4_scaling_beta", "0.1"),
    ] {
        number(rope, name, expected)?;
    }
    integer(rope, "original_max_position_embeddings", 16384)?;
    flag(rope, "truncate", true, false)
}

fn text(value: &Value) -> Result<(), Error> {
    let text = object(value, "text_config")?;
    whitelist(
        text,
        &[
            "model_type",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "max_position_embeddings",
            "hidden_act",
            "rms_norm_eps",
            "attention_dropout",
            "rope_parameters",
            "rope_theta",
            "rope_scaling",
            "rope_traditional",
            "rope_interleaved",
            "sliding_window",
            "use_sliding_window",
            "layer_types",
            "tie_word_embeddings",
            "use_cache",
            "attention_bias",
            "mlp_bias",
            "dtype",
            "torch_dtype",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "initializer_range",
            "transformers_version",
            "_name_or_path",
            "unsloth_fixed",
        ],
        "text_config",
    )?;
    string(text, "model_type", "ministral3")?;
    string(text, "hidden_act", "silu")?;
    for (name, expected) in [
        ("hidden_size", 3072),
        ("intermediate_size", 9216),
        ("num_hidden_layers", 26),
        ("num_attention_heads", 32),
        ("num_key_value_heads", 8),
        ("head_dim", 128),
        ("vocab_size", 131072),
        ("max_position_embeddings", 262144),
    ] {
        integer(text, name, expected)?;
    }
    number(text, "rms_norm_eps", "0.00001")?;
    number(text, "attention_dropout", "0")?;
    flag(text, "tie_word_embeddings", true, true)?;
    flag(text, "use_cache", true, true)?;
    for name in [
        "attention_bias",
        "mlp_bias",
        "rope_traditional",
        "rope_interleaved",
        "use_sliding_window",
    ] {
        flag(text, name, false, false)?;
    }
    for name in ["sliding_window", "rope_scaling"] {
        if text.get(name).is_some_and(|value| !value.is_null()) {
            return Err(invalid(format!("{name} must be absent or null")));
        }
    }
    if let Some(value) = text.get("layer_types")
        && !value.as_array().is_some_and(|layers| {
            layers.len() == 26
                && layers
                    .iter()
                    .all(|value| value.as_str() == Some("full_attention"))
        })
    {
        return Err(invalid("only 26 full_attention layers are selected"));
    }
    optional_number(text, "rope_theta", "1000000")?;
    yarn(
        text.get("rope_parameters")
            .ok_or_else(|| invalid("rope_parameters missing"))?,
    )?;
    dtype(text, false)?;
    metadata(text)
}

fn vision(value: &Value) -> Result<(), Error> {
    let vision = object(value, "vision_config")?;
    whitelist(
        vision,
        &[
            "model_type",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "head_dim",
            "num_channels",
            "image_size",
            "patch_size",
            "hidden_act",
            "attention_dropout",
            "initializer_range",
            "rope_parameters",
            "rope_theta",
            "dtype",
            "torch_dtype",
            "transformers_version",
            "_name_or_path",
        ],
        "vision metadata",
    )?;
    string(vision, "model_type", "pixtral")?;
    string(vision, "hidden_act", "silu")?;
    for (name, expected) in [
        ("hidden_size", 1024),
        ("intermediate_size", 4096),
        ("num_hidden_layers", 24),
        ("num_attention_heads", 16),
        ("head_dim", 64),
        ("num_channels", 3),
        ("image_size", 1540),
        ("patch_size", 14),
    ] {
        integer(vision, name, expected)?;
    }
    number(vision, "attention_dropout", "0")?;
    optional_number(vision, "rope_theta", "10000")?;
    let rope = object(
        vision
            .get("rope_parameters")
            .ok_or_else(|| invalid("vision rope_parameters missing"))?,
        "vision rope_parameters",
    )?;
    whitelist(rope, &["rope_type", "rope_theta"], "vision rope_parameters")?;
    string(rope, "rope_type", "default")?;
    number(rope, "rope_theta", "10000")?;
    dtype(vision, false)?;
    metadata(vision)
}

/// Validate selected semantics without mutating or retaining the document.
///
/// The caller owns bounded JSON parsing and duplicate-field rejection before
/// constructing `Value`; duplicate fields cannot be recovered from a `Value`.
/// Missing text/vision dtype inherits the required BF16 wrapper declaration;
/// every present dtype alias must itself be the identical supported string.
pub fn validate(config: &Value) -> Result<Profile, Error> {
    let parent = object(config, "wrapper")?;
    whitelist(
        parent,
        &[
            "model_type",
            "architectures",
            "text_config",
            "vision_config",
            "dtype",
            "torch_dtype",
            "image_token_index",
            "multimodal_projector_bias",
            "projector_hidden_act",
            "spatial_merge_size",
            "vision_feature_layer",
            "tie_word_embeddings",
            "use_cache",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "transformers_version",
            "_name_or_path",
            "unsloth_fixed",
            "initializer_range",
        ],
        "wrapper",
    )?;
    string(parent, "model_type", "mistral3")?;
    if !parent
        .get("architectures")
        .and_then(Value::as_array)
        .is_some_and(|classes| {
            classes.len() == 1 && classes[0].as_str() == Some("Mistral3ForConditionalGeneration")
        })
    {
        return Err(invalid(
            "selected Mistral3ForConditionalGeneration wrapper is required",
        ));
    }
    integer(parent, "image_token_index", 10)?;
    integer(parent, "spatial_merge_size", 2)?;
    if parent.get("vision_feature_layer").and_then(Value::as_i64) != Some(-1) {
        return Err(invalid("vision_feature_layer must be integer -1"));
    }
    flag(parent, "multimodal_projector_bias", false, true)?;
    string(parent, "projector_hidden_act", "gelu")?;
    flag(parent, "tie_word_embeddings", true, false)?;
    flag(parent, "use_cache", true, false)?;
    dtype(parent, true)?;
    metadata(parent)?;
    let text_config = parent
        .get("text_config")
        .ok_or_else(|| invalid("text_config missing"))?;
    text(text_config)?;
    tokens(parent, object(text_config, "text_config")?)?;
    vision(
        parent
            .get("vision_config")
            .ok_or_else(|| invalid("vision_config missing"))?,
    )?;
    Ok(Profile {
        geometry: Geometry {
            hidden_size: 3072,
            intermediate_size: 9216,
            layers: 26,
            query_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            query_width: 4096,
            kv_width: 1024,
            vocab_size: 131072,
            original_max_positions: 16384,
            source_max_positions: 262144,
        },
    })
}

#[cfg(test)]
mod tests;
