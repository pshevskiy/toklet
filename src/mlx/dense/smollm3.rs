//! SmolLM3's explicit per-layer RoPE mask. All attention remains full causal.
use super::{Error, Value, invalid};

pub(super) fn prepare(value: &mut Value) -> Result<Vec<bool>, Error> {
    let object = value
        .as_object_mut()
        .ok_or_else(|| invalid("SmolLM3 config object required"))?;
    let layers = object
        .get("num_hidden_layers")
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0 && n <= 682)
        .ok_or_else(|| invalid("SmolLM3 layer count exceeds source name-map bound"))?;
    let mask = object
        .remove("no_rope_layers")
        .ok_or_else(|| invalid("explicit SmolLM3 rotary layer mask required"))?;
    let mask = mask
        .as_array()
        .filter(|m| m.len() == layers)
        .ok_or_else(|| invalid("SmolLM3 rotary mask length differs"))?;
    // Despite its upstream name, 1 enables RoPE and 0 bypasses it. An explicit
    // mask takes precedence over the interval in stock MLX; do not infer a pattern.
    let mask = mask
        .iter()
        .map(|v| match v.as_u64() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err(invalid("SmolLM3 rotary mask requires integer zero/one")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if object
        .remove("no_rope_layer_interval")
        .is_some_and(|v| v.as_u64().is_none_or(|n| n == 0))
    {
        return Err(invalid(
            "SmolLM3 rotary interval must be a positive integer",
        ));
    }
    let types = object
        .remove("layer_types")
        .ok_or_else(|| invalid("explicit SmolLM3 attention types required"))?;
    if !types
        .as_array()
        .is_some_and(|v| v.len() == layers && v.iter().all(|t| t == "full_attention"))
    {
        return Err(invalid(
            "SmolLM3 supports only full causal attention layers",
        ));
    }
    if object
        .remove("max_window_layers")
        .is_some_and(|v| v.as_u64().is_none_or(|n| n > layers as u64))
    {
        return Err(invalid("invalid SmolLM3 full-attention export metadata"));
    }
    // This converted MLX export omits the field and stores one tied table.
    // Source's normal tied-output validation must still reject an absent or
    // incompatible tensor; no output weights are synthesized here.
    object
        .entry("tie_word_embeddings")
        .or_insert(Value::Bool(true));
    Ok(mask)
}
