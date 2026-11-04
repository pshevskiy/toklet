//! Granite changes scalar boundaries, not the dense weight or attention layout.
use super::{Error, Value, invalid};
use crate::model::DenseScales;

pub(super) fn prepare(value: &mut Value) -> Result<DenseScales, Error> {
    let object = value
        .as_object_mut()
        .ok_or_else(|| invalid("Granite config object required"))?;
    let mut scalar = |name| {
        object
            .remove(name)
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .filter(|v| v.is_normal() && *v > 0.)
            .ok_or_else(|| invalid(format!("Granite requires positive normal {name}")))
    };
    let scales = DenseScales {
        embedding: scalar("embedding_multiplier")?,
        attention: Some(scalar("attention_multiplier")?),
        residual: scalar("residual_multiplier")?,
        logits_divisor: scalar("logits_scaling")?,
    };
    scales.validate()?;
    Ok(scales)
}
