//! Granite changes scalar boundaries, not the dense weight or attention layout.
use super::{Error, Value, invalid};
use crate::model::DenseScales;

pub(super) fn prepare(value: &mut Value) -> Result<DenseScales, Error> {
    let object = value
        .as_object_mut()
        .ok_or_else(|| invalid("Granite config object required"))?;
    // Some exports carry both spellings. Admit only the same supported string,
    // before serde's single-field alias handling; never choose between conflicts.
    if let (Some(dtype), Some(torch_dtype)) = (object.get("dtype"), object.get("torch_dtype")) {
        if dtype != torch_dtype
            || !dtype
                .as_str()
                .is_some_and(|v| ["float32", "float16", "bfloat16"].contains(&v))
        {
            return Err(invalid(
                "Granite dtype declarations must be identical supported strings",
            ));
        }
        object.remove("dtype");
    }
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

#[cfg(test)]
mod tests {
    use super::super::{Architecture, config};
    use crate::{mlx::qwen::parse_json, model::PositionPlan, weights::LoadOptions};
    use serde_json::Value;
    use sha2::{Digest, Sha256};

    #[test]
    fn pinned_granite_configs_preserve_geometry_scales_and_context_ceiling() {
        let manifest: Value = serde_json::from_slice(include_bytes!(
            "../../../tests/fixtures/granite-config-v1/manifest.json"
        ))
        .unwrap();
        for (name, raw) in [
            (
                "granite_original",
                include_bytes!("../../../tests/fixtures/granite-config-v1/granite_original.json")
                    .as_slice(),
            ),
            (
                "granite_modified",
                include_bytes!("../../../tests/fixtures/granite-config-v1/granite_modified.json")
                    .as_slice(),
            ),
            (
                "granite_q4",
                include_bytes!("../../../tests/fixtures/granite-config-v1/granite_q4.json")
                    .as_slice(),
            ),
        ] {
            let pin = &manifest["sources"][name];
            assert_eq!(raw.len() as u64, pin["bytes"]);
            assert_eq!(format!("{:x}", Sha256::digest(raw)), pin["sha256"]);
            let mut document = parse_json(raw).unwrap();
            // The ordinary source boundary consumes the affine plan before
            // dense configuration. This is a configuration-only test.
            let object = document.as_object_mut().unwrap();
            let quantization = object.remove("quantization");
            assert_eq!(quantization, object.remove("quantization_config"));
            for context_limit in [1, 1024, 32768] {
                let parsed = config(
                    document.clone(),
                    LoadOptions {
                        context_limit,
                        ..Default::default()
                    },
                )
                .unwrap();
                let c = parsed.config;
                assert_eq!(parsed.architecture, Architecture::Granite);
                assert!(!parsed.tied_output);
                assert_eq!(parsed.position_plan, PositionPlan::Ordinary);
                assert_eq!(
                    (c.hidden_size, c.intermediate_size, c.num_layers),
                    (2560, 8192, 40)
                );
                assert_eq!((c.query_heads, c.kv_heads, c.head_dim), (40, 8, 64));
                assert_eq!(c.vocab_size, 100352);
                assert_eq!(c.max_positions, context_limit);
                assert_eq!(c.rope_theta, 10_000_000.);
                assert_eq!(c.rms_norm_eps, 1e-5);
                assert_eq!(parsed.scales.embedding, 1.);
                assert_eq!(parsed.scales.attention, Some(0.015625));
                assert_eq!(parsed.scales.residual, 1.);
                assert_eq!(parsed.scales.logits_divisor, 1.);
            }
            for context_limit in [0, 32769, 131072, usize::MAX] {
                assert!(
                    config(
                        document.clone(),
                        LoadOptions {
                            context_limit,
                            ..Default::default()
                        }
                    )
                    .is_err()
                );
            }
        }
    }
}
