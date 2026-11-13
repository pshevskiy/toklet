//! Selected packed Ministral source planning for one explicit Metal diagnostic.
//! Source bytes/configuration stay untouched. This module owns no numerical
//! weights, positional tables, tokenizer, session or application admission.

use super::*;
use safetensors::tensor::TensorInfo;

const CONTEXT_LIMIT: usize = 512;
const PAYLOAD_BYTES: usize = 2_662_004_736;

#[derive(Clone)]
struct PhysicalTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: usize,
}

struct Plan {
    config: Config,
    profile: crate::mlx::ministral3::Profile,
    tensors: BTreeMap<String, TensorSpec>,
    physical: BTreeMap<String, PhysicalTensor>,
    admitted_load_bytes: usize,
}

fn quantization() -> Value {
    let mut recipe = Map::from_iter([
        ("bits".into(), Value::from(4)),
        ("group_size".into(), Value::from(64)),
        ("mode".into(), Value::from("affine")),
    ]);
    let affine8 = || serde_json::json!({"bits":8,"group_size":64,"mode":"affine"});
    recipe.insert("language_model.model.embed_tokens".into(), affine8());
    for layer in 0..26 {
        for role in ["self_attn.o_proj", "mlp.down_proj"] {
            recipe.insert(
                format!("language_model.model.layers.{layer}.{role}"),
                affine8(),
            );
        }
    }
    Value::Object(recipe)
}

fn plan(mut document: Value, options: LoadOptions) -> Result<Plan, Error> {
    if options.context_limit == 0 || options.context_limit > CONTEXT_LIMIT {
        return Err(invalid("Ministral diagnostic requires1..=512 positions"));
    }
    Source::validate_budget(options)?;
    let object = document
        .as_object_mut()
        .ok_or_else(|| invalid("Ministral diagnostic config must be an object"))?;
    let supplied = object.remove("quantization");
    if object.contains_key("quantization_config") || supplied.as_ref() != Some(&quantization()) {
        return Err(invalid(
            "Ministral diagnostic requires the exact selected mixed-affine recipe",
        ));
    }
    // Parse the supplied declaration too: integer fields must not be accepted
    // through floating-point or Boolean equality with the selected literals.
    let selected = QuantizationPlan::parse(supplied.unwrap())?;
    let profile = crate::mlx::ministral3::validate(&document)?;
    let geometry = profile.geometry();
    let config = Config {
        vocab_size: geometry.vocab_size,
        hidden_size: geometry.hidden_size,
        intermediate_size: geometry.intermediate_size,
        num_layers: geometry.layers,
        query_heads: geometry.query_heads,
        kv_heads: geometry.kv_heads,
        head_dim: geometry.head_dim,
        max_positions: options.context_limit,
        rms_norm_eps: 1e-5,
        rope_theta: 1e6,
    };
    config.validate()?;
    let expected = weights::expected(&config)?;
    let mut physical = BTreeMap::new();
    let mut tensors = BTreeMap::new();
    let mut payload = 0;
    for (graph_name, shape) in expected {
        if graph_name.ends_with(".bias") {
            continue;
        }
        let name = format!("language_model.{graph_name}");
        let (storage, payload_bytes) = match shape.as_slice() {
            [rows, columns] => {
                let module = name.strip_suffix(".weight").unwrap();
                let parameters = selected
                    .resolve(module)
                    .ok_or_else(|| invalid("selected Ministral packed matrix excluded"))?;
                let layout = Layout::new(
                    *rows,
                    *columns,
                    parameters.bits,
                    parameters.group_size,
                    MetadataDtype::BF16,
                )
                .map_err(|error| invalid(error.to_string()))?;
                physical.insert(
                    name.clone(),
                    PhysicalTensor {
                        dtype: Dtype::U32,
                        shape: vec![*rows, layout.row_bytes() / 4],
                        bytes: layout.weight_bytes(),
                    },
                );
                for field in ["scales", "biases"] {
                    physical.insert(
                        format!("{module}.{field}"),
                        PhysicalTensor {
                            dtype: Dtype::BF16,
                            shape: vec![*rows, *columns / 64],
                            bytes: layout.metadata_bytes(),
                        },
                    );
                }
                (Storage::Affine(layout), layout.total_bytes())
            }
            [width] => {
                let bytes = width
                    .checked_mul(2)
                    .ok_or_else(|| invalid("Ministral norm byte overflow"))?;
                physical.insert(
                    name.clone(),
                    PhysicalTensor {
                        dtype: Dtype::BF16,
                        shape: shape.clone(),
                        bytes,
                    },
                );
                (Storage::Float(MetadataDtype::BF16), bytes)
            }
            _ => return Err(invalid("Ministral text tensor rank differs")),
        };
        payload = add(payload, payload_bytes)?;
        tensors.insert(
            name,
            TensorSpec {
                shape,
                storage,
                payload_bytes,
            },
        );
    }
    if physical.len() != 602 || tensors.len() != 236 || payload != PAYLOAD_BYTES {
        return Err(invalid("Ministral selected physical inventory differs"));
    }
    let admitted_load_bytes = add(add(payload, METADATA_ALLOWANCE)?, STREAM_BYTES)?;
    if admitted_load_bytes > options.memory_limit_bytes {
        return Err(Error::Budget {
            required: admitted_load_bytes,
            limit: options.memory_limit_bytes,
        });
    }
    Ok(Plan {
        config,
        profile,
        tensors,
        physical,
        admitted_load_bytes,
    })
}

fn validate_names<'a>(
    expected: &BTreeMap<String, PhysicalTensor>,
    names: impl Iterator<Item = &'a str>,
) -> Result<(), Error> {
    let mut actual = BTreeSet::new();
    for name in names {
        if !actual.insert(name) {
            return Err(invalid("duplicate Ministral physical tensor"));
        }
    }
    if actual.len() != expected.len() || expected.keys().any(|name| !actual.contains(name.as_str()))
    {
        return Err(invalid(
            "Ministral diagnostic physical tensor inventory differs",
        ));
    }
    Ok(())
}

fn validate_tensor(name: &str, expected: &PhysicalTensor, info: &TensorInfo) -> Result<(), Error> {
    if info.dtype != expected.dtype
        || info.shape != expected.shape
        || info.data_offsets.1.checked_sub(info.data_offsets.0) != Some(expected.bytes)
    {
        return Err(invalid(format!(
            "Ministral diagnostic physical shape/dtype/extent differs: {name}"
        )));
    }
    Ok(())
}

pub(super) fn open(directory: &Path, options: LoadOptions) -> Result<Source, Error> {
    // Reject unsupported capacities/budgets before even opening configuration.
    if options.context_limit == 0 || options.context_limit > CONTEXT_LIMIT {
        return Err(invalid("Ministral diagnostic requires1..=512 positions"));
    }
    Source::validate_budget(options)?;
    let planned = plan(read_config(&directory.join("config.json"))?, options)?;
    let source = weights::Source::open_with_audit(directory, None)?;
    validate_names(&planned.physical, source.names())?;
    for (name, expected) in &planned.physical {
        let info = source
            .info(name)
            .ok_or_else(|| invalid("validated Ministral physical tensor absent"))?;
        validate_tensor(name, expected, info)?;
    }
    let logical = logical::ministral3_aliases(&planned.tensors)?;
    Ok(Source {
        config: planned.config,
        declared_positions: planned.profile.geometry().source_max_positions,
        declared_sliding_window: None,
        architecture: Architecture::Mistral,
        tied_output: true,
        // The optional selected profile is authoritative for this Half graph.
        // Legacy CPU PositionPlan is deliberately unchanged and not YaRN.
        position_plan: crate::model::PositionPlan::Ordinary,
        ministral3_profile: Some(planned.profile),
        rotary_layers: None,
        scales: Default::default(),
        tensors: planned.tensors,
        logical,
        source,
        tensor_payload_bytes: PAYLOAD_BYTES,
        admitted_load_bytes: planned.admitted_load_bytes,
    })
}

#[cfg(test)]
mod tests;
