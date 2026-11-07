//! Pinned cold recipe for the selected Heretic MLX4 checkpoint.
//!
//! This describes original affine4/group64/BF16 matrices and the one original
//! BF16 PLE projection. It grants no payload integrity, memory admission,
//! numerical execution, quality or managed-app eligibility.
use super::*;

pub const REPOSITORY: &str = "deadbydawn101/gemma-4-E2B-Heretic-Uncensored-mlx-4bit";
pub const REVISION: &str = "cde01efcb11f8a12e88a953b752fc16062aa1486";
pub const FORMAT_VERSION: &str = "gemma4-heretic-affine4-bf16-ple-v1";
pub const CONTAINER_BYTES: u64 = 3_581_100_878;
/// Expected whole-file digest; inspection does not read or authenticate payload.
pub const CONTAINER_SHA256: &str =
    "d4a2995ce3fdafbbd4a3376bc8524a8d26bcbf0f51197701b4e21ab27faa37e1";
const CONFIG_BYTES: usize = 6035;
const HEADER_BYTES: usize = 335744;
const CONFIG_SHA256: &str = "7ff08145776f892538f06b2e6b574a62080959d3097c37e4f8a45664f4478da1";
const HEADER_SHA256: &str = "e12e790370839d89a88cb1ead5e27525011ed3f39a3879f6c2a7162080f4dc50";
const BF16_PROJECTION: &str = "language_model.model.per_layer_model_projection.weight";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixStorage {
    Affine(Layout),
    Bf16 { rows: usize, columns: usize },
}

#[derive(Debug)]
pub struct MatrixPlan {
    pub name: String,
    pub storage: MatrixStorage,
}

pub struct Plan {
    layers: [LayerPlan; LAYERS],
    matrices: Vec<MatrixPlan>,
    text: BTreeMap<String, PhysicalTensor>,
    unused_shared_kv: BTreeMap<String, PhysicalTensor>,
    media: BTreeMap<String, PhysicalTensor>,
    text_bytes: usize,
    unused_shared_kv_bytes: usize,
    media_bytes: usize,
}

impl Plan {
    pub fn layers(&self) -> &[LayerPlan; LAYERS] {
        &self.layers
    }
    pub fn matrices(&self) -> &[MatrixPlan] {
        &self.matrices
    }
    pub fn text_tensors(&self) -> &BTreeMap<String, PhysicalTensor> {
        &self.text
    }
    pub fn unused_shared_kv_tensors(&self) -> &BTreeMap<String, PhysicalTensor> {
        &self.unused_shared_kv
    }
    pub fn media_tensors(&self) -> &BTreeMap<String, PhysicalTensor> {
        &self.media
    }
    pub fn text_payload_bytes(&self) -> usize {
        self.text_bytes
    }
    pub fn unused_shared_kv_bytes(&self) -> usize {
        self.unused_shared_kv_bytes
    }
    pub fn media_bytes(&self) -> usize {
        self.media_bytes
    }
    /// Single `model.safetensors` payload start; offsets in tensors are relative.
    pub fn payload_offset(&self) -> usize {
        8 + HEADER_BYTES
    }
}

/// Inspect only the exact pinned config/header and explicit text-only policy.
///
/// No index is consumed: this recipe names one fixed `model.safetensors` file.
/// A later loader must authenticate its consumed payload against the whole-file
/// digest, validate coefficient values and admit all actual owners/temporaries.
pub fn inspect(
    config_bytes: &[u8],
    header_bytes: &[u8],
    container_bytes: u64,
    policy: ModalityPolicy,
) -> Result<Plan, Error> {
    if policy != ModalityPolicy::TextOnly
        || container_bytes != CONTAINER_BYTES
        || config_bytes.len() != CONFIG_BYTES
        || header_bytes.len() != HEADER_BYTES
        || format!("{:x}", Sha256::digest(config_bytes)) != CONFIG_SHA256
        || format!("{:x}", Sha256::digest(header_bytes)) != HEADER_SHA256
    {
        return Err(invalid(
            "pinned Heretic MLX4 metadata or text-only policy differs",
        ));
    }
    layout(config_bytes, header_bytes, container_bytes)
}

fn configuration(bytes: &[u8]) -> Result<(), Error> {
    let mut document = super::super::qwen::parse_json(bytes)?;
    let text = document
        .as_object_mut()
        .and_then(|d| d.get_mut("text_config"))
        .and_then(Value::as_object_mut)
        .ok_or_else(|| invalid("Heretic text configuration object required"))?;
    // This checkpoint adds exactly one null alias to an explicitly disabled
    // MoE branch. Other graph options still pass the existing exact validator.
    if text.remove("moe_intermediate_size") != Some(Value::Null) {
        return Err(invalid("Heretic inactive MoE alias differs"));
    }
    config(&document, ModalityPolicy::TextOnly, Format::Affine4)
}

fn unused_logical(layers: &[LayerPlan; LAYERS]) -> Result<Vec<TensorPlan>, Error> {
    let mut tensors = Vec::new();
    for (index, layer) in layers.iter().enumerate().skip(PRODUCERS) {
        for role in ["k_proj", "v_proj"] {
            tensors.push(TensorPlan {
                name: format!("language_model.model.layers.{index}.self_attn.{role}.weight"),
                shape: Shape::Matrix {
                    rows: layer.head_dim,
                    columns: HIDDEN,
                },
                storage: Storage::Affine(
                    Layout::new(layer.head_dim, HIDDEN, 4, 64, MetadataDtype::BF16)
                        .map_err(|e| invalid(e.0))?,
                ),
            });
        }
        tensors.push(TensorPlan {
            name: format!("language_model.model.layers.{index}.self_attn.k_norm.weight"),
            shape: Shape::Vector(layer.head_dim),
            storage: Storage::Bf16Vector,
        });
    }
    Ok(tensors)
}

fn layout(config_bytes: &[u8], header_bytes: &[u8], container_bytes: u64) -> Result<Plan, Error> {
    configuration(config_bytes)?;
    let payload = usize::try_from(container_bytes)
        .ok()
        .and_then(|n| n.checked_sub(8))
        .and_then(|n| n.checked_sub(header_bytes.len()))
        .ok_or_else(|| invalid("Heretic container size differs"))?;
    let document = super::super::qwen::parse_json(header_bytes)?;
    let header = document
        .as_object()
        .ok_or_else(|| invalid("Heretic tensor header required"))?;
    if header.get("__metadata__") != Some(&json!({"format":"mlx"})) || header.len() != 2650 {
        return Err(invalid("Heretic tensor count or metadata differs"));
    }
    let layers = layer_plans();
    let logical = logical(&layers, Format::Affine4)?;
    let mut expected = expected_physical(&logical);
    expected.remove("language_model.model.per_layer_model_projection.scales");
    expected.remove("language_model.model.per_layer_model_projection.biases");
    expected.insert(
        BF16_PROJECTION.into(),
        (
            Dtype::Bf16,
            vec![LAYERS * PLE, HIDDEN],
            LAYERS * PLE * HIDDEN * 2,
        ),
    );
    let unused = expected_physical(&unused_logical(&layers)?);
    let matrices = logical
        .into_iter()
        .filter_map(|t| match t.storage {
            Storage::Affine(l) => Some(MatrixPlan {
                storage: if t.name == BF16_PROJECTION {
                    MatrixStorage::Bf16 {
                        rows: LAYERS * PLE,
                        columns: HIDDEN,
                    }
                } else {
                    MatrixStorage::Affine(l)
                },
                name: t.name,
            }),
            Storage::Bf16Vector => None,
        })
        .collect();
    let mut plan = Plan {
        layers,
        matrices,
        text: BTreeMap::new(),
        unused_shared_kv: BTreeMap::new(),
        media: BTreeMap::new(),
        text_bytes: 0,
        unused_shared_kv_bytes: 0,
        media_bytes: 0,
    };
    let mut spans = Vec::new();
    for (name, value) in header {
        if name == "__metadata__" {
            continue;
        }
        if name.len() > 256 {
            return Err(invalid("Heretic tensor name exceeds bound"));
        }
        let tensor = descriptor(value, payload)?;
        let bytes = tensor.range.len();
        spans.push(tensor.range.clone());
        let (selected, sum) =
            if let Some((dtype, shape, size)) = expected.get(name).or_else(|| unused.get(name)) {
                if tensor.dtype != *dtype || tensor.shape != *shape || bytes != *size {
                    return Err(invalid("Heretic text tensor representation differs"));
                }
                if unused.contains_key(name) {
                    (&mut plan.unused_shared_kv, &mut plan.unused_shared_kv_bytes)
                } else {
                    (&mut plan.text, &mut plan.text_bytes)
                }
            } else {
                let root = name.split('.').next().unwrap_or("");
                if !["audio_tower", "vision_tower", "embed_audio", "embed_vision"].contains(&root) {
                    return Err(invalid("unexpected Heretic tensor"));
                }
                (&mut plan.media, &mut plan.media_bytes)
            };
        *sum = sum
            .checked_add(bytes)
            .ok_or_else(|| invalid("Heretic payload sum overflow"))?;
        selected.insert(name.clone(), tensor);
    }
    if plan.text.len() != expected.len()
        || plan.unused_shared_kv.len() != unused.len()
        || plan.media.len() != 1415
    {
        return Err(invalid("incomplete Heretic tensor inventory"));
    }
    spans.sort_unstable_by_key(|s| (s.start, s.end));
    let mut cursor = 0;
    for span in spans {
        if span.start != cursor {
            return Err(invalid("Heretic payload gaps or overlap"));
        }
        cursor = span.end;
    }
    if cursor != payload {
        return Err(invalid("Heretic payload coverage differs"));
    }
    Ok(plan)
}

#[cfg(test)]
mod tests;
