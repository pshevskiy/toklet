//! Cold text plan for the exact official Gemma 4 E2B BF16 shard.
use super::{
    Dtype, LAYERS, LayerPlan, MAX_HEADER_TENSORS, ModalityPolicy, PhysicalTensor, Shape,
    descriptor, expected_text_config, invalid, layer_plans, same_semantics, visit_text_tensors,
};
use crate::weights::Error;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub mod execution;

#[cfg(test)]
pub(crate) mod modified_parent;

#[cfg(unix)]
pub mod rows;

pub const CONTAINER_BYTES: u64 = 10_246_621_918;
const PREFIX: &str = "model.language_model.";
const PLE_TABLE: &str = "model.language_model.embed_tokens_per_layer.weight";
pub(crate) const CONFIG_BYTES: usize = 4_954;
const HEADER_BYTES: usize = 263_952;
pub(crate) const CONFIG_SHA256: &str =
    "1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330";
const HEADER_SHA256: &str = "12740d6fe7a66b316040fa4d77471a8e1809498a71992b3364a6d5417d10662e";

/// Checked shapes and relative payload ranges. The PLE table stays a distinct
/// source range. Byte totals are raw tensor spans, not a device memory budget.
pub struct TextPlan {
    layers: [LayerPlan; LAYERS],
    text: BTreeMap<String, PhysicalTensor>,
    media_bytes: usize,
    unused_shared_kv_bytes: usize,
    resident_tensor_bytes: usize,
}

impl TextPlan {
    pub fn layers(&self) -> &[LayerPlan; LAYERS] {
        &self.layers
    }

    pub fn text_tensors(&self) -> &BTreeMap<String, PhysicalTensor> {
        &self.text
    }

    pub fn payload_offset(&self) -> usize {
        HEADER_BYTES + 8
    }

    pub fn per_layer_embedding(&self) -> &PhysicalTensor {
        &self.text[PLE_TABLE]
    }

    pub fn media_bytes(&self) -> usize {
        self.media_bytes
    }

    pub fn unused_shared_kv_bytes(&self) -> usize {
        self.unused_shared_kv_bytes
    }

    pub fn resident_tensor_bytes(&self) -> usize {
        self.resident_tensor_bytes
    }
}

/// Only the pinned official single-shard artifact can use this public inspector.
/// Inspecting metadata neither reads nor authenticates the source payload.
pub fn inspect(
    config: &[u8],
    header: &[u8],
    container_bytes: u64,
    policy: ModalityPolicy,
) -> Result<TextPlan, Error> {
    if policy != ModalityPolicy::TextOnly {
        return Err(invalid("Gemma4 BF16 requires explicit text-only exclusion"));
    }
    if container_bytes != CONTAINER_BYTES
        || config.len() != CONFIG_BYTES
        || header.len() != HEADER_BYTES
        || format!("{:x}", Sha256::digest(config)) != CONFIG_SHA256
        || format!("{:x}", Sha256::digest(header)) != HEADER_SHA256
    {
        return Err(invalid("Gemma4 BF16 pinned source metadata differs"));
    }
    layout(
        crate::mlx::qwen::parse_json(config)?,
        header,
        container_bytes,
    )
}

fn layout(
    document: serde_json::Value,
    header: &[u8],
    container_bytes: u64,
) -> Result<TextPlan, Error> {
    if document["model_type"] != "gemma4"
        || document["architectures"] != json!(["Gemma4ForConditionalGeneration"])
        || !same_semantics(&document["text_config"], &expected_text_config())
    {
        return Err(invalid("Gemma4 BF16 text graph semantics differ"));
    }

    let header = crate::mlx::qwen::parse_json(header)?;
    let header = header
        .as_object()
        .filter(|h| h.len() <= MAX_HEADER_TENSORS + 1)
        .ok_or_else(|| invalid("Gemma4 BF16 bounded header object required"))?;
    if header.get("__metadata__") != Some(&json!({"format": "pt"})) {
        return Err(invalid("Gemma4 BF16 container format differs"));
    }
    let payload = usize::try_from(container_bytes - 8 - HEADER_BYTES as u64)
        .map_err(|_| invalid("Gemma4 BF16 payload size overflow"))?;
    let layers = layer_plans();
    let mut expected = BTreeMap::new();
    visit_text_tensors(&layers, PREFIX, |name, shape| {
        expected.insert(name, shape);
        Ok(())
    })?;

    let mut text = BTreeMap::new();
    let mut spans = Vec::new();
    spans
        .try_reserve_exact(header.len() - 1)
        .map_err(|_| Error::Allocation)?;
    let mut media = (0, 0_usize);
    let mut unused_shared = (0, 0_usize);
    let mut resident_bytes = 0_usize;
    for (name, value) in header {
        if name == "__metadata__" {
            continue;
        }
        let tensor = descriptor(value, payload)?;
        if tensor.dtype != Dtype::Bf16 || tensor.range.is_empty() {
            return Err(invalid("Gemma4 BF16 tensor encoding differs"));
        }
        spans.push(tensor.range.clone());
        let bytes = tensor.range.len();
        if let Some(shape) = expected.remove(name) {
            let matches = match shape {
                Shape::Matrix { rows, columns } => tensor.shape == [rows, columns],
                Shape::Vector(size) => tensor.shape == [size],
            };
            if !matches {
                return Err(invalid("Gemma4 BF16 selected text shape differs"));
            }
            if name != PLE_TABLE {
                resident_bytes = add_bytes(resident_bytes, bytes)?;
            }
            text.insert(name.clone(), tensor);
            continue;
        }
        let excluded = if unused_shared_kv(name) {
            &mut unused_shared
        } else if [
            "model.audio_tower.",
            "model.vision_tower.",
            "model.embed_audio.",
            "model.embed_vision.",
        ]
        .iter()
        .any(|prefix| name.starts_with(prefix))
        {
            &mut media
        } else {
            return Err(invalid("unrecognized excluded Gemma4 BF16 tensor"));
        };
        excluded.0 += 1;
        excluded.1 = add_bytes(excluded.1, bytes)?;
    }
    if !expected.is_empty() || text.len() != 540 || unused_shared.0 != 60 || media.0 != 1_411 {
        return Err(invalid("Gemma4 BF16 tensor inventory differs"));
    }
    spans.sort_unstable_by_key(|r| r.start);
    let mut cursor = 0;
    for span in spans {
        if span.start != cursor {
            return Err(invalid("Gemma4 BF16 payload gaps or overlaps"));
        }
        cursor = span.end;
    }
    if cursor != payload {
        return Err(invalid("Gemma4 BF16 payload coverage differs"));
    }
    Ok(TextPlan {
        layers,
        text,
        media_bytes: media.1,
        unused_shared_kv_bytes: unused_shared.1,
        resident_tensor_bytes: resident_bytes,
    })
}

fn add_bytes(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b)
        .ok_or_else(|| invalid("Gemma4 BF16 tensor byte count overflow"))
}

fn unused_shared_kv(name: &str) -> bool {
    let Some((layer, tensor)) = name
        .strip_prefix("model.language_model.layers.")
        .and_then(|name| name.split_once('.'))
    else {
        return false;
    };
    layer
        .parse::<usize>()
        .is_ok_and(|layer| (super::PRODUCERS..LAYERS).contains(&layer))
        && matches!(
            tensor,
            "self_attn.k_norm.weight" | "self_attn.k_proj.weight" | "self_attn.v_proj.weight"
        )
}

#[cfg(test)]
mod tests;
