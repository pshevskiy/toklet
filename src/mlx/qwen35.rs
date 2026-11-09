//! Pinned Qwen3.5-4B cold text recipe; no numerical or model admission.
//!
//! Recurrent matrices, convolution history and attention KV are distinct state
//! kinds. Source ranges remain bytes, including an unaligned payload origin.
use super::affine::{Layout, MetadataDtype};
pub use super::tensor_metadata::{Dtype, PhysicalTensor, Shape, Storage, TensorPlan};
use super::tensor_metadata::{descriptor, expected_physical, same_semantics};
use crate::weights::Error;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

mod configuration;
pub mod nine_b;
pub mod payload;
mod recipe;
#[cfg(test)]
mod tests;

pub const REPOSITORY: &str = "avan-ag/Qwen3.5-4B-Uncensored-MLX-4bit";
pub const REVISION: &str = "fb98fbc59c71a567f3e23435386c9e93951261db";
pub const FORMAT_VERSION: &str = "qwen35-4b-avan-affine4-bf16-v1";
pub const CONTAINER_BYTES: u64 = 2_367_237_149;
/// Declared Hub digest only; inspecting metadata does not authenticate payload.
pub const CONTAINER_SHA256: &str =
    "0dfd6217f56cadcc373c90283d58d5e673833650fed1d6c299313fb536999043";
pub const CONFIG_BYTES: usize = 3024;
pub const CONFIG_SHA256: &str = "ac6f8df4f4c1fd9d2583996bdd7070ea801b703e8a2b36f6d4b673a94ea419b5";
pub const HEADER_BYTES: usize = 119829;
pub const HEADER_SHA256: &str = "7284f68df36a7761569b48321066e92d616306e33b4a9487029673a0ff344cb2";
const HIDDEN: usize = 2560;
const INTERMEDIATE: usize = 9216;
const VOCAB: usize = 248320;
const LAYERS: usize = 32;
const PREFIX: &str = "language_model.model.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModalityPolicy {
    Reject,
    TextOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeltaPlan {
    pub recurrent_owner: usize,
    pub convolution_owner: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionPlan {
    pub kv_owner: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotated_dims: usize,
    pub output_gate: bool,
}

/// Owner indices are local to their state kind; a delta state is not a KV layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerRecipe {
    GatedDelta(DeltaPlan),
    GatedFullAttention(AttentionPlan),
}

/// Original BF16 `[channels, 4, 1]`. No transpose or convolution is executed.
#[derive(Debug)]
pub struct ConvolutionPlan {
    pub layer: usize,
    pub history_owner: usize,
    pub name: String,
    pub channels: usize,
    pub taps: usize,
}

/// Original tensor spans only, excluding headers and all future runtime owners.
/// These are neither allocated bytes nor a model/session admission budget.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PayloadBytes {
    pub affine_codes: usize,
    pub affine_metadata: usize,
    pub bf16_vectors: usize,
    pub bf16_convolutions: usize,
    pub total: usize,
}

pub struct Plan {
    layers: [LayerRecipe; LAYERS],
    common: Vec<TensorPlan>,
    convolutions: Vec<ConvolutionPlan>,
    physical: BTreeMap<String, PhysicalTensor>,
    bytes: PayloadBytes,
}
impl Plan {
    pub fn layers(&self) -> &[LayerRecipe; LAYERS] {
        &self.layers
    }
    /// 249 affine matrices and 153 original BF16 vectors, in fixed recipe order.
    pub fn logical_tensors(&self) -> &[TensorPlan] {
        &self.common
    }
    pub fn convolutions(&self) -> &[ConvolutionPlan] {
        &self.convolutions
    }
    pub fn logical_owner_count(&self) -> usize {
        self.common.len() + self.convolutions.len()
    }
    pub fn physical_tensors(&self) -> &BTreeMap<String, PhysicalTensor> {
        &self.physical
    }
    pub fn payload_bytes(&self) -> PayloadBytes {
        self.bytes
    }
    /// Absolute byte origin119,837 is deliberately not a typed alignment promise.
    pub fn payload_offset(&self) -> usize {
        HEADER_BYTES + 8
    }
    pub fn embedding(&self) -> &TensorPlan {
        &self.common[0]
    }
    /// The tied head aliases the embedding rather than adding a physical owner.
    pub fn tied_head(&self) -> &TensorPlan {
        self.embedding()
    }
    pub fn hidden_size(&self) -> usize {
        HIDDEN
    }
    pub fn intermediate_size(&self) -> usize {
        INTERMEDIATE
    }
    pub fn vocab_size(&self) -> usize {
        VOCAB
    }
    pub fn declared_total_positions(&self) -> usize {
        262144
    }
    pub fn implemented_total_positions(&self) -> Option<usize> {
        None
    }
    pub fn qualified_total_positions(&self) -> Option<usize> {
        None
    }
    pub fn declared_recurrent_state_dtype(&self) -> &'static str {
        "float32"
    }
    pub fn rope_sections(&self) -> [usize; 3] {
        [11, 11, 10]
    }
    pub fn rope_theta(&self) -> f64 {
        10_000_000.0
    }
}

/// One fixed single-shard MLX source and an explicit text-only request.
/// No index, tensor bytes, numerical backend or device capability is admitted.
pub fn inspect(
    config: &[u8],
    header: &[u8],
    container_bytes: u64,
    policy: ModalityPolicy,
) -> Result<Plan, Error> {
    if policy != ModalityPolicy::TextOnly
        || container_bytes != CONTAINER_BYTES
        || config.len() != CONFIG_BYTES
        || header.len() != HEADER_BYTES
        || format!("{:x}", Sha256::digest(config)) != CONFIG_SHA256
        || format!("{:x}", Sha256::digest(header)) != HEADER_SHA256
    {
        return Err(invalid(
            "pinned Qwen3.5-4B metadata/text-only policy differs",
        ));
    }
    layout(config, header, container_bytes)
}

fn layout(config: &[u8], header: &[u8], container_bytes: u64) -> Result<Plan, Error> {
    if config.len() > 64 * 1024 || header.len() > 512 * 1024 {
        return Err(invalid("bounded Qwen3.5 metadata required"));
    }
    configuration::validate(config)?;
    let payload = container_bytes
        .checked_sub((8 + HEADER_BYTES) as u64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| invalid("Qwen3.5 payload size overflow"))?;
    let (layers, common, convolutions) = recipe::build()?;
    let (physical, bytes) = physical_metadata(header, payload, &common, &convolutions)?;
    Ok(Plan {
        layers,
        common,
        convolutions,
        physical,
        bytes,
    })
}

// Both pinned sources use the same physical packing. Their public inspectors
// keep separate identity/config/shape factories and execution eligibility.
fn physical_metadata(
    header: &[u8],
    payload: usize,
    common: &[TensorPlan],
    convolutions: &[ConvolutionPlan],
) -> Result<(BTreeMap<String, PhysicalTensor>, PayloadBytes), Error> {
    let mut expected = expected_physical(common);
    for c in convolutions {
        expected.insert(
            c.name.clone(),
            (
                Dtype::Bf16,
                vec![c.channels, c.taps, 1],
                c.channels * c.taps * 2,
            ),
        );
    }
    let document = super::qwen::parse_json(header)?;
    let header = document
        .as_object()
        .filter(|v| v.len() == expected.len() + 1)
        .ok_or_else(|| invalid("exact Qwen3.5 physical tensor inventory required"))?;
    if header.get("__metadata__") != Some(&json!({"format":"mlx"})) {
        return Err(invalid("Qwen3.5 container metadata differs"));
    }
    let mut physical = BTreeMap::new();
    let mut spans = Vec::new();
    let mut bytes = PayloadBytes::default();
    for (name, value) in header {
        if name == "__metadata__" {
            continue;
        }
        let (dtype, shape, size) = expected
            .remove(name)
            .ok_or_else(|| invalid("unexpected Qwen3.5 tensor"))?;
        let tensor = descriptor(value, payload)?;
        if tensor.dtype != dtype || tensor.shape != shape || tensor.range.len() != size {
            return Err(invalid("Qwen3.5 tensor representation/shape differs"));
        }
        let bucket = if dtype == Dtype::U32 {
            &mut bytes.affine_codes
        } else if name.ends_with(".scales") || name.ends_with(".biases") {
            &mut bytes.affine_metadata
        } else if name.ends_with(".conv1d.weight") {
            &mut bytes.bf16_convolutions
        } else {
            &mut bytes.bf16_vectors
        };
        *bucket = add(*bucket, size)?;
        bytes.total = add(bytes.total, size)?;
        spans.push(tensor.range.clone());
        physical.insert(name.clone(), tensor);
    }
    if !expected.is_empty() {
        return Err(invalid("missing Qwen3.5 tensor"));
    }
    spans.sort_unstable_by_key(|r| r.start);
    let mut cursor = 0;
    for span in spans {
        if span.start != cursor {
            return Err(invalid("Qwen3.5 tensor spans gap or overlap"));
        }
        cursor = span.end;
    }
    if cursor != payload || bytes.total != payload {
        return Err(invalid("Qwen3.5 payload coverage differs"));
    }
    Ok((physical, bytes))
}

fn invalid(reason: &str) -> Error {
    Error::Invalid(reason.into())
}
fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b)
        .ok_or_else(|| invalid("Qwen3.5 byte count overflow"))
}
