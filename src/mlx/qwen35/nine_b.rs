//! Pinned9B cold text recipe with a distinct original output projection.
//!
//! Inspecting config/header bytes does not authenticate tensor payload, allocate
//! a model/session, admit a device, or qualify numerical execution or memory fit.
use super::{
    ConvolutionPlan, LayerRecipe, ModalityPolicy, PayloadBytes, PhysicalTensor, TensorPlan,
};
use crate::weights::Error;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const REPOSITORY: &str = "lukey03/Qwen3.5-9B-abliterated-MLX-4bit";
pub const REVISION: &str = "8ff61b6a3af49b4f3973a5ed8ff0937de812c0d3";
pub const FORMAT_VERSION: &str = "qwen35-9b-lukey-affine4-bf16-v1";
pub const CONTAINER_BYTES: u64 = 5_038_160_953;
/// Declared publisher identity; this metadata API does not read its payload.
pub const CONTAINER_SHA256: &str =
    "74855e1b61dc1955bdb835c2a3c9af8d6ce22aa9e46b1b99644fbda4afb540d2";
pub const CONFIG_BYTES: usize = 2431;
pub const CONFIG_SHA256: &str = "a765dc71b9fd04c2104920f0ce8c500c53a435250106fb833eba3ad6c8258a82";
pub const HEADER_BYTES: usize = 120881;
pub const HEADER_SHA256: &str = "af0f407b4caf8bcd824272e8bf20136a6bfd66b923619e6509c6c72b62cbc615";
const HIDDEN: usize = 4096;
const INTERMEDIATE: usize = 12288;
const HEAD: &str = "language_model.lm_head.weight";

pub mod payload;

pub struct Plan {
    layers: [LayerRecipe; super::LAYERS],
    common: Vec<TensorPlan>,
    convolutions: Vec<ConvolutionPlan>,
    physical: BTreeMap<String, PhysicalTensor>,
    bytes: PayloadBytes,
}
impl Plan {
    pub fn layers(&self) -> &[LayerRecipe; super::LAYERS] {
        &self.layers
    }
    ///250 affine matrices/153BF16 vectors in fixed recipe order. The24 original
    /// rank-three convolution owners are returned separately by `convolutions`.
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
    /// Absolute origin120,889 is not a typed-alignment guarantee.
    pub fn payload_offset(&self) -> usize {
        8 + HEADER_BYTES
    }
    pub fn embedding(&self) -> &TensorPlan {
        &self.common[0]
    }
    /// A separate physical source owner; never an alias of `embedding`.
    pub fn output_head(&self) -> &TensorPlan {
        &self.common[1]
    }
    pub fn hidden_size(&self) -> usize {
        HIDDEN
    }
    pub fn intermediate_size(&self) -> usize {
        INTERMEDIATE
    }
    pub fn vocab_size(&self) -> usize {
        super::VOCAB
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

/// Exact source metadata and explicit text-only policy. No payload, index,
/// numerical backend, context capacity or device budget is admitted here.
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
        return Err(super::invalid(
            "pinned Qwen3.5-9B metadata/text-only policy differs",
        ));
    }
    layout(config, header, container_bytes)
}

fn layout(config: &[u8], header: &[u8], container_bytes: u64) -> Result<Plan, Error> {
    if config.len() > 64 * 1024 || header.len() > 512 * 1024 {
        return Err(super::invalid("bounded Qwen3.5-9B metadata required"));
    }
    let mut expected = super::configuration::text(HIDDEN, INTERMEDIATE, false);
    let Value::Object(extra) = json!({
        "architectures":["Qwen3_5ForCausalLM"],
        "quantization":{"group_size":64,"bits":4,"mode":"affine"},
        "quantization_config":{"group_size":64,"bits":4,"mode":"affine"},
        "transformers_version":"5.2.0"
    }) else {
        unreachable!("literal source config")
    };
    expected.extend(extra);
    if !super::same_semantics(
        &super::super::qwen::parse_json(config)?,
        &Value::Object(expected),
    ) {
        return Err(super::invalid("Qwen3.5-9B source graph options differ"));
    }
    let payload = container_bytes
        .checked_sub((8 + HEADER_BYTES) as u64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| super::invalid("Qwen3.5-9B payload size overflow"))?;
    let (layers, common, convolutions) =
        super::recipe::build_for(HIDDEN, INTERMEDIATE, Some(HEAD))?;
    let (physical, bytes) = super::physical_metadata(header, payload, &common, &convolutions)?;
    Ok(Plan {
        layers,
        common,
        convolutions,
        physical,
        bytes,
    })
}

#[cfg(all(test, target_pointer_width = "64"))]
mod tests;
