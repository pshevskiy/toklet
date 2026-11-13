//! Explicit2B diagnostic child structure. External artifact pins authenticate
//! bytes; this cold API does not qualify parent lineage or value transforms.
use super::{
    AttentionPlan, ConvolutionPlan, DeltaPlan, Dtype, LayerRecipe, ModalityPolicy, PhysicalTensor,
    Shape, Storage, TensorPlan, add, invalid, same_semantics,
};
use crate::{
    mlx::{
        affine::{Layout, MetadataDtype},
        tensor_metadata::{descriptor_with_f32, expected_physical_with_f32},
    },
    weights::Error,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const REPOSITORY: &str = "TheCluster/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-MLX-bf16";
pub const REVISION: &str = "78efacde810bbe7d1b80410fc5ac60db4f24fd8a";
pub const FORMAT_VERSION: &str = "qwen35-2b-diagnostic-affine4-bf16-f32-v1";
const LAYERS: usize = 24;
const HIDDEN: usize = 2048;
const INTERMEDIATE: usize = 6144;
const VOCAB: usize = 248320;
const PREFIX: &str = "language_model.model.";
const BODY_BYTES: usize = 1_059_315_904;

pub mod artifact;
mod configuration;
pub mod payload;
mod recipe;
#[cfg(all(test, target_pointer_width = "64"))]
mod tests;

/// Supplied by a separately authenticated conversion contract. No placeholder
/// whole-child digest, general family admission or provenance claim is implied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArtifactPins {
    pub config_bytes: usize,
    pub config_sha256: [u8; 32],
    pub header_bytes: usize,
    pub header_sha256: [u8; 32],
    pub container_bytes: u64,
    pub container_sha256: [u8; 32],
}
impl ArtifactPins {
    fn validate(&self) -> Result<(), Error> {
        if self.config_bytes == 0
            || self.config_bytes > 64 * 1024
            || self.header_bytes == 0
            || self.header_bytes > 512 * 1024
            || self.config_sha256 == [0; 32]
            || self.header_sha256 == [0; 32]
            || self.container_sha256 == [0; 32]
            || (self.header_bytes as u64)
                .checked_add(8)
                .and_then(|n| n.checked_add(BODY_BYTES as u64))
                != Some(self.container_bytes)
        {
            return Err(invalid(
                "bounded explicit Qwen3.5-2B artifact pins required",
            ));
        }
        Ok(())
    }
}

/// Original source spans, distinct from the older BF16-only public byte record.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PayloadBytes {
    pub affine_codes: usize,
    pub affine_metadata: usize,
    pub bf16_vectors: usize,
    pub f32_vectors: usize,
    pub bf16_convolutions: usize,
    pub total: usize,
}
pub struct Plan {
    layers: [LayerRecipe; LAYERS],
    common: Vec<TensorPlan>,
    convolutions: Vec<ConvolutionPlan>,
    physical: BTreeMap<String, PhysicalTensor>,
    bytes: PayloadBytes,
    pins: ArtifactPins,
}
impl Plan {
    pub fn layers(&self) -> &[LayerRecipe; LAYERS] {
        &self.layers
    }
    ///187 packed matrices,97BF16 vectors and18 original FP32 A_log vectors.
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
    pub fn source_f32_vector_bytes(&self) -> usize {
        self.bytes.f32_vectors
    }
    /// Requested learned tensor bytes after exact small-parameter decoding.
    /// Allocation capacity, upload overlap, framework and sessions are additional.
    pub fn runtime_learned_bytes(&self) -> Result<u64, Error> {
        let b = self.bytes;
        [b.affine_codes, b.affine_metadata, b.f32_vectors]
            .into_iter()
            .try_fold(0_u64, |total, n| {
                total
                    .checked_add(n as u64)
                    .ok_or_else(|| invalid("Qwen3.5-2B runtime byte count overflow"))
            })?
            .checked_add(
                (b.bf16_vectors as u64)
                    .checked_add(b.bf16_convolutions as u64)
                    .and_then(|n| n.checked_mul(2))
                    .ok_or_else(|| invalid("Qwen3.5-2B runtime byte count overflow"))?,
            )
            .ok_or_else(|| invalid("Qwen3.5-2B runtime byte count overflow"))
    }
    pub fn artifact_pins(&self) -> &ArtifactPins {
        &self.pins
    }
    /// Source byte origin, never a promise of typed alignment.
    pub fn payload_offset(&self) -> usize {
        8 + self.pins.header_bytes
    }
    pub fn embedding(&self) -> &TensorPlan {
        &self.common[0]
    }
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

/// Exact externally pinned child metadata and explicit text-only policy. This
/// does not read payloads, admit contexts/devices, or qualify source transforms.
pub fn inspect(
    config: &[u8],
    header: &[u8],
    container_bytes: u64,
    policy: ModalityPolicy,
    pins: &ArtifactPins,
) -> Result<Plan, Error> {
    pins.validate()?;
    if policy != ModalityPolicy::TextOnly
        || container_bytes != pins.container_bytes
        || config.len() != pins.config_bytes
        || header.len() != pins.header_bytes
        || <[u8; 32]>::from(Sha256::digest(config)) != pins.config_sha256
        || <[u8; 32]>::from(Sha256::digest(header)) != pins.header_sha256
    {
        return Err(invalid(
            "pinned Qwen3.5-2B metadata/text-only policy differs",
        ));
    }
    layout(config, header, *pins)
}
fn layout(config: &[u8], header: &[u8], pins: ArtifactPins) -> Result<Plan, Error> {
    if config.len() > 64 * 1024 || header.len() > 512 * 1024 {
        return Err(invalid("bounded Qwen3.5-2B metadata required"));
    }
    configuration::validate(config)?;
    let (layers, common, convolutions) = recipe::build()?;
    let (physical, bytes) = physical_metadata(header, &common, &convolutions)?;
    Ok(Plan {
        layers,
        common,
        convolutions,
        physical,
        bytes,
        pins,
    })
}
fn physical_metadata(
    header: &[u8],
    common: &[TensorPlan],
    convolutions: &[ConvolutionPlan],
) -> Result<(BTreeMap<String, PhysicalTensor>, PayloadBytes), Error> {
    let mut expected = expected_physical_with_f32(common);
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
    let document = super::super::qwen::parse_json(header)?;
    let header = document
        .as_object()
        .filter(|h| h.len() == 695)
        .ok_or_else(|| invalid("exact694-tensor Qwen3.5-2B inventory required"))?;
    if header.get("__metadata__") != Some(&json!({"format":"mlx"})) {
        return Err(invalid("Qwen3.5-2B container metadata differs"));
    }
    let mut physical = BTreeMap::new();
    let mut spans = Vec::with_capacity(694);
    let mut bytes = PayloadBytes::default();
    for (name, value) in header {
        if name == "__metadata__" {
            continue;
        }
        let (dtype, shape, size) = expected
            .remove(name)
            .ok_or_else(|| invalid("unexpected Qwen3.5-2B tensor"))?;
        let tensor = descriptor_with_f32(value, BODY_BYTES)?;
        if tensor.dtype != dtype || tensor.shape != shape || tensor.range.len() != size {
            return Err(invalid("Qwen3.5-2B tensor representation/shape differs"));
        }
        let bucket = if dtype == Dtype::U32 {
            &mut bytes.affine_codes
        } else if dtype == Dtype::F32 {
            &mut bytes.f32_vectors
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
    spans.sort_unstable_by_key(|r| r.start);
    let mut cursor = 0;
    for span in spans {
        if span.start != cursor {
            return Err(invalid("Qwen3.5-2B tensor spans gap or overlap"));
        }
        cursor = span.end;
    }
    if !expected.is_empty()
        || cursor != BODY_BYTES
        || bytes
            != (PayloadBytes {
                affine_codes: 940_638_208,
                affine_metadata: 117_579_776,
                bf16_vectors: 212_032,
                f32_vectors: 1152,
                bf16_convolutions: 884_736,
                total: BODY_BYTES,
            })
    {
        return Err(invalid("Qwen3.5-2B typed payload coverage differs"));
    }
    Ok((physical, bytes))
}
