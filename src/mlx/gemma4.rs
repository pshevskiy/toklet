//! Cold structural adapter for the inspected Gemma4 E2B MLX candidate.
//!
//! This is separate from the executable dense-family registry. It describes
//! the inspected E2B graph and named representations, excludes modalities only by explicit policy,
//! and validates config/index/header bytes without allocating tensor payloads.
//! A plan grants no payload integrity, numerical execution, position capacity,
//! text profile, device fit or download/Chat eligibility.

use super::affine::{Layout, MetadataDtype};
use crate::model::shared_kv::KvKind;
use crate::weights::Error;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub mod payload;

#[cfg(test)]
mod tests;

const HIDDEN: usize = 1536;
const VOCAB: usize = 262144;
const LAYERS: usize = 35;
const PRODUCERS: usize = 15;
const PLE: usize = 256;
const MAX_HEADER_TENSORS: usize = 4096;
const FEED_FORWARD: [&str; 3] = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Format {
    Affine4,
    QatMixed,
}

impl Format {
    fn quantization(self) -> Value {
        let mut quant = json!({"group_size":64,"bits":4,"mode":"affine"});
        if self == Self::QatMixed {
            for layer in 0..LAYERS {
                for name in FEED_FORWARD {
                    quant[format!("language_model.model.layers.{layer}.{name}")] =
                        json!({"group_size":64,"bits":8});
                }
            }
        }
        quant
    }

    fn matrix_bits(self, tensor: &str) -> u8 {
        let feed_forward = tensor
            .strip_suffix(".weight")
            .is_some_and(|name| FEED_FORWARD.iter().any(|role| name.ends_with(role)));
        if self == Self::QatMixed && feed_forward {
            8
        } else {
            4
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModalityPolicy {
    Reject,
    /// Recognize but do not execute/allocate the known audio/vision components.
    TextOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerPlan {
    pub kind: KvKind,
    pub head_dim: usize,
    pub rotated_dims: usize,
    pub producer: usize,
    pub intermediate: usize,
}

pub use super::tensor_metadata::{Dtype, PhysicalTensor, Shape, Storage, TensorPlan};
use super::tensor_metadata::{descriptor, expected_physical, same_semantics};

pub struct E2bPlan {
    layers: [LayerPlan; LAYERS],
    logical: Vec<TensorPlan>,
    text: BTreeMap<String, PhysicalTensor>,
    excluded: BTreeMap<String, PhysicalTensor>,
    text_payload_bytes: usize,
    excluded_payload_bytes: usize,
    payload_bytes: usize,
}

impl E2bPlan {
    pub fn layers(&self) -> &[LayerPlan; LAYERS] {
        &self.layers
    }
    pub fn logical_tensors(&self) -> &[TensorPlan] {
        &self.logical
    }
    pub fn text_tensors(&self) -> &BTreeMap<String, PhysicalTensor> {
        &self.text
    }
    pub fn excluded_tensors(&self) -> &BTreeMap<String, PhysicalTensor> {
        &self.excluded
    }
    pub fn text_payload_bytes(&self) -> usize {
        self.text_payload_bytes
    }
    pub fn excluded_payload_bytes(&self) -> usize {
        self.excluded_payload_bytes
    }
    pub fn payload_bytes(&self) -> usize {
        self.payload_bytes
    }
    /// A configuration declaration, not a qualified engine/request capacity.
    pub fn declared_total_positions(&self) -> usize {
        131072
    }
}

fn invalid(reason: impl Into<String>) -> Error {
    Error::Invalid(reason.into())
}

fn expected_text_config() -> Value {
    // The initial adapter intentionally names the exact audited E2B branch.
    // Unknown fields/options and missing explicit candidate values reject.
    let types: Vec<_> = (0..LAYERS)
        .map(|i| {
            if i % 5 == 4 {
                "full_attention"
            } else {
                "sliding_attention"
            }
        })
        .collect();
    json!({"attention_bias":false,"attention_dropout":0.0,"attention_k_eq_v":false,
        "bos_token_id":2,"dtype":"bfloat16","enable_moe_block":false,"eos_token_id":1,
        "expert_intermediate_size":null,"final_logit_softcapping":30.0,"global_head_dim":512,
        "head_dim":256,"hidden_activation":"gelu_pytorch_tanh","hidden_size":HIDDEN,
        "hidden_size_per_layer_input":PLE,"initializer_range":0.02,"intermediate_size":6144,
        "layer_types":types,"max_position_embeddings":131072,"model_type":"gemma4_text",
        "num_attention_heads":8,"num_experts":null,"num_global_key_value_heads":null,
        "num_hidden_layers":LAYERS,"num_key_value_heads":1,"num_kv_shared_layers":20,
        "pad_token_id":0,"rms_norm_eps":1e-6,
        "rope_parameters":{"full_attention":{"partial_rotary_factor":0.25,"rope_theta":1000000.0,"rope_type":"proportional"},
            "sliding_attention":{"rope_theta":10000.0,"rope_type":"default"}},
        "sliding_window":512,"tie_word_embeddings":true,"top_k_experts":null,
        "use_bidirectional_attention":null,"use_cache":true,"use_double_wide_mlp":true,
        "vocab_size":VOCAB,"vocab_size_per_layer_input":VOCAB})
}

fn config(value: &Value, policy: ModalityPolicy, format: Format) -> Result<(), Error> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid("Gemma4 wrapper object required"))?;
    let allowed = [
        "architectures",
        "audio_config",
        "audio_token_id",
        "boa_token_id",
        "boi_token_id",
        "do_sample",
        "dtype",
        "eoa_token_id",
        "eoa_token_index",
        "eoi_token_id",
        "eos_token_id",
        "generation_config",
        "image_token_id",
        "initializer_range",
        "model_type",
        "quantization",
        "quantization_config",
        "temperature",
        "text_config",
        "tie_word_embeddings",
        "top_k",
        "top_p",
        "transformers_version",
        "video_token_id",
        "vision_config",
        "vision_soft_tokens_per_image",
    ];
    if object.keys().any(|key| !allowed.contains(&key.as_str())) {
        return Err(invalid("unsupported Gemma4 wrapper field"));
    }
    if value["model_type"] != "gemma4"
        || value["architectures"] != json!(["Gemma4ForConditionalGeneration"])
        || value["dtype"] != "bfloat16"
        || value["tie_word_embeddings"] != true
    {
        return Err(invalid(
            "only the inspected Gemma4 E2B wrapper is described",
        ));
    }
    if policy != ModalityPolicy::TextOnly {
        return Err(invalid(
            "Gemma4 requires explicit text-only modality exclusion",
        ));
    }
    for (name, kind) in [
        ("audio_config", "gemma4_audio"),
        ("vision_config", "gemma4_vision"),
    ] {
        if !value[name].is_object() || value[name]["model_type"] != kind {
            return Err(invalid(
                "unrecognized excluded Gemma4 modality configuration",
            ));
        }
    }
    let quant = format.quantization();
    if value["quantization"] != quant || value["quantization_config"] != quant {
        return Err(invalid(match format {
            Format::Affine4 => "Gemma4 E2B requires matching four-bit/group64 affine declarations",
            Format::QatMixed => "Gemma4 QAT requires its exact mixed group64 affine declarations",
        }));
    }
    let mut expected_text = expected_text_config();
    if format == Format::QatMixed {
        let text = expected_text.as_object_mut().unwrap();
        text.remove("expert_intermediate_size");
        text.insert("moe_intermediate_size".into(), Value::Null);
    }
    if !same_semantics(&value["text_config"], &expected_text) {
        return Err(invalid(
            "unsupported or incomplete Gemma4 E2B text graph semantics",
        ));
    }
    // Root generation metadata, special IDs and excluded modality internals
    // have no numerical role here. They do not grant tokenizer/chat semantics.
    Ok(())
}

fn layer_plans() -> [LayerPlan; LAYERS] {
    std::array::from_fn(|index| {
        let full = index % 5 == 4;
        LayerPlan {
            kind: if full { KvKind::Full } else { KvKind::Sliding },
            head_dim: if full { 512 } else { 256 },
            rotated_dims: if full { 128 } else { 256 },
            producer: if index < PRODUCERS {
                index
            } else if full {
                14
            } else {
                13
            },
            intermediate: if index < PRODUCERS { 6144 } else { 12288 },
        }
    })
}

fn logical(layers: &[LayerPlan; LAYERS], format: Format) -> Result<Vec<TensorPlan>, Error> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(540)
        .map_err(|_| Error::Allocation)?;
    let mut add = |name: String, shape: Shape| -> Result<(), Error> {
        let storage = match shape {
            Shape::Vector(_) => Storage::Bf16Vector,
            Shape::Matrix { rows, columns } => Storage::Affine(
                Layout::new(
                    rows,
                    columns,
                    format.matrix_bits(&name),
                    64,
                    MetadataDtype::BF16,
                )
                .map_err(|error| invalid(error.to_string()))?,
            ),
        };
        result.push(TensorPlan {
            name,
            shape,
            storage,
        });
        Ok(())
    };
    let prefix = "language_model.model.";
    for (name, rows, columns) in [
        ("embed_tokens", VOCAB, HIDDEN),
        ("embed_tokens_per_layer", VOCAB, LAYERS * PLE),
        ("per_layer_model_projection", LAYERS * PLE, HIDDEN),
    ] {
        add(
            format!("{prefix}{name}.weight"),
            Shape::Matrix { rows, columns },
        )?;
    }
    for (name, size) in [("per_layer_projection_norm", PLE), ("norm", HIDDEN)] {
        add(format!("{prefix}{name}.weight"), Shape::Vector(size))?;
    }
    for (index, layer) in layers.iter().enumerate() {
        let base = format!("{prefix}layers.{index}.");
        for name in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "post_per_layer_input_norm",
        ] {
            add(format!("{base}{name}.weight"), Shape::Vector(HIDDEN))?;
        }
        add(format!("{base}layer_scalar"), Shape::Vector(1))?;
        add(
            format!("{base}self_attn.q_norm.weight"),
            Shape::Vector(layer.head_dim),
        )?;
        for (name, rows, columns) in [
            ("self_attn.q_proj", 8 * layer.head_dim, HIDDEN),
            ("self_attn.o_proj", HIDDEN, 8 * layer.head_dim),
            (FEED_FORWARD[0], layer.intermediate, HIDDEN),
            (FEED_FORWARD[1], layer.intermediate, HIDDEN),
            (FEED_FORWARD[2], HIDDEN, layer.intermediate),
            ("per_layer_input_gate", PLE, HIDDEN),
            ("per_layer_projection", HIDDEN, PLE),
        ] {
            add(
                format!("{base}{name}.weight"),
                Shape::Matrix { rows, columns },
            )?;
        }
        if index < PRODUCERS {
            add(
                format!("{base}self_attn.k_norm.weight"),
                Shape::Vector(layer.head_dim),
            )?;
            for name in ["k", "v"] {
                add(
                    format!("{base}self_attn.{name}_proj.weight"),
                    Shape::Matrix {
                        rows: layer.head_dim,
                        columns: HIDDEN,
                    },
                )?;
            }
        }
    }
    Ok(result)
}

/// Inspect three exact in-memory JSON buffers. Each is bounded to 1 MiB and
/// rejects duplicate fields before parsing. Container length is an advertised
/// metadata input; the tensor payload is neither opened nor authenticated here.
pub fn inspect(
    config_bytes: &[u8],
    index_bytes: &[u8],
    header_bytes: &[u8],
    container_bytes: u64,
    policy: ModalityPolicy,
) -> Result<E2bPlan, Error> {
    inspect_format(
        config_bytes,
        index_bytes,
        header_bytes,
        container_bytes,
        policy,
        Format::Affine4,
    )
}

/// Inspect only the pinned mixed affine4/8 QAT metadata. The advertised
/// container identity remains separate from the actual payload bytes.
pub fn inspect_qat(
    config_bytes: &[u8],
    index_bytes: &[u8],
    header_bytes: &[u8],
    policy: ModalityPolicy,
) -> Result<E2bPlan, Error> {
    for (bytes, size, digest) in [
        (
            config_bytes,
            31_198,
            "e996bc6f3887d18d1176438680b17c24e988576d5c15a43ccf3a482a737a8f7e",
        ),
        (
            index_bytes,
            218_323,
            "8ee3112362c6dfa13a26df2da4592c0a9aa541013918469accd86d7e8a54b11a",
        ),
        (
            header_bytes,
            320_311,
            "330a4034f42da2ba124b7ac37925225a1959f0f2e5c08eb62c4132851773dea6",
        ),
    ] {
        if bytes.len() != size || format!("{:x}", Sha256::digest(bytes)) != digest {
            return Err(invalid("Gemma4 QAT pinned metadata differs"));
        }
    }
    inspect_format(
        config_bytes,
        index_bytes,
        header_bytes,
        4_329_240_325,
        policy,
        Format::QatMixed,
    )
}

fn inspect_format(
    config_bytes: &[u8],
    index_bytes: &[u8],
    header_bytes: &[u8],
    container_bytes: u64,
    policy: ModalityPolicy,
    format: Format,
) -> Result<E2bPlan, Error> {
    let document = super::qwen::parse_json(config_bytes)?;
    config(&document, policy, format)?;
    let index = super::qwen::parse_json(index_bytes)?;
    let header = super::qwen::parse_json(header_bytes)?;
    let container =
        usize::try_from(container_bytes).map_err(|_| invalid("Gemma4 container size overflow"))?;
    let payload = container
        .checked_sub(8)
        .and_then(|n| n.checked_sub(header_bytes.len()))
        .ok_or_else(|| invalid("truncated advertised Gemma4 container"))?;
    if index.as_object().is_none_or(|v| v.len() != 2)
        || index["metadata"] != json!({"total_size":payload})
    {
        return Err(invalid("Gemma4 index payload size or fields differ"));
    }
    let map = index["weight_map"]
        .as_object()
        .ok_or_else(|| invalid("Gemma4 index weight_map object required"))?;
    let header = header
        .as_object()
        .ok_or_else(|| invalid("Gemma4 header object required"))?;
    if header.get("__metadata__") != Some(&json!({"format":"mlx"}))
        || header.len() > MAX_HEADER_TENSORS + 1
        || header.len() != map.len() + 1
    {
        return Err(invalid(
            "Gemma4 header metadata or tensor inventory differs",
        ));
    }
    let layers = layer_plans();
    let logical = logical(&layers, format)?;
    let expected = expected_physical(&logical);
    let mut text = BTreeMap::new();
    let mut excluded = BTreeMap::new();
    let mut spans = Vec::new();
    let mut text_bytes = 0_usize;
    let mut excluded_bytes = 0_usize;
    for (name, value) in header {
        if name == "__metadata__" {
            continue;
        }
        if name.len() > 256 || map.get(name) != Some(&json!("model.safetensors")) {
            return Err(invalid("Gemma4 tensor/index name or shard differs"));
        }
        let tensor = descriptor(value, payload)?;
        let bytes = tensor.range.end - tensor.range.start;
        spans.push(tensor.range.clone());
        if name.starts_with("language_model.") {
            let Some((dtype, shape, size)) = expected.get(name) else {
                return Err(invalid("unexpected Gemma4 text tensor"));
            };
            if tensor.dtype != *dtype || tensor.shape != *shape || bytes != *size {
                return Err(invalid("Gemma4 text tensor representation differs"));
            }
            text_bytes = text_bytes
                .checked_add(bytes)
                .ok_or_else(|| invalid("text payload sum overflow"))?;
            text.insert(name.clone(), tensor);
        } else {
            let root = name.split('.').next().unwrap_or("");
            if !["audio_tower", "vision_tower", "embed_audio", "embed_vision"].contains(&root) {
                return Err(invalid("unknown excluded Gemma4 component"));
            }
            excluded_bytes = excluded_bytes
                .checked_add(bytes)
                .ok_or_else(|| invalid("excluded payload sum overflow"))?;
            excluded.insert(name.clone(), tensor);
        }
    }
    if text.len() != expected.len() {
        return Err(invalid("incomplete Gemma4 text tensor inventory"));
    }
    spans.sort_unstable_by_key(|span| (span.start, span.end));
    let mut cursor = 0;
    for span in spans {
        if span.start != cursor {
            return Err(invalid("Gemma4 payload has gaps or overlap"));
        }
        cursor = span.end;
    }
    if cursor != payload || text_bytes.checked_add(excluded_bytes) != Some(payload) {
        return Err(invalid("Gemma4 payload coverage differs"));
    }
    Ok(E2bPlan {
        layers,
        logical,
        text,
        excluded,
        text_payload_bytes: text_bytes,
        excluded_payload_bytes: excluded_bytes,
        payload_bytes: payload,
    })
}
