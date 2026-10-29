//! Cold metadata for the pinned LFM2-350M MLX export.
//!
//! Inspection reads no tensor payload and grants no backend, context, text or model admission
//! is granted. Convolution rank-three coefficients are explicit owners, separate
//! from the shared vector/matrix inventory. B/C/x views borrow one fused parent.
use super::affine::{Layout, MetadataDtype, RowSpan};
pub use super::tensor_metadata::{Dtype, PhysicalTensor, Shape, Storage, TensorPlan};
use super::tensor_metadata::{descriptor, expected_physical, same_semantics};
use crate::capacity::{KvOwner, KvStorage, Retention};
use crate::weights::Error;
use serde_json::{Value, json};
use std::collections::BTreeMap;

/// Original-byte loading is separate from metadata and numerical admission.
pub mod payload;

const LAYERS: usize = 16;
const HIDDEN: usize = 1024;
const INTERMEDIATE: usize = 4608;
const VOCAB: usize = 65536;
const ATTENTION_LAYERS: [usize; 6] = [2, 5, 8, 10, 12, 14];
const CONV_OWNERS: usize = 10;
const COMMON_OWNERS: usize = 138;
const PHYSICAL_TENSORS: usize = 334;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionPlan {
    pub kv_owner: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub attention_scale: f32,
}

/// Owner IDs index their own state kind; no convolution is a pretend KV layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LayerRecipe {
    FullAttention(AttentionPlan),
    ShortConv3 { history_owner: usize },
}

/// Stored `[channels, 3, 1]`, with taps ordered oldest, previous, current.
/// No payload/convolution operation or coefficient transpose occurs here.
#[derive(Debug)]
pub struct Depthwise3Plan {
    pub layer: usize,
    pub history_owner: usize,
    pub coefficient_name: String,
    pub channels: usize,
    pub taps: usize,
    pub input_group_channels: usize,
    pub fused_input_name: String,
    pub output_name: String,
    /// B, C, x in that order. Ranges are relative to the original fused
    /// weight/scales/biases tensors, not newly allocated tensor owners.
    pub bcx: [RowSpan; 3],
}

/// Pure numerical state payload only: no guards, owner metadata, staging,
/// scratch, model, tokenizer or framework allowance. Not an admission budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StatePayloadBytes {
    pub attention_kv: usize,
    pub convolution_committed: usize,
    pub convolution_candidate: usize,
    pub total: usize,
}

pub struct Plan {
    layers: [LayerRecipe; LAYERS],
    common: Vec<TensorPlan>,
    convolutions: Vec<Depthwise3Plan>,
    physical: BTreeMap<String, PhysicalTensor>,
    payload_bytes: usize,
}
impl Plan {
    pub fn layers(&self) -> &[LayerRecipe; LAYERS] {
        &self.layers
    }
    /// Only the 93 affine and 45 norm owners. The ten rank-three convolution
    /// owners are returned separately by `convolutions`.
    pub fn logical_tensors(&self) -> &[TensorPlan] {
        &self.common
    }
    pub fn convolutions(&self) -> &[Depthwise3Plan] {
        &self.convolutions
    }
    pub fn logical_owner_count(&self) -> usize {
        self.common.len() + self.convolutions.len()
    }
    pub fn physical_tensors(&self) -> &BTreeMap<String, PhysicalTensor> {
        &self.physical
    }
    pub fn payload_bytes(&self) -> usize {
        self.payload_bytes
    }
    pub fn declared_total_positions(&self) -> usize {
        128000
    }
    pub fn implemented_total_positions(&self) -> Option<usize> {
        None
    }
    pub fn qualified_total_positions(&self) -> Option<usize> {
        None
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
    pub fn norm_epsilon(&self) -> f32 {
        1e-5
    }
    /// Stored BF16 gains are direct weights; unlike Gemma3 there is no +1.
    /// This getter does not qualify any activation/reduction/store precision.
    pub fn norm_weights_are_direct(&self) -> bool {
        true
    }
    pub fn tied_output(&self) -> bool {
        true
    }
    /// Pure checked geometry at a hypothetical capacity; it grants neither the
    /// source-declared window nor an implemented/qualified backend interval.
    pub fn state_payload_bytes(
        &self,
        capacity: usize,
        storage: KvStorage,
    ) -> Result<StatePayloadBytes, Error> {
        let attention_kv = KvOwner {
            heads: 8,
            head_dim: 64,
            storage,
            retention: Retention::Full,
            guard_elements: 0,
            copies: ATTENTION_LAYERS.len(),
        }
        .at_capacity(capacity)
        .map_err(|e| invalid(e.to_string()))?
        .total_bytes;
        let element_bytes = match storage {
            KvStorage::F16 => 2,
            KvStorage::F32 => 4,
        };
        let convolution_committed = CONV_OWNERS
            .checked_mul(2)
            .and_then(|v| v.checked_mul(HIDDEN))
            .and_then(|v| v.checked_mul(element_bytes))
            .ok_or_else(|| invalid("LFM2 history payload overflow"))?;
        let total = convolution_committed
            .checked_mul(2)
            .and_then(|v| v.checked_add(attention_kv))
            .ok_or_else(|| invalid("LFM2 state payload overflow"))?;
        Ok(StatePayloadBytes {
            attention_kv,
            convolution_committed,
            convolution_candidate: convolution_committed,
            total,
        })
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}
fn expected_config() -> Value {
    json!({"architectures":["Lfm2ForCausalLM"],"block_auto_adjust_ff_dim":true,
        "block_dim":HIDDEN,"block_ff_dim":6656,"block_ffn_dim_multiplier":1.0,
        "block_mlp_init_scale":1.0,"block_multiple_of":256,"block_norm_eps":1e-5,
        "block_out_init_scale":1.0,"block_use_swiglu":true,"block_use_xavier_init":true,
        "bos_token_id":1,"conv_L_cache":3,"conv_bias":false,"conv_dim":HIDDEN,
        "conv_dim_out":HIDDEN,"conv_use_xavier_init":true,"eos_token_id":7,
        "full_attn_idxs":ATTENTION_LAYERS,"hidden_size":HIDDEN,"initializer_range":0.02,
        "max_position_embeddings":128000,"model_type":"lfm2","norm_eps":1e-5,
        "num_attention_heads":16,"num_heads":16,"num_hidden_layers":LAYERS,
        "num_key_value_heads":8,"pad_token_id":0,"quantization":{"group_size":64,"bits":4},
        "quantization_config":{"group_size":64,"bits":4},"rope_theta":1000000.0,
        "torch_dtype":"bfloat16","transformers_version":"4.54.0.dev0",
        "use_cache":true,"use_pos_enc":true,"vocab_size":VOCAB})
}
fn layers() -> [LayerRecipe; LAYERS] {
    let mut kv_owner = 0;
    let mut history_owner = 0;
    std::array::from_fn(|layer| {
        if ATTENTION_LAYERS.contains(&layer) {
            let result = LayerRecipe::FullAttention(AttentionPlan {
                kv_owner,
                query_heads: 16,
                kv_heads: 8,
                head_dim: 64,
                rope_theta: 1e6,
                attention_scale: 1.0 / 8.0,
            });
            kv_owner += 1;
            result
        } else {
            let result = LayerRecipe::ShortConv3 { history_owner };
            history_owner += 1;
            result
        }
    })
}
fn common_tensors() -> Result<Vec<TensorPlan>, Error> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(COMMON_OWNERS)
        .map_err(|_| Error::Allocation)?;
    let mut add = |name: String, shape: Shape| -> Result<(), Error> {
        let storage = match shape {
            Shape::Vector(_) => Storage::Bf16Vector,
            Shape::Matrix { rows, columns } => Storage::Affine(
                Layout::new(rows, columns, 4, 64, MetadataDtype::BF16)
                    .map_err(|e| invalid(e.to_string()))?,
            ),
        };
        result.push(TensorPlan {
            name,
            shape,
            storage,
        });
        Ok(())
    };
    add(
        "model.embed_tokens.weight".into(),
        Shape::Matrix {
            rows: VOCAB,
            columns: HIDDEN,
        },
    )?;
    add("model.embedding_norm.weight".into(), Shape::Vector(HIDDEN))?;
    for layer in 0..LAYERS {
        let base = format!("model.layers.{layer}.");
        for norm in ["operator_norm", "ffn_norm"] {
            add(format!("{base}{norm}.weight"), Shape::Vector(HIDDEN))?;
        }
        for (name, rows, columns) in [
            ("w1", INTERMEDIATE, HIDDEN),
            ("w3", INTERMEDIATE, HIDDEN),
            ("w2", HIDDEN, INTERMEDIATE),
        ] {
            add(
                format!("{base}feed_forward.{name}.weight"),
                Shape::Matrix { rows, columns },
            )?;
        }
        if ATTENTION_LAYERS.contains(&layer) {
            for name in ["q", "k"] {
                add(
                    format!("{base}self_attn.{name}_layernorm.weight"),
                    Shape::Vector(64),
                )?;
            }
            for (name, rows) in [
                ("q_proj", 1024),
                ("k_proj", 512),
                ("v_proj", 512),
                ("out_proj", 1024),
            ] {
                add(
                    format!("{base}self_attn.{name}.weight"),
                    Shape::Matrix {
                        rows,
                        columns: HIDDEN,
                    },
                )?;
            }
        } else {
            for (name, rows) in [("in_proj", 3 * HIDDEN), ("out_proj", HIDDEN)] {
                add(
                    format!("{base}conv.{name}.weight"),
                    Shape::Matrix {
                        rows,
                        columns: HIDDEN,
                    },
                )?;
            }
        }
    }
    Ok(result)
}
fn convolutions(common: &[TensorPlan]) -> Result<Vec<Depthwise3Plan>, Error> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(CONV_OWNERS)
        .map_err(|_| Error::Allocation)?;
    for layer in 0..LAYERS {
        if ATTENTION_LAYERS.contains(&layer) {
            continue;
        }
        let base = format!("model.layers.{layer}.conv.");
        let fused_input_name = format!("{base}in_proj.weight");
        let parent = common
            .iter()
            .find(|t| t.name == fused_input_name)
            .ok_or_else(|| invalid("missing LFM2 fused parent"))?;
        let Storage::Affine(layout) = parent.storage else {
            return Err(invalid("LFM2 fused parent must be affine"));
        };
        let rows = |start| {
            layout
                .row_span(start..start + HIDDEN)
                .map_err(|e| invalid(e.to_string()))
        };
        result.push(Depthwise3Plan {
            layer,
            history_owner: result.len(),
            coefficient_name: format!("{base}conv.weight"),
            channels: HIDDEN,
            taps: 3,
            input_group_channels: 1,
            fused_input_name,
            output_name: format!("{base}out_proj.weight"),
            bcx: [rows(0)?, rows(HIDDEN)?, rows(2 * HIDDEN)?],
        });
    }
    Ok(result)
}

/// Inspect bounded duplicate-free JSON metadata and advertised container size.
/// Exact candidate structure only; no payload integrity or execution claim.
pub fn inspect(
    config_bytes: &[u8],
    index_bytes: &[u8],
    header_bytes: &[u8],
    advertised_container_len: u64,
) -> Result<Plan, Error> {
    let config = super::qwen::parse_json(config_bytes)?;
    if !same_semantics(&config, &expected_config()) {
        return Err(invalid("LFM2 350M config differs"));
    }
    let index = super::qwen::parse_json(index_bytes)?;
    let header = super::qwen::parse_json(header_bytes)?;
    let payload = usize::try_from(advertised_container_len)
        .ok()
        .and_then(|n| n.checked_sub(8))
        .and_then(|n| n.checked_sub(header_bytes.len()))
        .ok_or_else(|| invalid("truncated advertised LFM2 container"))?;
    if index.as_object().is_none_or(|v| v.len() != 2)
        || index["metadata"] != json!({"total_size":payload,"total_parameters":354483968})
    {
        return Err(invalid("LFM2 index size/fields differ"));
    }
    let map = index["weight_map"]
        .as_object()
        .ok_or_else(|| invalid("LFM2 weight map required"))?;
    let header = header
        .as_object()
        .ok_or_else(|| invalid("LFM2 header object required"))?;
    let common = common_tensors()?;
    let convolutions = convolutions(&common)?;
    let mut expected = expected_physical(&common);
    for conv in &convolutions {
        expected.insert(
            conv.coefficient_name.clone(),
            (
                Dtype::Bf16,
                vec![conv.channels, conv.taps, conv.input_group_channels],
                conv.channels * conv.taps * 2,
            ),
        );
    }
    if expected.len() != PHYSICAL_TENSORS
        || header.get("__metadata__") != Some(&json!({"format":"mlx"}))
        || header.len() != expected.len() + 1
        || map.len() != expected.len()
    {
        return Err(invalid("LFM2 physical tensor inventory differs"));
    }
    let mut physical = BTreeMap::new();
    let mut spans = Vec::new();
    spans
        .try_reserve_exact(expected.len())
        .map_err(|_| Error::Allocation)?;
    for (name, (dtype, shape, bytes)) in expected {
        if map.get(&name) != Some(&json!("model.safetensors")) {
            return Err(invalid("LFM2 tensor shard differs"));
        }
        let item = descriptor(
            header
                .get(&name)
                .ok_or_else(|| invalid("missing LFM2 tensor"))?,
            payload,
        )?;
        if item.dtype != dtype || item.shape != shape || item.range.end - item.range.start != bytes
        {
            return Err(invalid("LFM2 tensor representation differs"));
        }
        spans.push(item.range.clone());
        physical.insert(name, item);
    }
    spans.sort_unstable_by_key(|s| (s.start, s.end));
    let mut cursor = 0;
    for span in spans {
        if span.start != cursor {
            return Err(invalid("LFM2 payload gaps/overlap"));
        }
        cursor = span.end;
    }
    if cursor != payload {
        return Err(invalid("LFM2 payload coverage differs"));
    }
    Ok(Plan {
        layers: layers(),
        common,
        convolutions,
        physical,
        payload_bytes: payload,
    })
}

#[cfg(test)]
mod tests;
