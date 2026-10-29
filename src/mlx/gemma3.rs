//! Cold metadata plan for the inspected Gemma3 270M text export.
//!
//! Inspection neither reads tensor payload nor grants execution/Chat admission.
//! The separate payload module authenticates original bytes and prepares explicit
//! raw-gamma inputs for a bounded concrete graph; it does not grant Chat admission.
use super::affine::{Layout, MetadataDtype};
pub use super::gemma_metadata::{Dtype, PhysicalTensor, Shape, Storage, TensorPlan};
use super::gemma_metadata::{descriptor, expected_physical, same_semantics};
use crate::{model::shared_kv::KvKind, weights::Error};
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub mod payload;

const LAYERS: usize = 18;
const HIDDEN: usize = 640;
const HEAD: usize = 256;
const QUERY_HEADS: usize = 4;
const VOCAB: usize = 262144;
const INTERMEDIATE: usize = 2048;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerPlan {
    pub kind: KvKind,
    pub rope_theta: f32,
    pub query_width: usize,
    pub kv_width: usize,
    pub head_dim: usize,
    pub attention_scale: f32,
}

pub struct Plan {
    layers: [LayerPlan; LAYERS],
    logical: Vec<TensorPlan>,
    physical: BTreeMap<String, PhysicalTensor>,
    payload_bytes: usize,
}
impl Plan {
    pub fn layers(&self) -> &[LayerPlan; LAYERS] {
        &self.layers
    }
    pub fn logical_tensors(&self) -> &[TensorPlan] {
        &self.logical
    }
    pub fn physical_tensors(&self) -> &BTreeMap<String, PhysicalTensor> {
        &self.physical
    }
    pub fn payload_bytes(&self) -> usize {
        self.payload_bytes
    }
    /// Source declaration only; no backend/session capacity is admitted.
    pub fn declared_total_positions(&self) -> usize {
        32768
    }
    pub fn declared_sliding_window(&self) -> usize {
        512
    }
    /// Source's BF16-rounded sqrt(640), before conversion to activation dtype.
    pub fn embedding_scale(&self) -> f32 {
        25.25
    }
    /// All stored norm vectors are raw BF16 gamma, not folded coefficients.
    /// Consumers must qualify their coefficient-addition/output-store precision.
    pub fn norm_gamma_offset(&self) -> f32 {
        1.0
    }
    /// One physical packed table has both embedding and output uses.
    pub fn tied_output(&self) -> bool {
        true
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}
fn layers() -> [LayerPlan; LAYERS] {
    std::array::from_fn(|index| {
        let global = index % 6 == 5;
        LayerPlan {
            kind: if global {
                KvKind::Full
            } else {
                KvKind::Sliding
            },
            rope_theta: if global { 1_000_000.0 } else { 10_000.0 },
            query_width: QUERY_HEADS * HEAD,
            kv_width: HEAD,
            head_dim: HEAD,
            attention_scale: 1.0 / 16.0,
        }
    })
}

fn expected_config() -> Value {
    let types: Vec<_> = (0..LAYERS)
        .map(|i| {
            if i % 6 == 5 {
                "full_attention"
            } else {
                "sliding_attention"
            }
        })
        .collect();
    json!({"_sliding_window_pattern":6,"architectures":["Gemma3ForCausalLM"],
        "attention_bias":false,"attention_dropout":0.0,"attn_logit_softcapping":null,
        "bos_token_id":2,"eos_token_id":[1,106],"final_logit_softcapping":null,
        "head_dim":HEAD,"hidden_activation":"gelu_pytorch_tanh","hidden_size":HIDDEN,
        "initializer_range":0.02,"intermediate_size":INTERMEDIATE,"layer_types":types,
        "max_position_embeddings":32768,"model_type":"gemma3_text",
        "num_attention_heads":QUERY_HEADS,"num_hidden_layers":LAYERS,"num_key_value_heads":1,
        "pad_token_id":0,"quantization":{"group_size":64,"bits":4},
        "quantization_config":{"group_size":64,"bits":4},"query_pre_attn_scalar":256,
        "rms_norm_eps":1e-6,"rope_local_base_freq":10000.0,"rope_scaling":null,
        "rope_theta":1000000.0,"sliding_window":512,"torch_dtype":"bfloat16",
        "transformers_version":"4.55.0.dev0","use_bidirectional_attention":false,
        "use_cache":true,"vocab_size":VOCAB})
}

fn logical() -> Result<Vec<TensorPlan>, Error> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(236)
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
    add("model.norm.weight".into(), Shape::Vector(HIDDEN))?;
    for index in 0..LAYERS {
        let base = format!("model.layers.{index}.");
        for name in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            add(format!("{base}{name}.weight"), Shape::Vector(HIDDEN))?;
        }
        for name in ["q", "k"] {
            add(
                format!("{base}self_attn.{name}_norm.weight"),
                Shape::Vector(HEAD),
            )?;
        }
        for (name, rows, columns) in [
            ("self_attn.q_proj", QUERY_HEADS * HEAD, HIDDEN),
            ("self_attn.k_proj", HEAD, HIDDEN),
            ("self_attn.v_proj", HEAD, HIDDEN),
            ("self_attn.o_proj", HIDDEN, QUERY_HEADS * HEAD),
            ("mlp.gate_proj", INTERMEDIATE, HIDDEN),
            ("mlp.up_proj", INTERMEDIATE, HIDDEN),
            ("mlp.down_proj", HIDDEN, INTERMEDIATE),
        ] {
            add(
                format!("{base}{name}.weight"),
                Shape::Matrix { rows, columns },
            )?;
        }
    }
    Ok(result)
}

/// Inspect exact metadata buffers (each bounded to1MiB with duplicate rejection).
/// The container length is an advertised input; no payload bytes are consumed.
/// Only the pinned 270M graph/representation is described, not all Gemma3 exports.
pub fn inspect(
    config_bytes: &[u8],
    index_bytes: &[u8],
    header_bytes: &[u8],
    container_bytes: u64,
) -> Result<Plan, Error> {
    let config = super::qwen::parse_json(config_bytes)?;
    if !same_semantics(&config, &expected_config()) {
        return Err(invalid("Gemma3 270M text config differs"));
    }
    let index = super::qwen::parse_json(index_bytes)?;
    let header = super::qwen::parse_json(header_bytes)?;
    let payload = usize::try_from(container_bytes)
        .ok()
        .and_then(|n| n.checked_sub(8))
        .and_then(|n| n.checked_sub(header_bytes.len()))
        .ok_or_else(|| invalid("truncated advertised Gemma3 container"))?;
    if index.as_object().is_none_or(|v| v.len() != 2)
        || index["metadata"] != json!({"total_size":payload,"total_parameters":268098176})
    {
        return Err(invalid("Gemma3 index size/fields differ"));
    }
    let map = index["weight_map"]
        .as_object()
        .ok_or_else(|| invalid("Gemma3 weight map required"))?;
    let header = header
        .as_object()
        .ok_or_else(|| invalid("Gemma3 header object required"))?;
    let logical = logical()?;
    let expected = expected_physical(&logical);
    if header.get("__metadata__") != Some(&json!({"format":"mlx"}))
        || header.len() != expected.len() + 1
        || map.len() != expected.len()
    {
        return Err(invalid("Gemma3 physical tensor inventory differs"));
    }
    let mut physical = BTreeMap::new();
    let mut spans = Vec::new();
    for (name, (dtype, shape, bytes)) in expected {
        if map.get(&name) != Some(&json!("model.safetensors")) {
            return Err(invalid("Gemma3 tensor shard differs"));
        }
        let item = descriptor(
            header
                .get(&name)
                .ok_or_else(|| invalid("missing Gemma3 tensor"))?,
            payload,
        )?;
        if item.dtype != dtype || item.shape != shape || item.range.end - item.range.start != bytes
        {
            return Err(invalid("Gemma3 tensor representation differs"));
        }
        spans.push(item.range.clone());
        physical.insert(name, item);
    }
    spans.sort_unstable_by_key(|s| (s.start, s.end));
    let mut cursor = 0;
    for span in spans {
        if span.start != cursor {
            return Err(invalid("Gemma3 tensor payload has gaps/overlap"));
        }
        cursor = span.end;
    }
    if cursor != payload {
        return Err(invalid("Gemma3 payload coverage differs"));
    }
    Ok(Plan {
        layers: layers(),
        logical,
        physical,
        payload_bytes: payload,
    })
}

#[cfg(test)]
mod tests;
