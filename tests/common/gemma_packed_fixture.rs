use super::engine;
use engine::mlx::affine::{Layout, Matrix, MetadataDtype};
use engine::model::gemma_packed::{KvWeights, LayerWeights, Weights};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub struct Fixture {
    pub data: Value,
    pub ids: Vec<u32>,
    norms: BTreeMap<String, Vec<f32>>,
}
const RAW: &[u8] = include_bytes!("../fixtures/gemma-packed-graph-v1/math.bin");
fn bytes(item: &Value) -> &'static [u8] {
    let start = item["offset"].as_u64().unwrap() as usize;
    let size = item["bytes"].as_u64().unwrap() as usize;
    &RAW[start..start + size]
}
impl Fixture {
    pub fn new() -> Self {
        let receipt = include_bytes!(
            "../../benchmarks/results/2026-10-25-gemma-packed-graph-reference-v1.json"
        );
        assert_eq!(
            format!("{:x}", Sha256::digest(receipt)),
            "985e3d06115533396321be974a34fe4eecad5e5495efcc1fb2c327577e96c819"
        );
        let accepted: Value = serde_json::from_slice(receipt).unwrap();
        let manifest = include_bytes!("../fixtures/gemma-packed-graph-v1/math.json");
        assert_eq!(
            format!("{:x}", Sha256::digest(manifest)),
            accepted["fixture_sha256"].as_str().unwrap()
        );
        let data: Value = serde_json::from_slice(manifest).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(RAW)),
            data["binary"]["sha256"].as_str().unwrap()
        );
        let norms = data["weights"]
            .as_object()
            .unwrap()
            .iter()
            .filter(|(name, _)| data["packed_original_bf16"].get(name.as_str()).is_none())
            .map(|(name, item)| {
                (
                    name.clone(),
                    bytes(item)
                        .chunks_exact(4)
                        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                        .collect(),
                )
            })
            .collect();
        let ids = data["token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        Self { data, ids, norms }
    }
    pub fn matrix(&self, name: &str) -> Matrix<'static> {
        let item = &self.data["packed_original_bf16"][name];
        let shape = &self.data["weights"][name]["shape"];
        Matrix::from_le_bytes(
            Layout::new(
                shape[0].as_u64().unwrap() as usize,
                shape[1].as_u64().unwrap() as usize,
                4,
                64,
                MetadataDtype::BF16,
            )
            .unwrap(),
            bytes(&item["weight"]),
            bytes(&item["scales"]),
            bytes(&item["biases"]),
        )
        .unwrap()
    }
    pub fn weights(&self) -> Weights<'_> {
        Weights {
            embedding: self.matrix("model.embed_tokens.weight"),
            per_layer_embedding: self.matrix("model.embed_tokens_per_layer.weight"),
            per_layer_projection: self.matrix("model.per_layer_model_projection.weight"),
            per_layer_norm: &self.norms["model.per_layer_projection_norm.weight"],
            final_norm: &self.norms["model.norm.weight"],
            layers: (0..4)
                .map(|i| {
                    let prefix = format!("model.layers.{i}.");
                    let m = |name: &str| self.matrix(&(prefix.clone() + name));
                    let n = |name: &str| self.norms[&(prefix.clone() + name)].as_slice();
                    LayerWeights {
                        input_norm: n("input_layernorm.weight"),
                        q: m("self_attn.q_proj.weight"),
                        q_norm: n("self_attn.q_norm.weight"),
                        o: m("self_attn.o_proj.weight"),
                        kv: (i < 2).then(|| KvWeights {
                            k: m("self_attn.k_proj.weight"),
                            v: m("self_attn.v_proj.weight"),
                            k_norm: n("self_attn.k_norm.weight"),
                        }),
                        post_attention_norm: n("post_attention_layernorm.weight"),
                        pre_ff_norm: n("pre_feedforward_layernorm.weight"),
                        post_ff_norm: n("post_feedforward_layernorm.weight"),
                        ff_gate: m("mlp.gate_proj.weight"),
                        ff_up: m("mlp.up_proj.weight"),
                        ff_down: m("mlp.down_proj.weight"),
                        ple_gate: m("per_layer_input_gate.weight"),
                        ple_projection: m("per_layer_projection.weight"),
                        post_ple_norm: n("post_per_layer_input_norm.weight"),
                        scalar: n("layer_scalar")[0],
                    }
                })
                .collect(),
        }
    }
    pub fn values(&self, contract: &str, name: &str) -> Vec<f64> {
        let item = &self.data[contract][name];
        if item["dtype"] == "<f8" {
            bytes(item)
                .chunks_exact(8)
                .map(|v| f64::from_le_bytes(v.try_into().unwrap()))
                .collect()
        } else {
            bytes(item)
                .chunks_exact(4)
                .map(|v| f64::from(f32::from_le_bytes(v.try_into().unwrap())))
                .collect()
        }
    }
}
