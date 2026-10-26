//! Exact cold role resolution. No strings/maps are retained for execution.
use super::*;
use crate::mlx::gemma4::{E2bPlan, LayerPlan, Shape, Storage};

struct KvBindings {
    k: usize,
    v: usize,
    norm: usize,
}
struct LayerBindings {
    geometry: LayerPlan,
    q: usize,
    o: usize,
    gate: usize,
    up: usize,
    down: usize,
    ple_gate: usize,
    ple_projection: usize,
    input_norm: usize,
    q_norm: usize,
    post_attention_norm: usize,
    pre_ff_norm: usize,
    post_ff_norm: usize,
    post_ple_norm: usize,
    scalar: usize,
    kv: Option<KvBindings>,
}
pub(super) struct Bindings {
    embedding: usize,
    ple_embedding: usize,
    ple_projection: usize,
    ple_norm: usize,
    final_norm: usize,
    layers: [LayerBindings; 35],
}
impl Bindings {
    pub(super) fn new(cold: &E2bPlan) -> Result<Self> {
        let matrix = |name: &str, rows: usize, columns: usize| {
            let found = cold
                .logical_tensors()
                .iter()
                .filter(|t| matches!(t.storage, Storage::Affine(_)))
                .enumerate()
                .find(|(_, t)| t.name == name);
            match found {
                Some((index, t)) if t.shape == (Shape::Matrix { rows, columns }) => Ok(index),
                _ => Err(Error::Options("Gemma resident named matrix differs")),
            }
        };
        let vector = |name: &str, values: usize| {
            let found = cold
                .logical_tensors()
                .iter()
                .filter(|t| matches!(t.storage, Storage::Bf16Vector))
                .enumerate()
                .find(|(_, t)| t.name == name);
            match found {
                Some((index, t)) if t.shape == Shape::Vector(values) => Ok(index),
                _ => Err(Error::Options("Gemma resident named vector differs")),
            }
        };
        let prefix = "language_model.model.";
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(35)
            .map_err(|_| Error::Native("Gemma role reservation failed".into()))?;
        for (index, &geometry) in cold.layers().iter().enumerate() {
            let base = format!("{prefix}layers.{index}.");
            let m =
                |name: &str, rows, columns| matrix(&format!("{base}{name}.weight"), rows, columns);
            let v = |name: &str, values| vector(&format!("{base}{name}.weight"), values);
            layers.push(LayerBindings {
                geometry,
                q: m("self_attn.q_proj", 8 * geometry.head_dim, 1536)?,
                o: m("self_attn.o_proj", 1536, 8 * geometry.head_dim)?,
                gate: m("mlp.gate_proj", geometry.intermediate, 1536)?,
                up: m("mlp.up_proj", geometry.intermediate, 1536)?,
                down: m("mlp.down_proj", 1536, geometry.intermediate)?,
                ple_gate: m("per_layer_input_gate", 256, 1536)?,
                ple_projection: m("per_layer_projection", 1536, 256)?,
                input_norm: v("input_layernorm", 1536)?,
                q_norm: v("self_attn.q_norm", geometry.head_dim)?,
                post_attention_norm: v("post_attention_layernorm", 1536)?,
                pre_ff_norm: v("pre_feedforward_layernorm", 1536)?,
                post_ff_norm: v("post_feedforward_layernorm", 1536)?,
                post_ple_norm: v("post_per_layer_input_norm", 1536)?,
                scalar: vector(&format!("{base}layer_scalar"), 1)?,
                kv: if index < 15 {
                    Some(KvBindings {
                        k: m("self_attn.k_proj", geometry.head_dim, 1536)?,
                        v: m("self_attn.v_proj", geometry.head_dim, 1536)?,
                        norm: v("self_attn.k_norm", geometry.head_dim)?,
                    })
                } else {
                    None
                },
            });
        }
        let result = Self {
            embedding: matrix(&format!("{prefix}embed_tokens.weight"), 262144, 1536)?,
            ple_embedding: matrix(
                &format!("{prefix}embed_tokens_per_layer.weight"),
                262144,
                8960,
            )?,
            ple_projection: matrix(
                &format!("{prefix}per_layer_model_projection.weight"),
                8960,
                1536,
            )?,
            ple_norm: vector(&format!("{prefix}per_layer_projection_norm.weight"), 256)?,
            final_norm: vector(&format!("{prefix}norm.weight"), 1536)?,
            layers: layers
                .try_into()
                .map_err(|_| Error::Options("Gemma resident layer count"))?,
        };
        result.validate()?;
        Ok(result)
    }
    fn validate(&self) -> Result<()> {
        fn mark<const N: usize>(seen: &mut [bool; N], indices: &[usize]) -> Result<()> {
            for &index in indices {
                let value = seen
                    .get_mut(index)
                    .ok_or(Error::Options("Gemma role index outside storage"))?;
                if *value {
                    return Err(Error::Options("Gemma role index repeated"));
                }
                *value = true;
            }
            Ok(())
        }
        let mut matrices = [false; 278];
        let mut vectors = [false; 262];
        mark(
            &mut matrices,
            &[self.embedding, self.ple_embedding, self.ple_projection],
        )?;
        mark(&mut vectors, &[self.ple_norm, self.final_norm])?;
        for layer in &self.layers {
            mark(
                &mut matrices,
                &[
                    layer.q,
                    layer.o,
                    layer.gate,
                    layer.up,
                    layer.down,
                    layer.ple_gate,
                    layer.ple_projection,
                ],
            )?;
            mark(
                &mut vectors,
                &[
                    layer.input_norm,
                    layer.q_norm,
                    layer.post_attention_norm,
                    layer.pre_ff_norm,
                    layer.post_ff_norm,
                    layer.post_ple_norm,
                    layer.scalar,
                ],
            )?;
            if let Some(kv) = &layer.kv {
                mark(&mut matrices, &[kv.k, kv.v])?;
                mark(&mut vectors, &[kv.norm])?;
            }
        }
        if matrices.iter().chain(&vectors).any(|&v| !v) {
            return Err(Error::Options("Gemma role coverage incomplete"));
        }
        Ok(())
    }
    pub(super) fn layer_geometry(&self, index: usize) -> Option<LayerPlan> {
        self.layers.get(index).map(|layer| layer.geometry)
    }
}
