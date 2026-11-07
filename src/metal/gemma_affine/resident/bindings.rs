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
pub(in crate::metal::gemma_affine) struct Bindings {
    embedding: usize,
    ple_embedding: usize,
    ple_projection: usize,
    ple_norm: usize,
    final_norm: usize,
    layers: [LayerBindings; 35],
}
impl Bindings {
    #[cfg(target_os = "macos")]
    pub(in crate::metal::gemma_affine) fn operation_identity(
        &self,
        source: usize,
    ) -> Result<crate::mlx::gemma4::bf16::execution::OperationId> {
        use crate::mlx::gemma4::bf16::execution::{OperationId, Role};
        for (index, role) in [
            (self.embedding, Role::TokenEmbedding),
            (self.ple_embedding, Role::PerLayerEmbedding),
            (self.ple_projection, Role::PerLayerInputProjection),
        ] {
            if source == index {
                return Ok(OperationId { role, layer: None });
            }
        }
        for (index, layer) in self.layers.iter().enumerate() {
            let layer_id = Some(
                u16::try_from(index)
                    .map_err(|_| Error::Options("Gemma operation layer overflow"))?,
            );
            for (slot, role) in [
                (layer.q, Role::Query),
                (layer.o, Role::AttentionOutput),
                (layer.gate, Role::FeedForwardGate),
                (layer.up, Role::FeedForwardUp),
                (layer.down, Role::FeedForwardDown),
                (layer.ple_gate, Role::PerLayerGate),
                (layer.ple_projection, Role::PerLayerProjection),
            ] {
                if slot == source {
                    return Ok(OperationId {
                        role,
                        layer: layer_id,
                    });
                }
            }
            if let Some(kv) = &layer.kv {
                for (slot, role) in [(kv.k, Role::Key), (kv.v, Role::Value)] {
                    if slot == source {
                        return Ok(OperationId {
                            role,
                            layer: layer_id,
                        });
                    }
                }
            }
        }
        Err(Error::Options("Gemma operation source absent"))
    }
    #[cfg(target_os = "macos")]
    pub(in crate::metal::gemma_affine) fn operation_source(
        &self,
        id: crate::mlx::gemma4::bf16::execution::OperationId,
    ) -> Result<usize> {
        use crate::mlx::gemma4::bf16::execution::Role;
        if let Some(index) = id.layer {
            let layer = self
                .layers
                .get(usize::from(index))
                .ok_or(Error::Options("Gemma operation layer differs"))?;
            return Ok(match id.role {
                Role::Query => layer.q,
                Role::AttentionOutput => layer.o,
                Role::FeedForwardGate => layer.gate,
                Role::FeedForwardUp => layer.up,
                Role::FeedForwardDown => layer.down,
                Role::PerLayerGate => layer.ple_gate,
                Role::PerLayerProjection => layer.ple_projection,
                Role::Key => {
                    layer
                        .kv
                        .as_ref()
                        .ok_or(Error::Options("Gemma consumer has no K projection"))?
                        .k
                }
                Role::Value => {
                    layer
                        .kv
                        .as_ref()
                        .ok_or(Error::Options("Gemma consumer has no V projection"))?
                        .v
                }
                _ => return Err(Error::Options("Gemma operation scope differs")),
            });
        }
        Ok(match id.role {
            Role::TokenEmbedding | Role::OutputHead => self.embedding,
            Role::PerLayerEmbedding => self.ple_embedding,
            Role::PerLayerInputProjection => self.ple_projection,
            _ => return Err(Error::Options("Gemma operation scope differs")),
        })
    }
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
        Self::resolve(cold.layers(), "language_model.model.", matrix, vector)
    }

    #[cfg(target_os = "macos")]
    pub(in crate::metal::gemma_affine) fn bf16(
        cold: &crate::mlx::gemma4::bf16::TextPlan,
    ) -> Result<Self> {
        let selected = cold.text_tensors();
        let find = |name: &str, shape: &[usize]| {
            selected
                .iter()
                .filter(|(_, tensor)| tensor.shape.len() == shape.len())
                .position(|(key, tensor)| key == name && tensor.shape == shape)
                .ok_or(Error::Options("Gemma BF16 named tensor differs"))
        };
        Self::resolve(
            cold.layers(),
            "model.language_model.",
            |name, rows, columns| find(name, &[rows, columns]),
            |name, values| find(name, &[values]),
        )
    }

    fn resolve(
        geometry: &[LayerPlan; 35],
        prefix: &str,
        matrix: impl Fn(&str, usize, usize) -> Result<usize>,
        vector: impl Fn(&str, usize) -> Result<usize>,
    ) -> Result<Self> {
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(35)
            .map_err(|_| Error::Native("Gemma role reservation failed".into()))?;
        for (index, &geometry) in geometry.iter().enumerate() {
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
    pub(in crate::metal::gemma_affine) fn embedding_index(&self) -> usize {
        self.embedding
    }
    pub(in crate::metal::gemma_affine) fn ple_embedding_index(&self) -> usize {
        self.ple_embedding
    }
    pub(in crate::metal::gemma_affine) fn geometry(&self) -> [LayerPlan; 35] {
        self.layers.each_ref().map(|layer| layer.geometry)
    }
    pub(super) fn graph_weights<'a, 'weights>(
        &self,
        norms: &'a super::hybrid::Norms<'weights>,
    ) -> Result<crate::model::gemma_packed::Weights<'a, super::hybrid::MatrixRef<'a, 'weights>>>
    {
        self.graph_weights_for(|index| norms.matrix(index), |index| norms.vector(index))
    }

    pub(in crate::metal::gemma_affine) fn graph_weights_for<'a, M>(
        &self,
        m: impl Fn(usize) -> Result<M>,
        v: impl Fn(usize) -> Result<&'a [f32]>,
    ) -> Result<crate::model::gemma_packed::Weights<'a, M>> {
        use crate::model::gemma_packed::{KvWeights, LayerWeights, Weights};
        let mut layers = owner_vec(self.layers.len())?;
        for layer in &self.layers {
            layers.push(LayerWeights {
                input_norm: v(layer.input_norm)?,
                q: m(layer.q)?,
                q_norm: v(layer.q_norm)?,
                o: m(layer.o)?,
                kv: layer
                    .kv
                    .as_ref()
                    .map(|kv| {
                        Ok::<_, Error>(KvWeights {
                            k: m(kv.k)?,
                            v: m(kv.v)?,
                            k_norm: v(kv.norm)?,
                        })
                    })
                    .transpose()?,
                post_attention_norm: v(layer.post_attention_norm)?,
                pre_ff_norm: v(layer.pre_ff_norm)?,
                post_ff_norm: v(layer.post_ff_norm)?,
                ff_gate: m(layer.gate)?,
                ff_up: m(layer.up)?,
                ff_down: m(layer.down)?,
                ple_gate: m(layer.ple_gate)?,
                ple_projection: m(layer.ple_projection)?,
                post_ple_norm: v(layer.post_ple_norm)?,
                scalar: *v(layer.scalar)?
                    .first()
                    .ok_or(Error::Options("Gemma hybrid scalar role"))?,
            });
        }
        Ok(Weights {
            embedding: m(self.embedding)?,
            per_layer_embedding: m(self.ple_embedding)?,
            per_layer_projection: m(self.ple_projection)?,
            per_layer_norm: v(self.ple_norm)?,
            layers,
            final_norm: v(self.final_norm)?,
        })
    }
    pub(super) fn layer_geometry(&self, index: usize) -> Option<LayerPlan> {
        self.layers.get(index).map(|layer| layer.geometry)
    }
}
