//! Cold exact E2B name/layout resolution; no tensor-name lookup during forward.
use super::*;
use crate::mlx::gemma4::{E2bPlan, payload::PayloadStore};

const PREFIX: &str = "language_model.model.";

impl Layout {
    fn e2b(plan: &E2bPlan) -> Result<Self, Error> {
        Self::e2b_layers(plan.layers())
    }
    fn e2b_layers(geometry: &[crate::mlx::gemma4::LayerPlan]) -> Result<Self, Error> {
        if geometry.len() != 35 {
            return Err(Error::InvalidConfig("Gemma E2B geometry count"));
        }
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(geometry.len())
            .map_err(|_| Error::Allocation)?;
        for g in geometry {
            layers.push(LayerLayout {
                kind: g.kind,
                head_dim: g.head_dim,
                rotated_dims: g.rotated_dims,
                intermediate: g.intermediate,
                theta: if g.kind == KvKind::Full { 1e6 } else { 1e4 },
            });
        }
        // The cold plan is constructible publicly only through exact audited
        // config/index/header admission. Declared131072 is not execution capacity.
        Ok(Self {
            hidden: 1536,
            vocab: 262144,
            ple: 256,
            heads: 8,
            producers: 15,
            positions: MAX_POSITIONS,
            window: 512,
            layers,
        })
    }
}

#[cfg(all(
    feature = "metal",
    any(target_os = "macos", target_os = "ios"),
    target_arch = "aarch64"
))]
impl<'a, M: MatrixSource> Model<'a, M> {
    pub(crate) fn from_e2b_parts(
        weights: Weights<'a, M>,
        geometry: &[crate::mlx::gemma4::LayerPlan],
        backing: Backing,
    ) -> Result<Self, Error> {
        let result = Self::with_storage(weights, Layout::e2b_layers(geometry)?, Some(backing))?;
        for (index, layer) in geometry.iter().enumerate() {
            if result.plan.producer_for(index)? != layer.producer {
                return Err(Error::InvalidConfig("Gemma E2B shared producer differs"));
            }
        }
        Ok(result)
    }
}

impl<'a> Model<'a> {
    /// Borrow the exact admitted packed4/group64/BF16 E2B text payload.
    ///
    /// This scalar diagnostic graph admits only32 total positions. It supplies
    /// raw logits, not tokenizer/chat/generation, a qualified source-BF16/GPU
    /// backend or app eligibility. Keep store -> model -> session owners alive;
    /// no tensors are copied or expanded by this adapter.
    pub fn from_e2b(store: &'a PayloadStore) -> Result<Self, Error> {
        let matrix = |name: &str| {
            store
                .matrix(&format!("{PREFIX}{name}"))
                .map_err(|_| Error::InvalidConfig("Gemma E2B packed tensor"))
        };
        let vector = |name: &str| {
            store
                .vector(&format!("{PREFIX}{name}"))
                .map_err(|_| Error::InvalidConfig("Gemma E2B vector"))
        };
        let layout = Layout::e2b(store.plan())?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(layout.layers.len())
            .map_err(|_| Error::Allocation)?;
        for i in 0..layout.layers.len() {
            let base = format!("layers.{i}.");
            let m = |name: &str| matrix(&format!("{base}{name}.weight"));
            let v = |name: &str| vector(&format!("{base}{name}.weight"));
            let scalar = vector(&format!("{base}layer_scalar"))?;
            length("Gemma E2B layer scalar", scalar.len(), 1)?;
            layers.push(LayerWeights {
                input_norm: v("input_layernorm")?,
                q: m("self_attn.q_proj")?,
                q_norm: v("self_attn.q_norm")?,
                o: m("self_attn.o_proj")?,
                kv: if i < layout.producers {
                    Some(KvWeights {
                        k: m("self_attn.k_proj")?,
                        v: m("self_attn.v_proj")?,
                        k_norm: v("self_attn.k_norm")?,
                    })
                } else {
                    None
                },
                post_attention_norm: v("post_attention_layernorm")?,
                pre_ff_norm: v("pre_feedforward_layernorm")?,
                post_ff_norm: v("post_feedforward_layernorm")?,
                ff_gate: m("mlp.gate_proj")?,
                ff_up: m("mlp.up_proj")?,
                ff_down: m("mlp.down_proj")?,
                ple_gate: m("per_layer_input_gate")?,
                ple_projection: m("per_layer_projection")?,
                post_ple_norm: v("post_per_layer_input_norm")?,
                scalar: scalar[0],
            });
        }
        let weights = Weights {
            embedding: matrix("embed_tokens.weight")?,
            per_layer_embedding: matrix("embed_tokens_per_layer.weight")?,
            per_layer_projection: matrix("per_layer_model_projection.weight")?,
            per_layer_norm: vector("per_layer_projection_norm.weight")?,
            layers,
            final_norm: vector("norm.weight")?,
        };
        let mut model = Self::with_storage(weights, layout, Some(Backing::cpu(store.report())))?;
        for (i, layer) in store.plan().layers().iter().enumerate() {
            if model.plan.producer_for(i)? != layer.producer {
                return Err(Error::InvalidConfig("Gemma E2B shared producer differs"));
            }
        }
        model.content_identity = Some(*store.content_identity());
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlx::gemma4::{ModalityPolicy, inspect};

    #[test]
    fn real_plan_uses_named_geometry_without_declared_context_allocations() {
        let plan = inspect(
            include_bytes!("../../../tests/fixtures/gemma4-e2b-metadata-v1/config.json"),
            include_bytes!("../../../tests/fixtures/gemma4-e2b-metadata-v1/index.json"),
            include_bytes!("../../../tests/fixtures/gemma4-e2b-metadata-v1/header.json"),
            3550670554,
            ModalityPolicy::TextOnly,
        )
        .unwrap();
        let layout = Layout::e2b(&plan).unwrap();
        assert_eq!(
            (layout.hidden, layout.vocab, layout.ple, layout.heads),
            (1536, 262144, 256, 8)
        );
        assert_eq!((layout.layers.len(), layout.producers), (35, 15));
        assert_eq!((layout.positions, layout.window), (32, 512));
        assert_eq!(plan.declared_total_positions(), 131072);
        for (i, layer) in layout.layers.iter().enumerate() {
            let full = i % 5 == 4;
            assert_eq!(
                layer.kind,
                if full { KvKind::Full } else { KvKind::Sliding }
            );
            assert_eq!(layer.head_dim, if full { 512 } else { 256 });
            assert_eq!(layer.rotated_dims, if full { 128 } else { 256 });
            assert_eq!(layer.theta, if full { 1e6 } else { 1e4 });
            assert_eq!(layer.intermediate, if i < 15 { 6144 } else { 12288 });
        }
        let counts = Scratch::counts(&layout, MAX_POSITIONS).unwrap();
        assert_eq!(counts[4], 32 * 8 * 512);
        assert_eq!(counts[16], 32 * 12288);
        assert_eq!(counts[21], 32 * 262144);
        assert_eq!(counts[22], 32 * 262144);
        println!(
            "Gemma E2B bounded scratch: {} bytes",
            total_bytes(&counts).unwrap()
        );
    }
}
