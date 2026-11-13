//! Concrete pinned source binding, resolved once before numerical execution.
use super::*;
use crate::mlx::qwen35::{LayerRecipe, PayloadBytes};

struct SourceBinding<'p> {
    config: Config,
    layers: &'p [LayerRecipe],
    embedding: &'p str,
    output_head: Option<&'p str>,
    runtime_learned_bytes: usize,
}
fn promoted_bf16_bytes(bytes: PayloadBytes) -> Result<usize> {
    add(
        add(bytes.affine_codes, bytes.affine_metadata)?,
        mul(add(bytes.bf16_vectors, bytes.bf16_convolutions)?, 2)?,
    )
}
fn geometry(
    hidden: usize,
    intermediate: usize,
    vocab: usize,
    layers: &[LayerRecipe],
    max_positions: usize,
    rope_theta: f64,
) -> Result<Config> {
    let Some(LayerRecipe::GatedDelta(d)) = layers.first() else {
        return Err(invalid("Qwen3.5 first delta recipe"));
    };
    let Some(LayerRecipe::GatedFullAttention(a)) = layers.get(3) else {
        return Err(invalid("Qwen3.5 fourth attention recipe"));
    };
    let config = Config {
        hidden,
        intermediate,
        vocab,
        layers: layers.len(),
        query_heads: a.query_heads,
        kv_heads: a.kv_heads,
        head_dim: a.head_dim,
        rotated_dims: a.rotated_dims,
        key_heads: d.key_heads,
        value_heads: d.value_heads,
        key_dim: d.key_dim,
        value_dim: d.value_dim,
        max_positions,
        rope_theta: rope_theta as f32,
        rms_epsilon: 1e-6,
    };
    config.validate()?;
    Ok(config)
}
impl SourceBinding<'_> {
    fn four_b(plan: &crate::mlx::qwen35::Plan) -> Result<SourceBinding<'_>> {
        Ok(SourceBinding {
            config: geometry(
                plan.hidden_size(),
                plan.intermediate_size(),
                plan.vocab_size(),
                plan.layers(),
                plan.declared_total_positions(),
                plan.rope_theta(),
            )?,
            layers: plan.layers(),
            embedding: &plan.embedding().name,
            output_head: None,
            runtime_learned_bytes: promoted_bf16_bytes(plan.payload_bytes())?,
        })
    }
    fn nine_b(plan: &crate::mlx::qwen35::nine_b::Plan) -> Result<SourceBinding<'_>> {
        Ok(SourceBinding {
            config: geometry(
                plan.hidden_size(),
                plan.intermediate_size(),
                plan.vocab_size(),
                plan.layers(),
                plan.declared_total_positions(),
                plan.rope_theta(),
            )?,
            layers: plan.layers(),
            embedding: &plan.embedding().name,
            output_head: Some(&plan.output_head().name),
            runtime_learned_bytes: promoted_bf16_bytes(plan.payload_bytes())?,
        })
    }
    fn two_b(plan: &crate::mlx::qwen35::two_b::Plan) -> Result<SourceBinding<'_>> {
        Ok(SourceBinding {
            config: geometry(
                plan.hidden_size(),
                plan.intermediate_size(),
                plan.vocab_size(),
                plan.layers(),
                plan.declared_total_positions(),
                plan.rope_theta(),
            )?,
            layers: plan.layers(),
            embedding: &plan.embedding().name,
            output_head: None,
            // F32 A_log source spans are retained exactly, while only BF16
            // vectors/convolutions double in size on promotion to CPU f32.
            runtime_learned_bytes: usize::try_from(plan.runtime_learned_bytes()?)
                .map_err(|_| cpu::Error::SizeOverflow)?,
        })
    }
}
impl<'a, M: MatrixStorage> Model<'a, M> {
    // The4B CPU/Metal callers and their tied semantics keep this entrypoint.
    pub(super) fn bind_source(
        plan: &crate::mlx::qwen35::Plan,
        matrix: impl FnMut(&str) -> Result<M>,
        vector: impl FnMut(&str) -> Result<&'a [f32]>,
        convolution: impl FnMut(&str) -> Result<&'a [f32]>,
        backing_bytes: usize,
    ) -> Result<Self> {
        Self::bind_recipe(
            SourceBinding::four_b(plan)?,
            matrix,
            vector,
            convolution,
            backing_bytes,
        )
    }
    pub(super) fn bind_nine_b_source(
        plan: &crate::mlx::qwen35::nine_b::Plan,
        matrix: impl FnMut(&str) -> Result<M>,
        vector: impl FnMut(&str) -> Result<&'a [f32]>,
        convolution: impl FnMut(&str) -> Result<&'a [f32]>,
        backing_bytes: usize,
    ) -> Result<Self> {
        Self::bind_recipe(
            SourceBinding::nine_b(plan)?,
            matrix,
            vector,
            convolution,
            backing_bytes,
        )
    }
    pub(super) fn bind_two_b_source(
        plan: &crate::mlx::qwen35::two_b::Plan,
        matrix: impl FnMut(&str) -> Result<M>,
        vector: impl FnMut(&str) -> Result<&'a [f32]>,
        convolution: impl FnMut(&str) -> Result<&'a [f32]>,
        backing_bytes: usize,
    ) -> Result<Self> {
        Self::bind_recipe(
            SourceBinding::two_b(plan)?,
            matrix,
            vector,
            convolution,
            backing_bytes,
        )
    }
    fn bind_recipe(
        plan: SourceBinding<'_>,
        mut matrix: impl FnMut(&str) -> Result<M>,
        mut vector: impl FnMut(&str) -> Result<&'a [f32]>,
        mut convolution: impl FnMut(&str) -> Result<&'a [f32]>,
        backing_bytes: usize,
    ) -> Result<Self> {
        let config = plan.config;
        let prefix = "language_model.model.";
        let mut layers = reserved(config.layers)?;
        for (i, kind) in plan.layers.iter().enumerate() {
            let b = format!("{prefix}layers.{i}.");
            let mut mat = |role: &str| matrix(&format!("{b}{role}.weight"));
            let mut vec = |role: &str| vector(&format!("{b}{role}"));
            let mixing = match kind {
                LayerRecipe::GatedDelta(_) => MixingWeights::Delta(DeltaWeights {
                    qkv: mat("linear_attn.in_proj_qkv")?,
                    z: mat("linear_attn.in_proj_z")?,
                    a: mat("linear_attn.in_proj_a")?,
                    b: mat("linear_attn.in_proj_b")?,
                    out: mat("linear_attn.out_proj")?,
                    convolution: convolution(&format!("{b}linear_attn.conv1d.weight"))?,
                    a_log: vec("linear_attn.A_log")?,
                    dt_bias: vec("linear_attn.dt_bias")?,
                    norm: vec("linear_attn.norm.weight")?,
                }),
                LayerRecipe::GatedFullAttention(_) => MixingWeights::Attention(AttentionWeights {
                    q_gate: mat("self_attn.q_proj")?,
                    k: mat("self_attn.k_proj")?,
                    v: mat("self_attn.v_proj")?,
                    out: mat("self_attn.o_proj")?,
                    q_norm: vec("self_attn.q_norm.weight")?,
                    k_norm: vec("self_attn.k_norm.weight")?,
                }),
            };
            layers.push(LayerWeights {
                input_norm: vec("input_layernorm.weight")?,
                post_norm: vec("post_attention_layernorm.weight")?,
                gate: mat("mlp.gate_proj")?,
                up: mat("mlp.up_proj")?,
                down: mat("mlp.down_proj")?,
                mixing,
            });
        }
        let weights = Weights {
            embedding: matrix(plan.embedding)?,
            final_norm: vector(&format!("{prefix}norm.weight"))?,
            layers,
        };
        let mut model = match plan.output_head {
            Some(name) => Self::new_untied(config, weights, matrix(name)?)?,
            None => Self::new(config, weights)?,
        };
        let wanted = plan.runtime_learned_bytes;
        if model.learned_bytes != wanted || wanted > backing_bytes {
            return Err(invalid("Qwen3.5 source backing count"));
        }
        model.learned_bytes = backing_bytes;
        Ok(model)
    }
}
impl<'a> Model<'a> {
    /// Explicit tied-head2B diagnostic binding. F32 recurrent decay parameters
    /// remain borrowed at their original precision; no codec, numerical, quality
    /// or ordinary managed-route qualification is implied.
    pub fn from_two_b_cpu_store(
        store: &'a crate::mlx::qwen35::two_b::payload::PayloadStore,
    ) -> Result<Self> {
        Self::bind_two_b_source(
            store.plan(),
            |name| Ok(store.matrix(name)?),
            |name| Ok(store.vector(name)?),
            |name| Ok(store.convolution(name)?),
            store.report().retained_tensor_capacity_bytes,
        )
    }
    /// Diagnostic binding of the completely authenticated9B CPU store. This
    /// selects its distinct source head; it grants no numerical/context, codec,
    /// GPU, quality, request or managed-route qualification.
    pub fn from_nine_b_cpu_store(
        store: &'a crate::mlx::qwen35::nine_b::payload::PayloadStore,
    ) -> Result<Self> {
        Self::bind_nine_b_source(
            store.plan(),
            |name| Ok(store.matrix(name)?),
            |name| Ok(store.vector(name)?),
            |name| Ok(store.convolution(name)?),
            store.report().retained_tensor_capacity_bytes,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlx::affine::Layout;
    use crate::mlx::qwen35::{AttentionPlan, DeltaPlan};

    struct ShapeMatrix(Layout);
    impl execution::sealed::Matrix for ShapeMatrix {}
    impl MatrixStorage for ShapeMatrix {
        fn layout(&self) -> Layout {
            self.0
        }
    }

    fn two_b_layers() -> Vec<LayerRecipe> {
        (0..24)
            .map(|layer| {
                if layer % 4 == 3 {
                    LayerRecipe::GatedFullAttention(AttentionPlan {
                        kv_owner: layer / 4,
                        query_heads: 8,
                        kv_heads: 2,
                        head_dim: 256,
                        rotated_dims: 64,
                        output_gate: true,
                    })
                } else {
                    LayerRecipe::GatedDelta(DeltaPlan {
                        recurrent_owner: layer - layer / 4,
                        convolution_owner: layer - layer / 4,
                        key_heads: 16,
                        value_heads: 16,
                        key_dim: 128,
                        value_dim: 128,
                    })
                }
            })
            .collect()
    }

    #[test]
    fn two_b_recipe_derives_distinct_ratio_one_geometry_without_payloads() {
        let layers = two_b_layers();
        let c = geometry(2048, 6144, 248320, &layers, 262144, 10_000_000.).unwrap();
        assert_eq!(c.layers, 24);
        assert_eq!(c.delta_count(), 18);
        assert_eq!(c.attention_count(), 6);
        assert_eq!(c.query_width().unwrap(), 2048);
        assert_eq!(c.kv_width().unwrap(), 512);
        assert_eq!(c.key_width().unwrap(), 2048);
        assert_eq!(c.value_width().unwrap(), 2048);
        assert_eq!(c.conv_width().unwrap(), 6144);
        assert_eq!(c.key_heads, c.value_heads);
        assert_eq!(c.rotated_dims, 64);
        for count in [0, 1, 3, 23] {
            let prefix = &layers[..count];
            assert!(geometry(2048, 6144, 248320, prefix, 262144, 10_000_000.).is_err());
        }
        let mut wrong = layers.clone();
        wrong.swap(0, 3);
        assert!(geometry(2048, 6144, 248320, &wrong, 262144, 10_000_000.).is_err());
    }

    #[test]
    fn legacy_bf16_runtime_accounting_preserves_promotions_and_overflow_checks() {
        assert_eq!(
            promoted_bf16_bytes(PayloadBytes {
                affine_codes: 32,
                affine_metadata: 8,
                bf16_vectors: 6,
                bf16_convolutions: 10,
                total: 56,
            })
            .unwrap(),
            72
        );
        for bytes in [
            PayloadBytes {
                affine_codes: usize::MAX,
                affine_metadata: 1,
                ..PayloadBytes::default()
            },
            PayloadBytes {
                bf16_vectors: usize::MAX,
                ..PayloadBytes::default()
            },
            PayloadBytes {
                bf16_vectors: usize::MAX,
                bf16_convolutions: 1,
                ..PayloadBytes::default()
            },
        ] {
            assert!(promoted_bf16_bytes(bytes).is_err());
        }
    }

    fn shape(name: &str) -> Result<ShapeMatrix> {
        let dimensions = if name.ends_with("embed_tokens.weight") {
            (248320, 2048)
        } else if name.ends_with("mlp.gate_proj.weight")
            || name.ends_with("mlp.up_proj.weight")
            || name.ends_with("linear_attn.in_proj_qkv.weight")
        {
            (6144, 2048)
        } else if name.ends_with("mlp.down_proj.weight") {
            (2048, 6144)
        } else if name.ends_with("linear_attn.in_proj_a.weight")
            || name.ends_with("linear_attn.in_proj_b.weight")
        {
            (16, 2048)
        } else if name.ends_with("self_attn.q_proj.weight") {
            (4096, 2048)
        } else if name.ends_with("self_attn.k_proj.weight")
            || name.ends_with("self_attn.v_proj.weight")
        {
            (512, 2048)
        } else if name.ends_with("linear_attn.in_proj_z.weight")
            || name.ends_with("linear_attn.out_proj.weight")
            || name.ends_with("self_attn.o_proj.weight")
        {
            (2048, 2048)
        } else {
            return Err(invalid("unexpected test matrix name"));
        };
        Ok(ShapeMatrix(
            Layout::new(dimensions.0, dimensions.1, 4, 64, MetadataDtype::BF16).unwrap(),
        ))
    }

    fn counted_two_b(
        values: &[f32],
        wanted: usize,
        backing: usize,
        wrong_decay_length: bool,
    ) -> Result<Model<'_, ShapeMatrix>> {
        let layers = two_b_layers();
        Model::bind_recipe(
            SourceBinding {
                config: geometry(2048, 6144, 248320, &layers, 262144, 10_000_000.)?,
                layers: &layers,
                embedding: "language_model.model.embed_tokens.weight",
                output_head: None,
                runtime_learned_bytes: wanted,
            },
            shape,
            |name| {
                let n = if name.ends_with("linear_attn.A_log") {
                    if wrong_decay_length { 15 } else { 16 }
                } else if name.ends_with("linear_attn.dt_bias") {
                    16
                } else if name.ends_with("linear_attn.norm.weight") {
                    128
                } else if name.ends_with("self_attn.q_norm.weight")
                    || name.ends_with("self_attn.k_norm.weight")
                {
                    256
                } else {
                    2048
                };
                Ok(&values[..n])
            },
            |_| Ok(&values[..6144 * 4]),
            backing,
        )
    }

    #[test]
    fn two_b_tied_shape_binding_counts_f32_once_and_preserves_decay_bits() {
        // Shape-only matrices carry no payload; this is a cold binding witness,
        // not numerical or full-model memory qualification.
        let values = vec![f32::from_bits(0x3f80_1234); 6144 * 4];
        let wanted = 1_060_412_672;
        let model = counted_two_b(&values, wanted, wanted + 64, false).unwrap();
        assert!(model.output_head.is_none());
        assert_eq!(model.layers.len(), 24);
        assert_eq!(model.learned_bytes, wanted + 64);
        let MixingWeights::Delta(delta) = &model.layers[0].weights.mixing else {
            panic!("first layer is delta")
        };
        assert_eq!(delta.a_log.as_ptr(), values.as_ptr());
        assert!(delta.a_log.iter().all(|v| v.to_bits() == 0x3f80_1234));
        assert!(counted_two_b(&values, wanted - 1, wanted, false).is_err());
        assert!(counted_two_b(&values, wanted, wanted - 1, false).is_err());
        assert!(counted_two_b(&values, wanted, wanted, true).is_err());
    }
}
