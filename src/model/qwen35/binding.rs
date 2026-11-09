//! Concrete pinned source binding, resolved once before numerical execution.
use super::*;
use crate::mlx::qwen35::{LayerRecipe, PayloadBytes};

struct SourceBinding<'p> {
    config: Config,
    layers: &'p [LayerRecipe],
    embedding: &'p str,
    output_head: Option<&'p str>,
    bytes: PayloadBytes,
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
            bytes: plan.payload_bytes(),
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
            bytes: plan.payload_bytes(),
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
        let bytes = plan.bytes;
        let wanted = add(
            add(bytes.affine_codes, bytes.affine_metadata)?,
            mul(add(bytes.bf16_vectors, bytes.bf16_convolutions)?, 2)?,
        )?;
        if model.learned_bytes != wanted || wanted > backing_bytes {
            return Err(invalid("Qwen3.5 source backing count"));
        }
        model.learned_bytes = backing_bytes;
        Ok(model)
    }
}
impl<'a> Model<'a> {
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
