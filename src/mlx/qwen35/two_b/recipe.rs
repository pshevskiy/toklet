//! Separate24-layer factory; the existing4B/9B factories are unchanged.
use super::*;
type Recipe = ([LayerRecipe; LAYERS], Vec<TensorPlan>, Vec<ConvolutionPlan>);
fn vector(common: &mut Vec<TensorPlan>, name: String, size: usize, f32: bool) {
    common.push(TensorPlan {
        name,
        shape: Shape::Vector(size),
        storage: if f32 {
            Storage::F32Vector
        } else {
            Storage::Bf16Vector
        },
    });
}
fn matrix(
    common: &mut Vec<TensorPlan>,
    name: String,
    rows: usize,
    columns: usize,
) -> Result<(), Error> {
    let layout = Layout::new(rows, columns, 4, 64, MetadataDtype::BF16)
        .map_err(|e| Error::Invalid(e.0.into()))?;
    common.push(TensorPlan {
        name,
        shape: Shape::Matrix { rows, columns },
        storage: Storage::Affine(layout),
    });
    Ok(())
}
pub(super) fn build() -> Result<Recipe, Error> {
    let mut common = Vec::with_capacity(302);
    let mut convolutions = Vec::with_capacity(18);
    matrix(
        &mut common,
        format!("{PREFIX}embed_tokens.weight"),
        VOCAB,
        HIDDEN,
    )?;
    vector(&mut common, format!("{PREFIX}norm.weight"), HIDDEN, false);
    let (mut kv_owner, mut delta_owner) = (0, 0);
    let layers = std::array::from_fn(|i| {
        if i % 4 == 3 {
            let recipe = LayerRecipe::GatedFullAttention(AttentionPlan {
                kv_owner,
                query_heads: 8,
                kv_heads: 2,
                head_dim: 256,
                rotated_dims: 64,
                output_gate: true,
            });
            kv_owner += 1;
            recipe
        } else {
            let recipe = LayerRecipe::GatedDelta(DeltaPlan {
                recurrent_owner: delta_owner,
                convolution_owner: delta_owner,
                key_heads: 16,
                value_heads: 16,
                key_dim: 128,
                value_dim: 128,
            });
            delta_owner += 1;
            recipe
        }
    });
    for (i, layer) in layers.iter().enumerate() {
        let base = format!("{PREFIX}layers.{i}.");
        for role in ["input_layernorm", "post_attention_layernorm"] {
            vector(&mut common, format!("{base}{role}.weight"), HIDDEN, false);
        }
        match layer {
            LayerRecipe::GatedDelta(plan) => {
                for (role, rows, columns) in [
                    ("in_proj_a", 16, HIDDEN),
                    ("in_proj_b", 16, HIDDEN),
                    ("in_proj_qkv", 6144, HIDDEN),
                    ("in_proj_z", 2048, HIDDEN),
                    ("out_proj", HIDDEN, 2048),
                ] {
                    matrix(
                        &mut common,
                        format!("{base}linear_attn.{role}.weight"),
                        rows,
                        columns,
                    )?;
                }
                vector(&mut common, format!("{base}linear_attn.A_log"), 16, true);
                vector(&mut common, format!("{base}linear_attn.dt_bias"), 16, false);
                vector(
                    &mut common,
                    format!("{base}linear_attn.norm.weight"),
                    128,
                    false,
                );
                convolutions.push(ConvolutionPlan {
                    layer: i,
                    history_owner: plan.convolution_owner,
                    name: format!("{base}linear_attn.conv1d.weight"),
                    channels: 6144,
                    taps: 4,
                });
            }
            LayerRecipe::GatedFullAttention(_) => {
                for (role, rows, columns) in [
                    ("q_proj", 4096, HIDDEN),
                    ("k_proj", 512, HIDDEN),
                    ("v_proj", 512, HIDDEN),
                    ("o_proj", HIDDEN, 2048),
                ] {
                    matrix(
                        &mut common,
                        format!("{base}self_attn.{role}.weight"),
                        rows,
                        columns,
                    )?;
                }
                for role in ["q_norm", "k_norm"] {
                    vector(
                        &mut common,
                        format!("{base}self_attn.{role}.weight"),
                        256,
                        false,
                    );
                }
            }
        }
        for (role, rows, columns) in [
            ("gate_proj", INTERMEDIATE, HIDDEN),
            ("up_proj", INTERMEDIATE, HIDDEN),
            ("down_proj", HIDDEN, INTERMEDIATE),
        ] {
            matrix(
                &mut common,
                format!("{base}mlp.{role}.weight"),
                rows,
                columns,
            )?;
        }
    }
    if common.len() != 302 || convolutions.len() != 18 || delta_owner != 18 || kv_owner != 6 {
        return Err(invalid("Qwen3.5-2B cold recipe inventory differs"));
    }
    Ok((layers, common, convolutions))
}
