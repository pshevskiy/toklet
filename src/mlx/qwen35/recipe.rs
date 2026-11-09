//! Fixed names/representations resolved cold, not through maps in a forward loop.
use super::*;

type Recipe = ([LayerRecipe; LAYERS], Vec<TensorPlan>, Vec<ConvolutionPlan>);

fn vector(common: &mut Vec<TensorPlan>, name: String, size: usize) {
    common.push(TensorPlan {
        name,
        shape: Shape::Vector(size),
        storage: Storage::Bf16Vector,
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
    build_for(HIDDEN, INTERMEDIATE, None)
}

// Cold factories only:4B has one tied table;9B has a distinct output projection.
pub(super) fn build_for(
    hidden: usize,
    intermediate: usize,
    output_head: Option<&str>,
) -> Result<Recipe, Error> {
    let mut common = Vec::new();
    let mut convolutions = Vec::new();
    matrix(
        &mut common,
        format!("{PREFIX}embed_tokens.weight"),
        VOCAB,
        hidden,
    )?;
    if let Some(name) = output_head {
        matrix(&mut common, name.into(), VOCAB, hidden)?;
    }
    vector(&mut common, format!("{PREFIX}norm.weight"), hidden);
    let mut kv_owner = 0;
    let mut delta_owner = 0;
    let layers = std::array::from_fn(|i| {
        if i % 4 == 3 {
            let plan = LayerRecipe::GatedFullAttention(AttentionPlan {
                kv_owner,
                query_heads: 16,
                kv_heads: 4,
                head_dim: 256,
                rotated_dims: 64,
                output_gate: true,
            });
            kv_owner += 1;
            plan
        } else {
            let plan = LayerRecipe::GatedDelta(DeltaPlan {
                recurrent_owner: delta_owner,
                convolution_owner: delta_owner,
                key_heads: 16,
                value_heads: 32,
                key_dim: 128,
                value_dim: 128,
            });
            delta_owner += 1;
            plan
        }
    });
    for (i, layer) in layers.iter().enumerate() {
        let base = format!("{PREFIX}layers.{i}.");
        for role in ["input_layernorm", "post_attention_layernorm"] {
            vector(&mut common, format!("{base}{role}.weight"), hidden);
        }
        match layer {
            LayerRecipe::GatedDelta(plan) => {
                for (role, rows, columns) in [
                    ("in_proj_a", 32, hidden),
                    ("in_proj_b", 32, hidden),
                    ("in_proj_qkv", 8192, hidden),
                    ("in_proj_z", 4096, hidden),
                    ("out_proj", hidden, 4096),
                ] {
                    matrix(
                        &mut common,
                        format!("{base}linear_attn.{role}.weight"),
                        rows,
                        columns,
                    )?;
                }
                for role in ["A_log", "dt_bias"] {
                    vector(&mut common, format!("{base}linear_attn.{role}"), 32);
                }
                vector(&mut common, format!("{base}linear_attn.norm.weight"), 128);
                convolutions.push(ConvolutionPlan {
                    layer: i,
                    history_owner: plan.convolution_owner,
                    name: format!("{base}linear_attn.conv1d.weight"),
                    channels: 8192,
                    taps: 4,
                });
            }
            LayerRecipe::GatedFullAttention(_) => {
                // Q and its output gate are one original projection, with twice
                // the query-head width. No split/copy is performed by this plan.
                for (role, rows, columns) in [
                    ("q_proj", 8192, hidden),
                    ("k_proj", 1024, hidden),
                    ("v_proj", 1024, hidden),
                    ("o_proj", hidden, 4096),
                ] {
                    matrix(
                        &mut common,
                        format!("{base}self_attn.{role}.weight"),
                        rows,
                        columns,
                    )?;
                }
                for role in ["q_norm", "k_norm"] {
                    vector(&mut common, format!("{base}self_attn.{role}.weight"), 256);
                }
            }
        }
        for (role, rows, columns) in [
            ("gate_proj", intermediate, hidden),
            ("up_proj", intermediate, hidden),
            ("down_proj", hidden, intermediate),
        ] {
            matrix(
                &mut common,
                format!("{base}mlp.{role}.weight"),
                rows,
                columns,
            )?;
        }
    }
    if common.len() != 402 + usize::from(output_head.is_some())
        || convolutions.len() != 24
        || kv_owner != 8
        || delta_owner != 24
    {
        return Err(invalid("Qwen3.5 cold recipe inventory differs"));
    }
    Ok((layers, common, convolutions))
}
