//! Borrowed physical text weights in the original MLX parameter convention.

use super::{invalid, payload::TextPayload};
use crate::{
    mlx::affine::Matrix,
    model::qwen35::{AttentionWeights, DeltaWeights, LayerWeights, MixingWeights},
    model::sensenova_loop::{PHYSICAL_LAYERS, mhc::Router},
    weights::Error,
};
use half::bf16;
use safetensors::Dtype;
use std::collections::BTreeMap;

pub struct SmallWeights<'a> {
    payload: &'a TextPayload,
    values: BTreeMap<String, Vec<f32>>,
    promoted_bytes: usize,
}

pub struct TextWeights<'a> {
    pub(crate) embedding: Matrix<'a>,
    pub(crate) head: Matrix<'a>,
    pub(crate) final_norm: &'a [f32],
    pub(crate) layers: Vec<PhysicalLayer<'a>>,
}

pub(crate) struct PhysicalLayer<'a> {
    /// Norms contain the literal checkpoint coefficients. The branch execution
    /// must select the correct normalization convention for each norm.
    pub(crate) weights: LayerWeights<'a>,
    pub(crate) routers: Option<[Router<'a>; 2]>,
}

fn element_bytes(dtype: Dtype, rank: usize) -> Option<usize> {
    match (dtype, rank) {
        (Dtype::BF16, 1 | 3) => Some(2),
        (Dtype::F32, 1 | 2) => Some(4),
        _ => None,
    }
}

impl<'a> SmallWeights<'a> {
    /// The caller reserves a separate allowance for promoted parameters in
    /// addition to the still-borrowed original packed payload and metadata.
    pub fn new(payload: &'a TextPayload, promotion_limit_bytes: usize) -> Result<Self, Error> {
        let mut required = 0usize;
        for tensor in payload.plan().text_tensors().values() {
            if let Some(size) = element_bytes(tensor.dtype, tensor.shape.len()) {
                let count =
                    ((tensor.file_range.end - tensor.file_range.start) / size as u64) as usize;
                required = required
                    .checked_add(count * size_of::<f32>())
                    .ok_or_else(|| invalid("SenseNova promoted weight extent overflow"))?;
            }
        }
        if required > promotion_limit_bytes {
            return Err(Error::Budget {
                required,
                limit: promotion_limit_bytes,
            });
        }
        let mut values = BTreeMap::new();
        for (name, tensor) in payload.plan().text_tensors() {
            if let Some(size) = element_bytes(tensor.dtype, tensor.shape.len()) {
                let data = payload
                    .tensor(name)
                    .ok_or_else(|| invalid("SenseNova missing parameter payload"))?;
                let mut decoded = Vec::new();
                decoded
                    .try_reserve_exact(data.len() / size)
                    .map_err(|_| Error::Allocation)?;
                for value in data.chunks_exact(size) {
                    let value = if size == 2 {
                        bf16::from_le_bytes([value[0], value[1]]).to_f32()
                    } else {
                        f32::from_le_bytes([value[0], value[1], value[2], value[3]])
                    };
                    decoded.push(value);
                }
                values.insert(name.clone(), decoded);
            }
        }
        let promoted_bytes = values
            .values()
            .map(|v| v.capacity() * size_of::<f32>())
            .sum();
        if promoted_bytes > promotion_limit_bytes {
            return Err(Error::Budget {
                required: promoted_bytes,
                limit: promotion_limit_bytes,
            });
        }
        Ok(Self {
            payload,
            values,
            promoted_bytes,
        })
    }

    pub fn promoted_bytes(&self) -> usize {
        self.promoted_bytes
    }

    fn params(&self, name: &str, shape: &[usize]) -> Result<&[f32], Error> {
        let plan = self
            .payload
            .plan()
            .text_tensors()
            .get(name)
            .ok_or_else(|| invalid("SenseNova missing parameter descriptor"))?;
        if plan.shape != shape {
            return Err(invalid("SenseNova parameter shape differs"));
        }
        self.values
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| invalid("SenseNova missing promoted parameter"))
    }

    fn matrix(&self, name: &str) -> Result<Matrix<'_>, Error> {
        let layout = *self
            .payload
            .plan()
            .affine_layouts()
            .get(name)
            .ok_or_else(|| invalid("SenseNova missing affine layout"))?;
        let component = |part| {
            self.payload
                .tensor(&format!("{name}.{part}"))
                .ok_or_else(|| invalid("SenseNova missing affine component"))
        };
        Matrix::from_le_bytes(
            layout,
            component("weight")?,
            component("scales")?,
            component("biases")?,
        )
        .map_err(|error| invalid(error.0))
    }

    /// Bind once per physical table, then borrow across repeated execution visits.
    pub fn view(&self) -> Result<TextWeights<'_>, Error> {
        let mut layers = Vec::with_capacity(PHYSICAL_LAYERS);
        for i in 0..PHYSICAL_LAYERS {
            let prefix = format!("language_model.model.layers.{i}");
            let matrix = |suffix| self.matrix(&format!("{prefix}.{suffix}"));
            let params =
                |suffix, shape: &[usize]| self.params(&format!("{prefix}.{suffix}"), shape);
            let mixing = if i % 4 == 3 {
                MixingWeights::Attention(AttentionWeights {
                    q_gate: matrix("self_attn.q_proj")?,
                    k: matrix("self_attn.k_proj")?,
                    v: matrix("self_attn.v_proj")?,
                    out: matrix("self_attn.o_proj")?,
                    q_norm: params("self_attn.q_norm.weight", &[256])?,
                    k_norm: params("self_attn.k_norm.weight", &[256])?,
                })
            } else {
                MixingWeights::Delta(DeltaWeights {
                    qkv: matrix("linear_attn.in_proj_qkv")?,
                    z: matrix("linear_attn.in_proj_z")?,
                    a: matrix("linear_attn.in_proj_a")?,
                    b: matrix("linear_attn.in_proj_b")?,
                    out: matrix("linear_attn.out_proj")?,
                    convolution: params("linear_attn.conv1d.weight", &[8192, 4, 1])?,
                    a_log: params("linear_attn.A_log", &[32])?,
                    dt_bias: params("linear_attn.dt_bias", &[32])?,
                    norm: params("linear_attn.norm.weight", &[128])?,
                })
            };
            let routers = if (4..28).contains(&i) {
                let router = |branch| {
                    let weight = self.params(&format!("{prefix}.{branch}.weight"), &[8, 8192])?;
                    let bias = self.params(&format!("{prefix}.{branch}.bias"), &[8])?;
                    let bias = bias
                        .try_into()
                        .map_err(|_| invalid("SenseNova router bias extent differs"))?;
                    Router::new(4096, weight, bias).map_err(Error::Model)
                };
                Some([router("mhc_attn")?, router("mhc_ffn")?])
            } else {
                None
            };
            layers.push(PhysicalLayer {
                weights: LayerWeights {
                    input_norm: params("input_layernorm.weight", &[4096])?,
                    post_norm: params("post_attention_layernorm.weight", &[4096])?,
                    gate: matrix("mlp.gate_proj")?,
                    up: matrix("mlp.up_proj")?,
                    down: matrix("mlp.down_proj")?,
                    mixing,
                },
                routers,
            });
        }
        Ok(TextWeights {
            embedding: self.matrix("language_model.model.embed_tokens")?,
            head: self.matrix("language_model.lm_head")?,
            final_norm: self.params("language_model.model.norm.weight", &[4096])?,
            layers,
        })
    }
}
