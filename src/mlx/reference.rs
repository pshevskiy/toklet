//! Explicit diagnostic-only FP32 expansion of a local MLX Qwen artifact.
//!
//! Preserves the scalar engine as an independent full-graph reference before
//! composing resident GPU operations. Never selected by a performance backend.

use super::{
    affine::Matrix,
    qwen::{Source, Storage},
};
use crate::{
    model::{self, LayerWeights, Model, Projection, Weights},
    weights::{Error, LoadOptions},
};
use std::path::Path;

pub struct Loaded {
    pub model: Model,
    pub fp32_weights_bytes: usize,
    pub temporary_packed_bytes: usize,
    /// FP32 weights + largest packed tensor + stream + metadata estimate.
    /// Session, diagnostic trace, tokenizer and application require admission.
    pub admitted_load_bytes: usize,
}
fn invalid(reason: &'static str) -> Error {
    Error::Invalid(reason.into())
}
fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b)
        .ok_or_else(|| invalid("MLX reference size overflow"))
}
fn product(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("MLX reference size overflow"))
}
fn bytes(count: usize) -> Result<Vec<u8>, Error> {
    if count > isize::MAX as usize {
        return Err(invalid("MLX reference allocation exceeds index range"));
    }
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| Error::Allocation)?;
    result.resize(count, 0);
    Ok(result)
}
fn read(source: &mut Source, name: &str) -> Result<Vec<f32>, Error> {
    let spec = source
        .tensors()
        .get(name)
        .ok_or_else(|| invalid("validated reference tensor absent"))?;
    let count = spec
        .shape()
        .iter()
        .try_fold(1usize, |n, &d| product(n, d))?;
    let storage = spec.storage();
    let mut result = model::zeros(count)?;
    match storage {
        Storage::Float(_) => {
            let mut index = 0;
            source.visit_float(name, |x| {
                *result
                    .get_mut(index)
                    .ok_or_else(|| invalid("MLX reference float element count differs"))? = x;
                index += 1;
                Ok(())
            })?;
            if index != count {
                return Err(invalid("MLX reference float element count differs"));
            }
        }
        Storage::Affine(layout) => {
            let mut weights = bytes(layout.weight_bytes())?;
            let mut scales = bytes(layout.metadata_bytes())?;
            let mut biases = bytes(layout.metadata_bytes())?;
            source.read_affine(name, &mut weights, &mut scales, &mut biases)?;
            let matrix = Matrix::from_le_bytes(layout, &weights, &scales, &biases)
                .map_err(|e| invalid(e.0))?;
            for row in 0..layout.rows() {
                for column in 0..layout.columns() {
                    result[row * layout.columns() + column] =
                        matrix.value(row, column).map_err(|e| invalid(e.0))?;
                }
            }
        }
    }
    Ok(result)
}
/// Explicit correctness/debug path. Expands once to the existing FP32 scalar
/// model, retaining no source packed model copy. Admission precedes expansion;
/// a redundant output head stays tied to the single embedding allocation.
pub fn load_qwen(directory: &Path, options: LoadOptions) -> Result<Loaded, Error> {
    let mut source = Source::open(directory, options)?;
    let config = source.config().clone();
    let mut fp32_weights_bytes = 0;
    let mut temporary_packed_bytes = 0;
    for spec in source.tensors().values() {
        let count = spec
            .shape()
            .iter()
            .try_fold(1usize, |n, &d| product(n, d))?;
        fp32_weights_bytes = add(fp32_weights_bytes, product(count, 4)?)?;
        if matches!(spec.storage(), Storage::Affine(_)) {
            temporary_packed_bytes = temporary_packed_bytes.max(spec.payload_bytes());
        }
    }
    let admitted_load_bytes = add(
        add(add(fp32_weights_bytes, temporary_packed_bytes)?, 64 * 1024)?,
        64 * 1024 * 1024,
    )?;
    if admitted_load_bytes > options.memory_limit_bytes {
        return Err(Error::Budget {
            required: admitted_load_bytes,
            limit: options.memory_limit_bytes,
        });
    }
    let embedding = read(&mut source, "model.embed_tokens.weight")?;
    let final_norm = read(&mut source, "model.norm.weight")?;
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(config.num_layers)
        .map_err(|_| Error::Allocation)?;
    for i in 0..config.num_layers {
        let p = format!("model.layers.{i}");
        let mut projection = |name| -> Result<Projection, Error> {
            Ok(Projection {
                weight: read(&mut source, &format!("{p}.self_attn.{name}_proj.weight"))?,
                bias: read(&mut source, &format!("{p}.self_attn.{name}_proj.bias"))?,
            })
        };
        let q = projection("q")?;
        let k = projection("k")?;
        let v = projection("v")?;
        layers.push(LayerWeights {
            input_norm: read(&mut source, &format!("{p}.input_layernorm.weight"))?,
            post_attention_norm: read(
                &mut source,
                &format!("{p}.post_attention_layernorm.weight"),
            )?,
            q,
            k,
            v,
            o: read(&mut source, &format!("{p}.self_attn.o_proj.weight"))?,
            gate: read(&mut source, &format!("{p}.mlp.gate_proj.weight"))?,
            up: read(&mut source, &format!("{p}.mlp.up_proj.weight"))?,
            down: read(&mut source, &format!("{p}.mlp.down_proj.weight"))?,
        });
    }
    let model = Model::new(
        config,
        Weights {
            embedding,
            layers,
            final_norm,
        },
    )?;
    Ok(Loaded {
        model,
        fp32_weights_bytes,
        temporary_packed_bytes,
        admitted_load_bytes,
    })
}
