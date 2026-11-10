//! Audited loader for the canonical LFM3 BF16 repackaging.
use std::{path::Path, time::Instant};

use half::bf16;
use safetensors::{Dtype, tensor::Metadata};

use super::{
    Error, INITIAL_MEMORY_LIMIT_BYTES,
    content::{ExpectedFile, FileAudit, Manifest},
    invalid, parse_safetensors_header,
};
use crate::model::{
    lfm3::{HEAD_DIM, HIDDEN, INTERMEDIATE, KV_HEADS, LAYERS, Model, QUERY_HEADS, VOCAB, reserve},
    lfm3_state::CONV_LAYERS,
};

pub(crate) const PARAMETERS: usize = 1_728_801_536;
const BYTES: u64 = 3_457_635_448;
const ORIGIN: usize = 32_376;
const SHA256: [u8; 32] = [
    0xb6, 0xff, 0x17, 0xac, 0x99, 0x3d, 0xdd, 0xf0, 0xf6, 0xf5, 0x1b, 0xd8, 0x72, 0xb1, 0x2a, 0xe1,
    0x43, 0x7f, 0x44, 0x53, 0xa5, 0x4a, 0x0b, 0x48, 0x78, 0xce, 0xdc, 0x78, 0x64, 0xfb, 0xc0, 0x7e,
];
// Exact 32 KiB header, parsed metadata, cold 64 KiB reader and temporary names.
const LOAD_ALLOWANCE: usize = 2 * 1024 * 1024;

#[derive(Clone, Copy)]
pub(crate) struct Tensor {
    begin: usize,
    end: usize,
}

#[derive(Clone, Copy)]
pub(crate) enum Operator {
    Convolution {
        bcx: Tensor,
        coefficients: Tensor,
        output: Tensor,
    },
    Attention {
        q: Tensor,
        k: Tensor,
        v: Tensor,
        gate: Tensor,
        q_norm: Tensor,
        k_norm: Tensor,
        output: Tensor,
    },
}

#[derive(Clone, Copy)]
pub(crate) struct Layer {
    pub operator: Operator,
    pub operator_norm: Tensor,
    pub post_operator_norm: Tensor,
    pub ffn_norm: Tensor,
    pub post_ffn_norm: Tensor,
    pub gate: Tensor,
    pub up: Tensor,
    pub down: Tensor,
}

pub(crate) struct Weights {
    data: Vec<bf16>,
    pub embedding: Tensor,
    pub final_norm: Tensor,
    pub layers: [Layer; LAYERS],
}

impl Weights {
    pub fn get(&self, tensor: Tensor) -> &[bf16] {
        &self.data[tensor.begin..tensor.end]
    }
}

fn tensor(metadata: &Metadata, name: &str, shape: &[usize]) -> Result<Tensor, Error> {
    let info = metadata
        .info(name)
        .ok_or_else(|| invalid(format!("missing LFM3 tensor {name}")))?;
    if info.dtype != Dtype::BF16 || info.shape != shape {
        return Err(invalid(format!("LFM3 tensor shape or dtype: {name}")));
    }
    let (begin, end) = info.data_offsets;
    if begin % 2 != 0 || end % 2 != 0 {
        return Err(invalid("unaligned LFM3 tensor offset"));
    }
    Ok(Tensor {
        begin: begin / 2,
        end: end / 2,
    })
}

fn parse_layer(metadata: &Metadata, i: usize) -> Result<Layer, Error> {
    let take = |name: &str, shape: &[usize]| {
        tensor(metadata, &format!("model.layers.{i}.{name}.weight"), shape)
    };
    let operator = if CONV_LAYERS.contains(&i) {
        Operator::Convolution {
            bcx: take("conv.in_proj", &[3 * HIDDEN, HIDDEN])?,
            coefficients: take("conv.conv", &[HIDDEN, 7, 1])?,
            output: take("conv.out_proj", &[HIDDEN, HIDDEN])?,
        }
    } else {
        let kv_width = KV_HEADS * HEAD_DIM;
        Operator::Attention {
            q: take("self_attn.q_proj", &[QUERY_HEADS * HEAD_DIM, HIDDEN])?,
            k: take("self_attn.k_proj", &[kv_width, HIDDEN])?,
            v: take("self_attn.v_proj", &[kv_width, HIDDEN])?,
            gate: take("self_attn.gate_proj", &[QUERY_HEADS, HIDDEN])?,
            q_norm: take("self_attn.q_norm", &[HEAD_DIM])?,
            k_norm: take("self_attn.k_norm", &[HEAD_DIM])?,
            output: take("self_attn.o_proj", &[HIDDEN, QUERY_HEADS * HEAD_DIM])?,
        }
    };
    Ok(Layer {
        operator,
        operator_norm: take("operator_norm", &[HIDDEN])?,
        post_operator_norm: take("post_operator_norm", &[HIDDEN])?,
        ffn_norm: take("ffn_norm", &[HIDDEN])?,
        post_ffn_norm: take("post_ffn_norm", &[HIDDEN])?,
        gate: take("feed_forward.w1", &[INTERMEDIATE, HIDDEN])?,
        up: take("feed_forward.w3", &[INTERMEDIATE, HIDDEN])?,
        down: take("feed_forward.w2", &[HIDDEN, INTERMEDIATE])?,
    })
}

pub struct LoadedLfm3 {
    pub model: Model,
    pub report: Lfm3LoadReport,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lfm3LoadReport {
    pub source_sha256: [u8; 32],
    pub source_bytes: u64,
    pub temporary_allowance_bytes: usize,
    pub authenticated_ns: u128,
    pub layout_ns: u128,
    pub delivery_ns: u128,
    pub rotary_ns: u128,
}

/// Load one immutable BF16 bank from the authenticated single-file candidate.
/// Admit the requested model and exclusive session before reading payload.
pub fn load_lfm3(
    directory: &Path,
    capacity: usize,
    chunk: usize,
    memory_limit_bytes: usize,
) -> Result<LoadedLfm3, Error> {
    let memory = Model::plan(capacity, chunk)?;
    let limit = memory_limit_bytes.min(INITIAL_MEMORY_LIMIT_BYTES);
    let required = memory
        .total_bytes
        .checked_add(LOAD_ALLOWANCE)
        .ok_or_else(|| invalid("LFM3 working-set size overflow"))?;
    if required > limit {
        return Err(Error::Budget { required, limit });
    }
    let files = [ExpectedFile {
        name: "model.safetensors",
        bytes: BYTES,
        sha256: SHA256,
    }];
    let manifest = Manifest::new(
        "mchen04/LFM3-1.7B-Heretic-Abliterated-MLX-bf16",
        "88636d45d403e7c8ab6a40e85c9d0fa90cc5a253",
        &files,
    )?;
    let started = Instant::now();
    let mut audit = FileAudit::open(
        directory,
        &manifest,
        0,
        &[ORIGIN as u64, BYTES],
        LOAD_ALLOWANCE,
    )?;
    let authenticated_ns = started.elapsed().as_nanos();
    let started = Instant::now();
    let mut header = reserve(ORIGIN)?;
    audit.visit(0, |bytes| {
        header.extend_from_slice(bytes);
        Ok(())
    })?;
    let length = u64::from_le_bytes(header[..8].try_into().unwrap());
    if length != (ORIGIN - 8) as u64 {
        return Err(invalid("unexpected LFM3 header size"));
    }
    let metadata = parse_safetensors_header(&header[8..], ORIGIN as u64, BYTES)?;
    if metadata.offset_keys().len() != 286 || metadata.data_len() != PARAMETERS * 2 {
        return Err(invalid("unexpected LFM3 tensor inventory"));
    }
    let embedding = tensor(&metadata, "model.embed_tokens.weight", &[VOCAB, HIDDEN])?;
    let final_norm = tensor(&metadata, "model.embedding_norm.weight", &[HIDDEN])?;
    let layers = (0..LAYERS)
        .map(|i| parse_layer(&metadata, i))
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| invalid("LFM3 layer count"))?;
    let layout_ns = started.elapsed().as_nanos();
    let started = Instant::now();
    let mut data = reserve(PARAMETERS)?;
    audit.visit(1, |bytes| {
        if !bytes.len().is_multiple_of(2) {
            return Err(invalid("unaligned LFM3 BF16 delivery"));
        }
        for pair in bytes.chunks_exact(2) {
            let value = bf16::from_le_bytes([pair[0], pair[1]]);
            if !value.is_finite() {
                return Err(invalid("nonfinite LFM3 checkpoint word"));
            }
            data.push(value);
        }
        Ok(())
    })?;
    let delivery = audit.finish()?;
    if delivery.file_sha256() != SHA256 || data.len() != PARAMETERS {
        return Err(invalid("incomplete LFM3 weight delivery"));
    }
    let delivery_ns = started.elapsed().as_nanos();
    let started = Instant::now();
    let model = Model::from_verified(
        Weights {
            data,
            embedding,
            final_norm,
            layers,
        },
        capacity,
        chunk,
        memory,
    )?;
    let rotary_ns = started.elapsed().as_nanos();
    Ok(LoadedLfm3 {
        model,
        report: Lfm3LoadReport {
            source_sha256: delivery.file_sha256(),
            source_bytes: delivery.bytes(),
            temporary_allowance_bytes: LOAD_ALLOWANCE,
            authenticated_ns,
            layout_ns,
            delivery_ns,
            rotary_ns,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_rejects_before_source_open() {
        let memory = Model::plan(2048, 16).unwrap();
        let required = memory.total_bytes + LOAD_ALLOWANCE;
        let path = Path::new("");
        assert!(matches!(
            load_lfm3(path, 2048, 16, required - 1),
            Err(Error::Budget { required: r, limit }) if r == required && limit == required - 1
        ));
        assert!(matches!(
            load_lfm3(path, 262_144, 262_144, usize::MAX),
            Err(Error::Budget {
                limit: INITIAL_MEMORY_LIMIT_BYTES,
                ..
            })
        ));
    }
}
