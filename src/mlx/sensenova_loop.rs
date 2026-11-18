//! Metadata binding for one pinned SenseNova-Loop-9B MLX conversion.

use super::affine::{Layout, MetadataDtype};
use crate::weights::{Error, parse_safetensors_header};
use safetensors::Dtype;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, ops::Range};

pub mod payload;
pub mod text_weights;

pub const REVISION: &str = "b9a36a9750f28b7a0916dd403a87d1f79a02e1cc";
pub const TEXT_BYTES: u64 = 5_050_626_048;
pub const VISION_BYTES: u64 = 912_020_960;
pub const CONFIG_SHA256: &str = "6f3e408c5d8d0bf31ad4eac8c17e4e5f3087789056811ff74d3866169e716501";
pub const INDEX_SHA256: &str = "948806e8cb94433391c4e61df80273ab502264b460e49faffacbc8e85f011b0e";

pub struct ShardPin {
    pub name: &'static str,
    pub file_bytes: u64,
    pub file_sha256: &'static str,
    pub header_bytes: usize,
    pub header_sha256: &'static str,
}

pub const SHARDS: [ShardPin; 2] = [
    ShardPin {
        name: "model-00001-of-00002.safetensors",
        file_bytes: 5_362_366_858,
        file_sha256: "d7c7fdf188525e267611b84edf4e34be98d8f973575f2937619f6dbdb595db06",
        header_bytes: 168_866,
        header_sha256: "80439f28e27e753fe537301782e2555fc2294e7620d5f5c28d838da5a4af75ac",
    },
    ShardPin {
        name: "model-00002-of-00002.safetensors",
        file_bytes: 600_449_850,
        file_sha256: "86706e23cbdba8bd6f1d2c33890a06708786fd7e7d76410e40c7b6cc56b934de",
        header_bytes: 818,
        header_sha256: "f54ed378e5e37a3699177cd6303363c2ee47b38399db0f43e9093eff9f4204e1",
    },
];

#[derive(Debug)]
pub struct TextTensor {
    pub shard: usize,
    /// Absolute range in the original shard, including its header prefix.
    pub file_range: Range<u64>,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
}

pub struct Plan {
    text: BTreeMap<String, TextTensor>,
    affine: BTreeMap<String, Layout>,
}

impl Plan {
    pub fn text_tensors(&self) -> &BTreeMap<String, TextTensor> {
        &self.text
    }

    /// Keys are module prefixes. The physical components append .weight,
    /// .scales and .biases. Embedding and output head are separate modules.
    pub fn affine_layouts(&self) -> &BTreeMap<String, Layout> {
        &self.affine
    }
}

#[derive(Deserialize)]
struct Index {
    metadata: IndexMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct IndexMetadata {
    total_size: u64,
}

fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}

fn pinned(bytes: &[u8], limit: usize, digest: &str) -> bool {
    bytes.len() <= limit && format!("{:x}", Sha256::digest(bytes)) == digest
}

/// Inspect the exact JSON config, index and two raw Safetensors JSON headers,
/// in published shard order. The header spans exclude the eight-byte length
/// prefix. Measured shard file lengths may come from local stat or trusted
/// metadata. Generation eligibility and payload authentication are separate.
pub fn inspect(
    config: &[u8],
    index: &[u8],
    headers: [&[u8]; 2],
    file_bytes: [u64; 2],
) -> Result<Plan, Error> {
    if !pinned(config, 8192, CONFIG_SHA256) || !pinned(index, 256 << 10, INDEX_SHA256) {
        return Err(invalid("SenseNova pinned config or index differs"));
    }
    let mut index: Index = serde_json::from_slice(index)?;
    if index.metadata.total_size != TEXT_BYTES + VISION_BYTES {
        return Err(invalid("SenseNova index payload size differs"));
    }

    let mut text = BTreeMap::new();
    let mut text_bytes = 0;
    let mut vision_bytes = 0;
    for (shard, pin) in SHARDS.iter().enumerate() {
        let header = headers[shard];
        if file_bytes[shard] != pin.file_bytes
            || header.len() != pin.header_bytes
            || !pinned(header, pin.header_bytes, pin.header_sha256)
        {
            return Err(invalid("SenseNova pinned shard header differs"));
        }
        let start = 8 + header.len() as u64;
        let metadata = parse_safetensors_header(header, start, pin.file_bytes)?;
        for name in metadata.offset_keys() {
            if index.weight_map.remove(&name).as_deref() != Some(pin.name) {
                return Err(invalid("SenseNova index and header ownership differ"));
            }
            let info = metadata
                .info(&name)
                .ok_or_else(|| invalid("SenseNova missing tensor descriptor"))?;
            let (first, end) = info.data_offsets;
            let file_range = (start + first as u64)..(start + end as u64);
            let bytes = (end - first) as u64;
            if name.starts_with("language_model.") {
                text_bytes += bytes;
                text.insert(
                    name,
                    TextTensor {
                        shard,
                        file_range,
                        dtype: info.dtype,
                        shape: info.shape.clone(),
                    },
                );
            } else if name.starts_with("vision_tower.") {
                vision_bytes += bytes;
            } else {
                return Err(invalid("SenseNova unrecognized tensor owner"));
            }
        }
    }
    if !index.weight_map.is_empty()
        || text.len() != 1023
        || text_bytes != TEXT_BYTES
        || vision_bytes != VISION_BYTES
    {
        return Err(invalid("SenseNova exact text and vision inventory differs"));
    }

    let mut affine = BTreeMap::new();
    for (name, tensor) in &text {
        if tensor.dtype != Dtype::U32 {
            continue;
        }
        let prefix = name
            .strip_suffix(".weight")
            .ok_or_else(|| invalid("SenseNova packed weight name differs"))?;
        let [rows, packed_columns] = tensor.shape.as_slice() else {
            return Err(invalid("SenseNova packed weight rank differs"));
        };
        // Eight four-bit values occupy one little-endian U32.
        let columns = packed_columns
            .checked_mul(8)
            .ok_or_else(|| invalid("SenseNova packed column count overflow"))?;
        let layout = Layout::new(*rows, columns, 4, 64, MetadataDtype::BF16)
            .map_err(|error| invalid(error.0))?;
        for suffix in ["scales", "biases"] {
            let component = text
                .get(&format!("{prefix}.{suffix}"))
                .ok_or_else(|| invalid("SenseNova missing affine metadata"))?;
            if component.dtype != Dtype::BF16 || component.shape != [*rows, columns / 64] {
                return Err(invalid("SenseNova affine metadata shape differs"));
            }
        }
        affine.insert(prefix.to_owned(), layout);
    }
    let embedding = affine.get("language_model.model.embed_tokens");
    let head = affine.get("language_model.lm_head");
    if affine.len() != 250
        || [embedding, head].into_iter().any(|layout| {
            !layout.is_some_and(|layout| layout.rows() == 248_320 && layout.columns() == 4096)
        })
    {
        return Err(invalid(
            "SenseNova exact affine or vocabulary inventory differs",
        ));
    }
    for layer in 4..28 {
        for branch in ["mhc_attn", "mhc_ffn"] {
            let prefix = format!("language_model.model.layers.{layer}.{branch}");
            for (suffix, shape) in [("weight", &[8, 8192][..]), ("bias", &[8][..])] {
                let tensor = text
                    .get(&format!("{prefix}.{suffix}"))
                    .ok_or_else(|| invalid("SenseNova missing FP32 routing parameter"))?;
                if tensor.dtype != Dtype::F32 || tensor.shape != shape {
                    return Err(invalid("SenseNova FP32 routing parameter shape differs"));
                }
            }
        }
    }
    Ok(Plan { text, affine })
}

#[cfg(test)]
mod tests;
