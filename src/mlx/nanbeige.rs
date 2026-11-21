//! Pinned Nanbeige4.2-3B affine Q4 artifact and shared-layer execution plan.
//!
//! Packed groups contain 64 codes in eight little-endian U32 words, low
//! nibble first. Each group has a BF16 scale and additive BF16 bias. An
//! independently aligned resident bank is required for each source tensor.

use super::affine::{Layout, MetadataDtype};
use crate::{
    capacity::{KvBytes, KvOwner, KvStorage, Retention},
    weights::{self, Error},
};
use safetensors::Dtype;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

pub const PROFILE: &str = "nanbeige4.2-3b-mlx-q4";
pub const REVISION: &str = "c21f850c4096b67a1b53a2c8eb4f8c8652f89375";
pub const HIDDEN: usize = 3072;
pub const INTERMEDIATE: usize = 10752;
pub const HEAD_DIM: usize = 128;
pub const QUERY_HEADS: usize = 48;
pub const KV_HEADS: usize = 8;
pub const QUERY_WIDTH: usize = QUERY_HEADS * HEAD_DIM;
pub const KV_WIDTH: usize = KV_HEADS * HEAD_DIM;
pub const PHYSICAL_LAYERS: usize = 22;
pub const PASSES: usize = 2;
pub const VISITS: usize = PHYSICAL_LAYERS * PASSES;
pub const LOGIT_ROWS: usize = 166144;
pub const TOKENIZER_VOCAB: u32 = 166107;
pub const EOS: u32 = 166101;
pub const ROPE_THETA: f32 = 70_000_000.;
pub const EPSILON: f32 = 1e-5;
pub const MAX_CHUNK: usize = 128;
pub const MAX_CAPACITY: usize = weights::INITIAL_CONTEXT_LIMIT;

const FILE_NAME: &str = "model.safetensors";
const FILE_BYTES: u64 = 2345769894;
const FILE_DIGEST: &str = "916df8e6495e9f9e896bf74a3515a8c1829d7b8dd4b275820f2315a74f5a49bf";
const HEADER_BYTES: usize = 58278; // includes eight-byte little-endian length
const HEADER_DIGEST: &str = "8bed74a15f7cb08b6b5a5d283461c40cd7bf26282a6e30cfc9a727deaadf989e";
pub const WEIGHT_BYTES: usize = FILE_BYTES as usize - HEADER_BYTES;
const CONFIG: Asset = (
    "config.json",
    1305,
    "d0eab082cf933bd2900d837963b5e6d7d1d71423960ae7b48cd239ec18e88272",
);
const INDEX: Asset = (
    "model.safetensors.index.json",
    36121,
    "24fa6e7c1e3aa25573276c9735d093af096f71cccbaa036ddfe28c45f741e7e1",
);
pub(crate) type Asset = (&'static str, usize, &'static str);

/// Prefill dispatch follows chunk shape. One-row chunks, including any prompt
/// remainder of length one, use grouped vector arithmetic. For a multirow
/// chunk the retained last-row head uses the same prefill arithmetic even
/// though generation does not retain earlier logits. This experimental
/// schedule uses the tiled kernel for all multirow chunks. MLX can select
/// vector kernels for some small multirow shapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Projection {
    GroupedVector,
    TiledBf16,
}

impl Projection {
    pub fn for_chunk(tokens: usize) -> Self {
        if tokens == 1 {
            Self::GroupedVector
        } else {
            Self::TiledBf16
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Visit {
    pub physical_layer: usize,
    pub kv_slot: usize,
    pub normalize_after: bool,
}

pub fn visits() -> [Visit; VISITS] {
    std::array::from_fn(|i| Visit {
        physical_layer: i % PHYSICAL_LAYERS,
        kv_slot: i,
        normalize_after: (i + 1).is_multiple_of(PHYSICAL_LAYERS),
    })
}

pub fn kv_bytes(capacity: usize) -> Result<KvBytes, crate::cpu::Error> {
    KvOwner {
        heads: KV_HEADS,
        head_dim: HEAD_DIM,
        storage: KvStorage::Bf16,
        retention: Retention::Full,
        guard_elements: 0,
        copies: VISITS,
    }
    .at_capacity(capacity)
}

#[derive(Clone, Copy)]
pub struct Matrix {
    pub layout: Layout,
    pub weights: usize,
    pub scales: usize,
    pub biases: usize,
}

pub struct Block {
    pub input_norm: usize,
    pub query: Matrix,
    pub key: Matrix,
    pub value: Matrix,
    pub attention_out: Matrix,
    pub post_attention_norm: usize,
    pub gate: Matrix,
    pub up: Matrix,
    pub down: Matrix,
}

#[derive(Clone, Debug)]
struct Bank {
    name: String,
    bytes: usize,
    start: usize,
}

pub struct Plan {
    banks: Vec<Bank>,
    pub embedding: Matrix,
    pub head: Matrix,
    pub final_norm: usize,
    pub blocks: [Block; PHYSICAL_LAYERS],
}

#[derive(Deserialize)]
struct Index {
    metadata: IndexMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct IndexMetadata {
    total_size: usize,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

fn io_error(path: &Path, source: std::io::Error) -> Error {
    Error::Io {
        path: path.into(),
        source,
    }
}

fn matches_pin(bytes: &[u8], expected_bytes: usize, digest: &str) -> bool {
    bytes.len() == expected_bytes && format!("{:x}", Sha256::digest(bytes)) == digest
}

pub(crate) fn pinned_asset(
    directory: &Path,
    (name, size, digest): Asset,
) -> Result<Vec<u8>, Error> {
    let path = directory.join(name);
    let file = weights::audit::open_regular(&path)?;
    if file.metadata().map_err(|e| io_error(&path, e))?.len() != size as u64 {
        return Err(invalid(format!("Nanbeige {name} asset length differs")));
    }
    let mut raw = Vec::new();
    raw.try_reserve_exact(size).map_err(|_| Error::Allocation)?;
    file.take(size as u64 + 1)
        .read_to_end(&mut raw)
        .map_err(|e| io_error(&path, e))?;
    if !matches_pin(&raw, size, digest) {
        return Err(invalid(format!("Nanbeige {name} asset identity differs")));
    }
    Ok(raw)
}

pub fn inspect(config: &[u8], index: &[u8], header_with_prefix: &[u8]) -> Result<Plan, Error> {
    if !matches_pin(config, CONFIG.1, CONFIG.2)
        || !matches_pin(index, INDEX.1, INDEX.2)
        || !matches_pin(header_with_prefix, HEADER_BYTES, HEADER_DIGEST)
    {
        return Err(invalid(
            "Nanbeige pinned config, index or shard header differs",
        ));
    }
    let prefix = u64::from_le_bytes(header_with_prefix[..8].try_into().unwrap());
    if prefix != (HEADER_BYTES - 8) as u64 {
        return Err(invalid("Nanbeige safetensors header length differs"));
    }
    let index: Index = serde_json::from_slice(index)?;
    let offsets = weights::parse_safetensors_header(
        &header_with_prefix[8..],
        HEADER_BYTES as u64,
        FILE_BYTES,
    )?;
    const TENSORS: usize = 513;
    let offset_names = offsets.offset_keys();
    if index.metadata.total_size != WEIGHT_BYTES
        || offset_names.len() != TENSORS
        || index.weight_map.len() != TENSORS
    {
        return Err(invalid("Nanbeige tensor inventory differs"));
    }

    let mut banks = Vec::with_capacity(TENSORS);
    for name in offset_names {
        let info = offsets
            .info(&name)
            .ok_or_else(|| invalid("Nanbeige missing tensor descriptor"))?;
        let (start, end) = info.data_offsets;
        let bytes = end - start;
        if index.weight_map.get(&name).map(String::as_str) != Some(FILE_NAME)
            || bytes == 0
            || !bytes.is_multiple_of(4)
        {
            return Err(invalid(format!(
                "Nanbeige tensor owner or extent differs for {name}"
            )));
        }
        banks.push(Bank { name, bytes, start });
    }
    banks.sort_unstable_by_key(|b| b.start);
    let mut names: BTreeMap<_, _> = banks
        .iter()
        .enumerate()
        .map(|(id, b)| (b.name.as_str(), id))
        .collect();

    let mut tensor = |name: &str, dtype: Dtype, shape: &[usize]| {
        let id = names
            .remove(name)
            .ok_or_else(|| invalid(format!("Nanbeige missing tensor {name}")))?;
        let info = offsets.info(name).unwrap();
        if info.dtype != dtype || info.shape != shape {
            return Err(invalid(format!(
                "Nanbeige dtype or shape differs for {name}"
            )));
        }
        if dtype == Dtype::U32 && (HEADER_BYTES + banks[id].start) % 4 != 2 {
            return Err(invalid("Nanbeige packed source alignment differs"));
        }
        Ok(id)
    };
    fn matrix(
        tensor: &mut impl FnMut(&str, Dtype, &[usize]) -> Result<usize, Error>,
        name: &str,
        rows: usize,
        columns: usize,
    ) -> Result<Matrix, Error> {
        let layout =
            Layout::new(rows, columns, 4, 64, MetadataDtype::BF16).map_err(|e| invalid(e.0))?;
        Ok(Matrix {
            layout,
            weights: tensor(&format!("{name}.weight"), Dtype::U32, &[rows, columns / 8])?,
            scales: tensor(
                &format!("{name}.scales"),
                Dtype::BF16,
                &[rows, columns / 64],
            )?,
            biases: tensor(
                &format!("{name}.biases"),
                Dtype::BF16,
                &[rows, columns / 64],
            )?,
        })
    }
    let embedding = matrix(&mut tensor, "model.embed_tokens", LOGIT_ROWS, HIDDEN)?;
    let head = matrix(&mut tensor, "lm_head", LOGIT_ROWS, HIDDEN)?;
    let final_norm = tensor("model.norm.weight", Dtype::BF16, &[HIDDEN])?;
    let mut blocks = Vec::with_capacity(PHYSICAL_LAYERS);
    for i in 0..PHYSICAL_LAYERS {
        let prefix = format!("model.layers.{i}");
        blocks.push(Block {
            input_norm: tensor(
                &format!("{prefix}.input_layernorm.weight"),
                Dtype::BF16,
                &[HIDDEN],
            )?,
            query: matrix(
                &mut tensor,
                &format!("{prefix}.self_attn.q_proj"),
                QUERY_WIDTH,
                HIDDEN,
            )?,
            key: matrix(
                &mut tensor,
                &format!("{prefix}.self_attn.k_proj"),
                KV_WIDTH,
                HIDDEN,
            )?,
            value: matrix(
                &mut tensor,
                &format!("{prefix}.self_attn.v_proj"),
                KV_WIDTH,
                HIDDEN,
            )?,
            attention_out: matrix(
                &mut tensor,
                &format!("{prefix}.self_attn.o_proj"),
                HIDDEN,
                QUERY_WIDTH,
            )?,
            post_attention_norm: tensor(
                &format!("{prefix}.post_attention_layernorm.weight"),
                Dtype::BF16,
                &[HIDDEN],
            )?,
            gate: matrix(
                &mut tensor,
                &format!("{prefix}.mlp.gate_proj"),
                INTERMEDIATE,
                HIDDEN,
            )?,
            up: matrix(
                &mut tensor,
                &format!("{prefix}.mlp.up_proj"),
                INTERMEDIATE,
                HIDDEN,
            )?,
            down: matrix(
                &mut tensor,
                &format!("{prefix}.mlp.down_proj"),
                HIDDEN,
                INTERMEDIATE,
            )?,
        });
    }
    if !names.is_empty() {
        return Err(invalid("Nanbeige unexpected tensor"));
    }
    Ok(Plan {
        banks,
        embedding,
        head,
        final_norm,
        blocks: blocks
            .try_into()
            .map_err(|_| invalid("Nanbeige layer count differs"))?,
    })
}

/// Sequential authenticated loading directly into final owners. Each bank is
/// read once into its separately allocated destination. The caller must hold
/// all completed destinations private until finish authenticates this stream.
pub struct Source {
    path: PathBuf,
    file: File,
    hasher: Sha256,
    next: usize,
    plan: Plan,
}

impl Source {
    pub fn open(directory: &Path) -> Result<Self, Error> {
        let config = pinned_asset(directory, CONFIG)?;
        let index = pinned_asset(directory, INDEX)?;
        let path = directory.join(FILE_NAME);
        let mut file = weights::audit::open_regular(&path)?;
        if file.metadata().map_err(|e| io_error(&path, e))?.len() != FILE_BYTES {
            return Err(invalid("Nanbeige pinned shard length differs"));
        }
        let mut header = vec![0; HEADER_BYTES];
        file.read_exact(&mut header)
            .map_err(|e| io_error(&path, e))?;
        let plan = inspect(&config, &index, &header)?;
        let mut hasher = Sha256::new();
        hasher.update(&header);
        Ok(Self {
            path,
            file,
            hasher,
            next: 0,
            plan,
        })
    }

    pub fn next_bank_bytes(&self) -> Option<usize> {
        self.plan.banks.get(self.next).map(|bank| bank.bytes)
    }

    pub fn largest_bank_bytes(&self) -> usize {
        self.plan.banks.iter().map(|bank| bank.bytes).max().unwrap()
    }

    pub fn read_bank(&mut self, destination: &mut [u8]) -> Result<(), Error> {
        if self.next_bank_bytes() != Some(destination.len()) {
            return Err(invalid("Nanbeige resident tensor extent differs"));
        }
        self.file
            .read_exact(destination)
            .map_err(|e| io_error(&self.path, e))?;
        self.hasher.update(destination);
        self.next += 1;
        Ok(())
    }

    pub fn finish(mut self) -> Result<Plan, Error> {
        let mut tail = [0];
        if self.next_bank_bytes().is_some()
            || self
                .file
                .read(&mut tail)
                .map_err(|e| io_error(&self.path, e))?
                != 0
            || format!("{:x}", self.hasher.finalize()) != FILE_DIGEST
        {
            return Err(invalid("Nanbeige pinned shard payload identity differs"));
        }
        Ok(self.plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_blocks_keep_every_kv_slot_and_both_final_norms() {
        let visits = visits();
        for (i, visit) in visits.into_iter().enumerate() {
            assert_eq!(visit.physical_layer, i % 22);
            assert_eq!(visit.kv_slot, i);
            assert_eq!(visit.normalize_after, i == 21 || i == 43);
        }
        assert_eq!(kv_bytes(2048).unwrap().total_bytes, 352 * 1024 * 1024);
        assert_eq!(Projection::for_chunk(1), Projection::GroupedVector);
        assert_eq!(Projection::for_chunk(128), Projection::TiledBf16);
    }
}
