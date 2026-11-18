//! Bounded local loading for the pinned text payload.

use super::{Plan, SHARDS, ShardPin, TEXT_BYTES, inspect, invalid};
use crate::mlx::payload::{buffer, read_exact, validate_q4_bf16};
use crate::weights::Error;
use half::bf16;
use safetensors::Dtype;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{ErrorKind, Read},
    path::Path,
};

const TRANSFER_BYTES: usize = 64 << 10;
const METADATA_ALLOWANCE_BYTES: usize = 64 << 20;

/// Admission estimate for the model load only. Session and runtime admission
/// must apply their own limits before using the retained text payload.
pub const ADMITTED_LOAD_BYTES: usize =
    TEXT_BYTES as usize + METADATA_ALLOWANCE_BYTES + TRANSFER_BYTES;

pub struct TextPayload {
    plan: Plan,
    tensors: BTreeMap<String, Vec<u8>>,
}

impl TextPayload {
    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// Immutable original little-endian bytes. Each physical tensor has one owner.
    pub fn tensor(&self, name: &str) -> Option<&[u8]> {
        self.tensors.get(name).map(Vec::as_slice)
    }
}

fn io<T>(path: &Path, result: std::io::Result<T>) -> Result<T, Error> {
    result.map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })
}

fn read_metadata(path: &Path, limit: usize) -> Result<Vec<u8>, Error> {
    let mut file = io(path, File::open(path))?;
    let meta = io(path, file.metadata())?;
    if !meta.is_file() || meta.len() > limit as u64 {
        return Err(invalid("SenseNova metadata file extent differs"));
    }
    let mut data = buffer(meta.len() as usize)?;
    io(path, file.read_exact(&mut data))?;
    require_eof(&mut file)?;
    Ok(data)
}

fn require_eof(reader: &mut impl Read) -> Result<(), Error> {
    match reader.read_exact(&mut [0]) {
        Err(error) if error.kind() == ErrorKind::UnexpectedEof => Ok(()),
        _ => Err(invalid("SenseNova shard or metadata end differs")),
    }
}

/// Validate original metadata and authenticate both complete shards before
/// publishing immutable text tensors. Skipped vision bytes pass through the
/// same fixed-size hashing buffer as other gaps in the text inventory.
pub fn load(directory: &Path, memory_limit_bytes: usize) -> Result<TextPayload, Error> {
    if ADMITTED_LOAD_BYTES > memory_limit_bytes {
        return Err(Error::Budget {
            required: ADMITTED_LOAD_BYTES,
            limit: memory_limit_bytes,
        });
    }
    let config = read_metadata(&directory.join("config.json"), 8192)?;
    let index = read_metadata(&directory.join("model.safetensors.index.json"), 256 << 10)?;
    let mut sources = Vec::with_capacity(SHARDS.len());
    let mut headers = [Vec::new(), Vec::new()];
    let mut lengths = [0; 2];
    for (slot, pin) in SHARDS.iter().enumerate() {
        let path = directory.join(pin.name);
        let mut file = io(&path, File::open(&path))?;
        let meta = io(&path, file.metadata())?;
        if !meta.is_file() || meta.len() != pin.file_bytes {
            return Err(invalid("SenseNova shard file extent differs"));
        }
        lengths[slot] = meta.len();
        let (header, sha) = read_shard_header(&mut file, pin)?;
        headers[slot] = header;
        sources.push((file, sha));
    }
    let plan = inspect(&config, &index, [&headers[0], &headers[1]], lengths)?;
    let mut transfer = buffer(TRANSFER_BYTES)?;
    let mut tensors = BTreeMap::new();
    for (slot, (mut file, sha)) in sources.into_iter().enumerate() {
        consume_shard(&mut file, slot, &plan, sha, &mut transfer, &mut tensors)?;
    }
    finish(plan, tensors)
}

fn read_shard_header(reader: &mut impl Read, pin: &ShardPin) -> Result<(Vec<u8>, Sha256), Error> {
    let mut sha = Sha256::new();
    let mut prefix = [0; 8];
    read_exact(reader, &mut prefix, &mut sha)?;
    if u64::from_le_bytes(prefix) != pin.header_bytes as u64 {
        return Err(invalid("SenseNova shard header length differs"));
    }
    let mut header = buffer(pin.header_bytes)?;
    read_exact(reader, &mut header, &mut sha)?;
    Ok((header, sha))
}

fn consume_shard(
    reader: &mut impl Read,
    slot: usize,
    plan: &Plan,
    mut sha: Sha256,
    transfer: &mut [u8],
    tensors: &mut BTreeMap<String, Vec<u8>>,
) -> Result<(), Error> {
    let pin = &SHARDS[slot];
    let mut text: Vec<_> = plan
        .text_tensors()
        .iter()
        .filter(|(_, tensor)| tensor.shard == slot)
        .collect();
    text.sort_unstable_by_key(|(_, tensor)| tensor.file_range.start);
    let mut offset = 8 + pin.header_bytes as u64;
    for (name, tensor) in text {
        let range = &tensor.file_range;
        let skip = range
            .start
            .checked_sub(offset)
            .ok_or_else(|| invalid("SenseNova text range overlaps another range"))?;
        hash_skip(reader, skip, transfer, &mut sha)?;
        let size = usize::try_from(range.end - range.start)
            .map_err(|_| invalid("SenseNova tensor size exceeds address space"))?;
        let mut bytes = buffer(size)?;
        read_exact(reader, &mut bytes, &mut sha)?;
        tensors.insert(name.clone(), bytes);
        offset = range.end;
    }
    hash_skip(reader, pin.file_bytes - offset, transfer, &mut sha)?;
    require_eof(reader)?;
    if format!("{:x}", sha.finalize()) != pin.file_sha256 {
        return Err(invalid("SenseNova full shard SHA-256 differs"));
    }
    Ok(())
}

fn finish(plan: Plan, tensors: BTreeMap<String, Vec<u8>>) -> Result<TextPayload, Error> {
    for (name, tensor) in plan.text_tensors() {
        let data = &tensors[name];
        let finite = match tensor.dtype {
            Dtype::U32 => true,
            Dtype::BF16 => data
                .chunks_exact(2)
                .all(|by| bf16::from_le_bytes([by[0], by[1]]).is_finite()),
            Dtype::F32 => data
                .chunks_exact(4)
                .all(|by| f32::from_le_bytes([by[0], by[1], by[2], by[3]]).is_finite()),
            _ => false,
        };
        if !finite {
            return Err(invalid(
                "SenseNova floating-point parameter is nonfinite or unsupported",
            ));
        }
    }
    for (name, layout) in plan.affine_layouts() {
        validate_q4_bf16(
            *layout,
            &tensors[&format!("{name}.weight")],
            &tensors[&format!("{name}.scales")],
            &tensors[&format!("{name}.biases")],
        )?;
    }
    Ok(TextPayload { plan, tensors })
}

fn hash_skip(
    reader: &mut impl Read,
    mut remaining: u64,
    transfer: &mut [u8],
    sha: &mut Sha256,
) -> Result<(), Error> {
    while remaining != 0 {
        let take = remaining.min(transfer.len() as u64) as usize;
        read_exact(reader, &mut transfer[..take], sha)?;
        remaining -= take as u64;
    }
    Ok(())
}

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
mod streamed_smoke;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skipped_bytes_remain_in_the_shard_digest() {
        let mut transfer = [0; 2];
        let mut sha = Sha256::new();
        let mut data = &b"skipped text skipped"[..];
        hash_skip(&mut data, 8, &mut transfer, &mut sha).unwrap();
        let mut retained = [0; 4];
        read_exact(&mut data, &mut retained, &mut sha).unwrap();
        hash_skip(&mut data, 8, &mut transfer, &mut sha).unwrap();
        require_eof(&mut data).unwrap();
        assert_eq!(&retained, b"text");
        assert_eq!(
            sha.finalize()[..],
            Sha256::digest(b"skipped text skipped")[..]
        );
    }

    #[test]
    fn short_budget_precedes_any_file_open() {
        assert!(matches!(
            load(
                Path::new("missing-sensenova-artifact"),
                ADMITTED_LOAD_BYTES - 1
            ),
            Err(Error::Budget {
                required: ADMITTED_LOAD_BYTES,
                ..
            })
        ));
    }
}
