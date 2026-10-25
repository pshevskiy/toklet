//! Opt-in historical raw-consumption evidence for one frozen Smol checkpoint.
//! No model construction, tokenizer pairing or application capability follows.
use super::{CONVERSION_BYTES, Error, METADATA_ALLOWANCE_BYTES, invalid, io};
use safetensors::Dtype;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, mem::size_of, path::Path};

const WITNESS: &[u8] = include_bytes!("../../tests/fixtures/smol-load-identity-v1/witness.json");
const WITNESS_SHA: &str = "cf226bc5a0317b1156964c5e2454dd9242aa01b2aaf03e6949da60e058ce6680";
const CONTAINER_BYTES: u64 = 143030401;
const PAYLOAD_BYTES: usize = 142955136;
const DATA_START: usize = 75265;

/// Historical evidence that the admitted metadata and all required raw ranges
/// matched one authenticated reference. This is not a current disk snapshot or
/// proof of successful model assembly. A copied value survives later source
/// changes/failures. Obtain it at the final source-consumption boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceContentIdentity {
    config: [u8; 32],
    index: [u8; 32],
    header: [u8; 32],
    reference_container: [u8; 32],
    verified_reads: usize,
    physical_ranges: usize,
    payload_bytes: usize,
    container_bytes: u64,
}
impl SourceContentIdentity {
    pub const fn repository(&self) -> &'static str {
        "mlx-community/SmolLM2-135M-Instruct-8bit"
    }
    pub const fn revision(&self) -> &'static str {
        "0f0d9b8218915bc34d401e1a340b8c049d300d5e"
    }
    pub const fn config_sha256(&self) -> [u8; 32] {
        self.config
    }
    pub const fn index_sha256(&self) -> [u8; 32] {
        self.index
    }
    pub const fn header_sha256(&self) -> [u8; 32] {
        self.header
    }
    /// Authenticated reference anchor whose complete prefix/header/payload
    /// ranges matched. Per-range hashes do not recompute whole-file SHA-256.
    pub const fn reference_container_sha256(&self) -> [u8; 32] {
        self.reference_container
    }
    pub const fn verified_reads(&self) -> usize {
        self.verified_reads
    }
    pub const fn physical_ranges(&self) -> usize {
        self.physical_ranges
    }
    pub const fn payload_bytes(&self) -> usize {
        self.payload_bytes
    }
    pub const fn container_bytes(&self) -> u64 {
        self.container_bytes
    }
}
#[derive(Clone, Copy)]
pub(crate) enum MetadataKind {
    Config,
    Index,
    Prefix,
    Header,
}
impl MetadataKind {
    fn expected(self) -> (usize, &'static str, u8) {
        match self {
            Self::Config => (
                1097,
                "5da91905efdfde905f566d76d4031e6ca7c4cb2a01dc4e3812a70b855188fc73",
                1,
            ),
            Self::Index => (
                48979,
                "d016e7514e8bfed162eb9d0716c73244c1b73792db3143e92e54138c1772db44",
                2,
            ),
            Self::Prefix => (
                8,
                "db8cfabc0a617c1e247673bfaa67cf5320598e244ff9b6538bb01f1c23dc5625",
                4,
            ),
            Self::Header => (
                75257,
                "357edb41f2242613ddddcbe671be7c60923abf4850c77105c63a870df4ccb1f8",
                8,
            ),
        }
    }
}
#[derive(Deserialize)]
struct Witness {
    repository: String,
    revision: String,
    shard: WitnessShard,
}
#[derive(Deserialize)]
struct WitnessShard {
    bytes: u64,
    data_start: usize,
    tensors: Vec<WitnessRange>,
}
#[derive(Deserialize)]
struct WitnessRange {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    bytes: usize,
    payload_offsets: [usize; 2],
    sha256: String,
}
struct Range {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    begin: usize,
    end: usize,
    digest: [u8; 32],
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Clean,
    Reading { index: usize, ordinal: usize },
    Failed,
}
pub(crate) struct Audit {
    ranges: Vec<Range>,
    seen: Vec<bool>,
    unique: usize,
    reads: usize,
    state: State,
    metadata: u8,
    identity: SourceContentIdentity,
}
pub(crate) struct Ticket {
    index: usize,
    ordinal: usize,
    expected: usize,
    count: usize,
    hash: Sha256,
}
fn digest_hex(text: &str) -> Result<[u8; 32], Error> {
    if text.len() != 64
        || !text
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(invalid("invalid trusted digest"));
    }
    let mut result = [0; 32];
    for (out, pair) in result.iter_mut().zip(text.as_bytes().chunks_exact(2)) {
        let digit = |b: u8| if b <= b'9' { b - b'0' } else { b - b'a' + 10 };
        *out = 16 * digit(pair[0]) + digit(pair[1]);
    }
    Ok(result)
}
impl Audit {
    pub(crate) fn smol(budget: usize) -> Result<Self, Error> {
        let required = METADATA_ALLOWANCE_BYTES + CONVERSION_BYTES;
        if budget < required {
            return Err(Error::Budget {
                required,
                limit: budget,
            });
        }
        if <[u8; 32]>::from(Sha256::digest(WITNESS)) != digest_hex(WITNESS_SHA)? {
            return Err(invalid("trusted Smol witness changed"));
        }
        let witness: Witness = serde_json::from_slice(WITNESS)?;
        if witness.repository != "mlx-community/SmolLM2-135M-Instruct-8bit"
            || witness.revision != "0f0d9b8218915bc34d401e1a340b8c049d300d5e"
            || witness.shard.bytes != CONTAINER_BYTES
            || witness.shard.data_start != DATA_START
            || witness.shard.tensors.len() != 694
        {
            return Err(invalid("trusted Smol inventory changed"));
        }
        let mut ranges = Vec::new();
        ranges
            .try_reserve_exact(694)
            .map_err(|_| Error::Allocation)?;
        let mut end = 0;
        for r in witness.shard.tensors {
            let [begin, next] = r.payload_offsets;
            if begin != end
                || next <= begin
                || next - begin != r.bytes
                || r.name.is_empty()
                || r.name.len() > 256
                || r.shape.is_empty()
                || r.shape.len() > 2
            {
                return Err(invalid("trusted range geometry changed"));
            }
            let dtype = match r.dtype.as_str() {
                "F16" => Dtype::F16,
                "U32" => Dtype::U32,
                _ => return Err(invalid("trusted dtype changed")),
            };
            let bytes = r
                .shape
                .iter()
                .try_fold(dtype.bitsize() / 8, |n, &d| n.checked_mul(d))
                .ok_or_else(|| invalid("trusted shape overflow"))?;
            if bytes != r.bytes {
                return Err(invalid("trusted shape size changed"));
            }
            ranges.push(Range {
                name: r.name,
                dtype,
                shape: r.shape,
                begin,
                end: next,
                digest: digest_hex(&r.sha256)?,
            });
            end = next;
        }
        if end != PAYLOAD_BYTES || DATA_START.checked_add(end) != Some(CONTAINER_BYTES as usize) {
            return Err(invalid("trusted coverage changed"));
        }
        ranges.sort_unstable_by(|a, b| a.name.cmp(&b.name));
        if ranges.windows(2).any(|r| r[0].name == r[1].name) {
            return Err(invalid("duplicate trusted range"));
        }
        let mut seen = Vec::new();
        seen.try_reserve_exact(ranges.len())
            .map_err(|_| Error::Allocation)?;
        seen.resize(ranges.len(), false);
        let result = Self {
            ranges,
            seen,
            unique: 0,
            reads: 0,
            state: State::Clean,
            metadata: 0,
            identity: SourceContentIdentity {
                config: digest_hex(MetadataKind::Config.expected().1)?,
                index: digest_hex(MetadataKind::Index.expected().1)?,
                header: digest_hex(MetadataKind::Header.expected().1)?,
                reference_container: digest_hex(
                    "ec569c771e56424636be99625aba9031abc525a68a5f4ab16d678eca96fef12c",
                )?,
                verified_reads: 0,
                physical_ranges: 694,
                payload_bytes: PAYLOAD_BYTES,
                container_bytes: CONTAINER_BYTES,
            },
        };
        if result.owned_bytes() > METADATA_ALLOWANCE_BYTES {
            return Err(invalid("audit exceeds metadata policy"));
        }
        Ok(result)
    }
    pub(crate) fn owned_bytes(&self) -> usize {
        size_of::<Self>()
            + self.ranges.capacity() * size_of::<Range>()
            + self.seen.capacity()
            + self
                .ranges
                .iter()
                .map(|r| r.name.capacity() + r.shape.capacity() * size_of::<usize>())
                .sum::<usize>()
    }
    pub(crate) fn verify_metadata(&mut self, kind: MetadataKind, data: &[u8]) -> Result<(), Error> {
        let (bytes, expected, bit) = kind.expected();
        if data.len() != bytes || <[u8; 32]>::from(Sha256::digest(data)) != digest_hex(expected)? {
            self.poison();
            return Err(invalid("Smol metadata differs from trusted consumed bytes"));
        }
        self.metadata |= bit;
        Ok(())
    }
    pub(crate) fn read_metadata(
        &mut self,
        path: &Path,
        kind: MetadataKind,
    ) -> Result<Vec<u8>, Error> {
        let file = open_regular(path)?;
        let count = kind.expected().0;
        let mut data = Vec::new();
        data.try_reserve_exact(count + 1)
            .map_err(|_| Error::Allocation)?;
        io(path, file.take((count + 1) as u64).read_to_end(&mut data))?;
        self.verify_metadata(kind, &data)?;
        Ok(data)
    }
    pub(crate) fn verify_extent(&mut self, bytes: u64) -> Result<(), Error> {
        if bytes != self.identity.container_bytes {
            self.poison();
            return Err(invalid("audited shard extent changed"));
        }
        Ok(())
    }
    pub(crate) fn begin(&mut self, name: &str) -> Result<Ticket, Error> {
        if self.state != State::Clean || self.metadata != 15 {
            self.poison();
            return Err(invalid("source audit poisoned or incomplete"));
        }
        self.state = State::Failed;
        let index = self
            .ranges
            .binary_search_by(|r| r.name.as_str().cmp(name))
            .map_err(|_| invalid("untrusted tensor read"))?;
        let ordinal = self
            .reads
            .checked_add(1)
            .ok_or_else(|| invalid("audit read counter overflow"))?;
        self.state = State::Reading { index, ordinal };
        Ok(Ticket {
            index,
            ordinal,
            expected: self.ranges[index].end - self.ranges[index].begin,
            count: 0,
            hash: Sha256::new(),
        })
    }
    pub(crate) fn validate_range(
        &mut self,
        ticket: &Ticket,
        info: &safetensors::tensor::TensorInfo,
    ) -> Result<(), Error> {
        let r = &self.ranges[ticket.index];
        if info.data_offsets != (r.begin, r.end) || info.dtype != r.dtype || info.shape != r.shape {
            self.poison();
            return Err(invalid("audited tensor metadata changed"));
        }
        Ok(())
    }
    pub(crate) fn finish(&mut self, ticket: Ticket) -> Result<(), Error> {
        let expected = State::Reading {
            index: ticket.index,
            ordinal: ticket.ordinal,
        };
        let current = self.state;
        self.state = State::Failed;
        if current != expected
            || ticket.count != ticket.expected
            || <[u8; 32]>::from(ticket.hash.finalize()) != self.ranges[ticket.index].digest
        {
            return Err(invalid("consumed tensor bytes differ from trusted range"));
        }
        let unique = if self.seen[ticket.index] {
            self.unique
        } else {
            self.unique
                .checked_add(1)
                .ok_or_else(|| invalid("audit range counter overflow"))?
        };
        self.reads = ticket.ordinal;
        self.unique = unique;
        self.seen[ticket.index] = true;
        self.state = State::Clean;
        Ok(())
    }
    pub(crate) fn poison(&mut self) {
        self.state = State::Failed;
    }
    pub(crate) fn identity(&self) -> Option<SourceContentIdentity> {
        if self.state != State::Clean || self.metadata != 15 || self.unique != self.ranges.len() {
            return None;
        }
        Some(SourceContentIdentity {
            verified_reads: self.reads,
            ..self.identity
        })
    }
}
impl Ticket {
    pub(crate) fn update(&mut self, bytes: &[u8]) -> Result<(), Error> {
        self.count = self
            .count
            .checked_add(bytes.len())
            .filter(|&n| n <= self.expected)
            .ok_or_else(|| invalid("audited read size overflow"))?;
        self.hash.update(bytes);
        Ok(())
    }
}
/// No-follow applies to the final component. Parent directories may be linked;
/// authority always comes from the consumed bytes and the retained handle.
#[cfg(unix)]
pub(crate) fn open_regular(path: &Path) -> Result<File, Error> {
    use std::os::unix::fs::OpenOptionsExt;
    let file = io(
        path,
        File::options()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(path),
    )?;
    if !io(path, file.metadata())?.is_file() {
        return Err(invalid("audited input must be a regular file"));
    }
    Ok(file)
}
#[cfg(not(unix))]
pub(crate) fn open_regular(_: &Path) -> Result<File, Error> {
    Err(invalid(
        "audited opening requires supported Unix file flags",
    ))
}
#[cfg(test)]
pub(crate) mod tests;
