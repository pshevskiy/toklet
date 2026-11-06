//! Bounded cold file delivery against an expected complete artifact manifest.
//!
//! This module does not load a model, pair a codec, authenticate an arbitrary
//! caller's expectations, or authorize persisted settings. Receipts describe
//! historical bytes delivered successfully, not current disk or resident state.
use super::{Error, audit::open_regular, invalid};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    mem::size_of,
    path::Path,
};

pub(crate) const MAX_FILES: usize = 16;
pub(crate) const MAX_RANGES: usize = 4096;
pub(crate) const BUFFER_BYTES: usize = 64 * 1024;
pub(crate) const MAX_FILE_BYTES: u64 = 4 * 1024 * 1024 * 1024;
pub(crate) const MAX_TOTAL_BYTES: u64 = 6 * 1024 * 1024 * 1024;

/// Caller-owned expectations. Names are plain ASCII basenames, not paths.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ExpectedFile<'a> {
    pub name: &'a str,
    pub bytes: u64,
    pub sha256: [u8; 32],
}

/// A validated borrowed inventory. Validation is not publisher authentication.
/// The borrowed strings/table remain immutable and are accounted by the caller.
pub(crate) struct Manifest<'a> {
    files: &'a [ExpectedFile<'a>],
    identity: [u8; 32],
}
impl<'a> Manifest<'a> {
    /// Canonical encoding: domain `toklet-full-manifest-v1\0`, then repository
    /// and revision as u64-LE byte length + ASCII bytes, u64-LE file count, then
    /// entries sorted by raw basename bytes: length+name, u64-LE size, 32 digest
    /// bytes. Directory, input ordering, backend and runtime fit are excluded.
    pub(crate) fn new(
        repository: &str,
        revision: &str,
        files: &'a [ExpectedFile<'a>],
    ) -> Result<Self, Error> {
        let component = |text: &str| {
            !text.is_empty()
                && text.len() <= 128
                && text
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
                && text != "."
                && text != ".."
        };
        let mut parts = repository.split('/');
        if !parts.next().is_some_and(component)
            || !parts.next().is_some_and(component)
            || parts.next().is_some()
            || revision.len() != 40
            || !revision
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            || files.is_empty()
            || files.len() > MAX_FILES
        {
            return Err(invalid("invalid content manifest identity or count"));
        }
        let mut order = [0; MAX_FILES];
        let mut total = 0_u64;
        for (i, entry) in files.iter().enumerate() {
            if !component(entry.name) || entry.bytes == 0 || entry.bytes > MAX_FILE_BYTES {
                return Err(invalid("invalid content manifest file"));
            }
            total = total
                .checked_add(entry.bytes)
                .ok_or_else(|| invalid("manifest size overflow"))?;
            order[i] = i;
        }
        if total > MAX_TOTAL_BYTES {
            return Err(invalid("content manifest exceeds total byte policy"));
        }
        let order = &mut order[..files.len()];
        order.sort_unstable_by_key(|&i| files[i].name);
        if order
            .windows(2)
            .any(|p| files[p[0]].name == files[p[1]].name)
        {
            return Err(invalid("duplicate content manifest file"));
        }
        let identity =
            canonical_manifest_identity(repository, revision, order.iter().map(|&i| files[i]));
        Ok(Self { files, identity })
    }
    pub(crate) const fn identity(&self) -> [u8; 32] {
        self.identity
    }
    pub(crate) const fn files(&self) -> &[ExpectedFile<'a>] {
        self.files
    }
}

/// Hash an already validated inventory in ascending basename order.
/// Also used for the statically pinned official inventory, whose shard exceeds
/// this module's general bounded file-delivery policy.
pub(crate) fn canonical_manifest_identity<'a>(
    repository: &str,
    revision: &str,
    entries: impl ExactSizeIterator<Item = ExpectedFile<'a>>,
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"toklet-full-manifest-v1\0");
    let mut text = |value: &str| {
        hash.update((value.len() as u64).to_le_bytes());
        hash.update(value.as_bytes());
    };
    text(repository);
    text(revision);
    hash.update((entries.len() as u64).to_le_bytes());
    for entry in entries {
        hash.update((entry.name.len() as u64).to_le_bytes());
        hash.update(entry.name.as_bytes());
        hash.update(entry.bytes.to_le_bytes());
        hash.update(entry.sha256);
    }
    hash.finalize().into()
}

#[derive(Clone, Copy)]
struct Range {
    end: u64,
    digest: [u8; 32],
    seen: bool,
}

/// Historical successful delivery of every byte of one expected file. This
/// copy is not proof that a model/codec was built or that a path still matches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FileDelivery {
    manifest: [u8; 32],
    file: [u8; 32],
    name: [u8; 32],
    bytes: u64,
    ranges: usize,
    reads: usize,
}
impl FileDelivery {
    pub(crate) const fn manifest_identity(&self) -> [u8; 32] {
        self.manifest
    }
    pub(crate) const fn file_name_sha256(&self) -> [u8; 32] {
        self.name
    }
    pub(crate) const fn file_sha256(&self) -> [u8; 32] {
        self.file
    }
    pub(crate) const fn bytes(&self) -> u64 {
        self.bytes
    }
    pub(crate) const fn physical_ranges(&self) -> usize {
        self.ranges
    }
    pub(crate) const fn verified_reads(&self) -> usize {
        self.reads
    }
}

/// One retained descriptor, bounded range hashes and one reusable 64 KiB buffer.
/// Callbacks run only in this cold reader, never in a numerical hot loop.
pub(crate) struct FileAudit {
    file: File,
    ranges: Vec<Range>,
    buffer: Vec<u8>,
    receipt: FileDelivery,
    clean: bool,
}
impl FileAudit {
    /// Minimum requested Rust owner storage. Actual Vec capacities are checked
    /// again before reading. Excludes caller-owned manifest/range ends and OS
    /// descriptor/allocator overhead; this is not a process-memory prediction.
    pub(crate) fn required_bytes(range_count: usize) -> Result<usize, Error> {
        if range_count == 0 || range_count > MAX_RANGES {
            return Err(invalid("content range count out of bounds"));
        }
        size_of::<Self>()
            .checked_add(BUFFER_BYTES)
            .and_then(|n| {
                range_count
                    .checked_mul(size_of::<Range>())
                    .and_then(|r| n.checked_add(r))
            })
            .ok_or_else(|| invalid("content audit memory overflow"))
    }
    pub(crate) fn owned_bytes(&self) -> Result<usize, Error> {
        Self::storage_bytes(self.buffer.capacity(), self.ranges.capacity())
    }
    fn storage_bytes(buffer: usize, ranges: usize) -> Result<usize, Error> {
        ranges
            .checked_mul(size_of::<Range>())
            .and_then(|n| n.checked_add(buffer))
            .and_then(|n| n.checked_add(size_of::<Self>()))
            .ok_or_else(|| invalid("content audit memory overflow"))
    }
    /// Range ends must be strictly increasing and partition the complete file,
    /// including prefix/header bytes. The full digest prepass authenticates all
    /// range digests before any caller receives bytes. No source directory
    /// snapshot is implied. Parent path components may be symlinks.
    pub(crate) fn open(
        directory: &Path,
        manifest: &Manifest<'_>,
        file_index: usize,
        range_ends: &[u64],
        budget: usize,
    ) -> Result<Self, Error> {
        let required = Self::required_bytes(range_ends.len())?;
        if required > budget {
            return Err(Error::Budget {
                required,
                limit: budget,
            });
        }
        let entry = manifest
            .files
            .get(file_index)
            .ok_or_else(|| invalid("content file index out of bounds"))?;
        let mut previous = 0;
        for &end in range_ends {
            if end <= previous || end > entry.bytes {
                return Err(invalid("content ranges do not partition file"));
            }
            previous = end;
        }
        if previous != entry.bytes {
            return Err(invalid("content range coverage incomplete"));
        }
        let mut ranges = Vec::new();
        ranges
            .try_reserve_exact(range_ends.len())
            .map_err(|_| Error::Allocation)?;
        for &end in range_ends {
            ranges.push(Range {
                end,
                digest: [0; 32],
                seen: false,
            });
        }
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(BUFFER_BYTES)
            .map_err(|_| Error::Allocation)?;
        buffer.resize(BUFFER_BYTES, 0);
        let actual = Self::storage_bytes(buffer.capacity(), ranges.capacity())?;
        if actual > budget {
            return Err(Error::Budget {
                required: actual,
                limit: budget,
            });
        }
        let path = directory.join(entry.name);
        let file = open_regular(&path)?;
        let mut audit = Self {
            file,
            ranges,
            buffer,
            receipt: FileDelivery {
                manifest: manifest.identity,
                file: entry.sha256,
                name: Sha256::digest(entry.name.as_bytes()).into(),
                bytes: entry.bytes,
                ranges: range_ends.len(),
                reads: 0,
            },
            clean: false,
        };
        let mut whole = Sha256::new();
        audit.extent()?;
        let mut begin = 0;
        for range in &mut audit.ranges {
            let mut hash = Sha256::new();
            let mut remaining = range.end - begin;
            while remaining > 0 {
                let count = remaining.min(BUFFER_BYTES as u64) as usize;
                super::io(&path, audit.file.read_exact(&mut audit.buffer[..count]))?;
                whole.update(&audit.buffer[..count]);
                hash.update(&audit.buffer[..count]);
                remaining -= count as u64;
            }
            range.digest = hash.finalize().into();
            begin = range.end;
        }
        let mut extra = [0];
        if super::io(&path, audit.file.read(&mut extra))? != 0
            || <[u8; 32]>::from(whole.finalize()) != entry.sha256
        {
            return Err(invalid("content full-file digest or extent mismatch"));
        }
        audit.extent()?;
        audit.clean = true;
        Ok(audit)
    }
    fn extent(&self) -> Result<(), Error> {
        let metadata = self
            .file
            .metadata()
            .map_err(|e| invalid(format!("content file metadata: {e}")))?;
        if !metadata.is_file() || metadata.len() != self.receipt.bytes {
            return Err(invalid("content file extent changed"));
        }
        Ok(())
    }
    /// Deliver a complete authenticated physical range. The final digest check
    /// follows delivery: discard partial destinations on ANY returned error or
    /// unwind. A callback panic leaves this audit poisoned, even if caught by
    /// its caller. Repeated visits are reverified; arbitrary subranges are not
    /// supported. No receipt escapes this operation.
    pub(crate) fn visit(
        &mut self,
        index: usize,
        mut consume: impl FnMut(&[u8]) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let was_clean = self.clean;
        self.clean = false;
        if !was_clean {
            return Err(invalid("content audit poisoned"));
        }
        let range = *self
            .ranges
            .get(index)
            .ok_or_else(|| invalid("content range index out of bounds"))?;
        let begin = if index == 0 {
            0
        } else {
            self.ranges[index - 1].end
        };
        self.extent()?;
        self.file
            .seek(SeekFrom::Start(begin))
            .map_err(|e| invalid(format!("content seek: {e}")))?;
        let mut remaining = range.end - begin;
        let mut hash = Sha256::new();
        while remaining > 0 {
            let count = remaining.min(BUFFER_BYTES as u64) as usize;
            self.file
                .read_exact(&mut self.buffer[..count])
                .map_err(|e| invalid(format!("content read: {e}")))?;
            hash.update(&self.buffer[..count]);
            consume(&self.buffer[..count])?;
            remaining -= count as u64;
        }
        self.extent()?;
        if <[u8; 32]>::from(hash.finalize()) != range.digest {
            return Err(invalid("delivered bytes differ from authenticated range"));
        }
        self.receipt.reads = self
            .receipt
            .reads
            .checked_add(1)
            .ok_or_else(|| invalid("content read count overflow"))?;
        self.ranges[index].seen = true;
        self.clean = true;
        Ok(())
    }
    /// Consumes the audit. All expected ranges must have been successfully
    /// delivered; the prepass alone cannot produce a delivery receipt.
    pub(crate) fn finish(self) -> Result<FileDelivery, Error> {
        self.delivery()
            .ok_or_else(|| invalid("content delivery poisoned or incomplete"))
    }
    fn delivery(&self) -> Option<FileDelivery> {
        (self.clean && self.ranges.iter().all(|r| r.seen)).then_some(self.receipt)
    }
    fn poison(&mut self) {
        self.clean = false;
    }
    fn visit_extent(
        &mut self,
        begin: u64,
        end: u64,
        consume: impl FnMut(&[u8]) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let index = self.ranges.binary_search_by_key(&end, |r| r.end);
        match index {
            Ok(i) if (if i == 0 { 0 } else { self.ranges[i - 1].end }) == begin => {
                self.visit(i, consume)
            }
            _ => {
                self.poison();
                Err(invalid("unverified content range"))
            }
        }
    }
}

mod smol;
pub(crate) use smol::smol_manifest;
mod qwen2;
pub(crate) use qwen2::{qwen2_manifest, qwen15_manifest};
mod qwen;
mod qwen3;
pub(crate) use qwen::QwenProfile;
pub use qwen::QwenSourceContentIdentity;
mod dense;
pub use dense::DenseSourceContentIdentity;
pub(crate) use dense::{DenseAudit, DenseProfile};
mod llama3;
pub(crate) use llama3::llama3_manifest;
mod mistral;
pub(crate) use mistral::mistral_manifest;
pub(crate) use qwen3::qwen3_manifest;
#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) use dense::tests::SourceFixture;
