//! One pinned catalog expectation and its opt-in consumed-source reader.
use super::{ExpectedFile, Manifest};
use crate::weights::Error;

const FILES: [ExpectedFile<'static>; 6] = [
    ExpectedFile {
        name: "config.json",
        bytes: 937,
        sha256: [
            0x9f, 0x38, 0x7d, 0xcb, 0x1b, 0xf0, 0x45, 0xca, 0xdb, 0x34, 0xa3, 0x6e, 0x67, 0x56,
            0x9a, 0xbd, 0x73, 0xc0, 0x0e, 0x53, 0x22, 0x60, 0x3d, 0xf2, 0xb1, 0x9d, 0x0c, 0x0f,
            0x8c, 0xc9, 0xd2, 0x40,
        ],
    },
    ExpectedFile {
        name: "model.safetensors",
        bytes: 633442994,
        sha256: [
            0x3a, 0xd5, 0xc9, 0x6e, 0x0a, 0x47, 0x6d, 0x4e, 0xee, 0x48, 0xe9, 0xf5, 0x25, 0xd0,
            0xee, 0x0a, 0x3f, 0x18, 0x30, 0xb3, 0xe5, 0xb8, 0xe1, 0x7b, 0x44, 0x1d, 0xe0, 0x66,
            0x1e, 0x92, 0x13, 0x97,
        ],
    },
    ExpectedFile {
        name: "model.safetensors.index.json",
        bytes: 49731,
        sha256: [
            0x45, 0xad, 0xdf, 0x77, 0x79, 0xcd, 0xbd, 0x5a, 0xf1, 0x8d, 0x92, 0xf9, 0xf7, 0xb7,
            0x9c, 0x18, 0x8f, 0x6e, 0xbb, 0xf3, 0x2c, 0xff, 0x7b, 0x4a, 0xe7, 0xdb, 0xe6, 0x8a,
            0x91, 0x34, 0x0e, 0xbc,
        ],
    },
    ExpectedFile {
        name: "tokenizer.json",
        bytes: 11422654,
        sha256: [
            0xae, 0xb1, 0x33, 0x07, 0xa7, 0x1a, 0xcd, 0x8f, 0xe8, 0x18, 0x61, 0xd9, 0x4a, 0xd5,
            0x4a, 0xb6, 0x89, 0xdf, 0x77, 0x33, 0x18, 0x80, 0x9e, 0xed, 0x3c, 0xbe, 0x79, 0x4b,
            0x44, 0x92, 0xda, 0xe4,
        ],
    },
    ExpectedFile {
        name: "tokenizer_config.json",
        bytes: 10219,
        sha256: [
            0x8e, 0x7d, 0xd4, 0x25, 0x0d, 0xb5, 0x8b, 0x2a, 0x29, 0x79, 0x25, 0x62, 0xd7, 0x27,
            0xb2, 0x71, 0x5d, 0xc0, 0xfe, 0x79, 0xa9, 0x4b, 0x44, 0x6c, 0xaa, 0x4c, 0x07, 0xdb,
            0x39, 0xec, 0xeb, 0xa6,
        ],
    },
    ExpectedFile {
        name: "special_tokens_map.json",
        bytes: 613,
        sha256: [
            0x76, 0x86, 0x2e, 0x76, 0x52, 0x66, 0xb8, 0x5a, 0xa9, 0x45, 0x97, 0x67, 0xe3, 0x3c,
            0xba, 0xf1, 0x39, 0x70, 0xf3, 0x27, 0xa0, 0xe8, 0x8d, 0x1c, 0x65, 0x84, 0x6c, 0x2d,
            0xdd, 0x3a, 0x1e, 0xcd,
        ],
    },
];

/// Canonical expectation for the existing managed Qwen3 profile. This is not a
/// disk verification, supported audited loader, model or codec receipt. The
/// special-token map is a distribution asset; no inference consumption is implied.
pub(crate) fn qwen3_manifest() -> Result<Manifest<'static>, Error> {
    Manifest::new(
        "mlx-community/Qwen3-0.6B-8bit",
        "11de96878523501bcaa86104e3c186de07ff9068",
        &FILES,
    )
}

use super::{FileAudit, FileDelivery};
use crate::weights::{
    Index, METADATA_ALLOWANCE_BYTES, Metadata, Shard, Source, UniqueKeys, invalid, io,
};
use sha2::Digest;
use std::{collections::BTreeMap, fs::File, io::Read, mem::size_of, path::Path};

/// Historical consumed-source evidence for the one pinned Qwen3 artifact.
/// The six-file digest is an EXPECTED distribution anchor. This receipt covers
/// config, index and all container bytes only: no tokenizer, special-token map,
/// successful model assembly, live model ownership or codec pairing is proved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QwenSourceContentIdentity {
    config: FileDelivery,
    index: FileDelivery,
    container: FileDelivery,
}
impl QwenSourceContentIdentity {
    pub const fn expected_manifest_sha256(&self) -> [u8; 32] {
        self.config.manifest_identity()
    }
    pub const fn config_sha256(&self) -> [u8; 32] {
        self.config.file_sha256()
    }
    pub const fn index_sha256(&self) -> [u8; 32] {
        self.index.file_sha256()
    }
    pub const fn container_sha256(&self) -> [u8; 32] {
        self.container.file_sha256()
    }
    pub const fn container_bytes(&self) -> u64 {
        self.container.bytes()
    }
    /// Includes separately delivered length prefix and header, then all tensors.
    pub const fn container_physical_ranges(&self) -> usize {
        self.container.physical_ranges()
    }
    pub const fn verified_container_reads(&self) -> usize {
        self.container.verified_reads()
    }
}

pub(crate) struct QwenAudit {
    config: FileDelivery,
    index: FileDelivery,
    container: FileAudit,
}
fn verified_small(
    directory: &Path,
    manifest: &Manifest<'_>,
    file_index: usize,
) -> Result<(Vec<u8>, FileDelivery), Error> {
    let bytes = manifest.files()[file_index].bytes;
    if bytes > 1024 * 1024 {
        return Err(invalid("Qwen metadata exceeds bounded read"));
    }
    let mut audit = FileAudit::open(
        directory,
        manifest,
        file_index,
        &[bytes],
        METADATA_ALLOWANCE_BYTES,
    )?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(bytes as usize)
        .map_err(|_| Error::Allocation)?;
    audit.visit(0, |chunk| {
        output.extend_from_slice(chunk);
        Ok(())
    })?;
    Ok((output, audit.finish()?))
}
impl QwenAudit {
    pub(crate) fn config(
        directory: &Path,
        budget: usize,
    ) -> Result<(Vec<u8>, FileDelivery), Error> {
        // Same existing source metadata+stream admission before any file open.
        let required = METADATA_ALLOWANCE_BYTES + super::BUFFER_BYTES;
        if budget < required {
            return Err(Error::Budget {
                required,
                limit: budget,
            });
        }
        verified_small(directory, &qwen3_manifest()?, 0)
    }
    pub(crate) fn open_source(
        directory: &Path,
        config: FileDelivery,
        budget: usize,
    ) -> Result<Source, Error> {
        Self::open_source_expected(directory, config, &qwen3_manifest()?, budget)
    }
    fn open_source_expected(
        directory: &Path,
        config: FileDelivery,
        manifest: &Manifest<'_>,
        budget: usize,
    ) -> Result<Source, Error> {
        let files = manifest.files();
        if config.manifest_identity() != manifest.identity()
            || config.file_sha256() != files[0].sha256
            || config.file_name_sha256() != <[u8; 32]>::from(sha2::Sha256::digest(b"config.json"))
        {
            return Err(invalid("Qwen source config receipt mismatch"));
        }
        let (index_bytes, index_receipt) = verified_small(directory, manifest, 2)?;
        let index: Index = serde_json::from_slice(&index_bytes)?;
        drop(index_bytes);
        if index.weight_map.is_empty()
            || index.weight_map.values().any(|s| s != "model.safetensors")
        {
            return Err(invalid("audited Qwen source requires exact single shard"));
        }
        let payload = index
            .metadata
            .get("total_size")
            .and_then(serde_json::Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| invalid("audited Qwen index requires payload total"))?;
        let required = payload
            .checked_add(METADATA_ALLOWANCE_BYTES)
            .and_then(|n| n.checked_add(super::BUFFER_BYTES))
            .ok_or_else(|| invalid("Qwen source admission overflow"))?;
        if required > budget {
            return Err(Error::Budget {
                required,
                limit: budget,
            });
        }
        let path = directory.join("model.safetensors");
        // Bounded provisional header plans ranges only. None of these bytes are
        // parsed by the final loader until re-delivered and verified below.
        let mut provisional = crate::weights::audit::open_regular(&path)?;
        let mut length = [0; 8];
        io(&path, provisional.read_exact(&mut length))?;
        let header_count =
            usize::try_from(u64::from_le_bytes(length)).map_err(|_| invalid("header overflow"))?;
        if header_count == 0 || header_count > 128 * 1024 {
            return Err(invalid("Qwen header bound"));
        }
        let data_start = (header_count as u64)
            .checked_add(8)
            .ok_or_else(|| invalid("header overflow"))?;
        if data_start >= files[1].bytes {
            return Err(invalid("Qwen header extent"));
        }
        let mut provisional_header = Vec::new();
        provisional_header
            .try_reserve_exact(header_count)
            .map_err(|_| Error::Allocation)?;
        provisional_header.resize(header_count, 0);
        io(&path, provisional.read_exact(&mut provisional_header))?;
        drop(provisional);
        if provisional_header.first() != Some(&b'{') {
            return Err(invalid("Qwen header must be object"));
        }
        serde_json::from_slice::<UniqueKeys>(&provisional_header)?;
        let plan: Metadata = serde_json::from_slice(&provisional_header)?;
        let names = plan.offset_keys();
        if names.is_empty() || names.len() + 2 > super::MAX_RANGES {
            return Err(invalid("Qwen physical range count exceeds policy"));
        }
        let mut ends = Vec::new();
        ends.try_reserve_exact(names.len() + 2)
            .map_err(|_| Error::Allocation)?;
        ends.extend([8, data_start]);
        let mut offset = 0;
        for name in &names {
            let info = plan
                .info(name)
                .ok_or_else(|| invalid("missing planned tensor"))?;
            if info.data_offsets.0 != offset || info.data_offsets.1 <= offset {
                return Err(invalid("Qwen tensor range coverage"));
            }
            offset = info.data_offsets.1;
            ends.push(
                data_start
                    .checked_add(offset as u64)
                    .ok_or_else(|| invalid("tensor extent overflow"))?,
            );
        }
        if ends.last() != Some(&files[1].bytes) {
            return Err(invalid("Qwen full container coverage"));
        }
        drop(names);
        drop(plan);
        let mut audit = FileAudit::open(directory, manifest, 1, &ends, METADATA_ALLOWANCE_BYTES)?;
        drop(ends);
        // Avoid retaining another header. Compare verified delivery to the
        // provisional header, then parse that identical, now-verified buffer.
        let mut consumed = 0;
        audit.visit(0, |chunk| {
            if length.get(consumed..consumed + chunk.len()) != Some(chunk) {
                return Err(invalid("Qwen prefix changed"));
            }
            consumed += chunk.len();
            Ok(())
        })?;
        if consumed != 8 {
            return Err(invalid("Qwen prefix incomplete"));
        }
        consumed = 0;
        audit.visit(1, |chunk| {
            if provisional_header.get(consumed..consumed + chunk.len()) != Some(chunk) {
                return Err(invalid("Qwen header changed"));
            }
            consumed += chunk.len();
            Ok(())
        })?;
        if consumed != header_count {
            return Err(invalid("Qwen header incomplete"));
        }
        let metadata: Metadata = serde_json::from_slice(&provisional_header)?;
        drop(provisional_header);
        let mut locations = BTreeMap::new();
        for name in metadata.offset_keys() {
            if index.weight_map.get(&name).map(String::as_str) != Some("model.safetensors") {
                return Err(invalid("Qwen index disagrees with verified header"));
            }
            locations.insert(name, 0);
        }
        if locations.len() != index.weight_map.len() {
            return Err(invalid("Qwen index references absent tensor"));
        }
        if let Some(total) = index.metadata.get("total_size")
            && total.as_u64() != Some(metadata.data_len() as u64)
        {
            return Err(invalid("Qwen index payload total mismatch"));
        }
        // Keep the legacy Shard shape unchanged. This duplicate descriptor
        // refers to the same open file description; the Qwen visitor exclusively
        // uses the audit descriptor. No second file/model contents are retained.
        let file: File = io(&path, audit.file.try_clone())?;
        let shard = Shard {
            path,
            file,
            metadata,
            data_start,
            header_bytes: header_count,
        };
        let qwen = Self {
            config,
            index: index_receipt,
            container: audit,
        };
        if qwen.owned_bytes()? > METADATA_ALLOWANCE_BYTES {
            return Err(invalid("Qwen audit metadata budget"));
        }
        Ok(Source {
            shards: vec![shard],
            locations,
            buffer: Vec::new(),
            conversion_seconds: 0.,
            audit: None,
            qwen_audit: Some(qwen),
        })
    }
    pub(crate) fn owned_bytes(&self) -> Result<usize, Error> {
        self.container
            .owned_bytes()?
            .checked_add(size_of::<Self>() - size_of::<FileAudit>())
            .ok_or_else(|| invalid("Qwen audit memory overflow"))
    }
    pub(crate) fn poison(&mut self) {
        self.container.poison();
    }
    pub(crate) fn visit(
        &mut self,
        begin: u64,
        end: u64,
        consume: impl FnMut(&[u8]) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.container.visit_extent(begin, end, consume)
    }
    pub(crate) fn identity(&self) -> Option<QwenSourceContentIdentity> {
        Some(QwenSourceContentIdentity {
            config: self.config,
            index: self.index,
            container: self.container.delivery()?,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests;
