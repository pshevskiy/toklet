//! Shared cold config/index/container delivery for engine-owned dense profiles.
//! FileAudit and complete physical-range verification are unchanged.
use super::{Manifest, QwenProfile, llama3_manifest, mistral_manifest};
use crate::weights::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DenseProfile {
    Qwen(QwenProfile),
    Mistral,
    Llama3,
}
impl DenseProfile {
    pub(crate) fn manifest(self) -> Result<Manifest<'static>, Error> {
        match self {
            Self::Qwen(profile) => profile.manifest(),
            Self::Mistral => mistral_manifest(),
            Self::Llama3 => llama3_manifest(),
        }
    }
}

use super::{FileAudit, FileDelivery};
use crate::weights::{
    Index, METADATA_ALLOWANCE_BYTES, Metadata, Shard, Source, UniqueKeys, invalid, io,
};
use sha2::Digest;
use std::{collections::BTreeMap, fs::File, io::Read, mem::size_of, path::Path};

/// Historical consumed-source evidence for an engine-owned dense artifact.
/// The six-file digest is an EXPECTED distribution anchor. This receipt covers
/// config, index and all container bytes only: no tokenizer, special-token map,
/// successful model assembly, live model ownership or codec pairing is proved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseSourceContentIdentity {
    config: FileDelivery,
    index: FileDelivery,
    container: FileDelivery,
}
impl DenseSourceContentIdentity {
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

pub(crate) struct DenseAudit {
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
        return Err(invalid("Dense metadata exceeds bounded read"));
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
impl DenseAudit {
    pub(crate) fn config(
        profile: DenseProfile,
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
        verified_small(directory, &profile.manifest()?, 0)
    }
    pub(crate) fn open_source(
        profile: DenseProfile,
        directory: &Path,
        config: FileDelivery,
        budget: usize,
    ) -> Result<Source, Error> {
        Self::open_source_expected(directory, config, &profile.manifest()?, budget)
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
            return Err(invalid("Dense source config receipt mismatch"));
        }
        let (index_bytes, index_receipt) = verified_small(directory, manifest, 2)?;
        let index: Index = serde_json::from_slice(&index_bytes)?;
        drop(index_bytes);
        if index.weight_map.is_empty()
            || index.weight_map.values().any(|s| s != "model.safetensors")
        {
            return Err(invalid("audited dense source requires exact single shard"));
        }
        let payload = index
            .metadata
            .get("total_size")
            .and_then(serde_json::Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| invalid("audited dense index requires payload total"))?;
        let required = payload
            .checked_add(METADATA_ALLOWANCE_BYTES)
            .and_then(|n| n.checked_add(super::BUFFER_BYTES))
            .ok_or_else(|| invalid("Dense source admission overflow"))?;
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
            return Err(invalid("Dense header bound"));
        }
        let data_start = (header_count as u64)
            .checked_add(8)
            .ok_or_else(|| invalid("header overflow"))?;
        if data_start >= files[1].bytes {
            return Err(invalid("Dense header extent"));
        }
        let mut provisional_header = Vec::new();
        provisional_header
            .try_reserve_exact(header_count)
            .map_err(|_| Error::Allocation)?;
        provisional_header.resize(header_count, 0);
        io(&path, provisional.read_exact(&mut provisional_header))?;
        drop(provisional);
        if provisional_header.first() != Some(&b'{') {
            return Err(invalid("Dense header must be object"));
        }
        serde_json::from_slice::<UniqueKeys>(&provisional_header)?;
        let plan: Metadata = serde_json::from_slice(&provisional_header)?;
        let names = plan.offset_keys();
        if names.is_empty() || names.len() + 2 > super::MAX_RANGES {
            return Err(invalid("Dense physical range count exceeds policy"));
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
                return Err(invalid("Dense tensor range coverage"));
            }
            offset = info.data_offsets.1;
            ends.push(
                data_start
                    .checked_add(offset as u64)
                    .ok_or_else(|| invalid("tensor extent overflow"))?,
            );
        }
        if ends.last() != Some(&files[1].bytes) {
            return Err(invalid("Dense full container coverage"));
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
                return Err(invalid("Dense prefix changed"));
            }
            consumed += chunk.len();
            Ok(())
        })?;
        if consumed != 8 {
            return Err(invalid("Dense prefix incomplete"));
        }
        consumed = 0;
        audit.visit(1, |chunk| {
            if provisional_header.get(consumed..consumed + chunk.len()) != Some(chunk) {
                return Err(invalid("Dense header changed"));
            }
            consumed += chunk.len();
            Ok(())
        })?;
        if consumed != header_count {
            return Err(invalid("Dense header incomplete"));
        }
        let metadata: Metadata = serde_json::from_slice(&provisional_header)?;
        drop(provisional_header);
        let mut locations = BTreeMap::new();
        for name in metadata.offset_keys() {
            if index.weight_map.get(&name).map(String::as_str) != Some("model.safetensors") {
                return Err(invalid("Dense index disagrees with verified header"));
            }
            locations.insert(name, 0);
        }
        if locations.len() != index.weight_map.len() {
            return Err(invalid("Dense index references absent tensor"));
        }
        if let Some(total) = index.metadata.get("total_size")
            && total.as_u64() != Some(metadata.data_len() as u64)
        {
            return Err(invalid("Dense index payload total mismatch"));
        }
        // Keep the legacy Shard shape unchanged. This duplicate descriptor
        // refers to the same open file description; the dense visitor exclusively
        // uses the audit descriptor. No second file/model contents are retained.
        let file: File = io(&path, audit.file.try_clone())?;
        let shard = Shard {
            path,
            file,
            metadata,
            data_start,
            header_bytes: header_count,
        };
        let dense = Self {
            config,
            index: index_receipt,
            container: audit,
        };
        if dense.owned_bytes()? > METADATA_ALLOWANCE_BYTES {
            return Err(invalid("Dense audit metadata budget"));
        }
        Ok(Source {
            shards: vec![shard],
            locations,
            buffer: Vec::new(),
            conversion_seconds: 0.,
            audit: None,
            dense_audit: Some(dense),
        })
    }
    pub(crate) fn owned_bytes(&self) -> Result<usize, Error> {
        self.container
            .owned_bytes()?
            .checked_add(size_of::<Self>() - size_of::<FileAudit>())
            .ok_or_else(|| invalid("Dense audit memory overflow"))
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
    pub(crate) fn identity_for(&self, profile: DenseProfile) -> Option<DenseSourceContentIdentity> {
        let identity = self.identity()?;
        (identity.expected_manifest_sha256() == profile.manifest().ok()?.identity())
            .then_some(identity)
    }
    pub(crate) fn identity(&self) -> Option<DenseSourceContentIdentity> {
        Some(DenseSourceContentIdentity {
            config: self.config,
            index: self.index,
            container: self.container.delivery()?,
        })
    }
}

#[cfg(test)]
#[path = "qwen3/tests.rs"]
pub(crate) mod tests;

#[cfg(test)]
#[path = "dense/mistral_tests.rs"]
mod mistral_tests;
