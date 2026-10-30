//! Concrete consumed-asset evidence for the pinned Qwen3 source and codec.
//! A standalone codec receipt is historical; native pairing separately retains
//! the exact assembled model Rc. No template execution or app grant is added.
use super::{Codec, Profile, invalid};
use crate::{
    mlx::qwen::QwenSourceContentIdentity,
    weights::{
        audit::open_regular,
        content::{ExpectedFile, qwen3_manifest},
    },
};
use sha2::{Digest, Sha256};
use std::{
    io::{self, Read},
    path::Path,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QwenCodecContentIdentity {
    source: QwenSourceContentIdentity,
    tokenizer: [u8; 32],
    policy: [u8; 32],
    ancillary: [u8; 32],
}
impl QwenCodecContentIdentity {
    pub const fn source(&self) -> &QwenSourceContentIdentity {
        &self.source
    }
    /// All six expected assets have historical read evidence with distinct
    /// roles; this is not a simultaneous snapshot or current disk identity.
    pub const fn complete_manifest_sha256(&self) -> [u8; 32] {
        self.source.expected_manifest_sha256()
    }
    /// Exact bytes passed to Tokenizer::from_bytes.
    pub const fn tokenizer_sha256(&self) -> [u8; 32] {
        self.tokenizer
    }
    /// Verified against the fixed Rust policy; arbitrary Jinja is not parsed.
    pub const fn tokenizer_config_policy_sha256(&self) -> [u8; 32] {
        self.policy
    }
    /// Verified distribution asset, not inference input or parsed template.
    pub const fn ancillary_special_map_sha256(&self) -> [u8; 32] {
        self.ancillary
    }
    pub const fn text_policy(&self) -> &'static str {
        "qwen3-plain-nonthinking-v1"
    }
}

fn admit(source: &QwenSourceContentIdentity, allowance: usize) -> io::Result<()> {
    if allowance < Profile::Qwen3.allowance()
        || allowance > crate::weights::INITIAL_MEMORY_LIMIT_BYTES
    {
        return Err(invalid(
            "Audited Qwen3 codec requires a separate 256 MiB..=6 GiB allowance",
        ));
    }
    let manifest = qwen3_manifest().map_err(invalid)?;
    let files = manifest.files();
    if source.expected_manifest_sha256() != manifest.identity()
        || source.config_sha256() != files[0].sha256
        || source.index_sha256() != files[2].sha256
        || source.container_sha256() != files[1].sha256
        || source.container_bytes() != files[1].bytes
        || source.container_physical_ranges() < 3
        || source.container_physical_ranges() > crate::weights::content::MAX_RANGES
        || source.verified_container_reads() < source.container_physical_ranges()
    {
        return Err(invalid("Unsupported authenticated source for Qwen3 codec"));
    }
    Ok(())
}

fn verified(
    directory: &Path,
    expected: ExpectedFile<'_>,
    limit: usize,
) -> io::Result<(Vec<u8>, [u8; 32])> {
    let count = usize::try_from(expected.bytes).map_err(invalid)?;
    if count == 0 || count > limit {
        return Err(invalid("Audited codec asset size policy"));
    }
    let count_plus_one = count
        .checked_add(1)
        .ok_or_else(|| invalid("Audited codec size overflow"))?;
    let path = directory.join(expected.name);
    let file = open_regular(&path).map_err(invalid)?;
    if file.metadata()?.len() != expected.bytes {
        return Err(invalid("Audited codec asset extent differs"));
    }
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(count_plus_one).map_err(invalid)?;
    file.take(count_plus_one as u64).read_to_end(&mut bytes)?;
    let digest = Sha256::digest(&bytes).into();
    if bytes.len() != count || digest != expected.sha256 {
        return Err(invalid("Audited codec consumed bytes differ"));
    }
    Ok((bytes, digest))
}

struct Assets {
    tokenizer: Vec<u8>,
    tokenizer_sha: [u8; 32],
    policy_sha: [u8; 32],
    ancillary_sha: [u8; 32],
}
fn assets(
    directory: &Path,
    config_sha: [u8; 32],
    files: &[ExpectedFile<'_>],
) -> io::Result<Assets> {
    let (config, actual) = verified(directory, files[0], 16 * 1024)?;
    if actual != config_sha {
        return Err(invalid("Codec config differs from loaded source"));
    }
    drop(config);
    let (policy, policy_sha) = verified(directory, files[4], 16 * 1024)?;
    drop(policy);
    let (ancillary, ancillary_sha) = verified(directory, files[5], 16 * 1024)?;
    drop(ancillary);
    let (tokenizer, tokenizer_sha) = verified(directory, files[3], 16 * 1024 * 1024)?;
    Ok(Assets {
        tokenizer,
        tokenizer_sha,
        policy_sha,
        ancillary_sha,
    })
}
fn build(assets: Assets, source: &QwenSourceContentIdentity) -> io::Result<Codec> {
    let mut codec = Codec::from_tokenizer_bytes(&assets.tokenizer, Profile::Qwen3)?;
    codec.qwen3_content_identity = Some(QwenCodecContentIdentity {
        source: *source,
        tokenizer: assets.tokenizer_sha,
        policy: assets.policy_sha,
        ancillary: assets.ancillary_sha,
    });
    Ok(codec)
}
impl Codec {
    /// Check source evidence before files, then hash and parse the same tokenizer
    /// buffer. Public source-only evidence does not bind a live model allocation;
    /// native audited chat creation separately requires the assembled owner.
    pub fn load_qwen3_audited(
        directory: &Path,
        source: &QwenSourceContentIdentity,
        allowance: usize,
    ) -> io::Result<Self> {
        admit(source, allowance)?;
        let manifest = qwen3_manifest().map_err(invalid)?;
        let assets = assets(directory, source.config_sha256(), manifest.files())?;
        build(assets, source)
    }
    pub const fn qwen3_content_identity(&self) -> Option<&QwenCodecContentIdentity> {
        self.qwen3_content_identity.as_ref()
    }
}
#[cfg(test)]
mod tests;
