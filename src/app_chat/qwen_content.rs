//! Shared concrete consumed-asset evidence for pinned Qwen2/Qwen3 codecs.
//! A standalone codec receipt is historical; native pairing separately retains
//! the exact assembled model Rc. No template execution or app grant is added.
use super::{Codec, Profile, content::verified, invalid};
use crate::{
    mlx::qwen::QwenSourceContentIdentity,
    weights::content::{ExpectedFile, QwenProfile},
};
use std::{io, path::Path};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QwenCodecContentIdentity {
    profile: QwenProfile,
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
        match self.profile {
            QwenProfile::Qwen2 => "qwen-chatml-v1",
            QwenProfile::Qwen3 => "qwen3-plain-nonthinking-v1",
        }
    }
}
fn text_profile(profile: QwenProfile) -> Profile {
    match profile {
        QwenProfile::Qwen2 => Profile::Qwen,
        QwenProfile::Qwen3 => Profile::Qwen3,
    }
}
fn tokenizer_cap(profile: QwenProfile) -> usize {
    match profile {
        QwenProfile::Qwen2 => 8 * 1024 * 1024,
        QwenProfile::Qwen3 => 16 * 1024 * 1024,
    }
}
fn admit(
    profile: QwenProfile,
    source: &QwenSourceContentIdentity,
    allowance: usize,
) -> io::Result<()> {
    if allowance < text_profile(profile).allowance()
        || allowance > crate::weights::INITIAL_MEMORY_LIMIT_BYTES
    {
        return Err(invalid(match profile {
            QwenProfile::Qwen2 => {
                "Audited Qwen2 codec requires a separate 128 MiB..=6 GiB allowance"
            }
            QwenProfile::Qwen3 => {
                "Audited Qwen3 codec requires a separate 256 MiB..=6 GiB allowance"
            }
        }));
    }
    let manifest = profile.manifest().map_err(invalid)?;
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
        return Err(invalid(match profile {
            QwenProfile::Qwen2 => "Unsupported authenticated source for Qwen2 codec",
            QwenProfile::Qwen3 => "Unsupported authenticated source for Qwen3 codec",
        }));
    }
    Ok(())
}

struct Assets {
    tokenizer: Vec<u8>,
    tokenizer_sha: [u8; 32],
    policy_sha: [u8; 32],
    ancillary_sha: [u8; 32],
}
fn assets(
    profile: QwenProfile,
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
    let (tokenizer, tokenizer_sha) = verified(directory, files[3], tokenizer_cap(profile))?;
    Ok(Assets {
        tokenizer,
        tokenizer_sha,
        policy_sha,
        ancillary_sha,
    })
}
fn build(
    profile: QwenProfile,
    assets: Assets,
    source: &QwenSourceContentIdentity,
) -> io::Result<Codec> {
    let mut codec = Codec::from_tokenizer_bytes(&assets.tokenizer, text_profile(profile))?;
    codec.qwen_content_identity = Some(QwenCodecContentIdentity {
        profile,
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
        Self::load_qwen_audited(QwenProfile::Qwen3, directory, source, allowance)
    }
    /// Complete historical Qwen2 codec evidence using the original fixed ChatML
    /// formatter and a separate 128 MiB text allowance. No live owner is implied.
    pub fn load_qwen2_audited(
        directory: &Path,
        source: &QwenSourceContentIdentity,
        allowance: usize,
    ) -> io::Result<Self> {
        Self::load_qwen_audited(QwenProfile::Qwen2, directory, source, allowance)
    }
    fn load_qwen_audited(
        profile: QwenProfile,
        directory: &Path,
        source: &QwenSourceContentIdentity,
        allowance: usize,
    ) -> io::Result<Self> {
        admit(profile, source, allowance)?;
        let manifest = profile.manifest().map_err(invalid)?;
        let assets = assets(profile, directory, source.config_sha256(), manifest.files())?;
        build(profile, assets, source)
    }
    pub const fn qwen3_content_identity(&self) -> Option<&QwenCodecContentIdentity> {
        match self.qwen_content_identity.as_ref() {
            Some(value) if matches!(value.profile, QwenProfile::Qwen3) => Some(value),
            _ => None,
        }
    }
    pub const fn qwen2_content_identity(&self) -> Option<&QwenCodecContentIdentity> {
        match self.qwen_content_identity.as_ref() {
            Some(value) if matches!(value.profile, QwenProfile::Qwen2) => Some(value),
            _ => None,
        }
    }
}
#[cfg(test)]
#[path = "qwen_content/qwen2_tests.rs"]
mod qwen2_tests;
#[cfg(test)]
#[path = "qwen3_content/tests.rs"]
mod tests;
