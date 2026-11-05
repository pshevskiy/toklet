//! Shared concrete consumed-asset evidence for pinned dense codecs.
//! A standalone codec receipt is historical; native pairing separately retains
//! the exact assembled model Rc. No template execution or app grant is added.
use super::{Codec, Profile, content::verified, invalid};
use crate::{
    mlx::qwen::DenseSourceContentIdentity,
    weights::content::{DenseProfile, ExpectedFile, QwenProfile},
};
use std::{io, path::Path};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseCodecContentIdentity {
    profile: DenseProfile,
    source: DenseSourceContentIdentity,
    tokenizer: [u8; 32],
    policy: [u8; 32],
    ancillary: [u8; 32],
}
impl DenseCodecContentIdentity {
    pub const fn source(&self) -> &DenseSourceContentIdentity {
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
            DenseProfile::Qwen(QwenProfile::Qwen2 | QwenProfile::Qwen15) => "qwen-chatml-v1",
            DenseProfile::Qwen(QwenProfile::Qwen3) => "qwen3-plain-nonthinking-v1",
            DenseProfile::Mistral => "mistral-instruct-v1",
            DenseProfile::Llama3 => "llama3-plain-20240726-v1",
        }
    }
}
fn text_profile(profile: DenseProfile) -> Profile {
    match profile {
        DenseProfile::Qwen(QwenProfile::Qwen2) => Profile::Qwen,
        DenseProfile::Qwen(QwenProfile::Qwen15) => Profile::Qwen15,
        DenseProfile::Qwen(QwenProfile::Qwen3) => Profile::Qwen3,
        DenseProfile::Mistral => Profile::Mistral,
        DenseProfile::Llama3 => Profile::Llama3,
    }
}
fn tokenizer_cap(profile: DenseProfile) -> usize {
    match profile {
        DenseProfile::Qwen(QwenProfile::Qwen2 | QwenProfile::Qwen15) => 8 * 1024 * 1024,
        DenseProfile::Qwen(QwenProfile::Qwen3) => 16 * 1024 * 1024,
        DenseProfile::Mistral => 4 * 1024 * 1024,
        DenseProfile::Llama3 => 16 * 1024 * 1024,
    }
}
fn policy_cap(profile: DenseProfile) -> usize {
    match profile {
        DenseProfile::Qwen(_) => 16 * 1024,
        DenseProfile::Mistral => 256 * 1024,
        DenseProfile::Llama3 => 64 * 1024,
    }
}
fn admit(
    profile: DenseProfile,
    source: &DenseSourceContentIdentity,
    allowance: usize,
) -> io::Result<()> {
    if allowance < text_profile(profile).allowance()
        || allowance > crate::weights::INITIAL_MEMORY_LIMIT_BYTES
    {
        return Err(invalid(match profile {
            DenseProfile::Qwen(QwenProfile::Qwen2 | QwenProfile::Qwen15) => {
                "Audited Qwen2 codec requires a separate 128 MiB..=6 GiB allowance"
            }
            DenseProfile::Qwen(QwenProfile::Qwen3) => {
                "Audited Qwen3 codec requires a separate 256 MiB..=6 GiB allowance"
            }
            DenseProfile::Llama3 => {
                "Audited Llama3 codec requires a separate 256 MiB..=6 GiB allowance"
            }
            DenseProfile::Mistral => {
                "Audited Mistral codec requires a separate 64 MiB..=6 GiB allowance"
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
            DenseProfile::Qwen(QwenProfile::Qwen2 | QwenProfile::Qwen15) => {
                "Unsupported authenticated source for Qwen2 codec"
            }
            DenseProfile::Qwen(QwenProfile::Qwen3) => {
                "Unsupported authenticated source for Qwen3 codec"
            }
            DenseProfile::Mistral => "Unsupported authenticated source for Mistral codec",
            DenseProfile::Llama3 => "Unsupported authenticated source for Llama3 codec",
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
    profile: DenseProfile,
    directory: &Path,
    config_sha: [u8; 32],
    files: &[ExpectedFile<'_>],
) -> io::Result<Assets> {
    let (config, actual) = verified(directory, files[0], 16 * 1024)?;
    if actual != config_sha {
        return Err(invalid("Codec config differs from loaded source"));
    }
    drop(config);
    let (policy, policy_sha) = verified(directory, files[4], policy_cap(profile))?;
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
    profile: DenseProfile,
    assets: Assets,
    source: &DenseSourceContentIdentity,
) -> io::Result<Codec> {
    let mut codec = Codec::from_tokenizer_bytes(&assets.tokenizer, text_profile(profile))?;
    codec.dense_content_identity = Some(DenseCodecContentIdentity {
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
        source: &DenseSourceContentIdentity,
        allowance: usize,
    ) -> io::Result<Self> {
        Self::load_dense_audited(
            DenseProfile::Qwen(QwenProfile::Qwen3),
            directory,
            source,
            allowance,
        )
    }
    /// Complete historical Qwen2 codec evidence using the original fixed ChatML
    /// formatter and a separate 128 MiB text allowance. No live owner is implied.
    pub fn load_qwen2_audited(
        directory: &Path,
        source: &DenseSourceContentIdentity,
        allowance: usize,
    ) -> io::Result<Self> {
        Self::load_dense_audited(
            DenseProfile::Qwen(QwenProfile::Qwen2),
            directory,
            source,
            allowance,
        )
    }
    /// Verified larger Qwen2.5 plain ChatML assets, bound to the exact source receipt.
    pub fn load_qwen15_audited(
        directory: &Path,
        source: &DenseSourceContentIdentity,
        allowance: usize,
    ) -> io::Result<Self> {
        Self::load_dense_audited(
            DenseProfile::Qwen(QwenProfile::Qwen15),
            directory,
            source,
            allowance,
        )
    }
    /// Complete historical delivery for the existing fixed Mistral formatter.
    /// Native factories separately require and retain the exact assembled owner.
    pub fn load_mistral_audited(
        directory: &Path,
        source: &DenseSourceContentIdentity,
        allowance: usize,
    ) -> io::Result<Self> {
        Self::load_dense_audited(DenseProfile::Mistral, directory, source, allowance)
    }
    /// Consumed text buffers paired with the attached Llama3 source receipt.
    pub fn load_llama3_audited(
        directory: &Path,
        source: &DenseSourceContentIdentity,
        allowance: usize,
    ) -> io::Result<Self> {
        Self::load_dense_audited(DenseProfile::Llama3, directory, source, allowance)
    }
    fn load_dense_audited(
        profile: DenseProfile,
        directory: &Path,
        source: &DenseSourceContentIdentity,
        allowance: usize,
    ) -> io::Result<Self> {
        admit(profile, source, allowance)?;
        let manifest = profile.manifest().map_err(invalid)?;
        let assets = assets(profile, directory, source.config_sha256(), manifest.files())?;
        build(profile, assets, source)
    }
    pub const fn dense_content_identity(&self) -> Option<&DenseCodecContentIdentity> {
        self.dense_content_identity.as_ref()
    }
    pub const fn mistral_content_identity(&self) -> Option<&DenseCodecContentIdentity> {
        match self.dense_content_identity.as_ref() {
            Some(value) if matches!(value.profile, DenseProfile::Mistral) => Some(value),
            _ => None,
        }
    }
    pub const fn qwen3_content_identity(&self) -> Option<&DenseCodecContentIdentity> {
        match self.dense_content_identity.as_ref() {
            Some(value) if matches!(value.profile, DenseProfile::Qwen(QwenProfile::Qwen3)) => {
                Some(value)
            }
            _ => None,
        }
    }
    pub const fn qwen2_content_identity(&self) -> Option<&DenseCodecContentIdentity> {
        match self.dense_content_identity.as_ref() {
            Some(value) if matches!(value.profile, DenseProfile::Qwen(QwenProfile::Qwen2)) => {
                Some(value)
            }
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

#[cfg(test)]
#[path = "dense_content/mistral_tests.rs"]
mod mistral_tests;

#[cfg(test)]
mod llama3_tests;
