//! Cold consumed-asset pairing for one qualified Smol source profile.
//! This does not associate a codec with a live model or grant app capability.
use super::{CONFIG_SHA, Codec, Profile, TOKENIZER_ALLOWANCE, TOKENIZER_SHA, invalid};
use crate::{
    mlx::qwen::SourceContentIdentity,
    weights::{audit::open_regular, content::ExpectedFile},
};
use sha2::{Digest, Sha256};
use std::{
    io::{self, Read},
    path::Path,
};

const MODEL_CONFIG_SHA: &str = "5da91905efdfde905f566d76d4031e6ca7c4cb2a01dc4e3812a70b855188fc73";

/// Historical evidence that the exact consumed tokenizer buffers matched the
/// supplied authenticated source content. Copies survive model/codec drops and
/// later file changes. This is neither a resident-model hash nor live ownership.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodecContentIdentity {
    source: SourceContentIdentity,
    tokenizer_config: [u8; 32],
    tokenizer: [u8; 32],
    // Absent on the original three-asset route. Never enrich an existing handle.
    complete: Option<CompleteAssets>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CompleteAssets {
    manifest: [u8; 32],
    ancillary: [u8; 32],
}
impl CodecContentIdentity {
    /// Historical six-file completion, absent for the legacy audited codec.
    pub const fn complete_manifest_sha256(&self) -> Option<[u8; 32]> {
        match self.complete {
            Some(value) => Some(value.manifest),
            None => None,
        }
    }
    /// Verified ancillary distribution bytes, not parsed template input.
    pub const fn ancillary_special_map_sha256(&self) -> Option<[u8; 32]> {
        match self.complete {
            Some(value) => Some(value.ancillary),
            None => None,
        }
    }
    pub const fn source(&self) -> &SourceContentIdentity {
        &self.source
    }
    pub const fn tokenizer_config_sha256(&self) -> [u8; 32] {
        self.tokenizer_config
    }
    pub const fn tokenizer_sha256(&self) -> [u8; 32] {
        self.tokenizer
    }
}

fn matches(digest: [u8; 32], expected: &str) -> bool {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    expected.len() == 64
        && digest
            .iter()
            .zip(expected.as_bytes().chunks_exact(2))
            .all(|(&b, pair)| {
                pair[0] == HEX[(b >> 4) as usize] && pair[1] == HEX[(b & 15) as usize]
            })
}
fn admit(source: &SourceContentIdentity, allowance: usize) -> io::Result<()> {
    if allowance < TOKENIZER_ALLOWANCE {
        return Err(invalid(
            "Audited Smol codec requires a separate 64 MiB allowance",
        ));
    }
    if source.repository() != "mlx-community/SmolLM2-135M-Instruct-8bit"
        || source.revision() != "0f0d9b8218915bc34d401e1a340b8c049d300d5e"
        || !matches(source.config_sha256(), MODEL_CONFIG_SHA)
        || !matches(
            source.index_sha256(),
            "d016e7514e8bfed162eb9d0716c73244c1b73792db3143e92e54138c1772db44",
        )
        || !matches(
            source.header_sha256(),
            "357edb41f2242613ddddcbe671be7c60923abf4850c77105c63a870df4ccb1f8",
        )
        || !matches(
            source.reference_container_sha256(),
            "ec569c771e56424636be99625aba9031abc525a68a5f4ab16d678eca96fef12c",
        )
        || source.physical_ranges() != 694
        || source.verified_reads() < 694
        || source.payload_bytes() != 142955136
        || source.container_bytes() != 143030401
    {
        return Err(invalid(
            "Unsupported authenticated source content for Smol chat",
        ));
    }
    Ok(())
}
fn bounded_regular(path: &Path, limit: usize) -> io::Result<Vec<u8>> {
    let file = open_regular(path).map_err(invalid)?;
    let mut bytes = Vec::new();
    file.take((limit + 1) as u64).read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err(invalid("Audited chat asset exceeds its size bound"));
    }
    Ok(bytes)
}
fn authenticate(bytes: &[u8], expected: &str) -> io::Result<[u8; 32]> {
    let digest = Sha256::digest(bytes).into();
    if !matches(digest, expected) {
        return Err(invalid(
            "Audited chat asset differs from the supported content",
        ));
    }
    Ok(digest)
}
impl Codec {
    /// Validate consumed assets against supplied historical source evidence.
    /// The caller reserves this independent allowance before model/codec
    /// allocation. Source-only loading can also supply evidence: live model
    /// association must be established separately by the engine owner.
    pub fn load_smol_audited(
        directory: &Path,
        source: &SourceContentIdentity,
        allowance: usize,
    ) -> io::Result<Self> {
        // No file opening or tokenizer parsing occurs before admission.
        admit(source, allowance)?;
        let config = bounded_regular(&directory.join("config.json"), 16 * 1024)?;
        let config_digest = authenticate(&config, MODEL_CONFIG_SHA)?;
        if config_digest != source.config_sha256() {
            return Err(invalid("Consumed chat config differs from source content"));
        }
        let tokenizer_config =
            bounded_regular(&directory.join("tokenizer_config.json"), 16 * 1024)?;
        let tokenizer_config = authenticate(&tokenizer_config, CONFIG_SHA)?;
        let bytes = bounded_regular(&directory.join("tokenizer.json"), 4 * 1024 * 1024)?;
        let tokenizer = authenticate(&bytes, TOKENIZER_SHA)?;
        let mut codec = Self::from_tokenizer_bytes(&bytes, Profile::Smol)?;
        // Publish only after the same authenticated bytes parse and pass every
        // legacy policy check. Neither path nor shape serves as authority.
        codec.content_identity = Some(CodecContentIdentity {
            source: *source,
            tokenizer_config,
            tokenizer,
            complete: None,
        });
        Ok(codec)
    }
    pub const fn content_identity(&self) -> Option<&CodecContentIdentity> {
        self.content_identity.as_ref()
    }
}

pub(super) fn verified(
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

mod complete;

#[cfg(test)]
pub(crate) mod tests;
