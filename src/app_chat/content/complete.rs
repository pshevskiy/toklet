//! Complete Smol distribution evidence layered on the unchanged audited codec.
use super::*;
use crate::weights::content::smol_manifest;

fn complete_admit(source: &SourceContentIdentity, allowance: usize) -> io::Result<()> {
    // Preserve the original helper contract; this new route also caps its total
    // text allowance consistently with the native boundary before file access.
    admit(source, allowance)?;
    if allowance > crate::weights::INITIAL_MEMORY_LIMIT_BYTES {
        return Err(invalid("Complete Smol text allowance exceeds 6 GiB"));
    }
    let manifest = smol_manifest().map_err(invalid)?;
    let files = manifest.files();
    if source.config_sha256() != files[0].sha256
        || source.reference_container_sha256() != files[1].sha256
        || source.index_sha256() != files[2].sha256
        || source.container_bytes() != files[1].bytes
    {
        return Err(invalid("Smol source differs from complete manifest"));
    }
    Ok(())
}

impl Codec {
    /// Complete the six-file historical evidence without changing the existing
    /// three-asset constructor. Source evidence alone does not bind a live model.
    /// Ancillary buffer is released before tokenizer construction; it is covered
    /// by the existing separate 64 MiB text allowance, not an added model copy.
    pub fn load_smol_complete_audited(
        directory: &Path,
        source: &SourceContentIdentity,
        allowance: usize,
    ) -> io::Result<Self> {
        complete_admit(source, allowance)?;
        let manifest = smol_manifest().map_err(invalid)?;
        let (ancillary, digest) = verified(directory, manifest.files()[5], 16 * 1024)?;
        drop(ancillary);
        let mut codec = Self::load_smol_audited(directory, source, allowance)?;
        // The old constructor publishes only after same-buffer tokenizer parse
        // and policy checks. This local codec is still unpublished on failure.
        let identity = codec
            .content_identity
            .as_mut()
            .ok_or_else(|| invalid("Missing audited Smol codec evidence"))?;
        identity.complete = Some(CompleteAssets {
            manifest: manifest.identity(),
            ancillary: digest,
        });
        Ok(codec)
    }
}

#[cfg(test)]
mod tests;
