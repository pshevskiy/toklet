//! Consuming, externally pinned2B CPU ownership. Original FP32 A_log values
//! never pass through BF16. No numerical or parent-lineage claim is made.
use super::{ArtifactPins, ModalityPolicy, Plan, inspect};
use crate::{
    mlx::{
        affine::Matrix,
        qwen35::{invalid, payload as cold},
    },
    weights::Error,
};
use sha2::{Digest, Sha256};
use std::io::Read;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;
#[cfg(all(test, target_pointer_width = "64"))]
mod tests;
pub use cold::{ContentIdentity, LoadOptions, LoadReport};

pub struct PayloadStore {
    plan: Plan,
    components: cold::Components,
}
impl PayloadStore {
    pub fn plan(&self) -> &Plan {
        &self.plan
    }
    pub fn identity(&self) -> ContentIdentity {
        self.components.identity
    }
    pub fn report(&self) -> &LoadReport {
        &self.components.report
    }
    pub fn matrix(&self, name: &str) -> Result<Matrix<'_>, Error> {
        let p = self
            .components
            .matrices
            .get(name)
            .ok_or_else(|| invalid("Qwen3.5-2B matrix not found"))?;
        Matrix::from_le_bytes(p.layout, &p.codes, &p.scales, &p.biases).map_err(|e| invalid(e.0))
    }
    pub fn vector(&self, name: &str) -> Result<&[f32], Error> {
        self.components
            .vectors
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| invalid("Qwen3.5-2B vector not found"))
    }
    /// OriginalBF16 `[channels,4,1]`: four consecutive taps per channel.
    pub fn convolution(&self, name: &str) -> Result<&[f32], Error> {
        self.components
            .convolutions
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| invalid("Qwen3.5-2B convolution not found"))
    }
}
fn source_recipe(plan: &Plan) -> cold::SourceRecipe<'_> {
    cold::SourceRecipe {
        common: plan.logical_tensors(),
        convolutions: plan.convolutions(),
        physical: plan.physical_tensors(),
        source_payload_bytes: plan.payload_bytes().total,
        source_f32_vector_bytes: plan.source_f32_vector_bytes(),
    }
}
fn digest_text(digest: &[u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Bind the explicit conversion contract pins, exact shape/typed recipe and
/// complete streamed container including EOF before publishing any ownership.
/// The cold budget covers CPU tensors/temporary conversion, not process RSS or
/// simultaneous sessions/application state. No ordinary source is admitted.
pub fn load<R: Read>(
    config: &[u8],
    mut reader: R,
    options: LoadOptions,
    pins: &ArtifactPins,
) -> Result<PayloadStore, Error> {
    cold::metadata_admission(options.memory_limit_bytes)?;
    pins.validate()?;
    if options.modality != ModalityPolicy::TextOnly
        || config.len() != pins.config_bytes
        || <[u8; 32]>::from(Sha256::digest(config)) != pins.config_sha256
    {
        return Err(invalid(
            "pinned Qwen3.5-2B config/text-only request required",
        ));
    }
    let (header, sha) = cold::source_header(&mut reader, pins.header_bytes)?;
    let plan = inspect(
        config,
        &header,
        pins.container_bytes,
        options.modality,
        pins,
    )?;
    let identity = ContentIdentity::pending(config, &header, pins.container_bytes);
    let components = cold::read_components(
        &mut reader,
        source_recipe(&plan),
        sha,
        identity,
        plan.payload_offset() as u64,
        &digest_text(&pins.container_sha256),
        options.memory_limit_bytes,
    )?;
    Ok(PayloadStore { plan, components })
}
