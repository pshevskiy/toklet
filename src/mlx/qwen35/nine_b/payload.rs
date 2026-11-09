//! Separate pinned9B CPU ownership; no model, GPU, codec or device admission.
use super::{
    CONFIG_BYTES, CONTAINER_BYTES, CONTAINER_SHA256, HEADER_BYTES, ModalityPolicy, Plan, inspect,
};
use crate::{
    mlx::{affine::Matrix, qwen35::payload as cold},
    weights::Error,
};
use std::io::Read;

pub use cold::{ContentIdentity, LoadOptions, LoadReport};

/// Immutable original packed components and exactly promoted small BF16 values.
/// The distinct head remains an independent owner, without numerical binding.
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
            .ok_or_else(|| super::super::invalid("Qwen3.5-9B matrix not found"))?;
        Matrix::from_le_bytes(p.layout, &p.codes, &p.scales, &p.biases)
            .map_err(|e| super::super::invalid(e.0))
    }
    pub fn vector(&self, name: &str) -> Result<&[f32], Error> {
        self.components
            .vectors
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| super::super::invalid("Qwen3.5-9B vector not found"))
    }
    /// Original `[channels,4,1]`, four consecutive coefficients per channel.
    pub fn convolution(&self, name: &str) -> Result<&[f32], Error> {
        self.components
            .convolutions
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| super::super::invalid("Qwen3.5-9B convolution not found"))
    }
}

fn source_recipe(plan: &Plan) -> cold::SourceRecipe<'_> {
    cold::SourceRecipe {
        common: plan.logical_tensors(),
        convolutions: plan.convolutions(),
        physical: plan.physical_tensors(),
        bytes: plan.payload_bytes(),
    }
}

/// Authenticate one exact9B stream, including EOF, before returning any store,
/// identity or report. Budget covers cold CPU ownership and bounded temporaries;
/// it does not establish total process, numerical/GPU/session or application fit.
pub fn load<R: Read>(
    config: &[u8],
    mut reader: R,
    options: LoadOptions,
) -> Result<PayloadStore, Error> {
    cold::metadata_admission(options.memory_limit_bytes)?;
    if options.modality != ModalityPolicy::TextOnly || config.len() != CONFIG_BYTES {
        return Err(super::super::invalid(
            "pinned Qwen3.5-9B config/text-only request required",
        ));
    }
    let (header, sha) = cold::source_header(&mut reader, HEADER_BYTES)?;
    let plan = inspect(config, &header, CONTAINER_BYTES, options.modality)?;
    let identity = ContentIdentity::pending(config, &header, CONTAINER_BYTES);
    let components = cold::read_components(
        &mut reader,
        source_recipe(&plan),
        sha,
        identity,
        (8 + HEADER_BYTES) as u64,
        CONTAINER_SHA256,
        options.memory_limit_bytes,
    )?;
    Ok(PayloadStore { plan, components })
}

#[cfg(all(test, target_pointer_width = "64"))]
mod tests;
