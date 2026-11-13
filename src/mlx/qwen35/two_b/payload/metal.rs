//! Distinct2B consuming GPU owner; source-specific cold admission stays explicit.
use super::{ContentIdentity, PayloadStore};
pub use crate::mlx::qwen35::payload::metal::LoadReport;
use crate::{
    metal::{
        Error,
        gemma_affine::{Context, Weights},
    },
    mlx::qwen35::{
        invalid,
        payload::metal::{GpuComponents, upload_components},
        two_b::Plan,
    },
};
use std::{collections::BTreeMap, mem::size_of};
type Result<T> = std::result::Result<T, Error>;

/// Immutable packed GPU weights and exact FP32 small CPU owners. The original
/// FP32 A_log bits survive unchanged; other small parameters are widened BF16.
pub struct MetalStore {
    plan: Plan,
    matrices: Vec<Weights>,
    slots: BTreeMap<String, usize>,
    vectors: BTreeMap<String, Vec<f32>>,
    convolutions: BTreeMap<String, Vec<f32>>,
    identity: ContentIdentity,
    report: LoadReport,
}
impl PayloadStore {
    /// Consume authenticated CPU ownership, dropping each original packed
    /// matrix after its bounded upload. Failure publishes no partial store.
    pub fn into_metal(self, context: &Context, memory_limit_bytes: usize) -> Result<MetalStore> {
        let Self { plan, components } = self;
        let GpuComponents {
            matrices,
            slots,
            vectors,
            convolutions,
            identity,
            report,
        } = upload_components(
            context,
            components,
            memory_limit_bytes,
            size_of::<MetalStore>(),
        )?;
        Ok(MetalStore {
            plan,
            matrices,
            slots,
            vectors,
            convolutions,
            identity,
            report,
        })
    }
}
impl MetalStore {
    pub fn plan(&self) -> &Plan {
        &self.plan
    }
    pub fn identity(&self) -> ContentIdentity {
        self.identity
    }
    pub fn report(&self) -> LoadReport {
        self.report
    }
    /// Resolve this cold name once when assembling borrowed numerical views.
    pub fn matrix(
        &self,
        name: &str,
    ) -> std::result::Result<(usize, &Weights), crate::weights::Error> {
        let slot = *self
            .slots
            .get(name)
            .ok_or_else(|| invalid("Qwen3.5-2B Metal matrix not found"))?;
        Ok((slot, &self.matrices[slot]))
    }
    pub fn vector(&self, name: &str) -> std::result::Result<&[f32], crate::weights::Error> {
        self.vectors
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| invalid("Qwen3.5-2B Metal vector not found"))
    }
    pub fn convolution(&self, name: &str) -> std::result::Result<&[f32], crate::weights::Error> {
        self.convolutions
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| invalid("Qwen3.5-2B Metal convolution not found"))
    }
}
