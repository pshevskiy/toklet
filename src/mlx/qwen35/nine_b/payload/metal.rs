//! Separate pinned9B consuming GPU ownership; no implicit4B/source eligibility.
use super::{ContentIdentity, PayloadStore};
pub use crate::mlx::qwen35::payload::metal::LoadReport;
use crate::{
    metal::{
        Error,
        gemma_affine::{Context, Weights},
    },
    mlx::qwen35::{
        invalid,
        nine_b::Plan,
        payload::metal::{GpuComponents, upload_components},
    },
};
use std::{collections::BTreeMap, mem::size_of};

type Result<T> = std::result::Result<T, Error>;

/// Original GPU packed matrices and small exact BF16-to-FP32 CPU tensors.
/// No Context borrow, session state, codec, or CPU matrix backing remains.
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
    /// Consume only this completely authenticated9B source, dropping original
    /// CPU packed owners after each upload. No full CPU matrix copy remains.
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
    /// Cold name resolution. Numerical model views carry the resolved slot.
    pub fn matrix(
        &self,
        name: &str,
    ) -> std::result::Result<(usize, &Weights), crate::weights::Error> {
        let slot = *self
            .slots
            .get(name)
            .ok_or_else(|| invalid("Qwen3.5-9B Metal matrix not found"))?;
        Ok((slot, &self.matrices[slot]))
    }
    pub fn vector(&self, name: &str) -> std::result::Result<&[f32], crate::weights::Error> {
        self.vectors
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| invalid("Qwen3.5-9B Metal vector not found"))
    }
    pub fn convolution(&self, name: &str) -> std::result::Result<&[f32], crate::weights::Error> {
        self.convolutions
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| invalid("Qwen3.5-9B Metal convolution not found"))
    }
}
