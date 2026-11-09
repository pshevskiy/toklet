//! Consuming authenticated packed ownership; no retained full CPU matrix copy.
use super::{Components, ContentIdentity, PayloadStore};
use crate::{
    metal::{
        Error,
        gemma_affine::{CONTEXT_ALLOWANCE_BYTES, Context, Weights},
    },
    mlx::qwen35::{Plan, invalid},
};
use std::{collections::BTreeMap, mem::size_of};

type Result<T> = std::result::Result<T, Error>;

const CHUNK: usize = 16384;
const LIMIT: usize = 6 * 1024 * 1024 * 1024;
const METADATA: usize = 64 * 1024 * 1024;
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or(Error::Options("Qwen Metal ownership overflow"))
}
fn admit(required: usize, limit: usize) -> Result<()> {
    if required > limit {
        return Err(Error::Budget { required, limit });
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct LoadReport {
    pub matrices: usize,
    pub packed_gpu_bytes: usize,
    pub small_cpu_capacity_bytes: usize,
    pub retained_requested_bytes: usize,
    pub retained_policy_bytes: usize,
    pub admitted_load_policy_bytes: usize,
    pub maximum_upload_source_capacity_bytes: usize,
    pub retained_cpu_matrix_bytes: usize,
}

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
fn retained_bytes(
    resident: usize,
    small: usize,
    unused_slots: usize,
    store_bytes: usize,
) -> Result<usize> {
    let unused = unused_slots
        .checked_mul(size_of::<Weights>())
        .ok_or(Error::Options("Qwen GPU slot capacity overflow"))?;
    add(add(add(resident, small)?, unused)?, store_bytes)
}

// Cold ownership only. Public wrappers retain concrete source Plans. The common
// mechanism neither selects a source nor participates in numerical dispatch.
pub(in crate::mlx::qwen35) struct GpuComponents {
    pub(in crate::mlx::qwen35) matrices: Vec<Weights>,
    pub(in crate::mlx::qwen35) slots: BTreeMap<String, usize>,
    pub(in crate::mlx::qwen35) vectors: BTreeMap<String, Vec<f32>>,
    pub(in crate::mlx::qwen35) convolutions: BTreeMap<String, Vec<f32>>,
    pub(in crate::mlx::qwen35) identity: ContentIdentity,
    pub(in crate::mlx::qwen35) report: LoadReport,
}

pub(in crate::mlx::qwen35) fn upload_components(
    context: &Context,
    components: Components,
    memory_limit_bytes: usize,
    store_bytes: usize,
) -> Result<GpuComponents> {
    let limit = memory_limit_bytes.min(LIMIT);
    let policy = add(CONTEXT_ALLOWANCE_BYTES, METADATA)?;
    let mut remaining = components.report.retained_tensor_capacity_bytes;
    let mut peak = add(remaining, policy)?;
    admit(peak, limit)?;
    let Components {
        matrices,
        vectors,
        convolutions,
        identity,
        report: _,
    } = components;
    let mut gpu = Vec::new();
    gpu.try_reserve_exact(matrices.len())
        .map_err(|_| Error::Native("Qwen GPU matrix slots unavailable".into()))?;
    let mut slots = BTreeMap::new();
    let mut resident = 0;
    let mut source_capacity = 0;
    for (name, p) in matrices {
        let base = add(add(remaining, resident)?, policy)?;
        admit(base, limit)?;
        let uploaded = context.weights_from_fill_reported(
            p.layout,
            CHUNK,
            limit - base,
            |first, child, codes, scales, biases| {
                let span = p
                    .layout
                    .row_span(first..first + child.rows())
                    .map_err(|e| Error::Options(e.0))?;
                if span.layout() != child {
                    return Err(Error::Options("Qwen upload child layout mismatch"));
                }
                codes.copy_from_slice(&p.codes[span.weight_range()]);
                scales.copy_from_slice(&p.scales[span.metadata_range()]);
                biases.copy_from_slice(&p.biases[span.metadata_range()]);
                Ok(())
            },
        );
        let (weights, stats) = uploaded.map_err(|e| match e {
            Error::Budget { required, .. } => Error::Budget {
                required: base.saturating_add(required),
                limit,
            },
            other => other,
        })?;
        peak = peak.max(add(base, stats.working_bytes)?);
        source_capacity = source_capacity.max(stats.source_capacity_bytes);
        resident = add(resident, weights.owned_bytes())?;
        slots.insert(name, gpu.len());
        gpu.push(weights);
        let old = add(
            add(p.codes.capacity(), p.scales.capacity())?,
            p.biases.capacity(),
        )?;
        remaining = remaining
            .checked_sub(old)
            .ok_or(Error::Options("Qwen consumed CPU ownership mismatch"))?;
        drop(p);
    }
    let small = vectors
        .values()
        .chain(convolutions.values())
        .try_fold(0, |n, v| {
            add(
                n,
                v.capacity()
                    .checked_mul(4)
                    .ok_or(Error::Options("Qwen small tensor capacity overflow"))?,
            )
        })?;
    if small != remaining || slots.len() != gpu.len() {
        return Err(Error::Options("Qwen retained CPU ownership mismatch"));
    }
    let retained = retained_bytes(resident, small, gpu.capacity() - gpu.len(), store_bytes)?;
    let retained_policy = add(retained, policy)?;
    peak = peak.max(retained_policy);
    admit(peak, limit)?;
    let report = LoadReport {
        matrices: gpu.len(),
        packed_gpu_bytes: resident,
        small_cpu_capacity_bytes: small,
        retained_requested_bytes: retained,
        retained_policy_bytes: retained_policy,
        admitted_load_policy_bytes: peak,
        maximum_upload_source_capacity_bytes: source_capacity,
        retained_cpu_matrix_bytes: 0,
    };
    Ok(GpuComponents {
        matrices: gpu,
        slots,
        vectors,
        convolutions,
        identity,
        report,
    })
}

impl PayloadStore {
    /// The consumed source is already completely authenticated. Original CPU
    /// bytes survive each upload callback, then drop before the next matrix.
    /// Failure drops unpublished GPU owners and all remaining consumed storage.
    /// Limit includes this Context policy and both source/destination overlap.
    pub fn into_metal(self, context: &Context, memory_limit_bytes: usize) -> Result<MetalStore> {
        let Self {
            plan,
            matrices,
            vectors,
            convolutions,
            identity,
            report,
        } = self;
        let GpuComponents {
            matrices,
            slots,
            vectors,
            convolutions,
            identity,
            report,
        } = upload_components(
            context,
            Components {
                matrices,
                vectors,
                convolutions,
                identity,
                report,
            },
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
            .ok_or_else(|| invalid("Qwen Metal matrix not found"))?;
        Ok((slot, &self.matrices[slot]))
    }
    pub fn vector(&self, name: &str) -> std::result::Result<&[f32], crate::weights::Error> {
        self.vectors
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| invalid("Qwen Metal vector not found"))
    }
    pub fn convolution(&self, name: &str) -> std::result::Result<&[f32], crate::weights::Error> {
        self.convolutions
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| invalid("Qwen Metal convolution not found"))
    }
}

#[cfg(test)]
mod accounting_tests {
    use super::*;
    #[test]
    fn retained_capacity_and_metadata_are_counted_once_without_a_gpu() {
        assert_eq!(
            retained_bytes(1024, 256, 2, 100).unwrap(),
            1380 + 2 * size_of::<Weights>()
        );
        assert_eq!(retained_bytes(0, 0, 0, 0).unwrap(), 0);
        assert!(admit(100, 100).is_ok());
        assert!(matches!(
            admit(100, 99),
            Err(Error::Budget {
                required: 100,
                limit: 99
            })
        ));
    }
    #[test]
    fn checked_ownership_arithmetic_rejects_overflow_without_a_gpu() {
        assert!(add(usize::MAX, 1).is_err());
        for args in [
            (usize::MAX, 1, 0, 0),
            (1, usize::MAX, 0, 0),
            (0, 0, usize::MAX, 0),
            (1, 0, 0, usize::MAX),
        ] {
            assert!(retained_bytes(args.0, args.1, args.2, args.3).is_err());
        }
    }
}
