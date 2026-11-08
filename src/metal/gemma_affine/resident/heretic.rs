//! Explicit pinned Mac mixed-format diagnostic. No managed-model eligibility.
use super::super::bf16::shared::{Matrix as Bf16Matrix, Workspace as Bf16Workspace};

mod graph;
use super::*;
use crate::mlx::gemma4::uncensored::{self, MatrixStorage};
pub use graph::{Backend, GenerationSession, MatrixRef, Model, Norms, Session};
use std::time::Instant;

pub const CONTEXT_BYTES: usize = 2 * CONTEXT_ALLOWANCE_BYTES;
const BF16_ROWS: usize = 8960;
const BF16_COLUMNS: usize = 1536;
const SHA256: [u8; 32] = [
    0xd4, 0xa2, 0x99, 0x5c, 0xe3, 0xfd, 0xaf, 0xbb, 0xd4, 0xa3, 0x37, 0x6b, 0xc8, 0x52, 0x4a, 0x8d,
    0x26, 0xbc, 0xbf, 0x0f, 0x51, 0x19, 0x77, 0x01, 0xb4, 0xe2, 0x1a, 0xb2, 0x7f, 0xaa, 0x37, 0xe1,
];

/// Two concrete precise pipelines; selection occurs once per projection.
pub struct HereticContext {
    affine: super::super::Context,
    bf16: Bf16Context,
    creation_seconds: f64,
}
impl HereticContext {
    pub fn new(available: usize) -> Result<Self> {
        process_budget(available)?;
        admit(CONTEXT_BYTES, available)?;
        let start = Instant::now();
        let affine = super::super::Context::new(CONTEXT_ALLOWANCE_BYTES)?;
        let bf16 = Bf16Context::new(CONTEXT_ALLOWANCE_BYTES)?;
        Ok(Self {
            affine,
            bf16,
            creation_seconds: start.elapsed().as_secs_f64(),
        })
    }
    /// Host time for device/library/pipeline constructors, separate from load.
    pub fn creation_seconds(&self) -> f64 {
        self.creation_seconds
    }
    pub fn device_name(&self) -> String {
        self.affine.device_name()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct SourceIdentity {
    pub config_sha256: [u8; 32],
    pub header_sha256: [u8; 32],
    pub container_sha256: [u8; 32],
    pub container_bytes: u64,
    pub matched_text_components: usize,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct HereticLoadReport {
    pub resident_requested_bytes: usize,
    pub admitted_load_policy_bytes: usize,
    pub maximum_source_capacity_bytes: usize,
    pub affine_matrices: usize,
    pub bf16_matrices: usize,
    pub vectors: usize,
    pub selected_source_bytes: usize,
    pub unused_shared_kv_source_bytes: usize,
    pub media_source_bytes: usize,
    pub prepare_seconds: f64,
    pub authenticated_affine_load_seconds: f64,
    pub bf16_load_seconds: f64,
    pub total_load_seconds: f64,
}

#[derive(Clone, Copy)]
enum MatrixSlot {
    Affine(usize),
    Bf16,
}

/// Original GPU coefficients and FP32 vector owners; no KV/session scratch.
/// The whole-file anchor is historical, with every selected second-pass byte
/// matched before publication. Excluded second-pass/live-path integrity is not
/// implied. No shard index is consumed by this fixed single-file loader.
pub struct HereticWeights {
    storage: Storage,
    bf16: Bf16Matrix,
    slots: Vec<MatrixSlot>,
    bindings: Bindings,
    identity: SourceIdentity,
    report: HereticLoadReport,
    limit: usize,
}

impl HereticWeights {
    /// Budget includes both context policies, cold metadata, hash/source overlap
    /// and all retained owners. Caller reader/config/codec/app storage is separate.
    pub fn load(
        context: &HereticContext,
        config: &[u8],
        mut reader: impl Read + Seek,
        available: usize,
    ) -> Result<Self> {
        process_budget(available)?;
        let total_start = Instant::now();
        admit(add(CONTEXT_BYTES, COLD_POLICY)?, available)?;
        let options = LoadOptions {
            container_bytes: uncensored::CONTAINER_BYTES,
            container_sha256: SHA256,
            available_bytes: available,
            modality: ModalityPolicy::TextOnly,
        };
        let mut prepared = source::heretic::Prepared::read(config, &mut reader, &options)?;
        let bindings = Bindings::heretic(&prepared.cold)?;
        let mut slots = owner_vec(278)?;
        let mut affine_index = 0;
        for matrix in prepared.cold.matrices() {
            slots.push(match matrix.storage {
                MatrixStorage::Affine(layout) => {
                    if prepared
                        .inputs
                        .matrices
                        .get(affine_index)
                        .is_none_or(|m| m.layout != layout)
                    {
                        return Err(Error::Options("Heretic affine role order differs"));
                    }
                    let slot = MatrixSlot::Affine(affine_index);
                    affine_index += 1;
                    slot
                }
                MatrixStorage::Bf16 { rows, columns }
                    if (rows, columns) == (BF16_ROWS, BF16_COLUMNS) =>
                {
                    MatrixSlot::Bf16
                }
                _ => return Err(Error::Options("Heretic BF16 matrix geometry differs")),
            });
        }
        if affine_index != 277
            || slots.len() != 278
            || slots
                .iter()
                .filter(|m| matches!(m, MatrixSlot::Bf16))
                .count()
                != 1
        {
            return Err(Error::Options("Heretic mixed role count differs"));
        }
        let slot_bytes = multiply(slots.capacity(), size_of::<MatrixSlot>())?;
        let new_base = add(size_of::<Self>(), slot_bytes)?;
        let extra_base = new_base.saturating_sub(size_of::<E2bWeights>());
        let bf16_payload = Bf16Matrix::payload_bytes(BF16_ROWS, BF16_COLUMNS)?;
        let raw_bytes = multiply(multiply(BF16_ROWS, BF16_COLUMNS)?, 2)?;
        // Conservative simultaneous reserve. Affine and BF16 source buffers
        // actually have disjoint lifetimes; neither is retained with the model.
        let reserve = add(
            add(extra_base, bf16_payload)?,
            add(raw_bytes, CONTEXT_ALLOWANCE_BYTES)?,
        )?;
        let local_available = available.checked_sub(reserve).ok_or(Error::Budget {
            required: reserve,
            limit: available,
        })?;
        let mut affine_options = options;
        affine_options.available_bytes = local_available;
        let prepare_seconds = total_start.elapsed().as_secs_f64();
        let start = Instant::now();
        let (storage, legacy) = load_storage(
            &context.affine,
            &mut prepared.inputs,
            &mut reader,
            &affine_options,
        )
        .map_err(|e| match e {
            Error::Budget { required, .. } => Error::Budget {
                required: required.saturating_add(reserve),
                limit: available,
            },
            other => other,
        })?;
        let authenticated_affine_load_seconds = start.elapsed().as_secs_f64();
        let base = add(
            subtract(legacy.resident_requested_bytes, size_of::<E2bWeights>())?,
            new_base,
        )?;
        let start = Instant::now();
        let mut raw = owner_vec::<u8>(raw_bytes)?;
        let full_policy = add(
            add(legacy.admitted_load_policy_bytes, extra_base)?,
            add(bf16_payload, add(raw.capacity(), CONTEXT_ALLOWANCE_BYTES)?)?,
        )?;
        admit(full_policy, available)?;
        raw.resize(raw_bytes, 0);
        prepared.read_bf16(&mut reader, &mut raw)?;
        // Matrix's inline descriptor is already counted in this owner's base.
        let other = add(base, add(raw.capacity(), add(CONTEXT_BYTES, COLD_POLICY)?)?)?;
        let local = add(
            available.checked_sub(other).ok_or(Error::Budget {
                required: other,
                limit: available,
            })?,
            size_of::<Bf16Matrix>(),
        )?;
        let bf16 = context
            .bf16
            .resident_matrix(BF16_ROWS, BF16_COLUMNS, &raw, local)?;
        let maximum_source_capacity_bytes =
            legacy.maximum_source_capacity_bytes.max(raw.capacity());
        drop(raw);
        let bf16_load_seconds = start.elapsed().as_secs_f64();
        let proof = prepared.inputs.finish(&options)?;
        let resident_requested_bytes = add(base, bf16.physical_bytes())?;
        let report = HereticLoadReport {
            resident_requested_bytes,
            admitted_load_policy_bytes: full_policy,
            maximum_source_capacity_bytes,
            affine_matrices: storage.matrices.len(),
            bf16_matrices: 1,
            vectors: storage.vectors.len(),
            selected_source_bytes: prepared.cold.text_payload_bytes(),
            unused_shared_kv_source_bytes: prepared.cold.unused_shared_kv_bytes(),
            media_source_bytes: prepared.cold.media_bytes(),
            prepare_seconds,
            authenticated_affine_load_seconds,
            bf16_load_seconds,
            total_load_seconds: total_start.elapsed().as_secs_f64(),
        };
        let identity = SourceIdentity {
            config_sha256: *proof.config_sha256(),
            header_sha256: *proof.header_sha256(),
            container_sha256: *proof.container_sha256(),
            container_bytes: proof.container_bytes(),
            matched_text_components: proof.text_components(),
        };
        Ok(Self {
            storage,
            bf16,
            slots,
            bindings,
            identity,
            report,
            limit: available,
        })
    }
    pub fn identity(&self) -> &SourceIdentity {
        &self.identity
    }
    pub fn report(&self) -> &HereticLoadReport {
        &self.report
    }
}

fn process_budget(bytes: usize) -> Result<()> {
    if bytes == 0 || bytes > crate::weights::INITIAL_MEMORY_LIMIT_BYTES {
        return Err(Error::Options("Heretic process budget outside6GiB policy"));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
