//! Cold, streaming text-only storage admission for the inspected E2B layout.
//!
//! Every container byte is hashed; excluded modalities are read but never
//! retained. No partial store is published on failure. This is storage/value
//! admission only, not an executable family, numerical or device qualification.

use super::{E2bPlan, ModalityPolicy, Storage, inspect, invalid};
use crate::mlx::affine::{Layout, Matrix};
use crate::weights::Error;
use half::bf16;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Read;

#[cfg(test)]
mod tests;

const MAX_HEADER_BYTES: usize = 1024 * 1024;
const STREAM_BYTES: usize = 64 * 1024;
/// Cold config/index/header/plan, maps, names and descriptor reservation. This
/// covers caller metadata during loading, not tokenizer, session or process RSS.
const METADATA_RESERVATION: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy)]
pub struct LoadOptions {
    /// Obtained from an independently admitted immutable download identity.
    pub container_bytes: u64,
    pub container_sha256: [u8; 32],
    pub memory_limit_bytes: usize,
    pub modality: ModalityPolicy,
}

pub struct LoadReport {
    pub container_bytes: u64,
    pub container_sha256: [u8; 32],
    pub text_source_bytes: usize,
    pub excluded_source_bytes: usize,
    pub packed_matrices: usize,
    pub fp32_vectors: usize,
    /// Actual retained tensor Vec capacities, excluding metadata reservations.
    pub tensor_capacity_bytes: usize,
    pub metadata_reservation_bytes: usize,
    /// Includes retained tensors, vector conversion overlap and streaming buffer.
    pub admitted_load_bytes: usize,
}

struct Packed {
    layout: Layout,
    weight: Vec<u8>,
    scales: Vec<u8>,
    biases: Vec<u8>,
}

/// Keep this owner alive while its borrowed weights/model/session are used.
/// Resolve these cold name lookups once when constructing a concrete model.
/// No mutable storage or self-referential model/native handle is exposed.
pub struct PayloadStore {
    plan: E2bPlan,
    matrices: BTreeMap<String, Packed>,
    vectors: BTreeMap<String, Vec<f32>>,
    report: LoadReport,
}
impl PayloadStore {
    pub fn plan(&self) -> &E2bPlan {
        &self.plan
    }
    pub fn report(&self) -> &LoadReport {
        &self.report
    }
    /// Cold borrowed view construction checks metadata again. Never call this
    /// name-based builder in a numerical loop; retain the resulting Matrix.
    pub fn matrix(&self, name: &str) -> Result<Matrix<'_>, Error> {
        let m = self
            .matrices
            .get(name)
            .ok_or_else(|| invalid("Gemma packed tensor not found"))?;
        Matrix::from_le_bytes(m.layout, &m.weight, &m.scales, &m.biases).map_err(|e| invalid(e.0))
    }
    pub fn vector(&self, name: &str) -> Result<&[f32], Error> {
        self.vectors
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| invalid("Gemma vector not found"))
    }
}
fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b)
        .ok_or_else(|| invalid("Gemma payload size overflow"))
}
fn multiply(a: usize, b: usize) -> Result<usize, Error> {
    let n = a
        .checked_mul(b)
        .ok_or_else(|| invalid("Gemma payload size overflow"))?;
    if n > isize::MAX as usize {
        return Err(invalid("Gemma payload allocation bound"));
    }
    Ok(n)
}
fn buffer(size: usize) -> Result<Vec<u8>, Error> {
    if size > isize::MAX as usize {
        return Err(invalid("Gemma payload allocation bound"));
    }
    let mut v = Vec::new();
    v.try_reserve_exact(size).map_err(|_| Error::Allocation)?;
    v.resize(size, 0);
    Ok(v)
}
fn read_exact<R: Read>(reader: &mut R, bytes: &mut [u8], sha: &mut Sha256) -> Result<(), Error> {
    reader
        .read_exact(bytes)
        .map_err(|e| invalid(format!("Gemma payload read: {e}")))?;
    sha.update(bytes);
    Ok(())
}

struct Reservation {
    tensors: usize,
    overlap: usize,
    total: usize,
}
fn reservation(plan: &E2bPlan) -> Result<Reservation, Error> {
    let mut tensors = 0;
    let mut overlap = 0;
    for t in &plan.logical {
        let n = match t.storage {
            Storage::Affine(l) => l.total_bytes(),
            Storage::Bf16Vector => {
                let super::Shape::Vector(n) = t.shape else {
                    return Err(invalid("Gemma vector shape"));
                };
                overlap = overlap.max(multiply(n, 2)?);
                multiply(n, 4)?
            }
        };
        tensors = add(tensors, n)?;
    }
    let total = add(
        add(add(tensors, overlap)?, STREAM_BYTES)?,
        METADATA_RESERVATION,
    )?;
    Ok(Reservation {
        tensors,
        overlap,
        total,
    })
}
fn check_budget(required: usize, limit: usize) -> Result<(), Error> {
    if required > limit {
        return Err(Error::Budget { required, limit });
    }
    Ok(())
}
fn admit_capacity(peak: &mut usize, required: usize, limit: usize) -> Result<(), Error> {
    check_budget(required, limit)?;
    *peak = (*peak).max(required);
    Ok(())
}
fn vector(raw: &[u8]) -> Result<Vec<f32>, Error> {
    if !raw.len().is_multiple_of(2) {
        return Err(invalid("Gemma BF16 vector size"));
    }
    let count = raw.len() / 2;
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| Error::Allocation)?;
    for pair in raw.chunks_exact(2) {
        let v = bf16::from_le_bytes(pair.try_into().unwrap()).to_f32();
        if !v.is_finite() {
            return Err(invalid("nonfinite Gemma vector"));
        }
        result.push(v);
    }
    Ok(result)
}
fn validate_packed(p: &Packed) -> Result<(), Error> {
    if p.layout.bits() != 4 || p.layout.metadata_dtype() != crate::mlx::affine::MetadataDtype::BF16
    {
        return Err(invalid("Gemma payload packed representation"));
    }
    Matrix::from_le_bytes(p.layout, &p.weight, &p.scales, &p.biases).map_err(|e| invalid(e.0))?;
    // With a linear affine reconstruction, finite FP32 endpoints at codes0/15
    // imply finite results for every intervening four-bit code. This stronger
    // admission intentionally includes unused codes; no whole matrix expansion.
    for (s, b) in p.scales.chunks_exact(2).zip(p.biases.chunks_exact(2)) {
        let scale = f64::from(bf16::from_le_bytes(s.try_into().unwrap()).to_f32());
        let bias = f64::from(bf16::from_le_bytes(b.try_into().unwrap()).to_f32());
        for code in [0., 15.] {
            if !((scale * code + bias) as f32).is_finite() {
                return Err(invalid("Gemma affine reconstruction overflow"));
            }
        }
    }
    Ok(())
}

/// Consume one complete safetensors stream with an externally pinned identity.
/// Config/index/actual bounded header are admitted together before payload
/// buffers. All data stays private until byte count, EOF, digest and finite
/// values pass. The returned reservation is numerical storage/load accounting;
/// it neither admits a session nor grants a process/device allowance.
pub fn load<R: Read>(
    config: &[u8],
    index: &[u8],
    mut reader: R,
    options: LoadOptions,
) -> Result<PayloadStore, Error> {
    check_budget(
        add(METADATA_RESERVATION, STREAM_BYTES)?,
        options.memory_limit_bytes,
    )?;
    // Bound caller metadata before any header allocation or payload read.
    if config.len() > MAX_HEADER_BYTES || index.len() > MAX_HEADER_BYTES {
        return Err(invalid("Gemma payload metadata bound"));
    }
    let mut sha = Sha256::new();
    let mut length = [0; 8];
    read_exact(&mut reader, &mut length, &mut sha)?;
    let size = usize::try_from(u64::from_le_bytes(length))
        .map_err(|_| invalid("Gemma header size overflow"))?;
    if size == 0 || size > MAX_HEADER_BYTES {
        return Err(invalid("Gemma payload header bound"));
    }
    let mut header = buffer(size)?;
    read_exact(&mut reader, &mut header, &mut sha)?;
    let plan = inspect(
        config,
        index,
        &header,
        options.container_bytes,
        options.modality,
    )?;
    load_admitted(&mut reader, plan, sha, options)
}

fn load_admitted<R: Read>(
    reader: &mut R,
    plan: E2bPlan,
    mut sha: Sha256,
    options: LoadOptions,
) -> Result<PayloadStore, Error> {
    let reserve = reservation(&plan)?;
    check_budget(reserve.total, options.memory_limit_bytes)?;
    let mut admitted = reserve.total;
    let mut physical = BTreeMap::new();
    let mut spans: Vec<_> = plan
        .text
        .iter()
        .map(|(n, t)| (Some(n), t))
        .chain(plan.excluded.values().map(|t| (None, t)))
        .collect();
    spans.sort_unstable_by_key(|(_, t)| t.range.start);
    let mut stream = [0; STREAM_BYTES];
    let mut actual_raw = 0_usize;
    for (name, t) in spans {
        let count = t.range.end - t.range.start;
        if let Some(name) = name {
            let mut bytes = buffer(count)?;
            actual_raw = add(actual_raw, bytes.capacity())?;
            // Bound actual capacity before retaining any additional payload.
            admit_capacity(
                &mut admitted,
                add(
                    add(add(actual_raw, reserve.overlap)?, STREAM_BYTES)?,
                    METADATA_RESERVATION,
                )?,
                options.memory_limit_bytes,
            )?;
            for chunk in bytes.chunks_mut(STREAM_BYTES) {
                read_exact(reader, chunk, &mut sha)?;
            }
            physical.insert(name.clone(), bytes);
        } else {
            let mut left = count;
            while left > 0 {
                let n = left.min(stream.len());
                read_exact(reader, &mut stream[..n], &mut sha)?;
                left -= n;
            }
        }
    }
    let mut tail = [0];
    if reader
        .read(&mut tail)
        .map_err(|e| invalid(format!("Gemma payload EOF: {e}")))?
        != 0
    {
        return Err(invalid("Gemma payload has trailing bytes"));
    }
    let digest: [u8; 32] = sha.finalize().into();
    if digest != options.container_sha256 {
        return Err(invalid("Gemma payload SHA256 differs"));
    }
    let mut matrices = BTreeMap::new();
    let mut vectors = BTreeMap::new();
    let mut capacity = 0;
    let mut live = actual_raw;
    for t in &plan.logical {
        let take = |p: &mut BTreeMap<String, Vec<u8>>, n: &str| {
            p.remove(n)
                .ok_or_else(|| invalid("Gemma payload physical tensor missing"))
        };
        match t.storage {
            Storage::Affine(layout) => {
                let base = t
                    .name
                    .strip_suffix(".weight")
                    .ok_or_else(|| invalid("Gemma affine tensor name"))?;
                let p = Packed {
                    layout,
                    weight: take(&mut physical, &t.name)?,
                    scales: take(&mut physical, &format!("{base}.scales"))?,
                    biases: take(&mut physical, &format!("{base}.biases"))?,
                };
                validate_packed(&p)?;
                for n in [
                    p.weight.capacity(),
                    p.scales.capacity(),
                    p.biases.capacity(),
                ] {
                    capacity = add(capacity, n)?;
                }
                matrices.insert(t.name.clone(), p);
            }
            Storage::Bf16Vector => {
                let raw = take(&mut physical, &t.name)?;
                let wanted = multiply(raw.len() / 2, 4)?;
                admit_capacity(
                    &mut admitted,
                    add(add(add(live, wanted)?, STREAM_BYTES)?, METADATA_RESERVATION)?,
                    options.memory_limit_bytes,
                )?;
                let v = vector(&raw)?;
                let bytes = multiply(v.capacity(), 4)?;
                admit_capacity(
                    &mut admitted,
                    add(add(add(live, bytes)?, STREAM_BYTES)?, METADATA_RESERVATION)?,
                    options.memory_limit_bytes,
                )?;
                live = add(live - raw.capacity(), bytes)?;
                drop(raw);
                capacity = add(capacity, bytes)?;
                vectors.insert(t.name.clone(), v);
            }
        }
        admit_capacity(
            &mut admitted,
            add(
                add(add(capacity, reserve.overlap)?, STREAM_BYTES)?,
                METADATA_RESERVATION,
            )?,
            options.memory_limit_bytes,
        )?;
    }
    if capacity != live {
        return Err(invalid("Gemma payload capacity accounting"));
    }
    if !physical.is_empty() {
        return Err(invalid("Gemma payload unused text tensors"));
    }
    let actual = add(
        add(add(capacity, reserve.overlap)?, STREAM_BYTES)?,
        METADATA_RESERVATION,
    )?;
    admit_capacity(&mut admitted, actual, options.memory_limit_bytes)?;
    let report = LoadReport {
        container_bytes: options.container_bytes,
        container_sha256: digest,
        text_source_bytes: plan.text_payload_bytes,
        excluded_source_bytes: plan.excluded_payload_bytes,
        packed_matrices: matrices.len(),
        fp32_vectors: vectors.len(),
        tensor_capacity_bytes: capacity,
        metadata_reservation_bytes: METADATA_RESERVATION,
        admitted_load_bytes: admitted,
    };
    // Retained capacity can exceed the reservation on an allocator which rounds
    // requests upward, but it must still fit the caller's original allowance.
    debug_assert!(capacity >= reserve.tensors);
    Ok(PayloadStore {
        plan,
        matrices,
        vectors,
        report,
    })
}
