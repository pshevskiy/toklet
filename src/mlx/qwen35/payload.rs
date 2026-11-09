//! Cold packed CPU ownership, not a hybrid executor or device admission.
//!
//! The source is consumed sequentially as bytes, including its unaligned header
//! origin. No tensor, receipt or store escapes before complete authentication.
use super::*;
use crate::mlx::affine::Matrix;
use crate::mlx::payload::{buffer, read_exact, validate_q4_bf16};
use half::bf16;
use std::io::Read;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;
#[cfg(test)]
mod tests;

const STREAM_BYTES: usize = 64 * 1024;
/// Cold plans/maps/names/header and caller metadata, not process/framework RSS.
const METADATA_BYTES: usize = 64 * 1024 * 1024;
const MAX_LOAD_BYTES: usize = 6 * 1024 * 1024 * 1024;

#[derive(Clone, Copy)]
pub struct LoadOptions {
    pub memory_limit_bytes: usize,
    pub modality: ModalityPolicy,
}

/// Historical exact bytes consumed, not current path, codec or runtime proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContentIdentity {
    config: [u8; 32],
    header: [u8; 32],
    container: [u8; 32],
    bytes: u64,
}
impl ContentIdentity {
    pub(super) fn pending(config: &[u8], header: &[u8], bytes: u64) -> Self {
        Self {
            config: Sha256::digest(config).into(),
            header: Sha256::digest(header).into(),
            container: [0; 32],
            bytes,
        }
    }
    pub fn config_sha256(&self) -> &[u8; 32] {
        &self.config
    }
    pub fn header_sha256(&self) -> &[u8; 32] {
        &self.header
    }
    pub fn container_sha256(&self) -> &[u8; 32] {
        &self.container
    }
    pub fn container_bytes(&self) -> u64 {
        self.bytes
    }
}

pub struct LoadReport {
    pub source_payload_bytes: usize,
    pub packed_matrices: usize,
    pub fp32_vectors: usize,
    pub fp32_convolutions: usize,
    pub retained_tensor_capacity_bytes: usize,
    pub metadata_reservation_bytes: usize,
    pub admitted_load_bytes: usize,
}

pub(super) struct Packed {
    pub(super) layout: Layout,
    pub(super) codes: Vec<u8>,
    pub(super) scales: Vec<u8>,
    pub(super) biases: Vec<u8>,
}

// Borrowed cold descriptors only. Public stores keep their source-specific Plan;
// the streaming mechanism never chooses a source or a numerical backend.
pub(super) struct SourceRecipe<'a> {
    pub(super) common: &'a [TensorPlan],
    pub(super) convolutions: &'a [ConvolutionPlan],
    pub(super) physical: &'a BTreeMap<String, PhysicalTensor>,
    pub(super) bytes: PayloadBytes,
}
pub(super) struct Components {
    pub(super) matrices: BTreeMap<String, Packed>,
    pub(super) vectors: BTreeMap<String, Vec<f32>>,
    pub(super) convolutions: BTreeMap<String, Vec<f32>>,
    pub(super) identity: ContentIdentity,
    pub(super) report: LoadReport,
}
fn source_recipe(plan: &Plan) -> SourceRecipe<'_> {
    SourceRecipe {
        common: &plan.common,
        convolutions: &plan.convolutions,
        physical: &plan.physical,
        bytes: plan.bytes,
    }
}

/// Immutable original matrix components plus exact BF16-to-FP32 small tensors.
/// Resolve these cold names once when building a concrete future executor.
/// A GPU performance loader need not retain this full CPU reference owner.
pub struct PayloadStore {
    plan: Plan,
    matrices: BTreeMap<String, Packed>,
    vectors: BTreeMap<String, Vec<f32>>,
    convolutions: BTreeMap<String, Vec<f32>>,
    identity: ContentIdentity,
    report: LoadReport,
}
impl PayloadStore {
    pub fn plan(&self) -> &Plan {
        &self.plan
    }
    pub fn identity(&self) -> ContentIdentity {
        self.identity
    }
    pub fn report(&self) -> &LoadReport {
        &self.report
    }
    /// Borrowed original components; no expanded matrix or tensor clone.
    pub fn matrix(&self, name: &str) -> Result<Matrix<'_>, Error> {
        let p = self
            .matrices
            .get(name)
            .ok_or_else(|| invalid("Qwen3.5 matrix not found"))?;
        Matrix::from_le_bytes(p.layout, &p.codes, &p.scales, &p.biases).map_err(|e| invalid(e.0))
    }
    pub fn vector(&self, name: &str) -> Result<&[f32], Error> {
        self.vectors
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| invalid("Qwen3.5 vector not found"))
    }
    /// Original `[channels,4,1]` order: consecutive four taps per channel.
    pub fn convolution(&self, name: &str) -> Result<&[f32], Error> {
        self.convolutions
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| invalid("Qwen3.5 convolution not found"))
    }
}

fn mul(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_mul(b)
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| invalid("Qwen3.5 payload size overflow"))
}
fn budget(required: usize, limit: usize) -> Result<(), Error> {
    let limit = limit.min(MAX_LOAD_BYTES);
    if required > limit {
        return Err(Error::Budget { required, limit });
    }
    Ok(())
}
pub(super) struct Reservation {
    tensors: usize,
    overlap: usize,
    pub(super) total: usize,
}
#[cfg(test)]
fn reservation(plan: &Plan) -> Result<Reservation, Error> {
    reservation_for(&source_recipe(plan))
}
pub(super) fn reservation_for(plan: &SourceRecipe<'_>) -> Result<Reservation, Error> {
    let mut tensors = 0;
    let mut overlap = 0;
    for t in plan.common {
        let size = match (t.storage, t.shape) {
            (Storage::Affine(l), Shape::Matrix { .. }) => l.total_bytes(),
            (Storage::Bf16Vector, Shape::Vector(n)) => {
                overlap = overlap.max(mul(n, 2)?);
                mul(n, 4)?
            }
            _ => return Err(invalid("Qwen3.5 logical storage differs")),
        };
        tensors = add(tensors, size)?;
    }
    for c in plan.convolutions {
        let n = mul(c.channels, c.taps)?;
        overlap = overlap.max(mul(n, 2)?);
        tensors = add(tensors, mul(n, 4)?)?;
    }
    let total = add(add(add(tensors, overlap)?, STREAM_BYTES)?, METADATA_BYTES)?;
    Ok(Reservation {
        tensors,
        overlap,
        total,
    })
}
fn admit(peak: &mut usize, required: usize, limit: usize) -> Result<(), Error> {
    budget(required, limit)?;
    *peak = (*peak).max(required);
    Ok(())
}
fn with_metadata(tensors: usize, overlap: usize) -> Result<usize, Error> {
    add(add(add(tensors, overlap)?, STREAM_BYTES)?, METADATA_BYTES)
}
fn floats(raw: &[u8]) -> Result<Vec<f32>, Error> {
    if !raw.len().is_multiple_of(2) {
        return Err(invalid("Qwen3.5 BF16 byte count differs"));
    }
    let mut result = Vec::new();
    result
        .try_reserve_exact(raw.len() / 2)
        .map_err(|_| Error::Allocation)?;
    for bytes in raw.chunks_exact(2) {
        let v = bf16::from_le_bytes(bytes.try_into().unwrap()).to_f32();
        if !v.is_finite() {
            return Err(invalid("nonfinite Qwen3.5 BF16 value"));
        }
        result.push(v);
    }
    Ok(result)
}

/// Consume exactly one pinned source stream, without accepting caller-selected
/// payload hashes or layouts. Limits cover cold storage/temporary ownership,
/// not sessions, tokenizer, Metal or process RSS. Nothing is published on error.
pub fn load<R: Read>(
    config: &[u8],
    mut reader: R,
    options: LoadOptions,
) -> Result<PayloadStore, Error> {
    metadata_admission(options.memory_limit_bytes)?;
    if options.modality != ModalityPolicy::TextOnly || config.len() != CONFIG_BYTES {
        return Err(invalid("pinned Qwen3.5 config/text-only request required"));
    }
    let (header, sha) = source_header(&mut reader, HEADER_BYTES)?;
    let plan = inspect(config, &header, CONTAINER_BYTES, options.modality)?;
    let identity = ContentIdentity::pending(config, &header, CONTAINER_BYTES);
    load_admitted(
        &mut reader,
        plan,
        sha,
        identity,
        (8 + HEADER_BYTES) as u64,
        CONTAINER_SHA256,
        options.memory_limit_bytes,
    )
}

pub(super) fn metadata_admission(limit: usize) -> Result<(), Error> {
    budget(with_metadata(0, 0)?, limit)
}
pub(super) fn source_header<R: Read>(
    reader: &mut R,
    expected: usize,
) -> Result<(Vec<u8>, Sha256), Error> {
    let mut sha = Sha256::new();
    let mut length = [0; 8];
    read_exact(reader, &mut length, &mut sha)?;
    if u64::from_le_bytes(length) != expected as u64 {
        return Err(invalid("pinned Qwen3.5 header length differs"));
    }
    let mut header = buffer(expected)?;
    read_exact(reader, &mut header, &mut sha)?;
    Ok((header, sha))
}

fn take(physical: &mut BTreeMap<String, Vec<u8>>, name: &str) -> Result<Vec<u8>, Error> {
    physical
        .remove(name)
        .ok_or_else(|| invalid("Qwen3.5 physical tensor missing"))
}
fn load_admitted<R: Read>(
    reader: &mut R,
    plan: Plan,
    sha: Sha256,
    identity: ContentIdentity,
    consumed: u64,
    expected: &str,
    limit: usize,
) -> Result<PayloadStore, Error> {
    let Components {
        matrices,
        vectors,
        convolutions,
        identity,
        report,
    } = read_components(
        reader,
        source_recipe(&plan),
        sha,
        identity,
        consumed,
        expected,
        limit,
    )?;
    Ok(PayloadStore {
        plan,
        matrices,
        vectors,
        convolutions,
        identity,
        report,
    })
}

pub(super) fn read_components<R: Read>(
    reader: &mut R,
    plan: SourceRecipe<'_>,
    mut sha: Sha256,
    mut identity: ContentIdentity,
    mut consumed: u64,
    expected: &str,
    limit: usize,
) -> Result<Components, Error> {
    let reserve = reservation_for(&plan)?;
    budget(reserve.total, limit)?;
    let mut admitted = reserve.total;
    let mut spans: Vec<_> = plan.physical.iter().collect();
    spans.sort_unstable_by_key(|(_, t)| t.range.start);
    let mut physical = BTreeMap::new();
    let mut live = 0;
    let mut cursor = 0;
    for (name, t) in spans {
        if t.range.start != cursor {
            return Err(invalid("Qwen3.5 payload spans differ"));
        }
        let mut raw = buffer(t.range.len())?;
        live = add(live, raw.capacity())?;
        admit(&mut admitted, with_metadata(live, reserve.overlap)?, limit)?;
        for chunk in raw.chunks_mut(STREAM_BYTES) {
            read_exact(reader, chunk, &mut sha)?;
            consumed = consumed
                .checked_add(chunk.len() as u64)
                .ok_or_else(|| invalid("Qwen3.5 consumed size overflow"))?;
        }
        physical.insert(name.clone(), raw);
        cursor = t.range.end;
    }
    if cursor != plan.bytes.total || consumed != identity.bytes {
        return Err(invalid("Qwen3.5 consumed extent differs"));
    }
    let mut tail = [0];
    if reader
        .read(&mut tail)
        .map_err(|e| Error::Invalid(format!("Qwen3.5 payload EOF: {e}")))?
        != 0
    {
        return Err(invalid("Qwen3.5 payload trailing bytes"));
    }
    let digest = sha.finalize();
    if format!("{digest:x}") != expected {
        return Err(invalid("Qwen3.5 complete source SHA256 differs"));
    }
    identity.container = digest.into();
    let mut matrices = BTreeMap::new();
    let mut vectors = BTreeMap::new();
    let mut convolutions = BTreeMap::new();
    let mut retained = 0;
    for t in plan.common {
        match t.storage {
            Storage::Affine(layout) => {
                let base = t
                    .name
                    .strip_suffix(".weight")
                    .ok_or_else(|| invalid("Qwen3.5 affine name differs"))?;
                let packed = Packed {
                    layout,
                    codes: take(&mut physical, &t.name)?,
                    scales: take(&mut physical, &format!("{base}.scales"))?,
                    biases: take(&mut physical, &format!("{base}.biases"))?,
                };
                validate_q4_bf16(layout, &packed.codes, &packed.scales, &packed.biases)?;
                for n in [
                    packed.codes.capacity(),
                    packed.scales.capacity(),
                    packed.biases.capacity(),
                ] {
                    retained = add(retained, n)?;
                }
                matrices.insert(t.name.clone(), packed);
            }
            Storage::Bf16Vector => {
                let v = convert(&mut physical, &t.name, &mut live, &mut admitted, limit)?;
                retained = add(retained, mul(v.capacity(), 4)?)?;
                vectors.insert(t.name.clone(), v);
            }
        }
    }
    for c in plan.convolutions {
        let v = convert(&mut physical, &c.name, &mut live, &mut admitted, limit)?;
        if v.len() != mul(c.channels, c.taps)? {
            return Err(invalid("Qwen3.5 convolution length differs"));
        }
        retained = add(retained, mul(v.capacity(), 4)?)?;
        convolutions.insert(c.name.clone(), v);
    }
    if !physical.is_empty() || live != retained || retained < reserve.tensors {
        return Err(invalid("Qwen3.5 retained ownership differs"));
    }
    admit(
        &mut admitted,
        with_metadata(retained, reserve.overlap)?,
        limit,
    )?;
    let report = LoadReport {
        source_payload_bytes: plan.bytes.total,
        packed_matrices: matrices.len(),
        fp32_vectors: vectors.len(),
        fp32_convolutions: convolutions.len(),
        retained_tensor_capacity_bytes: retained,
        metadata_reservation_bytes: METADATA_BYTES,
        admitted_load_bytes: admitted,
    };
    Ok(Components {
        matrices,
        vectors,
        convolutions,
        identity,
        report,
    })
}
fn convert(
    physical: &mut BTreeMap<String, Vec<u8>>,
    name: &str,
    live: &mut usize,
    peak: &mut usize,
    limit: usize,
) -> Result<Vec<f32>, Error> {
    let raw = take(physical, name)?;
    admit(
        peak,
        with_metadata(add(*live, mul(raw.len() / 2, 4)?)?, 0)?,
        limit,
    )?;
    let result = floats(&raw)?;
    let capacity = mul(result.capacity(), 4)?;
    admit(peak, with_metadata(add(*live, capacity)?, 0)?, limit)?;
    *live = add(
        live.checked_sub(raw.capacity())
            .ok_or_else(|| invalid("Qwen3.5 raw capacity underflow"))?,
        capacity,
    )?;
    Ok(result)
}
