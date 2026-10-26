//! Complete immutable E2B text weights. Storage admission is not graph execution.
use super::*;
use crate::mlx::gemma4::ModalityPolicy;
use std::io::{Read, Seek};

mod bindings;
mod plan;
mod source;
#[cfg(test)]
mod tests;
use bindings::Bindings;
use plan::{Plan, add, admit, multiply, subtract, vector_bytes};

const TILE: usize = 16384;
// Conservative reserve for bounded parsed metadata/maps/descriptors, not an
// observation of allocator totals or arbitrary caller-owned reader storage.
const COLD_POLICY: usize = 64 * 1024 * 1024;
const HASH_STACK_POLICY: usize = 128 * 1024;
const OVERFLOW: &str = "Gemma resident storage size overflow";
fn policy_overhead() -> Result<usize> {
    add(
        add(CONTEXT_ALLOWANCE_BYTES, COLD_POLICY)?,
        HASH_STACK_POLICY,
    )
}
#[derive(Clone, Copy)]
pub struct LoadOptions {
    pub container_bytes: u64,
    pub container_sha256: [u8; 32],
    /// Remaining budget after other models/sessions/app state, caller-owned
    /// config/index buffers and arbitrary Read+Seek adapter/backing allocations.
    /// Includes this load's Context/cold policies, weights and source overlap.
    pub available_bytes: usize,
    pub modality: ModalityPolicy,
}
/// Historical whole-container first-pass anchor plus matched selected text
/// consumption. No current-path or excluded-second-pass assertion is implied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Identity {
    config_sha256: [u8; 32],
    index_sha256: [u8; 32],
    header_sha256: [u8; 32],
    container_sha256: [u8; 32],
    container_bytes: u64,
    text_components: usize,
}
impl Identity {
    pub fn config_sha256(&self) -> &[u8; 32] {
        &self.config_sha256
    }
    pub fn index_sha256(&self) -> &[u8; 32] {
        &self.index_sha256
    }
    pub fn header_sha256(&self) -> &[u8; 32] {
        &self.header_sha256
    }
    pub fn container_sha256(&self) -> &[u8; 32] {
        &self.container_sha256
    }
    pub fn container_bytes(&self) -> u64 {
        self.container_bytes
    }
    pub fn text_components(&self) -> usize {
        self.text_components
    }
}
#[derive(Clone, Copy, Debug)]
pub struct LoadReport {
    /// Requested retained ownership, including actual Vec slots and GPU guards.
    pub resident_requested_bytes: usize,
    /// Conservative final ownership plus maximum source scratch and policies;
    /// this is not measured physical residency or allocator totals.
    pub admitted_load_policy_bytes: usize,
    pub maximum_source_capacity_bytes: usize,
    pub matrix_owner_slots: usize,
    pub vector_owner_slots: usize,
    pub matrices: usize,
    pub vectors: usize,
}
struct ResidentVector {
    buffer: Buffer<u32>,
    values: usize,
}
struct Storage {
    matrices: Vec<Weights>,
    vectors: Vec<ResidentVector>,
}
/// Independently owned immutable original packed matrices and resident FP32
/// learned vectors, with cold validated role indices. No Context borrow, full
/// CPU source copy, activation scratch, KV, tokenizer or execution capability.
pub struct E2bWeights {
    storage: Storage,
    bindings: Bindings,
    identity: Identity,
    report: LoadReport,
}
impl E2bWeights {
    pub fn load(
        context: &Context,
        config: &[u8],
        index: &[u8],
        mut reader: impl Read + Seek,
        options: LoadOptions,
    ) -> Result<Self> {
        admit(policy_overhead()?, options.available_bytes)?;
        let mut prepared = source::Prepared::read(config, index, &mut reader, &options)?;
        let bindings = Bindings::new(&prepared.cold)?;
        let (storage, report) = load_storage(context, &mut prepared.inputs, &mut reader, &options)?;
        let identity = prepared.inputs.finish(&options)?;
        Ok(Self {
            storage,
            bindings,
            identity,
            report,
        })
    }
    pub fn identity(&self) -> &Identity {
        &self.identity
    }
    pub fn report(&self) -> &LoadReport {
        &self.report
    }
    pub fn matrix_count(&self) -> usize {
        self.storage.matrices.len()
    }
    pub fn vector_count(&self) -> usize {
        self.storage.vectors.len()
    }
    /// Cold admitted geometry, not a numerical execution capability.
    pub fn layer_geometry(&self, index: usize) -> Option<crate::mlx::gemma4::LayerPlan> {
        self.bindings.layer_geometry(index)
    }
}
fn owner_vec<T>(count: usize) -> Result<Vec<T>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| Error::Native("Gemma resident owner reservation failed".into()))?;
    Ok(result)
}
fn load_storage(
    context: &Context,
    inputs: &mut source::Inputs,
    reader: &mut (impl Read + Seek),
    options: &LoadOptions,
) -> Result<(Storage, LoadReport)> {
    let minimum = Plan::new(inputs, inputs.matrices.len(), inputs.vectors.len())?;
    inputs.check_metadata_capacities()?;
    admit(
        minimum.working(0, 0, minimum.source)?,
        options.available_bytes,
    )?;
    minimum.validate_device(inputs, context.gpu.device.maxBufferLength())?;
    let mut matrices = owner_vec(inputs.matrices.len())?;
    let mut vectors = owner_vec(inputs.vectors.len())?;
    let plan = Plan::new(inputs, matrices.capacity(), vectors.capacity())?;
    admit(plan.working(0, 0, plan.source)?, options.available_bytes)?;
    inputs.authenticate(reader, options)?;
    let mut actual_payload = 0;
    let mut completed_minimum = 0;
    let mut maximum_source = plan.source;
    for index in 0..inputs.matrices.len() {
        let layout = inputs.matrices[index].layout;
        let base = add(plan.policy_base, actual_payload)?;
        let remaining = subtract(options.available_bytes, base)?;
        // Every inline owner slot was reserved and charged in base already.
        let local_budget = add(remaining, size_of::<Weights>())?;
        let mut source = inputs.matrix_read(index)?;
        let (weights, stats) = context.weights_from_fill_reported(
            layout,
            TILE,
            local_budget,
            |first, child, codes, scales, biases| {
                source.fill(reader, first, child, [codes, scales, biases])
            },
        )?;
        source.finish(inputs)?;
        let actual_peak = subtract(add(base, stats.working_bytes)?, size_of::<Weights>())?;
        admit(actual_peak, options.available_bytes)?;
        maximum_source = maximum_source.max(stats.source_capacity_bytes);
        actual_payload = add(
            actual_payload,
            subtract(weights.owned_bytes(), size_of::<Weights>())?,
        )?;
        completed_minimum = add(completed_minimum, plan.matrix_payloads[index])?;
        admit(
            plan.working(actual_payload, completed_minimum, maximum_source)?,
            options.available_bytes,
        )?;
        matrices.push(weights);
    }
    // All three-array matrix source owners have dropped before vector scratch.
    let mut raw = owner_vec::<u8>(plan.maximum_vector_raw)?;
    maximum_source = maximum_source.max(add(raw.capacity(), size_of::<Vec<u8>>())?);
    admit(
        plan.working(actual_payload, completed_minimum, maximum_source)?,
        options.available_bytes,
    )?;
    raw.resize(plan.maximum_vector_raw, 0);
    for index in 0..inputs.vectors.len() {
        let values = inputs.vectors[index].values;
        let bytes = multiply(values, 2)?;
        let active = &mut raw[..bytes];
        inputs.vector_read(reader, index, active)?;
        if active
            .chunks_exact(2)
            .any(|b| !f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16).is_finite())
        {
            return Err(Error::Options("Gemma resident vector nonfinite"));
        }
        let resident = vector_bytes(values)?;
        let next_payload = add(actual_payload, resident)?;
        let next_minimum = add(completed_minimum, resident)?;
        admit(
            plan.working(next_payload, next_minimum, maximum_source)?,
            options.available_bytes,
        )?;
        let mut buffer = Buffer::new(&context.gpu, add(values, GUARD)?)?;
        buffer.with_mut(|words| {
            words.fill(SENTINEL);
            for (word, pair) in words.iter_mut().zip(active.chunks_exact(2)) {
                *word = u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16;
            }
        });
        vectors.push(ResidentVector { buffer, values });
        actual_payload = next_payload;
        completed_minimum = next_minimum;
    }
    if completed_minimum != plan.payload {
        return Err(Error::Options("Gemma resident payload incomplete"));
    }
    let storage = Storage { matrices, vectors };
    let actual_resident = storage.resident_bytes()?;
    if actual_resident != add(plan.base, actual_payload)? {
        return Err(Error::Options("Gemma resident actual accounting differs"));
    }
    let report = LoadReport {
        resident_requested_bytes: actual_resident,
        admitted_load_policy_bytes: plan.working(
            actual_payload,
            completed_minimum,
            maximum_source,
        )?,
        maximum_source_capacity_bytes: maximum_source,
        matrix_owner_slots: storage.matrices.capacity(),
        vector_owner_slots: storage.vectors.capacity(),
        matrices: storage.matrices.len(),
        vectors: storage.vectors.len(),
    };
    Ok((storage, report))
}
impl Storage {
    fn resident_bytes(&self) -> Result<usize> {
        let mut bytes = add(
            size_of::<E2bWeights>(),
            add(
                multiply(self.matrices.capacity(), size_of::<Weights>())?,
                multiply(self.vectors.capacity(), size_of::<ResidentVector>())?,
            )?,
        )?;
        for weights in &self.matrices {
            bytes = add(
                bytes,
                subtract(weights.owned_bytes(), size_of::<Weights>())?,
            )?;
        }
        for vector in &self.vectors {
            if vector.buffer.len != add(vector.values, GUARD)? {
                return Err(Error::Options("Gemma resident vector storage differs"));
            }
            bytes = add(bytes, multiply(vector.buffer.len, 4)?)?;
        }
        Ok(bytes)
    }
}
