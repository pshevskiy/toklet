//! Original-byte storage for the strictly inspected LFM2-350M layout.
//!
//! This module resolves direct BF16 gains and rank-three convolution taps. It
//! neither constructs a numerical backend nor changes the tiny scalar profile.
use super::{LayerRecipe, Plan, Shape, Storage, inspect, invalid};
use crate::{
    mlx::{
        affine::Matrix,
        payload::{buffer, read_exact, validate_q4_bf16},
    },
    model::{
        Config,
        lfm2::{AttentionWeights, ConvWeights, Kind, LayerWeights, Weights},
    },
    weights::Error,
};
use half::bf16;
use sha2::{Digest, Sha256};
use std::{io::Read, mem::size_of};

#[cfg(test)]
mod tests;
const MAX_HEADER_BYTES: usize = 1024 * 1024;
const STREAM_BYTES: usize = 64 * 1024;
/// Conservative allowance for caller config/index, parsed JSON,
/// Plan/maps/names and loader objects. Header capacity is added separately
/// during loading. This allowance is not measured process memory.
const METADATA_RESERVATION: usize = 64 * 1024 * 1024;

/// Expected identities come from independently authenticated bytes. A successful
/// store records the identities actually consumed; this is not a provenance or
/// numerical-execution capability and it does not reopen filesystem paths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContentIdentity {
    pub config_sha256: [u8; 32],
    pub index_sha256: [u8; 32],
    pub header_sha256: [u8; 32],
    pub container_sha256: [u8; 32],
    pub container_bytes: u64,
}
#[derive(Clone, Copy)]
pub struct LoadOptions {
    pub identity: ContentIdentity,
    /// Loader numerical storage plus the explicit metadata allowance. Reader
    /// owners and application/tokenizer state outside metadata are caller-owned.
    pub memory_limit_bytes: usize,
    /// Other caller owners, including a full container if `reader` is in-memory.
    /// A File reader does not retain a second model-sized CPU allocation.
    pub external_resident_bytes: usize,
}
#[derive(Clone, Copy, Debug)]
pub struct LoadReport {
    /// Complete authenticated file length, including 8-byte prefix and header.
    pub container_bytes: u64,
    pub header_bytes: usize,
    pub header_capacity_bytes: usize,
    pub external_resident_bytes: usize,
    pub payload_bytes: usize,
    pub payload_capacity_bytes: usize,
    pub packed_matrices: usize,
    pub direct_bf16_norms: usize,
    pub rank3_bf16_convolutions: usize,
    pub metadata_reservation_bytes: usize,
    /// Construction bound including actual payload/header capacities and caller
    /// external residency. Header is dropped before returning. No expanded
    /// matrix, second payload, typed gain/tap or session is retained by the store.
    pub admitted_load_bytes: usize,
}

/// Immutable original payload plus cold descriptors. Matrix views borrow its
/// exact original ranges. Keep this store alive until a borrowing upload returns;
/// a later copying upload can then release this original source owner.
pub struct PayloadStore {
    plan: Plan,
    payload: Vec<u8>,
    report: LoadReport,
    identity: ContentIdentity,
}
/// Concrete once-resolved data for a later bounded real constructor. Only
/// gain/tap vectors and small descriptor vectors are owned; all matrices borrow the
/// original store. Dropping this input frees temporary typed BF16 gains/taps.
pub struct ModelInput<'a> {
    pub config: Config,
    pub weights: Weights<'a>,
    pub source_resident_bytes: usize,
}
impl PayloadStore {
    pub fn plan(&self) -> &Plan {
        &self.plan
    }
    pub fn report(&self) -> &LoadReport {
        &self.report
    }
    pub fn content_identity(&self) -> &ContentIdentity {
        &self.identity
    }

    fn bytes(&self, name: &str) -> Result<&[u8], Error> {
        let tensor = self
            .plan
            .physical_tensors()
            .get(name)
            .ok_or_else(|| invalid("LFM2 physical tensor missing"))?;
        self.payload
            .get(tensor.range.clone())
            .ok_or_else(|| invalid("LFM2 physical tensor range"))
    }
    /// Cold name resolution; retain the returned concrete view outside loops.
    pub fn matrix(&self, name: &str) -> Result<Matrix<'_>, Error> {
        let tensor = self
            .plan
            .logical_tensors()
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| invalid("LFM2 matrix missing"))?;
        let Storage::Affine(layout) = tensor.storage else {
            return Err(invalid("LFM2 requested tensor is not affine"));
        };
        let base = name
            .strip_suffix(".weight")
            .ok_or_else(|| invalid("LFM2 matrix suffix"))?;
        Matrix::from_le_bytes(
            layout,
            self.bytes(name)?,
            self.bytes(&format!("{base}.scales"))?,
            self.bytes(&format!("{base}.biases"))?,
        )
        .map_err(|e| invalid(e.0))
    }
    /// Original direct BF16 gain words, with no +1 or activation cast.
    pub fn direct_gain(&self, name: &str) -> Result<Vec<bf16>, Error> {
        let tensor = self
            .plan
            .logical_tensors()
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| invalid("LFM2 norm missing"))?;
        let (Shape::Vector(count), Storage::Bf16Vector) = (tensor.shape, tensor.storage) else {
            return Err(invalid("LFM2 requested tensor is not a BF16 norm"));
        };
        decode_bf16(self.bytes(name)?, count)
    }
    /// Original [channels,3,1] taps, oldest/previous/current per channel.
    /// The explicit rank-three inventory is never treated as an affine matrix.
    pub fn convolution_coefficients(&self, name: &str) -> Result<Vec<bf16>, Error> {
        let tensor = self
            .plan
            .convolutions()
            .iter()
            .find(|t| t.coefficient_name == name)
            .ok_or_else(|| invalid("LFM2 convolution missing"))?;
        if tensor.taps != 3 || tensor.input_group_channels != 1 {
            return Err(invalid("LFM2 convolution layout"));
        }
        decode_bf16(self.bytes(name)?, multiply(tensor.channels, 3)?)
    }

    /// Cold resolution for the selected original artifact, capacity 1..=512.
    /// This is data preparation only: neither existing tiny backend accepts
    /// these real dimensions, and source-declared128000 is not admitted here.
    /// `external_resident_bytes` must include all caller owners still alive,
    /// including any whole file reader/mapping; it replaces the load-time value.
    pub fn prepare_direct_weights(
        &self,
        capacity: usize,
        external_resident_bytes: usize,
        memory_limit_bytes: usize,
    ) -> Result<ModelInput<'_>, Error> {
        let config = execution_config(capacity)?;
        if self.plan.logical_tensors().len() != 138
            || self.plan.convolutions().len() != 10
            || self.plan.physical_tensors().len() != 334
        {
            return Err(invalid("LFM2 preparation requires inspected350M plan"));
        }
        let source = add(
            add(self.payload.capacity(), METADATA_RESERVATION)?,
            external_resident_bytes,
        )?;
        check_budget(
            add(source, temporary_bytes(&self.plan, 16, 6, 10)?)?,
            memory_limit_bytes,
        )?;
        let mut layers = reserved(16)?;
        let mut attention = reserved(6)?;
        let mut convolution = reserved(10)?;
        for (index, recipe) in self.plan.layers().iter().enumerate() {
            let base = format!("model.layers.{index}.");
            let matrix = |suffix: &str| self.matrix(&format!("{base}{suffix}.weight"));
            let gain = |suffix: &str| self.direct_gain(&format!("{base}{suffix}.weight"));
            let kind = match *recipe {
                LayerRecipe::FullAttention(plan) => {
                    if plan.kv_owner != attention.len() {
                        return Err(invalid("LFM2 attention owner order"));
                    }
                    attention.push(AttentionWeights {
                        q: matrix("self_attn.q_proj")?,
                        k: matrix("self_attn.k_proj")?,
                        v: matrix("self_attn.v_proj")?,
                        o: matrix("self_attn.out_proj")?,
                        q_gamma: gain("self_attn.q_layernorm")?,
                        k_gamma: gain("self_attn.k_layernorm")?,
                    });
                    Kind::Attention(plan.kv_owner)
                }
                LayerRecipe::ShortConv3 { history_owner } => {
                    if history_owner != convolution.len() {
                        return Err(invalid("LFM2 history owner order"));
                    }
                    convolution.push(ConvWeights {
                        bcx: matrix("conv.in_proj")?,
                        output: matrix("conv.out_proj")?,
                        coefficients: self
                            .convolution_coefficients(&format!("{base}conv.conv.weight"))?,
                    });
                    Kind::Convolution(history_owner)
                }
            };
            layers.push(LayerWeights {
                kind,
                gate: matrix("feed_forward.w1")?,
                up: matrix("feed_forward.w3")?,
                down: matrix("feed_forward.w2")?,
                operator_gamma: gain("operator_norm")?,
                ffn_gamma: gain("ffn_norm")?,
            });
        }
        if attention.len() != 6 || convolution.len() != 10 {
            return Err(invalid("LFM2 state owner counts"));
        }
        let weights = Weights {
            embedding: self.matrix("model.embed_tokens.weight")?,
            layers,
            attention,
            convolution,
            final_gamma: self.direct_gain("model.embedding_norm.weight")?,
        };
        let source_resident_bytes = add(source, input_capacity(&weights)?)?;
        check_budget(source_resident_bytes, memory_limit_bytes)?;
        Ok(ModelInput {
            config,
            weights,
            source_resident_bytes,
        })
    }
}

fn execution_config(capacity: usize) -> Result<Config, Error> {
    if !(1..=512).contains(&capacity) {
        return Err(invalid("LFM2 requested capacity outside1..=512"));
    }
    Ok(Config {
        vocab_size: 65536,
        hidden_size: 1024,
        intermediate_size: 4608,
        num_layers: 16,
        query_heads: 16,
        kv_heads: 8,
        head_dim: 64,
        max_positions: capacity,
        rms_norm_eps: 1e-5,
        rope_theta: 1_000_000.0,
    })
}
fn reserved<T>(count: usize) -> Result<Vec<T>, Error> {
    let mut out = Vec::new();
    out.try_reserve_exact(count)
        .map_err(|_| Error::Allocation)?;
    Ok(out)
}

fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b)
        .ok_or_else(|| invalid("LFM2 payload size overflow"))
}
fn multiply(a: usize, b: usize) -> Result<usize, Error> {
    let value = a
        .checked_mul(b)
        .ok_or_else(|| invalid("LFM2 payload size overflow"))?;
    if value > isize::MAX as usize {
        return Err(invalid("LFM2 payload allocation bound"));
    }
    Ok(value)
}
fn check_budget(required: usize, limit: usize) -> Result<(), Error> {
    if required > limit {
        Err(Error::Budget { required, limit })
    } else {
        Ok(())
    }
}
fn descriptor_bytes(layers: usize, attention: usize, convolution: usize) -> Result<usize, Error> {
    add(
        add(
            multiply(layers, size_of::<LayerWeights<'_>>())?,
            multiply(attention, size_of::<AttentionWeights<'_>>())?,
        )?,
        multiply(convolution, size_of::<ConvWeights<'_>>())?,
    )
}
fn temporary_bytes(
    plan: &Plan,
    layers: usize,
    attention: usize,
    convolution: usize,
) -> Result<usize, Error> {
    let mut bytes = descriptor_bytes(layers, attention, convolution)?;
    for tensor in plan.logical_tensors() {
        if let (Shape::Vector(count), Storage::Bf16Vector) = (tensor.shape, tensor.storage) {
            bytes = add(bytes, multiply(count, size_of::<bf16>())?)?;
        }
    }
    for tensor in plan.convolutions() {
        bytes = add(
            bytes,
            multiply(multiply(tensor.channels, 3)?, size_of::<bf16>())?,
        )?;
    }
    Ok(bytes)
}
fn input_capacity(weights: &Weights<'_>) -> Result<usize, Error> {
    let mut bytes = descriptor_bytes(
        weights.layers.capacity(),
        weights.attention.capacity(),
        weights.convolution.capacity(),
    )?;
    bytes = add(
        bytes,
        multiply(weights.final_gamma.capacity(), size_of::<bf16>())?,
    )?;
    for layer in &weights.layers {
        for vector in [&layer.operator_gamma, &layer.ffn_gamma] {
            bytes = add(bytes, multiply(vector.capacity(), size_of::<bf16>())?)?;
        }
    }
    for layer in &weights.attention {
        for vector in [&layer.q_gamma, &layer.k_gamma] {
            bytes = add(bytes, multiply(vector.capacity(), size_of::<bf16>())?)?;
        }
    }
    for layer in &weights.convolution {
        bytes = add(
            bytes,
            multiply(layer.coefficients.capacity(), size_of::<bf16>())?,
        )?;
    }
    Ok(bytes)
}
fn decode_bf16(raw: &[u8], count: usize) -> Result<Vec<bf16>, Error> {
    if raw.len() != multiply(count, 2)? {
        return Err(invalid("LFM2 BF16 norm length"));
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| Error::Allocation)?;
    for pair in raw.chunks_exact(2) {
        let value = bf16::from_le_bytes(pair.try_into().unwrap());
        if !value.is_finite() {
            return Err(invalid("nonfinite LFM2 BF16 norm"));
        }
        values.push(value);
    }
    Ok(values)
}
fn validate_values(store: &PayloadStore) -> Result<(), Error> {
    for tensor in store.plan.logical_tensors() {
        match (tensor.shape, tensor.storage) {
            (Shape::Matrix { .. }, Storage::Affine(layout)) => {
                let base = tensor
                    .name
                    .strip_suffix(".weight")
                    .ok_or_else(|| invalid("LFM2 matrix suffix"))?;
                // Selected NativeHalf source reconstructs finite FP32(q*scale)
                // then adds bias. A finite product is exact for Q4/BF16, so it
                // equals FMA reconstruction; cancellation may not rescue an
                // overflowing product in this explicitly supported domain.
                for pair in store.bytes(&format!("{base}.scales"))?.chunks_exact(2) {
                    if !(bf16::from_le_bytes(pair.try_into().unwrap()).to_f32() * 15.0).is_finite()
                    {
                        return Err(invalid("LFM2 affine product overflow"));
                    }
                }
                validate_q4_bf16(
                    layout,
                    store.bytes(&tensor.name)?,
                    store.bytes(&format!("{base}.scales"))?,
                    store.bytes(&format!("{base}.biases"))?,
                )?;
            }
            (Shape::Vector(count), Storage::Bf16Vector) => {
                let raw = store.bytes(&tensor.name)?;
                if raw.len() != multiply(count, 2)? {
                    return Err(invalid("LFM2 norm length"));
                }
                for pair in raw.chunks_exact(2) {
                    if !bf16::from_le_bytes(pair.try_into().unwrap()).is_finite() {
                        return Err(invalid("nonfinite LFM2 BF16 norm"));
                    }
                }
            }
            _ => return Err(invalid("LFM2 logical tensor representation")),
        }
    }
    for tensor in store.plan.convolutions() {
        let raw = store.bytes(&tensor.coefficient_name)?;
        if raw.len() != multiply(multiply(tensor.channels, 3)?, 2)? {
            return Err(invalid("LFM2 rank-three coefficient length"));
        }
        for pair in raw.chunks_exact(2) {
            if !bf16::from_le_bytes(pair.try_into().unwrap()).is_finite() {
                return Err(invalid("nonfinite LFM2 BF16 convolution"));
            }
        }
    }
    Ok(())
}
/// Stream and hash every byte; retain only original payload bytes. Metadata
/// identities and the strict family plan pass before allocating the payload.
/// Complete byte count, EOF, digest and value checks precede returning a store.
/// Ordinary callers use a File reader to avoid a second full CPU container.
pub fn load<R: Read>(
    config: &[u8],
    index: &[u8],
    mut reader: R,
    options: LoadOptions,
) -> Result<PayloadStore, Error> {
    check_budget(
        add(
            add(METADATA_RESERVATION, STREAM_BYTES)?,
            options.external_resident_bytes,
        )?,
        options.memory_limit_bytes,
    )?;
    if config.len() > MAX_HEADER_BYTES || index.len() > MAX_HEADER_BYTES {
        return Err(invalid("LFM2 payload metadata bound"));
    }
    let identity = options.identity;
    if <[u8; 32]>::from(Sha256::digest(config)) != identity.config_sha256
        || <[u8; 32]>::from(Sha256::digest(index)) != identity.index_sha256
    {
        return Err(invalid("LFM2 config/index identity differs"));
    }
    let mut sha = Sha256::new();
    let mut prefix = [0; 8];
    read_exact(&mut reader, &mut prefix, &mut sha)?;
    let size =
        usize::try_from(u64::from_le_bytes(prefix)).map_err(|_| invalid("LFM2 header overflow"))?;
    if size == 0 || size > MAX_HEADER_BYTES {
        return Err(invalid("LFM2 bounded header required"));
    }
    let consumed = add(size, 8)? as u64;
    if consumed > identity.container_bytes {
        return Err(invalid("LFM2 truncated container"));
    }
    check_budget(
        add(
            add(add(METADATA_RESERVATION, STREAM_BYTES)?, size)?,
            options.external_resident_bytes,
        )?,
        options.memory_limit_bytes,
    )?;
    let mut header = buffer(size)?;
    check_budget(
        add(
            add(add(METADATA_RESERVATION, STREAM_BYTES)?, header.capacity())?,
            options.external_resident_bytes,
        )?,
        options.memory_limit_bytes,
    )?;
    read_exact(&mut reader, &mut header, &mut sha)?;
    if <[u8; 32]>::from(Sha256::digest(&header)) != identity.header_sha256 {
        return Err(invalid("LFM2 header identity differs"));
    }
    let plan = inspect(config, index, &header, identity.container_bytes)?;
    load_admitted(&mut reader, plan, sha, consumed, header.capacity(), options)
}
fn load_admitted<R: Read>(
    reader: &mut R,
    plan: Plan,
    mut sha: Sha256,
    consumed: u64,
    header_capacity: usize,
    options: LoadOptions,
) -> Result<PayloadStore, Error> {
    let header_bytes = usize::try_from(
        consumed
            .checked_sub(8)
            .ok_or_else(|| invalid("LFM2 header count"))?,
    )
    .map_err(|_| invalid("LFM2 header count"))?;
    let overhead = add(
        add(add(METADATA_RESERVATION, STREAM_BYTES)?, header_capacity)?,
        options.external_resident_bytes,
    )?;
    let required = add(plan.payload_bytes(), overhead)?;
    check_budget(required, options.memory_limit_bytes)?;
    let mut payload = buffer(plan.payload_bytes())?;
    let admitted = add(payload.capacity(), overhead)?;
    check_budget(admitted, options.memory_limit_bytes)?;
    let mut consumed = consumed;
    for chunk in payload.chunks_mut(STREAM_BYTES) {
        read_exact(reader, chunk, &mut sha)?;
        consumed = consumed
            .checked_add(chunk.len() as u64)
            .ok_or_else(|| invalid("LFM2 byte count overflow"))?;
    }
    let mut tail = [0];
    if reader
        .read(&mut tail)
        .map_err(|e| invalid(format!("LFM2 payload EOF: {e}")))?
        != 0
    {
        return Err(invalid("LFM2 payload trailing bytes"));
    }
    let digest: [u8; 32] = sha.finalize().into();
    if consumed != options.identity.container_bytes || digest != options.identity.container_sha256 {
        return Err(invalid("LFM2 complete container identity differs"));
    }
    let packed_matrices = plan
        .logical_tensors()
        .iter()
        .filter(|t| matches!(t.storage, Storage::Affine(_)))
        .count();
    let direct_bf16_norms = plan.logical_tensors().len() - packed_matrices;
    let rank3_bf16_convolutions = plan.convolutions().len();
    let store = PayloadStore {
        report: LoadReport {
            container_bytes: options.identity.container_bytes,
            header_bytes,
            header_capacity_bytes: header_capacity,
            external_resident_bytes: options.external_resident_bytes,
            payload_bytes: payload.len(),
            payload_capacity_bytes: payload.capacity(),
            packed_matrices,
            direct_bf16_norms,
            rank3_bf16_convolutions,
            metadata_reservation_bytes: METADATA_RESERVATION,
            admitted_load_bytes: admitted,
        },
        plan,
        payload,
        identity: options.identity,
    };
    validate_values(&store)?;
    Ok(store)
}
