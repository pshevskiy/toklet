//! Complete original-byte storage for the inspected Gemma3 270M layout.
//!
//! Byte/layout/value admission is separate from the explicitly requested raw
//! BF16 gamma interpretation and from numerical, text or application admission.
use super::{Plan, Shape, Storage, inspect, invalid};
use crate::{
    mlx::{
        affine::Matrix,
        payload::{buffer, read_exact, validate_q4_bf16},
    },
    model::{
        Config,
        gemma3::{Gamma, LayerSpec, LayerWeights, Weights},
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
/// Conservative allowance for caller config/index, bounded header, parsed JSON,
/// Plan/maps/names and loader objects. It is not measured process memory.
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
}
#[derive(Clone, Copy, Debug)]
pub struct LoadReport {
    pub payload_bytes: usize,
    pub payload_capacity_bytes: usize,
    pub packed_matrices: usize,
    pub raw_bf16_norms: usize,
    pub metadata_reservation_bytes: usize,
    /// Planned bound checked against actual payload Vec capacity. No expanded
    /// matrix, second payload, typed gamma or session is retained by the store.
    pub admitted_load_bytes: usize,
}

/// Immutable original payload plus cold descriptors. Matrix views borrow its
/// exact original ranges. Keep this store alive until a borrowing upload returns;
/// the resident Metal model retains no reference to it.
pub struct PayloadStore {
    plan: Plan,
    payload: Vec<u8>,
    report: LoadReport,
    identity: ContentIdentity,
}
/// Concrete once-resolved input to the existing raw-gamma constructor. Only
/// norm vectors and small descriptor vectors are owned; all matrices borrow the
/// original store. Dropping this input frees its temporary typed BF16 gammas.
pub struct ModelInput<'a> {
    pub config: Config,
    pub specs: Vec<LayerSpec>,
    pub local_window: usize,
    pub weights: Weights<Matrix<'a>>,
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
            .ok_or_else(|| invalid("Gemma3 physical tensor missing"))?;
        self.payload
            .get(tensor.range.clone())
            .ok_or_else(|| invalid("Gemma3 physical tensor range"))
    }
    /// Cold name resolution; retain the returned concrete view outside loops.
    pub fn matrix(&self, name: &str) -> Result<Matrix<'_>, Error> {
        let tensor = self
            .plan
            .logical_tensors()
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| invalid("Gemma3 matrix missing"))?;
        let Storage::Affine(layout) = tensor.storage else {
            return Err(invalid("Gemma3 requested tensor is not affine"));
        };
        let base = name
            .strip_suffix(".weight")
            .ok_or_else(|| invalid("Gemma3 matrix suffix"))?;
        Matrix::from_le_bytes(
            layout,
            self.bytes(name)?,
            self.bytes(&format!("{base}.scales"))?,
            self.bytes(&format!("{base}.biases"))?,
        )
        .map_err(|e| invalid(e.0))
    }
    /// Explicit semantic request: interpret original BF16 words as raw gamma,
    /// without folding one or promoting them to FP32 here. Dtype alone does not
    /// authorize this interpretation; the caller authenticates conversion lineage.
    pub fn raw_gamma(&self, name: &str) -> Result<Vec<bf16>, Error> {
        let tensor = self
            .plan
            .logical_tensors()
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| invalid("Gemma3 norm missing"))?;
        let (Shape::Vector(count), Storage::Bf16Vector) = (tensor.shape, tensor.storage) else {
            return Err(invalid("Gemma3 requested tensor is not a BF16 norm"));
        };
        decode_gamma(self.bytes(name)?, count)
    }

    /// Prepare the inspected 18-layer profile at requested capacity 1..=544.
    /// Physical window=min(512,capacity), equivalent throughout that admitted
    /// prefix. Source window 512/context 32768 remain metadata, not admission.
    /// `external_resident_bytes` includes other caller owners (for example an
    /// in-memory reader/container or tokenizer), beyond the metadata allowance.
    pub fn prepare_raw_gamma(
        &self,
        capacity: usize,
        external_resident_bytes: usize,
        memory_limit_bytes: usize,
    ) -> Result<ModelInput<'_>, Error> {
        let config = execution_config(capacity)?;
        if self.plan.logical_tensors().len() != 236 || self.plan.physical_tensors().len() != 490 {
            return Err(invalid(
                "Gemma3 model preparation requires the inspected 270M plan",
            ));
        }
        let source = add(
            add(self.payload.capacity(), METADATA_RESERVATION)?,
            external_resident_bytes,
        )?;
        let wanted = temporary_bytes(&self.plan, 18, 18)?;
        check_budget(add(source, wanted)?, memory_limit_bytes)?;
        let mut specs = Vec::new();
        specs.try_reserve_exact(18).map_err(|_| Error::Allocation)?;
        for layer in self.plan.layers() {
            specs.push(LayerSpec {
                kind: layer.kind,
                theta: layer.rope_theta,
            });
        }
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(18)
            .map_err(|_| Error::Allocation)?;
        for index in 0..18 {
            let base = format!("model.layers.{index}.");
            let matrix = |suffix: &str| self.matrix(&format!("{base}{suffix}.weight"));
            let gamma = |suffix: &str| self.raw_gamma(&format!("{base}{suffix}.weight"));
            layers.push(LayerWeights {
                q: matrix("self_attn.q_proj")?,
                k: matrix("self_attn.k_proj")?,
                v: matrix("self_attn.v_proj")?,
                o: matrix("self_attn.o_proj")?,
                gate: matrix("mlp.gate_proj")?,
                up: matrix("mlp.up_proj")?,
                down: matrix("mlp.down_proj")?,
                gamma: Gamma {
                    input: gamma("input_layernorm")?,
                    post_attention: gamma("post_attention_layernorm")?,
                    pre_ff: gamma("pre_feedforward_layernorm")?,
                    post_ff: gamma("post_feedforward_layernorm")?,
                    q: gamma("self_attn.q_norm")?,
                    k: gamma("self_attn.k_norm")?,
                },
            });
        }
        let weights = Weights {
            embedding: self.matrix("model.embed_tokens.weight")?,
            layers,
            final_gamma: self.raw_gamma("model.norm.weight")?,
        };
        let actual = input_capacity(&weights, specs.capacity())?;
        let source_resident_bytes = add(source, actual)?;
        check_budget(source_resident_bytes, memory_limit_bytes)?;
        Ok(ModelInput {
            config,
            specs,
            local_window: 512.min(capacity),
            weights,
            source_resident_bytes,
        })
    }
}
fn execution_config(capacity: usize) -> Result<Config, Error> {
    if !(1..=544).contains(&capacity) {
        return Err(invalid("Gemma3 requested capacity outside 1..=544"));
    }
    Ok(Config {
        vocab_size: 262144,
        hidden_size: 640,
        intermediate_size: 2048,
        num_layers: 18,
        query_heads: 4,
        kv_heads: 1,
        head_dim: 256,
        max_positions: capacity,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
    })
}
fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b)
        .ok_or_else(|| invalid("Gemma3 payload size overflow"))
}
fn multiply(a: usize, b: usize) -> Result<usize, Error> {
    let value = a
        .checked_mul(b)
        .ok_or_else(|| invalid("Gemma3 payload size overflow"))?;
    if value > isize::MAX as usize {
        return Err(invalid("Gemma3 payload allocation bound"));
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
fn temporary_bytes(
    plan: &Plan,
    layer_capacity: usize,
    spec_capacity: usize,
) -> Result<usize, Error> {
    let mut bytes = add(
        multiply(layer_capacity, size_of::<LayerWeights<Matrix<'_>>>())?,
        multiply(spec_capacity, size_of::<LayerSpec>())?,
    )?;
    for tensor in plan.logical_tensors() {
        if let (Shape::Vector(count), Storage::Bf16Vector) = (tensor.shape, tensor.storage) {
            bytes = add(bytes, multiply(count, size_of::<bf16>())?)?;
        }
    }
    Ok(bytes)
}
fn input_capacity(weights: &Weights<Matrix<'_>>, specs: usize) -> Result<usize, Error> {
    let mut bytes = add(
        multiply(
            weights.layers.capacity(),
            size_of::<LayerWeights<Matrix<'_>>>(),
        )?,
        multiply(specs, size_of::<LayerSpec>())?,
    )?;
    bytes = add(
        bytes,
        multiply(weights.final_gamma.capacity(), size_of::<bf16>())?,
    )?;
    for layer in &weights.layers {
        for vector in [
            &layer.gamma.input,
            &layer.gamma.post_attention,
            &layer.gamma.pre_ff,
            &layer.gamma.post_ff,
            &layer.gamma.q,
            &layer.gamma.k,
        ] {
            bytes = add(bytes, multiply(vector.capacity(), size_of::<bf16>())?)?;
        }
    }
    Ok(bytes)
}
fn decode_gamma(raw: &[u8], count: usize) -> Result<Vec<bf16>, Error> {
    if raw.len() != multiply(count, 2)? {
        return Err(invalid("Gemma3 BF16 norm length"));
    }
    let mut gamma = Vec::new();
    gamma
        .try_reserve_exact(count)
        .map_err(|_| Error::Allocation)?;
    for pair in raw.chunks_exact(2) {
        let value = bf16::from_le_bytes(pair.try_into().unwrap());
        if !value.is_finite() {
            return Err(invalid("nonfinite Gemma3 BF16 norm"));
        }
        gamma.push(value);
    }
    Ok(gamma)
}
fn validate_values(store: &PayloadStore) -> Result<(), Error> {
    for tensor in store.plan.logical_tensors() {
        match (tensor.shape, tensor.storage) {
            (Shape::Matrix { .. }, Storage::Affine(layout)) => {
                let base = tensor
                    .name
                    .strip_suffix(".weight")
                    .ok_or_else(|| invalid("Gemma3 matrix suffix"))?;
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
                    return Err(invalid("Gemma3 norm length"));
                }
                for pair in raw.chunks_exact(2) {
                    if !bf16::from_le_bytes(pair.try_into().unwrap()).is_finite() {
                        return Err(invalid("nonfinite Gemma3 BF16 norm"));
                    }
                }
            }
            _ => return Err(invalid("Gemma3 logical tensor representation")),
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
        add(METADATA_RESERVATION, STREAM_BYTES)?,
        options.memory_limit_bytes,
    )?;
    if config.len() > MAX_HEADER_BYTES || index.len() > MAX_HEADER_BYTES {
        return Err(invalid("Gemma3 payload metadata bound"));
    }
    let identity = options.identity;
    if <[u8; 32]>::from(Sha256::digest(config)) != identity.config_sha256
        || <[u8; 32]>::from(Sha256::digest(index)) != identity.index_sha256
    {
        return Err(invalid("Gemma3 config/index identity differs"));
    }
    let mut sha = Sha256::new();
    let mut prefix = [0; 8];
    read_exact(&mut reader, &mut prefix, &mut sha)?;
    let size = usize::try_from(u64::from_le_bytes(prefix))
        .map_err(|_| invalid("Gemma3 header overflow"))?;
    if size == 0 || size > MAX_HEADER_BYTES {
        return Err(invalid("Gemma3 bounded header required"));
    }
    let consumed = add(size, 8)? as u64;
    if consumed > identity.container_bytes {
        return Err(invalid("Gemma3 truncated container"));
    }
    let mut header = buffer(size)?;
    read_exact(&mut reader, &mut header, &mut sha)?;
    if <[u8; 32]>::from(Sha256::digest(&header)) != identity.header_sha256 {
        return Err(invalid("Gemma3 header identity differs"));
    }
    let plan = inspect(config, index, &header, identity.container_bytes)?;
    load_admitted(&mut reader, plan, sha, consumed, options)
}
fn load_admitted<R: Read>(
    reader: &mut R,
    plan: Plan,
    mut sha: Sha256,
    consumed: u64,
    options: LoadOptions,
) -> Result<PayloadStore, Error> {
    let required = add(
        add(plan.payload_bytes(), METADATA_RESERVATION)?,
        STREAM_BYTES,
    )?;
    check_budget(required, options.memory_limit_bytes)?;
    let mut payload = buffer(plan.payload_bytes())?;
    let admitted = add(add(payload.capacity(), METADATA_RESERVATION)?, STREAM_BYTES)?;
    check_budget(admitted, options.memory_limit_bytes)?;
    let mut consumed = consumed;
    for chunk in payload.chunks_mut(STREAM_BYTES) {
        read_exact(reader, chunk, &mut sha)?;
        consumed = consumed
            .checked_add(chunk.len() as u64)
            .ok_or_else(|| invalid("Gemma3 byte count overflow"))?;
    }
    let mut tail = [0];
    if reader
        .read(&mut tail)
        .map_err(|e| invalid(format!("Gemma3 payload EOF: {e}")))?
        != 0
    {
        return Err(invalid("Gemma3 payload trailing bytes"));
    }
    let digest: [u8; 32] = sha.finalize().into();
    if consumed != options.identity.container_bytes || digest != options.identity.container_sha256 {
        return Err(invalid("Gemma3 complete container identity differs"));
    }
    let packed_matrices = plan
        .logical_tensors()
        .iter()
        .filter(|t| matches!(t.storage, Storage::Affine(_)))
        .count();
    let raw_bf16_norms = plan.logical_tensors().len() - packed_matrices;
    let store = PayloadStore {
        report: LoadReport {
            payload_bytes: payload.len(),
            payload_capacity_bytes: payload.capacity(),
            packed_matrices,
            raw_bf16_norms,
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
