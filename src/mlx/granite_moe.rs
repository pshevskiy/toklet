//! Strict single-file BF16 Granite MoE storage/import. No inference admission.
//! Receipts describe delivered bytes, not a published model, codec or GPU owner.
use crate::weights::{
    self, Error,
    content::{ExpectedFile, FileAudit, Manifest},
};
use half::{bf16, f16};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, io::Read, mem::size_of, ops::Range, path::Path};
mod config;
const MAX_CONFIG: usize = 64 * 1024;
const MAX_HEADER: usize = 1024 * 1024;
const METADATA_RESERVATION: usize = 64 * 1024 * 1024;
fn invalid(s: impl Into<String>) -> Error {
    Error::Invalid(s.into())
}
fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or_else(|| invalid("MoE byte overflow"))
}
fn mul(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("MoE extent overflow"))
}
#[derive(Clone, Debug)]
pub struct Config {
    pub dense: crate::model::Config,
    pub experts: usize,
    pub selected: usize,
    pub scales: crate::model::DenseScales,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Embedding,
    Norm,
    Query,
    Key,
    Value,
    AttentionOutput,
    Router,
    GateUp,
    ExpertDown,
}
#[derive(Debug)]
pub struct Tensor {
    name: String,
    shape: [usize; 3],
    rank: usize,
    role: Role,
    range: Range<usize>,
    elements: usize,
}
impl Tensor {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.rank]
    }
    pub fn role(&self) -> Role {
        self.role
    }
    pub fn elements(&self) -> usize {
        self.elements
    }
    pub fn storage_bytes(&self) -> usize {
        self.elements * 2
    }
    /// Tight row-major Half storage; no device-specific padding/guards.
    pub fn alignment(&self) -> usize {
        std::mem::align_of::<u16>()
    }
    pub fn gate_up_slices(&self, expert: usize) -> Result<(Range<usize>, Range<usize>), Error> {
        if self.role != Role::GateUp || expert >= self.shape[0] {
            return Err(invalid("not a fused expert gate/up slice"));
        }
        let stride = self.shape[1] * self.shape[2];
        let start = expert * stride;
        let mid = start + stride / 2;
        Ok((start..mid, mid..start + stride))
    }
    pub fn down_slice(&self, expert: usize) -> Result<Range<usize>, Error> {
        if self.role != Role::ExpertDown || expert >= self.shape[0] {
            return Err(invalid("not an expert down slice"));
        }
        let stride = self.shape[1] * self.shape[2];
        Ok(expert * stride..(expert + 1) * stride)
    }
}
#[derive(Clone, Copy, Debug)]
pub struct StoragePlan {
    pub half_weights_bytes: usize,
    pub largest_tensor_bytes: usize,
    /// Exact Rust owner/range/stream capacities for the admitted import schema.
    pub import_owned_bytes: usize,
    /// Separate reservation for transient parsing/allocator/metadata overhead.
    pub metadata_reservation_bytes: usize,
    pub admitted_load_bytes: usize,
}
#[derive(Debug)]
pub struct Plan {
    config: Config,
    tensors: Vec<Tensor>,
    storage: StoragePlan,
    prefix_bytes: u64,
    container_bytes: u64,
}
impl Plan {
    /// Unauthenticated metadata inspection; no payload, numerical or session admission.
    pub fn inspect(
        config_bytes: &[u8],
        header: &[u8],
        container_bytes: u64,
        budget: usize,
    ) -> Result<Self, Error> {
        if config_bytes.len() > MAX_CONFIG
            || header.len() > MAX_HEADER
            || budget > weights::INITIAL_MEMORY_LIMIT_BYTES
            || container_bytes > weights::content::MAX_FILE_BYTES
        {
            return Err(invalid("MoE metadata/budget bounds"));
        }
        let config = config::parse(config_bytes)?;
        let prefix_bytes = add(header.len(), 8)? as u64;
        let metadata = weights::parse_safetensors_header(header, prefix_bytes, container_bytes)?;
        let c = &config.dense;
        let h = c.hidden_size;
        let i = c.intermediate_size;
        let e = config.experts;
        let mut expected = BTreeMap::new();
        expected.insert(
            "model.embed_tokens.weight".to_owned(),
            (vec![c.vocab_size, h], Role::Embedding),
        );
        expected.insert("model.norm.weight".to_owned(), (vec![h], Role::Norm));
        for layer in 0..c.num_layers {
            let prefix = format!("model.layers.{layer}.");
            for (name, shape, role) in [
                ("input_layernorm.weight", vec![h], Role::Norm),
                ("post_attention_layernorm.weight", vec![h], Role::Norm),
                ("self_attn.q_proj.weight", vec![h, h], Role::Query),
                (
                    "self_attn.k_proj.weight",
                    vec![mul(c.kv_heads, c.head_dim)?, h],
                    Role::Key,
                ),
                (
                    "self_attn.v_proj.weight",
                    vec![mul(c.kv_heads, c.head_dim)?, h],
                    Role::Value,
                ),
                ("self_attn.o_proj.weight", vec![h, h], Role::AttentionOutput),
                (
                    "block_sparse_moe.router.layer.weight",
                    vec![e, h],
                    Role::Router,
                ),
                (
                    "block_sparse_moe.input_linear.weight",
                    vec![e, mul(2, i)?, h],
                    Role::GateUp,
                ),
                (
                    "block_sparse_moe.output_linear.weight",
                    vec![e, h, i],
                    Role::ExpertDown,
                ),
            ] {
                expected.insert(prefix.clone() + name, (shape, role));
            }
        }
        let physical = metadata.tensors();
        if expected.len() != physical.len() {
            return Err(invalid("MoE exact tensor inventory differs"));
        }
        let mut tensors = Vec::new();
        tensors
            .try_reserve_exact(expected.len())
            .map_err(|_| Error::Allocation)?;
        for (name, (shape, role)) in expected {
            let info = physical
                .get(&name)
                .ok_or_else(|| invalid("MoE missing tensor"))?;
            if info.dtype != safetensors::Dtype::BF16 || info.shape != shape {
                return Err(invalid("MoE tensor dtype/shape differs"));
            }
            let elements = shape.iter().try_fold(1usize, |n, &v| mul(n, v))?;
            if info.data_offsets.1.checked_sub(info.data_offsets.0) != Some(mul(elements, 2)?) {
                return Err(invalid("MoE tensor byte extent differs"));
            }
            let mut dims = [1; 3];
            dims[..shape.len()].copy_from_slice(&shape);
            tensors.push(Tensor {
                name,
                shape: dims,
                rank: shape.len(),
                role,
                range: info.data_offsets.0..info.data_offsets.1,
                elements,
            });
        }
        tensors.sort_by_key(|t| t.range.start);
        let mut end = 0;
        let mut largest = 0;
        for t in &tensors {
            if t.range.start != end {
                return Err(invalid("MoE spans must partition payload"));
            }
            end = t.range.end;
            largest = largest.max(t.storage_bytes());
        }
        let table = mul(tensors.capacity(), size_of::<Tensor>())?;
        let names = tensors
            .iter()
            .try_fold(0usize, |n, t| add(n, t.name.capacity()))?;
        let audit = FileAudit::required_bytes(add(tensors.len(), 1)?)?;
        let owned = add(
            add(size_of::<Source>() - size_of::<FileAudit>(), table)?,
            add(names, audit)?,
        )?;
        let admitted = add(add(end, owned)?, METADATA_RESERVATION)?;
        if admitted > budget {
            return Err(Error::Budget {
                required: admitted,
                limit: budget,
            });
        }
        Ok(Self {
            config,
            tensors,
            storage: StoragePlan {
                half_weights_bytes: end,
                largest_tensor_bytes: largest,
                import_owned_bytes: owned,
                metadata_reservation_bytes: METADATA_RESERVATION,
                admitted_load_bytes: admitted,
            },
            prefix_bytes,
            container_bytes,
        })
    }
    pub fn config(&self) -> &Config {
        &self.config
    }
    pub fn tensors(&self) -> &[Tensor] {
        &self.tensors
    }
    pub fn storage(&self) -> StoragePlan {
        self.storage
    }
    /// Prospective F16 full-cache bytes only, not implemented session admission.
    pub fn prospective_kv_bytes(&self, capacity: usize) -> Result<usize, Error> {
        if capacity == 0 || capacity > self.config.dense.max_positions {
            return Err(invalid("MoE prospective KV capacity out of declaration"));
        }
        let c = &self.config.dense;
        [c.num_layers, 2, c.kv_heads, capacity, c.head_dim, 2]
            .into_iter()
            .try_fold(1, mul)
    }
}
/// Caller expectations are verified, not publisher authentication by this API.
pub struct ExpectedArtifact<'a> {
    pub repository: &'a str,
    pub revision: &'a str,
    pub config_bytes: u64,
    pub config_sha256: [u8; 32],
    pub model_bytes: u64,
    pub model_sha256: [u8; 32],
}
#[derive(Clone, Copy, Debug)]
pub struct ImportReceipt {
    manifest: [u8; 32],
    model: [u8; 32],
    half: [u8; 32],
    values: usize,
    underflows: usize,
}
impl ImportReceipt {
    pub fn manifest_identity(&self) -> [u8; 32] {
        self.manifest
    }
    pub fn source_model_sha256(&self) -> [u8; 32] {
        self.model
    }
    pub fn half_payload_sha256(&self) -> [u8; 32] {
        self.half
    }
    pub fn converted_values(&self) -> usize {
        self.values
    }
    pub fn underflowed_values(&self) -> usize {
        self.underflows
    }
}
pub struct Source {
    plan: Plan,
    audit: Option<FileAudit>,
    healthy: bool,
    next: usize,
    converted: Sha256,
    underflows: usize,
    manifest: [u8; 32],
    model: [u8; 32],
}
impl Source {
    pub fn open(
        directory: &Path,
        expected: ExpectedArtifact<'_>,
        budget: usize,
    ) -> Result<Self, Error> {
        if budget < METADATA_RESERVATION {
            return Err(Error::Budget {
                required: METADATA_RESERVATION,
                limit: budget,
            });
        }
        if expected.config_bytes == 0 || expected.config_bytes > MAX_CONFIG as u64 {
            return Err(invalid("MoE config bound"));
        }
        match fs::symlink_metadata(directory.join("model.safetensors.index.json")) {
            Ok(_) => return Err(invalid("MoE import requires one unindexed shard")),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(invalid(format!("MoE index inspection: {e}"))),
        }
        let entries = [
            ExpectedFile {
                name: "config.json",
                bytes: expected.config_bytes,
                sha256: expected.config_sha256,
            },
            ExpectedFile {
                name: "model.safetensors",
                bytes: expected.model_bytes,
                sha256: expected.model_sha256,
            },
        ];
        let manifest = Manifest::new(expected.repository, expected.revision, &entries)?;
        let mut config = vec![0; expected.config_bytes as usize];
        let mut file = weights::audit::open_regular(&directory.join("config.json"))?;
        if file.metadata().map_err(|e| invalid(e.to_string()))?.len() != expected.config_bytes {
            return Err(invalid("MoE config extent differs"));
        }
        file.read_exact(&mut config)
            .map_err(|e| invalid(e.to_string()))?;
        if <[u8; 32]>::from(Sha256::digest(&config)) != expected.config_sha256 {
            return Err(invalid("MoE config digest differs"));
        }
        let mut file = weights::audit::open_regular(&directory.join("model.safetensors"))?;
        if file.metadata().map_err(|e| invalid(e.to_string()))?.len() != expected.model_bytes {
            return Err(invalid("MoE container extent differs"));
        }
        let mut length = [0; 8];
        file.read_exact(&mut length)
            .map_err(|e| invalid(e.to_string()))?;
        let n = usize::try_from(u64::from_le_bytes(length))
            .map_err(|_| invalid("MoE header length overflow"))?;
        if n > MAX_HEADER {
            return Err(invalid("MoE header exceeds bound"));
        }
        let mut header = vec![0; n];
        file.read_exact(&mut header)
            .map_err(|e| invalid(e.to_string()))?;
        let plan = Plan::inspect(&config, &header, expected.model_bytes, budget)?;
        let mut prefix_sha = Sha256::new();
        prefix_sha.update(length);
        prefix_sha.update(&header);
        let prefix: [u8; 32] = prefix_sha.finalize().into();
        drop(header);
        drop(config);
        drop(file);
        let mut ends = Vec::with_capacity(plan.tensors.len() + 1);
        ends.push(plan.prefix_bytes);
        for t in &plan.tensors {
            ends.push(plan.prefix_bytes + t.range.end as u64);
        }
        let mut audit = FileAudit::open(directory, &manifest, 1, &ends, budget)?;
        if audit.owned_bytes()? != FileAudit::required_bytes(ends.len())? {
            return Err(invalid("MoE audit capacity differs from plan"));
        }
        let mut delivered = Sha256::new();
        audit.visit(0, |b| {
            delivered.update(b);
            Ok(())
        })?;
        if <[u8; 32]>::from(delivered.finalize()) != prefix {
            return Err(invalid(
                "MoE inspected header differs from authenticated prefix",
            ));
        }
        Ok(Self {
            plan,
            audit: Some(audit),
            healthy: true,
            next: 0,
            converted: Sha256::new(),
            underflows: 0,
            manifest: manifest.identity(),
            model: expected.model_sha256,
        })
    }
    pub fn plan(&self) -> &Plan {
        &self.plan
    }
    /// Direct final tight Half slice; exact length is checked before writes.
    /// Any error/panic poisons completion; discard ALL partial destinations.
    pub fn import_next(&mut self, destination: &mut [u16]) -> Result<(), Error> {
        let clean = self.healthy;
        self.healthy = false;
        if !clean {
            return Err(invalid("MoE import poisoned"));
        }
        let t = self
            .plan
            .tensors
            .get(self.next)
            .ok_or_else(|| invalid("MoE import already complete"))?;
        if destination.len() != t.elements {
            return Err(invalid(
                "MoE destination capacity differs from storage plan",
            ));
        }
        let mut cursor = 0;
        let hash = &mut self.converted;
        let underflows = &mut self.underflows;
        self.audit
            .as_mut()
            .ok_or_else(|| invalid("MoE audit unavailable"))?
            .visit(self.next + 1, |bytes| {
                if !bytes.len().is_multiple_of(2) {
                    return Err(invalid("MoE BF16 stream alignment"));
                }
                for pair in bytes.chunks_exact(2) {
                    let source = bf16::from_le_bytes([pair[0], pair[1]]).to_f32();
                    let (bits, underflow) = convert_value(source)?;
                    if underflow {
                        *underflows = add(*underflows, 1)?;
                    }
                    destination[cursor] = bits;
                    cursor += 1;
                    hash.update(bits.to_le_bytes());
                }
                Ok(())
            })?;
        if cursor != t.elements {
            return Err(invalid("MoE converted extent differs"));
        }
        self.next += 1;
        self.healthy = true;
        Ok(())
    }
    /// Historical delivery/conversion only; does not attest destination lifetime.
    pub fn finish(mut self) -> Result<ImportReceipt, Error> {
        if !self.healthy || self.next != self.plan.tensors.len() {
            return Err(invalid("MoE import incomplete or poisoned"));
        }
        let delivery = self
            .audit
            .take()
            .ok_or_else(|| invalid("MoE missing delivery"))?
            .finish()?;
        if delivery.manifest_identity() != self.manifest
            || delivery.file_sha256() != self.model
            || delivery.bytes() != self.plan.container_bytes
        {
            return Err(invalid("MoE delivery identity differs"));
        }
        Ok(ImportReceipt {
            manifest: self.manifest,
            model: self.model,
            half: self.converted.finalize().into(),
            values: self.plan.storage.half_weights_bytes / 2,
            underflows: self.underflows,
        })
    }
}

fn convert_value(source: f32) -> Result<(u16, bool), Error> {
    let value = f16::from_f32(source);
    if !source.is_finite() || !value.is_finite() {
        return Err(invalid("MoE nonfinite or Half overflow"));
    }
    Ok((value.to_bits(), source != 0. && value.to_f32() == 0.))
}
