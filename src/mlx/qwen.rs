//! Validated local MLX Qwen2/Llama artifact source, without full-model expansion.
//!
//! Opening validates config, index and tensor metadata. Payload reads check
//! floating values/affine metadata; no GPU allocation or model execution occurs.

use super::{
    affine::{Layout, Matrix, MetadataDtype},
    dense::{self, Architecture},
};
use crate::{
    model::Config,
    weights::{self, Error, LoadOptions},
};
use safetensors::Dtype;
use serde::{
    Deserialize,
    de::{self, MapAccess, SeqAccess, Visitor},
};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    fs::File,
    io::Read,
    ops::Range,
    path::Path,
};

mod logical;
mod phi3;
pub(crate) use logical::LogicalTensorSpec;

const MAX_CONFIG_BYTES: usize = 1024 * 1024;
const METADATA_ALLOWANCE: usize = 64 * 1024 * 1024;
const STREAM_BYTES: usize = 64 * 1024;

fn invalid(reason: impl Into<String>) -> Error {
    Error::Invalid(reason.into())
}
fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b)
        .ok_or_else(|| invalid("MLX artifact size overflow"))
}

/// A floating tensor retains its source precision; affine matrices retain all
/// three original packed buffers. This is an artifact plan, not a GPU layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Storage {
    Float(MetadataDtype),
    Affine(Layout),
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    shape: Vec<usize>,
    storage: Storage,
    payload_bytes: usize,
}
impl TensorSpec {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn storage(&self) -> Storage {
        self.storage
    }
    pub fn payload_bytes(&self) -> usize {
        self.payload_bytes
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Quantization {
    bits: u8,
    group_size: usize,
}
impl Quantization {
    fn parse(value: &Map<String, Value>, defaults: bool) -> Result<Self, Error> {
        for name in value.keys() {
            if !["bits", "group_size", "mode"].contains(&name.as_str()) && !defaults {
                return Err(invalid(format!(
                    "unsupported MLX quantization option: {name}"
                )));
            }
        }
        if value
            .get("mode")
            .is_some_and(|v| v.as_str() != Some("affine"))
        {
            return Err(invalid("only MLX affine quantization is supported"));
        }
        let bits = value
            .get("bits")
            .and_then(Value::as_u64)
            .and_then(|v| u8::try_from(v).ok())
            .ok_or_else(|| invalid("MLX bits must be an integer"))?;
        let group_size = value
            .get("group_size")
            .and_then(Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| invalid("MLX group_size must be an integer"))?;
        Layout::new(1, 128, bits, group_size, MetadataDtype::F16)
            .map_err(|e| invalid(e.to_string()))?;
        Ok(Self { bits, group_size })
    }
}
struct QuantizationPlan {
    default: Quantization,
    overrides: BTreeMap<String, Option<Quantization>>,
}
impl QuantizationPlan {
    fn parse(value: Value) -> Result<Self, Error> {
        let object = value
            .as_object()
            .ok_or_else(|| invalid("MLX quantization must be an object"))?;
        let default = Quantization::parse(object, true)?;
        let mut overrides = BTreeMap::new();
        for (name, value) in object {
            if ["bits", "group_size", "mode"].contains(&name.as_str()) {
                continue;
            }
            let quantization = if value == &Value::Bool(false) {
                None
            } else {
                Some(Quantization::parse(
                    value
                        .as_object()
                        .ok_or_else(|| invalid(format!("invalid MLX layer instruction: {name}")))?,
                    false,
                )?)
            };
            overrides.insert(name.clone(), quantization);
        }
        Ok(Self { default, overrides })
    }
    fn resolve(&self, module: &str) -> Option<Quantization> {
        self.overrides
            .get(module)
            .copied()
            .unwrap_or(Some(self.default))
    }
}

pub use crate::weights::audit::SourceContentIdentity;
pub use crate::weights::content::{DenseSourceContentIdentity, QwenSourceContentIdentity};

/// Immutable metadata and open shard handles. Tensor-name lookups belong to
/// loading only; execution should retain resolved concrete buffers/operations.
pub struct Source {
    config: Config,
    declared_positions: usize,
    declared_sliding_window: Option<usize>,
    architecture: Architecture,
    tied_output: bool,
    position_plan: crate::model::PositionPlan,
    rotary_layers: Option<Vec<bool>>,
    scales: crate::model::DenseScales,
    tensors: BTreeMap<String, TensorSpec>,
    logical: BTreeMap<String, LogicalTensorSpec>,
    source: weights::Source,
    tensor_payload_bytes: usize,
    admitted_load_bytes: usize,
}
impl Source {
    /// No weights are expanded/allocated. Admit source-precision tensor payloads
    /// plus a 64 MiB metadata estimate and 64 KiB shared stream buffer. This does
    /// not admit a session, GPU padding/conversion, tokenizer or application.
    /// A redundant tied head must be representation- and byte-identical; it is
    /// streamed for verification but omitted from owned tensor payload totals.
    pub fn open(directory: &Path, options: LoadOptions) -> Result<Self, Error> {
        Self::open_impl(directory, options, false)
    }

    /// Additive dense admission. The original `open` remains Qwen-only so
    /// existing consumers cannot accidentally opt into another graph.
    pub fn open_dense(directory: &Path, options: LoadOptions) -> Result<Self, Error> {
        Self::open_impl(directory, options, true)
    }

    /// Explicit Metal-only context diagnostic. Validate the ordinary source at
    /// its existing ceiling first, retaining the same document's declaration.
    /// Context does not affect tensor shapes or payload ownership. No generic
    /// parser/CPU/ordinary compatibility limit is widened by this route.
    #[cfg(any(
        test,
        all(
            feature = "metal",
            any(target_os = "macos", target_os = "ios"),
            target_arch = "aarch64"
        )
    ))]
    pub(crate) fn open_qwen3_context_diagnostic(
        directory: &Path,
        options: LoadOptions,
    ) -> Result<Self, Error> {
        let requested = options.context_limit;
        if requested == 0 || requested > super::execution::QWEN3_CONTEXT_DIAGNOSTIC_LIMIT {
            return Err(invalid(
                "Qwen3 context diagnostic requires1..=40960 positions",
            ));
        }
        let mut source = Self::open_dense(
            directory,
            LoadOptions {
                context_limit: requested.min(weights::MAX_CONTEXT_LIMIT),
                ..options
            },
        )?;
        if requested > source.declared_positions {
            return Err(invalid(
                "Qwen3 context diagnostic exceeds source position declaration",
            ));
        }
        let config = Config {
            max_positions: requested,
            ..source.config.clone()
        };
        super::execution::qwen3_context_geometry(&config, source.architecture)
            .map_err(|error| invalid(error.message()))?;
        source.config = config;
        Ok(source)
    }

    /// Opt in to the exact frozen Smol checkpoint's consumed-byte audit. This
    /// is source evidence only, not model assembly, codec pairing or capability.
    /// Any failed public consuming operation, including invalid arguments,
    /// poisons later identity publication. Discard partial destinations on error.
    pub fn open_smol_audited(directory: &Path, options: LoadOptions) -> Result<Self, Error> {
        Self::validate_budget(options)?;
        let mut audit = weights::audit::Audit::smol(options.memory_limit_bytes)?;
        let bytes = audit.read_metadata(
            &directory.join("config.json"),
            weights::audit::MetadataKind::Config,
        )?;
        let document = parse_json(&bytes)?;
        Self::open_document(directory, options, true, document, Some(audit), None)
    }
    /// Opt-in consumed-source audit for the pinned managed Qwen3-0.6B 8-bit
    /// inventory. Full-file prepasses authenticate range hashes, then the same
    /// bytes delivered to consumers are reverified. This adds cold I/O, not a
    /// new architecture/context/mode admission. No model or codec is assembled.
    /// Discard partial destinations after any error. Invalid public consuming
    /// operations poison later source-identity publication, as in the Smol path.
    pub fn open_qwen3_audited(directory: &Path, options: LoadOptions) -> Result<Self, Error> {
        Self::open_dense_audited(
            directory,
            options,
            weights::content::DenseProfile::Qwen(weights::content::QwenProfile::Qwen3),
        )
    }
    /// Opt-in config/index/container delivery for the existing pinned Qwen2.5
    /// 0.5B 8-bit artifact. No model assembly, codec or native mode is implied.
    /// Failed consuming operations poison evidence; discard partial destinations.
    pub fn open_qwen2_audited(directory: &Path, options: LoadOptions) -> Result<Self, Error> {
        Self::open_dense_audited(
            directory,
            options,
            weights::content::DenseProfile::Qwen(weights::content::QwenProfile::Qwen2),
        )
    }
    /// Opt-in delivered config/index/container evidence for the pinned Mistral
    /// artifact. This does not assemble a model, pair a codec or change modes.
    pub fn open_mistral_audited(directory: &Path, options: LoadOptions) -> Result<Self, Error> {
        Self::open_dense_audited(directory, options, weights::content::DenseProfile::Mistral)
    }
    /// Pinned Llama3 audit through the qualified scaled-RoPE Mac512 graph.
    pub fn open_llama3_audited(directory: &Path, options: LoadOptions) -> Result<Self, Error> {
        if options.context_limit > 512 {
            return Err(invalid("audited Llama3 requires context <=512"));
        }
        Self::open_dense_audited(directory, options, weights::content::DenseProfile::Llama3)
    }
    /// Authenticated numerical diagnostic only; no complete app/codec identity.
    #[cfg(any(
        test,
        all(
            feature = "metal",
            any(target_os = "macos", target_os = "ios"),
            target_arch = "aarch64"
        )
    ))]
    pub(crate) fn open_llama3_quality_diagnostic(
        directory: &Path,
        options: LoadOptions,
    ) -> Result<Self, Error> {
        if options.context_limit == 0 || options.context_limit > 1025 {
            return Err(invalid(
                "Llama quality diagnostic requires1..=1025 positions",
            ));
        }
        let source =
            Self::open_dense_audited(directory, options, weights::content::DenseProfile::Llama3)?;
        super::execution::llama3_quality_geometry(&source)
            .map_err(|error| invalid(error.message()))?;
        Ok(source)
    }
    fn open_dense_audited(
        directory: &Path,
        options: LoadOptions,
        profile: weights::content::DenseProfile,
    ) -> Result<Self, Error> {
        Self::validate_budget(options)?;
        let (bytes, receipt) =
            weights::content::DenseAudit::config(profile, directory, options.memory_limit_bytes)?;
        let document = parse_json(&bytes)?;
        drop(bytes);
        Self::open_document(
            directory,
            options,
            true,
            document,
            None,
            Some((profile, receipt)),
        )
    }
    /// Historical config/index/container consumption only. The embedded full
    /// manifest is an expected anchor, not proof of tokenizer/ancillary reads,
    /// model assembly, codec pairing or persistent preference eligibility.
    pub fn qwen3_content_identity(&self) -> Option<QwenSourceContentIdentity> {
        self.source
            .qwen_profile_content_identity(weights::content::QwenProfile::Qwen3)
    }
    /// Historical delivery for the pinned Qwen2 profile only; never Qwen3.
    pub fn qwen2_content_identity(&self) -> Option<QwenSourceContentIdentity> {
        self.source
            .qwen_profile_content_identity(weights::content::QwenProfile::Qwen2)
    }
    /// Historical config/index/container delivery for either admitted Qwen
    /// profile. The expected manifest identifies its scope, not model ownership.
    pub fn qwen_content_identity(&self) -> Option<QwenSourceContentIdentity> {
        self.source.qwen_content_identity()
    }
    /// Historical delivered bytes for any admitted dense audit profile. Not
    /// model ownership, codec pairing or persistence authority.
    pub fn dense_content_identity(&self) -> Option<DenseSourceContentIdentity> {
        self.source.dense_content_identity()
    }
    /// Historical delivered bytes for the pinned Mistral profile only.
    pub fn mistral_content_identity(&self) -> Option<DenseSourceContentIdentity> {
        self.source
            .dense_profile_content_identity(weights::content::DenseProfile::Mistral)
    }
    /// Historical evidence only after every trusted physical range was consumed
    /// successfully. Prefer querying once at the final consumption boundary.
    pub fn content_identity(&self) -> Option<SourceContentIdentity> {
        self.source.content_identity()
    }
    /// Retained ledger capacities, counted inside the existing metadata policy.
    pub fn content_audit_owned_bytes(&self) -> Option<usize> {
        self.source.audit_owned_bytes()
    }

    fn open_impl(directory: &Path, options: LoadOptions, allow_dense: bool) -> Result<Self, Error> {
        Self::validate_budget(options)?;
        let document = read_config(&directory.join("config.json"))?;
        Self::open_document(directory, options, allow_dense, document, None, None)
    }

    fn validate_budget(options: LoadOptions) -> Result<(), Error> {
        if options.memory_limit_bytes == 0
            || options.memory_limit_bytes > weights::INITIAL_MEMORY_LIMIT_BYTES
        {
            return Err(invalid("MLX loading budget must be within 6 GiB"));
        }
        Ok(())
    }

    /// The inspection path supplies the same bounded, duplicate-checked document
    /// it classified. Reopening config here could otherwise report two versions.
    pub(crate) fn from_inspected_config(
        directory: &Path,
        options: LoadOptions,
        document: Value,
    ) -> Result<Self, Error> {
        Self::open_document(directory, options, true, document, None, None)
    }

    fn open_document(
        directory: &Path,
        options: LoadOptions,
        allow_dense: bool,
        mut document: Value,
        audit: Option<weights::audit::Audit>,
        dense_config: Option<(
            weights::content::DenseProfile,
            weights::content::FileDelivery,
        )>,
    ) -> Result<Self, Error> {
        Self::validate_budget(options)?;
        let object = document
            .as_object_mut()
            .ok_or_else(|| invalid("MLX config must be an object"))?;
        let quantization = object.remove("quantization");
        let alternate = object.remove("quantization_config");
        let quantization = match (quantization, alternate) {
            (Some(a), Some(b)) if a != b => {
                return Err(invalid("conflicting MLX quantization declarations"));
            }
            (Some(a), _) | (_, Some(a)) => a,
            _ => {
                return Err(invalid(
                    "MLX affine artifact requires quantization settings",
                ));
            }
        };
        let plan = QuantizationPlan::parse(quantization)?;
        // Retain the declaration from this exact duplicate-checked document,
        // before architecture validation narrows Config to requested context.
        let declared_positions = object
            .get("max_position_embeddings")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|&value| value > 0)
            .ok_or_else(|| invalid("positive model position declaration required"))?;
        let implicit_smollm3_tied = object.get("model_type").and_then(Value::as_str)
            == Some("smollm3")
            && !object.contains_key("tie_word_embeddings");
        let dense::ParsedConfig {
            config,
            architecture,
            tied_output,
            position_plan,
            declared_sliding_window,
            rotary_layers,
            scales,
        } = if allow_dense {
            dense::config(document, options)?
        } else {
            dense::ParsedConfig {
                config: weights::qwen_config(document, options)?,
                architecture: Architecture::Qwen2,
                tied_output: true,
                position_plan: crate::model::PositionPlan::Ordinary,
                declared_sliding_window: None,
                rotary_layers: None,
                scales: Default::default(),
            }
        };
        if let Some((profile, _)) = dense_config {
            let expected = match profile {
                weights::content::DenseProfile::Qwen(weights::content::QwenProfile::Qwen2) => {
                    Architecture::Qwen2
                }
                weights::content::DenseProfile::Qwen(weights::content::QwenProfile::Qwen3) => {
                    Architecture::Qwen3
                }
                weights::content::DenseProfile::Mistral => Architecture::Mistral,
                weights::content::DenseProfile::Llama3 => Architecture::Llama,
            };
            if audit.is_some() || architecture != expected {
                return Err(invalid("Dense source audit profile mismatch"));
            }
        }
        // Bound name-map construction before looping over declared layers.
        let count = config
            .num_layers
            .checked_mul(12)
            .and_then(|v| v.checked_add(2))
            .ok_or_else(|| invalid("MLX tensor count overflow"))?;
        if count > 8192 {
            return Err(invalid(
                "MLX dense loader supports at most 8192 logical tensors",
            ));
        }
        let mut expected = weights::expected(&config)?;
        if !architecture.has_projection_bias() {
            expected.retain(|name, _| !name.ends_with(".bias"));
        }
        if architecture == Architecture::Qwen3 {
            for layer in 0..config.num_layers {
                for name in ["q_norm", "k_norm"] {
                    expected.insert(
                        format!("model.layers.{layer}.self_attn.{name}.weight"),
                        vec![config.head_dim],
                    );
                }
            }
        }
        let partitions = if architecture == Architecture::Phi3 {
            phi3::inventory(&config, &mut expected)?
        } else {
            Vec::new()
        };
        let source = if let Some((profile, config_receipt)) = dense_config {
            weights::content::DenseAudit::open_source(
                profile,
                directory,
                config_receipt,
                options.memory_limit_bytes,
            )?
        } else {
            weights::Source::open_with_audit(directory, audit)?
        };
        let has_head = source.info("lm_head.weight").is_some();
        if implicit_smollm3_tied && has_head {
            return Err(invalid(
                "implicit SmolLM3 tying requires a single embedding table",
            ));
        }
        if has_head || !tied_output {
            expected.insert(
                "lm_head.weight".into(),
                vec![config.vocab_size, config.hidden_size],
            );
        }
        let modules: BTreeSet<_> = expected
            .iter()
            .filter(|(_, shape)| shape.len() == 2)
            .map(|(name, _)| name.strip_suffix(".weight").unwrap())
            .collect();
        for name in plan.overrides.keys() {
            if !modules.contains(name.as_str()) {
                return Err(invalid(format!("unknown MLX quantized module: {name}")));
            }
        }
        let mut physical = BTreeSet::new();
        let mut tensors = BTreeMap::new();
        let mut tensor_payload_bytes = 0;
        for (name, shape) in &expected {
            let info = source
                .info(name)
                .ok_or_else(|| invalid(format!("missing tensor: {name}")))?;
            physical.insert(name.clone());
            let module = name.strip_suffix(".weight");
            let scales = module.map(|p| format!("{p}.scales"));
            let packed =
                shape.len() == 2 && scales.as_ref().is_some_and(|p| source.info(p).is_some());
            let storage = if packed {
                let module = module.unwrap();
                let quantization = plan.resolve(module).ok_or_else(|| {
                    invalid(format!("excluded MLX module has packed tensors: {module}"))
                })?;
                let scales = scales.as_ref().unwrap();
                let biases = format!("{module}.biases");
                let scale_info = source.info(scales).unwrap();
                let bias_info = source
                    .info(&biases)
                    .ok_or_else(|| invalid(format!("missing tensor: {biases}")))?;
                let dtype = metadata_dtype(scale_info.dtype)?;
                let layout = Layout::new(
                    shape[0],
                    shape[1],
                    quantization.bits,
                    quantization.group_size,
                    dtype,
                )
                .map_err(|e| invalid(format!("{name}: {e}")))?;
                let metadata_shape = [shape[0], shape[1] / layout.group_size()];
                if info.dtype != Dtype::U32
                    || info.shape != [shape[0], layout.row_bytes() / 4]
                    || scale_info.shape != metadata_shape
                    || bias_info.shape != metadata_shape
                    || bias_info.dtype != scale_info.dtype
                {
                    return Err(invalid(format!(
                        "MLX packed tensor shape/dtype mismatch: {name}"
                    )));
                }
                physical.insert(scales.clone());
                physical.insert(biases);
                Storage::Affine(layout)
            } else {
                if &info.shape != shape {
                    return Err(invalid(format!("tensor shape mismatch: {name}")));
                }
                Storage::Float(metadata_dtype(info.dtype)?)
            };
            let payload_bytes = match storage {
                Storage::Affine(layout) => layout.total_bytes(),
                Storage::Float(_) => info.data_offsets.1 - info.data_offsets.0,
            };
            if name != "lm_head.weight" || !tied_output {
                tensor_payload_bytes = add(tensor_payload_bytes, payload_bytes)?;
            }
            tensors.insert(
                name.clone(),
                TensorSpec {
                    shape: shape.clone(),
                    storage,
                    payload_bytes,
                },
            );
        }
        if source.names().any(|name| !physical.contains(name)) {
            return Err(invalid("unexpected tensor in MLX dense artifact"));
        }
        let admitted_load_bytes =
            add(add(tensor_payload_bytes, METADATA_ALLOWANCE)?, STREAM_BYTES)?;
        if admitted_load_bytes > options.memory_limit_bytes {
            return Err(Error::Budget {
                required: admitted_load_bytes,
                limit: options.memory_limit_bytes,
            });
        }
        let mut result = Self {
            config,
            declared_positions,
            declared_sliding_window,
            architecture,
            tied_output,
            position_plan,
            rotary_layers,
            scales,
            tensors,
            logical: BTreeMap::new(),
            source,
            tensor_payload_bytes,
            admitted_load_bytes,
        };
        if has_head && tied_output {
            result.check_tied_output()?;
            result.tensors.remove("lm_head.weight");
        }
        result.logical = logical::plan(&result.tensors, &partitions)?;
        Ok(result)
    }
    pub fn config(&self) -> &Config {
        &self.config
    }
    /// Validated declared total positions, before the requested load limit.
    /// Not proof of numerical qualification, artifact provenance or device fit.
    pub fn declared_positions(&self) -> usize {
        self.declared_positions
    }
    /// Original source window. The initial Phi path uses a bounded equivalent
    /// full prefix, not a ring cache; this is not a residency or fit estimate.
    pub fn declared_sliding_window(&self) -> Option<usize> {
        self.declared_sliding_window
    }
    pub fn architecture(&self) -> Architecture {
        self.architecture
    }
    pub fn tied_output(&self) -> bool {
        self.tied_output
    }
    pub fn position_plan(&self) -> crate::model::PositionPlan {
        self.position_plan
    }
    pub fn rotary_layers(&self) -> Option<&[bool]> {
        self.rotary_layers.as_deref()
    }
    pub fn dense_scales(&self) -> crate::model::DenseScales {
        self.scales
    }
    pub fn tensors(&self) -> &BTreeMap<String, TensorSpec> {
        &self.tensors
    }
    /// Original payloads counted once, without a second tied output table.
    pub fn tensor_payload_bytes(&self) -> usize {
        self.tensor_payload_bytes
    }
    pub fn admitted_load_bytes(&self) -> usize {
        self.admitted_load_bytes
    }

    /// Stream decoded floating values. A failing read or consumer may already
    /// have written a partial destination; discard it before publishing a model.
    pub fn visit_float(
        &mut self,
        name: &str,
        consume: impl FnMut(f32) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let result = (|| {
            if !matches!(
                self.tensors.get(name).map(|s| s.storage),
                Some(Storage::Float(_))
            ) {
                return Err(invalid("MLX tensor is not a planned floating tensor"));
            }
            self.source.visit(name, consume)
        })();
        if result.is_err() {
            self.source.poison_audit();
        }
        result
    }
    /// Stream rows of a planned rank-two floating tensor. Dtype is selected once
    /// and decoded from little-endian bytes without pointer alignment assumptions.
    /// Only selected values are checked for finiteness; no whole-tensor identity
    /// is established. Audited sources reject this path, even for all rows.
    /// A read/consumer failure requires discarding any partial destination.
    pub fn visit_float_rows(
        &mut self,
        name: &str,
        rows: Range<usize>,
        consume: impl FnMut(f32) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let result = (|| {
            let spec = self
                .tensors
                .get(name)
                .ok_or_else(|| invalid("MLX tensor is not a planned floating tensor"))?;
            let Storage::Float(dtype) = spec.storage else {
                return Err(invalid("MLX tensor is not a planned floating tensor"));
            };
            if spec.shape.len() != 2
                || spec.shape[1] == 0
                || rows.start >= rows.end
                || rows.end > spec.shape[0]
            {
                return Err(invalid("MLX floating row range out of bounds"));
            }
            let stride = spec.shape[1]
                .checked_mul(dtype.bytes())
                .ok_or_else(|| invalid("MLX floating row size overflow"))?;
            let start = rows
                .start
                .checked_mul(stride)
                .ok_or_else(|| invalid("MLX floating row size overflow"))?;
            let end = rows
                .end
                .checked_mul(stride)
                .ok_or_else(|| invalid("MLX floating row size overflow"))?;
            fn stream<const WIDTH: usize>(
                source: &mut weights::Source,
                name: &str,
                range: Range<usize>,
                decode: impl Fn([u8; WIDTH]) -> f32,
                mut consume: impl FnMut(f32) -> Result<(), Error>,
            ) -> Result<(), Error> {
                source.visit_raw_range(name, range, |bytes| {
                    // The shared buffer and admitted ranges are multiples of
                    // both supported scalar widths; never cast a byte pointer.
                    let chunks = bytes.chunks_exact(WIDTH);
                    if !chunks.remainder().is_empty() {
                        return Err(invalid("MLX floating range byte alignment"));
                    }
                    for raw in chunks {
                        let value = decode(raw.try_into().unwrap());
                        if !value.is_finite() {
                            return Err(invalid("nonfinite weight value in tensor payload"));
                        }
                        consume(value)?;
                    }
                    Ok(())
                })
            }
            match dtype {
                MetadataDtype::F32 => stream(
                    &mut self.source,
                    name,
                    start..end,
                    f32::from_le_bytes,
                    consume,
                ),
                MetadataDtype::F16 => stream(
                    &mut self.source,
                    name,
                    start..end,
                    |v| half::f16::from_le_bytes(v).to_f32(),
                    consume,
                ),
                MetadataDtype::BF16 => stream(
                    &mut self.source,
                    name,
                    start..end,
                    |v| half::bf16::from_le_bytes(v).to_f32(),
                    consume,
                ),
            }
        })();
        if result.is_err() {
            self.source.poison_audit();
        }
        result
    }

    /// Read original packed row spans into three caller-owned allocations.
    /// All destination sizes are checked before any write. Selected metadata is
    /// checked after streaming; unselected metadata is not read or validated.
    /// Discard all three partial destinations on failure. No whole-tensor proof
    /// is produced; audited sources reject even a full-row-range request.
    pub fn read_affine_rows(
        &mut self,
        name: &str,
        rows: Range<usize>,
        weights: &mut [u8],
        scales: &mut [u8],
        biases: &mut [u8],
    ) -> Result<Layout, Error> {
        let result = (|| {
            let Some(Storage::Affine(layout)) = self.tensors.get(name).map(|s| s.storage) else {
                return Err(invalid("MLX tensor is not a planned affine tensor"));
            };
            let span = layout.row_span(rows).map_err(|e| invalid(e.to_string()))?;
            let selected = span.layout();
            if weights.len() != selected.weight_bytes()
                || scales.len() != selected.metadata_bytes()
                || biases.len() != selected.metadata_bytes()
            {
                return Err(invalid(
                    "MLX destination sizes do not match planned row span",
                ));
            }
            let module = name
                .strip_suffix(".weight")
                .ok_or_else(|| invalid("MLX affine tensor name lacks weight suffix"))?;
            self.read_raw_range(name, span.weight_range(), weights)?;
            self.read_raw_range(&format!("{module}.scales"), span.metadata_range(), scales)?;
            self.read_raw_range(&format!("{module}.biases"), span.metadata_range(), biases)?;
            Matrix::from_le_bytes(selected, weights, scales, biases)
                .map_err(|e| invalid(e.to_string()))?;
            Ok(selected)
        })();
        if result.is_err() {
            self.source.poison_audit();
        }
        result
    }
    fn read_raw_range(
        &mut self,
        name: &str,
        range: Range<usize>,
        destination: &mut [u8],
    ) -> Result<(), Error> {
        let mut offset = 0;
        self.source.visit_raw_range(name, range, |bytes| {
            let end = add(offset, bytes.len())?;
            destination
                .get_mut(offset..end)
                .ok_or_else(|| invalid("MLX raw range size mismatch"))?
                .copy_from_slice(bytes);
            offset = end;
            Ok(())
        })?;
        if offset != destination.len() {
            return Err(invalid("MLX raw range size mismatch"));
        }
        Ok(())
    }

    /// Read directly into three caller-admitted packed allocations, then check
    /// finite affine metadata. Buffers are partial/unpublished on any error.
    pub fn read_affine(
        &mut self,
        name: &str,
        weights: &mut [u8],
        scales: &mut [u8],
        biases: &mut [u8],
    ) -> Result<Layout, Error> {
        let result = (|| {
            let Some(Storage::Affine(layout)) = self.tensors.get(name).map(|s| s.storage) else {
                return Err(invalid("MLX tensor is not a planned affine tensor"));
            };
            if weights.len() != layout.weight_bytes()
                || scales.len() != layout.metadata_bytes()
                || biases.len() != layout.metadata_bytes()
            {
                return Err(invalid("MLX destination sizes do not match planned tensor"));
            }
            let module = name.strip_suffix(".weight").unwrap();
            self.read_raw(name, weights)?;
            self.read_raw(&format!("{module}.scales"), scales)?;
            self.read_raw(&format!("{module}.biases"), biases)?;
            Matrix::from_le_bytes(layout, weights, scales, biases)
                .map_err(|e| invalid(e.to_string()))?;
            Ok(layout)
        })();
        if result.is_err() {
            self.source.poison_audit();
        }
        result
    }
    fn read_raw(&mut self, name: &str, destination: &mut [u8]) -> Result<(), Error> {
        let mut offset = 0;
        self.source.visit_raw(name, |bytes| {
            let end = add(offset, bytes.len())?;
            let target = destination
                .get_mut(offset..end)
                .ok_or_else(|| invalid("MLX raw read size mismatch"))?;
            target.copy_from_slice(bytes);
            offset = end;
            Ok(())
        })?;
        if offset != destination.len() {
            return Err(invalid("MLX raw read size mismatch"));
        }
        Ok(())
    }
    fn check_tied_output(&mut self) -> Result<(), Error> {
        if self.tensors["lm_head.weight"] != self.tensors["model.embed_tokens.weight"] {
            return Err(invalid(
                "redundant tied MLX head has a different representation",
            ));
        }
        let suffixes: &[&str] = match self.tensors["lm_head.weight"].storage {
            Storage::Float(_) => &["weight"],
            Storage::Affine(_) => &["weight", "scales", "biases"],
        };
        for suffix in suffixes {
            let mut digest = |name: &str| -> Result<[u8; 32], Error> {
                let mut hash = Sha256::new();
                self.source.visit_raw(name, |bytes| {
                    hash.update(bytes);
                    Ok(())
                })?;
                Ok(hash.finalize().into())
            };
            if digest(&format!("model.embed_tokens.{suffix}"))?
                != digest(&format!("lm_head.{suffix}"))?
            {
                return Err(invalid(
                    "redundant tied MLX head differs from embedding bytes",
                ));
            }
        }
        Ok(())
    }
}

fn metadata_dtype(dtype: Dtype) -> Result<MetadataDtype, Error> {
    match dtype {
        Dtype::F16 => Ok(MetadataDtype::F16),
        Dtype::BF16 => Ok(MetadataDtype::BF16),
        Dtype::F32 => Ok(MetadataDtype::F32),
        _ => Err(invalid(format!(
            "unsupported MLX floating dtype: {dtype:?}"
        ))),
    }
}

// Validate nested uniqueness before serde_json::Value can erase duplicate
// quantization instructions. The ordinary parser retains exact integer syntax.
struct Unique;
impl<'de> Deserialize<'de> for Unique {
    fn deserialize<D: de::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct Check;
        impl<'de> Visitor<'de> for Check {
            type Value = Unique;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("JSON with unique object keys")
            }
            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Unique, M::Error> {
                let mut keys = BTreeSet::new();
                while let Some(key) = map.next_key::<String>()? {
                    if !keys.insert(key) {
                        return Err(de::Error::custom("duplicate MLX config key"));
                    }
                    map.next_value::<Unique>()?;
                }
                Ok(Unique)
            }
            fn visit_seq<S: SeqAccess<'de>>(self, mut seq: S) -> Result<Unique, S::Error> {
                while seq.next_element::<Unique>()?.is_some() {}
                Ok(Unique)
            }
            fn visit_bool<E: de::Error>(self, _: bool) -> Result<Unique, E> {
                Ok(Unique)
            }
            fn visit_i64<E: de::Error>(self, _: i64) -> Result<Unique, E> {
                Ok(Unique)
            }
            fn visit_u64<E: de::Error>(self, _: u64) -> Result<Unique, E> {
                Ok(Unique)
            }
            fn visit_f64<E: de::Error>(self, _: f64) -> Result<Unique, E> {
                Ok(Unique)
            }
            fn visit_str<E: de::Error>(self, _: &str) -> Result<Unique, E> {
                Ok(Unique)
            }
            fn visit_unit<E: de::Error>(self) -> Result<Unique, E> {
                Ok(Unique)
            }
        }
        d.deserialize_any(Check)
    }
}
pub(crate) fn read_config(path: &Path) -> Result<Value, Error> {
    let io = |source| Error::Io {
        path: path.to_owned(),
        source,
    };
    let file = File::open(path).map_err(io)?;
    if file.metadata().map_err(io)?.len() > MAX_CONFIG_BYTES as u64 {
        return Err(invalid("MLX config exceeds 1 MiB"));
    }
    let mut data = Vec::new();
    // Limit again while reading in case a local file grows after stat.
    file.take((MAX_CONFIG_BYTES + 1) as u64)
        .read_to_end(&mut data)
        .map_err(io)?;
    if data.len() > MAX_CONFIG_BYTES {
        return Err(invalid("MLX config exceeds 1 MiB"));
    }
    parse_json(&data)
}

/// Shared cold metadata boundary: one bounded, duplicate-free JSON buffer.
/// Used for configs and the Gemma preparation adapter's index/header snapshots.
pub(crate) fn parse_json(data: &[u8]) -> Result<Value, Error> {
    if data.len() > MAX_CONFIG_BYTES {
        return Err(invalid("MLX metadata exceeds 1 MiB"));
    }
    serde_json::from_slice::<Unique>(data)?;
    Ok(serde_json::from_slice(data)?)
}

#[cfg(test)]
mod context_diagnostic_tests {
    use super::*;
    #[test]
    fn llama_quality_rejects_invalid_capacity_before_opening_source() {
        for context_limit in [0, 1026, usize::MAX] {
            let result = Source::open_llama3_quality_diagnostic(
                Path::new("/nonexistent-llama-quality"),
                LoadOptions {
                    context_limit,
                    memory_limit_bytes: usize::MAX,
                },
            );
            assert!(
                matches!(result, Err(Error::Invalid(ref message)) if message == "Llama quality diagnostic requires1..=1025 positions")
            );
        }
        assert!(
            matches!(Source::open_llama3_audited(Path::new("/nonexistent-llama-quality"),
            LoadOptions { context_limit: 513, memory_limit_bytes: usize::MAX }),
            Err(Error::Invalid(ref message)) if message == "audited Llama3 requires context <=512")
        );
    }
    #[cfg(unix)]
    #[test]
    #[ignore = "Explicit authenticated pinned Qwen3 source40960 metadata probe; no GPU/upload"]
    fn pinned_qwen3_context40960_source_metadata() {
        use std::io::{Seek, SeekFrom};
        use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
        type Identity = (u64, u64, u64, i64, i64, i64, i64);
        fn identity(file: &File) -> Identity {
            let m = file.metadata().unwrap();
            (
                m.dev(),
                m.ino(),
                m.len(),
                m.mtime(),
                m.mtime_nsec(),
                m.ctime(),
                m.ctime_nsec(),
            )
        }
        fn open(path: &Path) -> File {
            let file = std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
                .open(path)
                .unwrap();
            assert!(file.metadata().unwrap().is_file());
            file
        }
        fn digest(file: &mut File) -> String {
            let mut remaining = file.metadata().unwrap().len();
            assert!(remaining > 0 && remaining <= 1024 * 1024 * 1024);
            file.seek(SeekFrom::Start(0)).unwrap();
            let mut sha = Sha256::new();
            let mut bytes = [0u8; 65536];
            while remaining > 0 {
                let n = usize::try_from(remaining.min(bytes.len() as u64)).unwrap();
                file.read_exact(&mut bytes[..n]).unwrap();
                sha.update(&bytes[..n]);
                remaining -= n as u64;
            }
            assert_eq!(file.read(&mut bytes[..1]).unwrap(), 0);
            format!("{:x}", sha.finalize())
        }
        // Reuse the frozen checkpoint identities without opening validation inputs.
        const CONTRACT: &str =
            include_str!("../../benchmarks/qwen3-native-profile-v1/contract.json");
        assert_eq!(
            format!("{:x}", Sha256::digest(CONTRACT.as_bytes())),
            "05c6046c10945b637735c9a86de82d17f42372dd6e0fec1124b22c4ff47afb24"
        );
        let contract: Value = serde_json::from_str(CONTRACT).unwrap();
        assert_eq!(
            contract["model"]["repository"],
            "mlx-community/Qwen3-0.6B-8bit"
        );
        assert_eq!(
            contract["model"]["revision"],
            "11de96878523501bcaa86104e3c186de07ff9068"
        );
        let pins = contract["model"]["files"].as_object().unwrap();
        assert_eq!(pins.len(), 6);
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let directory = root.join("models/MLX-Qwen3-0.6B-8bit-11de968");
        let mut owners = Vec::with_capacity(pins.len());
        for (name, expected) in pins {
            assert_eq!(Path::new(name).file_name().unwrap(), name.as_str());
            let path = directory.join(name);
            let mut file = open(&path);
            let before = identity(&file);
            let sha = digest(&mut file);
            println!(
                "{}",
                serde_json::json!({"event":"source_pin_before","file":name,
                "identity":before,"sha256":sha})
            );
            assert_eq!(sha, expected.as_str().unwrap());
            assert_eq!(identity(&file), before);
            assert_eq!(identity(&open(&path)), before);
            owners.push((name.clone(), path, file, before, sha));
        }
        // Source validation owns bounded metadata and streaming file readers.
        // It performs no tensor expansion, GPU creation or numerical allocation.
        let result = Source::open_qwen3_context_diagnostic(
            &directory,
            LoadOptions {
                context_limit: 40960,
                memory_limit_bytes: 1536 * 1024 * 1024,
            },
        );
        let observation = match &result {
            Ok(source) => serde_json::json!({"event":"source_opened","requested":40960,
                "loaded_positions":source.config().max_positions,
                "declared_positions":source.declared_positions(),
                "architecture":format!("{:?}",source.architecture()),
                "tied_output":source.tied_output(),
                "tensor_payload_bytes":source.tensor_payload_bytes()}),
            Err(error) => {
                serde_json::json!({"event":"source_open_error","error":format!("{error:?}")})
            }
        };
        println!("{observation}");
        let passed = result.as_ref().is_ok_and(|source| {
            source.config().max_positions == 40960
                && source.declared_positions() == 40960
                && source.architecture() == Architecture::Qwen3
                && source.tied_output()
        });
        drop(result);
        // Retain every source descriptor through validation and check full bytes,
        // descriptor metadata and current path identities again even on open error.
        for (name, path, file, before, expected) in &mut owners {
            let after = identity(file);
            let sha = digest(file);
            let current = identity(&open(path));
            println!(
                "{}",
                serde_json::json!({"event":"source_pin_after","file":name,
                "before_identity":before,"after_identity":after,"path_identity":current,"sha256":sha})
            );
            assert_eq!(after, *before);
            assert_eq!(sha, *expected);
            assert_eq!(identity(file), *before);
            assert_eq!(current, *before);
        }
        assert!(
            passed,
            "pinned long-context source did not preserve requested and declared40960"
        );
        println!(
            "{}",
            serde_json::json!({"event":"source_metadata_probe_complete","passed":true,
            "requested_positions":40960,"declared_positions":40960,"loaded_positions":40960,
            "model_files_authenticated":6,"gpu_created":false,"tensor_upload":false,
            "numerical_buffers_allocated":false,"context_qualified":false,"app_capacity_changed":false})
        );
    }

    #[test]
    fn out_of_scope_context_rejects_before_source_io() {
        for capacity in [0, 40961, usize::MAX] {
            let result = Source::open_qwen3_context_diagnostic(
                Path::new("/nonexistent-qwen3-context-diagnostic"),
                LoadOptions {
                    context_limit: capacity,
                    memory_limit_bytes: weights::INITIAL_MEMORY_LIMIT_BYTES,
                },
            );
            assert!(
                matches!(result,Err(Error::Invalid(ref message)) if message=="Qwen3 context diagnostic requires1..=40960 positions")
            );
        }
    }
    #[test]
    fn tiny_dense_source_is_not_a_long_context_diagnostic_model() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mlx-qwen3-v1/tied");
        let options = LoadOptions {
            context_limit: 32,
            memory_limit_bytes: 256 * 1024 * 1024,
        };
        assert!(Source::open_dense(&path, options).is_ok());
        assert!(matches!(
            Source::open_qwen3_context_diagnostic(&path, options),
            Err(Error::Invalid(_))
        ));
    }
}

#[cfg(test)]
#[path = "qwen/content_tests.rs"]
mod content_tests;

#[cfg(test)]
#[path = "qwen/row_tests.rs"]
mod row_tests;

#[cfg(test)]
#[path = "qwen/logical_tests.rs"]
pub(crate) mod logical_tests;

#[cfg(test)]
#[path = "qwen/qwen3_content_tests.rs"]
mod qwen3_content_tests;

#[cfg(test)]
#[path = "qwen/qwen2_content_tests.rs"]
mod qwen2_content_tests;

#[cfg(test)]
#[path = "qwen/mistral_content_tests.rs"]
mod mistral_content_tests;
