//! Cold local MLX metadata inspection shared by future CLI/native/Hub callers.
//! No model expansion, GPU creation, tokenizer or qualification is performed.
use super::{
    affine::MetadataDtype,
    dense::Architecture,
    execution,
    qwen::{self, Source, Storage},
    reference,
};
use crate::{
    model::Config,
    weights::{self, LoadOptions},
};
use serde::Serialize;
use std::{collections::BTreeMap, path::Path};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    ScalarReference,
    Metal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Unknown,
    Unsupported,
    CompatibleUntested,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Reason {
    InvalidRequest,
    InvalidConfiguration,
    UnknownArchitecture,
    ArtifactRejected,
    UnsupportedGeometry,
    LoadBudgetExceeded,
    BackendUnavailable,
    MetadataAccepted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadAdmission {
    NotEvaluated,
    ExceedsBudget,
    FitsScalarLoadEstimate,
    DevicePlanningRequired,
}

#[derive(Debug, Serialize)]
pub struct Geometry {
    pub hidden_size: usize,
    pub query_width: usize,
    pub kv_width: usize,
    pub intermediate_size: usize,
    pub layers: usize,
    pub vocabulary: usize,
    pub head_dim: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
}
impl Geometry {
    fn validated(c: &Config) -> Self {
        // Config::validate has already checked these products at admission.
        Self {
            hidden_size: c.hidden_size,
            query_width: c.query_heads * c.head_dim,
            kv_width: c.kv_heads * c.head_dim,
            intermediate_size: c.intermediate_size,
            layers: c.num_layers,
            vocabulary: c.vocab_size,
            head_dim: c.head_dim,
            query_heads: c.query_heads,
            kv_heads: c.kv_heads,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Representation {
    pub storage: &'static str,
    /// Floating tensor dtype, or scale/bias metadata dtype for MLX affine storage.
    /// MLX affine packed matrix words are U32; `bits` is their element bit width.
    pub dtype: &'static str,
    pub bits: Option<u8>,
    pub group_size: Option<usize>,
    pub logical_tensors: usize,
    pub source_payload_bytes: usize,
}

#[derive(Debug, Serialize)]
pub struct Resources {
    /// Owned source payloads, counting tied embedding/output only once.
    pub source_payload_bytes: usize,
    /// Existing Source admission: payloads + 64 MiB metadata + 64 KiB stream.
    pub source_admitted_load_bytes: usize,
    /// ScalarReference only: exact existing expansion estimate, excluding KV/UI.
    pub scalar_fp32_weights_bytes: Option<usize>,
    pub scalar_temporary_packed_bytes: Option<usize>,
    pub scalar_admitted_load_bytes: Option<usize>,
    /// Metal stride/padding depends on MPS/device; never infer it from source.
    pub metal_runtime_planning_required: bool,
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub schema_version: u32,
    pub backend: Backend,
    pub status: Status,
    pub reason: Reason,
    pub message: String,
    pub architecture: Option<&'static str>,
    pub declared_model_type: Option<String>,
    pub configured_max_positions: Option<usize>,
    pub requested_context: usize,
    pub requested_memory_limit_bytes: usize,
    pub geometry: Option<Geometry>,
    pub tied_output: Option<bool>,
    pub representations: Vec<Representation>,
    pub resources: Option<Resources>,
    pub tensor_metadata_validated: bool,
    pub backend_available_in_build: bool,
    pub load_admission: LoadAdmission,
    pub payload_validation_required: bool,
    pub runtime_device_checks_required: bool,
    pub session_admission_required: bool,
    pub text_profile_checked: bool,
    /// This probe never evaluates a qualification receipt or upgrades a label.
    pub qualification_evaluated: bool,
}
impl Report {
    fn rejected(
        backend: Backend,
        options: LoadOptions,
        reason: Reason,
        message: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: 1,
            backend,
            status: Status::Unsupported,
            reason,
            message: message.into(),
            architecture: None,
            declared_model_type: None,
            configured_max_positions: None,
            requested_context: options.context_limit,
            requested_memory_limit_bytes: options.memory_limit_bytes,
            geometry: None,
            tied_output: None,
            representations: Vec::new(),
            resources: None,
            tensor_metadata_validated: false,
            backend_available_in_build: backend == Backend::ScalarReference
                || cfg!(all(
                    feature = "metal",
                    any(target_os = "macos", target_os = "ios"),
                    target_arch = "aarch64"
                )),
            load_admission: LoadAdmission::NotEvaluated,
            payload_validation_required: true,
            runtime_device_checks_required: backend == Backend::Metal,
            session_admission_required: true,
            text_profile_checked: false,
            qualification_evaluated: false,
        }
    }
}

fn dtype(dtype: MetadataDtype) -> &'static str {
    match dtype {
        MetadataDtype::F16 => "f16",
        MetadataDtype::BF16 => "bf16",
        MetadataDtype::F32 => "f32",
    }
}
fn name(architecture: Architecture) -> &'static str {
    match architecture {
        Architecture::Qwen2 => "qwen2",
        Architecture::Qwen3 => "qwen3",
        Architecture::Llama => "llama",
        Architecture::Mistral => "mistral",
    }
}

/// Inspect a local artifact with the loader's exact config/tensor rules.
/// Size/count bounds come from the existing parser. Source may stream a
/// redundant tied head for identity validation; arbitrary payload finiteness and
/// Metal value-domain/device checks still occur only during numerical loading.
/// Compatibility, source/scalar load estimates and runtime fit are separate.
pub fn inspect(directory: &Path, backend: Backend, options: LoadOptions) -> Report {
    let mut report = Report::rejected(
        backend,
        options,
        Reason::InvalidRequest,
        "Context and memory request are outside engine bounds",
    );
    if options.context_limit == 0
        || options.context_limit > weights::MAX_CONTEXT_LIMIT
        || options.memory_limit_bytes == 0
        || options.memory_limit_bytes > weights::INITIAL_MEMORY_LIMIT_BYTES
    {
        return report;
    }
    let document = match qwen::read_config(&directory.join("config.json")) {
        Ok(document) => document,
        Err(error) => {
            report.reason = Reason::InvalidConfiguration;
            report.message = error.to_string();
            return report;
        }
    };
    report.configured_max_positions = document
        .get("max_position_embeddings")
        .and_then(|v| v.as_u64())
        .and_then(|v| usize::try_from(v).ok());
    let model_type = match document.get("model_type").and_then(|v| v.as_str()) {
        Some(value) if !value.is_empty() && value.len() <= 128 => value,
        _ => {
            report.reason = Reason::InvalidConfiguration;
            report.message = "A bounded, nonempty model_type is required".into();
            return report;
        }
    };
    report.declared_model_type = Some(model_type.to_owned());
    if !["qwen2", "qwen3", "llama", "mistral"].contains(&model_type) {
        report.status = Status::Unknown;
        report.reason = Reason::UnknownArchitecture;
        report.message = "The declared architecture has no native MLX decoder".into();
        return report;
    }
    // Inspect metadata against the engine ceiling to report a smaller caller
    // budget separately; this does not allocate/expand the source payloads.
    let source = match Source::from_inspected_config(
        directory,
        LoadOptions {
            memory_limit_bytes: weights::INITIAL_MEMORY_LIMIT_BYTES,
            ..options
        },
        document,
    ) {
        Ok(source) => source,
        Err(error) => {
            if matches!(error, weights::Error::Budget { .. }) {
                report.reason = Reason::LoadBudgetExceeded;
                report.load_admission = LoadAdmission::ExceedsBudget;
            } else {
                report.reason = Reason::ArtifactRejected;
            }
            report.message = error.to_string();
            return report;
        }
    };
    report.architecture = Some(name(source.architecture()));
    report.geometry = Some(Geometry::validated(source.config()));
    report.tied_output = Some(source.tied_output());
    report.tensor_metadata_validated = true;
    let mut groups = BTreeMap::new();
    for logical in source.logical_tensors().values() {
        let spec = logical.tensor();
        let key = match spec.storage() {
            Storage::Float(kind) => ("floating", dtype(kind), None, None),
            Storage::Affine(layout) => (
                "mlx_affine",
                dtype(layout.metadata_dtype()),
                Some(layout.bits()),
                Some(layout.group_size()),
            ),
        };
        let group = groups.entry(key).or_insert(Representation {
            storage: key.0,
            dtype: key.1,
            bits: key.2,
            group_size: key.3,
            logical_tensors: 0,
            source_payload_bytes: 0,
        });
        group.logical_tensors += 1;
        let Some(total) = group.source_payload_bytes.checked_add(spec.payload_bytes()) else {
            report.reason = Reason::ArtifactRejected;
            report.message = "Representation size overflow".into();
            return report;
        };
        group.source_payload_bytes = total;
    }
    report.representations = groups.into_values().collect();
    let mut resources = Resources {
        source_payload_bytes: source.tensor_payload_bytes(),
        source_admitted_load_bytes: source.admitted_load_bytes(),
        scalar_fp32_weights_bytes: None,
        scalar_temporary_packed_bytes: None,
        scalar_admitted_load_bytes: None,
        metal_runtime_planning_required: backend == Backend::Metal,
    };
    if backend == Backend::Metal {
        if let Err(error) = execution::metal_source_geometry(&source) {
            report.reason = Reason::UnsupportedGeometry;
            report.message = error.message().into();
            report.resources = Some(resources);
            return report;
        }
    } else {
        let plan = match reference::load_plan(&source) {
            Ok(plan) => plan,
            Err(error) => {
                report.reason = Reason::ArtifactRejected;
                report.message = error.to_string();
                return report;
            }
        };
        resources.scalar_fp32_weights_bytes = Some(plan.fp32_weights_bytes);
        resources.scalar_temporary_packed_bytes = Some(plan.temporary_packed_bytes);
        resources.scalar_admitted_load_bytes = Some(plan.admitted_load_bytes);
    }
    let required = resources
        .scalar_admitted_load_bytes
        .unwrap_or(resources.source_admitted_load_bytes);
    report.status = Status::CompatibleUntested;
    if required > options.memory_limit_bytes {
        report.reason = Reason::LoadBudgetExceeded;
        report.load_admission = LoadAdmission::ExceedsBudget;
        report.message = "The existing loader estimate exceeds the requested memory limit".into();
    } else {
        report.reason = if report.backend_available_in_build {
            Reason::MetadataAccepted
        } else {
            Reason::BackendUnavailable
        };
        report.load_admission = if backend == Backend::Metal {
            LoadAdmission::DevicePlanningRequired
        } else {
            LoadAdmission::FitsScalarLoadEstimate
        };
        report.message = if report.backend_available_in_build {
            "Configuration and tensor metadata are admitted; payload/session/text and qualification checks remain"
        } else {
            "Configuration and tensor metadata are admitted, but this backend is not compiled for the current target"
        }.into();
    }
    report.resources = Some(resources);
    report
}
