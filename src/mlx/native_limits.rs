//! Cold ordinary native MLX context facts, using actual loader admission policy.
//! A snapshot is not a payload identity, load permission, budget fit or qualification.
use super::{
    compatibility::{Reason, Status},
    dense::Architecture,
    execution,
    qwen::{self, Source},
};
use crate::weights::{self, LoadOptions};
use serde::Serialize;
use std::path::Path;

/// These correspond to existing distinct native load entrypoints. Diagnostic
/// Qwen3, specialized Gemma/LFM and scalar-reference routes are not selectable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadProfile {
    Ordinary,
    SmolAudited,
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub schema_version: u32,
    pub report_kind: &'static str,
    pub load_profile: LoadProfile,
    pub status: Status,
    pub reason: Reason,
    pub message: String,
    pub architecture: Option<&'static str>,
    /// Present only after the strict source parser validated this declaration.
    pub declared_total_positions: Option<usize>,
    pub engine_global_position_ceiling: usize,
    /// Phi3 declares a window that bounds its implemented full-attention prefix.
    pub declared_sliding_window: Option<usize>,
    /// None means no extra graph cap, or that metadata was not accepted. Consult
    /// status/tensor_metadata_validated; never substitute the global ceiling.
    pub graph_position_cap: Option<usize>,
    /// Min(declaration, global ceiling, optional graph cap/window), after every
    /// ordinary source/geometry check succeeds. Not a loaded session capacity.
    pub factory_position_upper: Option<usize>,
    pub backend_available_in_build: bool,
    pub tensor_metadata_validated: bool,
    /// Audited profile only: authenticated config/index/header, not complete
    /// consumed payload evidence. Source::content_identity remains unpublished.
    pub audited_source_metadata_validated: bool,
    pub complete_source_authentication_required: bool,
    pub payload_validation_required: bool,
    pub runtime_device_checks_required: bool,
    pub session_admission_required: bool,
    pub budget_fit_evaluated: bool,
    pub qualification_evaluated: bool,
    pub text_profile_checked: bool,
    pub whole_app_device_fit_evaluated: bool,
    /// The existing metadata-reader admission ceiling, not a caller budget or
    /// a claim that the actual padded resident weights fit it.
    pub metadata_reader_limit_bytes: usize,
}
impl Report {
    fn rejected(profile: LoadProfile) -> Self {
        Self {
            schema_version: 1,
            report_kind: "native_mlx_context_limits",
            load_profile: profile,
            status: Status::Unsupported,
            reason: Reason::ArtifactRejected,
            message: String::new(),
            architecture: None,
            declared_total_positions: None,
            engine_global_position_ceiling: weights::MAX_CONTEXT_LIMIT,
            declared_sliding_window: None,
            graph_position_cap: None,
            factory_position_upper: None,
            backend_available_in_build: cfg!(all(
                feature = "metal",
                any(target_os = "macos", target_os = "ios"),
                target_arch = "aarch64"
            )),
            tensor_metadata_validated: false,
            audited_source_metadata_validated: false,
            complete_source_authentication_required: profile == LoadProfile::SmolAudited,
            payload_validation_required: true,
            runtime_device_checks_required: true,
            session_admission_required: true,
            budget_fit_evaluated: false,
            qualification_evaluated: false,
            text_profile_checked: false,
            whole_app_device_fit_evaluated: false,
            metadata_reader_limit_bytes: weights::INITIAL_MEMORY_LIMIT_BYTES,
        }
    }
}

/// Read bounded config/header/index metadata without model, GPU or session
/// creation. As in existing Source inspection, redundant tied heads may be
/// streamed for equality. No whole payload is allocated or authenticated here.
/// A seed context of one avoids requiring a caller to guess an admissible upper;
/// it is never published as a loaded position limit. Actual load revalidates.
pub fn inspect(directory: &Path, profile: LoadProfile) -> Report {
    let mut report = Report::rejected(profile);
    let options = LoadOptions {
        context_limit: 1,
        memory_limit_bytes: weights::INITIAL_MEMORY_LIMIT_BYTES,
    };
    let source = match profile {
        LoadProfile::Ordinary => {
            let document = match qwen::read_config(&directory.join("config.json")) {
                Ok(document) => document,
                Err(error) => {
                    report.reason = Reason::InvalidConfiguration;
                    report.message = error.to_string();
                    return report;
                }
            };
            match document.get("model_type").and_then(|value| value.as_str()) {
                Some("qwen2" | "qwen3" | "llama" | "mistral" | "phi3") => {}
                Some(name) if !name.is_empty() && name.len() <= 128 => {
                    report.status = Status::Unknown;
                    report.reason = Reason::UnknownArchitecture;
                    report.message = "No ordinary native MLX decoder for this architecture".into();
                    return report;
                }
                _ => {
                    report.reason = Reason::InvalidConfiguration;
                    report.message = "A bounded, nonempty model_type is required".into();
                    return report;
                }
            }
            // Consume the same document classified above, avoiding a second
            // config read with potentially different declaration/architecture.
            Source::from_inspected_config(directory, options, document)
        }
        LoadProfile::SmolAudited => Source::open_smol_audited(directory, options),
    };
    let source = match source {
        Ok(source) => source,
        Err(error) => {
            report.message = error.to_string();
            if matches!(error, weights::Error::Budget { .. }) {
                report.reason = Reason::LoadBudgetExceeded;
            }
            return report;
        }
    };
    report.architecture = Some(match source.architecture() {
        Architecture::Qwen2 => "qwen2",
        Architecture::Qwen3 => "qwen3",
        Architecture::Llama => "llama",
        Architecture::Mistral => "mistral",
        Architecture::Phi3 => "phi3",
        Architecture::SmolLm3 => "smollm3",
    });
    report.declared_total_positions = Some(source.declared_positions());
    report.tensor_metadata_validated = true;
    report.audited_source_metadata_validated = profile == LoadProfile::SmolAudited;
    let limits = match execution::metal_source_position_limits(&source) {
        Ok(limits) => limits,
        Err(error) => {
            report.reason = Reason::UnsupportedGeometry;
            report.message = error.message().into();
            return report;
        }
    };
    report.graph_position_cap = limits.graph_cap;
    report.declared_sliding_window = limits.declared_window;
    report.factory_position_upper = Some(limits.upper);
    report.status = Status::CompatibleUntested;
    report.reason = if report.backend_available_in_build {
        Reason::MetadataAccepted
    } else {
        Reason::BackendUnavailable
    };
    report.message = "Ordinary native context policy accepted; actual loading must revalidate payload, profile, budgets and runtime requirements".into();
    report
}
