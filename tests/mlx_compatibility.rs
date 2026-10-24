//! Offline metadata/admission contracts; real checkpoint inspection is explicit.
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};
use toklet::{
    mlx::{
        compatibility::{self, Backend, LoadAdmission, Reason, Status},
        reference,
    },
    weights::{self, LoadOptions},
};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}
fn options() -> LoadOptions {
    LoadOptions {
        context_limit: 32,
        memory_limit_bytes: 128 * 1024 * 1024,
    }
}
fn fixture(name: &str) -> PathBuf {
    root().join("tests/fixtures").join(name)
}
struct Temporary(PathBuf);
impl Temporary {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "toklet-compatibility-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn copy(name: &str) -> Self {
        let result = Self::new();
        for entry in fs::read_dir(fixture(name)).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_file() {
                fs::copy(entry.path(), result.0.join(entry.file_name())).unwrap();
            }
        }
        result
    }
    fn config(&self, value: Value) {
        fs::write(
            self.0.join("config.json"),
            serde_json::to_vec(&value).unwrap(),
        )
        .unwrap();
    }
}
impl Drop for Temporary {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn supported_families_share_loader_metadata_and_scalar_load_estimates() {
    for (path, architecture) in [
        ("mlx-qwen-v1/model", "qwen2"),
        ("mlx-llama-v1/tied", "llama"),
        ("mlx-llama-v1/untied", "llama"),
        ("mlx-mistral-v1/untied", "mistral"),
        ("mlx-qwen3-v1/tied", "qwen3"),
        ("mlx-qwen3-v1/untied", "qwen3"),
    ] {
        let report = compatibility::inspect(&fixture(path), Backend::ScalarReference, options());
        assert_eq!(
            report.status,
            Status::CompatibleUntested,
            "{}",
            report.message
        );
        assert_eq!(report.architecture, Some(architecture));
        assert_eq!(report.reason, Reason::MetadataAccepted);
        assert!(report.tensor_metadata_validated && report.payload_validation_required);
        assert!(!report.text_profile_checked && !report.qualification_evaluated);
        assert!(report.session_admission_required && !report.runtime_device_checks_required);
        let resource = report.resources.unwrap();
        let loaded = reference::load_dense(&fixture(path), options()).unwrap();
        assert_eq!(
            resource.scalar_fp32_weights_bytes,
            Some(loaded.fp32_weights_bytes)
        );
        assert_eq!(
            resource.scalar_temporary_packed_bytes,
            Some(loaded.temporary_packed_bytes)
        );
        assert_eq!(
            resource.scalar_admitted_load_bytes,
            Some(loaded.admitted_load_bytes)
        );
        assert_eq!(report.load_admission, LoadAdmission::FitsScalarLoadEstimate);
        let exact = loaded.admitted_load_bytes;
        let below = LoadOptions {
            memory_limit_bytes: exact - 1,
            ..options()
        };
        let rejected = compatibility::inspect(&fixture(path), Backend::ScalarReference, below);
        assert_eq!(rejected.status, Status::CompatibleUntested);
        assert_eq!(rejected.load_admission, LoadAdmission::ExceedsBudget);
        assert_eq!(rejected.reason, Reason::LoadBudgetExceeded);
        assert!(
            matches!(reference::load_dense(&fixture(path),below),Err(weights::Error::Budget{ required,limit }) if required==exact && limit==exact-1)
        );
        assert!(
            reference::load_dense(
                &fixture(path),
                LoadOptions {
                    memory_limit_bytes: exact,
                    ..options()
                }
            )
            .is_ok()
        );
    }
}

#[test]
fn metal_report_never_infers_padding_device_session_or_qualification() {
    for (path, architecture) in [
        ("mlx-qwen-v1/model", "qwen2"),
        ("mlx-llama-v1/tied", "llama"),
        ("mlx-mistral-v1/untied", "mistral"),
        ("mlx-qwen3-v1/tied", "qwen3"),
    ] {
        let report = compatibility::inspect(&fixture(path), Backend::Metal, options());
        assert_eq!(
            report.status,
            Status::CompatibleUntested,
            "{}",
            report.message
        );
        assert_eq!(report.architecture, Some(architecture));
        assert_eq!(report.load_admission, LoadAdmission::DevicePlanningRequired);
        assert!(
            report.runtime_device_checks_required
                && report.payload_validation_required
                && report.session_admission_required
        );
        assert!(!report.qualification_evaluated && !report.text_profile_checked);
        let expected_build = cfg!(all(
            feature = "metal",
            any(target_os = "macos", target_os = "ios"),
            target_arch = "aarch64"
        ));
        assert_eq!(report.backend_available_in_build, expected_build);
        assert_eq!(
            report.reason,
            if expected_build {
                Reason::MetadataAccepted
            } else {
                Reason::BackendUnavailable
            }
        );
        let resources = report.resources.unwrap();
        assert!(resources.metal_runtime_planning_required);
        assert!(resources.scalar_admitted_load_bytes.is_none());
    }
    let report = compatibility::inspect(&fixture("mlx-qwen3-v1/tied"), Backend::Metal, options());
    let geometry = report.geometry.unwrap();
    assert_eq!(
        (
            geometry.hidden_size,
            geometry.query_width,
            geometry.kv_width
        ),
        (32, 64, 32)
    );
    assert_eq!(
        report
            .representations
            .iter()
            .map(|group| group.logical_tensors)
            .sum::<usize>(),
        24
    );
    assert_eq!(
        report
            .representations
            .iter()
            .filter(|group| group.storage == "mlx_affine")
            .map(|group| group.logical_tensors)
            .sum::<usize>(),
        14
    );
    let bytes = report
        .representations
        .iter()
        .map(|group| group.source_payload_bytes)
        .sum::<usize>();
    assert_eq!(bytes, report.resources.unwrap().source_payload_bytes);
}

#[test]
fn unknown_architecture_is_not_confused_with_malformed_configuration() {
    let folder = Temporary::new();
    folder.config(json!({"model_type":"some_unimplemented_graph"}));
    let unknown = compatibility::inspect(&folder.0, Backend::Metal, options());
    assert_eq!(unknown.status, Status::Unknown);
    assert_eq!(unknown.reason, Reason::UnknownArchitecture);
    assert_eq!(
        unknown.declared_model_type.as_deref(),
        Some("some_unimplemented_graph")
    );
    assert!(!unknown.tensor_metadata_validated && unknown.resources.is_none());
    for raw in [
        b"{}".as_slice(),
        b"[]",
        b"{\"model_type\":1}",
        b"{\"model_type\":\"\"}",
        b"{\"model_type\":\"unknown\",\"model_type\":\"qwen3\"}",
        b"{\"model_type\":\"unknown\",\"x\":{\"a\":1,\"a\":2}}",
    ] {
        fs::write(folder.0.join("config.json"), raw).unwrap();
        let report = compatibility::inspect(&folder.0, Backend::ScalarReference, options());
        assert_eq!(report.status, Status::Unsupported);
        assert_eq!(report.reason, Reason::InvalidConfiguration);
    }
    folder.config(json!({"model_type":"x".repeat(129)}));
    assert_eq!(
        compatibility::inspect(&folder.0, Backend::Metal, options()).reason,
        Reason::InvalidConfiguration
    );
    fs::write(folder.0.join("config.json"), vec![b' '; 1024 * 1024 + 1]).unwrap();
    assert_eq!(
        compatibility::inspect(&folder.0, Backend::Metal, options()).reason,
        Reason::InvalidConfiguration
    );
}

#[test]
fn requested_bounds_reject_before_io_and_context_is_separate_from_source() {
    let missing = Path::new("/a-model-that-does-not-exist");
    for invalid in [
        LoadOptions {
            context_limit: 0,
            ..options()
        },
        LoadOptions {
            context_limit: 32769,
            ..options()
        },
        LoadOptions {
            memory_limit_bytes: 0,
            ..options()
        },
        LoadOptions {
            memory_limit_bytes: weights::INITIAL_MEMORY_LIMIT_BYTES + 1,
            ..options()
        },
    ] {
        assert_eq!(
            compatibility::inspect(missing, Backend::Metal, invalid).reason,
            Reason::InvalidRequest
        );
    }
    let report = compatibility::inspect(
        &fixture("mlx-qwen3-v1/tied"),
        Backend::ScalarReference,
        LoadOptions {
            context_limit: 16,
            ..options()
        },
    );
    assert_eq!(report.configured_max_positions, Some(32));
    assert_eq!(report.requested_context, 16);
    assert_eq!(report.status, Status::CompatibleUntested);
    let over = compatibility::inspect(
        &fixture("mlx-qwen3-v1/tied"),
        Backend::ScalarReference,
        LoadOptions {
            context_limit: 33,
            ..options()
        },
    );
    assert_eq!(over.configured_max_positions, Some(32));
    assert_eq!(over.requested_context, 33);
    assert_eq!(over.reason, Reason::ArtifactRejected);
}

#[test]
fn unsupported_settings_and_missing_tensors_never_grant_metadata_acceptance() {
    let folder = Temporary::copy("mlx-qwen3-v1/tied");
    let mut config: Value =
        serde_json::from_slice(&fs::read(folder.0.join("config.json")).unwrap()).unwrap();
    config["rope_scaling"] = json!({"type":"linear","factor":2});
    folder.config(config);
    let report = compatibility::inspect(&folder.0, Backend::Metal, options());
    assert_eq!(report.status, Status::Unsupported);
    assert_eq!(report.reason, Reason::ArtifactRejected);
    assert!(!report.tensor_metadata_validated);
    let folder = Temporary::copy("mlx-qwen3-v1/tied");
    for entry in fs::read_dir(&folder.0).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|ext| ext == "safetensors") {
            fs::remove_file(path).unwrap();
        }
    }
    assert_eq!(
        compatibility::inspect(&folder.0, Backend::Metal, options()).reason,
        Reason::ArtifactRejected
    );
}

#[test]
fn overflowing_public_configuration_rejects_before_geometry_reporting() {
    let folder = Temporary::copy("mlx-qwen3-v1/tied");
    let mut config: Value =
        serde_json::from_slice(&fs::read(folder.0.join("config.json")).unwrap()).unwrap();
    config["num_attention_heads"] = json!(usize::MAX);
    config["num_key_value_heads"] = json!(1);
    config["head_dim"] = json!(2);
    folder.config(config);
    let report = compatibility::inspect(&folder.0, Backend::ScalarReference, options());
    assert_eq!(report.status, Status::Unsupported);
    assert_eq!(report.reason, Reason::ArtifactRejected);
    assert!(report.geometry.is_none());
}

#[test]
fn nan_payload_is_not_silently_qualified_by_header_inspection() {
    let folder = Temporary::copy("mlx-qwen3-v1/tied");
    let path = fs::read_dir(&folder.0)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
        .unwrap();
    let mut data = fs::read(&path).unwrap();
    let length = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&data[8..8 + length]).unwrap();
    let tensor = &header["model.norm.weight"];
    assert_eq!(tensor["dtype"], "F32");
    let offset = 8 + length + tensor["data_offsets"][0].as_u64().unwrap() as usize;
    data[offset..offset + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    fs::write(&path, data).unwrap();
    let report = compatibility::inspect(&folder.0, Backend::ScalarReference, options());
    assert_eq!(report.status, Status::CompatibleUntested);
    assert!(report.tensor_metadata_validated && report.payload_validation_required);
    assert!(!report.qualification_evaluated);
    assert!(reference::load_dense(&folder.0, options()).is_err());
}

#[test]
fn report_json_uses_explicit_stable_states_not_a_fit_or_quality_boolean() {
    let report = compatibility::inspect(
        &fixture("mlx-qwen3-v1/tied"),
        Backend::Metal,
        LoadOptions {
            memory_limit_bytes: 1,
            ..options()
        },
    );
    let value = serde_json::to_value(report).unwrap();
    assert_eq!(value["schema_version"], 1);
    assert_eq!(value["status"], "compatible_untested");
    assert_eq!(value["reason"], "load_budget_exceeded");
    assert_eq!(value["load_admission"], "exceeds_budget");
    assert_eq!(value["qualification_evaluated"], false);
    assert_eq!(value["text_profile_checked"], false);
}

#[cfg(all(
    feature = "metal",
    any(target_os = "macos", target_os = "ios"),
    target_arch = "aarch64"
))]
#[test]
fn rejected_geometry_matches_actual_loader_without_creating_a_gpu() {
    for name in [
        "mlx-qwen-v1/model",
        "mlx-llama-v1/tied",
        "mlx-mistral-v1/untied",
        "mlx-qwen3-v1/tied",
    ] {
        let folder = Temporary::copy(name);
        let mut config: Value =
            serde_json::from_slice(&fs::read(folder.0.join("config.json")).unwrap()).unwrap();
        config["rope_theta"] = json!(12345);
        folder.config(config);
        let report = compatibility::inspect(&folder.0, Backend::Metal, options());
        assert_eq!(report.reason, Reason::UnsupportedGeometry);
        let error = match toklet::metal::Model::load_mlx(&folder.0, options()) {
            Ok(_) => panic!("Unqualified geometry must reject before GPU creation"),
            Err(error) => error,
        };
        assert!(matches!(error, toklet::metal::Error::Unsupported(_)));
        assert_eq!(report.message, error.to_string());
    }
}

#[test]
#[ignore = "Explicit local real Qwen3 checkpoint metadata probe; no GPU or inference"]
fn real_qwen3_metadata_probe_reports_bf16_without_qualifying_execution() {
    let path = root().join("models/MLX-Qwen3-0.6B-8bit-11de968");
    let report = compatibility::inspect(
        &path,
        Backend::Metal,
        LoadOptions {
            context_limit: 512,
            memory_limit_bytes: 1536 * 1024 * 1024,
        },
    );
    assert_eq!(
        report.status,
        Status::CompatibleUntested,
        "{}",
        report.message
    );
    assert_eq!(report.architecture, Some("qwen3"));
    assert_eq!(report.configured_max_positions, Some(40960));
    assert_eq!(report.requested_context, 512);
    assert_eq!(report.geometry.as_ref().unwrap().query_width, 2048);
    assert_eq!(
        report.resources.as_ref().unwrap().source_payload_bytes,
        633364480
    );
    assert_eq!(report.representations.len(), 2);
    assert_eq!(
        report
            .representations
            .iter()
            .map(|group| group.logical_tensors)
            .sum::<usize>(),
        310
    );
    assert!(
        report
            .representations
            .iter()
            .all(|group| group.dtype == "bf16")
    );
    assert!(report.payload_validation_required && report.runtime_device_checks_required);
    assert_eq!(report.load_admission, LoadAdmission::DevicePlanningRequired);
    assert!(!report.qualification_evaluated);
    println!(
        "MLX_COMPATIBILITY_REAL {}",
        serde_json::to_string(&report).unwrap()
    );
}
