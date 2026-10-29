//! Download-independent context facts. No model, tokenizer or GPU is constructed.
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};
use toklet::{
    mlx::{
        compatibility::{self, Backend, Reason, Status},
        native_limits::{self, LoadProfile},
    },
    weights::{self, LoadOptions},
};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}
struct Temporary(PathBuf);
impl Temporary {
    fn copy(name: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "toklet-native-limits-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        for item in fs::read_dir(fixture(name)).unwrap() {
            let item = item.unwrap();
            if item.file_type().unwrap().is_file() {
                fs::copy(item.path(), path.join(item.file_name())).unwrap();
            }
        }
        Self(path)
    }
    fn change(&self, name: &str, value: Value) {
        let file = self.0.join("config.json");
        let mut config: Value = serde_json::from_slice(&fs::read(&file).unwrap()).unwrap();
        config[name] = value;
        fs::write(file, serde_json::to_vec(&config).unwrap()).unwrap();
    }
}
impl Drop for Temporary {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn accepted(path: &Path, declaration: usize, cap: Option<usize>, upper: usize) {
    let report = native_limits::inspect(path, LoadProfile::Ordinary);
    assert_eq!(
        report.status,
        Status::CompatibleUntested,
        "{}",
        report.message
    );
    assert_eq!(report.declared_total_positions, Some(declaration));
    assert_eq!(report.graph_position_cap, cap);
    assert_eq!(report.factory_position_upper, Some(upper));
    assert_eq!(report.engine_global_position_ceiling, 32768);
    assert!(report.tensor_metadata_validated && report.payload_validation_required);
    assert!(report.runtime_device_checks_required && report.session_admission_required);
    assert!(!report.audited_source_metadata_validated);
    assert!(!report.complete_source_authentication_required);
    assert!(!report.budget_fit_evaluated && !report.qualification_evaluated);
    assert!(!report.text_profile_checked && !report.whole_app_device_fit_evaluated);
    let available = cfg!(all(
        feature = "metal",
        any(target_os = "macos", target_os = "ios"),
        target_arch = "aarch64"
    ));
    assert_eq!(report.backend_available_in_build, available);
    assert_eq!(
        report.reason,
        if available {
            Reason::MetadataAccepted
        } else {
            Reason::BackendUnavailable
        }
    );
    // The ordinary loader/planner's metadata path accepts this exact upper and
    // rejects one beyond it. This checks the report against the shared validator.
    let inspect = |context_limit| {
        compatibility::inspect(
            path,
            Backend::Metal,
            LoadOptions {
                context_limit,
                memory_limit_bytes: weights::INITIAL_MEMORY_LIMIT_BYTES,
            },
        )
    };
    assert_eq!(inspect(upper).status, Status::CompatibleUntested);
    assert_ne!(inspect(upper + 1).status, Status::CompatibleUntested);
}

#[test]
fn native_limits_preserve_declaration_and_exact_ordinary_boundaries() {
    for (name, cap) in [
        ("mlx-qwen3-v1/tied", Some(32)),
        ("mlx-mistral-v1/untied", Some(32)),
        ("mlx-llama-v1/tied", Some(512)),
        ("mlx-phi3-v1/tied", Some(512)),
        ("mlx-qwen-v1/model", None),
    ] {
        let copy = Temporary::copy(name);
        if name.contains("phi3") {
            copy.change("sliding_window", Value::Null);
        }
        for declared in [7usize, 32, 513, 40960] {
            copy.change("max_position_embeddings", json!(declared));
            let upper = declared.min(32768).min(cap.unwrap_or(usize::MAX));
            accepted(&copy.0, declared, cap, upper);
        }
    }
}

#[test]
fn phi3_window_is_an_enforced_prefix_bound_not_a_ring_cache_claim() {
    let copy = Temporary::copy("mlx-phi3-v1/tied");
    for window in [7usize, 511, 512, 2047] {
        copy.change("sliding_window", json!(window));
        let report = native_limits::inspect(&copy.0, LoadProfile::Ordinary);
        assert_eq!(report.declared_sliding_window, Some(window));
        accepted(&copy.0, 4096, Some(512), window.min(512));
    }
    copy.change("sliding_window", json!(0));
    let rejected = native_limits::inspect(&copy.0, LoadProfile::Ordinary);
    assert_eq!(rejected.status, Status::Unsupported);
    assert!(rejected.factory_position_upper.is_none());
}

#[test]
fn limits_do_not_promote_scalar_geometry_or_specialized_diagnostic_routes() {
    let copy = Temporary::copy("mlx-qwen3-v1/tied");
    copy.change("rope_theta", json!(12345.0));
    let scalar = compatibility::inspect(
        &copy.0,
        Backend::ScalarReference,
        LoadOptions {
            context_limit: 32,
            memory_limit_bytes: weights::INITIAL_MEMORY_LIMIT_BYTES,
        },
    );
    assert_eq!(scalar.status, Status::CompatibleUntested);
    let native = native_limits::inspect(&copy.0, LoadProfile::Ordinary);
    assert_eq!(native.reason, Reason::UnsupportedGeometry);
    assert!(native.factory_position_upper.is_none());
    assert!(native.tensor_metadata_validated);
    for family in ["gemma3", "lfm2", "unknown_decoder"] {
        copy.change("model_type", json!(family));
        let report = native_limits::inspect(&copy.0, LoadProfile::Ordinary);
        assert_eq!(report.status, Status::Unknown);
        assert_eq!(report.reason, Reason::UnknownArchitecture);
        assert!(report.factory_position_upper.is_none());
        assert!(!report.tensor_metadata_validated);
    }
}

#[test]
fn rejected_metadata_never_publishes_a_guessed_upper_or_audited_identity() {
    let copy = Temporary::copy("mlx-qwen3-v1/tied");
    let ordinary = native_limits::inspect(&copy.0, LoadProfile::Ordinary);
    assert_eq!(ordinary.factory_position_upper, Some(32));
    let audited = native_limits::inspect(&copy.0, LoadProfile::SmolAudited);
    assert_eq!(audited.load_profile, LoadProfile::SmolAudited);
    assert_eq!(audited.status, Status::Unsupported);
    assert!(audited.factory_position_upper.is_none());
    assert!(audited.complete_source_authentication_required);
    assert!(!audited.audited_source_metadata_validated);
    copy.change("max_position_embeddings", json!(0));
    let invalid = native_limits::inspect(&copy.0, LoadProfile::Ordinary);
    assert!(invalid.declared_total_positions.is_none());
    assert!(invalid.factory_position_upper.is_none());
    // Reinspect genuinely changed metadata; the earlier owned report stays historical.
    assert_eq!(ordinary.factory_position_upper, Some(32));
    copy.change("max_position_embeddings", json!(8));
    fs::remove_file(copy.0.join("model.safetensors")).unwrap();
    let missing = native_limits::inspect(&copy.0, LoadProfile::Ordinary);
    assert_eq!(missing.status, Status::Unsupported);
    assert!(missing.factory_position_upper.is_none());
    assert!(!missing.tensor_metadata_validated);
}
