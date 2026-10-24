//! Portable JSON inspection; never downloads weights or creates a device.
use serde_json::Value;
use std::{
    path::Path,
    process::{Command, Output},
};
fn fixture(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}
fn inspect(path: &Path, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_toklet"))
        .arg("inspect")
        .arg("--model")
        .arg(path)
        .args(extra)
        .output()
        .unwrap()
}
fn record(out: Output) -> Value {
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out.stderr.is_empty());
    assert_eq!(out.stdout.iter().filter(|&&byte| byte == b'\n').count(), 1);
    serde_json::from_slice(&out.stdout).unwrap()
}
#[test]
fn cli_preserves_engine_reports_for_each_family_backend_and_budget() {
    for (path, family) in [
        ("mlx-qwen-v1/model", "qwen2"),
        ("mlx-llama-v1/tied", "llama"),
        ("mlx-mistral-v1/untied", "mistral"),
        ("mlx-qwen3-v1/tied", "qwen3"),
    ] {
        for backend in ["scalar-reference", "metal"] {
            let value = record(inspect(
                &fixture(path),
                &["--backend", backend, "--context-capacity", "32"],
            ));
            assert_eq!(value["architecture"], family);
            assert_eq!(value["status"], "compatible_untested");
            assert_eq!(value["requested_memory_limit_bytes"], 6442450944u64);
            assert_eq!(value["qualification_evaluated"], false);
            let below = record(inspect(
                &fixture(path),
                &[
                    "--backend",
                    backend,
                    "--context-capacity",
                    "32",
                    "--memory-limit-bytes",
                    "1",
                ],
            ));
            assert_eq!(below["reason"], "load_budget_exceeded");
            assert_eq!(below["load_admission"], "exceeds_budget");
        }
    }
}
#[test]
fn valid_inspect_requests_always_return_reports_including_rejections() {
    let missing = record(inspect(Path::new("missing-inspection-fixture"), &[]));
    assert_eq!(missing["reason"], "invalid_configuration");
    let invalid = record(inspect(
        Path::new("missing-inspection-fixture"),
        &["--context-capacity", "0"],
    ));
    assert_eq!(invalid["reason"], "invalid_request");
    let oversized = record(inspect(
        &fixture("mlx-qwen3-v1/tied"),
        &["--context-capacity", "33"],
    ));
    assert_eq!(oversized["reason"], "artifact_rejected");
}
#[test]
fn malformed_inspection_arguments_never_emit_a_report() {
    for args in [
        vec![],
        vec!["--model"],
        vec!["--model", "x", "--backend", "cpu"],
        vec!["--model", "x", "--tokens", "1"],
        vec!["--model", "x", "--model", "x"],
        vec!["--model", "x", "--context-capacity", "-1"],
        vec!["--model", "x", "--memory-limit-bytes", "no"],
    ] {
        let out = Command::new(env!("CARGO_BIN_EXE_toklet"))
            .arg("inspect")
            .args(args)
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(2));
        assert!(out.stdout.is_empty());
        assert!(!out.stderr.is_empty());
    }
    for args in [vec!["--help"], vec!["inspect", "--help"]] {
        let out = Command::new(env!("CARGO_BIN_EXE_toklet"))
            .args(args)
            .output()
            .unwrap();
        assert!(out.status.success());
        assert!(out.stderr.is_empty());
        assert!(
            String::from_utf8(out.stdout)
                .unwrap()
                .contains("toklet inspect")
        );
    }
}
