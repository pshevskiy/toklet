//! Portable argument checks; explicit runtime GPU tests are ignored by default.
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
use serde_json::Value;
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
use std::path::Path;
use std::process::{Command, Output};
fn command(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_toklet"))
        .arg("plan")
        .args(args)
        .output()
        .unwrap()
}
#[test]
fn malformed_runtime_plan_arguments_never_initialize_or_emit_a_report() {
    for args in [
        vec![],
        vec!["--model"],
        vec!["--model", "x", "--model", "y"],
        vec!["--model", "x", "--backend", "cpu"],
        vec!["--model", "x", "--attention", "auto"],
        vec!["--model", "x", "--readback", "token"],
        vec!["--model", "x", "--normalization", "fast"],
        vec!["--model", "x", "--route", "implicit"],
        vec![
            "--model",
            "x",
            "--route",
            "official-gemma-bf16",
            "--attention",
            "reference",
        ],
        vec![
            "--model",
            "x",
            "--route",
            "official-gemma-bf16",
            "--readback",
            "full-logits",
        ],
        vec![
            "--model",
            "x",
            "--route",
            "official-gemma-bf16",
            "--normalization",
            "compensated",
        ],
        vec!["--model", "x", "--prefill-chunk", "-1"],
        vec![
            "--model",
            "x",
            "--position-offset",
            "18446744073709551615",
            "--context-capacity",
            "1",
        ],
    ] {
        let out = command(&args);
        assert_eq!(out.status.code(), Some(2));
        assert!(out.stdout.is_empty());
        assert!(!out.stderr.is_empty());
    }
    let help = command(&["--help"]);
    assert!(help.status.success() && help.stderr.is_empty());
    assert!(String::from_utf8_lossy(&help.stdout).contains("Initializes"));
}
#[test]
#[cfg(not(all(
    feature = "metal",
    any(target_os = "macos", target_os = "ios"),
    target_arch = "aarch64"
)))]
fn runtime_plan_unavailable_build_has_no_implicit_metadata_fallback() {
    let out = command(&["--model", "missing"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty());
    assert!(String::from_utf8_lossy(&out.stderr).contains("unavailable"));
}
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "Explicit Apple GPU runtime CLI/direct equality and budget/status contracts"]
fn runtime_cli_exactly_matches_direct_engine_reports() {
    use toklet::{
        metal::{self, AttentionMode, NormalizationMode, ReadbackMode},
        session, weights,
    };
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mlx-qwen3-v1/tied");
    for readback in [ReadbackMode::FullLogits, ReadbackMode::GreedyToken] {
        for limit in [1, 256 * 1024 * 1024] {
            let mut args = vec![
                "--model".to_owned(),
                p.to_str().unwrap().to_owned(),
                "--context-capacity".to_owned(),
                "31".to_owned(),
                "--position-offset".to_owned(),
                "1".to_owned(),
                "--prefill-chunk".to_owned(),
                "3".to_owned(),
                "--load-memory-limit-bytes".to_owned(),
                limit.to_string(),
                "--session-memory-limit-bytes".to_owned(),
                limit.to_string(),
                "--readback".to_owned(),
                if readback == ReadbackMode::FullLogits {
                    "full-logits".to_owned()
                } else {
                    "greedy-token".to_owned()
                },
            ];
            let borrowed: Vec<_> = args.iter().map(String::as_str).collect();
            let out = command(&borrowed);
            assert!(out.status.success() && out.stderr.is_empty());
            assert_eq!(out.stdout.iter().filter(|&&v| v == b'\n').count(), 1);
            let value: Value = serde_json::from_slice(&out.stdout).unwrap();
            let direct = metal::planning::inspect_mlx(
                &p,
                weights::LoadOptions {
                    context_limit: 32,
                    memory_limit_bytes: limit,
                },
                session::Options {
                    capacity: 31,
                    prefill_chunk_size: 3,
                    position_offset: 1,
                    memory_limit_bytes: limit,
                },
                NormalizationMode::Compensated,
                readback,
                AttentionMode::Reference,
            )
            .unwrap();
            assert_eq!(value, serde_json::to_value(direct).unwrap());
            assert_eq!(value["fits_requested_budgets"], limit > 1);
            args.extend(["--attention".to_owned(), "online".to_owned()]);
            let borrowed: Vec<_> = args.iter().map(String::as_str).collect();
            let rejected = command(&borrowed);
            assert_eq!(rejected.status.code(), Some(1));
            assert!(rejected.stdout.is_empty());
        }
    }
    for extra in [
        vec!["--context-capacity", "0"],
        vec!["--prefill-chunk", "0"],
        vec!["--load-context-capacity", "31"],
        vec!["--load-memory-limit-bytes", "0"],
    ] {
        let mut args = vec!["--model", p.to_str().unwrap(), "--context-capacity", "32"];
        if extra[0] == "--context-capacity" {
            args.truncate(2);
        }
        args.extend(extra);
        let out = command(&args);
        assert_eq!(out.status.code(), Some(1));
        assert!(out.stdout.is_empty());
    }
    let mut child = Command::new(env!("CARGO_BIN_EXE_toklet"))
        .args([
            "plan",
            "--model",
            p.to_str().unwrap(),
            "--context-capacity",
            "32",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    drop(child.stdout.take());
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success() && out.stderr.is_empty());
    println!(
        "RUNTIME_CLI_PLANNING {}",
        serde_json::json!({"direct_reports_equal":4,"budget_fit_false_is_success":true,"invalid_engine_requests_rejected":true,"broken_pipe_success":true,"passed":true})
    );
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "Explicit official BF16 Mac CLI/direct policy and minimum-budget reports"]
fn official_bf16_cli_matches_explicit_selected_route_and_budget_reports() {
    let model = std::path::PathBuf::from(std::env::var_os("TOKLET_GEMMA_BF16_MODEL").unwrap());
    for budget in [1, toklet::weights::INITIAL_MEMORY_LIMIT_BYTES] {
        let out = command(&[
            "--model",
            model.to_str().unwrap(),
            "--route",
            "official-gemma-bf16",
            "--load-memory-limit-bytes",
            &budget.to_string(),
            "--session-memory-limit-bytes",
            &budget.to_string(),
        ]);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let value: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        let direct = toklet::metal::gemma_affine::inspect_official_bf16(
            &model,
            toklet::weights::LoadOptions {
                context_limit: 512,
                memory_limit_bytes: budget,
            },
            toklet::session::Options {
                capacity: 512,
                prefill_chunk_size: 32,
                position_offset: 0,
                memory_limit_bytes: budget,
            },
        )
        .unwrap();
        assert_eq!(value, serde_json::to_value(direct).unwrap());
        assert_eq!(value["minimum_fits_load_budget"], budget > 1);
        assert_eq!(value["minimum_fits_session_budget"], budget > 1);
        assert_eq!(value["requested_chunk_rows"], 32);
    }
}
