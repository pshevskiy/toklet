//! Independent portable controls; these never load a model or require a GPU.
use serde_json::Value;
use std::{
    ffi::OsString,
    fs,
    io::{self, Write},
    path::PathBuf,
    process::{Command, Output},
    sync::atomic::{AtomicU64, Ordering},
};

// The collector is pure, but its typed enum includes branches used only by the
// production caller. Keep dead-code allowance local to this included module.
#[allow(dead_code)]
#[path = "../src/bin/cli/gemma4/summary.rs"]
mod summary;
use summary::{CacheAction, Collector, ErrorCategory, Kind, Manifest, Memory, Outcome, Phase};
use summary::{Record, State, Timing, Work};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Files(PathBuf);
impl Files {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "toklet-gemma-cli-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}
impl Drop for Files {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn invoke(args: &[OsString]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_toklet"))
        .args(args)
        .output()
        .expect("portable CLI process")
}
fn arguments(files: &Files) -> Vec<OsString> {
    vec![
        "--model-profile".into(),
        "gemma4-bf16".into(),
        "--model".into(),
        files.path("model-must-not-be-opened").into_os_string(),
        "--tokens".into(),
        "2,105".into(),
        "--backend".into(),
        "metal".into(),
    ]
}
fn rejected(files: &Files, args: &[OsString]) -> String {
    let output = invoke(args);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(!files.path("metrics.jsonl").exists());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains("error:"));
    assert!(!error.contains("No such file or directory"), "{error}");
    error
}

#[test]
fn gemma_cli_help_names_explicit_route_and_metrics_without_model_io() {
    let output = invoke(&["--help".into()]);
    assert!(output.status.success());
    let help = String::from_utf8(output.stdout).unwrap();
    for flag in [
        "gemma4-bf16",
        "--submission",
        "--request-metrics",
        "--metrics-output",
    ] {
        assert!(help.contains(flag), "missing {flag}");
    }
}

#[test]
fn gemma_cli_rejects_unknown_duplicate_and_incomplete_flags_before_io() {
    let files = Files::new();
    for tail in [
        vec!["--submission", "unknown"],
        vec!["--request-metrics", "unknown"],
        vec!["--request-metrics"],
        vec!["--metrics-output"],
        vec![
            "--submission",
            "per-tile",
            "--submission",
            "batched-single-row",
        ],
    ] {
        // Put syntax errors before backend selection so CPU builds exercise
        // the syntax rejection rather than merely platform admission.
        let mut args: Vec<OsString> = tail.into_iter().map(Into::into).collect();
        args.extend(arguments(&files));
        rejected(&files, &args);
    }
}

#[test]
fn gemma_specific_flags_do_not_change_older_profiles() {
    let files = Files::new();
    for profile in [
        "qwen",
        "lfm3-bf16",
        "ministral3-diagnostic",
        "qwen35-2b-diagnostic",
    ] {
        for flag in [
            vec!["--submission", "per-tile"],
            vec!["--request-metrics", "disabled"],
            vec!["--metrics-output", "unused.jsonl"],
        ] {
            let mut args: Vec<OsString> = [
                "--model-profile",
                profile,
                "--model",
                "unopened",
                "--tokens",
                "2",
            ]
            .into_iter()
            .map(Into::into)
            .collect();
            args.extend(flag.into_iter().map(OsString::from));
            let error = rejected(&files, &args);
            assert!(error.contains("require gemma4-bf16"), "{error}");
        }
    }
}

#[test]
fn gemma_admission_rejects_modes_bounds_and_metrics_pairing_before_io() {
    let files = Files::new();
    let mut cases = Vec::new();
    for tail in [
        vec!["--context-capacity", "0"],
        vec!["--context-capacity", "513"],
        vec!["--prefill-chunk", "0"],
        vec!["--prefill-chunk", "33"],
        vec!["--context-capacity", "1", "--prefill-chunk", "2"],
        vec!["--request-metrics", "summary"],
        vec!["--metrics-output", "unused.jsonl"],
        vec!["--tools", "must-not-be-opened"],
        vec!["--metal-projection", "original"],
        vec!["--metal-attention", "reference"],
        vec!["--cpu-projection", "scalar"],
        vec!["--q4-weights", "must-not-be-opened"],
    ] {
        let mut args = arguments(&files);
        args.extend(tail.into_iter().map(OsString::from));
        cases.push(args);
    }
    for args in cases {
        rejected(&files, &args);
    }
    let mut cpu = arguments(&files);
    cpu[7] = "cpu".into();
    assert!(rejected(&files, &cpu).contains("requires Metal"));
}

#[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
#[test]
fn gemma_unsupported_platform_is_explicit_and_never_falls_back_to_cpu() {
    let files = Files::new();
    let error = rejected(&files, &arguments(&files));
    assert!(error.contains("Metal") && error.contains("Apple Silicon"));
}

fn manifest() -> Manifest<'static> {
    Manifest {
        session_id: "portable-independent",
        capacity: 512,
        prefill_chunk_size: 32,
        submission: "per-tile",
        load_ms: 10.0,
        route_start_to_ready_ms: 20.0,
        load_admitted_bytes: 100 * 1024 * 1024,
        // Numerical50MiB plus1GiB owner and all five explicit allowances.
        session_admitted_bytes: 1_299_202_048,
        logical_memory: toklet::session::Memory {
            weights_bytes: 32 * 1024 * 1024,
            positional_bytes: 1024 * 1024,
            kv_bytes: 2 * 1024 * 1024,
            scratch_bytes: 1024 * 1024,
            history_bytes: 4096,
            logits_bytes: 2 * 1024 * 1024,
            metadata_bytes: 4096,
            metadata_allowance_bytes: 12 * 1024 * 1024 - 8192,
            admitted_bytes: 50 * 1024 * 1024,
        },
    }
}
fn record(request_id: u64) -> Record<'static> {
    Record {
        session_id: "portable-independent",
        request_id,
        attempt_id: request_id,
        owner_generation: 1,
        request_kind: Kind::Generate,
        terminal_status: Outcome::Success,
        failed_phase: None,
        error_category: None,
        cache_action: CacheAction::Fresh,
        replay_required: false,
        before: State::default(),
        after: State {
            history_tokens: 4,
            committed_tokens: 3,
            pending_tokens: 1,
        },
        work: Work {
            prompt_tokens: Some(2),
            reused_prefix_tokens: Some(0),
            prefill_rows: Some(2),
            decode_evaluations: Some(1),
            sampled_tokens: Some(2),
            emitted_tokens: Some(2),
            callback_tokens: 2,
            replayed_rows: Some(0),
        },
        timing: Timing {
            request_total_ms: 100.0,
            native_total_ms: Some(60.0),
            tokenization_ms: Some(5.0),
            prefill_ms: Some(40.0),
            decode_ms: Some(10.0),
            sampling_ms: Some(1.0),
            ttft_event_ms: Some(50.0),
            first_text_publication_ms: Some(55.0),
            last_token_event_ms: Some(70.0),
            token_interval_count: 1,
            token_interval_min_ms: Some(20.0),
            token_interval_max_ms: Some(20.0),
            token_interval_sum_ms: 20.0,
            output_decode_ms: 1.0,
            output_publication_ms: 2.0,
            cancel_to_terminal_ms: None,
        },
        memory: Memory {
            weights_owned_bytes: Some(32 * 1024 * 1024),
            kv_allocated_bytes: Some(2 * 1024 * 1024),
            scratch_bytes: Some(1024 * 1024),
            kv_valid_positions: 3,
            owner_reserved_bytes: 1024 * 1024 * 1024,
            load_admitted_bytes: 100 * 1024 * 1024,
            session_admitted_bytes: 1_299_202_048,
            app_allowance_bytes: 128 * 1024 * 1024,
            stream_allowance_bytes: 4 * 1024 * 1024,
            telemetry_allowance_bytes: summary::ALLOWANCE_BYTES,
            helper_reserve_bytes: 32 * 1024 * 1024,
        },
    }
}
fn rows(bytes: &[u8]) -> Vec<Value> {
    assert!(bytes.ends_with(b"\n"));
    bytes
        .split(|b| *b == b'\n')
        .filter(|s| !s.is_empty())
        .map(|s| {
            assert!(s.len() <= 4096);
            serde_json::from_slice(s).unwrap()
        })
        .collect()
}

#[test]
fn disabled_collector_has_no_ring_or_retained_records() {
    let mut disabled = Collector::new(false);
    disabled.set_manifest(&manifest());
    disabled.record(&record(1));
    assert_eq!(disabled.status().retained, 0);
    assert!(disabled.allocated_bytes() <= 16 * 1024);
}

#[test]
fn unassignable_request_identity_marks_outcome_denominator_unknown_without_fake_record() {
    let mut collector = Collector::new(true);
    collector.set_manifest(&manifest());
    collector.record(&record(1));
    let before = collector.status();
    collector.note_counter_overflow();
    let status = collector.status();
    assert!(status.counter_overflow);
    assert_eq!(status.retained, before.retained);
    assert_eq!(status.telemetry_dropped, before.telemetry_dropped + 1);
    assert_eq!(status.invalid_records, before.invalid_records);
    assert_eq!(status.export_errors, before.export_errors);
    let mut bytes = Vec::new();
    collector.export_to(&mut bytes).unwrap();
    let data = rows(&bytes);
    assert_eq!(data.len(), 3);
    assert_eq!(data[1]["request_id"], 1);
    assert_eq!(data[2]["retained"], 1);
    assert_eq!(data[2]["telemetry_dropped"], 1);
    assert_eq!(data[2]["counter_overflow"], true);
}

#[test]
fn collector_exports_independent_rates_and_explicit_unknown_observations() {
    let mut collector = Collector::new(true);
    collector.set_manifest(&manifest());
    collector.record(&record(1));
    assert!(collector.allocated_bytes() <= 16 * 1024 + 1024 * 1024);
    let mut bytes = Vec::new();
    let status = collector.export_to(&mut bytes).unwrap();
    assert_eq!(status.retained, 1);
    let data = rows(&bytes);
    assert_eq!(data.len(), 3);
    assert_eq!(data[0]["event"], "request_metrics_manifest");
    assert_eq!(data[0]["model_profile"], "gemma4-bf16");
    assert_eq!(data[0]["model_repository"], "google/gemma-4-E2B-it");
    assert_eq!(
        data[0]["model_revision"],
        "70af34e20bd4b7a91f0de6b22675850c43922a03"
    );
    assert_eq!(data[0]["backend"], "metal");
    assert_eq!(data[0]["weight_dtype"], "bf16");
    assert_eq!(data[0]["kv_dtype"], "fp32");
    assert_eq!(data[0]["sampling"], "greedy");
    assert_eq!(data[0]["thinking_enabled"], false);
    assert_eq!(data[0]["instrumentation"], "summary");
    assert_eq!(data[0]["engine_crate_version"], env!("CARGO_PKG_VERSION"));
    for field in [
        "engine_revision",
        "build_flags",
        "device_identity",
        "device_power",
        "model_content_sha256",
        "tokenizer_content_sha256",
        "conversion_ms",
        "compile_ms",
        "process_start_to_ready_ms",
    ] {
        assert!(data[0].get(field).unwrap().is_null(), "invented {field}");
    }
    assert_eq!(
        data[0]["external_provenance_reason"],
        "external_capture_manifest_required"
    );
    assert_eq!(data[1]["prefill_tokens_per_s"], 50.0);
    assert_eq!(data[1]["decode_tokens_per_s"], 50.0);
    assert_eq!(data[1]["decode_evaluations_per_s"], 100.0);
    assert_eq!(data[1]["token_interval_mean_ms"], 20.0);
    assert!(
        data[1]["ordinary_observations"]
            .as_object()
            .unwrap()
            .values()
            .all(Value::is_null)
    );
    assert_eq!(data[2]["event"], "request_metrics_export");
    assert_eq!(data[2]["complete"], true);
}

#[test]
fn logical_prompt_covers_history_on_continue_append_and_partial_cancellation() {
    let mut collector = Collector::new(true);
    let mut header = manifest();
    header.prefill_chunk_size = 2;
    collector.set_manifest(&header);
    let mut cancel = record(1);
    cancel.terminal_status = Outcome::Cancelled;
    cancel.after = State {
        history_tokens: 5,
        committed_tokens: 2,
        pending_tokens: 3,
    };
    cancel.work = Work {
        prompt_tokens: Some(5),
        reused_prefix_tokens: Some(0),
        prefill_rows: Some(2),
        decode_evaluations: Some(0),
        sampled_tokens: Some(0),
        emitted_tokens: Some(0),
        callback_tokens: 0,
        replayed_rows: Some(0),
    };
    cancel.timing = Timing {
        request_total_ms: 15.0,
        native_total_ms: Some(12.0),
        tokenization_ms: Some(1.0),
        prefill_ms: Some(10.0),
        decode_ms: Some(0.0),
        sampling_ms: Some(0.0),
        ..Timing::default()
    };
    cancel.memory.kv_valid_positions = 2;
    collector.record(&cancel);
    let mut continuation = record(2);
    continuation.request_kind = Kind::Continue;
    continuation.cache_action = CacheAction::Pending;
    continuation.before = cancel.after;
    continuation.after = State {
        history_tokens: 7,
        committed_tokens: 6,
        pending_tokens: 1,
    };
    continuation.work.prompt_tokens = Some(5);
    continuation.work.reused_prefix_tokens = Some(2);
    continuation.work.prefill_rows = Some(3);
    continuation.memory.kv_valid_positions = 6;
    collector.record(&continuation);
    let mut append = record(3);
    append.request_kind = Kind::Append;
    append.cache_action = CacheAction::Append;
    append.before = continuation.after;
    append.after = State {
        history_tokens: 10,
        committed_tokens: 9,
        pending_tokens: 1,
    };
    append.work = Work {
        prompt_tokens: Some(9),
        reused_prefix_tokens: Some(6),
        prefill_rows: Some(3),
        decode_evaluations: Some(0),
        sampled_tokens: Some(1),
        emitted_tokens: Some(1),
        callback_tokens: 1,
        replayed_rows: Some(0),
    };
    append.timing = Timing {
        request_total_ms: 80.0,
        native_total_ms: Some(50.0),
        tokenization_ms: Some(1.0),
        prefill_ms: Some(35.0),
        decode_ms: Some(0.0),
        sampling_ms: Some(1.0),
        ttft_event_ms: Some(40.0),
        first_text_publication_ms: Some(45.0),
        last_token_event_ms: Some(40.0),
        output_decode_ms: 1.0,
        output_publication_ms: 2.0,
        ..Timing::default()
    };
    append.memory.kv_valid_positions = 9;
    collector.record(&append);
    assert_eq!(collector.status().retained, 3);
    assert_eq!(collector.status().invalid_records, 0);
    // Appending two IDs still admits the nine-token logical input history.
    append.request_id = 4;
    append.attempt_id = 4;
    append.work.prompt_tokens = Some(2);
    collector.record(&append);
    // Continue has no new input IDs, but its logical prompt is the existing five.
    continuation.request_id = 5;
    continuation.attempt_id = 5;
    continuation.work.prompt_tokens = Some(0);
    collector.record(&continuation);
    append.request_id = 6;
    append.attempt_id = 6;
    append.work.prompt_tokens = Some(9);
    append.work.replayed_rows = Some(4);
    collector.record(&append);
    assert_eq!(collector.status().retained, 3);
    assert_eq!(collector.status().invalid_records, 3);
    let mut bytes = Vec::new();
    collector.export_to(&mut bytes).unwrap();
    let data = rows(&bytes);
    assert_eq!(data[1]["work"]["prompt_tokens"], 5);
    assert_eq!(data[1]["after"]["pending_tokens"], 3);
    assert_eq!(data[2]["work"]["prompt_tokens"], 5);
    assert_eq!(data[3]["work"]["prompt_tokens"], 9);
    assert_eq!(data[3]["after"]["history_tokens"], 10);
}

#[test]
fn collector_drops_invalid_records_without_publishing_nonfinite_or_impossible_state() {
    let mut collector = Collector::new(true);
    collector.set_manifest(&manifest());
    let mut bad = record(1);
    bad.timing.request_total_ms = f64::NAN;
    collector.record(&bad);
    let mut bad = record(2);
    bad.timing.decode_ms = Some(-1.0);
    collector.record(&bad);
    let mut bad = record(3);
    bad.after.committed_tokens = 5;
    collector.record(&bad);
    let mut bad = record(4);
    bad.session_id = "another-session";
    collector.record(&bad);
    assert_eq!(collector.status().invalid_records, 4);
    assert_eq!(collector.status().retained, 0);
    collector.record(&record(5));
    let mut bytes = Vec::new();
    collector.export_to(&mut bytes).unwrap();
    let data = rows(&bytes);
    assert_eq!(data.len(), 3);
    assert_eq!(data[1]["request_id"], 5);
    assert_eq!(data[2]["invalid_records"], 4);
}

#[test]
fn collector_ring_wrap_preserves_a_complete_ordered_suffix_and_bounded_payload() {
    let mut collector = Collector::new(true);
    collector.set_manifest(&manifest());
    // Each fixed-field summary exceeds1KiB, so1024 cannot fit the1MiB ring.
    assert!(serde_json::to_vec(&record(1)).unwrap().len() > 1024);
    for id in 1..=1024 {
        collector.record(&record(id));
    }
    let status = collector.status();
    assert!(status.retained > 0 && status.retained < 1024);
    assert_eq!(status.retained + status.telemetry_dropped, 1024);
    assert_eq!(status.invalid_records, 0);
    let mut bytes = Vec::new();
    collector.export_to(&mut bytes).unwrap();
    assert!(bytes.len() <= 1024 * 1024 + 2 * 4096);
    let data = rows(&bytes);
    assert_eq!(data.len() as u64, status.retained + 2);
    let ids: Vec<_> = data[1..data.len() - 1]
        .iter()
        .map(|r| r["request_id"].as_u64().unwrap())
        .collect();
    assert_eq!(
        ids,
        (status.telemetry_dropped + 1..=1024).collect::<Vec<_>>()
    );
}

#[test]
fn terminal_metadata_failure_corrects_only_the_latest_matching_attempt() {
    let mut collector = Collector::new(true);
    collector.set_manifest(&manifest());
    collector.record(&record(1));
    collector.record(&record(2));
    let before = collector.status();
    let mut error = record(2);
    error.terminal_status = Outcome::OutputError;
    error.failed_phase = Some(Phase::Output);
    error.error_category = Some(ErrorCategory::Output);
    error.replay_required = true;
    collector.correct_last_terminal(&error);
    assert_eq!(collector.status().retained, before.retained);
    assert_eq!(
        collector.status().telemetry_dropped,
        before.telemetry_dropped
    );
    let mut stale = record(1);
    stale.terminal_status = Outcome::OutputError;
    stale.failed_phase = Some(Phase::Output);
    stale.error_category = Some(ErrorCategory::Output);
    collector.correct_last_terminal(&stale);
    let mut clear_replay = record(2);
    clear_replay.terminal_status = Outcome::OutputError;
    clear_replay.failed_phase = Some(Phase::Output);
    clear_replay.error_category = Some(ErrorCategory::Output);
    collector.correct_last_terminal(&clear_replay);
    let mut changed_work = record(2);
    changed_work.terminal_status = Outcome::OutputError;
    changed_work.failed_phase = Some(Phase::Output);
    changed_work.error_category = Some(ErrorCategory::Output);
    changed_work.replay_required = true;
    changed_work.work.sampled_tokens = Some(3);
    collector.correct_last_terminal(&changed_work);
    assert_eq!(collector.status().retained, before.retained);
    assert_eq!(
        collector.status().telemetry_dropped,
        before.telemetry_dropped
    );
    let mut bytes = Vec::new();
    collector.export_to(&mut bytes).unwrap();
    let data = rows(&bytes);
    assert_eq!(data.len(), 4);
    assert_eq!(data[1]["request_id"], 1);
    assert_eq!(data[1]["terminal_status"], "success");
    assert_eq!(data[2]["request_id"], 2);
    assert_eq!(data[2]["terminal_status"], "output_error");
    assert_eq!(data[2]["replay_required"], true);
    assert_eq!(data[2]["work"]["sampled_tokens"], 2);
    assert_eq!(data[2]["after"]["committed_tokens"], 3);
    assert_eq!(data[3]["retained"], 2);
}

struct FailingWriter;
impl Write for FailingWriter {
    fn write(&mut self, _bytes: &[u8]) -> io::Result<usize> {
        Err(io::Error::other(
            "independent injected telemetry sink failure",
        ))
    }
    fn flush(&mut self) -> io::Result<()> {
        Err(io::Error::other("independent injected flush failure"))
    }
}
#[test]
fn export_failure_preserves_records_and_existing_file_bytes() {
    let files = Files::new();
    let mut collector = Collector::new(true);
    collector.set_manifest(&manifest());
    collector.record(&record(1));
    assert!(collector.export_to(FailingWriter).is_err());
    assert_eq!(collector.status().retained, 1);
    assert_eq!(collector.status().export_errors, 1);
    let target = files.path("metrics.jsonl");
    fs::write(&target, b"user artifact\n").unwrap();
    assert!(collector.export(&target).is_err());
    assert_eq!(fs::read(&target).unwrap(), b"user artifact\n");
    let mut bytes = Vec::new();
    collector.export_to(&mut bytes).unwrap();
    let data = rows(&bytes);
    assert_eq!(data[1]["terminal_status"], "success");
    assert_eq!(data[1]["after"]["history_tokens"], 4);
    assert_eq!(data[2]["export_errors"], 2);
}

#[cfg(unix)]
#[test]
fn export_never_follows_a_symlink() {
    let files = Files::new();
    let target = files.path("sentinel");
    fs::write(&target, b"user artifact\n").unwrap();
    let link = files.path("link");
    std::os::unix::fs::symlink(&target, &link).unwrap();
    let mut collector = Collector::new(true);
    collector.set_manifest(&manifest());
    collector.record(&record(1));
    assert!(collector.export(&link).is_err());
    assert_eq!(fs::read(target).unwrap(), b"user artifact\n");
}
