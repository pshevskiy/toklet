use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, io::Read, path::PathBuf};
use toklet::{
    cpu::ProjectionKernel,
    sampling,
    session::{Options, Session},
    weights::{self, LoadOptions},
};

#[test]
#[ignore = "requires the explicitly downloaded pinned Qwen checkpoint; Mac footprint requires own-process inspection"]
fn pinned_qwen_cached_sessions_match_full_prefix() {
    check_sessions(ProjectionKernel::Scalar);
}

#[test]
#[ignore = "requires the explicitly downloaded pinned Qwen checkpoint; Mac footprint requires own-process inspection"]
fn pinned_qwen_tiled_cached_sessions_match_full_prefix() {
    check_sessions(ProjectionKernel::Tiled);
}

fn check_sessions(kernel: ProjectionKernel) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let probes: Value = serde_json::from_slice(
        &fs::read(root.join("tests/fixtures/qwen-v2/manifest.json")).unwrap(),
    )
    .unwrap();
    let mut source =
        fs::File::open(root.join("models/Qwen2.5-0.5B-Instruct/model.safetensors")).unwrap();
    let mut digest = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let count = source.read(&mut buffer).unwrap();
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    assert_eq!(
        format!("{:x}", digest.finalize()),
        probes["model"]["weights"]["sha256"].as_str().unwrap()
    );
    let loaded = weights::load_qwen(
        &root.join("models/Qwen2.5-0.5B-Instruct"),
        LoadOptions::default(),
    )
    .unwrap();
    let owned_model = loaded.model.with_projection_kernel(kernel);
    let model = &owned_model;
    let width = model.config().vocab_size;
    let mut scratch = model.scratch(5).unwrap();
    let mut results = BTreeMap::new();
    let mut maximum_admitted_bytes = 0;
    for (case, spec) in probes["cases"].as_object().unwrap() {
        let tokens: Vec<u32> = spec["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_u64().unwrap() as u32)
            .collect();
        let offset = spec["offset"].as_u64().unwrap() as usize;
        let trace = model
            .forward_diagnostic(&tokens, offset, &mut scratch)
            .unwrap();
        let mut comparisons = 0;
        for chunk in [tokens.len(), 2, 1] {
            let opts = Options {
                capacity: 2048 - offset,
                prefill_chunk_size: chunk,
                position_offset: offset,
                ..Options::default()
            };
            let mut session = Session::new(model, opts).unwrap();
            let memory = session.memory();
            maximum_admitted_bytes = maximum_admitted_bytes.max(memory.admitted_bytes);
            assert_eq!(memory.weights_bytes, loaded.report.fp32_weights_bytes);
            assert_eq!(memory.kv_bytes, 2 * 24 * 2 * (2048 - offset) * 64 * 4);
            assert!(memory.admitted_bytes <= weights::INITIAL_MEMORY_LIMIT_BYTES);
            for part in tokens.chunks(chunk) {
                session.append(part).unwrap();
                assert!(session.logits().is_none());
                session.evaluate().unwrap();
                let position = session.evaluated_len() - 1;
                assert_eq!(
                    session.logits().unwrap(),
                    &trace.logits[position * width..(position + 1) * width],
                    "{case} chunk {chunk}"
                );
                assert_eq!(
                    sampling::greedy(session.logits().unwrap()).unwrap() as u64,
                    probes["argmax"][case][position].as_u64().unwrap()
                );
                comparisons += 1;
            }
            session.set_tokens(&tokens).unwrap();
            assert_eq!(session.evaluated_len(), tokens.len());
            assert_eq!(session.evaluate_next_chunk().unwrap(), 0);
            // A sampled token is pending in history, then evaluated exactly once.
            let sampled = sampling::greedy(session.logits().unwrap()).unwrap();
            session.append(&[sampled]).unwrap();
            assert_eq!(session.evaluated_len(), tokens.len());
            assert_eq!(session.pending_len(), 1);
            assert!(session.logits().is_none());
            session.evaluate().unwrap();
            let mut reference = vec![0.0; width];
            model
                .forward(session.history(), offset, &mut scratch, &mut reference)
                .unwrap();
            assert_eq!(session.logits().unwrap(), reference);
            comparisons += 1;
            assert_eq!(session.memory(), memory);
            // Shortening retains all but its final token; replay reconstructs logits.
            session.set_tokens(&tokens).unwrap();
            assert_eq!(session.evaluated_len(), tokens.len() - 1);
            session.evaluate().unwrap();
            assert_eq!(
                session.logits().unwrap(),
                &trace.logits[(tokens.len() - 1) * width..]
            );
            comparisons += 1;
        }
        eprintln!(
            "{case}: {comparisons} exact full-logit cache/full-prefix comparisons pass, including append, sampled pending token, fully cached prompt and replay"
        );
        results.insert(case.clone(), json!({"exact_full_logit_comparisons": comparisons, "position_offset": offset, "tokens": tokens}));
    }
    // Keep a fully admitted 2048-token allocation live for footprint inspection;
    // the real-model numerical scope above is short probes, not long-context runs.
    let admitted = Session::new(model, Options::default()).unwrap();
    let m = admitted.memory();
    maximum_admitted_bytes = maximum_admitted_bytes.max(m.admitted_bytes);
    assert_eq!(m.kv_bytes, 50_331_648);
    let peak = physical_peak();
    let report = json!({"model": probes["model"], "results": results, "cpu_projection":kernel.name(),
        "session_memory": {"weights_bytes": m.weights_bytes, "kv_bytes": m.kv_bytes,
            "scratch_bytes": m.scratch_bytes, "history_bytes": m.history_bytes,
            "logits_bytes": m.logits_bytes, "metadata_bytes": m.metadata_bytes,
            "metadata_allowance_bytes": m.metadata_allowance_bytes,
            "admitted_bytes": m.admitted_bytes, "maximum_admitted_bytes": maximum_admitted_bytes},
        "physical_process_peak_bytes": peak,
        "scope": "short real-model numerical checks; 2048-token capacity admitted, full 2048-token execution checked with the small CPU model; no performance/held-out quality qualification"});
    let path = std::env::var_os("TOKLET_QWEN_SESSION_REPORT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            root.join(if kernel == ProjectionKernel::Scalar {
                "artifacts/qwen-session-results.json"
            } else {
                "artifacts/qwen-session-tiled-results.json"
            })
        });
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let bytes = serde_json::to_vec_pretty(&report).unwrap();
    if kernel == ProjectionKernel::Tiled || std::env::var_os("TOKLET_QWEN_SESSION_REPORT").is_some()
    {
        use std::io::Write;
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap()
            .write_all(&bytes)
            .unwrap();
    } else {
        fs::write(&path, bytes).unwrap();
    }
    eprintln!("Session report: {}", path.display());
}

#[cfg(target_os = "macos")]
fn physical_peak() -> Option<u64> {
    // macOS SDK rusage_info_v4: 16 UUID bytes followed by35 u64 fields.
    // This bounded own-process syscall avoids an external footprint sampler.
    #[repr(C)]
    struct Usage {
        uuid: [u8; 16],
        fields: [u64; 35],
    }
    #[link(name = "proc")]
    unsafe extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, usage: *mut Usage) -> i32;
    }
    assert_eq!(std::mem::size_of::<Usage>(), 296);
    let mut usage = Usage {
        uuid: [0; 16],
        fields: [0; 35],
    };
    // SAFETY: initialized, aligned v4 storage has the SDK's296-byte size;
    // the call writes only this exclusive stack allocation synchronously.
    let status = unsafe { proc_pid_rusage(std::process::id() as i32, 4, &mut usage) };
    assert_eq!(status, 0, "own-process libproc v4 query failed");
    let peak = usage.fields[28]; // ri_lifetime_max_phys_footprint
    assert!(peak <= weights::INITIAL_MEMORY_LIMIT_BYTES as u64);
    eprintln!(
        "Own-process physical session peak: {peak} bytes; includes test/hash/diagnostic setup"
    );
    Some(peak)
}

#[cfg(not(target_os = "macos"))]
fn physical_peak() -> Option<u64> {
    None
}
