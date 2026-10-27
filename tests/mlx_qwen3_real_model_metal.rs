//! Explicit selected-checkpoint qualification; ordinary tests need only the manifest.
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const MANIFEST_SHA: &str = "4594118378d2c583c80c8f0987e06d3bbca0fa77188d4869e670252070b23fc7";
const ARRAY_SHA: &str = "812c8302de38ca6bf2c073a23940cfef27a64a950c1bfa8e841444606d3dd010";
fn manifest() -> Value {
    let bytes = include_bytes!("fixtures/mlx-qwen3-real-v1/manifest.json");
    assert_eq!(format!("{:x}", Sha256::digest(bytes)), MANIFEST_SHA);
    serde_json::from_slice(bytes).unwrap()
}

#[test]
fn qwen3_real_mlx_manifest_freezes_complete_independent_schema_and_gates() {
    let m = manifest();
    assert_eq!(m["format"], "toklet-mlx-qwen3-real-reference-v1");
    assert_eq!(m["model"]["repository"], "mlx-community/Qwen3-0.6B-8bit");
    assert_eq!(
        m["model"]["revision"],
        "11de96878523501bcaa86104e3c186de07ff9068"
    );
    assert_eq!(m["array_sha256"], ARRAY_SHA);
    assert_eq!(m["array_bytes"], 41_115_648);
    assert_eq!(m["layers"], json!([0, 1, 27]));
    assert_eq!(m["greedy"]["fp32_tokens"], json!([1096, 374, 264, 3491]));
    assert_eq!(m["greedy"]["fp64_tokens"], m["greedy"]["fp32_tokens"]);
    assert_eq!(
        m["greedy"]["teacher_forcing"],
        "Independent FP64-accumulation greedy history"
    );
    assert_eq!(m["tokenization"]["ids"], json!([9707, 11, 1879, 0]));
    assert_eq!(m["tokenization"]["chat_template_evaluated"], false);
    for (stage, shape) in [
        ("q_projection", json!([4, 2048])),
        ("k_projection", json!([4, 1024])),
        ("q_norm", json!([4, 16, 128])),
        ("k_norm", json!([4, 8, 128])),
        ("q_rotary", json!([4, 16, 128])),
        ("k_rotary", json!([4, 8, 128])),
        ("attention_context", json!([4, 2048])),
    ] {
        assert_eq!(
            m["arrays"][format!("cases.short.layers.27.{stage}")]["shape"],
            shape
        );
    }
    assert_eq!(m["cases"].as_object().unwrap().len(), 6);
    assert_eq!(m["arrays"].as_object().unwrap().len(), 385);
    let mut cursor = 0;
    let mut stages = std::collections::BTreeMap::new();
    for (name, item) in m["arrays"].as_object().unwrap() {
        let shape = item["shape"].as_array().unwrap();
        let elements: usize = shape.iter().map(|v| v.as_u64().unwrap() as usize).product();
        assert!(shape.iter().all(|v| v.as_u64().unwrap() > 0));
        let width = match item["dtype"].as_str().unwrap() {
            "<f2" => 2,
            "<f4" => 4,
            _ => panic!("Frozen array dtype differs"),
        };
        assert_eq!(item["offset"].as_u64().unwrap() as usize, cursor);
        assert_eq!(item["bytes"].as_u64().unwrap() as usize, elements * width);
        cursor += elements * width;
        let gate = &item["gate"];
        let arithmetic = gate["pooled_fp32_fp64_max_abs"].as_f64().unwrap();
        let prefix = gate["equivalent_prefix_max_abs"].as_f64().unwrap();
        assert!(arithmetic.is_finite() && arithmetic >= 0.0 && prefix.is_finite() && prefix >= 0.0);
        assert_eq!(
            gate["atol"].as_f64().unwrap(),
            1e-4f64.max(2.0 * arithmetic).max(2.0 * prefix)
        );
        assert_eq!(gate["rtol"], 1e-3);
        let stage = name.splitn(3, '.').nth(2).unwrap();
        if let Some(previous) = stages.insert(stage, gate) {
            assert_eq!(previous, gate);
        }
    }
    assert_eq!(cursor, 41_115_648);
    for flag in [
        "target_gpu_executed",
        "held_out_quality_qualified",
        "physical_device_qualified",
    ] {
        assert_eq!(m[flag], false);
    }
}

#[cfg(all(
    feature = "metal",
    any(target_os = "macos", target_os = "ios"),
    target_arch = "aarch64"
))]
mod gpu {
    use super::*;
    use half::f16;
    use std::{collections::BTreeSet, fs, io::Read, path::PathBuf};
    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }
    use toklet::{
        metal::{self, AttentionMode, NormalizationMode, ProjectionMode, ReadbackMode},
        session,
        weights::LoadOptions,
    };
    const BUDGET: usize = 1536 * 1024 * 1024;
    const MODEL: &str = "models/MLX-Qwen3-0.6B-8bit-11de968";
    struct Reference {
        manifest: Value,
        bytes: Vec<u8>,
    }
    impl Reference {
        fn open() -> Self {
            let manifest = manifest();
            let path = root().join("artifacts/m8-qwen3-real-reference-v1/arrays.bin");
            let mut file =
                fs::File::open(path).expect("Run the explicitly reviewed CPU export first");
            let mut bytes = Vec::new();
            file.by_ref()
                .take(41_115_649)
                .read_to_end(&mut bytes)
                .unwrap();
            assert_eq!(bytes.len(), 41_115_648);
            assert_eq!(format!("{:x}", Sha256::digest(&bytes)), ARRAY_SHA);
            Self { manifest, bytes }
        }
        fn array(&self, name: &str) -> Vec<f32> {
            let item = &self.manifest["arrays"][name];
            let offset = item["offset"].as_u64().unwrap() as usize;
            let size = item["bytes"].as_u64().unwrap() as usize;
            let bytes = &self.bytes[offset..offset + size];
            assert_eq!(format!("{:x}", Sha256::digest(bytes)), item["sha256"]);
            match item["dtype"].as_str().unwrap() {
                "<f2" => bytes
                    .chunks_exact(2)
                    .map(|b| f16::from_le_bytes(b.try_into().unwrap()).to_f32())
                    .collect(),
                "<f4" => bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
                _ => panic!("Frozen dtype differs"),
            }
        }
        fn qualify(&self, name: &str, actual: &[f32], start: usize) -> f64 {
            let expected = self.array(name);
            let expected = &expected[start..];
            assert_eq!(actual.len(), expected.len(), "{name}");
            let gate = &self.manifest["arrays"][name]["gate"];
            let atol = gate["atol"].as_f64().unwrap();
            let rtol = gate["rtol"].as_f64().unwrap();
            let mut maximum = 0.0f64;
            for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
                let delta = (f64::from(a) - f64::from(b)).abs();
                assert!(
                    a.is_finite() && b.is_finite() && delta <= atol + rtol * f64::from(b).abs(),
                    "{name}[{index}]: {a} versus {b}, gate {gate}"
                );
                maximum = maximum.max(delta);
            }
            maximum
        }
        fn checkpoint(&self) {
            for (name, pin) in self.manifest["model"]["file_pins"].as_object().unwrap() {
                let mut file = fs::File::open(root().join(MODEL).join(name)).unwrap();
                let mut hash = Sha256::new();
                let mut buffer = [0u8; 65536];
                loop {
                    let count = file.read(&mut buffer).unwrap();
                    if count == 0 {
                        break;
                    }
                    hash.update(&buffer[..count]);
                }
                assert_eq!(
                    format!("{:x}", hash.finalize()),
                    pin.as_str().unwrap(),
                    "{name}"
                );
            }
        }
    }
    fn model(reference: &Reference) -> metal::Model {
        reference.checkpoint();
        metal::Model::load_mlx(
            &root().join(MODEL),
            LoadOptions {
                context_limit: 32,
                memory_limit_bytes: BUDGET,
            },
        )
        .unwrap()
    }
    fn options(chunk: usize, offset: usize) -> session::Options {
        session::Options {
            capacity: 32 - offset,
            prefill_chunk_size: chunk,
            position_offset: offset,
            memory_limit_bytes: BUDGET,
        }
    }
    fn make_session(
        model: &metal::Model,
        chunk: usize,
        offset: usize,
        readback: ReadbackMode,
        attention: AttentionMode,
    ) -> metal::Session<'_> {
        metal::Session::with_attention_mode(
            model,
            options(chunk, offset),
            ProjectionMode::MlxAffine,
            NormalizationMode::Compensated,
            readback,
            attention,
        )
        .unwrap()
    }
    fn stage(case: &str, d: &metal::Diagnostic) -> String {
        let name = match d.stage {
            "q" => "q_projection",
            "k" => "k_projection",
            "v" => "v_projection",
            "gate" => "gate_projection",
            "silu" => "gate_silu",
            "up" => "up_projection",
            "down" => "down_projection",
            other => other,
        };
        match d.layer {
            Some(layer) => format!("cases.{case}.layers.{layer}.{name}"),
            None => format!("cases.{case}.{name}"),
        }
    }
    fn tokens(value: &Value) -> Vec<u32> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect()
    }

    #[test]
    #[ignore = "Explicit Apple GPU and immutable public checkpoint: full selected-stage/chunk/offset parity"]
    fn native_mlx_qwen3_real_graph_matches_prefrozen_cpu_reference() {
        let reference = Reference::open();
        let model = model(&reference);
        assert!(metal::Session::new(&model, options(3, 0)).is_err());
        // Exact artifact payload plus seven Half guard elements on each of
        // the 113 floating norms; tied weights occur once.
        assert_eq!(model.weights_bytes(), 633_366_062);
        assert_eq!(
            model.mlx_architecture(),
            Some(toklet::mlx::dense::Architecture::Qwen3)
        );
        assert!(
            metal::Session::with_attention_mode(
                &model,
                options(3, 0),
                ProjectionMode::MlxAffine,
                NormalizationMode::Compensated,
                ReadbackMode::FullLogits,
                AttentionMode::Online
            )
            .is_err()
        );
        let mut snapshots = 0;
        let mut values = 0;
        let mut maximum = 0.0f64;
        let mut admitted = 0;
        for (case, spec) in reference.manifest["cases"].as_object().unwrap() {
            let tokens = tokens(&spec["tokens"]);
            let offset = spec["offset"].as_u64().unwrap() as usize;
            let chunks: BTreeSet<_> = [tokens.len(), 3, 1].into_iter().collect();
            for chunk in chunks {
                let mut session = make_session(
                    &model,
                    chunk,
                    offset,
                    ReadbackMode::FullLogits,
                    AttentionMode::Reference,
                );
                session.enable_diagnostics(&[0, 1, 27]).unwrap();
                admitted = admitted.max(session.memory().admitted_bytes);
                assert!(admitted <= BUDGET);
                session.set_tokens(&tokens).unwrap();
                session.evaluate().unwrap();
                let prefix = format!("cases.{case}.");
                let mut expected: BTreeSet<_> = reference.manifest["arrays"]
                    .as_object()
                    .unwrap()
                    .keys()
                    .filter(|name| {
                        name.starts_with(&prefix)
                            && !name.ends_with("attention_scores")
                            && !name.ends_with("attention_probabilities")
                    })
                    .cloned()
                    .collect();
                assert_eq!(expected.len(), 57);
                for diagnostic in session.diagnostics().unwrap() {
                    let name = stage(case, &diagnostic);
                    assert!(expected.remove(&name), "Unexpected diagnostic: {name}");
                    let shape = reference.manifest["arrays"][&name]["shape"]
                        .as_array()
                        .unwrap();
                    let columns: usize = shape[1..]
                        .iter()
                        .map(|d| d.as_u64().unwrap() as usize)
                        .product();
                    assert_eq!(diagnostic.columns, columns);
                    assert_eq!(diagnostic.rows, (tokens.len() - 1) % chunk + 1);
                    assert_eq!(diagnostic.values.len(), diagnostic.rows * columns);
                    let start = (tokens.len() - diagnostic.rows) * columns;
                    maximum = maximum.max(reference.qualify(&name, &diagnostic.values, start));
                    snapshots += 1;
                    values += diagnostic.values.len();
                }
                assert!(expected.is_empty(), "Missing diagnostics: {expected:?}");
                assert_eq!(session.evaluated_len(), tokens.len());
                assert_eq!(session.pending_len(), 0);
                let saved = session.logits().unwrap().to_vec();
                session.set_tokens(&tokens).unwrap();
                assert_eq!(session.evaluate_next_chunk().unwrap(), 0);
                assert_eq!(session.logits().unwrap(), saved);
            }
        }
        assert_eq!(snapshots, 912);
        reference.checkpoint();
        println!(
            "MLX_QWEN3_REAL_GRAPH {}",
            json!({"device":model.device_name(),"manifest_sha256":MANIFEST_SHA,"snapshots":snapshots,"compared_values":values,"max_absolute_error":maximum,"weights_bytes":model.weights_bytes(),"max_logical_session_admitted_bytes":admitted,"full_quality_qualified":false,"performance_qualified":false,"iphone_qualified":false})
        );
    }

    #[test]
    #[ignore = "Explicit Apple GPU and immutable public checkpoint: cached/reference/token-only/state qualification"]
    fn native_mlx_qwen3_real_cached_and_token_only_execution() {
        let reference = Reference::open();
        let model = model(&reference);
        let prompt = tokens(&reference.manifest["greedy"]["prompt"]);
        let expected = tokens(&reference.manifest["greedy"]["fp64_tokens"]);
        let mut selected = 0;
        let mut distributions = 0;
        let mut maximum = 0.0f64;
        for attention in [AttentionMode::Reference] {
            for readback in [ReadbackMode::FullLogits, ReadbackMode::GreedyToken] {
                let mut session = make_session(&model, 3, 0, readback, attention);
                session.set_tokens(&prompt).unwrap();
                for (step, &token) in expected.iter().enumerate() {
                    session.evaluate().unwrap();
                    match readback {
                        ReadbackMode::FullLogits => {
                            maximum = maximum.max(
                                reference.qualify(
                                    &format!("greedy.step_{step}.logits"),
                                    session
                                        .logits()
                                        .expect("FullLogits must publish a full distribution"),
                                    0,
                                ),
                            );
                            distributions += 1;
                        }
                        ReadbackMode::GreedyToken => assert!(session.logits().is_none()),
                    }
                    assert_eq!(session.greedy_token().unwrap().unwrap(), token);
                    selected += 1;
                    session.append(&[token]).unwrap();
                    assert_eq!(session.pending_len(), 1);
                    assert!(session.greedy_token().is_none());
                }
                assert_eq!(session.evaluated_len(), 7);
                assert_eq!(session.history().len(), 8);
                let history = session.history().to_vec();
                assert!(session.append(&[151936]).is_err());
                assert_eq!(session.history(), history);
                assert_eq!(session.evaluated_len(), 7);
                assert_eq!(session.pending_len(), 1);
                session.set_tokens(&history).unwrap();
                session.evaluate().unwrap();
                assert_eq!(session.evaluated_len(), 8);
                let changed = tokens(&reference.manifest["cases"]["future_changed"]["tokens"]);
                session.set_tokens(&changed).unwrap();
                session.evaluate().unwrap();
                let mut fresh = make_session(&model, 3, 0, readback, attention);
                fresh.set_tokens(&changed).unwrap();
                fresh.evaluate().unwrap();
                assert_eq!(
                    session.greedy_token().unwrap().unwrap(),
                    fresh.greedy_token().unwrap().unwrap()
                );
                match readback {
                    ReadbackMode::FullLogits => assert_eq!(
                        session
                            .logits()
                            .expect("Changed-prefix FullLogits distribution"),
                        fresh.logits().expect("Fresh FullLogits distribution"),
                    ),
                    ReadbackMode::GreedyToken => {
                        assert!(session.logits().is_none());
                        assert!(fresh.logits().is_none());
                    }
                }
                session.reset();
                assert_eq!(session.evaluated_len(), 0);
                assert!(session.history().is_empty());
            }
        }
        // A real committed partial prefill followed by a changed suffix preserves its exact prefix.
        let mut session = make_session(
            &model,
            3,
            0,
            ReadbackMode::FullLogits,
            AttentionMode::Reference,
        );
        let tile = tokens(&reference.manifest["cases"]["tile17"]["tokens"]);
        session.set_tokens(&tile).unwrap();
        assert_eq!(session.evaluate_next_chunk().unwrap(), 3);
        assert_eq!(session.evaluated_len(), 3);
        let changed = tokens(&reference.manifest["cases"]["future_changed"]["tokens"]);
        session.set_tokens(&changed).unwrap();
        assert_eq!(session.evaluated_len(), 3);
        session.evaluate().unwrap();
        let name = "cases.future_changed.logits";
        maximum = maximum.max(reference.qualify(name, session.logits().unwrap(), 3 * 151936));
        // The independent tokenizer's three next-token prefixes have full distributions.
        let scoring = tokens(&reference.manifest["cases"]["short"]["tokens"]);
        for length in 1..scoring.len() {
            session.set_tokens(&scoring[..length]).unwrap();
            session.evaluate().unwrap();
            maximum = maximum.max(reference.qualify(
                &format!("scoring.prefix_{length}.logits"),
                session.logits().unwrap(),
                0,
            ));
        }
        // Exact capacity and overflow preserve the committed prefix and output.
        session.set_tokens(&[9707; 32]).unwrap();
        session.evaluate().unwrap();
        let saved = session.logits().unwrap().to_vec();
        assert!(session.append(&[11]).is_err());
        assert_eq!(session.evaluated_len(), 32);
        assert_eq!(session.logits().unwrap(), saved);
        assert!(
            metal::Session::with_attention_mode(
                &model,
                session::Options {
                    memory_limit_bytes: 1,
                    ..options(3, 0)
                },
                ProjectionMode::MlxAffine,
                NormalizationMode::Compensated,
                ReadbackMode::FullLogits,
                AttentionMode::Reference
            )
            .is_err()
        );
        reference.checkpoint();
        assert_eq!(selected, 8);
        assert_eq!(distributions, 4);
        println!(
            "MLX_QWEN3_REAL_STATE {}",
            json!({"device":model.device_name(),"selected_ids":selected,"full_distributions":distributions,"max_logit_absolute_error":maximum,"pending_rejection_reset_changed_prefix_passed":true,"partial_prefill_committed":3,"full_quality_qualified":false,"performance_qualified":false,"iphone_qualified":false})
        );
    }
    #[test]
    #[ignore = "Explicit Apple GPU real Qwen3 public generation cancellation, output failure and hidden stops"]
    fn native_mlx_qwen3_real_generation_callbacks_preserve_pending_state() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use toklet::generation::{self, Input, Request, StopReason};
        let reference = Reference::open();
        let model = model(&reference);
        let prompt = tokens(&reference.manifest["greedy"]["prompt"]);
        let expected = tokens(&reference.manifest["greedy"]["fp64_tokens"]);
        let changed = tokens(&reference.manifest["cases"]["future_changed"]["tokens"]);
        for readback in [ReadbackMode::FullLogits, ReadbackMode::GreedyToken] {
            let create = || make_session(&model, 3, 0, readback, AttentionMode::Reference);
            let mut session = create();
            let cancel = AtomicBool::new(false);
            let mut emitted = Vec::new();
            let request = |input, max_new_tokens| Request {
                input,
                max_new_tokens,
                stop_tokens: &[],
            };
            let stopped = generation::generate(
                &mut session,
                request(Input::Prompt(&prompt), 4),
                &cancel,
                |event| {
                    assert_eq!(event.index, 0);
                    emitted.push(event.token_id);
                    cancel.store(true, Ordering::Relaxed);
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(stopped.reason, StopReason::Cancelled);
            assert_eq!((stopped.sampled_tokens, stopped.emitted_tokens), (1, 1));
            assert_eq!(
                (
                    stopped.evaluated_tokens,
                    stopped.pending_tokens,
                    stopped.history_tokens
                ),
                (4, 1, 5)
            );
            assert_eq!(
                session.history(),
                [prompt.as_slice(), &expected[..1]].concat()
            );
            let before = (
                session.history().to_vec(),
                session.evaluated_len(),
                session.pending_len(),
                session.logits().map(<[f32]>::to_vec),
                session.greedy_token(),
            );
            let precancelled = generation::generate(
                &mut session,
                request(Input::Reset(&changed), 1),
                &cancel,
                |_| panic!("Pre-cancelled reset must not emit"),
            )
            .unwrap();
            assert_eq!(precancelled.reason, StopReason::Cancelled);
            assert_eq!(
                (
                    precancelled.sampled_tokens,
                    precancelled.emitted_tokens,
                    precancelled.prefill_tokens
                ),
                (0, 0, 0)
            );
            assert_eq!(
                (
                    session.history().to_vec(),
                    session.evaluated_len(),
                    session.pending_len(),
                    session.logits().map(<[f32]>::to_vec),
                    session.greedy_token()
                ),
                before
            );
            cancel.store(false, Ordering::Relaxed);
            let resumed = generation::generate(
                &mut session,
                request(Input::Continue, 3),
                &cancel,
                |event| {
                    emitted.push(event.token_id);
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(resumed.reason, StopReason::OutputLimit);
            assert_eq!(
                (
                    resumed.reused_tokens,
                    resumed.prefill_tokens,
                    resumed.sampled_tokens,
                    resumed.emitted_tokens
                ),
                (4, 1, 3, 3)
            );
            assert_eq!(
                (
                    resumed.evaluated_tokens,
                    resumed.pending_tokens,
                    resumed.history_tokens
                ),
                (7, 1, 8)
            );
            assert_eq!(emitted, expected);
            let mut uninterrupted = create();
            let mut fresh_ids = Vec::new();
            let fresh = generation::generate(
                &mut uninterrupted,
                request(Input::Prompt(&prompt), 4),
                &cancel,
                |event| {
                    fresh_ids.push(event.token_id);
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(fresh.reason, StopReason::OutputLimit);
            assert_eq!(fresh_ids, emitted);
            assert_eq!(session.history(), uninterrupted.history());
            assert_eq!(
                (session.evaluated_len(), session.pending_len()),
                (uninterrupted.evaluated_len(), uninterrupted.pending_len())
            );
            assert!(session.logits().is_none() && uninterrupted.logits().is_none());
            assert!(session.greedy_token().is_none() && uninterrupted.greedy_token().is_none());
            session.discard_pending();
            uninterrupted.discard_pending();
            assert_eq!(session.history(), uninterrupted.history());
            assert_eq!(session.greedy_token(), uninterrupted.greedy_token());
            assert_eq!(session.greedy_token().unwrap().unwrap(), expected[3]);
            match readback {
                ReadbackMode::FullLogits => {
                    let actual = session.logits().expect("Resumed full distribution");
                    assert_eq!(
                        actual,
                        uninterrupted
                            .logits()
                            .expect("Uninterrupted full distribution")
                    );
                    reference.qualify("greedy.step_3.logits", actual, 0);
                }
                ReadbackMode::GreedyToken => {
                    assert!(session.logits().is_none() && uninterrupted.logits().is_none())
                }
            }

            // A failed sink also leaves the sampled ID pending, but returns an
            // output error rather than a successful cancellation report.
            let mut failed_ids = Vec::new();
            let error = generation::generate(
                &mut session,
                request(Input::Reset(&prompt), 4),
                &cancel,
                |event| {
                    failed_ids.push(event.token_id);
                    Err(std::io::Error::other("synthetic Qwen3 sink failure"))
                },
            )
            .unwrap_err();
            match error {
                generation::Error::Output(error) => {
                    assert_eq!(error.to_string(), "synthetic Qwen3 sink failure")
                }
                other => panic!("Expected output error, got {other}"),
            }
            assert_eq!(failed_ids, expected[..1]);
            assert_eq!((session.evaluated_len(), session.pending_len()), (4, 1));
            assert_eq!(
                session.history(),
                [prompt.as_slice(), &expected[..1]].concat()
            );
            assert!(session.logits().is_none() && session.greedy_token().is_none());
            let after_error = generation::generate(
                &mut session,
                request(Input::Continue, 1),
                &cancel,
                |event| {
                    assert_eq!(event.token_id, expected[1]);
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(after_error.reason, StopReason::OutputLimit);
            assert_eq!(
                (
                    after_error.reused_tokens,
                    after_error.prefill_tokens,
                    after_error.sampled_tokens,
                    after_error.emitted_tokens
                ),
                (4, 1, 1, 1)
            );
            generation::generate(
                &mut uninterrupted,
                request(Input::Reset(&prompt), 2),
                &cancel,
                |_| Ok(()),
            )
            .unwrap();
            assert_eq!(session.history(), uninterrupted.history());
            assert_eq!((session.evaluated_len(), session.pending_len()), (5, 1));
            session.discard_pending();
            uninterrupted.discard_pending();
            assert_eq!(session.greedy_token(), uninterrupted.greedy_token());
            assert_eq!(session.greedy_token().unwrap().unwrap(), expected[1]);
            match readback {
                ReadbackMode::FullLogits => {
                    let actual = session.logits().expect("Output-error resumed distribution");
                    assert_eq!(
                        actual,
                        uninterrupted.logits().expect("Fresh two-step distribution")
                    );
                    reference.qualify("greedy.step_1.logits", actual, 0);
                }
                ReadbackMode::GreedyToken => {
                    assert!(session.logits().is_none() && uninterrupted.logits().is_none())
                }
            }

            let hidden = generation::generate(
                &mut session,
                Request {
                    input: Input::Reset(&prompt),
                    max_new_tokens: 4,
                    stop_tokens: &expected[..1],
                },
                &cancel,
                |_| panic!("Hidden stop must not emit"),
            )
            .unwrap();
            assert_eq!(
                hidden.reason,
                StopReason::StopToken {
                    token_id: expected[0]
                }
            );
            assert_eq!(
                (
                    hidden.sampled_tokens,
                    hidden.emitted_tokens,
                    hidden.evaluated_tokens,
                    hidden.pending_tokens,
                    hidden.history_tokens
                ),
                (1, 0, 4, 1, 5)
            );
            assert_eq!(
                session.history(),
                [prompt.as_slice(), &expected[..1]].concat()
            );
            let continued = generation::generate(
                &mut session,
                request(Input::Continue, 1),
                &cancel,
                |event| {
                    assert_eq!(event.token_id, expected[1]);
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(
                (
                    continued.reused_tokens,
                    continued.prefill_tokens,
                    continued.sampled_tokens,
                    continued.emitted_tokens
                ),
                (4, 1, 1, 1)
            );
            assert_eq!(
                session.history(),
                [prompt.as_slice(), &expected[..2]].concat()
            );
            assert_eq!((session.evaluated_len(), session.pending_len()), (5, 1));
            session.discard_pending();
            assert_eq!(session.greedy_token().unwrap().unwrap(), expected[1]);
            let before = (
                session.history().to_vec(),
                session.evaluated_len(),
                session.pending_len(),
                session.logits().map(<[f32]>::to_vec),
                session.greedy_token(),
            );
            for invalid in [
                request(Input::Reset(&[151936]), 1),
                request(Input::Continue, usize::MAX),
                Request {
                    input: Input::Continue,
                    max_new_tokens: 1,
                    stop_tokens: &[151936],
                },
            ] {
                assert!(matches!(
                    generation::generate(&mut session, invalid, &cancel, |_| panic!(
                        "Invalid request must not emit"
                    )),
                    Err(generation::Error::InvalidRequest(_))
                ));
                assert_eq!(
                    (
                        session.history().to_vec(),
                        session.evaluated_len(),
                        session.pending_len(),
                        session.logits().map(<[f32]>::to_vec),
                        session.greedy_token()
                    ),
                    before
                );
            }
            println!(
                "MLX_QWEN3_REAL_GENERATION {}",
                json!({"readback":format!("{readback:?}"),"callback_cancelled_after_emitted_tokens":1,"cancelled_evaluated_tokens":4,"cancelled_pending_tokens":1,"exact_uninterrupted_replay":true,"output_failure_and_hidden_stop_resume":true,"precancelled_reset_and_invalid_requests_preserve_state":true,"mid_gpu_or_prefill_cancellation_qualified":false,"iphone_qualified":false})
            );
        }
        reference.checkpoint();
    }

    #[test]
    #[ignore = "Explicit Apple GPU real Qwen3 exact session and diagnostic budget boundaries"]
    fn native_mlx_qwen3_real_memory_boundaries_preserve_admitted_state() {
        let reference = Reference::open();
        let model = model(&reference);
        for readback in [ReadbackMode::FullLogits, ReadbackMode::GreedyToken] {
            let make = |limit| {
                metal::Session::with_attention_mode(
                    &model,
                    session::Options {
                        memory_limit_bytes: limit,
                        ..options(3, 0)
                    },
                    ProjectionMode::MlxAffine,
                    NormalizationMode::Compensated,
                    readback,
                    AttentionMode::Reference,
                )
            };
            let baseline = make(BUDGET).unwrap();
            let plan = metal::planning::inspect_mlx(
                &root().join(MODEL),
                LoadOptions {
                    context_limit: 32,
                    memory_limit_bytes: BUDGET,
                },
                options(3, 0),
                NormalizationMode::Compensated,
                readback,
                AttentionMode::Reference,
            )
            .unwrap();
            assert_eq!(plan.resident_weights_bytes, 633_366_062);
            assert_eq!(plan.resident_weights_bytes, model.weights_bytes());
            assert_eq!(plan.session_memory, baseline.memory());
            assert!(plan.fits_requested_budgets);
            println!(
                "MLX_QWEN3_REAL_RESOURCE_PLAN {}",
                serde_json::to_string(&plan).unwrap()
            );
            let exact = baseline.memory().admitted_bytes;
            assert!(
                matches!(make(exact-1), Err(metal::Error::Budget { required, limit }) if required == exact && limit == exact-1)
            );
            assert_eq!(make(exact).unwrap().memory(), baseline.memory());
            drop(baseline);
            let mut diagnostic = make(BUDGET).unwrap();
            diagnostic.enable_diagnostics(&[0, 1, 27]).unwrap();
            let with_diagnostics = diagnostic.memory().admitted_bytes;
            drop(diagnostic);
            let mut below = make(with_diagnostics - 1).unwrap();
            assert!(
                matches!(below.enable_diagnostics(&[0,1,27]), Err(metal::Error::Budget { required, limit }) if required == with_diagnostics && limit == with_diagnostics-1)
            );
            assert_eq!(below.memory().admitted_bytes, exact);
            assert_eq!(below.evaluated_len(), 0);
            assert_eq!(below.pending_len(), 0);
            drop(below);
            let mut admitted = make(with_diagnostics).unwrap();
            admitted.enable_diagnostics(&[0, 1, 27]).unwrap();
            assert_eq!(admitted.memory().admitted_bytes, with_diagnostics);
            let prompt = tokens(&reference.manifest["cases"]["short"]["tokens"]);
            admitted.set_tokens(&prompt).unwrap();
            admitted.evaluate().unwrap();
            let logits = admitted
                .diagnostics()
                .unwrap()
                .into_iter()
                .find(|entry| entry.stage == "logits")
                .unwrap();
            reference.qualify("cases.short.logits", &logits.values, 3 * 151936);
            assert_eq!(admitted.greedy_token().unwrap().unwrap(), 1096);
            println!(
                "MLX_QWEN3_REAL_BUDGET {}",
                json!({"readback":format!("{readback:?}"),"session_admitted_bytes":exact,"diagnostic_admitted_bytes":with_diagnostics,"exact_and_limit_minus_one_passed":true})
            );
        }
        reference.checkpoint();
    }

    #[test]
    #[ignore = "Explicit Apple GPU diagnostic-only Qwen3 1025 capacity/state; no numerical or quality qualification"]
    fn native_qwen3_quality_diagnostic_1025_preserves_boundaries_and_pending_state() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use toklet::generation::{self, Input, Request, StopReason};
        let reference = Reference::open();
        reference.checkpoint();
        let load = LoadOptions {
            context_limit: 1025,
            memory_limit_bytes: BUDGET,
        };
        let directory = root().join(MODEL);
        assert!(matches!(
            metal::Model::load_mlx(&directory, load),
            Err(metal::Error::Unsupported(
                "Metal Qwen3 requires the admitted tiny/0.6B geometry, context and ordinary RoPE"
            ))
        ));
        assert!(matches!(
            metal::Model::load_mlx_qwen3_quality_diagnostic(
                &directory,
                LoadOptions {
                    context_limit: 1026,
                    ..load
                }
            ),
            Err(metal::Error::Unsupported(
                "quality diagnostic requires Qwen3 0.6B and at most 1025 positions"
            ))
        ));
        let model = metal::Model::load_mlx_qwen3_quality_diagnostic(&directory, load).unwrap();
        assert_eq!(model.config().max_positions, 1025);
        assert_eq!(model.mlx_declared_positions(), Some(40960));
        let options = session::Options {
            capacity: 1025,
            prefill_chunk_size: 128,
            position_offset: 0,
            memory_limit_bytes: BUDGET,
        };
        let create = |options| {
            metal::Session::with_attention_mode(
                &model,
                options,
                ProjectionMode::MlxAffine,
                NormalizationMode::Compensated,
                ReadbackMode::FullLogits,
                AttentionMode::Reference,
            )
        };
        for invalid in [
            session::Options {
                capacity: 1026,
                ..options
            },
            session::Options {
                position_offset: 1,
                ..options
            },
        ] {
            assert!(matches!(
                create(invalid),
                Err(metal::Error::Options(
                    "Metal session options outside admitted model/context bounds"
                ))
            ));
        }
        let mut active = create(options).unwrap();
        let admitted = active.memory().admitted_bytes;
        assert!(admitted <= BUDGET);
        assert!(matches!(
            create(session::Options {
                memory_limit_bytes: admitted - 1,
                ..options
            }),
            Err(metal::Error::Budget { required, limit }) if required == admitted && limit == admitted - 1
        ));
        let exact = create(session::Options {
            memory_limit_bytes: admitted,
            ..options
        })
        .unwrap();
        assert_eq!(exact.memory().admitted_bytes, admitted);
        drop(exact);
        let prompt: Vec<_> = [9707, 11, 1879, 0].into_iter().cycle().take(1025).collect();
        active.set_tokens(&prompt[..1024]).unwrap();
        assert_eq!(active.evaluate_next_chunk().unwrap(), 128);
        active.discard_pending();
        assert_eq!(active.history(), &prompt[..128]);
        assert_eq!((active.evaluated_len(), active.pending_len()), (128, 0));
        active.append(&prompt[128..1024]).unwrap();
        active.evaluate().unwrap();
        let prefix_logits = active.logits().unwrap().to_vec();
        active.append(&prompt[1024..]).unwrap();
        active.evaluate().unwrap();
        let full_logits = active.logits().unwrap().to_vec();
        assert!(full_logits.iter().all(|v| v.is_finite()));
        assert_eq!((active.evaluated_len(), active.pending_len()), (1025, 0));
        let assert_full = |s: &metal::Session<'_>| {
            assert_eq!(s.history(), prompt);
            assert_eq!((s.evaluated_len(), s.pending_len()), (1025, 0));
            assert_eq!(s.logits().unwrap(), full_logits);
        };
        assert!(matches!(
            active.append(&[11]),
            Err(metal::Error::Options(
                "token ID/count outside session capacity"
            ))
        ));
        assert_full(&active);
        assert!(matches!(
            active.set_tokens(&[9707; 1026]),
            Err(metal::Error::Options(
                "token ID/count outside session capacity"
            ))
        ));
        assert_full(&active);
        active.set_tokens(&prompt).unwrap();
        assert_eq!(active.evaluate_next_chunk().unwrap(), 0);
        drop(active);

        // A different chunk schedule and future tail must preserve the exact
        // preceding distribution. This tests causal state, not source parity.
        let mut replay = create(session::Options {
            prefill_chunk_size: 37,
            ..options
        })
        .unwrap();
        replay.enable_diagnostics(&[]).unwrap();
        let diagnostic_admitted = replay.memory().admitted_bytes;
        assert!(diagnostic_admitted <= BUDGET);
        replay.set_tokens(&prompt[..1024]).unwrap();
        replay.evaluate().unwrap();
        assert_eq!(replay.logits().unwrap(), prefix_logits);
        replay.append(&prompt[1024..]).unwrap();
        replay.evaluate().unwrap();
        assert_eq!(replay.logits().unwrap(), full_logits);
        replay.reset();
        let mut changed = prompt.clone();
        changed[1024] = 17;
        replay.set_tokens(&changed).unwrap();
        replay.evaluate().unwrap();
        let snapshots = replay.diagnostics().unwrap();
        let logits = snapshots.iter().find(|s| s.stage == "logits").unwrap();
        assert_eq!(logits.rows, 26);
        assert_eq!(logits.columns, 151936);
        let row = logits.rows - 2;
        assert_eq!(
            &logits.values[row * logits.columns..(row + 1) * logits.columns],
            prefix_logits
        );
        assert_eq!(replay.evaluated_len(), 1025);
        drop(snapshots);
        drop(replay);

        // One sampled output at the boundary remains pending. Cancellation,
        // overflow rejection and later commitment preserve its valid prefix.
        let mut generated = create(options).unwrap();
        let cancel = AtomicBool::new(false);
        let stopped = generation::generate(
            &mut generated,
            Request {
                input: Input::Prompt(&prompt[..1024]),
                max_new_tokens: 1,
                stop_tokens: &[],
            },
            &cancel,
            |_| {
                cancel.store(true, Ordering::Relaxed);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(stopped.reason, StopReason::Cancelled);
        assert_eq!((stopped.sampled_tokens, stopped.emitted_tokens), (1, 1));
        assert_eq!(
            (
                generated.evaluated_len(),
                generated.pending_len(),
                generated.history().len()
            ),
            (1024, 1, 1025)
        );
        let history = generated.history().to_vec();
        let generated_logits = generated.logits().map(<[f32]>::to_vec);
        assert!(generated_logits.is_none()); // Sampled history is still pending.
        cancel.store(false, Ordering::Relaxed);
        assert!(matches!(
            generation::generate(
                &mut generated,
                Request {
                    input: Input::Continue,
                    max_new_tokens: 1,
                    stop_tokens: &[],
                },
                &cancel,
                |_| panic!("full diagnostic must reject before emission")
            ),
            Err(generation::Error::InvalidRequest(
                "prompt plus requested output exceeds session capacity"
            ))
        ));
        assert_eq!(generated.history(), history);
        assert_eq!(
            (generated.evaluated_len(), generated.pending_len()),
            (1024, 1)
        );
        assert_eq!(generated.logits().map(<[f32]>::to_vec), generated_logits);
        generated.evaluate().unwrap();
        assert_eq!(
            (generated.evaluated_len(), generated.pending_len()),
            (1025, 0)
        );
        drop(generated);
        reference.checkpoint();
        println!(
            "QWEN3_QUALITY_DIAGNOSTIC_1025 {}",
            json!({
                "device":model.device_name(),"capacity":1025,"chunks":[128,37],
                "exact_chunk_prefix_replay":true,"maximum_plus_one_rejected":true,
                "pending_cancel_commit_checked":true,"session_admitted_bytes":admitted,
                "diagnostic_session_admitted_bytes":diagnostic_admitted,
                "experimental_only":true,"c512_numerical_qualified":false,
                "known_c512_numerical_gate_status":"rejected_unchanged",
                "quality_qualified":false,"app_limits_changed":false,"iphone_qualified":false
            })
        );
    }

    #[test]
    #[ignore = "Explicit Apple GPU real Qwen3 512 capacity/resume/replay witness, separate from short numerical oracle"]
    fn native_mlx_qwen3_real_512_capacity_preserves_committed_state() {
        let reference = Reference::open();
        reference.checkpoint();
        let model = metal::Model::load_mlx(
            &root().join(MODEL),
            LoadOptions {
                context_limit: 512,
                memory_limit_bytes: BUDGET,
            },
        )
        .unwrap();
        let options = session::Options {
            capacity: 512,
            prefill_chunk_size: 128,
            position_offset: 0,
            memory_limit_bytes: BUDGET,
        };
        let create = || {
            metal::Session::with_attention_mode(
                &model,
                options,
                ProjectionMode::MlxAffine,
                NormalizationMode::Compensated,
                ReadbackMode::FullLogits,
                AttentionMode::Reference,
            )
            .unwrap()
        };
        let prompt: Vec<_> = [9707, 11, 1879, 0].into_iter().cycle().take(512).collect();
        let mut session = create();
        session.set_tokens(&prompt).unwrap();
        assert_eq!(session.evaluate_next_chunk().unwrap(), 128);
        assert_eq!(session.evaluated_len(), 128);
        assert_eq!(session.pending_len(), 384);
        session.discard_pending();
        assert_eq!(session.history(), &prompt[..128]);
        assert_eq!(session.pending_len(), 0);
        session.append(&prompt[128..]).unwrap();
        session.evaluate().unwrap();
        assert_eq!(session.history(), prompt);
        assert_eq!(session.evaluated_len(), 512);
        assert_eq!(session.pending_len(), 0);
        let saved = session.logits().unwrap().to_vec();
        assert!(session.append(&[11]).is_err());
        assert!(session.set_tokens(&[9707; 513]).is_err());
        assert_eq!(session.evaluated_len(), 512);
        assert_eq!(session.history(), prompt);
        assert_eq!(session.logits().unwrap(), saved);
        session.set_tokens(&prompt).unwrap();
        assert_eq!(session.evaluate_next_chunk().unwrap(), 0);
        assert_eq!(session.logits().unwrap(), saved);
        let admitted = session.memory().admitted_bytes;
        assert!(admitted <= BUDGET);
        drop(session);
        let mut fresh = create();
        fresh.set_tokens(&prompt).unwrap();
        fresh.evaluate().unwrap();
        assert_eq!(fresh.logits().unwrap(), saved);
        reference.checkpoint();
        println!(
            "MLX_QWEN3_REAL_512 {}",
            json!({"device":model.device_name(),
            "capacity":512,"chunk":128,"partial_prefill_committed":128,
            "exact_fresh_full_distribution":true,"max_logical_session_admitted_bytes":admitted,
            "long_context_numerical_oracle_qualified":false,"iphone_qualified":false})
        );
    }
}
