//! Explicit selected-checkpoint qualification; ordinary tests need only the manifest.
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const MANIFEST_SHA: &str = "3cd3af44e0d4d57cc7cf23c8dd7cc7b60ccf0968e48bfae90adba35517e039bd";
const ARRAY_SHA: &str = "4f471fe2f15c257c8fbf4169a88b99b7ba28fa1e9bf881eaa91f77a7f8cdcca2";
fn manifest() -> Value {
    let bytes = include_bytes!("fixtures/mlx-smollm-real-v1/manifest.json");
    assert_eq!(format!("{:x}", Sha256::digest(bytes)), MANIFEST_SHA);
    serde_json::from_slice(bytes).unwrap()
}

#[test]
fn smollm_mlx_manifest_freezes_complete_independent_schema_and_gates() {
    let m = manifest();
    assert_eq!(m["format"], "toklet-mlx-smollm-real-reference-v1");
    assert_eq!(
        m["model"]["repository"],
        "mlx-community/SmolLM2-135M-Instruct-8bit"
    );
    assert_eq!(
        m["model"]["revision"],
        "0f0d9b8218915bc34d401e1a340b8c049d300d5e"
    );
    assert_eq!(m["array_sha256"], ARRAY_SHA);
    assert_eq!(m["array_bytes"], 13_945_632);
    assert_eq!(m["layers"], json!([0, 1, 29]));
    assert_eq!(m["greedy"]["fp32_tokens"], json!([339, 3543, 719, 1891]));
    assert_eq!(m["greedy"]["fp64_tokens"], m["greedy"]["fp32_tokens"]);
    assert_eq!(m["cases"].as_object().unwrap().len(), 6);
    assert_eq!(m["arrays"].as_object().unwrap().len(), 349);
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
    assert_eq!(cursor, 13_945_632);
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
    const MODEL: &str = "models/MLX-SmolLM2-135M-Instruct-8bit-0f0d9b8";
    struct Reference {
        manifest: Value,
        bytes: Vec<u8>,
    }
    impl Reference {
        fn open() -> Self {
            let manifest = manifest();
            let path = root().join("artifacts/m8-smollm-reference-v1/arrays.bin");
            let mut file =
                fs::File::open(path).expect("Run the explicitly reviewed CPU export first");
            let mut bytes = Vec::new();
            file.by_ref()
                .take(13_945_633)
                .read_to_end(&mut bytes)
                .unwrap();
            assert_eq!(bytes.len(), 13_945_632);
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
    fn native_mlx_smollm_graph_matches_prefrozen_cpu_reference() {
        let reference = Reference::open();
        let model = model(&reference);
        assert!(metal::Session::new(&model, options(3, 0)).is_err());
        // Exact artifact payload plus seven Half guard elements on each of
        // the 61 floating norms; tied weights occur once.
        assert_eq!(model.weights_bytes(), 142_955_990);
        assert_eq!(
            model.mlx_architecture(),
            Some(toklet::mlx::dense::Architecture::Llama)
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
                session.enable_diagnostics(&[0, 1, 29]).unwrap();
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
                assert_eq!(expected.len(), 51);
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
        assert_eq!(snapshots, 816);
        reference.checkpoint();
        println!(
            "MLX_SMOLLM_GRAPH {}",
            json!({"device":model.device_name(),"manifest_sha256":MANIFEST_SHA,"snapshots":snapshots,"compared_values":values,"max_absolute_error":maximum,"weights_bytes":model.weights_bytes(),"max_logical_session_admitted_bytes":admitted,"full_quality_qualified":false,"performance_qualified":false,"iphone_qualified":false})
        );
    }

    #[test]
    #[ignore = "Explicit Apple GPU and immutable public checkpoint: cached/reference/token-only/state qualification"]
    fn native_mlx_smollm_cached_and_token_only_execution() {
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
                assert!(session.append(&[49152]).is_err());
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
        maximum = maximum.max(reference.qualify(name, session.logits().unwrap(), 3 * 49152));
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
        session.set_tokens(&[19556; 32]).unwrap();
        session.evaluate().unwrap();
        let saved = session.logits().unwrap().to_vec();
        assert!(session.append(&[28]).is_err());
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
            "MLX_SMOLLM_STATE {}",
            json!({"device":model.device_name(),"selected_ids":selected,"full_distributions":distributions,"max_logit_absolute_error":maximum,"pending_rejection_reset_changed_prefix_passed":true,"partial_prefill_committed":3,"full_quality_qualified":false,"performance_qualified":false,"iphone_qualified":false})
        );
    }
    #[test]
    #[ignore = "Explicit Apple GPU real SmolLM512 capacity/resume/replay witness, separate from short numerical oracle"]
    fn native_mlx_smollm_512_capacity_preserves_committed_state() {
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
        let prompt: Vec<_> = [19556, 28, 905, 17].into_iter().cycle().take(512).collect();
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
        assert!(session.append(&[28]).is_err());
        assert!(session.set_tokens(&[19556; 513]).is_err());
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
            "MLX_SMOLLM_512 {}",
            json!({"device":model.device_name(),
            "capacity":512,"chunk":128,"partial_prefill_committed":128,
            "exact_fresh_full_distribution":true,"max_logical_session_admitted_bytes":admitted,
            "long_context_numerical_oracle_qualified":false,"iphone_qualified":false})
        );
    }
}
