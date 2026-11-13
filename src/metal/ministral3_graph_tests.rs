//! Explicit packed graph composition witnesses; no tiny source admission.
use super::*;
use crate::mlx::affine::{Layout as AffineLayout, Matrix, MetadataDtype};
use half::{bf16, f16};
use serde_json::Value;
use sha2::{Digest, Sha256};

fn fixture() -> Value {
    let value: Value = serde_json::from_slice(include_bytes!(
        "../../tests/fixtures/ministral3-graph-v1/fixture.json"
    ))
    .unwrap();
    assert_eq!(value["format"], "toklet-ministral3-half-graph-v1");
    for (path, bytes) in [
        (
            "tools/reference/ministral3_graph_fixture.py",
            include_bytes!("../../tools/reference/ministral3_graph_fixture.py").as_slice(),
        ),
        (
            "tools/reference/ministral3_graph_protocol.py",
            include_bytes!("../../tools/reference/ministral3_graph_protocol.py").as_slice(),
        ),
        (
            "tools/reference/ministral3_rotary_fixture.py",
            include_bytes!("../../tools/reference/ministral3_rotary_fixture.py").as_slice(),
        ),
    ] {
        assert_eq!(value["producers"][path]["bytes"], bytes.len() as u64);
        assert_eq!(
            value["producers"][path]["sha256"],
            format!("{:x}", Sha256::digest(bytes))
        );
    }
    let profile = include_bytes!("../../tests/fixtures/ministral3-rotary-v1/profile.json");
    let binding = &value["inputs"]["tests/fixtures/ministral3-rotary-v1/profile.json"];
    assert_eq!(binding["bytes"], profile.len() as u64);
    assert_eq!(binding["sha256"], format!("{:x}", Sha256::digest(profile)));
    assert_eq!(
        value["schedules"]["chunks"],
        serde_json::json!([1, 2, 3, 4])
    );
    assert_eq!(value["comparison"]["both_profiles_required"], true);
    assert_eq!(
        value["comparison"]["native_derived_enlargement_forbidden"],
        true
    );
    let control = &value["composition_control"];
    assert_eq!(control["diagnostic_only"], true);
    let high = value["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == control["case"])
        .unwrap();
    let target = &high["arrays"][control["target_stage"].as_str().unwrap()];
    for profile in ["main_bits", "wide_bits"] {
        let expected = words16(&target[profile]);
        let wrong = words16(&control["wrong_keys_kind"][profile]);
        let atol = target["gate"]["atol"].as_f64().unwrap();
        assert!(
            expected.iter().zip(&wrong).any(|(&a, &b)| {
                let expected = f64::from(f16::from_bits(a).to_f32());
                let wrong = f64::from(f16::from_bits(b).to_f32());
                (expected - wrong).abs() > atol + 1e-3 * expected.abs()
            }),
            "high-position wrong Keys kind must fail the original query gate"
        );
    }
    value
}

fn number(value: &Value, key: &str) -> usize {
    value[key].as_u64().unwrap() as usize
}

fn words16(value: &Value) -> Vec<u16> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u16::try_from(v.as_u64().unwrap()).unwrap())
        .collect()
}

fn bytes16(value: &Value) -> Vec<u8> {
    words16(value)
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect()
}

fn bytes32(value: &Value) -> Vec<u8> {
    value
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|v| u32::try_from(v.as_u64().unwrap()).unwrap().to_le_bytes())
        .collect()
}

fn tokens(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
        .collect()
}

fn tiny_model(fixture: &Value) -> Model {
    let c = &fixture["config"];
    let config = model::Config {
        hidden_size: number(c, "hidden"),
        intermediate_size: number(c, "intermediate"),
        num_layers: number(c, "layers"),
        query_heads: number(c, "query_heads"),
        kv_heads: number(c, "kv_heads"),
        head_dim: number(c, "head_dim"),
        vocab_size: number(c, "vocabulary"),
        max_positions: 512,
        rms_norm_eps: c["rms_eps"].as_f64().unwrap() as f32,
        rope_theta: c["theta"].as_f64().unwrap() as f32,
    };
    assert_eq!(
        (
            config.hidden_size,
            config.intermediate_size,
            config.num_layers,
            config.query_heads,
            config.kv_heads,
            config.head_dim,
            config.vocab_size
        ),
        (64, 128, 1, 2, 1, 128, 64)
    );
    config.validate().unwrap();
    let gpu = Gpu::new().unwrap();
    let pipelines = mlx::compile(&gpu).unwrap();
    let plan = positional::Plan::ministral3(512).unwrap();
    let position = plan.upload(&gpu, plan.compile(&gpu).unwrap()).unwrap();
    assert_eq!(position.owned_bytes(), plan.owned_bytes());
    let mut tensors = BTreeMap::new();
    let mut weight_bytes = 0;
    for (name, spec) in fixture["weights"].as_object().unwrap() {
        let shape = spec["shape"].as_array().unwrap();
        let (rows, columns) = if shape.len() == 1 {
            (1, shape[0].as_u64().unwrap() as usize)
        } else {
            (
                shape[0].as_u64().unwrap() as usize,
                shape[1].as_u64().unwrap() as usize,
            )
        };
        let layout = Layout::new(rows, columns, true).unwrap();
        let tensor = if shape.len() == 1 {
            let words = words16(&spec["bf16_words"]);
            assert_eq!(words.len(), columns);
            let mut tensor = Tensor::allocate(&gpu, layout).unwrap();
            tensor.buffer.with_mut(|out| {
                for (out, word) in out.iter_mut().zip(words) {
                    *out = f16::from_f32(bf16::from_bits(word).to_f32()).to_bits();
                }
            });
            weight_bytes += layout.bytes(2).unwrap();
            tensor
        } else {
            let bits = u8::try_from(number(spec, "bits")).unwrap();
            let expected_bits = if name == "model.embed_tokens.weight"
                || name.ends_with(".self_attn.o_proj.weight")
                || name.ends_with(".mlp.down_proj.weight")
            {
                8
            } else {
                4
            };
            assert_eq!(bits, expected_bits);
            let affine = AffineLayout::new(
                rows,
                columns,
                bits,
                number(spec, "group_size"),
                MetadataDtype::BF16,
            )
            .unwrap();
            let weights = bytes32(&spec["weight_words"]);
            let scales = bytes16(&spec["scales_bf16"]);
            let biases = bytes16(&spec["biases_bf16"]);
            let matrix = Matrix::from_le_bytes(affine, &weights, &scales, &biases).unwrap();
            let mut buffer = Buffer::new(&gpu, affine.weight_bytes() / 2).unwrap();
            let mut packed = mlx::Packed::allocate(&gpu, affine, &pipelines).unwrap();
            packed.upload_matrix(&matrix, &mut buffer).unwrap();
            weight_bytes += mlx::packed_bytes(affine).unwrap();
            Tensor {
                buffer,
                layout,
                storage: TensorStorage::Mlx(packed),
            }
        };
        tensors.insert(name.clone(), tensor);
    }
    assert_eq!(tensors.len(), 11);
    let mut model =
        Model::assemble(config, gpu, tensors, weight_bytes, true, false, false).unwrap();
    model.positional = Some(position);
    model.mlx_architecture = Some(crate::mlx::dense::Architecture::Mistral);
    assert!(model.output.is_none());
    model
}

fn session(model: &Model, chunk: usize, offset: usize) -> Session<'_> {
    Session::with_attention_mode(
        model,
        session::Options {
            capacity: 512 - offset,
            prefill_chunk_size: chunk,
            position_offset: offset,
            memory_limit_bytes: weights::INITIAL_MEMORY_LIMIT_BYTES,
        },
        ProjectionMode::MlxAffine,
        NormalizationMode::Compensated,
        ReadbackMode::FullLogits,
        AttentionMode::Reference,
    )
    .unwrap()
}

fn prefix(session: &Session<'_>) -> Vec<u16> {
    prefix_len(session, session.evaluated_len())
}

fn prefix_len(session: &Session<'_>, count: usize) -> Vec<u16> {
    assert!(count <= session.evaluated_len());
    let c = &session.model.config;
    let mut result = Vec::new();
    for cache in &session.cache {
        for buffer in [&cache.k, &cache.v] {
            buffer.with_slice(|data| {
                for head in 0..c.kv_heads {
                    let base = head * session.capacity() * c.head_dim;
                    result.extend_from_slice(&data[base..base + count * c.head_dim]);
                }
            });
        }
    }
    result
}

fn tensor_bits(value: &Value, key: &str) -> Vec<f64> {
    let half = value["dtype"] == "f16le";
    value[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            if half {
                f64::from(f16::from_bits(u16::try_from(v.as_u64().unwrap()).unwrap()).to_f32())
            } else {
                f64::from(f32::from_bits(u32::try_from(v.as_u64().unwrap()).unwrap()))
            }
        })
        .collect()
}

fn compare_stage(witness: &Value, start: usize, actual: &[f32], label: &str) {
    let main = tensor_bits(witness, "main_bits");
    let wide = tensor_bits(witness, "wide_bits");
    assert_eq!(main.len(), wide.len());
    let envelope = main
        .iter()
        .zip(&wide)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    assert_eq!(witness["precision_envelope"].as_f64().unwrap(), envelope);
    let atol = witness["gate"]["atol"].as_f64().unwrap();
    let rtol = witness["gate"]["rtol"].as_f64().unwrap();
    assert_eq!(atol, (2. * envelope).max(1e-4));
    assert_eq!(rtol, 1e-3);
    for (profile, expected) in [("main", main), ("wide", wide)] {
        assert!(start + actual.len() <= expected.len());
        for (index, (&got, &want)) in actual.iter().zip(&expected[start..]).enumerate() {
            assert!(
                got.is_finite()
                    && want.is_finite()
                    && (f64::from(got) - want).abs() <= atol + rtol * want.abs(),
                "{label}/{profile}/{index}: {got} vs {want}"
            );
        }
    }
}

#[test]
fn selected_loader_and_position_plans_reject_unadmitted_capacity_before_gpu() {
    for context_limit in [0, 513, usize::MAX] {
        assert!(matches!(
            Model::load_ministral3_diagnostic(
                Path::new("/nonexistent-ministral3"),
                weights::LoadOptions {
                    context_limit,
                    ..Default::default()
                }
            ),
            Err(Error::Options(
                "Ministral 3 diagnostic requires1..=512 positions"
            ))
        ));
        assert!(positional::Plan::ministral3(context_limit).is_err());
    }
    for memory_limit_bytes in [0, weights::INITIAL_MEMORY_LIMIT_BYTES + 1] {
        assert!(matches!(
            Model::load_ministral3_diagnostic(
                Path::new("/nonexistent-ministral3"),
                weights::LoadOptions {
                    context_limit: 512,
                    memory_limit_bytes
                }
            ),
            Err(Error::Options("Ministral 3 diagnostic budget must fit6GiB"))
        ));
    }
    let plan = positional::Plan::ministral3(512).unwrap();
    assert!(plan.setup_bytes().unwrap() < 512 * 1024 * 1024);
    assert_eq!(
        plan.pipeline_allowance(),
        ministral3_rotary::PIPELINE_ALLOWANCE
    );
    assert!(plan.owned_bytes() > 512 * (128 * 4 + 2));
    let largest = plan.validate_buffers(usize::MAX).unwrap();
    assert!(plan.validate_buffers(largest - 1).is_err());
}

#[test]
#[ignore = "Explicit Apple GPU independent mixed packed Ministral distinct-width graph"]
fn packed_graph_matches_both_independent_profiles_for_every_stage() {
    autoreleasepool(|_| {
        let fixture = fixture();
        let model = tiny_model(&fixture);
        let (mut schedules, mut commands, mut arrays, mut coordinates) = (0, 0, 0, 0);
        for case in fixture["cases"].as_array().unwrap() {
            let offset = number(case, "position_offset");
            if offset >= 512 {
                continue;
            } // High positions are a separate primitive control.
            let ids = tokens(&case["tokens"]);
            for chunk in [1, 2, 3, 4]
                .into_iter()
                .filter(|&chunk| chunk <= 512 - offset)
            {
                let mut state = session(&model, chunk, offset);
                schedules += 1;
                state.enable_diagnostics(&[0]).unwrap();
                state.enable_profile().unwrap();
                state.append(&ids).unwrap();
                let mut start = 0;
                while state.pending_len() > 0 {
                    let count = state.evaluate_next_chunk().unwrap();
                    commands += 1;
                    assert_eq!(state.timing().unwrap().command_count, 1);
                    let profile = state.profile().unwrap();
                    assert_eq!(
                        profile
                            .events
                            .iter()
                            .filter(|e| e.name == "ministral3_rotary_queries")
                            .count(),
                        1
                    );
                    assert_eq!(
                        profile
                            .events
                            .iter()
                            .filter(|e| e.name == "ministral3_rotary_keys")
                            .count(),
                        1
                    );
                    for data in state.diagnostics().unwrap() {
                        let name = data.layer.map_or_else(
                            || data.stage.to_owned(),
                            |layer| format!("layers.{layer}.{}", data.stage),
                        );
                        let witness = &case["arrays"][&name];
                        assert!(!witness.is_null(), "missing {name}");
                        compare_stage(witness, start * data.columns, &data.values, &name);
                        arrays += 1;
                        coordinates += 2 * data.values.len();
                    }
                    start += count;
                }
                assert_eq!(state.evaluated_len(), ids.len());
            }
        }
        println!(
            "MINISTRAL3_GRAPH_SUMMARY {}",
            serde_json::json!({
                "device":model.gpu.device.name().to_string(), "tiny_cases":5,
                "schedules":schedules, "completed_commands":commands,
                "diagnostic_arrays":arrays,"profile_coordinates":coordinates,
                "both_profiles":true,"commands_per_chunk":1,
                "selected_full_model_quality_fit_performance_qualified":false
            })
        );
    });
}

#[test]
#[ignore = "Explicit Apple GPU packed Ministral state/causal/reset/failure/replay and fit checks"]
fn packed_graph_state_and_memory_are_transactional() {
    autoreleasepool(|_| {
        let fixture = fixture();
        let model = tiny_model(&fixture);
        let prompt = [1, 7, 13, 3];
        let mut full = session(&model, 4, 0);
        full.append(&prompt).unwrap();
        full.evaluate().unwrap();
        let expected_logits = full.logits().unwrap().to_vec();
        let expected_prefix = prefix(&full);
        let mut state = session(&model, 1, 0);
        state.append(&prompt).unwrap();
        state.evaluate().unwrap();
        assert_eq!(state.logits().unwrap(), expected_logits);
        assert_eq!(prefix(&state), expected_prefix);
        assert_eq!(
            state.memory().positional_bytes,
            model.positional_resident_bytes()
        );
        assert_eq!(
            state.memory().metadata_allowance_bytes,
            ALLOWANCE + model.positional_pipeline_allowance()
        );
        let predicted = model.native_session_plan(state.options).unwrap().0;
        let actual = state.native_owned_memory().unwrap();
        assert_eq!(predicted.admitted_bytes, state.memory().admitted_bytes);
        assert_eq!(
            actual.metadata_allowance_bytes,
            predicted.metadata_allowance_bytes
        );
        assert_eq!(actual.positional_bytes, predicted.positional_bytes);
        let mut low = state.options;
        low.memory_limit_bytes = predicted.admitted_bytes - 1;
        assert!(matches!(
            Session::with_attention_mode(
                &model,
                low,
                ProjectionMode::MlxAffine,
                NormalizationMode::Compensated,
                ReadbackMode::FullLogits,
                AttentionMode::Reference
            ),
            Err(Error::Budget { .. })
        ));
        let mut over = state.options;
        over.capacity = 513;
        assert!(
            Session::with_attention_mode(
                &model,
                over,
                ProjectionMode::MlxAffine,
                NormalizationMode::Compensated,
                ReadbackMode::FullLogits,
                AttentionMode::Reference
            )
            .is_err()
        );
        assert!(
            Session::with_execution_modes(
                &model,
                state.options,
                ProjectionMode::MlxAffine,
                NormalizationMode::Original
            )
            .is_err()
        );
        assert!(
            Session::with_attention_mode(
                &model,
                state.options,
                ProjectionMode::MlxAffine,
                NormalizationMode::Compensated,
                ReadbackMode::FullLogits,
                AttentionMode::CompensatedReference
            )
            .is_err()
        );
        assert!(state.enable_affine_prefill_tile4().is_err());
        assert!(state.enable_rms_parallel_store().is_err());
        state.set_tokens(&prompt).unwrap();
        assert_eq!(state.pending_len(), 0);
        assert_eq!(state.evaluate_next_chunk().unwrap(), 0);
        state.append(&[5]).unwrap();
        assert!(state.logits().is_none());
        state.discard_pending();
        assert_eq!(state.logits().unwrap(), expected_logits);
        state.set_tokens(&[1, 7, 13, 5]).unwrap();
        assert_eq!(state.evaluated_len(), 3);
        assert_eq!(prefix(&state), prefix_len(&full, 3));
        state.evaluate().unwrap();
        let mut fresh = session(&model, 1, 0);
        fresh.append(&[1, 7, 13, 5]).unwrap();
        fresh.evaluate().unwrap();
        assert_eq!(state.logits().unwrap(), fresh.logits().unwrap());
        assert_eq!(prefix(&state), prefix(&fresh));
        state.reset();
        state.append(&prompt).unwrap();
        assert_eq!(state.evaluate_next_chunk().unwrap(), 1);
        let committed = prefix(&state);
        state.discard_pending(); // Cancellation preserves the completed prefix.
        assert_eq!(state.history(), &[1]);
        assert_eq!(prefix(&state), committed);
        state.append(&prompt[1..]).unwrap();
        let before = state.evaluated_len();
        assert!(matches!(
            state.finish_chunk(1, Err(Error::Native("injected encode failure".into()))),
            Err(Error::Native(_))
        ));
        assert_eq!(state.evaluated_len(), before);
        state.evaluate().unwrap();
        assert_eq!(state.logits().unwrap(), expected_logits);
        state.append(&[5]).unwrap();
        state.diagnostic_rows = 0;
        state.run_chunk(1, state.evaluated, None).unwrap(); // Completed but unpublished.
        assert!(matches!(
            state.finish_chunk(1, Err(Error::Nonfinite)),
            Err(Error::Nonfinite)
        ));
        assert_eq!(state.evaluated_len(), 4);
        state.discard_pending();
        assert_eq!(state.logits().unwrap(), expected_logits);
        state.append(&[5]).unwrap();
        state.evaluate().unwrap(); // Rewrites failed suffix.
        let retry = state.logits().unwrap().to_vec();
        assert!(matches!(
            state.finish_chunk(1, Err(Error::Completion)),
            Err(Error::Completion)
        ));
        assert_eq!(state.evaluated_len(), 0);
        assert!(state.logits().is_none());
        assert!(matches!(state.append(&[1]), Err(Error::ReplayRequired)));
        state.set_tokens(&[1, 7, 13, 3, 5]).unwrap();
        state.evaluate().unwrap();
        assert_eq!(state.logits().unwrap(), retry);
        state.reset();
        state.append(&prompt).unwrap();
        state.evaluate().unwrap();
        assert_eq!(state.logits().unwrap(), expected_logits);
        assert_eq!(prefix(&state), expected_prefix);
    });
}

#[test]
#[ignore = "Explicit Apple GPU finite packed embedding overflow preserves committed Ministral prefix"]
fn finite_packed_overflow_retries_without_publishing_failed_kv() {
    autoreleasepool(|_| {
        let mut fixture = fixture();
        // Only this fault witness changes an independently authored tensor:
        // finite BF16 scale32768/code255 reconstructs finite FP32 weights but
        // overflows Half embedding storage for token5. The tied Float head
        // remains finite for the successful control tokens.
        let embedding = &mut fixture["weights"]["model.embed_tokens.weight"];
        for word in &mut embedding["weight_words"].as_array_mut().unwrap()[5 * 16..6 * 16] {
            *word = Value::from(u32::MAX);
        }
        embedding["scales_bf16"][5] = Value::from(0x4700_u16);
        embedding["biases_bf16"][5] = Value::from(0_u16);
        let model = tiny_model(&fixture);
        let mut state = session(&model, 1, 0);
        state.append(&[1]).unwrap();
        state.evaluate().unwrap();
        let committed = prefix(&state);
        let logits = state.logits().unwrap().to_vec();
        state.append(&[5]).unwrap();
        assert!(matches!(state.evaluate_next_chunk(), Err(Error::Nonfinite)));
        assert_eq!(state.evaluated_len(), 1);
        assert_eq!(prefix(&state), committed);
        assert!(state.logits().is_none());
        state.discard_pending();
        assert_eq!(state.logits().unwrap(), logits);
        state.append(&[7]).unwrap();
        state.evaluate().unwrap();
        let mut fresh = session(&model, 1, 0);
        fresh.append(&[1, 7]).unwrap();
        fresh.evaluate().unwrap();
        assert_eq!(state.logits().unwrap(), fresh.logits().unwrap());
        assert_eq!(prefix(&state), prefix(&fresh));
    });
}

#[test]
#[ignore = "Explicit release Apple GPU warmed packed Ministral decode Rust allocation measurement"]
fn warmed_packed_graph_decode_allocates_no_rust_objects() {
    use crate::model::gemma_reference::tests::Measurement;
    autoreleasepool(|_| {
        let model = tiny_model(&fixture());
        let mut state = session(&model, 1, 0);
        state.append(&[1, 7]).unwrap();
        state.evaluate().unwrap();
        state.reset();
        state.append(&[1, 7, 13, 3, 2, 11, 9, 5]).unwrap();
        let mut completed = 0;
        while state.pending_len() > 0 {
            let measurement = Measurement::start();
            let result = state.evaluate_next_chunk();
            let allocations = measurement.finish();
            assert_eq!(result.unwrap(), 1);
            assert_eq!(allocations, 0, "warmed completed decode Rust allocations");
            completed += 1;
        }
        assert_eq!(completed, 8);
    });
}

#[test]
#[ignore = "Explicit Apple GPU shared graph positional dispatcher high-position query-gain control"]
fn graph_dispatcher_distinguishes_queries_and_keys_at_frozen_high_position() {
    autoreleasepool(|_| {
        let raw = include_bytes!("../../tests/fixtures/ministral3-rotary-v1/fixture.json");
        assert_eq!(
            format!("{:x}", Sha256::digest(raw)),
            "0e1eb4f6a9056bf1725d8265e905ec7fe8ddc3c690e8a802f58121d96f36c427"
        );
        let witness: Value = serde_json::from_slice(raw).unwrap();
        let case = witness["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["name"] == "first_epoch")
            .unwrap();
        let gpu = Gpu::new().unwrap();
        let plan = positional::Plan::ministral3_control(16384, 1).unwrap();
        let operation = plan.upload(&gpu, plan.compile(&gpu).unwrap()).unwrap();
        for (kind, input_name, output_name, heads) in [
            (
                ministral3_rotary::Kind::Queries,
                "query_input",
                "queries",
                3,
            ),
            (ministral3_rotary::Kind::Keys, "key_input", "keys", 1),
        ] {
            let words = words16(&case[input_name]);
            let mut input = Buffer::new(&gpu, words.len() + 7).unwrap();
            input.with_mut(|out| out[..words.len()].copy_from_slice(&words));
            let output = Buffer::<u16>::new(&gpu, words.len() + 7).unwrap();
            let status = Buffer::<u32>::new(&gpu, 8).unwrap();
            let raw = gpu.command().unwrap();
            operation
                .encode(
                    &Command {
                        raw: &raw,
                        trace: None,
                    },
                    kind,
                    positional::Invocation {
                        input: &input,
                        output: &output,
                        status: &status,
                        rows: 1,
                        heads,
                        offset: 16384,
                        position_limit: 16385,
                    },
                )
                .unwrap();
            raw.commit();
            raw.waitUntilCompleted();
            assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
            assert_eq!(status.with_slice(|v| v[0]), 0);
            let main = words16(&case["references"]["main"][output_name]);
            let wide = words16(&case["references"]["wide"][output_name]);
            let envelope = main
                .iter()
                .zip(&wide)
                .map(|(&a, &b)| {
                    f64::from((f16::from_bits(a).to_f32() - f16::from_bits(b).to_f32()).abs())
                })
                .fold(0., f64::max);
            output.with_slice(|out| {
                for expected in [&main, &wide] {
                    for (&actual, &expected) in out.iter().zip(expected) {
                        let actual = f64::from(f16::from_bits(actual).to_f32());
                        let expected = f64::from(f16::from_bits(expected).to_f32());
                        assert!(
                            (actual - expected).abs()
                                <= (2. * envelope).max(1e-4) + 1e-3 * expected.abs()
                        );
                    }
                }
            });
        }
    });
}
