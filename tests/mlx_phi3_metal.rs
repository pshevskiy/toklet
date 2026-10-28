#![cfg(all(
    feature = "metal",
    any(target_os = "macos", target_os = "ios"),
    target_arch = "aarch64"
))]
//! Explicit tiny Phi tests. No real-model, full-window, app or phone claim.
#[path = "support/phi3.rs"]
mod fixture;

use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::atomic::AtomicBool,
};
use toklet::{
    generation::{self, Input, Request, StopReason},
    metal::{self, AttentionMode, NormalizationMode, ProjectionMode, ReadbackMode},
    session,
    weights::LoadOptions,
};

const BUDGET: usize = 256 * 1024 * 1024;
const VOCAB: usize = 64;
fn load_options() -> LoadOptions {
    LoadOptions {
        context_limit: 512,
        memory_limit_bytes: BUDGET,
    }
}
fn options(chunk: usize, offset: usize, capacity: usize) -> session::Options {
    session::Options {
        capacity,
        prefill_chunk_size: chunk,
        position_offset: offset,
        memory_limit_bytes: BUDGET,
    }
}
fn create(
    model: &metal::Model,
    opts: session::Options,
    readback: ReadbackMode,
) -> metal::Session<'_> {
    metal::Session::with_attention_mode(
        model,
        opts,
        ProjectionMode::MlxAffine,
        NormalizationMode::Compensated,
        readback,
        AttentionMode::Reference,
    )
    .unwrap()
}
fn tokens(case: &Value) -> Vec<u32> {
    case["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
        .collect()
}
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}
// Existing B1 exact policy is finite numeric equality, including equal signed
// zeros. Raw bits() remains unchanged for source/native evidence and failures.
fn numeric_bits(values: &[f32]) -> Vec<u32> {
    values
        .iter()
        .map(|&value| {
            assert!(value.is_finite());
            if value == 0.0 { 0 } else { value.to_bits() }
        })
        .collect()
}
fn stage(d: &metal::Diagnostic) -> String {
    let name = match d.stage {
        "q" => "q_projection",
        "k" => "k_projection",
        "v" => "v_projection",
        "gate" => "gate_projection",
        "up" => "up_projection",
        "silu" => "gate_silu",
        "down" => "down_projection",
        other => other,
    };
    match d.layer {
        Some(layer) => format!("layers.{layer}.{name}"),
        None => name.to_owned(),
    }
}
fn selected<'a>(manifest: &'a Value, variant: &str, name: &str) -> &'a Value {
    manifest["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["variant"] == variant && c["name"] == name)
        .unwrap()
}
fn main_wide(raw: &[u8], spec: &Value) -> (Vec<f32>, Vec<f32>) {
    let main = fixture::values(raw, &spec["profiles"]["main"]);
    let wide = fixture::values(raw, &spec["profiles"]["wide"]);
    assert_eq!(main.len(), wide.len());
    assert!(main.iter().chain(&wide).all(|v| v.is_finite()));
    let envelope = main
        .iter()
        .zip(&wide)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
        .fold(0.0_f64, f64::max);
    assert_eq!(spec["precision_envelope"].as_f64().unwrap(), envelope);
    assert_eq!(
        spec["half_gate"]["atol"].as_f64().unwrap(),
        1e-4_f64.max(2.0 * envelope)
    );
    assert_eq!(spec["half_gate"]["rtol"].as_f64().unwrap(), 1e-3);
    (main, wide)
}

fn qualify(actual: &[f32], main: &[f32], wide: &[f32], gate: &Value, label: &Value) -> Value {
    assert_eq!(actual.len(), main.len());
    assert_eq!(actual.len(), wide.len());
    let atol = gate["atol"].as_f64().unwrap();
    let rtol = gate["rtol"].as_f64().unwrap();
    assert!(atol >= 1e-4 && atol.is_finite());
    assert_eq!(rtol, 1e-3);
    let mut maxima = [0.0_f64; 2];
    let mut ratios = [0.0_f64; 2];
    let mut main_failures = 0;
    let mut wide_exceedances = 0;
    let mut nonfinite = 0;
    let mut first_main = None;
    let mut first_wide = None;
    for (index, ((&actual, &main), &wide)) in actual.iter().zip(main).zip(wide).enumerate() {
        if !actual.is_finite() || !main.is_finite() || !wide.is_finite() {
            nonfinite += 1;
        }
        for (profile, expected) in [main, wide].into_iter().enumerate() {
            let error = (f64::from(actual) - f64::from(expected)).abs();
            let bound = atol + rtol * f64::from(expected).abs();
            let within = actual.is_finite() && expected.is_finite() && error <= bound;
            maxima[profile] = maxima[profile].max(error);
            ratios[profile] = ratios[profile].max(error / bound);
            if !within {
                let witness =
                    json!({"index":index,"actual":actual.to_bits(),"reference":expected.to_bits()});
                if profile == 0 {
                    main_failures += 1;
                    first_main.get_or_insert(witness);
                } else {
                    wide_exceedances += 1;
                    first_wide.get_or_insert(witness);
                }
            }
        }
    }
    if main_failures != 0 || nonfinite != 0 {
        // At most17×256 values per stage. Preserve original IEEE bits, including
        // signed zeros; numeric canonicalization belongs only to exact controls.
        println!(
            "PHI_NUMERICAL_FAILURE {}",
            json!({"label":label,"primary_main_gate":true,
            "main_gate_failures":main_failures,"first_main":first_main,"nonfinite_values":nonfinite,
            "wide_diagnostic_only":true,"wide_diagnostic_exceedances":wide_exceedances,"first_wide":first_wide,
            "actual_bits":bits(actual),"main_bits":bits(main),"wide_bits":bits(wide),"gate":gate})
        );
    }
    assert_eq!(
        nonfinite, 0,
        "{label}: finite values required in all profiles"
    );
    assert_eq!(main_failures, 0, "{label}: frozen primary main gate");
    json!({"primary_main_gate":true,"main_gate_values":actual.len(),"main_gate_failures":main_failures,
        "wide_diagnostic_only":true,"wide_diagnostic_values":actual.len(),"wide_diagnostic_exceedances":wide_exceedances,
        "first_wide_exceedance":first_wide,"max_abs_main":maxima[0],"max_abs_wide":maxima[1],
        "max_ratio_main":ratios[0],"max_ratio_wide":ratios[1]})
}

fn exact(actual: &[u32], expected: &[u32], label: &str) {
    // Preserve original raw vectors for witnesses; compare their finite numeric
    // representations only. This is not a tighter signed-zero identity gate.
    let actual_numeric = numeric_bits(
        &actual
            .iter()
            .copied()
            .map(f32::from_bits)
            .collect::<Vec<_>>(),
    );
    let expected_numeric = numeric_bits(
        &expected
            .iter()
            .copied()
            .map(f32::from_bits)
            .collect::<Vec<_>>(),
    );
    if actual_numeric != expected_numeric {
        println!(
            "PHI_EXACT_FAILURE {}",
            json!({"label":label,"actual_bits":actual,"expected_bits":expected})
        );
    }
    assert_eq!(actual_numeric, expected_numeric, "{label}");
}

#[test]
#[ignore = "Explicit tiny Phi Apple GPU stages and exact schedules; not real checkpoint qualification"]
fn native_phi3_every_chunk_matches_frozen_independent_half_profiles() {
    let (root, manifest, raw) = fixture::open();
    assert_eq!(manifest["stage_groups"], 195);
    assert_eq!(manifest["stored_arrays"], 585);
    assert_eq!(manifest["oracle_stages_per_case"], 39);
    assert_eq!(manifest["native_diagnostic_stages_per_case"], 35);
    let mut snapshots = 0;
    let mut values = 0;
    let mut wide_exceedances = 0_u64;
    let mut schedules = 0;
    let mut total_rows = 0;
    for variant in ["untied", "tied"] {
        let model = metal::Model::load_mlx(&root.join(variant), load_options()).unwrap();
        assert_eq!(
            model.mlx_architecture(),
            Some(toklet::mlx::dense::Architecture::Phi3)
        );
        assert_eq!(model.positional_resident_bytes(), 0);
        let mut causal = BTreeMap::new();
        for case in manifest["cases"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|c| c["variant"] == variant)
        {
            let input = tokens(case);
            let offset = usize::try_from(case["position_offset"].as_u64().unwrap()).unwrap();
            let name = case["name"].as_str().unwrap();
            assert_eq!(case["initial_kv_tokens"], 0);
            let capacity = 512 - offset;
            let mut chunks = vec![
                1.min(capacity),
                3.min(capacity),
                16.min(capacity),
                input.len().min(capacity),
            ];
            chunks.sort_unstable();
            chunks.dedup();
            let mut baseline: Option<Vec<u32>> = None;
            for chunk in chunks {
                let mut s = create(
                    &model,
                    options(chunk, offset, capacity),
                    ReadbackMode::FullLogits,
                );
                s.enable_diagnostics(&[0, 1]).unwrap();
                s.set_tokens(&input).unwrap();
                let mut start = 0;
                let mut all_logits = Vec::with_capacity(input.len() * VOCAB);
                let mut stats = Vec::new();
                loop {
                    let rows = s.evaluate_next_chunk().unwrap();
                    if rows == 0 {
                        break;
                    }
                    assert_eq!(rows, chunk.min(input.len() - start));
                    let mut names: BTreeSet<_> = case["arrays"]
                        .as_object()
                        .unwrap()
                        .keys()
                        .filter(|name| {
                            !name.ends_with("attention_scores")
                                && !name.ends_with("attention_probabilities")
                        })
                        .cloned()
                        .collect();
                    assert_eq!(names.len(), 35);
                    let diagnostics = s.diagnostics().unwrap();
                    assert_eq!(diagnostics.len(), 35);
                    for d in diagnostics {
                        let stage = stage(&d);
                        assert!(names.remove(&stage), "duplicate/unknown diagnostic {stage}");
                        let spec = &case["arrays"][&stage];
                        let shape = spec["shape"].as_array().unwrap();
                        assert_eq!(shape[0].as_u64().unwrap() as usize, input.len());
                        let columns: usize = shape[1..]
                            .iter()
                            .map(|v| v.as_u64().unwrap() as usize)
                            .product();
                        assert_eq!(
                            (d.rows, d.columns, d.values.len()),
                            (rows, columns, rows * columns)
                        );
                        let (main, wide) = main_wide(&raw, spec);
                        let range = start * columns..(start + rows) * columns;
                        let label = json!({"variant":variant,"case":name,"offset":offset,"chunk":chunk,"start":start,"rows":rows,"stage":stage});
                        let comparison = qualify(
                            &d.values,
                            &main[range.clone()],
                            &wide[range],
                            &spec["half_gate"],
                            &label,
                        );
                        wide_exceedances +=
                            comparison["wide_diagnostic_exceedances"].as_u64().unwrap();
                        stats.push(json!({"label":label,"comparison":comparison}));
                        if stage == "logits" {
                            all_logits.extend(bits(&d.values));
                        }
                        snapshots += 1;
                        values += d.values.len();
                    }
                    assert!(names.is_empty());
                    start += rows;
                    assert_eq!(s.evaluated_len(), start);
                    assert_eq!(s.pending_len(), input.len() - start);
                }
                assert_eq!(start, input.len());
                assert_eq!(all_logits.len(), input.len() * VOCAB);
                exact(
                    &bits(s.logits().unwrap()),
                    &all_logits[all_logits.len() - VOCAB..],
                    "public last row",
                );
                println!(
                    "PHI_GRAPH_SCHEDULE {}",
                    json!({"variant":variant,"case":name,"offset":offset,
                    "chunk":chunk,"local_keys":input.len(),"logits_bits":all_logits,"stages":stats})
                );
                if let Some(expected) = &baseline {
                    exact(&all_logits, expected, "all-row schedule equality");
                } else {
                    baseline = Some(all_logits.clone());
                }
                let before = bits(s.logits().unwrap());
                s.set_tokens(&input).unwrap();
                assert_eq!(s.evaluate_next_chunk().unwrap(), 0);
                exact(&bits(s.logits().unwrap()), &before, "fully cached reuse");
                if offset == 495 {
                    assert_eq!(
                        (s.capacity(), s.evaluated_len(), offset + s.evaluated_len()),
                        (17, 17, 512)
                    );
                    let error = s.append(&[1]).unwrap_err();
                    assert!(matches!(
                        error,
                        metal::Error::Options("token ID/count outside session capacity")
                    ));
                    let error = s.set_tokens(&[1; 18]).unwrap_err();
                    assert!(matches!(
                        error,
                        metal::Error::Options("token ID/count outside session capacity")
                    ));
                    assert_eq!(s.history(), input);
                    assert_eq!(s.evaluated_len(), 17);
                    exact(
                        &bits(s.logits().unwrap()),
                        &before,
                        "boundary rejection atomicity",
                    );
                }
                schedules += 1;
                total_rows += start;
            }
            causal.insert(name.to_owned(), baseline.unwrap());
        }
        if variant == "untied" {
            exact(
                &causal["prompt"][..16 * VOCAB],
                &causal["future_changed"][..16 * VOCAB],
                "future-tail causal prefix",
            );
        }
    }
    assert_eq!(
        (schedules, total_rows, snapshots, values),
        (19, 281, 3815, 1_996_224)
    );
    println!(
        "PHI_HALF_GRAPH {}",
        json!({"schedules":schedules,"committed_rows":total_rows,"snapshots":snapshots,
        "main_gate_values":values,"main_gate_failures":0,
        "wide_diagnostic_values":values,"wide_diagnostic_exceedances":wide_exceedances,"wide_diagnostic_only":true,"captured_stages_per_chunk":35,
        "source_stages_per_case":39,"absolute_last_position":511,"boundary_local_keys":17,
        "full512_key_numerics_tested":false,"real_checkpoint_qualified":false,"iphone_qualified":false})
    );
}

fn lowest_id(logits: &[f32]) -> u32 {
    assert_eq!(logits.len(), VOCAB);
    assert!(logits.iter().all(|v| v.is_finite()));
    let mut best = 0;
    for (index, &value) in logits.iter().enumerate().skip(1) {
        if value > logits[best] {
            best = index;
        }
    }
    best as u32
}
#[derive(Debug, PartialEq, Eq)]
struct State {
    history: Vec<u32>,
    evaluated: usize,
    pending: usize,
    logits: Option<Vec<u32>>,
    greedy: Option<u32>,
}
fn state(s: &metal::Session<'_>) -> State {
    State {
        history: s.history().to_vec(),
        evaluated: s.evaluated_len(),
        pending: s.pending_len(),
        logits: s.logits().map(numeric_bits),
        greedy: s.greedy_token().map(Result::unwrap),
    }
}

fn memory_control(root: &Path, model: &metal::Model) {
    let opts = options(16, 0, 512);
    for readback in [ReadbackMode::FullLogits, ReadbackMode::GreedyToken] {
        let error = metal::Session::with_attention_mode(
            model,
            opts,
            ProjectionMode::MlxAffine,
            NormalizationMode::Compensated,
            readback,
            AttentionMode::Online,
        )
        .err()
        .expect("Phi Online must reject");
        assert!(matches!(error, metal::Error::Unsupported(_)));
        assert!(error.to_string().contains("Online"));
    }

    let report = metal::planning::inspect_mlx_capacity(
        root,
        load_options(),
        opts,
        NormalizationMode::Compensated,
        ReadbackMode::FullLogits,
        AttentionMode::Reference,
    )
    .unwrap();
    assert!(!report.resources.numerical_buffers_allocated);
    assert_eq!(
        report.resources.resident_weights_bytes,
        model.weights_bytes()
    );
    let first = create(model, opts, ReadbackMode::FullLogits);
    assert_eq!(first.memory(), report.resources.session_memory);
    let admitted = first.memory().admitted_bytes;
    drop(first);
    let exact = create(
        model,
        session::Options {
            memory_limit_bytes: admitted,
            ..opts
        },
        ReadbackMode::FullLogits,
    );
    assert_eq!(exact.memory().admitted_bytes, admitted);
    drop(exact);
    assert!(matches!(
        metal::Session::with_attention_mode(
            model,
            session::Options {
                memory_limit_bytes: admitted - 1,
                ..opts
            },
            ProjectionMode::MlxAffine,
            NormalizationMode::Compensated,
            ReadbackMode::FullLogits,
            AttentionMode::Reference
        ),
        Err(metal::Error::Budget { .. })
    ));
    for invalid in [options(16, 0, 513), options(16, 1, 512)] {
        assert!(
            metal::Session::with_attention_mode(
                model,
                invalid,
                ProjectionMode::MlxAffine,
                NormalizationMode::Compensated,
                ReadbackMode::FullLogits,
                AttentionMode::Reference
            )
            .is_err()
        );
    }
    println!(
        "PHI_MEMORY {}",
        json!({"report":report,"exact_session_budget":admitted,"measured_process_peak":false})
    );
}

#[test]
#[ignore = "Explicit tiny Phi GPU state/readback/raw-generation and admission; no text or performance claim"]
fn native_phi3_state_readback_raw_generation_and_budget_are_bounded() {
    let (root, manifest, raw) = fixture::open();
    let prompt = tokens(selected(&manifest, "untied", "prompt"));
    let changed = tokens(selected(&manifest, "untied", "future_changed"));
    let mut total_samples = 0;
    for variant in ["untied", "tied"] {
        let path = root.join(variant);
        let model = metal::Model::load_mlx(&path, load_options()).unwrap();
        memory_control(&path, &model);
        let mut baseline: Option<Vec<u32>> = None;
        // Same prompt is a native-only readback/state control for the tied model;
        // its independent graph fixture remains the three-token tied_control.
        for readback in [ReadbackMode::FullLogits, ReadbackMode::GreedyToken] {
            let mut s = create(&model, options(3, 0, 512), readback);
            s.set_tokens(&prompt).unwrap();
            s.evaluate().unwrap();
            let prompt_state = state(&s);
            if variant == "untied" && readback == ReadbackMode::FullLogits {
                let spec = &selected(&manifest, variant, "prompt")["arrays"]["logits"];
                let (main, wide) = main_wide(&raw, spec);
                qualify(
                    s.logits().unwrap(),
                    &main[16 * VOCAB..],
                    &wide[16 * VOCAB..],
                    &spec["half_gate"],
                    &json!("state prompt"),
                );
            }
            s.set_tokens(&prompt).unwrap();
            assert_eq!(s.evaluate_next_chunk().unwrap(), 0);
            assert_eq!(state(&s), prompt_state);
            let first_id = s.greedy_token().unwrap().unwrap();
            s.append(&[first_id]).unwrap();
            assert_eq!(s.pending_len(), 1);
            assert!(s.logits().is_none() && s.greedy_token().is_none());
            s.discard_pending();
            assert_eq!(state(&s), prompt_state);
            let mut expected = Vec::new();
            for _ in 0..4 {
                s.evaluate().unwrap();
                let id = s.greedy_token().unwrap().unwrap();
                match readback {
                    ReadbackMode::FullLogits => assert_eq!(id, lowest_id(s.logits().unwrap())),
                    ReadbackMode::GreedyToken => assert!(s.logits().is_none()),
                }
                expected.push(id);
                s.append(&[id]).unwrap();
            }
            assert_eq!(
                (s.evaluated_len(), s.pending_len(), s.history().len()),
                (20, 1, 21)
            );
            if let Some(ids) = &baseline {
                assert_eq!(&expected, ids, "readback generated IDs");
            } else {
                baseline = Some(expected.clone());
            }
            s.reset();
            s.set_tokens(&prompt).unwrap();
            assert_eq!(s.evaluate_next_chunk().unwrap(), 3);
            s.discard_pending();
            assert_eq!(s.history(), &prompt[..3]);
            s.append(&prompt[3..]).unwrap();
            s.evaluate().unwrap();
            assert_eq!(
                state(&s),
                prompt_state,
                "reset and partial-prefill continuation"
            );
            s.set_tokens(&changed).unwrap();
            assert_eq!((s.evaluated_len(), s.pending_len()), (16, 1));
            s.evaluate().unwrap();
            let changed_state = state(&s);
            s.reset();
            s.set_tokens(&changed).unwrap();
            s.evaluate().unwrap();
            assert_eq!(
                state(&s),
                changed_state,
                "edited prefix versus fresh replay"
            );
            let flag = AtomicBool::new(false);
            let mut emitted = Vec::new();
            let report = generation::generate(
                &mut s,
                Request {
                    input: Input::Reset(&prompt),
                    max_new_tokens: 4,
                    stop_tokens: &[],
                },
                &flag,
                |e| {
                    assert_eq!(e.index, emitted.len());
                    emitted.push(e.token_id);
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(emitted, expected);
            assert_eq!(
                s.history(),
                [prompt.as_slice(), expected.as_slice()].concat()
            );
            assert_eq!(report.reason, StopReason::OutputLimit);
            assert_eq!(
                (
                    report.sampled_tokens,
                    report.emitted_tokens,
                    report.evaluated_tokens,
                    report.pending_tokens
                ),
                (4, 4, 20, 1)
            );
            total_samples += report.sampled_tokens;
            let before = state(&s);
            let report = generation::generate(
                &mut s,
                Request {
                    input: Input::Reset(&changed),
                    max_new_tokens: 1,
                    stop_tokens: &[],
                },
                &AtomicBool::new(true),
                |_| panic!("pre-cancelled request emitted"),
            )
            .unwrap();
            assert_eq!(report.reason, StopReason::Cancelled);
            assert_eq!((report.sampled_tokens, report.emitted_tokens), (0, 0));
            assert_eq!(state(&s), before, "pre-cancelled reset is atomic");
            let report = generation::generate(
                &mut s,
                Request {
                    input: Input::Reset(&prompt),
                    max_new_tokens: 4,
                    stop_tokens: &expected[..1],
                },
                &flag,
                |_| panic!("stop ID emitted"),
            )
            .unwrap();
            assert_eq!(
                report.reason,
                StopReason::StopToken {
                    token_id: expected[0]
                }
            );
            assert_eq!(
                (
                    report.sampled_tokens,
                    report.emitted_tokens,
                    report.evaluated_tokens,
                    report.pending_tokens
                ),
                (1, 0, 17, 1)
            );
            let failed = generation::generate(
                &mut s,
                Request {
                    input: Input::Reset(&prompt),
                    max_new_tokens: 1,
                    stop_tokens: &[],
                },
                &flag,
                |_| Err(std::io::Error::other("injected output failure")),
            );
            assert!(matches!(failed, Err(generation::Error::Output(_))));
            assert_eq!((s.evaluated_len(), s.pending_len()), (17, 1));
            assert_eq!(s.history().last(), Some(&expected[0]));
            s.discard_pending();
            assert_eq!(state(&s), prompt_state);
            println!(
                "PHI_STATE {}",
                json!({"variant":variant,"readback":format!("{readback:?}"),"generated_ids":expected,
                "capacity_allocated":512,"largest_numerical_history":21,"raw_token_only":true})
            );
        }
    }
    assert_eq!(total_samples, 16);
}
