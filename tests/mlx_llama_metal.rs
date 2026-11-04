#![cfg(all(
    feature = "metal",
    any(target_os = "macos", target_os = "ios"),
    target_arch = "aarch64"
))]
use half::f16;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};
use toklet::{
    metal::{self, AttentionMode, NormalizationMode, ProjectionMode, ReadbackMode},
    session,
    weights::LoadOptions,
};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join("mlx-llama-v1")
}
fn manifest() -> Value {
    let bytes = fs::read(root().join("manifest.json")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        "824d2ee2ae9dd5b623c5aa9db10c13b2e4dee32647ea5fa9d78274fb3529a1f5"
    );
    let m: Value = serde_json::from_slice(&bytes).unwrap();
    for (name, item) in m["files"].as_object().unwrap() {
        let data = fs::read(root().join(name)).unwrap();
        assert_eq!(data.len() as u64, item["bytes"]);
        assert_eq!(format!("{:x}", Sha256::digest(data)), item["sha256"]);
    }
    m
}
fn array(m: &Value, name: &str) -> Vec<f32> {
    let bytes = fs::read(root().join("arrays.bin")).unwrap();
    let e = &m["arrays"][name];
    let offset = e["offset"].as_u64().unwrap() as usize;
    let len = e["bytes"].as_u64().unwrap() as usize;
    let data = &bytes[offset..offset + len];
    assert_eq!(format!("{:x}", Sha256::digest(data)), e["sha256"]);
    match e["dtype"].as_str().unwrap() {
        "<f2" => data
            .chunks_exact(2)
            .map(|b| f16::from_le_bytes(b.try_into().unwrap()).to_f32())
            .collect(),
        "<f4" => data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        _ => panic!(),
    }
}
fn qualify(a: &[f32], b: &[f32], gate: &Value, context: &str) {
    assert_eq!(a.len(), b.len());
    let atol = gate["atol"].as_f64().unwrap();
    let rtol = gate["rtol"].as_f64().unwrap();
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!(
            a.is_finite()
                && b.is_finite()
                && (f64::from(a) - f64::from(b)).abs() <= atol + rtol * f64::from(b).abs(),
            "{context} index{i}: {a} vs {b}, gate {gate}"
        );
    }
}
fn options(chunk: usize, offset: usize) -> session::Options {
    session::Options {
        capacity: 32 - offset,
        prefill_chunk_size: chunk,
        position_offset: offset,
        memory_limit_bytes: 256 * 1024 * 1024,
    }
}
fn session(
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
fn model(variant: &str) -> metal::Model {
    metal::Model::load_mlx(
        &root().join(variant),
        LoadOptions {
            context_limit: 32,
            memory_limit_bytes: 128 * 1024 * 1024,
        },
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
        Some(i) => format!("cases.{case}.layers.{i}.{name}"),
        None => format!("cases.{case}.{name}"),
    }
}

#[test]
#[ignore = "Explicit Apple GPU tied/untied Llama graph against frozen independent Half gates"]
fn native_llama_graph_matches_frozen_tied_and_untied_oracles() {
    let m = manifest();
    let mut snapshots = 0;
    let mut compared = 0;
    for variant in ["tied", "untied"] {
        let model = model(variant);
        for (case, spec) in m["variants"][variant]["cases"].as_object().unwrap() {
            let tokens: Vec<u32> = spec["tokens"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            let offset = spec["position_offset"].as_u64().unwrap() as usize;
            {
                let attention = AttentionMode::Reference;
                for chunk in [tokens.len(), 3, 1] {
                    let mut s = session(&model, chunk, offset, ReadbackMode::FullLogits, attention);
                    s.enable_diagnostics(&[0, 1]).unwrap();
                    s.set_tokens(&tokens).unwrap();
                    s.evaluate().unwrap();
                    let prefix = format!("{variant}.half.cases.{case}.");
                    let mut expected_names: std::collections::BTreeSet<_> = m["arrays"]
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
                    assert_eq!(expected_names.len(), 35);
                    let mut diagnostics = s.diagnostics().unwrap();
                    diagnostics.sort_by_key(|d| match d.layer {
                        None => match d.stage {
                            "embedding" => 0,
                            "final_norm" => 33,
                            "logits" => 34,
                            _ => panic!("unexpected global diagnostic"),
                        },
                        Some(layer) => {
                            1 + layer * 16
                                + [
                                    "input_norm",
                                    "q",
                                    "k",
                                    "v",
                                    "q_rotary",
                                    "k_rotary",
                                    "attention_context",
                                    "attention_projection",
                                    "attention_residual",
                                    "post_attention_norm",
                                    "gate",
                                    "silu",
                                    "up",
                                    "gated",
                                    "down",
                                    "output",
                                ]
                                .iter()
                                .position(|stage| *stage == d.stage)
                                .expect("known graph stage")
                        }
                    });
                    for d in diagnostics {
                        let name = format!("{variant}.half.{}", stage(case, &d));
                        assert!(expected_names.remove(&name), "{name}");
                        let shape = m["arrays"][&name]["shape"].as_array().unwrap();
                        let columns = shape[1..]
                            .iter()
                            .map(|v| v.as_u64().unwrap() as usize)
                            .product::<usize>();
                        assert_eq!(d.columns, columns);
                        assert_eq!(d.rows, (tokens.len() - 1) % chunk + 1);
                        let expected = array(&m, &name);
                        let start = (tokens.len() - d.rows) * columns;
                        qualify(
                            &d.values,
                            &expected[start..],
                            &m["arrays"][&name]["gate"],
                            &format!("{name} {attention:?} chunk{chunk}"),
                        );
                        snapshots += 1;
                        compared += d.values.len();
                    }
                    assert!(expected_names.is_empty());
                    assert_eq!(s.evaluated_len(), tokens.len());
                    let before = s.logits().unwrap().to_vec();
                    s.set_tokens(&tokens).unwrap();
                    assert_eq!(s.evaluate_next_chunk().unwrap(), 0);
                    assert_eq!(s.logits().unwrap(), before);
                }
            }
        }
        // Ordinary last-row output uses a nonzero input matrix offset.
        let mut s = session(
            &model,
            32,
            0,
            ReadbackMode::FullLogits,
            AttentionMode::Reference,
        );
        s.set_tokens(&[1, 7, 13, 3]).unwrap();
        s.evaluate().unwrap();
        let key = format!("{variant}.half.cases.prompt.logits");
        let expected = array(&m, &key);
        qualify(
            s.logits().unwrap(),
            &expected[expected.len() - 64..],
            &m["arrays"][&key]["gate"],
            &key,
        );
        println!(
            "LLAMA_MODEL {}",
            serde_json::json!({"variant":variant,"device":model.device_name(),"weights_bytes":model.weights_bytes(),"real_checkpoint_qualified":false,"iphone_qualified":false})
        );
    }
    assert_eq!(snapshots, 1050);
    println!(
        "LLAMA_GRAPH {}",
        serde_json::json!({"snapshots":snapshots,"compared_values":compared})
    );
}
#[test]
#[ignore = "Explicit Apple GPU Llama cache/pending/token-only/replay/boundary and admission"]
fn native_llama_pending_state_token_selection_and_memory_are_explicit() {
    let m = manifest();
    let mut selections = 0;
    let mut distributions = 0;
    let mut weights = Vec::new();
    for variant in ["tied", "untied"] {
        let model = model(variant);
        assert_eq!(
            model.mlx_architecture(),
            Some(toklet::mlx::dense::Architecture::Llama)
        );
        for readback in [ReadbackMode::FullLogits, ReadbackMode::GreedyToken] {
            let error = metal::Session::with_attention_mode(
                &model,
                options(3, 0),
                ProjectionMode::MlxAffine,
                NormalizationMode::Compensated,
                readback,
                AttentionMode::Online,
            )
            .err()
            .expect("unqualified explicit Online must reject");
            assert!(error.to_string().contains("not qualified"));
        }
        weights.push(model.weights_bytes());
        let source = toklet::mlx::qwen::Source::open_dense(
            &root().join(variant),
            LoadOptions {
                context_limit: 32,
                memory_limit_bytes: 128 * 1024 * 1024,
            },
        )
        .unwrap();
        let load_limit = source.admitted_load_bytes();
        assert!(matches!(
            metal::Model::load_mlx(
                &root().join(variant),
                LoadOptions {
                    context_limit: 32,
                    memory_limit_bytes: load_limit - 1
                }
            ),
            Err(metal::Error::Weights(toklet::weights::Error::Budget { .. }))
        ));
        let exact = metal::Model::load_mlx(
            &root().join(variant),
            LoadOptions {
                context_limit: 32,
                memory_limit_bytes: load_limit,
            },
        )
        .unwrap();
        assert_eq!(exact.weights_bytes(), model.weights_bytes());
        drop(exact);
        {
            let attention = AttentionMode::Reference;
            for readback in [ReadbackMode::FullLogits, ReadbackMode::GreedyToken] {
                let mut s = session(&model, 32, 0, readback, attention);
                s.set_tokens(&[1, 7, 13, 3]).unwrap();
                for step in 0..4 {
                    s.evaluate().unwrap();
                    let key = format!("{variant}.greedy.step_{step}.logits");
                    match readback {
                        ReadbackMode::FullLogits => {
                            qualify(
                                s.logits().expect("FullLogits must publish distribution"),
                                &array(&m, &key),
                                &m["arrays"][&key]["gate"],
                                &format!("{key} {attention:?} {readback:?}"),
                            );
                            distributions += 1;
                        }
                        ReadbackMode::GreedyToken => assert!(s.logits().is_none()),
                    }
                    let id = s.greedy_token().unwrap().unwrap();
                    assert_eq!(
                        u64::from(id),
                        m["variants"][variant]["greedy"]["tokens"][step]
                    );
                    selections += 1;
                    s.append(&[id]).unwrap();
                    assert_eq!(s.pending_len(), 1);
                    assert!(s.greedy_token().is_none());
                }
                let history = s.history().to_vec();
                s.set_tokens(&history).unwrap();
                assert_eq!(s.evaluated_len(), 7);
                s.evaluate().unwrap();
                assert_eq!(s.evaluated_len(), 8);
                let changed = [1, 7, 13, 17];
                s.set_tokens(&changed).unwrap();
                s.evaluate().unwrap();
                let mut fresh = session(&model, 32, 0, readback, attention);
                fresh.set_tokens(&changed).unwrap();
                fresh.evaluate().unwrap();
                assert_eq!(
                    s.greedy_token().unwrap().unwrap(),
                    fresh.greedy_token().unwrap().unwrap()
                );
                match readback {
                    ReadbackMode::FullLogits => assert_eq!(
                        s.logits().expect("FullLogits must publish distribution"),
                        fresh.logits().unwrap()
                    ),
                    ReadbackMode::GreedyToken => {
                        assert!(s.logits().is_none());
                        assert!(fresh.logits().is_none());
                    }
                }
                s.reset();
                assert_eq!(s.evaluated_len(), 0);
            }
        }
        let mut s = session(
            &model,
            3,
            0,
            ReadbackMode::FullLogits,
            AttentionMode::Reference,
        );
        let tokens: Vec<u32> = (0..32).map(|i| (7 * i + 1) % 64).collect();
        s.set_tokens(&tokens).unwrap();
        assert_eq!(s.evaluate_next_chunk().unwrap(), 3);
        assert_eq!(s.evaluated_len(), 3);
        s.evaluate().unwrap();
        assert_eq!(s.evaluated_len(), 32);
        let before = s.logits().unwrap().to_vec();
        assert!(s.append(&[1]).is_err());
        assert!(s.set_tokens(&[1; 33]).is_err());
        assert_eq!(s.evaluated_len(), 32);
        assert_eq!(s.history(), tokens);
        assert_eq!(s.logits().unwrap(), before);
        s.reset();
        s.set_tokens(&[1, 7, 13, 3]).unwrap();
        s.evaluate().unwrap();
        let mut bad = options(3, 0);
        bad.memory_limit_bytes = 1;
        assert!(
            metal::Session::with_attention_mode(
                &model,
                bad,
                ProjectionMode::MlxAffine,
                NormalizationMode::Compensated,
                ReadbackMode::FullLogits,
                AttentionMode::Reference
            )
            .is_err()
        );
    }
    assert_eq!(weights[1] - weights[0], 2560);
    assert_eq!(distributions, 8);
    assert_eq!(selections, 16);
    println!(
        "LLAMA_STATE {}",
        serde_json::json!({"selections":selections,"full_distributions":distributions,"weights_bytes":weights,"partial_prefill_committed":3,"boundary":32,"iphone_qualified":false})
    );
}

// This fixture has its own frozen Half-storage oracle. It does not replace
// the ordinary-RoPE graph fixture or the scalar FP32 source comparison.
fn scaled_fixture() -> Value {
    let raw = include_bytes!("fixtures/llama3-half-graph-v1/fixture.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        "7bab31c5552c3954e9ac5a97919881eee3657e036e71ddc072713929aebd3405"
    );
    let fixture: Value = serde_json::from_slice(raw).unwrap();
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for category in ["inputs", "producers"] {
        for (name, pin) in fixture[category].as_object().unwrap() {
            let bytes = fs::read(repo.join(name)).unwrap();
            assert_eq!(bytes.len() as u64, pin["bytes"]);
            assert_eq!(format!("{:x}", Sha256::digest(bytes)), pin["sha256"]);
        }
    }
    fixture
}

fn scaled_array(spec: &Value) -> Vec<f32> {
    spec["main_bits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            let bits = v.as_u64().unwrap();
            match spec["dtype"].as_str().unwrap() {
                "f16le" => f16::from_bits(u16::try_from(bits).unwrap()).to_f32(),
                "f32le" => f32::from_bits(u32::try_from(bits).unwrap()),
                _ => panic!("unknown frozen storage"),
            }
        })
        .collect()
}

struct ScaledArtifact(PathBuf);
impl ScaledArtifact {
    fn new(variant: &str, fixture: &Value) -> Self {
        let path = std::env::temp_dir().join(format!(
            "toklet-half-{}-{}-{variant}",
            fixture["format"].as_str().unwrap(),
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        let artifact = Self(path);
        fs::copy(
            root().join(variant).join("model.safetensors"),
            artifact.0.join("model.safetensors"),
        )
        .unwrap();
        fs::write(
            artifact.0.join("config.json"),
            serde_json::to_vec(&fixture["config_templates"][variant]).unwrap(),
        )
        .unwrap();
        artifact
    }
}
impl Drop for ScaledArtifact {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
#[ignore = "Explicit Apple GPU scaled Llama Half graph and session checks; not real checkpoint qualification"]
fn scaled_llama_half_graph_and_session_match_frozen_independent_oracle() {
    check_half_graph(&scaled_fixture(), true);
}

fn check_half_graph(fixture: &Value, positional: bool) {
    let mut snapshots = 0;
    let mut compared = 0;
    for variant in ["tied", "untied"] {
        let artifact = ScaledArtifact::new(variant, fixture);
        let mut causal_logits = std::collections::BTreeMap::new();
        let model = metal::Model::load_mlx(
            &artifact.0,
            LoadOptions {
                context_limit: 512,
                memory_limit_bytes: 128 * 1024 * 1024,
            },
        )
        .unwrap();
        assert_eq!(model.positional_resident_bytes() > 0, positional);
        for case in fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|c| c["variant"] == variant)
        {
            let tokens: Vec<u32> = case["tokens"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
                .collect();
            let offset = case["position_offset"].as_u64().unwrap() as usize;
            assert_eq!(case["initial_kv_tokens"], 0);
            let make_session = |chunk| {
                metal::Session::with_attention_mode(
                    &model,
                    session::Options {
                        capacity: 512 - offset,
                        position_offset: offset,
                        prefill_chunk_size: chunk,
                        memory_limit_bytes: 256 * 1024 * 1024,
                    },
                    ProjectionMode::MlxAffine,
                    NormalizationMode::Compensated,
                    ReadbackMode::FullLogits,
                    AttentionMode::Reference,
                )
                .unwrap()
            };
            let logits_spec = &case["arrays"]["logits"];
            let logits = scaled_array(logits_spec);
            let mut full_schedule_logits: Option<Vec<f32>> = None;
            for chunk in [tokens.len(), 3, 1] {
                let mut s = make_session(chunk);
                s.enable_diagnostics(&[0, 1]).unwrap();
                s.set_tokens(&tokens).unwrap();
                let mut start = 0;
                let mut actual_logits = Vec::new();
                loop {
                    let rows = s.evaluate_next_chunk().unwrap();
                    if rows == 0 {
                        break;
                    }
                    let mut names: std::collections::BTreeSet<_> = case["arrays"]
                        .as_object()
                        .unwrap()
                        .keys()
                        .filter(|n| {
                            !n.ends_with("attention_scores")
                                && !n.ends_with("attention_probabilities")
                        })
                        .cloned()
                        .collect();
                    assert_eq!(names.len(), 35);
                    let diagnostics = s.diagnostics().unwrap();
                    assert_eq!(diagnostics.len(), 35);
                    if let Some(mask) =
                        fixture["config_templates"][variant]["no_rope_layers"].as_array()
                    {
                        for (layer, enabled) in mask.iter().enumerate() {
                            if enabled == 0 {
                                for (before, after) in [("q", "q_rotary"), ("k", "k_rotary")] {
                                    let values = |stage| {
                                        &diagnostics
                                            .iter()
                                            .find(|d| d.layer == Some(layer) && d.stage == stage)
                                            .unwrap()
                                            .values
                                    };
                                    assert!(
                                        values(before)
                                            .iter()
                                            .zip(values(after))
                                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                                        "bypass must preserve every Q/K bit"
                                    );
                                }
                            }
                        }
                    }
                    for d in diagnostics {
                        let full_name = stage("scaled", &d);
                        let name = full_name.strip_prefix("cases.scaled.").unwrap();
                        assert!(names.remove(name), "unexpected/duplicate {name}");
                        let spec = &case["arrays"][name];
                        let shape = spec["shape"].as_array().unwrap();
                        assert_eq!(shape[0].as_u64().unwrap() as usize, tokens.len());
                        let columns = shape[1..]
                            .iter()
                            .map(|v| v.as_u64().unwrap() as usize)
                            .product::<usize>();
                        assert_eq!((d.rows, d.columns), (rows, columns));
                        let expected = scaled_array(spec);
                        qualify(
                            &d.values,
                            &expected[start * columns..(start + rows) * columns],
                            &spec["gate"],
                            &format!(
                                "{variant} {} offset{offset} chunk{chunk} rows{start} {name}",
                                case["name"]
                            ),
                        );
                        if name == "logits" {
                            actual_logits.extend_from_slice(&d.values);
                        }
                        snapshots += 1;
                        compared += d.values.len();
                    }
                    assert!(names.is_empty());
                    start += rows;
                    assert_eq!(s.evaluated_len(), start);
                }
                assert_eq!(start, tokens.len());
                assert_eq!(actual_logits.len(), tokens.len() * 64);
                if let Some(full) = &full_schedule_logits {
                    assert_eq!(&actual_logits, full, "full/chunk schedule logits");
                } else {
                    full_schedule_logits = Some(actual_logits.clone());
                    causal_logits.insert(case["name"].as_str().unwrap(), actual_logits);
                }
                qualify(
                    s.logits().unwrap(),
                    &logits[logits.len() - 64..],
                    &logits_spec["gate"],
                    "last row",
                );
                let before = s.logits().unwrap().to_vec();
                s.set_tokens(&tokens).unwrap();
                assert_eq!(s.evaluate_next_chunk().unwrap(), 0);
                assert_eq!(s.logits().unwrap(), before);
                if offset == 509 {
                    // Empty initial cache, three keys at absolute positions
                    // 509..511: a position boundary, not a 512-key quality run.
                    assert!(s.append(&[1]).is_err());
                    assert!(s.set_tokens(&[1; 4]).is_err());
                    assert_eq!(s.history(), tokens);
                    assert_eq!(s.evaluated_len(), 3);
                    assert_eq!(s.logits().unwrap(), before);
                } else {
                    s.append(&[1]).unwrap();
                    assert!(s.logits().is_none());
                    s.discard_pending();
                    assert_eq!(s.history(), tokens);
                    assert_eq!(s.logits().unwrap(), before);
                }
                s.reset();
                assert_eq!(s.evaluated_len(), 0);
                s.set_tokens(&tokens).unwrap();
                let committed = s.evaluate_next_chunk().unwrap();
                s.discard_pending();
                assert_eq!(s.evaluated_len(), committed);
                assert_eq!(s.history(), &tokens[..committed]);
                s.set_tokens(&tokens).unwrap();
                s.evaluate().unwrap();
                qualify(
                    s.logits().unwrap(),
                    &logits[logits.len() - 64..],
                    &logits_spec["gate"],
                    "resume",
                );
                assert_eq!(s.logits().unwrap(), before, "reset/resume distribution");
                if case["name"] == "prompt" {
                    s.set_tokens(&[1, 7, 13, 17]).unwrap();
                    assert_eq!(s.evaluated_len(), 3);
                    s.evaluate().unwrap();
                    let changed = fixture["cases"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|c| c["variant"] == variant && c["name"] == "future_changed")
                        .unwrap();
                    let spec = &changed["arrays"]["logits"];
                    let expected = scaled_array(spec);
                    qualify(
                        s.logits().unwrap(),
                        &expected[expected.len() - 64..],
                        &spec["gate"],
                        "prefix edit",
                    );
                }
            }
        }
        assert_eq!(
            &causal_logits["prompt"][..3 * 64],
            &causal_logits["future_changed"][..3 * 64],
            "future tokens must not change earlier distributions"
        );
    }
    assert_eq!(snapshots, 1680);
    println!(
        "{} {}",
        if positional {
            "LLAMA3_HALF_GRAPH"
        } else if fixture["format"] == "toklet-smollm3-half-graph-v1" {
            "SMOLLM3_HALF_GRAPH"
        } else {
            "GRANITE_HALF_GRAPH"
        },
        serde_json::json!({"format":fixture["format"],"snapshots":snapshots,"compared_values":compared,
        "diagnostic_stages_per_chunk":35,"fixture_stages":39,"cases":8,"schedules":3,
        "absolute_position_boundary":511,"boundary_initial_kv_tokens":0,
        "real_checkpoint_qualified":false,"iphone_qualified":false})
    );
}

#[test]
#[ignore = "Explicit Apple GPU mixed rotary mask graph at nonzero offsets and cached chunks"]
fn smollm3_half_graph_and_session_match_frozen_independent_oracle() {
    let raw = include_bytes!("fixtures/smollm3-half-graph-v1/fixture.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        "a6cf4fbc824e8a0af25721181be477015b907dd3f6d3c093501b4eba88813466"
    );
    let fixture: Value = serde_json::from_slice(raw).unwrap();
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for category in ["inputs", "producers"] {
        for (name, pin) in fixture[category].as_object().unwrap() {
            let bytes = fs::read(repo.join(name)).unwrap();
            assert_eq!(bytes.len() as u64, pin["bytes"]);
            assert_eq!(format!("{:x}", Sha256::digest(bytes)), pin["sha256"]);
        }
    }
    check_half_graph(&fixture, false);
}

#[test]
#[ignore = "Explicit Apple GPU Granite checkpoint scales, nonzero offsets and cached chunks"]
fn granite_half_graph_and_session_match_frozen_independent_oracle() {
    let raw = include_bytes!("fixtures/granite-half-graph-v1/fixture.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        "852711a8bfffd68f389f5db8a987ff6e4f56dd5e5d747bcbe89cccb089b58dd9"
    );
    let fixture: Value = serde_json::from_slice(raw).unwrap();
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for category in ["inputs", "producers"] {
        for (name, pin) in fixture[category].as_object().unwrap() {
            let bytes = fs::read(repo.join(name)).unwrap();
            assert_eq!(bytes.len() as u64, pin["bytes"]);
            assert_eq!(format!("{:x}", Sha256::digest(bytes)), pin["sha256"]);
        }
    }
    check_half_graph(&fixture, false);
}
