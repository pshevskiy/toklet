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
