use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
use toklet::{
    mlx::{
        affine::MetadataDtype,
        compatibility::{self, Backend, Reason, Status},
        dense::Architecture,
        qwen::{Source, Storage},
        reference,
    },
    model::{PositionPlan, Trace},
    session::{self, Session},
    weights::{Error, LoadOptions},
};

#[path = "support/phi3.rs"]
mod fixture;

fn load_options() -> LoadOptions {
    LoadOptions {
        context_limit: 512,
        memory_limit_bytes: 128 * 1024 * 1024,
    }
}

fn trace_arrays(trace: &Trace) -> BTreeMap<String, &[f32]> {
    let mut arrays = BTreeMap::from([
        ("embedding".into(), trace.embedding.as_slice()),
        ("final_norm".into(), trace.final_norm.as_slice()),
        ("logits".into(), trace.logits.as_slice()),
    ]);
    for (index, layer) in trace.layers.iter().enumerate() {
        assert!(layer.q_norm.is_empty() && layer.k_norm.is_empty());
        for (name, values) in [
            ("input_norm", &layer.input_norm),
            ("q_projection", &layer.q_projection),
            ("k_projection", &layer.k_projection),
            ("v_projection", &layer.v_projection),
            ("q_rotary", &layer.q_rotary),
            ("k_rotary", &layer.k_rotary),
            ("attention_scores", &layer.attention_scores),
            ("attention_probabilities", &layer.attention_probabilities),
            ("attention_context", &layer.attention_context),
            ("attention_projection", &layer.attention_projection),
            ("attention_residual", &layer.attention_residual),
            ("post_attention_norm", &layer.post_attention_norm),
            ("gate_projection", &layer.gate_projection),
            ("up_projection", &layer.up_projection),
            ("gate_silu", &layer.gate_silu),
            ("gated", &layer.gated),
            ("down_projection", &layer.down_projection),
            ("output", &layer.output),
        ] {
            arrays.insert(format!("layers.{index}.{name}"), values.as_slice());
        }
    }
    arrays
}

fn close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite()
                && b.is_finite()
                && (f64::from(a) - f64::from(b)).abs() <= 1e-5 + 1e-4 * f64::from(b).abs(),
            "{label} index{index}: {a} vs {b}"
        );
    }
}

#[test]
fn original_fused_inventory_preserves_source_precision_and_bounded_semantics() {
    let (root, manifest, _) = fixture::open();
    for (variant, physical_count, payload, fp32) in [
        ("untied", 15, 347_520, 2_461_440),
        ("tied", 14, 340_608, 2_412_288),
    ] {
        let path = root.join(variant);
        let source = Source::open_dense(&path, load_options()).unwrap();
        assert_eq!(source.architecture(), Architecture::Phi3);
        assert_eq!(source.position_plan(), PositionPlan::Ordinary);
        assert_eq!(source.declared_positions(), 4096);
        assert_eq!(source.declared_sliding_window(), Some(2047));
        assert_eq!(source.config().max_positions, 512);
        assert_eq!(source.config().head_dim, 96);
        assert_eq!(source.tied_output(), variant == "tied");
        assert_eq!(source.tensors().len(), physical_count);
        assert_eq!(source.tensor_payload_bytes(), payload);
        assert_eq!(
            source
                .tensors()
                .values()
                .map(|s| s.payload_bytes())
                .sum::<usize>(),
            payload
        );
        assert!(
            !source
                .tensors()
                .contains_key("model.layers.0.self_attn.q_proj.weight")
        );
        let qkv = &source.tensors()["model.layers.0.self_attn.qkv_proj.weight"];
        assert_eq!(qkv.shape(), [576, 192]);
        let Storage::Affine(layout) = qkv.storage() else {
            panic!()
        };
        assert_eq!(
            (layout.bits(), layout.group_size(), layout.metadata_dtype()),
            (4, 64, MetadataDtype::F16)
        );
        assert_eq!(
            manifest["source_details"][variant]["data_start"]
                .as_u64()
                .unwrap()
                % 4,
            1
        );
        assert!(Source::open(&path, load_options()).is_err());

        let loaded = reference::load_dense(&path, load_options()).unwrap();
        assert_eq!(loaded.fp32_weights_bytes, fp32);
        assert_eq!(loaded.temporary_packed_bytes, 27_648);
        assert!(loaded.temporary_packed_bytes < qkv.payload_bytes());
        assert_eq!(
            loaded.admitted_load_bytes,
            fp32 + 27_648 + 64 * 1024 + 64 * 1024 * 1024
        );
        let lower = LoadOptions {
            memory_limit_bytes: loaded.admitted_load_bytes - 1,
            ..load_options()
        };
        assert!(
            matches!(reference::load_dense(&path, lower), Err(Error::Budget { required, limit }) if required == loaded.admitted_load_bytes && limit == lower.memory_limit_bytes)
        );
    }
}

#[test]
fn complete_cpu_head96_graph_and_cached_schedules_match_frozen_independent_source() {
    let (root, manifest, arrays) = fixture::open();
    // arbitrary_precision JSON retains exponent spelling; compare numeric policy.
    assert_eq!(manifest["cpu_gate"]["atol"].as_f64(), Some(1e-5));
    assert_eq!(manifest["cpu_gate"]["rtol"].as_f64(), Some(1e-4));
    let mut stages = 0;
    let mut compared = 0;
    for variant in ["untied", "tied"] {
        let loaded = reference::load_dense(&root.join(variant), load_options()).unwrap();
        let mut previous_prefix = None;
        for case in manifest["cases"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|c| c["variant"] == variant)
        {
            let tokens: Vec<u32> = case["tokens"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            let offset = case["position_offset"].as_u64().unwrap() as usize;
            let mut scratch = loaded.model.scratch(tokens.len()).unwrap();
            let trace = loaded
                .model
                .forward_diagnostic(&tokens, offset, &mut scratch)
                .unwrap();
            let actual = trace_arrays(&trace);
            assert_eq!(actual.len(), 39);
            for (name, values) in &actual {
                let expected = fixture::values(&arrays, &case["arrays"][name]["profiles"]["cpu"]);
                close(values, &expected, &format!("{} {name}", case["name"]));
                stages += 1;
                compared += values.len();
            }
            if case["name"] == "prompt" {
                previous_prefix = Some(trace.logits[..16 * 64].to_vec());
            }
            if case["name"] == "future_changed" {
                assert_eq!(
                    &trace.logits[..16 * 64],
                    previous_prefix.as_ref().unwrap().as_slice()
                );
            }
            let mut schedules = vec![1, 3, 16, tokens.len()];
            schedules.retain(|&n| n <= tokens.len());
            schedules.sort_unstable();
            schedules.dedup();
            for chunk in schedules {
                let mut session = Session::new(
                    &loaded.model,
                    session::Options {
                        capacity: tokens.len(),
                        prefill_chunk_size: chunk,
                        position_offset: offset,
                        memory_limit_bytes: 128 * 1024 * 1024,
                    },
                )
                .unwrap();
                session.set_tokens(&tokens).unwrap();
                let mut cached = Vec::new();
                while session.pending_len() > 0 {
                    let count = chunk.min(session.pending_len());
                    let mut rows = vec![0.; count * 64];
                    assert_eq!(
                        session.evaluate_next_chunk_with_logits(&mut rows).unwrap(),
                        count
                    );
                    cached.extend(rows);
                }
                assert_eq!(cached, trace.logits);
                assert_eq!(session.evaluated_len(), tokens.len());
                assert_eq!(session.evaluate_next_chunk().unwrap(), 0);
                assert!(session.append(&[1]).is_err());
                assert_eq!(session.evaluated_len(), tokens.len());
                session.reset();
                session.set_tokens(&tokens).unwrap();
                session.evaluate().unwrap();
                assert_eq!(
                    session.logits().unwrap(),
                    &trace.logits[trace.logits.len() - 64..]
                );
            }
        }
    }
    assert_eq!(stages, 195);
    println!(
        "PHI_CPU {}",
        json!({"stage_groups":stages,"compared_values":compared,"real_model_qualified":false})
    );
}

struct CopyArtifact(PathBuf);
impl CopyArtifact {
    fn new(root: &Path) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "toklet-phi3-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        for name in ["config.json", "model.safetensors"] {
            fs::copy(root.join(name), path.join(name)).unwrap();
        }
        Self(path)
    }
    fn change(&self, key: &str, value: Value) {
        let path = self.0.join("config.json");
        let mut config: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        config[key] = value;
        fs::write(path, serde_json::to_vec(&config).unwrap()).unwrap();
    }
}
impl Drop for CopyArtifact {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn window_and_absolute_position_limits_reject_before_payload_access_or_session_mutation() {
    let (root, _, _) = fixture::open();
    let a = CopyArtifact::new(&root.join("untied"));
    a.change("sliding_window", json!(31));
    for positions in [1, 17, 31] {
        let options = LoadOptions {
            context_limit: positions,
            ..load_options()
        };
        let source = Source::open_dense(&a.0, options).unwrap();
        assert_eq!(source.config().max_positions, positions);
        assert_eq!(source.declared_positions(), 4096);
        assert_eq!(source.declared_sliding_window(), Some(31));
    }
    let loaded = reference::load_dense(
        &a.0,
        LoadOptions {
            context_limit: 31,
            ..load_options()
        },
    )
    .unwrap();
    for (capacity, offset) in [(32, 0), (1, 31), (1, usize::MAX)] {
        assert!(
            Session::new(
                &loaded.model,
                session::Options {
                    capacity,
                    prefill_chunk_size: 1,
                    position_offset: offset,
                    memory_limit_bytes: 128 * 1024 * 1024
                }
            )
            .is_err()
        );
    }
    fs::remove_file(a.0.join("model.safetensors")).unwrap();
    for positions in [32, 513, 4097] {
        assert!(matches!(
            Source::open_dense(
                &a.0,
                LoadOptions {
                    context_limit: positions,
                    ..load_options()
                }
            ),
            Err(Error::Invalid(_))
        ));
    }
}

#[test]
fn compatibility_reports_source_window_and_keeps_native_domain_and_text_qualification_separate() {
    let (root, _, _) = fixture::open();
    let path = root.join("untied");
    for backend in [Backend::ScalarReference, Backend::Metal] {
        let report = compatibility::inspect(&path, backend, load_options());
        assert_eq!(report.status, Status::CompatibleUntested);
        assert_eq!(report.configured_max_positions, Some(4096));
        assert_eq!(report.configured_sliding_window, Some(2047));
        assert_eq!(report.requested_context, 512);
        assert!(!report.text_profile_checked && !report.qualification_evaluated);
    }
    let old = compatibility::inspect(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mlx-llama-v1/untied"),
        Backend::ScalarReference,
        LoadOptions {
            context_limit: 32,
            ..load_options()
        },
    );
    assert!(
        !serde_json::to_value(old)
            .unwrap()
            .as_object()
            .unwrap()
            .contains_key("configured_sliding_window")
    );
    for (key, value) in [("rope_theta", json!(0.5)), ("rms_norm_eps", json!(1e-40))] {
        let a = CopyArtifact::new(&path);
        a.change(key, value);
        assert!(Source::open_dense(&a.0, load_options()).is_ok());
        assert_eq!(
            compatibility::inspect(&a.0, Backend::Metal, load_options()).reason,
            Reason::UnsupportedGeometry
        );
    }
}
