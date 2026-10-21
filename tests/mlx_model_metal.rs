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

fn root(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}
fn manifest() -> Value {
    let bytes = fs::read(root("mlx-half-v1").join("manifest.json")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        "1b394305deb39cee1076bfc867080dc2106eb05560b3a906e4a0ace526627a27"
    );
    serde_json::from_slice(&bytes).unwrap()
}
fn array(m: &Value, name: &str) -> Vec<f32> {
    let bytes = fs::read(root("mlx-half-v1").join("arrays.bin")).unwrap();
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
fn qualify(a: &[f32], b: &[f32], gate: &Value) {
    assert_eq!(a.len(), b.len());
    let atol = gate["atol"].as_f64().unwrap();
    let rtol = gate["rtol"].as_f64().unwrap();
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!(
            a.is_finite()
                && b.is_finite()
                && (f64::from(a) - f64::from(b)).abs() <= atol + rtol * f64::from(b).abs(),
            "index{i}: {a} vs {b}, gate {gate}"
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
fn model() -> metal::Model {
    metal::Model::load_mlx(
        &root("mlx-qwen-v1").join("model"),
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
#[ignore = "Explicit Apple GPU native MLX complete tiny graph, independent gates and strided logits"]
fn native_mlx_tiny_graph_matches_prefrozen_independent_gates() {
    let m = manifest();
    let model = model();
    assert!(metal::Session::new(&model, options(3, 0)).is_err());
    let mut snapshots = 0;
    let mut values = 0;
    for (case, spec) in m["cases"].as_object().unwrap() {
        let tokens: Vec<u32> = spec["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_u64().unwrap() as u32)
            .collect();
        let offset = spec["position_offset"].as_u64().unwrap() as usize;
        for chunk in [tokens.len(), 3, 1] {
            let mut s = session(
                &model,
                chunk,
                offset,
                ReadbackMode::FullLogits,
                AttentionMode::Reference,
            );
            s.enable_diagnostics(&[0, 1]).unwrap();
            s.set_tokens(&tokens).unwrap();
            s.evaluate().unwrap();
            let prefix = format!("cases.{case}.");
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
            for d in s.diagnostics().unwrap() {
                let name = stage(case, &d);
                assert!(
                    expected_names.remove(&name),
                    "duplicate/unexpected diagnostic {name}"
                );
                let shape = m["arrays"][&name]["shape"].as_array().unwrap();
                let columns = shape[1..]
                    .iter()
                    .map(|d| d.as_u64().unwrap() as usize)
                    .product::<usize>();
                assert_eq!(d.columns, columns);
                assert_eq!(d.rows, (tokens.len() - 1) % chunk + 1);
                assert_eq!(d.values.len(), d.rows * d.columns);
                let expected = array(&m, &name);
                let start = (tokens.len() - d.rows) * d.columns;
                qualify(&d.values, &expected[start..], &m["arrays"][&name]["gate"]);
                snapshots += 1;
                values += d.values.len();
            }
            assert!(
                expected_names.is_empty(),
                "missing diagnostics {expected_names:?}"
            );
            assert_eq!(s.evaluated_len(), tokens.len());
            assert_eq!(s.timing().unwrap().command_count, 1);
            let saved = s.logits().unwrap().to_vec();
            s.set_tokens(&tokens).unwrap();
            assert_eq!(s.evaluate_next_chunk().unwrap(), 0);
            assert_eq!(s.logits().unwrap(), saved);
        }
    }
    assert_eq!(snapshots, 525);
    // No diagnostic rows: prefill projects only the last input view, whose
    // nonzero MPS byte offset must be honored by the packed tied output.
    let tokens = [1, 7, 13, 3];
    let mut s = session(
        &model,
        32,
        0,
        ReadbackMode::FullLogits,
        AttentionMode::Reference,
    );
    s.enable_profile().unwrap();
    s.set_tokens(&tokens).unwrap();
    s.evaluate().unwrap();
    let expected = array(&m, "cases.prompt.logits");
    qualify(
        s.logits().unwrap(),
        &expected[expected.len() - 64..],
        &m["arrays"]["cases.prompt.logits"]["gate"],
    );
    let profile = s.profile().unwrap();
    assert_eq!(profile.events.len(), 38);
    assert!(profile.events.iter().any(|e| e.name == "mlx_product"));
    println!(
        "MLX_MODEL_GRAPH {}",
        serde_json::json!({"device":model.device_name(),"snapshots":snapshots,"compared_values":values,"weights_bytes":model.weights_bytes(),"profile_events":profile.events.len(),"quality_qualified":false,"performance_qualified":false,"iphone_qualified":false})
    );
}

#[test]
#[ignore = "Explicit Apple GPU native MLX cached prefix, pending samples, token-only and online attention"]
fn native_mlx_cached_pending_and_token_only_execution() {
    let m = manifest();
    let model = model();
    let mut selected = 0;
    for attention in [AttentionMode::Reference, AttentionMode::Online] {
        for readback in [ReadbackMode::FullLogits, ReadbackMode::GreedyToken] {
            let mut s = session(&model, 32, 0, readback, attention);
            s.set_tokens(&[1, 7, 13, 3]).unwrap();
            for step in 0..4 {
                s.evaluate().unwrap();
                let name = format!("greedy.step_{step}.logits");
                if let Some(logits) = s.logits() {
                    qualify(logits, &array(&m, &name), &m["arrays"][&name]["gate"]);
                }
                let token = s.greedy_token().unwrap().unwrap();
                assert_eq!(token, 47);
                selected += 1;
                s.append(&[token]).unwrap();
                assert_eq!(s.pending_len(), 1);
                assert!(s.greedy_token().is_none());
            }
            assert_eq!(s.evaluated_len(), 7);
            assert_eq!(s.history().len(), 8);
            let history = s.history().to_vec();
            s.set_tokens(&history).unwrap();
            assert_eq!(s.evaluated_len(), 7);
            s.evaluate().unwrap();
            assert_eq!(s.evaluated_len(), 8);
            // Changed prefix and reset/replay agree with a fresh same-mode session.
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
            if let Some(logits) = s.logits() {
                assert_eq!(logits, fresh.logits().unwrap());
            }
            s.reset();
            assert_eq!(s.evaluated_len(), 0);
        }
    }
    // Cancellation boundary: one real partial-prefill chunk, then changed input.
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
    let changed = [1, 7, 13, 17];
    s.set_tokens(&changed).unwrap();
    s.evaluate().unwrap();
    let expected = array(&m, "cases.future_changed.logits");
    qualify(
        s.logits().unwrap(),
        &expected[expected.len() - 64..],
        &m["arrays"]["cases.future_changed.logits"]["gate"],
    );
    println!(
        "MLX_MODEL_STATE {}",
        serde_json::json!({"device":model.device_name(),"selected_ids":selected,"partial_prefill_committed":3,"pending_and_changed_prefix_passed":true,"model_quality_qualified":false,"iphone_qualified":false})
    );
}

struct Artifact(PathBuf);
impl Artifact {
    fn copy() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "toklet-mlx-model-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        for name in ["config.json", "model.safetensors"] {
            fs::copy(
                root("mlx-qwen-v1").join("model").join(name),
                path.join(name),
            )
            .unwrap();
        }
        Self(path)
    }
    fn change(&self, name: &str, index: Option<usize>, value: f32) {
        let path = self.0.join("model.safetensors");
        let mut data = fs::read(&path).unwrap();
        let header_size = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
        let header: Value = serde_json::from_slice(&data[8..8 + header_size]).unwrap();
        assert_eq!(header[name]["dtype"], "F32");
        let begin = 8 + header_size + header[name]["data_offsets"][0].as_u64().unwrap() as usize;
        let end = 8 + header_size + header[name]["data_offsets"][1].as_u64().unwrap() as usize;
        if let Some(index) = index {
            data[begin + index * 4..begin + (index + 1) * 4].copy_from_slice(&value.to_le_bytes());
        } else {
            for raw in data[begin..end].chunks_exact_mut(4) {
                raw.copy_from_slice(&value.to_le_bytes());
            }
        }
        fs::write(path, data).unwrap();
    }
    fn floating() -> Self {
        use safetensors::{Dtype, tensor::TensorView};
        use toklet::mlx::{
            affine::Matrix,
            qwen::{Source, Storage},
        };
        let artifact = Self::copy();
        let mut config: Value =
            serde_json::from_slice(&fs::read(artifact.0.join("config.json")).unwrap()).unwrap();
        config.as_object_mut().unwrap().remove("quantization");
        let mut source = Source::open(
            &root("mlx-qwen-v1").join("model"),
            LoadOptions {
                context_limit: 32,
                ..Default::default()
            },
        )
        .unwrap();
        let specs = source.tensors().clone();
        let mut tensors = std::collections::BTreeMap::new();
        for (name, spec) in specs {
            let mut values = Vec::new();
            match spec.storage() {
                Storage::Float(_) => source
                    .visit_float(&name, |v| {
                        values.push(v);
                        Ok(())
                    })
                    .unwrap(),
                Storage::Affine(l) => {
                    let mut w = vec![0; l.weight_bytes()];
                    let mut s = vec![0; l.metadata_bytes()];
                    let mut b = s.clone();
                    source.read_affine(&name, &mut w, &mut s, &mut b).unwrap();
                    let m = Matrix::from_le_bytes(l, &w, &s, &b).unwrap();
                    for r in 0..l.rows() {
                        for c in 0..l.columns() {
                            values.push(m.value(r, c).unwrap());
                        }
                    }
                }
            }
            tensors.insert(
                name,
                (
                    spec.shape().to_vec(),
                    values
                        .into_iter()
                        .flat_map(f32::to_le_bytes)
                        .collect::<Vec<_>>(),
                ),
            );
        }
        let views: Vec<_> = tensors
            .iter()
            .map(|(n, (s, b))| {
                (
                    n.as_str(),
                    TensorView::new(Dtype::F32, s.clone(), b).unwrap(),
                )
            })
            .collect();
        safetensors::serialize_to_file(views, None, &artifact.0.join("model.safetensors")).unwrap();
        fs::write(
            artifact.0.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        artifact
    }
    fn load(&self) -> Result<metal::Model, metal::Error> {
        metal::Model::load_mlx(
            &self.0,
            LoadOptions {
                context_limit: 32,
                memory_limit_bytes: 128 * 1024 * 1024,
            },
        )
    }
}
impl Drop for Artifact {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
#[ignore = "Explicit Apple GPU MLX loader rejection, cleanup, mode admission and gather-failure publication"]
fn native_mlx_loader_and_gather_failure_preserve_model_session_boundaries() {
    let path = root("mlx-qwen-v1").join("model");
    assert!(matches!(
        metal::Model::load_mlx(
            &path,
            LoadOptions {
                context_limit: 32,
                memory_limit_bytes: 64 * 1024 * 1024
            }
        ),
        Err(metal::Error::Weights(toklet::weights::Error::Budget { .. }))
    ));
    let float = Artifact::copy();
    float.change("model.norm.weight", None, 1e10);
    let error = match float.load() {
        Ok(_) => panic!("accepted Half overflow"),
        Err(e) => e.to_string(),
    };
    assert!(error.contains("finite Half"), "{error}");
    let domain = Artifact::copy();
    domain.change("model.embed_tokens.scales", None, 0.);
    domain.change("model.embed_tokens.biases", None, 1e-30);
    assert!(matches!(domain.load(), Err(metal::Error::Unsupported(_))));
    let floating = Artifact::floating();
    let normal = metal::Model::load(
        &floating.0,
        LoadOptions {
            context_limit: 32,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(
        metal::Session::with_projection_mode(&normal, options(3, 0), ProjectionMode::MlxAffine)
            .is_err()
    );
    let overflow = Artifact::copy();
    overflow.change("model.embed_tokens.scales", Some(47), 0.);
    overflow.change("model.embed_tokens.biases", Some(47), 1e10);
    let overflow = overflow.load().unwrap();
    for readback in [ReadbackMode::FullLogits, ReadbackMode::GreedyToken] {
        let mut s = session(&overflow, 3, 0, readback, AttentionMode::Reference);
        s.set_tokens(&[1, 7, 13, 3]).unwrap();
        s.evaluate().unwrap();
        let committed = s.logits().map(<[f32]>::to_vec);
        let selected = s.greedy_token().unwrap().unwrap();
        assert_eq!(s.evaluated_len(), 4);
        s.append(&[47]).unwrap();
        assert!(matches!(
            s.evaluate_next_chunk(),
            Err(metal::Error::Nonfinite)
        ));
        assert_eq!(s.evaluated_len(), 4);
        assert!(s.logits().is_none());
        assert!(s.greedy_token().is_none());
        // A completed nonfinite chunk preserves the valid prior prefix; only
        // command failure requires reset/replay. Recover the cached distribution.
        s.set_tokens(&[1, 7, 13, 3]).unwrap();
        assert_eq!(s.evaluated_len(), 4);
        assert_eq!(s.evaluate_next_chunk().unwrap(), 0);
        assert_eq!(s.greedy_token().unwrap().unwrap(), selected);
        assert_eq!(s.logits(), committed.as_deref());
    }
    let valid = model();
    let mut s = session(
        &valid,
        3,
        0,
        ReadbackMode::FullLogits,
        AttentionMode::Reference,
    );
    s.set_tokens(&[1, 7, 13, 3]).unwrap();
    s.evaluate().unwrap();
    assert!(s.logits().unwrap().iter().all(|x| x.is_finite()));
    println!(
        "MLX_MODEL_ADMISSION {}",
        serde_json::json!({"device":valid.device_name(),"loader_rejections":3,"mode_mismatch_directions":2,"gather_failure_valid_prefix":4,"cleanup_and_valid_reload":true,"quality_qualified":false})
    );
}
