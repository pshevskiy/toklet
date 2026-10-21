use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};
use toklet::{
    mlx::reference,
    model::{LayerTrace, Trace},
    session::{self, Session},
    weights::{Error, LoadOptions},
};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mlx-qwen-v1")
}
fn manifest() -> Value {
    let r = root();
    let m: Value = serde_json::from_slice(&fs::read(r.join("manifest.json")).unwrap()).unwrap();
    for (name, e) in m["files"].as_object().unwrap() {
        let b = fs::read(r.join(name)).unwrap();
        assert_eq!(b.len() as u64, e["bytes"]);
        assert_eq!(format!("{:x}", Sha256::digest(&b)), e["sha256"]);
    }
    assert_eq!(m["policy"]["tolerances"]["logits"]["atol"], 1e-5);
    assert_eq!(m["policy"]["tolerances"]["logits"]["rtol"], 1e-4);
    m
}
fn array(m: &Value, name: &str) -> Vec<f32> {
    let e = &m["arrays"][name];
    let b = fs::read(root().join("arrays.bin")).unwrap();
    let p = e["offset"].as_u64().unwrap() as usize;
    let n = e["bytes"].as_u64().unwrap() as usize;
    let data = &b[p..p + n];
    assert_eq!(format!("{:x}", Sha256::digest(data)), e["sha256"]);
    data.chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect()
}
fn values(trace: &Trace) -> BTreeMap<String, &[f32]> {
    let mut result = BTreeMap::from([
        ("embedding".into(), trace.embedding.as_slice()),
        ("final_norm".into(), trace.final_norm.as_slice()),
        ("logits".into(), trace.logits.as_slice()),
    ]);
    for (i, l) in trace.layers.iter().enumerate() {
        for (name, data) in layer(l) {
            result.insert(format!("layers.{i}.{name}"), data);
        }
    }
    result
}
fn layer(l: &LayerTrace) -> [(&str, &[f32]); 18] {
    [
        ("input_norm", &l.input_norm),
        ("q_projection", &l.q_projection),
        ("k_projection", &l.k_projection),
        ("v_projection", &l.v_projection),
        ("q_rotary", &l.q_rotary),
        ("k_rotary", &l.k_rotary),
        ("attention_scores", &l.attention_scores),
        ("attention_probabilities", &l.attention_probabilities),
        ("attention_context", &l.attention_context),
        ("attention_projection", &l.attention_projection),
        ("attention_residual", &l.attention_residual),
        ("post_attention_norm", &l.post_attention_norm),
        ("gate_projection", &l.gate_projection),
        ("up_projection", &l.up_projection),
        ("gate_silu", &l.gate_silu),
        ("gated", &l.gated),
        ("down_projection", &l.down_projection),
        ("output", &l.output),
    ]
}
fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-5 + 1e-4 * b.abs(),
            "index{i}: {a} vs {b}"
        );
    }
}
fn load(directory: &Path, limit: usize) -> Result<reference::Loaded, Error> {
    reference::load_qwen(
        directory,
        LoadOptions {
            context_limit: 32,
            memory_limit_bytes: limit,
        },
    )
}

#[test]
fn imported_mixed_mlx_model_matches_every_numpy_trace_and_native_mlx_logits() {
    let m = manifest();
    let loaded = load(&root().join("model"), 128 * 1024 * 1024).unwrap();
    let model = loaded.model;
    assert_eq!(
        model.allocated_weights_bytes().unwrap(),
        loaded.fp32_weights_bytes
    );
    assert_eq!(
        loaded.admitted_load_bytes,
        loaded.fp32_weights_bytes + loaded.temporary_packed_bytes + 64 * 1024 + 64 * 1024 * 1024
    );
    let mut arrays = 0;
    for (name, case) in m["cases"].as_object().unwrap() {
        let tokens: Vec<u32> = case["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let offset = case["position_offset"].as_u64().unwrap() as usize;
        let mut scratch = model.scratch(tokens.len()).unwrap();
        let trace = model
            .forward_diagnostic(&tokens, offset, &mut scratch)
            .unwrap();
        for (stage, actual) in values(&trace) {
            close(actual, &array(&m, &format!("cases.{name}.{stage}")));
            arrays += 1;
        }
        if offset == 0 {
            for backend in ["CPU", "GPU"] {
                close(
                    &trace.logits,
                    &array(&m, &format!("native.{backend}.{name}.logits")),
                );
            }
        }
    }
    assert_eq!(arrays, 195);
}

#[test]
fn expanded_reference_cached_and_offset_execution_keeps_the_same_prefix() {
    let m = manifest();
    let loaded = load(&root().join("model"), 128 * 1024 * 1024).unwrap();
    for (name, case) in m["cases"].as_object().unwrap() {
        let tokens: Vec<u32> = case["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let offset = case["position_offset"].as_u64().unwrap() as usize;
        let mut s = Session::new(
            &loaded.model,
            session::Options {
                capacity: 32 - offset,
                prefill_chunk_size: 3,
                position_offset: offset,
                memory_limit_bytes: 128 * 1024 * 1024,
            },
        )
        .unwrap();
        s.set_tokens(&tokens).unwrap();
        s.evaluate().unwrap();
        let expected = array(&m, &format!("cases.{name}.logits"));
        close(s.logits().unwrap(), &expected[expected.len() - 64..]);
        assert_eq!(s.evaluated_len(), tokens.len());
        let previous = s.logits().unwrap().to_vec();
        s.set_tokens(&tokens).unwrap();
        s.evaluate().unwrap();
        assert_eq!(s.logits().unwrap(), previous);
        assert_eq!(s.evaluated_len(), tokens.len());
        s.reset();
        assert_eq!(s.evaluated_len(), 0);
        s.set_tokens(&tokens).unwrap();
        s.evaluate().unwrap();
        assert_eq!(s.logits().unwrap(), previous);
    }
}

#[test]
fn expansion_admits_fp32_plus_largest_packed_temporary_before_loading() {
    manifest();
    let path = root().join("model");
    let loaded = load(&path, 128 * 1024 * 1024).unwrap();
    let required = loaded.admitted_load_bytes;
    drop(loaded);
    assert!(matches!(load(&path,required-1),Err(Error::Budget{required:r,..}) if r==required));
    assert!(load(&path, required).is_ok());
}

#[test]
fn invalid_payload_never_publishes_an_expanded_model() {
    let folder = std::env::temp_dir().join(format!(
        "toklet-mlx-reference-invalid-{}",
        std::process::id()
    ));
    fs::create_dir(&folder).unwrap();
    struct Cleanup(PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }
    let _cleanup = Cleanup(folder.clone());
    fs::copy(root().join("model/config.json"), folder.join("config.json")).unwrap();
    let original = fs::read(root().join("model/model.safetensors")).unwrap();
    let header_size = u64::from_le_bytes(original[..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&original[8..8 + header_size]).unwrap();
    for packed in [false, true] {
        let mut data = original.clone();
        let mut change = |name: &str, value: f32| {
            let offsets = header[name]["data_offsets"].as_array().unwrap();
            let begin = 8 + header_size + offsets[0].as_u64().unwrap() as usize;
            let end = 8 + header_size + offsets[1].as_u64().unwrap() as usize;
            for raw in data[begin..end].chunks_exact_mut(4) {
                raw.copy_from_slice(&value.to_le_bytes());
            }
        };
        if packed {
            change("model.embed_tokens.scales", f32::MAX);
            change("model.embed_tokens.biases", f32::MAX);
        } else {
            change("model.norm.weight", f32::NAN);
        }
        fs::write(folder.join("model.safetensors"), data).unwrap();
        let result = load(&folder, 128 * 1024 * 1024);
        let error = match result {
            Ok(_) => panic!("invalid payload published a model"),
            Err(e) => e.to_string(),
        };
        assert!(
            error.contains(if packed {
                "reconstruction overflow"
            } else {
                "nonfinite"
            }),
            "{error}"
        );
    }
}
