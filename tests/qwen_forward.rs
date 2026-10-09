use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};
use toklet::{
    model::{LayerTrace, Trace},
    sampling,
    weights::{self, LoadOptions},
};

fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen-v2")
}
fn read_array(root: &Path, array: &Value) -> Vec<f32> {
    let bytes = fs::read(root.join(array["file"].as_str().unwrap())).unwrap();
    assert_eq!(bytes.len() as u64, array["bytes"].as_u64().unwrap());
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        array["sha256"].as_str().unwrap()
    );
    assert!(bytes.len().is_multiple_of(4));
    bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

#[test]
fn real_fp32_probes_have_frozen_identity_selection_and_independent_gates() {
    let root = fixture_root();
    let manifest = read_json(&root.join("manifest.json"));
    let sources =
        read_json(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/sources.json"));
    assert_eq!(manifest["model"], sources["model"]);
    assert_eq!(manifest["policy"]["atol_floor"], 1e-4);
    assert_eq!(manifest["policy"]["envelope_multiplier"], 2.0);
    assert_eq!(manifest["policy"]["rtol"], 1e-4);
    assert_eq!(manifest["layers"], json!([0, 1, 23]));
    assert_eq!(manifest["dtype"], "FP32");
    assert_eq!(manifest["device"], "CPU");
    assert_eq!(manifest["attention"], "eager");
    assert_eq!(manifest["cache"], "none");
    assert_eq!(manifest["arrays"].as_object().unwrap().len(), 228);
    for array in manifest["arrays"].as_object().unwrap().values() {
        let values = read_array(&root, array);
        let indices = array["indices"].as_array().unwrap();
        assert_eq!(values.len(), indices.len());
        assert!(values.iter().all(|x| x.is_finite()));
        let elements = array["shape"]
            .as_array()
            .unwrap()
            .iter()
            .try_fold(1_u64, |count, dimension| {
                count.checked_mul(dimension.as_u64().unwrap())
            })
            .unwrap();
        assert_eq!(array["full_array_bytes"].as_u64().unwrap(), elements * 4);
        assert_eq!(array["full_array_sha256"].as_str().unwrap().len(), 64);
        assert!(
            indices
                .iter()
                .all(|index| index.as_u64().unwrap() < elements)
        );
        assert!(
            indices
                .windows(2)
                .all(|pair| pair[0].as_u64().unwrap() < pair[1].as_u64().unwrap())
        );
        let envelope = &array["reference_envelope"];
        let atol = (2.0
            * envelope["pooled_fp32_fp64_max_absolute_error"]
                .as_f64()
                .unwrap())
        .max(
            2.0 * envelope["equivalent_prefix_max_absolute_error"]
                .as_f64()
                .unwrap(),
        )
        .max(1e-4);
        assert_eq!(envelope["atol"].as_f64().unwrap(), atol);
        assert!(envelope["maximum_tolerance_fraction"].as_f64().unwrap() <= 1.0);
    }
}

fn layer_values(layer: &LayerTrace) -> [(&'static str, &[f32]); 18] {
    [
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
    ]
}
fn trace_values(trace: &Trace) -> BTreeMap<String, &[f32]> {
    let mut values = BTreeMap::from([
        ("embedding".into(), trace.embedding.as_slice()),
        ("final_norm".into(), trace.final_norm.as_slice()),
        ("logits".into(), trace.logits.as_slice()),
    ]);
    for index in [0, 1, 23] {
        for (name, array) in layer_values(&trace.layers[index]) {
            values.insert(format!("layers.{index}.{name}"), array);
        }
    }
    values
}
fn nll(logits: &[f32], target: u32) -> f64 {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    maximum
        + logits
            .iter()
            .map(|&x| (f64::from(x) - maximum).exp())
            .sum::<f64>()
            .ln()
        - f64::from(logits[target as usize])
}

#[test]
#[ignore = "requires pinned Qwen weights and explicitly regenerated full Transformers probes"]
fn pinned_qwen_fp32_matches_full_transformers_forward() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let full = root.join("artifacts/qwen-forward-v2");
    let manifest = read_json(&full.join("manifest.json"));
    let probes = read_json(&fixture_root().join("manifest.json"));
    for key in [
        "policy",
        "model",
        "cases",
        "layers",
        "versions",
        "argmax",
        "exporter_sha256",
    ] {
        assert_eq!(
            manifest[key], probes[key],
            "full/committed identity mismatch for {key}"
        );
    }
    // Bind qualification to the actual checkpoint, not its path/config name.
    use std::io::Read;
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
        manifest["model"]["weights"]["sha256"].as_str().unwrap()
    );
    let loaded = weights::load_qwen(
        &root.join("models/Qwen2.5-0.5B-Instruct"),
        LoadOptions::default(),
    )
    .unwrap();
    let mut scratch = loaded.model.scratch(4).unwrap();
    let mut results = BTreeMap::new();
    let mut failures = Vec::new();
    for (case, spec) in manifest["cases"].as_object().unwrap() {
        let tokens: Vec<u32> = spec["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|token| token.as_u64().unwrap() as u32)
            .collect();
        let offset = spec["offset"].as_u64().unwrap() as usize;
        let trace = loaded
            .model
            .forward_diagnostic(&tokens, offset, &mut scratch)
            .unwrap();
        let values = trace_values(&trace);
        let mut metrics = BTreeMap::new();
        for (name, actual) in values {
            let key = format!("cases.{case}.{name}");
            let array = &manifest["arrays"][&key];
            let committed = &probes["arrays"][&key];
            assert_eq!(
                array["sha256"], committed["full_array_sha256"],
                "full identity for {key}"
            );
            assert_eq!(
                array["bytes"], committed["full_array_bytes"],
                "full byte count for {key}"
            );
            let expected = read_array(&full, array);
            assert_eq!(actual.len(), expected.len(), "{key}");
            let atol = array["reference_envelope"]["atol"].as_f64().unwrap();
            assert_eq!(
                array["reference_envelope"],
                probes["arrays"][&key]["reference_envelope"]
            );
            let mut maximum_error = 0.0_f64;
            let mut maximum_fraction = 0.0_f64;
            let mut maximum_relative_error = 0.0_f64;
            for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                let error = (f64::from(actual) - f64::from(expected)).abs();
                let fraction = error / (atol + 1e-4 * f64::from(expected).abs());
                assert!(
                    actual.is_finite(),
                    "{key} index {index}: nonfinite {actual}"
                );
                maximum_error = maximum_error.max(error);
                maximum_fraction = maximum_fraction.max(fraction);
                maximum_relative_error =
                    maximum_relative_error.max(error / f64::from(expected).abs().max(1e-6));
            }
            if maximum_fraction > 1.0 {
                failures.push(format!("{key}: maximum absolute error {maximum_error}, tolerance fraction {maximum_fraction}"));
            }
            let selected = read_array(&fixture_root(), &probes["arrays"][&key]);
            for (value, index) in selected
                .iter()
                .zip(probes["arrays"][&key]["indices"].as_array().unwrap())
            {
                assert_eq!(*value, expected[index.as_u64().unwrap() as usize]);
            }
            metrics.insert(name, json!({"maximum_absolute_error": maximum_error, "maximum_tolerance_fraction": maximum_fraction, "maximum_relative_error_floor_1e-6": maximum_relative_error}));
        }
        let width = loaded.model.config().vocab_size;
        let expected_logits =
            read_array(&full, &manifest["arrays"][format!("cases.{case}.logits")]);
        let argmax: Vec<u32> = trace
            .logits
            .chunks_exact(width)
            .map(|row| sampling::greedy(row).unwrap())
            .collect();
        assert_eq!(json!(argmax), manifest["argmax"][case]);
        let losses: Vec<f64> = tokens
            .iter()
            .skip(1)
            .enumerate()
            .map(|(position, &target)| {
                nll(
                    &trace.logits[position * width..(position + 1) * width],
                    target,
                ) - nll(
                    &expected_logits[position * width..(position + 1) * width],
                    target,
                )
            })
            .collect();
        assert!(
            losses.iter().all(|difference| difference.abs() <= 1e-3),
            "teacher-forced short-probe loss {losses:?}"
        );
        let mut last = vec![0.0; width];
        loaded
            .model
            .forward(&tokens, offset, &mut scratch, &mut last)
            .unwrap();
        assert_eq!(last, trace.logits[(tokens.len() - 1) * width..]);
        eprintln!(
            "{case}: full forward probes, every-position argmax and final-only path match; short-target NLL deltas {losses:?}"
        );
        results.insert(
            case.clone(),
            json!({"arrays": metrics, "argmax": argmax, "short_teacher_forced_nll_deltas": losses}),
        );
    }
    let report = json!({"model": manifest["model"], "policy": manifest["policy"], "results": results, "quality_scope":"short numerical probes only; no held-out loss/task qualification"});
    let output = root.join("artifacts/qwen-parity-results.json");
    fs::write(&output, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    eprintln!("Full numerical metrics: {}", output.display());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
