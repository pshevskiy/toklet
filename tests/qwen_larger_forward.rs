use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};
use toklet::{
    cpu::ProjectionKernel,
    model::{LayerTrace, Trace},
    sampling,
    session::{Options, Session},
    weights::{self, LoadOptions},
};
#[path = "support/trace_parity.rs"]
mod trace_parity;

fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen-larger-v1")
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
fn larger_fp32_probes_have_frozen_identity_selection_and_independent_gates() {
    let root = fixture_root();
    let manifest = read_json(&root.join("manifest.json"));
    let sources = read_json(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/m5-larger-sources-v1.json"),
    );
    assert_eq!(manifest["model"], sources["model"]);
    assert_eq!(manifest["policy"]["atol_floor"], 1e-4);
    assert_eq!(manifest["policy"]["envelope_multiplier"], 2.0);
    assert_eq!(manifest["policy"]["rtol"], 1e-4);
    assert_eq!(manifest["layers"], json!([0, 1, 27]));
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
    for index in [0, 1, 27] {
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
#[ignore = "requires the preselected larger Qwen checkpoint, independent frozen oracle, and own-process memory access"]
fn preselected_larger_tiled_fp32_matches_independent_forward() {
    check_forward(ProjectionKernel::Tiled);
}

fn check_forward(kernel: ProjectionKernel) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let full = root.join("artifacts/m5-larger-oracle-v1");
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
        fs::File::open(root.join("models/Qwen2.5-1.5B-Instruct/model.safetensors")).unwrap();
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
        &root.join("models/Qwen2.5-1.5B-Instruct"),
        LoadOptions::default(),
    )
    .unwrap();
    let mut model = loaded.model;
    assert_eq!(loaded.report.fp32_weights_bytes, 6_174_857_216);
    let mut scratch = model.scratch(4).unwrap();
    // Two full diagnostic traces + largest external arrays/JSON fit inside the
    // explicit128MiB test/framework allowance; no second weight model is held.
    assert!(
        loaded.report.fp32_weights_bytes + scratch.allocated_bytes() + 128 * 1024 * 1024
            <= weights::INITIAL_MEMORY_LIMIT_BYTES
    );
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
        let scalar = if kernel == ProjectionKernel::Tiled {
            model = model.with_projection_kernel(ProjectionKernel::Scalar);
            Some(
                model
                    .forward_diagnostic(&tokens, offset, &mut scratch)
                    .unwrap(),
            )
        } else {
            None
        };
        model = model.with_projection_kernel(kernel);
        let trace = model
            .forward_diagnostic(&tokens, offset, &mut scratch)
            .unwrap();
        let bitwise_arrays = scalar
            .as_ref()
            .map(|scalar| trace_parity::assert_bits(&trace, scalar))
            .unwrap_or(0);
        drop(scalar);
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
        let width = model.config().vocab_size;
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
        let mut cache_rows = 0;
        let mut cache_admission = 0;
        for chunk in [1, 3, 8] {
            let mut session = Session::new(
                &model,
                Options {
                    capacity: 1025,
                    prefill_chunk_size: chunk,
                    position_offset: offset,
                    memory_limit_bytes: weights::INITIAL_MEMORY_LIMIT_BYTES - 64 * 1024 * 1024,
                },
            )
            .unwrap();
            cache_admission =
                cache_admission.max(session.memory().admitted_bytes + 64 * 1024 * 1024);
            // One trace, bounded external output, existing scratch, oracle
            // arrays and JSON remain in the separately admitted64MiB allowance.
            let mut output = vec![0.; chunk * width];
            for part in tokens.chunks(chunk) {
                let start = session.evaluated_len();
                session.append(part).unwrap();
                let output = &mut output[..part.len() * width];
                assert_eq!(
                    session.evaluate_next_chunk_with_logits(output).unwrap(),
                    part.len()
                );
                assert_eq!(
                    output,
                    &trace.logits[start * width..(start + part.len()) * width]
                );
                cache_rows += part.len();
            }
        }
        let mut last = vec![0.0; width];
        model
            .forward(&tokens, offset, &mut scratch, &mut last)
            .unwrap();
        assert_eq!(last, trace.logits[(tokens.len() - 1) * width..]);
        eprintln!(
            "{case}: full forward probes, every-position argmax and final-only path match; short-target NLL deltas {losses:?}"
        );
        results.insert(
            case.clone(),
            json!({"arrays": metrics, "argmax": argmax, "short_teacher_forced_nll_deltas": losses,
                "scalar_bitwise_trace_arrays":bitwise_arrays,"all_row_cache_comparisons":cache_rows,
                "cache_admission_including_application_bytes":cache_admission}),
        );
    }
    let report = json!({"model": manifest["model"], "policy": manifest["policy"], "results": results,
        "cpu_projection":kernel.name(), "passed":failures.is_empty(),
        "physical_process_peak_bytes":physical_peak(), "quality_scope":"short independent larger numerical probes; no full held-out loss/task, GPU or performance qualification"});
    let output = std::env::var_os("TOKLET_LARGER_FORWARD_REPORT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            root.join(if kernel == ProjectionKernel::Scalar {
                "artifacts/m5-larger-parity-scalar-v1.json"
            } else {
                "artifacts/m5-larger-parity-tiled-v1.json"
            })
        });
    let bytes = serde_json::to_vec_pretty(&report).unwrap();
    if kernel == ProjectionKernel::Tiled
        || std::env::var_os("TOKLET_LARGER_FORWARD_REPORT").is_some()
    {
        use std::io::Write;
        fs::create_dir_all(output.parent().unwrap()).unwrap();
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&output)
            .unwrap()
            .write_all(&bytes)
            .unwrap();
    } else {
        fs::write(&output, bytes).unwrap();
    }
    eprintln!("Full numerical metrics: {}", output.display());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
#[cfg(target_os = "macos")]
fn physical_peak() -> Option<u64> {
    // macOS SDK rusage_info_v4: 16 UUID bytes followed by35 u64 fields.
    // This bounded own-process syscall avoids an external footprint sampler.
    #[repr(C)]
    struct Usage {
        uuid: [u8; 16],
        fields: [u64; 35],
    }
    #[link(name = "proc")]
    unsafe extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, usage: *mut Usage) -> i32;
    }
    assert_eq!(std::mem::size_of::<Usage>(), 296);
    let mut usage = Usage {
        uuid: [0; 16],
        fields: [0; 35],
    };
    // SAFETY: initialized, aligned v4 storage has the SDK's296-byte size;
    // the call writes only this exclusive stack allocation synchronously.
    let status = unsafe { proc_pid_rusage(std::process::id() as i32, 4, &mut usage) };
    assert_eq!(status, 0, "own-process libproc v4 query failed");
    let peak = usage.fields[28]; // ri_lifetime_max_phys_footprint
    assert!(peak <= weights::INITIAL_MEMORY_LIMIT_BYTES as u64);
    eprintln!(
        "Own-process physical session peak: {peak} bytes; includes test/hash/diagnostic setup"
    );
    Some(peak)
}

#[cfg(not(target_os = "macos"))]
fn physical_peak() -> Option<u64> {
    None
}
