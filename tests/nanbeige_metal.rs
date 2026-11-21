#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]

use half::bf16;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{BufWriter, ErrorKind, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};
use toklet::{
    metal::nanbeige::{Model, Session},
    mlx::nanbeige::{HIDDEN, KV_WIDTH, LOGIT_ROWS, VISITS},
    session::{InferenceSession, Options},
};

fn fixture(name: &str) -> PathBuf {
    std::env::var_os(name).map(PathBuf::from).expect(name)
}

fn document(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

fn words(values: &[f32]) -> Vec<u16> {
    values
        .iter()
        .map(|&x| bf16::from_f32(x).to_bits())
        .collect()
}

fn compare(native: &[u16], reference_root: &Path, receipt: &Value) -> Value {
    let name = receipt["path"].as_str().unwrap();
    assert_eq!(Path::new(name).components().count(), 1);
    let reference = fs::read(reference_root.join(name)).unwrap();
    assert_eq!(reference.len(), native.len() * 2);
    let shape: Vec<usize> = serde_json::from_value(receipt["shape"].clone()).unwrap();
    assert_eq!(shape.iter().product::<usize>(), native.len());
    assert_eq!(receipt["bytes"].as_u64().unwrap(), reference.len() as u64);
    assert_eq!(
        format!("{:x}", Sha256::digest(&reference)),
        receipt["sha256"]
    );
    let mut maximum = 0.0_f32;
    let mut squared = 0.0_f64;
    let mut exact = 0;
    let mut native_hash = Sha256::new();
    let mut reference_top = None;
    let mut maximum_value = f32::NEG_INFINITY;
    let mut highest_ulp = 0;
    let order = |x: u16| if x >= 0x8000 { !x } else { x | 0x8000 };
    for (i, (&ours, theirs)) in native.iter().zip(reference.chunks_exact(2)).enumerate() {
        let theirs = u16::from_le_bytes(theirs.try_into().unwrap());
        native_hash.update(ours.to_le_bytes());
        exact += usize::from(ours == theirs);
        highest_ulp = highest_ulp.max(order(ours).abs_diff(order(theirs)));
        let a = bf16::from_bits(ours).to_f32();
        let b = bf16::from_bits(theirs).to_f32();
        assert!(a.is_finite() && b.is_finite());
        if b > maximum_value {
            maximum_value = b;
            reference_top = Some(i);
        }
        let delta = (a - b).abs();
        maximum = maximum.max(delta);
        squared += f64::from(delta).powi(2);
    }
    let native_sha256 = format!("{:x}", native_hash.finalize());
    let mut result = json!({
        "reference": name, "shape": receipt["shape"], "values": native.len(),
        "exact": exact, "max_abs": maximum, "max_bf16_ulp": highest_ulp,
        "rms_abs": (squared / native.len() as f64).sqrt(),
        "reference_argmax": reference_top, "native_sha256": native_sha256,
    });
    if let Some(directory) = std::env::var_os("TOKLET_NANBEIGE_NATIVE_OUTPUT") {
        let output = format!("{name}.{native_sha256}.bf16le");
        let path = Path::new(&directory).join(&output);
        match fs::File::create_new(&path) {
            Ok(file) => {
                let mut file = BufWriter::new(file);
                for &word in native {
                    file.write_all(&word.to_le_bytes()).unwrap();
                }
                file.flush().unwrap();
            }
            Err(error) if error.kind() == ErrorKind::AlreadyExists => {
                let previously_recorded = fs::read(path).unwrap();
                assert_eq!(previously_recorded.len(), native.len() * 2);
                assert_eq!(
                    format!("{:x}", Sha256::digest(previously_recorded)),
                    native_sha256
                );
            }
            Err(error) => panic!("could not record {output}: {error}"),
        }
        result["native_output"] = json!(output);
    }
    result
}

#[test]
#[ignore = "Explicit Apple GPU, pinned checkpoint and independent MLX numerical fixture"]
fn pinned_mlx_partition_comparison() {
    let source = fixture("TOKLET_NANBEIGE_MODEL");
    let reference = fixture("TOKLET_NANBEIGE_REFERENCE");
    let receipt = document(&reference.join("result.json"));
    assert_eq!(receipt["passed"], true);
    let prompt: Vec<u32> = serde_json::from_value(receipt["prompt_token_ids"].clone()).unwrap();
    assert_eq!(prompt.len(), 128);
    let decode = receipt["decode_input_id"].as_u64().unwrap() as u32;
    let options = Options {
        capacity: 129,
        prefill_chunk_size: 128,
        ..Options::default()
    };
    let load = Instant::now();
    let model = Model::load(&source, options, true).unwrap();
    let load_seconds = load.elapsed().as_secs_f64();
    let cancelled = AtomicBool::new(false);
    let mut cases = Vec::new();
    for (name, chunk) in [("t128", 128), ("t64-t64", 64)] {
        let mut session = Session::new(
            &model,
            Options {
                prefill_chunk_size: chunk,
                ..options
            },
            true,
            &cancelled,
        )
        .unwrap();
        let case = &receipt["cases"][name];
        let mut prefill_loop_rows = std::array::from_fn::<_, 2, _>(|_| Vec::new());
        session.append(&prompt).unwrap();
        let clock = Instant::now();
        for _ in prompt.chunks_exact(chunk) {
            assert_eq!(session.evaluate_next_chunk().unwrap(), chunk);
            for (pass, buffer) in prefill_loop_rows.iter_mut().enumerate() {
                let first = buffer.len();
                buffer.resize(first + chunk * HIDDEN, 0);
                session.copy_pass_bf16(pass, &mut buffer[first..]).unwrap();
            }
        }
        let prefill_seconds = clock.elapsed().as_secs_f64();
        assert_eq!(session.evaluated_len(), prompt.len());
        let prefill_logits = words(session.logits().unwrap());
        let prefill_greedy = session.greedy_token().unwrap().unwrap();
        let expected_prefill_greedy = case["prefill_greedy_id"].as_u64().unwrap() as u32;
        assert_eq!(prefill_logits.len(), LOGIT_ROWS);
        assert_eq!(expected_prefill_greedy, 858);
        assert_eq!(prefill_greedy, expected_prefill_greedy);
        let prefill_batched_head = compare(
            &prefill_logits,
            &reference,
            &case["prefill_batched_head_last_logits"],
        );
        assert_eq!(prefill_batched_head["reference_argmax"], 858);
        let prefill: Vec<_> = prefill_loop_rows
            .iter()
            .enumerate()
            .map(|(pass, panel)| compare(panel, &reference, &case["prefill_loops"][pass]))
            .collect();

        session.append(&[decode]).unwrap();
        cancelled.store(true, Ordering::Relaxed);
        assert!(matches!(
            session.evaluate_next_decode_chunk(),
            Err(toklet::session::Error::Model(
                toklet::model::Error::Cancelled
            ))
        ));
        assert_eq!(session.evaluated_len(), prompt.len());
        assert_eq!(session.pending_len(), 1);
        assert_eq!(words(session.logits().unwrap()), prefill_logits);
        cancelled.store(false, Ordering::Relaxed);

        let clock = Instant::now();
        assert_eq!(session.evaluate_next_decode_chunk().unwrap(), 1);
        let decode_seconds = clock.elapsed().as_secs_f64();
        assert_eq!(session.evaluated_len(), prompt.len() + 1);
        assert_eq!(session.pending_len(), 0);
        let decode_greedy = session.greedy_token().unwrap().unwrap();
        let native_decode_logits = words(session.logits().unwrap());
        let decode_logits = compare(&native_decode_logits, &reference, &case["decode_logits"]);
        assert_eq!(decode_logits["reference_argmax"], 461);
        assert_eq!(decode_greedy, 461);
        let full_recompute = compare(
            &native_decode_logits,
            &reference,
            &receipt["cases"]["t129"]["decode_logits"],
        );
        let mut decode_loops = Vec::new();
        for pass in 0..2 {
            let mut output = vec![0; HIDDEN];
            session.copy_pass_bf16(pass, &mut output).unwrap();
            decode_loops.push(compare(&output, &reference, &case["decode_loops"][pass]));
        }

        let mut k = vec![0; (prompt.len() + 1) * KV_WIDTH];
        let mut v = vec![0; k.len()];
        let mut slots = Vec::new();
        let mut distinct = std::collections::BTreeSet::new();
        for slot in 0..VISITS {
            session.copy_kv_bf16(slot, &mut k, &mut v).unwrap();
            let h = |x: &[u16]| {
                let mut sha = Sha256::new();
                for &word in x {
                    sha.update(word.to_le_bytes());
                }
                format!("{:x}", sha.finalize())
            };
            assert!(distinct.insert((h(&k), h(&v))));
            if let Some(ref_slot) = case["decode_state"]["sampled_slots"].get(slot.to_string()) {
                slots.push(json!({"slot": slot,
                    "k": compare(&k, &reference, &ref_slot["k"]),
                    "v": compare(&v, &reference, &ref_slot["v"]),
                }));
            }
        }
        session.reset();
        assert_eq!(session.evaluated_len(), 0);
        assert!(session.history().is_empty() && session.logits().is_none());
        assert!(session.copy_kv_bf16(0, &mut [], &mut []).is_err());
        // Different partition orders keep separate numerical measurements.
        cases.push(json!({
            "partition": name, "chunks": case["chunks"],
            "prefill_passes": prefill,
            "prefill_batched_head": prefill_batched_head,
            "prefill_vector_head": compare(&prefill_logits, &reference, &case["prefill_single_row_head_logits"]),
            "prefill_greedy": prefill_greedy, "mlx_prefill_greedy": expected_prefill_greedy,
            "decode_passes": decode_loops, "decode_logits": decode_logits,
            "independent_t129_recompute": full_recompute,
            "decode_greedy": decode_greedy, "decode_input": decode,
            "decode_slots": slots, "distinct_kv_slots": distinct.len(),
            "cancel_and_retry": true, "reset": true,
            "prefill_seconds_diagnostic": prefill_seconds, "decode_seconds_diagnostic": decode_seconds,
        }));
    }
    let result = json!({
        "scope": "Nanbeige native Metal numerical comparison",
        "device": model.device_name(), "load_seconds": load_seconds,
        "cases": cases,
    });
    println!("{result}");
}
