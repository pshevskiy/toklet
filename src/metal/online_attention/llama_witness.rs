//! Reproduce the frozen Llama Online failure and replay identical attention inputs.
use super::*;
use half::f16;
use objc2::runtime::ProtocolObject;
use sha2::{Digest, Sha256};

#[test]
#[ignore = "Explicit Mac Metal witness for the frozen tiny Llama Online rejection"]
fn llama_online_same_input_witness() {
    let directory =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mlx-llama-v1");
    let manifest = std::fs::read(directory.join("manifest.json")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&manifest)),
        "824d2ee2ae9dd5b623c5aa9db10c13b2e4dee32647ea5fa9d78274fb3529a1f5"
    );
    let manifest: serde_json::Value = serde_json::from_slice(&manifest).unwrap();
    for (name, entry) in manifest["files"].as_object().unwrap() {
        let bytes = std::fs::read(directory.join(name)).unwrap();
        assert_eq!(entry["bytes"], bytes.len());
        assert_eq!(entry["sha256"], format!("{:x}", Sha256::digest(&bytes)));
    }
    let case = &manifest["variants"]["tied"]["cases"]["offset"];
    assert_eq!(case["tokens"], serde_json::json!([7, 13, 3]));
    assert_eq!(case["position_offset"], 5);
    let values = &manifest["arrays"]["tied.half.cases.offset.layers.1.attention_projection"];
    assert_eq!(values["dtype"], "<f2");
    let start = values["offset"].as_u64().unwrap() as usize + 68 * 2;
    let arrays = std::fs::read(directory.join("arrays.bin")).unwrap();
    let frozen = f16::from_le_bytes(arrays[start..start + 2].try_into().unwrap()).to_f32();
    let budget = 256 * 1024 * 1024;
    let model = Model::load_mlx(
        &directory.join("tied"),
        weights::LoadOptions {
            context_limit: 32,
            memory_limit_bytes: budget,
        },
    )
    .unwrap();
    let options = session::Options {
        capacity: 27,
        position_offset: 5,
        prefill_chunk_size: 3,
        memory_limit_bytes: budget,
    };
    let open = |attention| {
        Session::with_attention_mode(
            &model,
            options,
            ProjectionMode::MlxAffine,
            NormalizationMode::Compensated,
            ReadbackMode::FullLogits,
            attention,
        )
    };
    assert!(open(AttentionMode::Online).is_err());
    let mut reference = open(AttentionMode::Reference).unwrap();
    reference.enable_diagnostics(&[1]).unwrap();
    reference.append(&[7, 13, 3]).unwrap();
    reference.evaluate().unwrap();
    let ordinary = reference
        .diagnostics()
        .unwrap()
        .into_iter()
        .find(|row| row.layer == Some(1) && row.stage == "attention_projection")
        .unwrap()
        .values[68];
    let atol = values["gate"]["atol"].as_f64().unwrap();
    let rtol = values["gate"]["rtol"].as_f64().unwrap();
    let gate = atol + rtol * f64::from(frozen).abs();
    assert!(f64::from(ordinary - frozen).abs() <= gate);
    drop(reference);
    let mut s = open(AttentionMode::Reference).unwrap();
    s.enable_diagnostics(&[1]).unwrap();
    // Test-only override. The admitted Reference layout contains all buffers
    // required by Online. Keep the public Llama rejection in force.
    s.attention_mode = AttentionMode::Online;
    s.append(&[7, 13, 3]).unwrap();
    s.evaluate().unwrap();
    let snapshots = s.diagnostics().unwrap();
    let get = |stage| {
        &snapshots
            .iter()
            .find(|row| row.layer == Some(1) && row.stage == stage)
            .unwrap()
            .values
    };
    let rejected = get("attention_projection")[68];
    let q_words: Vec<_> = get("q_rotary")
        .iter()
        .map(|&x| f16::from_f32(x).to_bits())
        .collect();
    let g = &model.gpu;
    let mut q = Buffer::<u16>::new(g, q_words.len() + 7).unwrap();
    q.with_mut(|x| {
        x.fill(f16::NAN.to_bits());
        x[..q_words.len()].copy_from_slice(&q_words);
    });
    let capacity = s.capacity();
    let cache = &mut s.cache[1];
    let shape = [3, 4, 2, 8, capacity as u32, 3, 0, 3];
    let scratch = &mut s.scratch;
    let mut records = Vec::new();
    for poison in [f16::NAN.to_bits(), f16::MAX.to_bits()] {
        for buffer in [&mut cache.k, &mut cache.v] {
            buffer.with_mut(|data| {
                for head in 0..2 {
                    data[(head * capacity + 3) * 8..(head + 1) * capacity * 8].fill(poison);
                }
                data[2 * capacity * 8..].fill(poison);
            });
        }
        let mut contexts = Vec::new();
        let mut projections = Vec::new();
        for kernel in [
            Kernel::Attention,
            Kernel::AttentionPrefill,
            Kernel::AttentionDecode,
        ] {
            scratch.context.with_mut(|b| b.fill(0x3555));
            scratch.status.with_mut(|b| b.fill(0));
            let raw = g.command().unwrap();
            let command = Command {
                raw: &raw,
                trace: None,
            };
            let mut buffers: Vec<&ProtocolObject<dyn MTLBuffer>> =
                vec![&q.raw, &cache.k.raw, &cache.v.raw];
            if matches!(kernel, Kernel::Attention) {
                buffers.push(&scratch.scores.raw);
                buffers.push(&scratch.probs.raw);
            }
            buffers.push(&scratch.context.raw);
            let work = if matches!(kernel, Kernel::AttentionPrefill) {
                8
            } else {
                12
            };
            g.encode(
                &command,
                kernel,
                &buffers,
                &shape,
                work,
                Some(&scratch.status),
            )
            .unwrap();
            mlx::product(
                &command,
                &model.layers[1].o,
                &scratch.zero,
                &s.views[2].context,
                &scratch.delta.raw,
                s.h.stride,
                3,
                true,
                &scratch.status,
            )
            .unwrap();
            raw.commit();
            raw.waitUntilCompleted();
            assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
            assert_eq!(scratch.status.with_slice(|b| b[0]), 0);
            contexts.push(scratch.context.with_slice(|b| {
                assert!(b[96..].iter().all(|&x| x == 0x3555));
                b[..96].to_vec()
            }));
            projections.push(scratch.delta.with_slice(|b| f16::from_bits(b[68]).to_f32()));
        }
        records.push((contexts, projections));
    }
    assert_eq!(records[0], records[1]);
    let (contexts, projections) = &records[0];
    let captured_context: Vec<_> = get("attention_context")
        .iter()
        .map(|&value| f16::from_f32(value).to_bits())
        .collect();
    assert_eq!(contexts[1], captured_context);
    assert_eq!(projections[1].to_bits(), rejected.to_bits());
    let differences: Vec<_> = contexts[0]
        .iter()
        .zip(&contexts[1])
        .enumerate()
        .filter_map(|(i, (&a, &b))| {
            (a != b).then_some((i, f16::from_bits(a).to_f32(), f16::from_bits(b).to_f32()))
        })
        .collect();
    let k = cache.k.with_slice(|b| b.to_vec());
    let v = cache.v.with_slice(|b| b.to_vec());
    let half = |bits| f64::from(f16::from_bits(bits).to_f32());
    let mut maximum_error = [0.0f64; 3];
    for row in 0..3 {
        for head in 0..4 {
            let kv = head / 2;
            let q = &q_words[(row * 4 + head) * 8..][..8];
            let scores: Vec<_> = (0..=row)
                .map(|position| {
                    let k = &k[(kv * capacity + position) * 8..][..8];
                    q.iter()
                        .zip(k)
                        .map(|(&a, &b)| half(a) * half(b))
                        .sum::<f64>()
                        / 8f64.sqrt()
                })
                .collect();
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mass: Vec<_> = scores.iter().map(|score| (score - maximum).exp()).collect();
            let total: f64 = mass.iter().sum();
            for channel in 0..8 {
                let expected = mass
                    .iter()
                    .enumerate()
                    .map(|(position, weight)| {
                        weight * half(v[(kv * capacity + position) * 8 + channel])
                    })
                    .sum::<f64>()
                    / total;
                for (mode, actual) in contexts.iter().enumerate() {
                    let actual = half(actual[(row * 4 + head) * 8 + channel]);
                    assert!(actual.is_finite());
                    maximum_error[mode] = maximum_error[mode].max((actual - expected).abs());
                }
            }
        }
    }
    println!(
        "LLAMA_ONLINE {}",
        serde_json::json!({"projection":projections,"differences":differences,
                 "online_prefill_single_equal":contexts[1]==contexts[2],
                 "fp64_maximum_error":maximum_error,"poison_invariant":true,"frozen_reference_projection":frozen,
                 "ordinary_reference_projection":ordinary,"frozen_atol":atol,"frozen_rtol":rtol,
                 "frozen_online_gate_passed":f64::from(rejected-frozen).abs()<=gate})
    );
}
