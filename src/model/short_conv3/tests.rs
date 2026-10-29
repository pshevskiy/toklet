use super::*;
use serde_json::Value;
use sha2::{Digest, Sha256};

fn state(h: &History) -> (Vec<u32>, usize, u64) {
    (
        h.committed().iter().map(|x| x.to_bits()).collect(),
        h.offset(),
        h.generation(),
    )
}
fn execute(h: &mut History, input: &[f32], weight: &[f32], tokens: usize) -> Vec<f32> {
    let mut candidate = vec![f32::NAN; h.staging_elements()];
    let mut work = candidate.clone();
    let mut output = vec![f32::NAN; input.len()];
    let before = state(h);
    let frame = h
        .prepare_chunk(
            input,
            weight,
            &mut candidate,
            &mut work,
            &mut output,
            tokens,
        )
        .unwrap();
    assert_eq!(state(h), before);
    // The prepared proof borrows final candidate only, never shared work.
    work.fill(f32::NAN);
    h.commit(&frame).unwrap();
    output
}

#[test]
fn independent_gate_composition_and_all_chunk_partitions() {
    // Independently derived integer fixture supplied by the parent adviser.
    // u=B*x; convolution has NO SiLU; C is a direct multiplicative gate.
    let b = [1., -1., 2., 2., 3., 1., 4., -2., 5., 3.];
    let x = [2., 3., 3., -1., 4., 4., 5., 2., 6., -2.];
    let c = [1., 2., -1., 1., 2., -2., -2., 3., 3., -1.];
    let u = [2., -3., 6., -2., 12., 4., 20., -4., 30., -6.];
    let weight = [1., 10., 100., 3., -2., 5.];
    let z = [200., -15., 620., -4., 1262., 15., 2126., -34., 3212., -10.];
    let gated = [
        200., -30., -620., -4., 2524., -30., -4252., -102., 9636., 10.,
    ];
    for i in 0..u.len() {
        assert_eq!(b[i] * x[i], u[i]);
        assert_eq!(c[i] * z[i], gated[i]);
    }
    // Every composition of five positive token counts: 16 distinct schedules.
    for cuts in 0_u32..16 {
        let mut history = History::new(2, 5).unwrap();
        let pointer = history.committed().as_ptr();
        let mut start = 0;
        for end in 1..=5 {
            if end < 5 && cuts & (1 << (end - 1)) == 0 {
                continue;
            }
            let out = execute(&mut history, &u[start * 2..end * 2], &weight, end - start);
            assert_eq!(out, z[start * 2..end * 2]);
            let expected = if end == 1 {
                [0., 0., u[0], u[1]]
            } else {
                u[(end - 2) * 2..end * 2].try_into().unwrap()
            };
            assert_eq!(history.committed(), expected);
            assert_eq!(history.offset(), end);
            assert_eq!(history.committed().as_ptr(), pointer);
            start = end;
        }
        assert_eq!(history.committed(), [20., -4., 30., -6.]);
        history.reset().unwrap();
        assert!(history.committed().iter().all(|v| v.to_bits() == 0));
        assert_eq!(history.offset(), 0);
        assert_eq!(history.committed().as_ptr(), pointer);
        assert_eq!(execute(&mut history, &u, &weight, 5), z);
    }
}

fn fixture_values(blob: &[u8], row: &Value, name: &str) -> Vec<f64> {
    let d = &row["arrays"][name];
    let off = d["offset"].as_u64().unwrap() as usize;
    let n = d["bytes"].as_u64().unwrap() as usize;
    let b = &blob[off..off + n];
    assert_eq!(
        format!("{:x}", Sha256::digest(b)),
        d["sha256"].as_str().unwrap()
    );
    match d["dtype"].as_str().unwrap() {
        "<f4" => b
            .chunks_exact(4)
            .map(|x| f64::from(f32::from_le_bytes(x.try_into().unwrap())))
            .collect(),
        "<f8" => b
            .chunks_exact(8)
            .map(|x| f64::from_le_bytes(x.try_into().unwrap()))
            .collect(),
        _ => panic!("unsupported frozen fixture dtype"),
    }
}
#[test]
fn unchanged_independent_convolution_fixture_and_history_every_chunk() {
    let receipt =
        include_bytes!("../../../benchmarks/results/2026-10-24-causal-conv-reference-v1.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(receipt)),
        "945b0fc825adc144a8ab1b8ec74d30693478345c59b4cde30a67edd3ab955359"
    );
    let receipt: Value = serde_json::from_slice(receipt).unwrap();
    let m = include_bytes!("../../../tests/fixtures/causal-conv-v1/manifest.json");
    let binary = include_bytes!("../../../tests/fixtures/causal-conv-v1/arrays.bin");
    for (raw, key) in [
        (m.as_slice(), "fixture_manifest_sha256"),
        (binary.as_slice(), "fixture_binary_sha256"),
    ] {
        assert_eq!(
            format!("{:x}", Sha256::digest(raw)),
            receipt[key].as_str().unwrap()
        );
    }
    let manifest: Value = serde_json::from_slice(m).unwrap();
    let row = manifest["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["name"] == "channel-tail")
        .unwrap();
    assert_eq!(
        (
            row["channels"].as_u64(),
            row["kernel"].as_u64(),
            row["tokens"].as_u64()
        ),
        (Some(5), Some(3), Some(6))
    );
    let values = |name| fixture_values(binary, row, name);
    let f32s = |name| values(name).iter().map(|&x| x as f32).collect::<Vec<_>>();
    let input = f32s("input");
    let weight = f32s("weight");
    let initial = f32s("initial_history");
    let expected = values("output");
    let histories = values("history");
    for size in [1, 2, 3, 4, 6] {
        let mut h = History::new(5, 8).unwrap();
        execute(&mut h, &initial, &weight, 2); // two ordinary inputs recreate frozen initial history
        assert_eq!(h.committed(), initial);
        let mut start = 0;
        while start < 6 {
            let end = (start + size).min(6);
            let output = execute(&mut h, &input[start * 5..end * 5], &weight, end - start);
            for (&a, &e) in output.iter().zip(&expected[start * 5..end * 5]) {
                assert!(a.is_finite() && (f64::from(a) - e).abs() <= 1e-5 + 1e-4 * e.abs());
            }
            assert_eq!(
                h.committed()
                    .iter()
                    .map(|&x| f64::from(x))
                    .collect::<Vec<_>>(),
                histories[(end - 1) * 10..end * 10]
            );
            start = end;
        }
        assert_eq!(h.offset(), 8);
    }
}

#[test]
fn whole_chunk_prevalidation_preserves_all_scratch() {
    let h = History::new(2, 3).unwrap();
    let input = [1.; 6];
    let weights = [1.; 6];
    for bad in 0..5 {
        let mut candidate = [7.; 4];
        let mut work = [8.; 4];
        let mut output = [9.; 6];
        assert!(
            h.prepare_chunk(
                &input[..if bad == 0 { 5 } else { 6 }],
                &weights[..if bad == 1 { 5 } else { 6 }],
                &mut candidate[..if bad == 2 { 3 } else { 4 }],
                &mut work[..if bad == 3 { 3 } else { 4 }],
                &mut output[..if bad == 4 { 5 } else { 6 }],
                3
            )
            .is_err()
        );
        assert_eq!(candidate, [7.; 4]);
        assert_eq!(work, [8.; 4]);
        assert_eq!(output, [9.; 6]);
    }
    for which in 0..2 {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut input = input;
            let mut weights = weights;
            if which == 0 {
                input[5] = value;
            } else {
                weights[5] = value;
            }
            let mut candidate = [7.; 4];
            let mut work = [8.; 4];
            let mut output = [9.; 6];
            assert!(matches!(
                h.prepare_chunk(&input, &weights, &mut candidate, &mut work, &mut output, 3),
                Err(Error::Kernel(cpu::Error::NonFinite { .. }))
            ));
            assert_eq!(candidate, [7.; 4]);
            assert_eq!(work, [8.; 4]);
            assert_eq!(output, [9.; 6]);
        }
    }
    for tokens in [0, 4, usize::MAX] {
        let mut candidate = [7.; 4];
        let mut work = [8.; 4];
        let mut output = [9.; 6];
        assert!(
            h.prepare_chunk(
                &input,
                &weights,
                &mut candidate,
                &mut work,
                &mut output,
                tokens
            )
            .is_err()
        );
        assert_eq!(candidate, [7.; 4]);
        assert_eq!(work, [8.; 4]);
        assert_eq!(output, [9.; 6]);
    }
    assert_eq!(h.committed(), [0.; 4]);
    assert_eq!(h.offset(), 0);
    assert_eq!(h.generation(), 0);
}

#[test]
fn late_cast_failure_discard_retry_and_reset_replay() {
    let mut h = History::new(1, 8).unwrap();
    execute(&mut h, &[2., 3.], &[1., 1., 1.], 2);
    let before = state(&h);
    let mut candidate = [7.; 2];
    let mut work = [8.; 2];
    let mut output = [9.; 2];
    assert!(matches!(
        h.prepare_chunk(
            &[1., f32::MAX],
            &[0., 0., 2.],
            &mut candidate,
            &mut work,
            &mut output,
            2
        ),
        Err(Error::Kernel(cpu::Error::NonFinite { index: 0, .. }))
    ));
    assert_eq!(state(&h), before);
    assert_eq!(output[0], 2.);
    {
        let _frame = h
            .prepare_chunk(
                &[1., 4.],
                &[0., 0., 2.],
                &mut candidate,
                &mut work,
                &mut output,
                2,
            )
            .unwrap();
    }
    assert_eq!(state(&h), before); // cancellation/drop of a whole successful candidate
    assert_eq!(execute(&mut h, &[1., 4.], &[0., 0., 2.], 2), [2., 8.]);
    assert_eq!(h.committed(), [1., 4.]);
    // An earlier token prefix cannot be recovered by changing an offset. Reset
    // and replay its inputs, instead of retaining the suffix history above.
    h.reset().unwrap();
    execute(&mut h, &[2., 3.], &[1., 1., 1.], 2);
    assert_eq!(h.committed(), [2., 3.]);
    assert_eq!(h.offset(), 2);
}

#[test]
fn batch_foreign_stale_missing_duplicate_and_mixed_steps_are_atomic() {
    let mut hs = [History::new(1, 10).unwrap(), History::new(1, 10).unwrap()];
    let mut a = [0.; 2];
    let mut b = [0.; 2];
    let mut work = [0.; 2];
    let mut out = [0.; 2];
    let w = [1.; 3];
    let first = hs[0]
        .prepare_chunk(&[1.], &w, &mut a, &mut work, &mut out[..1], 1)
        .unwrap();
    let duplicate = hs[0]
        .prepare_chunk(&[2.], &w, &mut b, &mut work, &mut out[..1], 1)
        .unwrap();
    let before = hs.iter().map(state).collect::<Vec<_>>();
    assert!(commit_batch(&mut hs, &[first, duplicate]).is_err());
    assert_eq!(hs.iter().map(state).collect::<Vec<_>>(), before);
    let first = hs[0]
        .prepare_chunk(&[1.], &w, &mut a, &mut work, &mut out[..1], 1)
        .unwrap();
    let second = hs[1]
        .prepare_chunk(&[2., 3.], &w, &mut b, &mut work, &mut out, 2)
        .unwrap();
    assert!(commit_batch(&mut hs, &[first, second]).is_err());
    assert_eq!(hs.iter().map(state).collect::<Vec<_>>(), before);
    let first = hs[0]
        .prepare_chunk(&[1.], &w, &mut a, &mut work, &mut out[..1], 1)
        .unwrap();
    assert!(commit_batch(&mut hs, std::slice::from_ref(&first)).is_err());
    assert!(hs[1].commit(&first).is_err()); // identical geometry is not identity
    hs[0].commit(&first).unwrap();
    let after = state(&hs[0]);
    assert!(hs[0].commit(&first).is_err());
    assert_eq!(state(&hs[0]), after);
    let first = hs[0]
        .prepare_chunk(&[1.], &w, &mut a, &mut work, &mut out[..1], 1)
        .unwrap();
    let second = hs[1]
        .prepare_chunk(&[2.], &w, &mut b, &mut work, &mut out[..1], 1)
        .unwrap();
    let before = hs.iter().map(state).collect::<Vec<_>>();
    assert!(commit_batch(&mut hs, &[first, second]).is_err()); // different committed offsets
    assert_eq!(hs.iter().map(state).collect::<Vec<_>>(), before);
    reset_batch(&mut hs).unwrap();
    let first = hs[0]
        .prepare_chunk(&[1.], &w, &mut a, &mut work, &mut out[..1], 1)
        .unwrap();
    let second = hs[1]
        .prepare_chunk(&[2.], &w, &mut b, &mut work, &mut out[..1], 1)
        .unwrap();
    commit_batch(&mut hs, &[first, second]).unwrap();
    assert_eq!(hs[0].committed(), [0., 1.]);
    assert_eq!(hs[1].committed(), [0., 2.]);
    let frame = hs[0]
        .prepare_chunk(&[3.], &w, &mut a, &mut work, &mut out[..1], 1)
        .unwrap();
    hs[0].reset().unwrap();
    let before = state(&hs[0]);
    assert!(hs[0].commit(&frame).is_err());
    assert_eq!(state(&hs[0]), before);
    commit_batch(&mut [], &[]).unwrap();
    reset_batch(&mut []).unwrap();
}

#[test]
fn checked_accounting_position_and_reset_generation_exhaustion() {
    for (channels, positions) in [(0, 1), (1, 0), (usize::MAX, 1)] {
        assert!(History::planned_bytes(channels, positions).is_err());
        assert!(History::new(channels, positions).is_err());
    }
    let mut hs = [
        History::new(1024, 512).unwrap(),
        History::new(1024, 512).unwrap(),
    ];
    assert_eq!(
        hs[0].allocated_bytes().unwrap(),
        History::planned_bytes(1024, 512).unwrap()
    );
    assert_eq!(hs[0].staging_elements(), 2048);
    assert_eq!(
        hs[0].allocated_bytes().unwrap(),
        8192 + std::mem::size_of::<History>()
    );
    hs[0].committed[0] = -0.;
    hs[1].committed[0] = 3.;
    hs[1].generation = u64::MAX;
    let before = hs.iter().map(state).collect::<Vec<_>>();
    assert!(reset_batch(&mut hs).is_err());
    assert_eq!(hs.iter().map(state).collect::<Vec<_>>(), before);
    assert!(hs[1].admit(1).is_err());
    hs[1].generation = 0;
    reset_batch(&mut hs).unwrap();
    assert!(
        hs.iter()
            .flat_map(|h| h.committed())
            .all(|v| v.to_bits() == 0)
    );
    hs[0].offset = usize::MAX;
    assert!(matches!(
        hs[0].admit(1),
        Err(Error::Kernel(cpu::Error::SizeOverflow))
    ));
    hs[0].offset = 512;
    assert!(matches!(
        hs[0].admit(1),
        Err(Error::Capacity {
            offset: 512,
            tokens: 1,
            limit: 512
        })
    ));
}
