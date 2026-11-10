use super::*;
use crate::cpu;
use serde_json::Value;
use sha2::{Digest, Sha256};

fn cases() -> Vec<Value> {
    let raw = include_bytes!("../../../tests/fixtures/short-conv7-v1.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        "e1dd5d86bc81b25303d6c8de93f1553897e191948c215e291c996ba5aac9c7be"
    );
    let reference =
        include_bytes!("../../../benchmarks/results/2026-11-10-short-conv7-reference-v1.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(reference)),
        "48ec96749d239ceb9e2455eefb16b51893efc31a652a1d03b8e2333e98478b3d"
    );
    let receipt: Value = serde_json::from_slice(reference).unwrap();
    assert_eq!(receipt["failures"], 0);
    assert_eq!(receipt["fp32_bit_differences"], 0);
    assert_eq!(receipt["raw_history_matches"], true);
    let fixture: Value = serde_json::from_slice(raw).unwrap();
    assert_eq!(fixture["format"], "toklet-short-conv7-exact-v1");
    assert_eq!(
        fixture["source_coefficients"]["channels"],
        serde_json::json!([0, 1, 1023, 2047])
    );
    assert_eq!(
        fixture["source_coefficients"]["object_sha256"],
        "b6ff17ac993dddf0f6f51bd872b12ae1437f4453a54a0b4878cedc7864fbc07e"
    );
    fixture["cases"].as_array().unwrap().clone()
}
fn values(row: &Value, name: &str) -> Vec<f32> {
    row[name]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| f32::from_bits(u32::try_from(v.as_u64().unwrap()).unwrap()))
        .collect()
}
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|x| x.to_bits()).collect()
}
fn state(h: &History) -> (Vec<u32>, usize, u64) {
    (bits(h.committed()), h.offset(), h.generation())
}
fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &e) in actual.iter().zip(expected) {
        assert!(
            a.is_finite()
                && e.is_finite()
                && (f64::from(a) - f64::from(e)).abs() <= 1e-5 + 1e-4 * f64::from(e).abs(),
            "{a} differs from {e}"
        );
    }
}
fn partition(row: &Value, chunks: &[usize]) {
    let c = row["channels"].as_u64().unwrap() as usize;
    let t = row["tokens"].as_u64().unwrap() as usize;
    let input = values(row, "input_bits");
    let weight = values(row, "weight_bits");
    let initial = values(row, "initial_history_bits");
    let expected = values(row, "output_bits");
    let histories = values(row, "history_after_each_bits");
    let final_history = values(row, "final_history_bits");
    let mut h = History::new(c, t + 6).unwrap();
    let pointer = h.committed().as_ptr();
    let bytes = h.allocated_bytes().unwrap();
    let mut candidate = vec![f32::NAN; 6 * c];
    let mut work = vec![f32::NAN; 6 * c];
    let mut seed_output = vec![0.; 6 * c];
    let frame = h
        .prepare_chunk(
            &initial,
            &weight,
            &mut candidate,
            &mut work,
            &mut seed_output,
            6,
        )
        .unwrap();
    h.commit(&frame).unwrap();
    assert_eq!(bits(h.committed()), bits(&initial));
    let mut output = vec![f32::NAN; t * c];
    let mut offset = 0;
    for &n in chunks {
        let before = state(&h);
        let range = offset * c..(offset + n) * c;
        let frame = h
            .prepare_chunk(
                &input[range.clone()],
                &weight,
                &mut candidate,
                &mut work,
                &mut output[range.clone()],
                n,
            )
            .unwrap();
        assert_eq!(state(&h), before);
        h.commit(&frame).unwrap();
        close(&output[range.clone()], &expected[range]);
        offset += n;
        assert_eq!(
            bits(h.committed()),
            bits(&histories[(offset - 1) * 6 * c..offset * 6 * c])
        );
    }
    assert_eq!(offset, t);
    assert_eq!(h.offset(), t + 6);
    assert_eq!(bits(h.committed()), bits(&final_history));
    close(&output, &expected);
    h.reset().unwrap();
    assert_eq!(h.offset(), 0);
    assert!(h.committed().iter().all(|x| x.to_bits() == 0));
    assert_eq!(h.committed().as_ptr(), pointer);
    assert_eq!(h.allocated_bytes().unwrap(), bytes);
}
#[test]
fn exact_rational_fixtures_and_every_authentic_coefficient_chunk_partition() {
    let mut schedules = 0;
    for row in cases() {
        let t = row["tokens"].as_u64().unwrap() as usize;
        if row["all_chunk_partitions"] == true {
            for mask in 0..1_usize << (t - 1) {
                let mut chunks = Vec::new();
                let mut n = 1;
                for boundary in 0..t - 1 {
                    if mask & (1 << boundary) != 0 {
                        chunks.push(n);
                        n = 1;
                    } else {
                        n += 1;
                    }
                }
                chunks.push(n);
                partition(&row, &chunks);
                schedules += 1;
            }
        } else {
            for size in [1, 2, 3, 5, 6, 7, 8, 17] {
                let mut left = t;
                let mut chunks = Vec::new();
                while left > 0 {
                    let n = left.min(size);
                    chunks.push(n);
                    left -= n;
                }
                partition(&row, &chunks);
                schedules += 1;
            }
        }
    }
    assert_eq!(schedules, 288);
    println!("SHORT_CONV7_FIXTURES cases=5 partitions={schedules} exact_history=true");
}
#[test]
fn scalar_leaf_each_step_matches_independent_seven_tap_output_and_raw_history() {
    for row in cases() {
        let c = row["channels"].as_u64().unwrap() as usize;
        let t = row["tokens"].as_u64().unwrap() as usize;
        let input = values(&row, "input_bits");
        let weight = values(&row, "weight_bits");
        let mut history = values(&row, "initial_history_bits");
        let expected = values(&row, "output_bits");
        let histories = values(&row, "history_after_each_bits");
        let mut candidate = vec![0.; 6 * c];
        let mut output = vec![0.; c];
        for token in 0..t {
            cpu::causal_depthwise_conv_step(
                &input[token * c..(token + 1) * c],
                &weight,
                &history,
                &mut candidate,
                &mut output,
                c,
                7,
            )
            .unwrap();
            close(&output, &expected[token * c..(token + 1) * c]);
            assert_eq!(
                bits(&candidate),
                bits(&histories[token * 6 * c..(token + 1) * 6 * c])
            );
            std::mem::swap(&mut history, &mut candidate);
        }
        assert_eq!(bits(&history), bits(&values(&row, "final_history_bits")));
    }
}
#[test]
fn whole_chunk_rejections_preserve_all_scratch_and_committed_state() {
    let h = History::new(2, 4).unwrap();
    for bad in 0..9 {
        let mut input = [1.; 8];
        let mut weights = [0.5; 14];
        let mut candidate = [37.; 12];
        let mut work = [38.; 12];
        let mut output = [39.; 8];
        if bad == 7 {
            input[7] = f32::NAN;
        }
        if bad == 8 {
            weights[13] = f32::INFINITY;
        }
        let before = state(&h);
        let result = h.prepare_chunk(
            &input[..if bad == 0 { 7 } else { 8 }],
            &weights[..if bad == 1 { 13 } else { 14 }],
            &mut candidate[..if bad == 2 { 11 } else { 12 }],
            &mut work[..if bad == 3 { 11 } else { 12 }],
            &mut output[..if bad == 4 { 7 } else { 8 }],
            if bad == 5 {
                0
            } else if bad == 6 {
                5
            } else {
                4
            },
        );
        assert!(result.is_err());
        assert_eq!(state(&h), before);
        assert_eq!(candidate, [37.; 12]);
        assert_eq!(work, [38.; 12]);
        assert_eq!(output, [39.; 8]);
    }
}
#[test]
fn late_overflow_discards_then_retry_reset_and_replay_preserve_valid_prefix() {
    let mut h = History::new(1, 3).unwrap();
    let w = [1., 0., 0., 0., 0., 0., 2.];
    let mut candidate = [0.; 6];
    let mut work = [0.; 6];
    let mut out = [0.; 2];
    let frame = h
        .prepare_chunk(&[0.25], &w, &mut candidate, &mut work, &mut out[..1], 1)
        .unwrap();
    h.commit(&frame).unwrap();
    let before = state(&h);
    assert!(matches!(
        h.prepare_chunk(&[1., f32::MAX], &w, &mut candidate, &mut work, &mut out, 2),
        Err(Error::Kernel(cpu::Error::NonFinite { .. }))
    ));
    assert_eq!(state(&h), before);
    let frame = h
        .prepare_chunk(&[1., 2.], &w, &mut candidate, &mut work, &mut out, 2)
        .unwrap();
    h.commit(&frame).unwrap();
    let final_history = bits(h.committed());
    let expected = out;
    assert_eq!(h.offset(), 3);
    assert!(
        h.prepare_chunk(&[1.], &w, &mut candidate, &mut work, &mut out[..1], 1)
            .is_err()
    );
    h.reset().unwrap();
    let frame = h
        .prepare_chunk(&[0.25], &w, &mut candidate, &mut work, &mut out[..1], 1)
        .unwrap();
    h.commit(&frame).unwrap();
    let frame = h
        .prepare_chunk(&[1., 2.], &w, &mut candidate, &mut work, &mut out, 2)
        .unwrap();
    h.commit(&frame).unwrap();
    assert_eq!(out, expected);
    assert_eq!(bits(h.committed()), final_history);
}
#[test]
fn mixed_stale_foreign_and_cross_profile_frames_never_publish_partial_batches() {
    let mut hs = [History::new(1, 4).unwrap(), History::new(1, 4).unwrap()];
    let mut a = [0.; 6];
    let mut b = [0.; 6];
    let mut work = [0.; 6];
    let mut out = [0.; 2];
    let w = [1.; 7];
    let f0 = hs[0]
        .prepare_chunk(&[1.], &w, &mut a, &mut work, &mut out[..1], 1)
        .unwrap();
    let f1 = hs[1]
        .prepare_chunk(&[2., 3.], &w, &mut b, &mut work, &mut out, 2)
        .unwrap();
    let before = hs.iter().map(state).collect::<Vec<_>>();
    assert!(commit_batch(&mut hs, &[f0, f1]).is_err());
    assert_eq!(hs.iter().map(state).collect::<Vec<_>>(), before);
    let first = hs[0]
        .prepare_chunk(&[1.], &w, &mut a, &mut work, &mut out[..1], 1)
        .unwrap();
    let duplicate = hs[0]
        .prepare_chunk(&[2.], &w, &mut b, &mut work, &mut out[..1], 1)
        .unwrap();
    assert!(commit_batch(&mut hs, &[first, duplicate]).is_err());
    assert_eq!(hs.iter().map(state).collect::<Vec<_>>(), before);
    let frame = hs[0]
        .prepare_chunk(&[1.], &w, &mut a, &mut work, &mut out[..1], 1)
        .unwrap();
    assert!(commit_batch(&mut hs, std::slice::from_ref(&frame)).is_err());
    assert_eq!(hs.iter().map(state).collect::<Vec<_>>(), before);
    hs[0].reset().unwrap();
    assert!(hs[0].commit(&frame).is_err());
    let mut three = crate::model::short_conv3::History::new(3, 4).unwrap();
    let mut work3 = [0.; 6];
    let mut out3 = [0.; 3];
    let frame = three
        .prepare_chunk(&[1.; 3], &[1.; 9], &mut b, &mut work3, &mut out3, 1)
        .unwrap();
    assert!(hs[1].commit(&frame).is_err());
    assert_eq!(hs[1].offset(), 0);
    let frame = hs[1]
        .prepare_chunk(&[1.], &w, &mut a, &mut work, &mut out[..1], 1)
        .unwrap();
    assert!(three.commit(&frame).is_err());
    assert_eq!(three.offset(), 0);
    commit_batch(&mut [], &[]).unwrap();
    reset_batch(&mut []).unwrap();
}
#[test]
fn bounded_specializations_checked_accounting_and_generation_exhaustion() {
    assert!(crate::model::short_conv::History::<0>::new(1, 1).is_err());
    assert!(crate::model::short_conv::History::<1>::new(1, 1).is_err());
    assert!(crate::model::short_conv::History::<4>::new(1, 1).is_err());
    assert!(crate::model::short_conv::History::<8>::new(1, 1).is_err());
    for (c, p) in [(0, 1), (1, 0), (usize::MAX, 1)] {
        assert!(History::planned_bytes(c, p).is_err());
        assert!(History::new(c, p).is_err());
    }
    let mut hs = [
        History::new(2048, 1025).unwrap(),
        History::new(2048, 1025).unwrap(),
    ];
    assert_eq!(
        hs[0].allocated_bytes().unwrap(),
        6 * 2048 * 4 + std::mem::size_of::<History>()
    );
    assert_eq!(
        hs[0].allocated_bytes().unwrap(),
        History::planned_bytes(2048, 1025).unwrap()
    );
    assert_eq!(hs[0].staging_elements(), 6 * 2048);
    hs[1].set_generation_for_test(u64::MAX);
    let before = hs.iter().map(state).collect::<Vec<_>>();
    assert!(reset_batch(&mut hs).is_err());
    assert_eq!(hs.iter().map(state).collect::<Vec<_>>(), before);
}
