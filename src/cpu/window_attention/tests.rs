use super::*;
use serde_json::Value;
use sha2::{Digest, Sha256};

fn shape() -> WindowAttentionShape {
    WindowAttentionShape {
        queries: 1,
        keys: 2,
        capacity: 3,
        query_heads: 2,
        kv_heads: 1,
        head_dim: 2,
        query_start: 1,
        key_start: 0,
        window: Some(3),
        scale: 1.,
    }
}

fn close(actual: &[f32], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &e) in actual.iter().zip(expected) {
        assert!(
            a.is_finite() && (f64::from(a) - e).abs() <= 1e-5 + 1e-4 * e.abs(),
            "{a} differs from {e}"
        );
    }
}

#[test]
fn frozen_scores_probabilities_context_and_masks_match_with_unused_capacity() {
    let receipt = include_bytes!(
        "../../../benchmarks/results/2026-10-24-gemma-local-attention-reference-v1.json"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(receipt)),
        "4bd3f72204f9bba31f07d80db65b132965fd441133e7763058d1eb148bf69372"
    );
    let accepted: Value = serde_json::from_slice(receipt).unwrap();
    let raw = include_bytes!("../../../tests/fixtures/gemma-local-attention-v1/math.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        accepted["fixture_sha256"].as_str().unwrap()
    );
    let fixture: Value = serde_json::from_slice(raw).unwrap();
    for row in fixture["cases"].as_array().unwrap() {
        let n = |key: &str| row[key].as_u64().unwrap() as usize;
        if n("queries") == 0 {
            continue;
        }
        let s = WindowAttentionShape {
            queries: n("queries"),
            keys: n("keys"),
            capacity: n("keys") + 2,
            query_heads: n("q_heads"),
            kv_heads: n("kv_heads"),
            head_dim: n("dims"),
            query_start: n("origin") + n("keys") - n("queries"),
            key_start: n("origin"),
            window: row["window"].as_u64().map(|x| x as usize),
            scale: row["scale"].as_f64().unwrap() as f32,
        };
        let values = |name: &str| {
            row["arrays"][name]["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect::<Vec<_>>()
        };
        let transpose = |values: &[f64]| {
            let mut out = vec![0.; s.queries * s.query_heads * s.head_dim];
            for t in 0..s.queries {
                for h in 0..s.query_heads {
                    for f in 0..s.head_dim {
                        out[(t * s.query_heads + h) * s.head_dim + f] =
                            values[(h * s.queries + t) * s.head_dim + f];
                    }
                }
            }
            out
        };
        let q: Vec<f32> = transpose(&values("q")).iter().map(|&x| x as f32).collect();
        let padded = |name| {
            let source = values(name);
            let mut out = vec![f32::NAN; s.kv_heads * s.capacity * s.head_dim];
            for h in 0..s.kv_heads {
                for t in 0..s.keys {
                    for f in 0..s.head_dim {
                        out[(h * s.capacity + t) * s.head_dim + f] =
                            source[(h * s.keys + t) * s.head_dim + f] as f32;
                    }
                }
            }
            out
        };
        let (k, v) = (padded("k"), padded("v"));
        let mut context = vec![f32::NAN; q.len()];
        let mut scores = vec![f32::NAN; s.query_heads * s.queries * s.keys];
        let mut probabilities = scores.clone();
        window_attention(
            &q,
            &k,
            &v,
            &mut context,
            &mut scores,
            &mut probabilities,
            &s,
        )
        .unwrap();
        close(&context, &transpose(&values("output")));
        close(&scores, &values("scores"));
        close(&probabilities, &values("probabilities"));
        for h in 0..s.query_heads {
            for (i, mask) in row["arrays"]["mask"]["values"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
            {
                if !mask.as_bool().unwrap() {
                    assert_eq!(
                        probabilities[h * s.queries * s.keys + i].to_bits(),
                        0_f32.to_bits()
                    );
                }
            }
        }
    }
}

#[test]
fn hand_group_mapping_window_and_capacity_strides() {
    let s = WindowAttentionShape {
        queries: 1,
        keys: 3,
        capacity: 5,
        query_heads: 4,
        kv_heads: 2,
        head_dim: 1,
        query_start: 19,
        key_start: 17,
        window: Some(2),
        scale: 1.,
    };
    let k = [
        0.,
        0.,
        0.,
        f32::NAN,
        f32::NAN,
        0.,
        0.,
        0.,
        f32::NAN,
        f32::NAN,
    ];
    let v = [
        1.,
        3.,
        5.,
        f32::NAN,
        f32::NAN,
        11.,
        13.,
        15.,
        f32::NAN,
        f32::NAN,
    ];
    let mut context = [9.; 4];
    let mut scores = [9.; 12];
    let mut probabilities = [9.; 12];
    window_attention(
        &[0.; 4],
        &k,
        &v,
        &mut context,
        &mut scores,
        &mut probabilities,
        &s,
    )
    .unwrap();
    assert_eq!(context, [4., 4., 14., 14.]);
    assert_eq!(scores, [0.; 12]);
    assert_eq!(
        probabilities,
        [0., 0.5, 0.5, 0., 0.5, 0.5, 0., 0.5, 0.5, 0., 0.5, 0.5]
    );
}

#[test]
fn invalid_admission_and_late_fully_masked_query_preserve_all_outputs() {
    let mut invalid = vec![
        WindowAttentionShape {
            queries: 0,
            ..shape()
        },
        WindowAttentionShape { keys: 0, ..shape() },
        WindowAttentionShape {
            capacity: 1,
            ..shape()
        },
        WindowAttentionShape {
            query_heads: 0,
            ..shape()
        },
        WindowAttentionShape {
            kv_heads: 0,
            ..shape()
        },
        WindowAttentionShape {
            head_dim: 0,
            ..shape()
        },
        WindowAttentionShape {
            query_heads: 3,
            kv_heads: 2,
            ..shape()
        },
        WindowAttentionShape {
            window: Some(0),
            ..shape()
        },
        WindowAttentionShape {
            queries: 2,
            query_start: usize::MAX,
            ..shape()
        },
        WindowAttentionShape {
            key_start: usize::MAX,
            ..shape()
        },
        WindowAttentionShape {
            capacity: usize::MAX,
            ..shape()
        },
        WindowAttentionShape {
            queries: usize::MAX,
            ..shape()
        },
    ];
    for scale in [0., -1., f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        invalid.push(WindowAttentionShape { scale, ..shape() });
    }
    let mut context = [7.; 5];
    let mut scores = [8.; 5];
    let mut probabilities = [9.; 5];
    for s in invalid {
        assert!(
            window_attention(
                &[1.; 4],
                &[1.; 6],
                &[1.; 6],
                &mut context[..4],
                &mut scores[..4],
                &mut probabilities[..4],
                &s
            )
            .is_err()
        );
        assert_eq!(context, [7.; 5]);
        assert_eq!(scores, [8.; 5]);
        assert_eq!(probabilities, [9.; 5]);
    }
    for tensor in 0..6 {
        for longer in [false, true] {
            let q = [1.; 5];
            let k = [1.; 7];
            let v = [1.; 7];
            let size = |which, valid| {
                if tensor == which {
                    if longer { valid + 1 } else { valid - 1 }
                } else {
                    valid
                }
            };
            assert!(
                window_attention(
                    &q[..size(0, 4)],
                    &k[..size(1, 6)],
                    &v[..size(2, 6)],
                    &mut context[..size(3, 4)],
                    &mut scores[..size(4, 4)],
                    &mut probabilities[..size(5, 4)],
                    &shape()
                )
                .is_err()
            );
            assert_eq!(context, [7.; 5]);
            assert_eq!(scores, [8.; 5]);
            assert_eq!(probabilities, [9.; 5]);
        }
    }
    for tensor in 0..3 {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut q = [1.; 4];
            let mut k = [1.; 6];
            let mut v = [1.; 6];
            match tensor {
                0 => q[3] = value,
                1 => k[3] = value,
                _ => v[3] = value,
            };
            assert!(
                window_attention(
                    &q,
                    &k,
                    &v,
                    &mut context[..4],
                    &mut scores[..4],
                    &mut probabilities[..4],
                    &shape()
                )
                .is_err()
            );
            assert_eq!(context, [7.; 5]);
            assert_eq!(scores, [8.; 5]);
            assert_eq!(probabilities, [9.; 5]);
        }
    }
    for (qstart, kstart, queries, expected_row) in [(9, 10, 1, 0), (20, 10, 1, 0), (10, 10, 3, 2)] {
        let s = WindowAttentionShape {
            queries,
            keys: 2,
            capacity: 2,
            query_start: qstart,
            key_start: kstart,
            window: Some(1),
            ..shape()
        };
        let q = vec![1.; queries * 4];
        let mut c = vec![7.; q.len()];
        let mut a = vec![8.; queries * 4];
        let mut b = vec![9.; queries * 4];
        assert_eq!(
            window_attention(&q, &[1.; 4], &[1.; 4], &mut c, &mut a, &mut b, &s),
            Err(Error::FullyMasked { row: expected_row })
        );
        assert!(c.iter().all(|&x| x == 7.));
        assert!(a.iter().all(|&x| x == 8.));
        assert!(b.iter().all(|&x| x == 9.));
    }
}

#[test]
fn score_overflow_preserves_inputs_and_replay_succeeds() {
    let s = WindowAttentionShape {
        queries: 2,
        keys: 2,
        capacity: 2,
        query_heads: 1,
        kv_heads: 1,
        head_dim: 1,
        query_start: 0,
        key_start: 0,
        window: None,
        scale: 1.,
    };
    let q = [0., f32::MAX];
    let k = [1., f32::MAX];
    let v = [1., 2.];
    let mut c = [9.; 2];
    let mut a = [9.; 4];
    let mut b = [9.; 4];
    assert_eq!(
        window_attention(&q, &k, &v, &mut c, &mut a, &mut b, &s),
        Err(Error::NonFinite {
            tensor: "attention scores",
            index: 3
        })
    );
    assert_eq!(q, [0., f32::MAX]);
    assert_eq!(k, [1., f32::MAX]);
    assert_eq!(v, [1., 2.]);
    assert_eq!(c, [1., 9.]);
    window_attention(&[0., 1.], &k, &v, &mut c, &mut a, &mut b, &s).unwrap();
    assert_eq!(c, [1., 2.]);
}

#[test]
fn rounded_probability_context_overflow_rejects_and_replays() {
    let s = WindowAttentionShape {
        queries: 1,
        keys: 7,
        capacity: 7,
        query_heads: 1,
        kv_heads: 1,
        head_dim: 1,
        query_start: 6,
        key_start: 0,
        window: None,
        scale: 1.,
    };
    let v = [f32::MAX; 7];
    let mut c = [9.];
    let mut a = [9.; 7];
    let mut b = [9.; 7];
    // Seven FP32 rounded 1/7 probabilities sum slightly above one in FP64.
    assert_eq!(
        window_attention(&[0.], &[0.; 7], &v, &mut c, &mut a, &mut b, &s),
        Err(Error::NonFinite {
            tensor: "attention context",
            index: 0
        })
    );
    assert_eq!(v, [f32::MAX; 7]);
    assert_eq!(c, [9.]);
    window_attention(&[0.], &[0.; 7], &[1.; 7], &mut c, &mut a, &mut b, &s).unwrap();
    assert_eq!(c, [1.]);
}

#[test]
fn multi_head_prefill_equals_chunks_with_absolute_offsets() {
    let q: Vec<f32> = (0..5 * 4 * 3)
        .map(|i| ((i * 7 % 31) as f32 - 15.) * 0.125)
        .collect();
    let k: Vec<f32> = (0..2 * 5 * 3)
        .map(|i| ((i * 11 % 31) as f32 - 15.) * 0.125)
        .collect();
    let v: Vec<f32> = k.iter().map(|x| x - 2.).collect();
    let s = WindowAttentionShape {
        queries: 5,
        keys: 5,
        capacity: 5,
        query_heads: 4,
        kv_heads: 2,
        head_dim: 3,
        query_start: 17,
        key_start: 17,
        window: Some(3),
        scale: 1.,
    };
    let mut full = vec![0.; q.len()];
    let mut a = vec![0.; 100];
    let mut b = a.clone();
    window_attention(&q, &k, &v, &mut full, &mut a, &mut b, &s).unwrap();
    for chunk in [1, 2, 3] {
        let mut joined = vec![0.; q.len()];
        for first in (0..5).step_by(chunk) {
            let end = (first + chunk).min(5);
            let part = WindowAttentionShape {
                queries: end - first,
                keys: end,
                query_start: 17 + first,
                ..s
            };
            let mut a = vec![0.; 4 * (end - first) * end];
            let mut b = a.clone();
            window_attention(
                &q[first * 12..end * 12],
                &k,
                &v,
                &mut joined[first * 12..end * 12],
                &mut a,
                &mut b,
                &part,
            )
            .unwrap();
        }
        assert_eq!(joined, full);
    }
}
