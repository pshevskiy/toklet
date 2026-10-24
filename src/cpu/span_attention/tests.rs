use super::*;
use crate::cpu::{WindowAttentionShape, window_attention};
use crate::model::window_cache::WindowKvCache;
use serde_json::Value;
use sha2::{Digest, Sha256};

struct Storage {
    k: Vec<f32>,
    v: Vec<f32>,
    keys: usize,
    stride: usize,
    start: usize,
}
fn packed(
    k: &[f32],
    v: &[f32],
    heads: usize,
    keys: usize,
    dims: usize,
    start: usize,
    parts: [usize; 3],
) -> [Storage; 3] {
    let mut position = 0;
    std::array::from_fn(|i| {
        let n = parts[i];
        let stride = n * dims + 2;
        let count = if n == 0 {
            0
        } else {
            (heads - 1) * stride + n * dims
        };
        let mut out = Storage {
            k: vec![f32::NAN; count],
            v: vec![f32::NAN; count],
            keys: n,
            stride,
            start: start + position,
        };
        for h in 0..heads {
            for row in 0..n {
                let source = (h * keys + position + row) * dims;
                let target = h * stride + row * dims;
                out.k[target..target + dims].copy_from_slice(&k[source..source + dims]);
                out.v[target..target + dims].copy_from_slice(&v[source..source + dims]);
            }
        }
        position += n;
        out
    })
}
fn segments(data: &[Storage; 3]) -> [AttentionKvSegment<'_>; 3] {
    std::array::from_fn(|i| AttentionKvSegment {
        k: &data[i].k,
        v: &data[i].v,
        keys: data[i].keys,
        head_stride: data[i].stride,
        key_start: data[i].start,
    })
}
fn transpose(data: &[f64], heads: usize, tokens: usize, dims: usize) -> Vec<f64> {
    let mut out = vec![0.; data.len()];
    for h in 0..heads {
        for t in 0..tokens {
            for d in 0..dims {
                out[(t * heads + h) * dims + d] = data[(h * tokens + t) * dims + d];
            }
        }
    }
    out
}
fn close(actual: &[f32], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &e) in actual.iter().zip(expected) {
        assert!(
            a.is_finite() && (f64::from(a) - e).abs() <= 1e-5 + 1e-4 * e.abs(),
            "{a} vs {e}"
        );
    }
}
fn bits(data: &[f32]) -> Vec<u32> {
    data.iter().map(|x| x.to_bits()).collect()
}
fn basic() -> SpanAttentionShape {
    SpanAttentionShape {
        queries: 1,
        keys: 2,
        query_heads: 2,
        kv_heads: 1,
        head_dim: 2,
        query_start: 1,
        key_start: 0,
        window: Some(2),
        scale: 1.,
    }
}

#[test]
fn frozen_scores_probabilities_context_and_exact_contiguous_reference() {
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
    let data: Value = serde_json::from_slice(raw).unwrap();
    for case in data["cases"].as_array().unwrap() {
        let n = |key: &str| case[key].as_u64().unwrap() as usize;
        let (queries, keys, h, hk, d) = (
            n("queries"),
            n("keys"),
            n("q_heads"),
            n("kv_heads"),
            n("dims"),
        );
        if queries == 0 {
            continue;
        }
        let values = |key: &str| {
            case["arrays"][key]["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect::<Vec<_>>()
        };
        let q: Vec<f32> = transpose(&values("q"), h, queries, d)
            .iter()
            .map(|&x| x as f32)
            .collect();
        let k = values("k").iter().map(|&x| x as f32).collect::<Vec<_>>();
        let v = values("v").iter().map(|&x| x as f32).collect::<Vec<_>>();
        let s = SpanAttentionShape {
            queries,
            keys,
            query_heads: h,
            kv_heads: hk,
            head_dim: d,
            query_start: n("origin") + keys - queries,
            key_start: n("origin"),
            window: case["window"].as_u64().map(|v| v as usize),
            scale: case["scale"].as_f64().unwrap() as f32,
        };
        let first = keys.min(1);
        let last = (keys - first).min(1);
        let backing = packed(
            &k,
            &v,
            hk,
            keys,
            d,
            s.key_start,
            [first, keys - first - last, last],
        );
        let spans = segments(&backing);
        let before = backing
            .iter()
            .map(|a| (bits(&a.k), bits(&a.v)))
            .collect::<Vec<_>>();
        let mut context = vec![f32::NAN; q.len()];
        let mut scores = vec![f32::NAN; h * queries * keys];
        let mut probabilities = scores.clone();
        span_attention(
            &q,
            &spans,
            &mut context,
            &mut scores,
            &mut probabilities,
            &s,
        )
        .unwrap();
        close(&scores, &values("scores"));
        close(&probabilities, &values("probabilities"));
        let expected = transpose(&values("output"), h, queries, d);
        close(&context, &expected);
        let old = WindowAttentionShape {
            queries,
            keys,
            capacity: keys,
            query_heads: h,
            kv_heads: hk,
            head_dim: d,
            query_start: s.query_start,
            key_start: s.key_start,
            window: s.window,
            scale: s.scale,
        };
        let mut ctx = vec![0.; q.len()];
        let mut sc = vec![0.; scores.len()];
        let mut pr = sc.clone();
        window_attention(&q, &k, &v, &mut ctx, &mut sc, &mut pr, &old).unwrap();
        assert_eq!(bits(&context), bits(&ctx));
        assert_eq!(bits(&scores), bits(&sc));
        assert_eq!(bits(&probabilities), bits(&pr));
        assert_eq!(
            backing
                .iter()
                .map(|a| (bits(&a.k), bits(&a.v)))
                .collect::<Vec<_>>(),
            before
        );
    }
}

#[test]
fn actual_cache_ring_views_match_all_frozen_kv_contexts() {
    let receipt =
        include_bytes!("../../../benchmarks/results/2026-10-24-gemma-kv-reference-v1.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(receipt)),
        "d2972038be572fdf1af38cf09b02894c7d99353c62c79ecc96d3378834937498"
    );
    let accepted: Value = serde_json::from_slice(receipt).unwrap();
    let raw = include_bytes!("../../../tests/fixtures/gemma-kv-v1/math.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        accepted["fixture_sha256"].as_str().unwrap()
    );
    let data: Value = serde_json::from_slice(raw).unwrap();
    let mut count = 0;
    let mut wrapped_multihead = false;
    for case in data["cases"].as_array().unwrap() {
        let n = |key: &str| case[key].as_u64().unwrap() as usize;
        let (hk, h, d, cap) = (n("kv_heads"), n("query_heads"), n("dims"), n("capacity"));
        let max = case["chunks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .sum::<usize>()
            .max(cap);
        let mut cache =
            WindowKvCache::new(hk, d, cap, max, case["window"].as_u64().map(|v| v as usize))
                .unwrap();
        for step in case["steps"].as_array().unwrap() {
            let values = |name: &str| {
                step["arrays"][name]["values"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect::<Vec<_>>()
            };
            let tokens = step["count"].as_u64().unwrap() as usize;
            let q: Vec<f32> = transpose(&values("q"), h, tokens, d)
                .iter()
                .map(|&x| x as f32)
                .collect();
            let k = values("new_k")
                .iter()
                .map(|&v| v as f32)
                .collect::<Vec<_>>();
            let v = values("new_v")
                .iter()
                .map(|&v| v as f32)
                .collect::<Vec<_>>();
            let frame = cache.prepare(&k, &v, tokens).unwrap();
            let before = (
                bits(cache.committed().0),
                bits(cache.committed().1),
                cache.offset(),
                cache.generation(),
            );
            {
                let view = cache.view(&frame).unwrap();
                let shape = view.attention_shape(h, 1.);
                let spans = view.segments();
                if hk > 1 && spans[1].keys > 0 {
                    wrapped_multihead = true;
                }
                for head in 0..hk {
                    for (span, one) in spans.iter().zip(view.head(head).unwrap()) {
                        if span.keys == 0 {
                            assert!(span.k.is_empty() && span.v.is_empty());
                            continue;
                        }
                        assert_eq!(span.k[head * span.head_stride..].as_ptr(), one.k.as_ptr());
                        assert_eq!(span.v[head * span.head_stride..].as_ptr(), one.v.as_ptr());
                    }
                }
                let mut ctx = vec![0.; q.len()];
                let mut sc = vec![0.; h * tokens * shape.keys];
                let mut pr = sc.clone();
                span_attention(&q, &spans, &mut ctx, &mut sc, &mut pr, &shape).unwrap();
                let expected = transpose(&values("context"), h, tokens, d);
                close(&ctx, &expected);
            }
            assert_eq!(
                (
                    bits(cache.committed().0),
                    bits(cache.committed().1),
                    cache.offset(),
                    cache.generation()
                ),
                before
            );
            cache.commit(&frame).unwrap();
            count += 1;
        }
    }
    assert_eq!(count, 30);
    assert!(wrapped_multihead);
}

#[test]
fn empty_segments_and_signed_zero_padding_are_ignored_and_preserved() {
    let mut s = basic();
    s.query_heads = 4;
    s.kv_heads = 2;
    s.head_dim = 1;
    let q = [0.; 4];
    let k = [0.; 4];
    let v = [1., 3., 11., 13.];
    for parts in [[0, 2, 0], [2, 0, 0], [0, 0, 2], [1, 0, 1]] {
        let mut data = packed(&k, &v, 2, 2, 1, 0, parts);
        for item in &mut data {
            for value in item.k.iter_mut().chain(&mut item.v) {
                if value.is_nan() {
                    *value = -0.;
                }
            }
        }
        let before = data
            .iter()
            .map(|a| (bits(&a.k), bits(&a.v)))
            .collect::<Vec<_>>();
        let mut spans = segments(&data);
        for span in &mut spans {
            if span.keys == 0 {
                span.head_stride = usize::MAX;
                span.key_start = usize::MAX;
            }
        }
        let mut ctx = [0.; 4];
        let mut scores = [0.; 8];
        let mut probs = [0.; 8];
        span_attention(&q, &spans, &mut ctx, &mut scores, &mut probs, &s).unwrap();
        assert_eq!(ctx, [2., 2., 12., 12.]);
        assert_eq!(
            data.iter()
                .map(|a| (bits(&a.k), bits(&a.v)))
                .collect::<Vec<_>>(),
            before
        );
    }
}

#[test]
fn invalid_and_late_segments_reject_before_any_output_write() {
    let s = basic();
    let q = [1.; 4];
    let k = [1.; 4];
    let v = [2.; 4];
    let data = packed(&k, &v, 1, 2, 2, 0, [1, 0, 1]);
    let good = segments(&data);
    for mode in 0..8 {
        let mut spans = good;
        let bad = [f32::NAN, 0.];
        let one = [0.];
        match mode {
            0 => spans[2].key_start = 0,
            1 => spans[2].key_start = 3,
            2 => spans.swap(0, 2),
            3 => spans[2].head_stride = 1,
            4 => spans[2].k = &one,
            5 => spans[2].v = &bad,
            6 => spans[1].k = &one,
            7 => spans[2].keys = usize::MAX,
            _ => unreachable!(),
        }
        let (mut ctx, mut scores, mut probs) = ([77.; 4], [77.; 4], [77.; 4]);
        assert!(span_attention(&q, &spans, &mut ctx, &mut scores, &mut probs, &s).is_err());
        assert_eq!(ctx, [77.; 4]);
        assert_eq!(scores, [77.; 4]);
        assert_eq!(probs, [77.; 4]);
    }
    let mut more = basic();
    more.query_heads = 2;
    more.kv_heads = 2;
    let multi = packed(&[1.; 8], &[2.; 8], 2, 2, 2, 0, [1, 0, 1]);
    let mut spans = segments(&multi);
    spans[2].head_stride = usize::MAX;
    let (mut ctx, mut scores, mut probs) = ([77.; 4], [77.; 4], [77.; 4]);
    assert!(span_attention(&q, &spans, &mut ctx, &mut scores, &mut probs, &more).is_err());
    assert_eq!(ctx, [77.; 4]);
    assert_eq!(scores, [77.; 4]);
    assert_eq!(probs, [77.; 4]);
}

#[test]
fn invalid_common_geometry_lengths_finiteness_and_later_fully_masked_query_are_prewrite() {
    let q = [1.; 4];
    let k = [1.; 4];
    let v = [2.; 4];
    let data = packed(&k, &v, 1, 2, 2, 0, [1, 0, 1]);
    let spans = segments(&data);
    for mode in 0..14 {
        let mut s = basic();
        match mode {
            0 => s.queries = 0,
            1 => s.keys = 0,
            2 => s.query_heads = 0,
            3 => s.kv_heads = 0,
            4 => s.head_dim = 0,
            5 => s.kv_heads = 3,
            6 => s.window = Some(0),
            7 => s.scale = f32::NAN,
            8 => s.scale = 0.,
            9 => s.key_start = usize::MAX,
            10 => {
                s.query_start = usize::MAX;
                s.queries = 2;
            }
            11 => s.head_dim = usize::MAX,
            12 => s.query_heads = usize::MAX,
            13 => s.scale = f32::INFINITY,
            _ => unreachable!(),
        }
        let input = &q[..];
        let (mut ctx, mut scores, mut probs) = ([77.; 4], [77.; 4], [77.; 4]);
        assert!(span_attention(input, &spans, &mut ctx, &mut scores, &mut probs, &s).is_err());
        assert_eq!(ctx, [77.; 4]);
        assert_eq!(scores, [77.; 4]);
        assert_eq!(probs, [77.; 4]);
    }
    let s = basic();
    for mode in 0..4 {
        let (mut ctx, mut scores, mut probs) = ([77.; 4], [77.; 4], [77.; 4]);
        let mut bad_q = q;
        if mode == 0 {
            bad_q[3] = f32::NEG_INFINITY;
        }
        assert!(
            span_attention(
                &bad_q,
                &spans,
                if mode == 1 { &mut ctx[..3] } else { &mut ctx },
                if mode == 2 {
                    &mut scores[..3]
                } else {
                    &mut scores
                },
                if mode == 3 {
                    &mut probs[..3]
                } else {
                    &mut probs
                },
                &s
            )
            .is_err()
        );
        assert_eq!(ctx, [77.; 4]);
        assert_eq!(scores, [77.; 4]);
        assert_eq!(probs, [77.; 4]);
    }
    let s = SpanAttentionShape {
        queries: 2,
        keys: 1,
        query_heads: 1,
        kv_heads: 1,
        head_dim: 1,
        query_start: 0,
        key_start: 0,
        window: Some(1),
        scale: 1.,
    };
    let data = packed(&[0.], &[1.], 1, 1, 1, 0, [1, 0, 0]);
    let (mut ctx, mut sc, mut pr) = ([77.; 2], [77.; 2], [77.; 2]);
    assert_eq!(
        span_attention(&[0.; 2], &segments(&data), &mut ctx, &mut sc, &mut pr, &s),
        Err(Error::FullyMasked { row: 1 })
    );
    assert_eq!(ctx, [77.; 2]);
    assert_eq!(sc, [77.; 2]);
    assert_eq!(pr, [77.; 2]);
}

#[test]
fn arithmetic_overflow_outputs_are_discardable_and_inputs_support_replay() {
    let s = SpanAttentionShape {
        queries: 1,
        keys: 2,
        query_heads: 1,
        kv_heads: 1,
        head_dim: 1,
        query_start: 0,
        key_start: 0,
        window: Some(1),
        scale: 1.,
    };
    let q = [2.];
    let k = [0., f32::MAX];
    let v = [1., 2.];
    let data = packed(&k, &v, 1, 2, 1, 0, [1, 0, 1]);
    let before = (bits(&k), bits(&v));
    let (mut ctx, mut sc, mut pr) = ([0.], [0.; 2], [0.; 2]);
    assert!(span_attention(&q, &segments(&data), &mut ctx, &mut sc, &mut pr, &s).is_err());
    span_attention(&[0.], &segments(&data), &mut ctx, &mut sc, &mut pr, &s).unwrap();
    assert_eq!(ctx, [1.]);
    assert_eq!((bits(&k), bits(&v)), before);
    let s = SpanAttentionShape {
        queries: 1,
        keys: 7,
        query_heads: 1,
        kv_heads: 1,
        head_dim: 1,
        query_start: 6,
        key_start: 0,
        window: None,
        scale: 1.,
    };
    let data = packed(&[0.; 7], &[f32::MAX; 7], 1, 7, 1, 0, [2, 3, 2]);
    let (mut ctx, mut sc, mut pr) = ([0.], [0.; 7], [0.; 7]);
    assert!(span_attention(&[0.], &segments(&data), &mut ctx, &mut sc, &mut pr, &s).is_err());
    let good = packed(&[0.; 7], &[1.; 7], 1, 7, 1, 0, [2, 3, 2]);
    span_attention(&[0.], &segments(&good), &mut ctx, &mut sc, &mut pr, &s).unwrap();
    assert!(ctx[0].is_finite());
}

#[test]
fn failed_ring_attention_does_not_commit_and_retry_appends_once() {
    let mut cache = WindowKvCache::new(1, 1, 2, 6, Some(2)).unwrap();
    let old = [-0., 1.];
    let frame = cache.prepare(&old, &old, 2).unwrap();
    cache.commit(&frame).unwrap();
    let new = [f32::MAX];
    let values = [2.];
    let frame = cache.prepare(&new, &values, 1).unwrap();
    let before = (
        bits(cache.committed().0),
        bits(cache.committed().1),
        cache.offset(),
        cache.generation(),
    );
    {
        let view = cache.view(&frame).unwrap();
        let shape = view.attention_shape(1, 1.);
        let (mut ctx, mut sc, mut pr) = ([0.], [0.; 2], [0.; 2]);
        assert!(
            span_attention(&[2.], &view.segments(), &mut ctx, &mut sc, &mut pr, &shape).is_err()
        );
    }
    assert_eq!(
        (
            bits(cache.committed().0),
            bits(cache.committed().1),
            cache.offset(),
            cache.generation()
        ),
        before
    );
    {
        let view = cache.view(&frame).unwrap();
        let shape = view.attention_shape(1, 1.);
        let (mut ctx, mut sc, mut pr) = ([0.], [0.; 2], [0.; 2]);
        span_attention(&[0.], &view.segments(), &mut ctx, &mut sc, &mut pr, &shape).unwrap();
        assert_eq!(ctx, [1.5]);
    }
    cache.commit(&frame).unwrap();
    assert_eq!(cache.offset(), 3);
    assert_eq!(cache.generation(), 2);
    assert!(cache.commit(&frame).is_err());
}

#[test]
fn inclusive_maximum_position_and_hand_grouped_heads_match() {
    let s = SpanAttentionShape {
        queries: 1,
        keys: 2,
        query_heads: 4,
        kv_heads: 2,
        head_dim: 3,
        query_start: usize::MAX,
        key_start: usize::MAX - 1,
        window: Some(2),
        scale: 1.,
    };
    let q = [0.; 12];
    let k = [0.; 12];
    let v = [1., 2., 3., 3., 4., 5., 11., 12., 13., 13., 14., 15.];
    // Construct the final single-key segment without computing MAX+1.
    let first_k = [0.; 6];
    let first_v = [1., 2., 3., 11., 12., 13.];
    let last_k = [0.; 6];
    let last_v = [3., 4., 5., 13., 14., 15.];
    let spans = [
        AttentionKvSegment {
            k: &first_k,
            v: &first_v,
            keys: 1,
            head_stride: 3,
            key_start: usize::MAX - 1,
        },
        AttentionKvSegment {
            k: &[],
            v: &[],
            keys: 0,
            head_stride: usize::MAX,
            key_start: 0,
        },
        AttentionKvSegment {
            k: &last_k,
            v: &last_v,
            keys: 1,
            head_stride: 3,
            key_start: usize::MAX,
        },
    ];
    let (mut ctx, mut sc, mut pr) = ([0.; 12], [0.; 8], [0.; 8]);
    span_attention(&q, &spans, &mut ctx, &mut sc, &mut pr, &s).unwrap();
    assert_eq!(ctx, [2., 3., 4., 2., 3., 4., 12., 13., 14., 12., 13., 14.]);
    assert_eq!(pr, [0.5; 8]);
    let old = WindowAttentionShape {
        queries: 1,
        keys: 2,
        capacity: 2,
        query_heads: 4,
        kv_heads: 2,
        head_dim: 3,
        query_start: usize::MAX,
        key_start: usize::MAX - 1,
        window: Some(2),
        scale: 1.,
    };
    let (mut other, mut scores, mut probs) = ([0.; 12], [0.; 8], [0.; 8]);
    window_attention(&q, &k, &v, &mut other, &mut scores, &mut probs, &old).unwrap();
    assert_eq!(bits(&ctx), bits(&other));
}
