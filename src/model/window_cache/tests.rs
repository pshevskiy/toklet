use super::*;
use serde_json::Value;
use sha2::{Digest, Sha256};

type Snapshot = (Vec<u32>, Vec<u32>, usize, u64, Vec<Option<usize>>);

fn discard(_frame: PreparedKv<'_>) {}

fn snapshot(cache: &WindowKvCache) -> Snapshot {
    let (k, v) = cache.committed();
    (
        k.iter().map(|x| x.to_bits()).collect(),
        v.iter().map(|x| x.to_bits()).collect(),
        cache.offset(),
        cache.generation(),
        (0..cache.capacity)
            .map(|slot| cache.position_in_slot(slot))
            .collect(),
    )
}

fn values(step: &Value, name: &str) -> Vec<f32> {
    step["arrays"][name]["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}

#[test]
fn frozen_states_spans_visibility_abort_and_retry() {
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
    let mut checked = 0;
    for case in data["cases"].as_array().unwrap() {
        let n = |key: &str| case[key].as_u64().unwrap() as usize;
        let window = case["window"].as_u64().map(|x| x as usize);
        let max = case["chunks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .sum::<usize>()
            .max(n("capacity"));
        let mut cache =
            WindowKvCache::new(n("kv_heads"), n("dims"), n("capacity"), max, window).unwrap();
        for key in ["initial_k", "initial_v"] {
            assert_eq!(
                case[key]["values"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap() as f32)
                    .collect::<Vec<_>>(),
                vec![0.; cache.k.len()]
            );
        }
        assert_eq!(cache.offset(), 0);
        assert_eq!(cache.generation(), 0);
        for step in case["steps"].as_array().unwrap() {
            let tokens = step["count"].as_u64().unwrap() as usize;
            let (k, v) = (values(step, "new_k"), values(step, "new_v"));
            let before = snapshot(&cache);
            assert_eq!(cache.committed().0, values(step, "ring_before_k"));
            assert_eq!(cache.committed().1, values(step, "ring_before_v"));
            let frame = cache.prepare(&k, &v, tokens).unwrap();
            {
                let view = cache.view(&frame).unwrap();
                assert_eq!(
                    view.query_start(),
                    step["before"].as_u64().unwrap() as usize
                );
                assert_eq!(view.end(), step["after"].as_u64().unwrap() as usize);
                let positions = step["required_positions"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as usize)
                    .collect::<Vec<_>>();
                assert_eq!(view.start(), positions[0]);
                for head in 0..cache.heads {
                    let spans = view.head(head).unwrap();
                    let mut seen = Vec::new();
                    let mut keys = Vec::new();
                    let mut vals = Vec::new();
                    for span in spans {
                        seen.extend(span.position..span.position + span.k.len() / cache.dims);
                        keys.extend_from_slice(span.k);
                        vals.extend_from_slice(span.v);
                    }
                    assert_eq!(seen, positions);
                    let logical_k = values(step, "logical_k");
                    let logical_v = values(step, "logical_v");
                    let stride = view.end() * cache.dims;
                    let expected = |source: &[f32]| {
                        positions
                            .iter()
                            .flat_map(|&p| {
                                source[head * stride + p * cache.dims
                                    ..head * stride + (p + 1) * cache.dims]
                                    .iter()
                                    .copied()
                            })
                            .collect::<Vec<_>>()
                    };
                    assert_eq!(keys, expected(&logical_k));
                    assert_eq!(vals, expected(&logical_v));
                    for (index, span) in spans[..2].iter().enumerate() {
                        if span.k.is_empty() {
                            continue;
                        }
                        let frozen = &step["old_spans"][index];
                        let physical = frozen[0].as_u64().unwrap() as usize;
                        let count = frozen[1].as_u64().unwrap() as usize;
                        assert_eq!(
                            span.k.as_ptr(),
                            cache.k[(head * cache.capacity + physical) * cache.dims..].as_ptr()
                        );
                        assert_eq!(span.k.len(), count * cache.dims);
                    }
                    assert_eq!(
                        spans[2].k.as_ptr(),
                        k[head * tokens * cache.dims..].as_ptr()
                    );
                }
                for (row, query) in (view.query_start()..view.end()).enumerate() {
                    let frozen = &step["visible_ranges"][row];
                    assert_eq!(
                        view.visible_range(query).unwrap(),
                        frozen[0].as_u64().unwrap() as usize..frozen[1].as_u64().unwrap() as usize
                    );
                }
            }
            assert_eq!(snapshot(&cache), before);
            if step["abort_expected"].as_bool().unwrap() {
                // A discarded prepared frame did not modify any cache byte.
                discard(frame);
                assert_eq!(snapshot(&cache), before);
                let retried = cache.prepare(&k, &v, tokens).unwrap();
                cache.commit(&retried).unwrap();
            } else {
                cache.commit(&frame).unwrap();
            }
            assert_eq!(cache.committed().0, values(step, "ring_after_k"));
            assert_eq!(cache.committed().1, values(step, "ring_after_v"));
            assert_eq!(cache.offset(), step["after"].as_u64().unwrap() as usize);
            assert_eq!(
                cache.generation(),
                step["generation_after"].as_u64().unwrap()
            );
            let positions = step["positions_after"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().map(|p| p as usize))
                .collect::<Vec<_>>();
            assert_eq!(
                (0..cache.capacity)
                    .map(|slot| cache.position_in_slot(slot))
                    .collect::<Vec<_>>(),
                positions
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 30);
}

#[test]
fn batch_last_failure_preserves_every_owner_and_valid_retry_commits_once() {
    let mut caches = [
        WindowKvCache::new(1, 1, 3, 10, Some(3)).unwrap(),
        WindowKvCache::new(1, 1, 3, 10, Some(3)).unwrap(),
    ];
    let input = [-0., 1., 2.];
    let values = [3., 4., 5.];
    let first = [
        caches[0].prepare(&input, &values, 3).unwrap(),
        caches[1].prepare(&input, &values, 3).unwrap(),
    ];
    commit_batch(&mut caches, &first).unwrap();
    let new_k = [6., 7.];
    let new_v = [8., 9.];
    let frames = [
        caches[0].prepare(&new_k, &new_v, 2).unwrap(),
        caches[1].prepare(&new_k, &new_v, 2).unwrap(),
    ];
    let before = [snapshot(&caches[0]), snapshot(&caches[1])];
    for bad in [
        &frames[..1],
        &first[..],
        &[
            caches[0].prepare(&new_k, &new_v, 2).unwrap(),
            caches[0].prepare(&new_k, &new_v, 2).unwrap(),
        ][..],
        &[
            caches[1].prepare(&new_k, &new_v, 2).unwrap(),
            caches[0].prepare(&new_k, &new_v, 2).unwrap(),
        ][..],
    ] {
        assert!(commit_batch(&mut caches, bad).is_err());
        assert_eq!([snapshot(&caches[0]), snapshot(&caches[1])], before);
    }
    // First frame validates; last frame is stale. No first-owner write survives.
    let stale_last = [
        caches[0].prepare(&new_k, &new_v, 2).unwrap(),
        PreparedKv {
            k: first[1].k,
            v: first[1].v,
            owner: first[1].owner,
            generation: first[1].generation,
            offset: first[1].offset,
            tokens: first[1].tokens,
        },
    ];
    assert!(commit_batch(&mut caches, &stale_last).is_err());
    assert_eq!([snapshot(&caches[0]), snapshot(&caches[1])], before);
    commit_batch(&mut caches, &frames).unwrap();
    for cache in &caches {
        assert_eq!(cache.offset(), 5);
        assert_eq!(cache.generation(), 2);
        assert_eq!(cache.committed().0, [6., 7., 2.]);
    }
    let committed = [snapshot(&caches[0]), snapshot(&caches[1])];
    assert!(commit_batch(&mut caches, &frames).is_err());
    assert_eq!([snapshot(&caches[0]), snapshot(&caches[1])], committed);
    assert!(caches[0].view(&frames[0]).is_err());
}

#[test]
fn borrowed_views_preserve_early_queries_and_signed_zero_with_no_reallocation() {
    let mut cache = WindowKvCache::new(1, 1, 3, 20, Some(3)).unwrap();
    let pointers = (cache.k.as_ptr(), cache.v.as_ptr());
    let bytes = cache.allocated_bytes();
    let initial = [-0., 1.];
    let frame = cache.prepare(&initial, &initial, 2).unwrap();
    cache.commit(&frame).unwrap();
    let new = [2., 3.];
    let frame = cache.prepare(&new, &new, 2).unwrap();
    let view = cache.view(&frame).unwrap();
    let spans = view.head(0).unwrap();
    assert_eq!(spans[0].k[0].to_bits(), (-0_f32).to_bits());
    assert_eq!(view.visible_range(2).unwrap(), 0..3);
    assert_eq!(view.visible_range(3).unwrap(), 1..4);
    assert!(view.head(1).is_err());
    assert!(view.visible_range(1).is_err());
    assert!(view.visible_range(4).is_err());
    cache.commit(&frame).unwrap();
    assert_eq!(cache.committed().0, [3., 1., 2.]);
    let huge = [4., 5., 6., 7., 8.];
    let frame = cache.prepare(&huge, &huge, 5).unwrap();
    cache.commit(&frame).unwrap();
    assert_eq!(cache.committed().0, [6., 7., 8.]);
    assert_eq!((cache.k.as_ptr(), cache.v.as_ptr()), pointers);
    assert_eq!(cache.allocated_bytes(), bytes);
    assert_eq!(cache.position_in_slot(3), None);
}

#[test]
fn global_capacity_and_allocation_geometry_are_checked() {
    let planned = WindowKvCache::planned_bytes(2, 3, 4, 8, None).unwrap();
    assert_eq!(
        planned,
        2 * 2 * 3 * 4 * 4 + std::mem::size_of::<WindowKvCache>()
    );
    let mut cache = WindowKvCache::new(2, 3, 4, 8, None).unwrap();
    assert!(cache.allocated_bytes() >= planned);
    let chunk = vec![1.; 2 * 3 * 3];
    let frame = cache.prepare(&chunk, &chunk, 3).unwrap();
    cache.commit(&frame).unwrap();
    let before = snapshot(&cache);
    assert!(cache.prepare(&chunk, &chunk, 3).is_err());
    assert_eq!(snapshot(&cache), before);
    for args in [
        (0, 1, 1, 1, None),
        (1, 0, 1, 1, None),
        (1, 1, 0, 1, None),
        (1, 1, 1, 0, None),
        (1, 1, 2, 1, None),
        (1, 1, 2, 3, Some(1)),
        (1, 1, 1, 3, Some(0)),
        (usize::MAX, 2, 2, 2, None),
        (1, 1, usize::MAX, usize::MAX, None),
        (1, 1, isize::MAX as usize / 4 + 1, usize::MAX, None),
    ] {
        assert!(WindowKvCache::planned_bytes(args.0, args.1, args.2, args.3, args.4).is_err());
    }
}

#[test]
fn invalid_inputs_foreign_frames_and_overflow_preserve_exact_state() {
    let mut cache = WindowKvCache::new(1, 1, 3, 8, Some(3)).unwrap();
    let input = [-0.];
    let first = cache.prepare(&input, &input, 1).unwrap();
    cache.commit(&first).unwrap();
    let before = snapshot(&cache);
    for (k, v, n) in [
        (&[][..], &[][..], 0),
        (&[][..], &input[..], 1),
        (&input[..], &[][..], 1),
        (&input[..], &input[..], usize::MAX),
        (&[f32::NAN][..], &input[..], 1),
        (&input[..], &[f32::INFINITY][..], 1),
        (&[f32::NEG_INFINITY][..], &input[..], 1),
        (&input[..], &input[..], 8),
    ] {
        assert!(cache.prepare(k, v, n).is_err());
        assert_eq!(snapshot(&cache), before);
    }
    let other = WindowKvCache::new(1, 1, 3, 8, Some(3)).unwrap();
    let foreign = other.prepare(&input, &input, 1).unwrap();
    assert!(cache.commit(&foreign).is_err());
    assert!(cache.view(&foreign).is_err());
    assert_eq!(snapshot(&cache), before);
    cache.generation = u64::MAX;
    assert!(cache.prepare(&input, &input, 1).is_err());
    cache.generation = 1;
    cache.offset = usize::MAX;
    assert!(cache.prepare(&input, &input, 1).is_err());
    cache.offset = 1;
    assert_eq!(snapshot(&cache), before);
    // Even an internally corrupted proposal is re-admitted before writes.
    let mut frame = cache.prepare(&input, &input, 1).unwrap();
    frame.tokens = usize::MAX;
    assert!(cache.commit(&frame).is_err());
    assert_eq!(snapshot(&cache), before);
}

#[test]
fn moved_owner_preserves_identity_and_chunking_preserves_logical_ring() {
    for window in [Some(3), None] {
        let mut one = WindowKvCache::new(1, 1, window.unwrap_or(9), 9, window).unwrap();
        let data = [-0., 1., 2., 3., 4., 5., 6., 7., 8.];
        let proposal = one.prepare(&data, &data, 9).unwrap();
        let identity = one.owner;
        let mut moved = one;
        assert_eq!(moved.owner, identity);
        moved.commit(&proposal).unwrap();
        one = WindowKvCache::new(1, 1, window.unwrap_or(9), 9, window).unwrap();
        for input in data.chunks(2) {
            let frame = one.prepare(input, input, input.len()).unwrap();
            one.commit(&frame).unwrap();
        }
        assert_eq!(one.committed(), moved.committed());
        assert_eq!(one.offset(), moved.offset());
        assert_eq!(
            (0..one.capacity)
                .map(|s| one.position_in_slot(s))
                .collect::<Vec<_>>(),
            (0..moved.capacity)
                .map(|s| moved.position_in_slot(s))
                .collect::<Vec<_>>()
        );
    }
}
