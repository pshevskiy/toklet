use super::*;
use serde_json::Value;
use sha2::{Digest, Sha256};

mod capacity;

type Snapshot = Vec<(Vec<u32>, Vec<u32>, usize, u64)>;
fn snapshot(caches: &[WindowKvCache]) -> Snapshot {
    caches
        .iter()
        .map(|cache| {
            let (k, v) = cache.committed();
            (
                k.iter().map(|x| x.to_bits()).collect(),
                v.iter().map(|x| x.to_bits()).collect(),
                cache.offset(),
                cache.generation(),
            )
        })
        .collect()
}
fn geometry(kind: KvKind) -> LayerKvGeometry {
    LayerKvGeometry {
        kind,
        heads: 1,
        head_dim: 1,
    }
}
fn small() -> SharedKvPlan {
    SharedKvPlan::new(
        &[
            geometry(KvKind::Sliding),
            geometry(KvKind::Full),
            geometry(KvKind::Sliding),
            geometry(KvKind::Full),
        ],
        2,
        3,
        8,
        12,
    )
    .unwrap()
}
fn proposal<'a>(
    plan: &SharedKvPlan,
    caches: &[WindowKvCache],
    data: &'a [f32],
    n: usize,
) -> Vec<PreparedKv<'a>> {
    (0..plan.producer_count())
        .map(|i| plan.prepare_producer(i, caches, data, data, n).unwrap())
        .collect()
}

#[test]
fn frozen_routing_current_step_views_and_only_producer_commits() {
    let receipt =
        include_bytes!("../../../benchmarks/results/2026-10-24-gemma-shared-kv-reference-v1.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(receipt)),
        "aa5fd3d27375ce6822aae29bdacbf7c38336cc447b1aaebbf71ede918143554e"
    );
    let accepted: Value = serde_json::from_slice(receipt).unwrap();
    let raw = include_bytes!("../../../tests/fixtures/gemma-shared-kv-v1/expected.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        accepted["fixture_sha256"].as_str().unwrap()
    );
    let data: Value = serde_json::from_slice(raw).unwrap();
    let mut layers_checked = 0;
    let mut appends = 0;
    for case in data["cases"].as_array().unwrap() {
        let specs = case["types"]
            .as_array()
            .unwrap()
            .iter()
            .map(|kind| {
                let local = kind.as_str().unwrap() == "sliding_attention";
                LayerKvGeometry {
                    kind: if local { KvKind::Sliding } else { KvKind::Full },
                    heads: 1,
                    head_dim: if local { 256 } else { 512 },
                }
            })
            .collect::<Vec<_>>();
        let window = case["window"].as_u64().unwrap() as usize;
        let total = case["chunks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .sum::<usize>();
        let plan = SharedKvPlan::new(
            &specs,
            case["shared"].as_u64().unwrap() as usize,
            window,
            total,
            total.max(window),
        )
        .unwrap();
        let mut caches = plan.new_caches().unwrap();
        assert_eq!(caches.len(), plan.producer_count());
        for step in case["steps"].as_array().unwrap() {
            let n = step["count"].as_u64().unwrap() as usize;
            let offset = step["before"].as_u64().unwrap() as usize;
            let inputs = (0..plan.producer_count())
                .map(|producer| {
                    let dims = specs[producer].head_dim;
                    let values = (0..n * dims)
                        .map(|i| {
                            (producer * 10000 + offset + i / dims) as f32
                                + (i % dims) as f32 / 1024.
                        })
                        .collect::<Vec<_>>();
                    (values.clone(), values)
                })
                .collect::<Vec<_>>();
            let frames = inputs
                .iter()
                .enumerate()
                .map(|(i, (k, v))| plan.prepare_producer(i, &caches, k, v, n).unwrap())
                .collect::<Vec<_>>();
            let before = snapshot(&caches);
            for row in step["layers"].as_array().unwrap() {
                let layer = row["layer"].as_u64().unwrap() as usize;
                let producer = row["producer"].as_u64().unwrap() as usize;
                assert_eq!(plan.producer_for(layer).unwrap(), producer);
                let view = plan.view(layer, &caches, &frames).unwrap();
                let own = plan.view(producer, &caches, &frames).unwrap();
                assert_eq!(
                    view.query_start(),
                    row["returned_offset"].as_u64().unwrap() as usize
                );
                assert_eq!(view.start(), row["key_start"].as_u64().unwrap() as usize);
                assert_eq!(view.end(), row["key_end"].as_u64().unwrap() as usize);
                let spans = view.head(0).unwrap();
                let producer_spans = own.head(0).unwrap();
                for (span, other) in spans.iter().zip(producer_spans) {
                    assert_eq!(
                        (
                            span.position,
                            span.k.as_ptr(),
                            span.k.len(),
                            span.v.as_ptr(),
                            span.v.len()
                        ),
                        (
                            other.position,
                            other.k.as_ptr(),
                            other.k.len(),
                            other.v.as_ptr(),
                            other.v.len()
                        )
                    );
                }
                assert_eq!(spans[2].k.as_ptr(), inputs[producer].0.as_ptr());
                assert_eq!(spans[2].v.as_ptr(), inputs[producer].1.as_ptr());
                if layer >= plan.producer_count() {
                    assert!(
                        plan.prepare_producer(
                            layer,
                            &caches,
                            &inputs[producer].0,
                            &inputs[producer].1,
                            n
                        )
                        .is_err()
                    );
                }
                layers_checked += 1;
            }
            assert_eq!(snapshot(&caches), before);
            plan.commit(&mut caches, &frames).unwrap();
            for (producer, cache) in caches.iter().enumerate() {
                assert_eq!(cache.offset(), step["after"].as_u64().unwrap() as usize);
                assert_eq!(cache.generation(), step["index"].as_u64().unwrap() + 1);
                let capacity = if specs[producer].kind == KvKind::Sliding {
                    window
                } else {
                    total
                };
                for slot in 0..capacity {
                    if let Some(position) = cache.position_in_slot(slot) {
                        let dims = specs[producer].head_dim;
                        let k = cache.committed().0;
                        for feature in 0..dims {
                            assert_eq!(
                                k[slot * dims + feature],
                                (producer * 10000 + position) as f32 + feature as f32 / 1024.
                            );
                        }
                    }
                }
                appends += 1;
            }
        }
    }
    assert_eq!((layers_checked, appends), (226, 108));
}

#[test]
fn invalid_plan_geometry_kind_and_size_arithmetic_reject() {
    let a = geometry(KvKind::Sliding);
    let b = geometry(KvKind::Full);
    for (layers, shared, w, cap, max) in [
        (vec![], 0, 3, 8, 12),
        (vec![a], 1, 3, 8, 12),
        (vec![a, b], 1, 3, 8, 12),
        (vec![a], 0, 0, 8, 12),
        (vec![a], 0, 3, 0, 12),
        (vec![a], 0, 13, 8, 12),
        (vec![a], 0, 3, 13, 12),
        (vec![LayerKvGeometry { heads: 0, ..a }], 0, 3, 8, 12),
        (vec![LayerKvGeometry { head_dim: 0, ..a }], 0, 3, 8, 12),
        (vec![a, LayerKvGeometry { head_dim: 2, ..a }], 1, 3, 8, 12),
        (vec![a, LayerKvGeometry { heads: 2, ..a }], 1, 3, 8, 12),
        (
            vec![LayerKvGeometry {
                heads: usize::MAX,
                ..a
            }],
            0,
            3,
            8,
            12,
        ),
    ] {
        assert!(SharedKvPlan::new(&layers, shared, w, cap, max).is_err());
    }
    assert!(SharedKvPlan::planned_metadata_bytes(usize::MAX).is_err());
    assert!(SharedKvPlan::planned_metadata_bytes(0).is_err());
    let plan = small();
    assert!(plan.producer_for(plan.layer_count()).is_err());
}

#[test]
fn every_cache_geometry_field_is_checked_before_preparation_or_view() {
    let plan = small();
    let input = [1.];
    for (last, elements) in [
        (WindowKvCache::new(2, 1, 8, 12, None).unwrap(), 2),
        (WindowKvCache::new(1, 2, 8, 12, None).unwrap(), 2),
        (WindowKvCache::new(1, 1, 7, 12, None).unwrap(), 1),
        (WindowKvCache::new(1, 1, 8, 13, None).unwrap(), 1),
        (WindowKvCache::new(1, 1, 8, 12, Some(8)).unwrap(), 1),
    ] {
        let mut caches = vec![WindowKvCache::new(1, 1, 3, 12, Some(3)).unwrap(), last];
        let last_input = vec![1.; elements];
        let frames = vec![
            caches[0].prepare(&input, &input, 1).unwrap(),
            caches[1].prepare(&last_input, &last_input, 1).unwrap(),
        ];
        // Prepare for an otherwise valid first owner still rejects the bad last geometry.
        let before = snapshot(&caches);
        assert!(
            plan.prepare_producer(0, &caches, &input, &input, 1)
                .is_err()
        );
        assert_eq!(snapshot(&caches), before);
        assert!(plan.view(0, &caches, &frames).is_err());
        assert!(plan.commit(&mut caches, &frames).is_err());
        assert_eq!(snapshot(&caches), before);
    }
}

#[test]
fn missing_reordered_duplicate_foreign_and_stale_frames_preserve_state() {
    let plan = small();
    let mut caches = plan.new_caches().unwrap();
    let input = [1.];
    let frames = proposal(&plan, &caches, &input, 1);
    let before = snapshot(&caches);
    assert!(plan.view(2, &caches, &frames[..1]).is_err());
    assert!(plan.commit(&mut caches, &frames[..1]).is_err());
    let reversed = [
        plan.prepare_producer(1, &caches, &input, &input, 1)
            .unwrap(),
        plan.prepare_producer(0, &caches, &input, &input, 1)
            .unwrap(),
    ];
    assert!(plan.view(2, &caches, &reversed).is_err());
    assert!(plan.published_view(2, &caches, &reversed).is_err());
    assert!(plan.commit(&mut caches, &reversed).is_err());
    let duplicate = [
        plan.prepare_producer(0, &caches, &input, &input, 1)
            .unwrap(),
        plan.prepare_producer(0, &caches, &input, &input, 1)
            .unwrap(),
    ];
    assert!(plan.view(2, &caches, &duplicate).is_err());
    assert!(plan.published_view(2, &caches, &duplicate).is_err());
    assert!(plan.commit(&mut caches, &duplicate).is_err());
    let foreign = plan.new_caches().unwrap();
    let other = proposal(&plan, &foreign, &input, 1);
    assert!(plan.view(2, &caches, &other).is_err());
    assert!(plan.commit(&mut caches, &other).is_err());
    assert_eq!(snapshot(&caches), before);
    plan.commit(&mut caches, &frames).unwrap();
    let after = snapshot(&caches);
    assert!(plan.view(2, &caches, &frames).is_err());
    assert!(plan.commit(&mut caches, &frames).is_err());
    assert_eq!(snapshot(&caches), after);
}

#[test]
fn last_owner_offset_generation_or_chunk_mismatch_rejects_all_views_and_commit() {
    let plan = small();
    let input = [1.];
    let two = [1., 2.];
    for mode in 0..3 {
        let mut caches = plan.new_caches().unwrap();
        if mode == 0 {
            let frame = caches[1].prepare(&input, &input, 1).unwrap();
            caches[1].commit(&frame).unwrap();
        }
        if mode == 1 {
            let first = caches[0].prepare(&two, &two, 2).unwrap();
            caches[0].commit(&first).unwrap();
            for _ in 0..2 {
                let second = caches[1].prepare(&input, &input, 1).unwrap();
                caches[1].commit(&second).unwrap();
            }
        }
        let frames = vec![
            plan.prepare_producer(0, &caches, &input, &input, 1)
                .unwrap(),
            plan.prepare_producer(
                1,
                &caches,
                if mode == 2 { &two } else { &input },
                if mode == 2 { &two } else { &input },
                if mode == 2 { 2 } else { 1 },
            )
            .unwrap(),
        ];
        let before = snapshot(&caches);
        // Selected layer2 uses owner0, but inconsistent owner1 still rejects.
        assert!(plan.view(2, &caches, &frames).is_err());
        if mode != 2 {
            assert!(plan.published_view(2, &caches, &frames[..1]).is_err());
        }
        assert!(plan.published_view(2, &caches, &frames).is_err());
        assert!(plan.commit(&mut caches, &frames).is_err());
        assert_eq!(snapshot(&caches), before);
    }
}

#[test]
fn full_batch_discard_retry_moved_owners_and_global_capacity() {
    let plan = small();
    let caches = plan.new_caches().unwrap();
    let input = [-0., 1., 2., 3.];
    let discarded = proposal(&plan, &caches, &input, 4);
    let before = snapshot(&caches);
    {
        let view = plan.view(2, &caches, &discarded).unwrap();
        assert_eq!(view.start(), 0);
        assert_eq!(view.end(), 4);
    }
    let moved = caches;
    assert_eq!(snapshot(&moved), before);
    assert!(plan.view(2, &moved, &discarded).is_ok());
    let retried = proposal(&plan, &moved, &input, 4);
    let mut caches = moved;
    plan.commit(&mut caches, &retried).unwrap();
    assert_eq!(caches[0].committed().0, [3., 1., 2.]);
    assert_eq!(caches[1].committed().0[..4], input);
    assert!(plan.view(2, &caches, &discarded).is_err());
    let next = proposal(&plan, &caches, &input, 4);
    plan.commit(&mut caches, &next).unwrap();
    let before = snapshot(&caches);
    assert!(
        plan.prepare_producer(1, &caches, &input, &input, 4)
            .is_err()
    );
    assert_eq!(snapshot(&caches), before);
}

#[test]
fn metadata_and_owner_container_accounting_do_not_double_count_objects() {
    let plan = small();
    let mut caches = plan.new_caches().unwrap();
    let planned = plan.planned_cache_bytes().unwrap();
    assert!(plan.allocated_cache_bytes(&caches).unwrap() >= planned);
    assert_eq!(
        SharedKvPlan::planned_metadata_bytes(plan.layer_count()).unwrap(),
        plan.layer_count() * std::mem::size_of::<Route>() + std::mem::size_of::<SharedKvPlan>()
    );
    assert!(
        plan.allocated_metadata_bytes()
            >= SharedKvPlan::planned_metadata_bytes(plan.layer_count()).unwrap()
    );
    caches.reserve(7);
    let expected = caches
        .iter()
        .map(WindowKvCache::allocated_bytes)
        .sum::<usize>()
        + (caches.capacity() - caches.len()) * std::mem::size_of::<WindowKvCache>()
        + std::mem::size_of::<Vec<WindowKvCache>>();
    assert_eq!(plan.allocated_cache_bytes(&caches).unwrap(), expected);
}

#[test]
fn ordered_publication_supports_sequential_execution_and_rejects_late_bad_frames() {
    let plan = small();
    let mut caches = plan.new_caches().unwrap();
    let input = [-0.];
    let two = [1., 2.];
    let before = snapshot(&caches);
    let mut frames = Vec::with_capacity(plan.producer_count());
    assert!(plan.published_view(0, &caches, &frames).is_err());
    frames.push(
        plan.prepare_producer(0, &caches, &input, &input, 1)
            .unwrap(),
    );
    let capacity = frames.capacity();
    {
        let producer = plan.published_view(0, &caches, &frames).unwrap();
        let consumer = plan.published_view(2, &caches, &frames).unwrap();
        assert_eq!(
            producer.head(0).unwrap()[2].k.as_ptr(),
            consumer.head(0).unwrap()[2].k.as_ptr()
        );
        assert_eq!(consumer.head(0).unwrap()[2].k.as_ptr(), input.as_ptr());
    }
    assert!(plan.published_view(1, &caches, &frames).is_err());
    assert!(plan.published_view(3, &caches, &frames).is_err());
    assert!(plan.commit(&mut caches, &frames).is_err());
    assert_eq!(snapshot(&caches), before);
    // A later producer's wrong query length is checked on publication.
    frames.push(plan.prepare_producer(1, &caches, &two, &two, 2).unwrap());
    assert!(plan.published_view(0, &caches, &frames).is_err());
    assert!(plan.commit(&mut caches, &frames).is_err());
    assert_eq!(snapshot(&caches), before);
    frames.pop();
    let foreign = plan.new_caches().unwrap();
    frames.push(
        plan.prepare_producer(1, &foreign, &input, &input, 1)
            .unwrap(),
    );
    assert!(plan.published_view(0, &caches, &frames).is_err());
    assert_eq!(snapshot(&caches), before);
    frames.pop();
    frames.push(
        plan.prepare_producer(1, &caches, &input, &input, 1)
            .unwrap(),
    );
    let too_many = [
        plan.prepare_producer(0, &caches, &input, &input, 1)
            .unwrap(),
        plan.prepare_producer(1, &caches, &input, &input, 1)
            .unwrap(),
        plan.prepare_producer(0, &caches, &input, &input, 1)
            .unwrap(),
    ];
    assert!(plan.published_view(0, &caches, &too_many).is_err());
    assert!(plan.published_view(3, &caches, &frames).is_ok());
    assert_eq!(frames.capacity(), capacity);
    plan.commit(&mut caches, &frames).unwrap();
    let committed = snapshot(&caches);
    let stale = frames.pop().unwrap();
    let late = [
        plan.prepare_producer(0, &caches, &input, &input, 1)
            .unwrap(),
        stale,
    ];
    assert!(plan.published_view(0, &caches, &late).is_err());
    assert!(plan.commit(&mut caches, &late).is_err());
    assert_eq!(snapshot(&caches), committed);
}

mod long_context;
