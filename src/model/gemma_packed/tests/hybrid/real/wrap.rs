//! Explicit real local-window eviction capture; numerical decisions are external.
use super::*;
use sha2::{Digest, Sha256};

fn hash(values: &[f32]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

#[test]
#[ignore = "explicit real E2B diagnostic1025 local eviction/wrap capture"]
fn diagnostic1025_wrap_schedules_overflow_and_reset() {
    let raw = include_bytes!("../../../../../../benchmarks/gemma-e2b-wrap-v1/tokens.u32le");
    let ids: Vec<u32> = raw
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    assert_eq!(ids.len(), 1025);
    let mut first = vec![32; 16];
    first.push(1);
    first.extend([32; 15]);
    first.extend([31, 1]);
    let mut second = vec![32; 8];
    second.extend([31; 8]);
    second.push(32); // 504 -> 536 crosses the first eviction in one call.
    second.extend([32; 15]);
    second.push(9); // 1016 -> 1025 crosses the second wrap in one call.
    assert_eq!(first.iter().sum::<usize>(), 1025);
    assert_eq!(second.iter().sum::<usize>(), 1025);
    let selected = [0, 31, 255, 511, 512, 767, 1023, 1024];
    let mut capture = capture::Capture::bounded(DIAGNOSTIC);
    let (source, file, path, original) = load_original();
    let owners = resident_snapshot(&source);
    let norms = hybrid::Norms::new(&source, PROCESS - DIAGNOSTIC).unwrap();
    let model = norms.model_for_context_probe(1025).unwrap();
    assert!(model.content_identity().is_none());
    let context = gemma_affine::Context::new(gemma_affine::CONTEXT_ALLOWANCE_BYTES).unwrap();
    let mut expected256 = None;
    let mut expected1025 = None;
    let mut prefix_hash = None;
    let mut prefix_caches: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
    for (schedule_index, schedule) in [first, second].into_iter().enumerate() {
        emit(
            "WRAP_SCHEDULE",
            json!({"index":schedule_index,"chunks":schedule}),
        );
        let mut session = norms
            .new_session(&model, &context, PROCESS - DIAGNOSTIC)
            .unwrap();
        assert_eq!((session.capacity(), session.chunk_capacity()), (1025, 32));
        let allocated = session.allocated_bytes().unwrap();
        assert_eq!(allocated, 271117548);
        let pointers_before = pointers(&session);
        let mut offset = 0;
        for (step, &count) in schedule.iter().enumerate() {
            let rows = session.forward(&ids[offset..offset + count]).unwrap();
            if offset == 0 {
                assert_eq!(count, 32);
                let actual = hash(rows);
                if let Some(expected) = &prefix_hash {
                    assert_eq!(&actual, expected);
                } else {
                    prefix_hash = Some(actual.clone());
                }
                emit(
                    "WRAP_PREFIX",
                    json!({"schedule":schedule_index,"sha256":actual}),
                );
            }
            for position in selected {
                if (offset..offset + count).contains(&position) {
                    let start = (position - offset) * 262144;
                    capture.record("wrap.logits", &rows[start..start + 262144], position, 1);
                }
            }
            offset += count;
            assert_eq!(session.tokens(), &ids[..offset]);
            assert_eq!(session.caches.len(), 15);
            for (owner, cache) in session.caches.iter().enumerate() {
                let layer = &model.layout.layers[owner];
                let capacity = if layer.kind == KvKind::Full {
                    1025
                } else {
                    512
                };
                let (k, v) = cache.committed();
                assert_eq!(
                    (k.len(), v.len()),
                    (capacity * layer.head_dim, capacity * layer.head_dim)
                );
                assert_eq!(
                    (cache.offset(), cache.generation()),
                    (offset, step as u64 + 1)
                );
                let start = offset.saturating_sub(capacity);
                for slot in 0..capacity {
                    let position = start + (slot + capacity - start % capacity) % capacity;
                    assert_eq!(
                        cache.position_in_slot(slot),
                        (position < offset).then_some(position)
                    );
                }
                if schedule_index == 0 && [512, 513, 1024, 1025].contains(&offset) {
                    let slots: Vec<_> = (0..capacity)
                        .map(|slot| cache.position_in_slot(slot))
                        .collect();
                    emit(
                        "WRAP_SLOTS",
                        json!({"committed":offset,"owner":owner,"slots":slots}),
                    );
                    for (kind, values) in [("k", k), ("v", v)] {
                        let mut chronological =
                            Vec::with_capacity((offset - start) * layer.head_dim);
                        for position in start..offset {
                            let slot = position % capacity;
                            let begin = slot * layer.head_dim;
                            chronological.extend_from_slice(&values[begin..begin + layer.head_dim]);
                        }
                        capture.record(
                            &format!("wrap.layer.{owner}.{kind}"),
                            &chronological,
                            offset,
                            offset - start,
                        );
                    }
                }
            }
            if offset == 32 && prefix_caches.is_empty() {
                prefix_caches = session
                    .caches
                    .iter()
                    .enumerate()
                    .map(|(owner, cache)| {
                        let count = 32 * model.layout.layers[owner].head_dim;
                        let (k, v) = cache.committed();
                        (
                            k[..count].iter().map(|v| v.to_bits()).collect(),
                            v[..count].iter().map(|v| v.to_bits()).collect(),
                        )
                    })
                    .collect();
            }
            if offset == 256 {
                let state = canonical(snapshot(&session), 262144);
                if let Some(expected) = &expected256 {
                    assert_eq!(&state, expected);
                } else {
                    expected256 = Some(state);
                }
            }
            assert_eq!(pointers(&session), pointers_before);
            assert_eq!(session.allocated_bytes().unwrap(), allocated);
        }
        assert_eq!(offset, 1025);
        let before = snapshot(&session);
        assert!(matches!(
            session.forward(&[ids[0]]),
            Err(hybrid::Error::Model(Error::Capacity {
                offset: 1025,
                tokens: 1,
                limit: 1025
            }))
        ));
        assert_eq!(snapshot(&session), before);
        let final_state = canonical(before, 262144);
        if let Some(expected) = &expected1025 {
            assert_eq!(&final_state, expected);
        } else {
            expected1025 = Some(final_state);
        }
        session.reset().unwrap();
        assert!(session.tokens().is_empty() && session.logits().is_empty());
        for (owner, cache) in session.caches.iter().enumerate() {
            let capacity = cache.committed().0.len() / model.layout.layers[owner].head_dim;
            assert_eq!(
                (cache.offset(), cache.generation()),
                (0, schedule.len() as u64 + 1)
            );
            assert!((0..capacity).all(|slot| cache.position_in_slot(slot).is_none()));
        }
        assert_eq!(
            Some(hash(session.forward(&ids[..32]).unwrap())),
            prefix_hash
        );
        assert_eq!(session.tokens(), &ids[..32]);
        for (owner, cache) in session.caches.iter().enumerate() {
            let dim = model.layout.layers[owner].head_dim;
            let (k, v) = cache.committed();
            let expected = &prefix_caches[owner];
            for (actual, prior) in [(k, &expected.0), (v, &expected.1)] {
                assert!(
                    actual[..32 * dim]
                        .iter()
                        .zip(prior)
                        .all(|(a, b)| a.to_bits() == *b)
                );
            }
            assert_eq!(
                (cache.offset(), cache.generation()),
                (32, schedule.len() as u64 + 2)
            );
            for slot in 0..k.len() / dim {
                assert_eq!(cache.position_in_slot(slot), (slot < 32).then_some(slot));
            }
        }
        assert_eq!(pointers(&session), pointers_before);
        assert_eq!(session.allocated_bytes().unwrap(), allocated);
        emit(
            "WRAP_STATE",
            json!({"schedule":schedule_index,"committed":1025,
            "overflow_preserved":true,"reset_all32_rows_exact":true,"reset_all15_valid_kv_prefixes_exact":true,
            "allocated_bytes":allocated,"cross_schedule_endpoints":[256,1025]}),
        );
    }
    assert!(expected256.is_some() && expected1025.is_some());
    assert_eq!(resident_snapshot(&source), owners);
    assert_eq!(original, identity(&file.metadata().unwrap()));
    assert_eq!(
        original,
        identity(&std::fs::symlink_metadata(path).unwrap())
    );
    capture.finish();
    emit(
        "WRAP_COMPLETE",
        json!({"state_passed":true,"numerical_decision":"separate reference replay",
        "positions":1025,"selected_positions":selected,"retained_milestones":[512,513,1024,1025],
        "prior_precision_failures_waived":false,"app_qualified":false,"phone_qualified":false}),
    );
}

#[test]
#[ignore = "Explicit original E2B final head same-input diagnostic at position 1024"]
fn diagnostic1025_same_input_final_head() {
    let raw = include_bytes!("../../../../../../benchmarks/gemma-e2b-wrap-v1/tokens.u32le");
    let ids: Vec<u32> = raw
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    assert_eq!(ids.len(), 1025);
    let (source, file, path, original) = load_original();
    let owners = resident_snapshot(&source);
    let norms = hybrid::Norms::new(&source, PROCESS - DIAGNOSTIC).unwrap();
    let model = norms.model_for_context_probe(1025).unwrap();
    let context = gemma_affine::Context::new(gemma_affine::CONTEXT_ALLOWANCE_BYTES).unwrap();
    let mut session = norms
        .new_session(&model, &context, PROCESS - DIAGNOSTIC)
        .unwrap();
    assert_eq!((session.capacity(), session.chunk_capacity()), (1025, 32));
    let allocated = session.allocated_bytes().unwrap();
    let mut offset = 0;
    for count in [32; 16].into_iter().chain([1]).chain([32; 15]).chain([31]) {
        session.forward(&ids[offset..offset + count]).unwrap();
        offset += count;
    }
    assert_eq!(offset, 1024);
    let row = session.forward(&ids[offset..]).unwrap();
    assert_eq!(row.len(), 262144);
    emit(
        "FINAL_HEAD_ROW",
        json!({"position":1024,"sha256":hash(row),"allocated_bytes":allocated}),
    );
    assert_eq!(session.tokens(), ids);
    let mut capture = capture::Capture::bounded(H * 4);
    capture.record("head.normalized", &session.scratch.norm[..H], 1024, 1);
    capture.finish();
    assert_eq!(resident_snapshot(&source), owners);
    assert_eq!(original, identity(&file.metadata().unwrap()));
    assert_eq!(
        original,
        identity(&std::fs::symlink_metadata(path).unwrap())
    );
}
