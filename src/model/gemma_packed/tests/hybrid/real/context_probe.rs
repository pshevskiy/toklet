//! Admission and prefix parity only: no execution past the qualified 512 inputs.
use super::*;
use crate::model::gemma_reference::tests::Measurement;
use sha2::{Digest, Sha256};

fn hash(values: &[f32]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

#[test]
#[ignore = "explicit E2B diagnostic1025 admission and qualified512-prefix parity"]
fn diagnostic1025_admission_preserves_qualified512_prefix() {
    let raw = include_bytes!("../../../../../../benchmarks/gemma-e2b-context-v1/tokens.u32le");
    let ids: Vec<u32> = raw
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    assert_eq!(ids.len(), 512);
    let (source, file, path, original) = load_original();
    let owners = resident_snapshot(&source);
    let norms = hybrid::Norms::new(&source, PROCESS - DIAGNOSTIC).unwrap();
    for limit in [0, 1026, usize::MAX] {
        let measured = Measurement::start();
        let rejected = norms.model_for_context_probe(limit);
        let allocations = measured.finish();
        assert!(matches!(
            rejected,
            Err(hybrid::Error::Model(Error::InvalidConfig(
                "Gemma E2B diagnostic position limit"
            )))
        ));
        assert_eq!(allocations, 0);
    }
    let ordinary = norms.model().unwrap();
    assert_eq!(ordinary.layout.positions, 32);
    let baseline_model = norms.model_for_context_probe(512).unwrap();
    let context = gemma_affine::Context::new(gemma_affine::CONTEXT_ALLOWANCE_BYTES).unwrap();
    let mut baseline = norms
        .new_session(&baseline_model, &context, PROCESS - DIAGNOSTIC)
        .unwrap();
    let mut row_blocks = Vec::new();
    for (chunk, input) in ids.chunks(32).enumerate() {
        let rows = baseline.forward(input).unwrap();
        row_blocks.push(hash(rows));
        for position in [0, 31, 255, 511] {
            if (chunk * 32..(chunk + 1) * 32).contains(&position) {
                let start = (position % 32) * 262144;
                emit(
                    "PROBE1025_BASELINE_ROW",
                    json!({"position":position,"sha256":hash(&rows[start..start+262144])}),
                );
            }
        }
    }
    let state = snapshot(&baseline);
    let old_allocated = baseline.allocated_bytes().unwrap();
    assert_eq!(old_allocated, 263761128);
    drop(baseline);

    let model = norms.model_for_context_probe(1025).unwrap();
    assert_eq!(model.layout.positions, 1025);
    assert_eq!(model.model_bytes(), ordinary.model_bytes());
    assert!(model.content_identity().is_none());
    let warm = norms
        .new_session(&model, &context, PROCESS - DIAGNOSTIC)
        .unwrap();
    let allocated = warm.allocated_bytes().unwrap();
    assert_eq!(allocated - old_allocated, 7356420);
    let required = model.model_bytes() + allocated;
    drop(warm);
    assert!(matches!(norms.new_session(&model, &context, required - 1),
        Err(hybrid::Error::Metal(crate::metal::Error::Budget {required:r,limit}))
        if r == required && limit == required - 1));
    let mut session = norms.new_session(&model, &context, required).unwrap();
    assert_eq!((session.capacity(), session.chunk_capacity()), (1025, 32));
    let pointers_before = pointers(&session);
    for (chunk, input) in ids.chunks(32).enumerate() {
        assert_eq!(hash(session.forward(input).unwrap()), row_blocks[chunk]);
        assert_eq!(session.tokens(), &ids[..(chunk + 1) * 32]);
        assert_eq!(session.allocated_bytes().unwrap(), allocated);
        assert_eq!(pointers(&session), pointers_before);
    }
    assert_eq!(session.caches.len(), 15);
    for (owner, cache) in session.caches.iter().enumerate() {
        let layout = &model.layout.layers[owner];
        let capacity = if layout.kind == KvKind::Full {
            1025
        } else {
            512
        };
        let (k, v) = cache.committed();
        let expected = &state.caches[owner];
        assert_eq!(
            (k.len(), v.len()),
            (capacity * layout.head_dim, capacity * layout.head_dim)
        );
        assert_eq!((cache.offset(), cache.generation()), (512, 16));
        for (actual, previous) in [(k, &expected.0), (v, &expected.1)] {
            assert_eq!(previous.len(), 512 * layout.head_dim);
            assert!(
                actual[..previous.len()]
                    .iter()
                    .zip(previous)
                    .all(|(a, b)| a.to_bits() == *b)
            );
        }
        for slot in 0..capacity {
            assert_eq!(cache.position_in_slot(slot), (slot < 512).then_some(slot));
        }
    }
    session.reset().unwrap();
    assert!(session.tokens().is_empty() && session.logits().is_empty());
    for cache in &session.caches {
        assert_eq!((cache.offset(), cache.generation()), (0, 17));
    }
    for (owner, cache) in session.caches.iter().enumerate() {
        let capacity = cache.committed().0.len() / model.layout.layers[owner].head_dim;
        assert!((0..capacity).all(|slot| cache.position_in_slot(slot).is_none()));
    }
    assert_eq!(hash(session.forward(&ids[..32]).unwrap()), row_blocks[0]);
    assert_eq!(session.tokens(), &ids[..32]);
    for (owner, cache) in session.caches.iter().enumerate() {
        let dim = model.layout.layers[owner].head_dim;
        let (k, v) = cache.committed();
        let previous = &state.caches[owner];
        for (actual, expected) in [(k, &previous.0), (v, &previous.1)] {
            assert!(
                actual[..32 * dim]
                    .iter()
                    .zip(&expected[..32 * dim])
                    .all(|(a, b)| a.to_bits() == *b)
            );
        }
        assert_eq!((cache.offset(), cache.generation()), (32, 18));
        for slot in 0..k.len() / dim {
            assert_eq!(cache.position_in_slot(slot), (slot < 32).then_some(slot));
        }
    }
    assert_eq!(session.allocated_bytes().unwrap(), allocated);
    assert_eq!(pointers(&session), pointers_before);
    assert_eq!(resident_snapshot(&source), owners);
    assert_eq!(original, identity(&file.metadata().unwrap()));
    assert_eq!(
        original,
        identity(&std::fs::symlink_metadata(path).unwrap())
    );
    emit(
        "PROBE1025_COMPLETE",
        json!({
            "default_positions":32,"diagnostic_positions":1025,"chunk_rows":32,
            "tested_prefix_positions":512,"all512_logit_rows_exact":true,"all15_valid_kv_prefixes_exact":true,
            "baseline_allocated_bytes":old_allocated,"candidate_allocated_bytes":allocated,
            "required_bytes":required,"exact_admission":true,"minus_one_rejected":true,"reset_reuse":true,
            "positions_above511_qualified":false,"app_qualified":false,"phone_qualified":false
        }),
    );
}
