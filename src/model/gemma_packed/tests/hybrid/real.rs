//! Explicit original E2B graph capture; no runtime/context/app promotion.
use super::*;
use crate::metal::gemma_affine::resident::{E2bWeights, LoadOptions};
use crate::mlx::gemma4::ModalityPolicy;
use hybrid::test_support::resident_snapshot;
use serde_json::json;
use std::{
    fs::OpenOptions,
    os::unix::fs::{MetadataExt, OpenOptionsExt},
};
const IDS: [u32; 4] = [2, 100, 257, 100];
const H: usize = 1536;
const P: usize = 256;
const N: usize = 35;
const HEADS: usize = 8;
const PROCESS: usize = 6 * 1024 * 1024 * 1024;
const DIAGNOSTIC: usize = 128 * 1024 * 1024;
mod attention_boundary;
mod context_probe;
mod loss_block;
mod qat;
mod task_controller;
mod v_boundary;
mod wrap;
fn identity(m: &std::fs::Metadata) -> [u64; 7] {
    [
        m.dev(),
        m.ino(),
        m.len(),
        m.mtime() as u64,
        m.mtime_nsec() as u64,
        m.ctime() as u64,
        m.ctime_nsec() as u64,
    ]
}
fn emit(label: &str, value: serde_json::Value) {
    use std::io::Write;
    println!("GEMMA_HYBRID_REAL_{label} {value}");
    std::io::stdout().flush().unwrap();
}
fn buffers(phase: &str, values: &[serde_json::Value]) {
    for value in values {
        emit("BUFFER", json!({"phase":phase,"value":value}));
    }
}
fn load_original() -> (E2bWeights, std::fs::File, std::path::PathBuf, [u64; 7]) {
    let path = std::path::PathBuf::from(
        std::env::var_os("TOKLET_GEMMA_RESIDENT_MODEL").expect("explicit original model path"),
    );
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(&path)
        .unwrap();
    let metadata = file.metadata().unwrap();
    assert!(metadata.is_file());
    let original = identity(&metadata);
    assert_eq!(
        original,
        identity(&std::fs::symlink_metadata(&path).unwrap())
    );
    let source = autoreleasepool(|_| {
        let context = gemma_affine::Context::new(gemma_affine::CONTEXT_ALLOWANCE_BYTES).unwrap();
        E2bWeights::load(
            &context,
            include_bytes!("../../../../../tests/fixtures/gemma4-e2b-metadata-v1/config.json"),
            include_bytes!("../../../../../tests/fixtures/gemma4-e2b-metadata-v1/index.json"),
            &mut file,
            LoadOptions {
                container_bytes: 3550670554,
                container_sha256: [
                    3, 142, 57, 163, 122, 118, 103, 55, 61, 44, 57, 145, 55, 84, 70, 177, 12, 150,
                    174, 29, 113, 122, 104, 103, 72, 112, 52, 61, 179, 118, 183, 110,
                ],
                available_bytes: PROCESS - DIAGNOSTIC,
                modality: ModalityPolicy::TextOnly,
            },
        )
        .unwrap()
    });
    (source, file, path, original)
}

// Epochs count successful commits/reset operations, not numerical positions.
// Validate them separately before comparing the logical committed prefix.
fn canonical(mut state: Snapshot, vocab: usize) -> Snapshot {
    state.logits = state.logits[state.logits.len() - vocab..].to_vec();
    for cache in &mut state.caches {
        cache.3 = 0;
    }
    state
}

#[test]
#[ignore = "explicit original Gemma E2B hybrid numerical graph capture"]
fn original_e2b_hybrid_graph_preserves_frozen_states_and_resident_owners() {
    let mut capture = capture::Capture::bounded(DIAGNOSTIC);
    let (source, file, path, original) = load_original();
    let before = resident_snapshot(&source);
    buffers("loaded", &before);
    let norms = hybrid::Norms::new(&source, PROCESS - DIAGNOSTIC).unwrap();
    assert!(std::ptr::eq(
        norms.source_identity().unwrap(),
        source.identity()
    ));
    for vector in norms.copied_vectors() {
        emit("CPU_VECTOR", vector);
    }
    let after_copy = resident_snapshot(&source);
    buffers("copied", &after_copy);
    assert_eq!(after_copy, before);
    drop(after_copy);
    let model = norms.model().unwrap();
    assert!(model.content_identity().is_none());
    let context = gemma_affine::Context::new(gemma_affine::CONTEXT_ALLOWANCE_BYTES).unwrap();
    let numerical_budget = PROCESS - DIAGNOSTIC;
    let warm = norms
        .new_session(&model, &context, numerical_budget)
        .unwrap();
    let required = model.model_bytes() + warm.allocated_bytes().unwrap();
    drop(warm);
    let rejected = norms.new_session(&model, &context, required - 1);
    assert!(
        matches!(&rejected, Err(hybrid::Error::Metal(crate::metal::Error::Budget {required:r,limit}))
        if *r == required && *limit == required-1),
        "unexpected admission result: {:?}",
        rejected.err()
    );
    let exact = norms.new_session(&model, &context, required).unwrap();
    assert_eq!(
        model.model_bytes() + exact.allocated_bytes().unwrap(),
        required
    );
    drop(exact);
    emit(
        "ADMISSION",
        json!({"model_bytes":model.model_bytes(),"norm_copy_bytes":norms.owned_bytes(),"raw_admitted_bytes":required,
        "diagnostic_policy_bytes":DIAGNOSTIC,"process_policy_bytes":PROCESS,"numerical_budget":numerical_budget,
        "context_dropped":true,"same_device_fresh_context":true,"exact_admission":true,"minus_one_rejected":true,
        "positions":32,"input_tokens":IDS}),
    );
    for schedule in [&[4][..], &[1, 1, 1, 1][..], &[1, 3][..]] {
        emit("SCHEDULE", json!(schedule));
        let mut session = norms
            .new_session(&model, &context, numerical_budget)
            .unwrap();
        let pointers_before = pointers(&session);
        let allocated = session.allocated_bytes().unwrap();
        let mut offset = 0;
        for &count in schedule {
            let prior = snapshot(&session);
            for layer in 0..N {
                session.fault = Some(layer);
                let result = session.forward(&IDS[offset..offset + count]);
                assert!(
                    result.as_ref().err().is_some_and(|e| e.to_string()
                        == Error::InvalidConfig("injected Gemma layer failure").to_string()),
                    "expected diagnostic boundary at layer {layer}, actual: {result:?}"
                );
                assert_eq!(snapshot(&session), prior);
                assert_eq!(pointers(&session), pointers_before);
                assert_eq!(session.allocated_bytes().unwrap(), allocated);
                let s = &session.scratch;
                let g = &model.layout.layers[layer];
                let dim = g.head_dim;
                for (stage, values) in [
                    ("output", &s.hidden[..count * H]),
                    ("self_attn.o_proj", &s.attention[..count * H]),
                    ("post_per_layer_input_norm", &s.dense_norm[..count * H]),
                    ("per_layer_projection", &s.dense_work[..count * H]),
                    ("per_layer_input_gate", &s.gate[..count * P]),
                    ("mlp.up_proj", &s.up[..count * g.intermediate]),
                    ("self_attn.q_norm", &s.q_norm[..count * HEADS * dim]),
                ] {
                    capture.record(&format!("layer.{layer}.{stage}"), values, offset, count);
                }
                for (stage, values) in [
                    ("q_rotary", &s.q[..count * HEADS * dim]),
                    ("context", &s.context[..count * HEADS * dim]),
                ] {
                    capture.record(
                        &format!("layer.{layer}.{stage}"),
                        &unheads(values, HEADS, count, dim),
                        offset,
                        count,
                    );
                }
            }
            session.fault = None;
            capture.record(
                "logits",
                session.forward(&IDS[offset..offset + count]).unwrap(),
                offset,
                count,
            );
            let s = &session.scratch;
            for (name, values) in [
                ("final_norm", &s.norm[..count * H]),
                ("ple_lookup", &s.ple_lookup[..count * N * P]),
                ("ple_projection", &s.ple_projection[..count * N * P]),
            ] {
                capture.record(name, values, offset, count);
            }
            for owner in 0..15 {
                let dim = model.layout.layers[owner].head_dim;
                let stage = &session.staging[owner];
                for (name, values) in [
                    ("k_rotary", &stage.k[..count * dim]),
                    ("self_attn.v_norm", &stage.v[..count * dim]),
                ] {
                    capture.record(&format!("layer.{owner}.{name}"), values, offset, count);
                }
                let cache = &session.caches[owner];
                let (k, v) = cache.committed();
                for slot in 0..offset + count {
                    assert_eq!(cache.position_in_slot(slot), Some(slot));
                    for (name, values) in [
                        ("k_rotary", &k[slot * dim..(slot + 1) * dim]),
                        ("self_attn.v_norm", &v[slot * dim..(slot + 1) * dim]),
                    ] {
                        capture.record(&format!("layer.{owner}.{name}"), values, slot, 1);
                    }
                }
            }
            offset += count;
            assert_eq!(session.tokens(), &IDS[..offset]);
            assert_eq!(pointers(&session), pointers_before);
            assert_eq!(session.allocated_bytes().unwrap(), allocated);
        }
        assert_eq!(offset, 4);
        session.reset().unwrap();
        assert!(session.tokens().is_empty() && session.logits().is_empty());
        capture.record("logits", session.forward(&IDS).unwrap(), 0, 4);
        assert_eq!(pointers(&session), pointers_before);
        assert_eq!(session.allocated_bytes().unwrap(), allocated);
    }
    let after = resident_snapshot(&source);
    buffers("executed", &after);
    assert_eq!(after, before);
    assert_eq!(norms.copied_vectors().len(), 262);
    assert_eq!(original, identity(&file.metadata().unwrap()));
    assert_eq!(
        original,
        identity(&std::fs::symlink_metadata(&path).unwrap())
    );
    capture.finish();
    emit(
        "COMPLETE",
        json!({"captured":true,"positions":32,"tested_input_positions":4,"source_path_and_descriptor_unchanged":true,
        "resident_buffers_unchanged":true,"full_model_window_qualified":false,"performance_qualified":false,"phone_qualified":false,
        "original_precision_failures_waived":false}),
    );
}

#[test]
#[ignore = "explicit original Gemma E2B hybrid32 state and numerical capture"]
fn original_e2b_hybrid32_boundary_schedules_and_reset() {
    let input = include_bytes!("../../../../../benchmarks/gemma-hybrid32-v1/tokens.u32le");
    let ids: Vec<u32> = input
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    assert_eq!(ids.len(), 32);
    let mut capture = capture::Capture::bounded(DIAGNOSTIC);
    let (source, file, path, original) = load_original();
    let owners = resident_snapshot(&source);
    let norms = hybrid::Norms::new(&source, PROCESS - DIAGNOSTIC).unwrap();
    let model = norms.model().unwrap();
    assert!(model.content_identity().is_none());
    let context = gemma_affine::Context::new(gemma_affine::CONTEXT_ALLOWANCE_BYTES).unwrap();
    let mut expected = None;
    for schedule in [&[32][..], &[31, 1][..], &[1, 15, 16][..]] {
        emit("BOUNDARY_SCHEDULE", json!(schedule));
        let mut session = norms
            .new_session(&model, &context, PROCESS - DIAGNOSTIC)
            .unwrap();
        assert_eq!(session.caches.len(), 15);
        assert_eq!(session.capacity(), 32);
        assert_eq!(session.chunk_capacity(), 32);
        let allocated = session.allocated_bytes().unwrap();
        let pointers_before = pointers(&session);
        let mut offset = 0;
        for (step, &count) in schedule.iter().enumerate() {
            let rows = session.forward(&ids[offset..offset + count]).unwrap();
            for position in [0, 15, 30, 31] {
                if (offset..offset + count).contains(&position) {
                    let start = (position - offset) * 262144;
                    capture.record("boundary.logits", &rows[start..start + 262144], position, 1);
                }
            }
            offset += count;
            assert_eq!(session.tokens(), &ids[..offset]);
            for cache in &session.caches {
                assert_eq!(cache.offset(), offset);
                assert_eq!(cache.generation(), step as u64 + 1);
                for slot in 0..32 {
                    assert_eq!(
                        cache.position_in_slot(slot),
                        (slot < offset).then_some(slot)
                    );
                }
            }
            assert_eq!(pointers(&session), pointers_before);
            assert_eq!(session.allocated_bytes().unwrap(), allocated);
        }
        assert_eq!(session.committed, 32);
        let prior = snapshot(&session);
        let result = session.forward(&[ids[0]]);
        assert!(matches!(
            result,
            Err(hybrid::Error::Model(Error::Capacity {
                offset: 32,
                tokens: 1,
                limit: 32
            }))
        ));
        assert_eq!(snapshot(&session), prior);
        assert_eq!(pointers(&session), pointers_before);
        assert_eq!(session.allocated_bytes().unwrap(), allocated);
        let state = canonical(prior, 262144);
        if let Some(expected) = &expected {
            assert_eq!(&state, expected);
        } else {
            expected = Some(state);
        }
        session.reset().unwrap();
        assert!(session.tokens().is_empty() && session.logits().is_empty());
        for cache in &session.caches {
            assert_eq!(cache.offset(), 0);
            assert_eq!(cache.generation(), schedule.len() as u64 + 1);
            assert!((0..32).all(|slot| cache.position_in_slot(slot).is_none()));
        }
        let rows = session.forward(&ids).unwrap();
        for position in [0, 15, 30, 31] {
            let start = position * 262144;
            capture.record(
                "boundary.reset_logits",
                &rows[start..start + 262144],
                position,
                1,
            );
        }
        for cache in &session.caches {
            assert_eq!(cache.generation(), schedule.len() as u64 + 2);
        }
        assert_eq!(
            canonical(snapshot(&session), 262144),
            *expected.as_ref().unwrap()
        );
        assert_eq!(pointers(&session), pointers_before);
        assert_eq!(session.allocated_bytes().unwrap(), allocated);
        emit(
            "BOUNDARY_STATE",
            json!({"schedule":schedule,"positions":32,"overflow_preserved":true,"reset_reuse":true,"allocated_bytes":allocated}),
        );
    }
    assert_eq!(resident_snapshot(&source), owners);
    assert_eq!(original, identity(&file.metadata().unwrap()));
    assert_eq!(
        original,
        identity(&std::fs::symlink_metadata(path).unwrap())
    );
    capture.finish();
    emit(
        "BOUNDARY_COMPLETE",
        json!({"state_passed":true,"positions":32,"selected_rows":[0,15,30,31],"numerical_decision":"separate reference replay","prior_precision_failures_waived":false,"phone_qualified":false,"app_qualified":false}),
    );
}

#[test]
#[ignore = "explicit E2B diagnostic512 admission and qualified32-prefix parity"]
fn diagnostic512_admission_preserves_default32_prefix() {
    use crate::model::gemma_reference::tests::Measurement;
    use sha2::{Digest, Sha256};
    let input = include_bytes!("../../../../../benchmarks/gemma-hybrid32-v1/tokens.u32le");
    let ids: Vec<u32> = input
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect();
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
    let context = gemma_affine::Context::new(gemma_affine::CONTEXT_ALLOWANCE_BYTES).unwrap();
    let mut session = norms
        .new_session(&ordinary, &context, PROCESS - DIAGNOSTIC)
        .unwrap();
    session.forward(&ids).unwrap();
    let baseline = snapshot(&session);
    let old_allocated = session.allocated_bytes().unwrap();
    for position in [0, 15, 30, 31] {
        let mut hash = Sha256::new();
        let mut raw = [0u8; 4096];
        for part in baseline.logits[position * 262144..(position + 1) * 262144].chunks(1024) {
            for (value, to) in part.iter().zip(raw.chunks_exact_mut(4)) {
                to.copy_from_slice(&value.to_le_bytes());
            }
            hash.update(&raw[..part.len() * 4]);
        }
        emit(
            "PROBE_DEFAULT_ROW",
            json!({"position":position,"sha256":format!("{:x}",hash.finalize())}),
        );
    }
    drop(session);
    let probe = norms.model_for_context_probe(512).unwrap();
    assert_eq!(probe.layout.positions, 512);
    assert_eq!(probe.model_bytes(), ordinary.model_bytes());
    assert!(probe.content_identity().is_none());
    let warm = norms
        .new_session(&probe, &context, PROCESS - DIAGNOSTIC)
        .unwrap();
    assert_eq!(warm.capacity(), 512);
    assert_eq!(warm.chunk_capacity(), 32);
    let allocated = warm.allocated_bytes().unwrap();
    assert_eq!(allocated - old_allocated, 18679680);
    let required = probe.model_bytes() + allocated;
    drop(warm);
    assert!(matches!(norms.new_session(&probe, &context, required - 1),
        Err(hybrid::Error::Metal(crate::metal::Error::Budget {required:r,limit}))
        if r == required && limit == required - 1));
    let mut session = norms.new_session(&probe, &context, required).unwrap();
    assert_eq!(session.allocated_bytes().unwrap(), allocated);
    let pointers_before = pointers(&session);
    for replay in 0..2 {
        if replay == 1 {
            session.reset().unwrap();
        }
        let rows = session.forward(&ids).unwrap();
        assert_eq!(rows.len(), baseline.logits.len());
        assert!(
            rows.iter()
                .zip(&baseline.logits)
                .all(|(value, bits)| value.to_bits() == *bits)
        );
        assert_eq!(session.tokens(), &ids);
        assert_eq!(session.caches.len(), 15);
        for (owner, cache) in session.caches.iter().enumerate() {
            let dim = probe.layout.layers[owner].head_dim;
            let count = 32 * dim; // This exact E2B graph has one KV head per producer.
            let (k, v) = cache.committed();
            let expected = &baseline.caches[owner];
            assert!(
                k[..count]
                    .iter()
                    .zip(&expected.0)
                    .all(|(value, bits)| value.to_bits() == *bits)
            );
            assert!(
                v[..count]
                    .iter()
                    .zip(&expected.1)
                    .all(|(value, bits)| value.to_bits() == *bits)
            );
            assert_eq!(expected.0.len(), count);
            assert_eq!(expected.1.len(), count);
            assert_eq!(cache.offset(), 32);
            assert_eq!(cache.generation(), if replay == 0 { 1 } else { 3 });
            assert_eq!(k.len(), 512 * dim);
            assert_eq!(v.len(), 512 * dim);
            for slot in 0..512 {
                assert_eq!(cache.position_in_slot(slot), (slot < 32).then_some(slot));
            }
        }
        assert_eq!(pointers(&session), pointers_before);
        assert_eq!(session.allocated_bytes().unwrap(), allocated);
    }
    drop(session);
    assert_eq!(resident_snapshot(&source), owners);
    assert_eq!(original, identity(&file.metadata().unwrap()));
    assert_eq!(
        original,
        identity(&std::fs::symlink_metadata(path).unwrap())
    );
    emit(
        "PROBE_COMPLETE",
        json!({"default_positions":32,"diagnostic_positions":512,"chunk_rows":32,
        "tested_prefix_positions":32,"all32_logit_rows_exact":true,"all15_valid_kv_prefixes_exact":true,
        "default_allocated_bytes":old_allocated,"candidate_allocated_bytes":allocated,
        "required_bytes":required,"exact_admission":true,"minus_one_rejected":true,"reset_reuse":true,
        "positions_above31_qualified":false,"app_qualified":false,"phone_qualified":false}),
    );
}

#[test]
#[ignore = "explicit real E2B diagnostic512 state and numerical capture"]
fn diagnostic512_boundary_schedules_and_reset() {
    use sha2::{Digest, Sha256};
    fn hash(values: &[f32]) -> [u8; 32] {
        let mut hash = Sha256::new();
        let mut raw = [0u8; 4096];
        for part in values.chunks(1024) {
            for (value, to) in part.iter().zip(raw.chunks_exact_mut(4)) {
                to.copy_from_slice(&value.to_le_bytes());
            }
            hash.update(&raw[..part.len() * 4]);
        }
        hash.finalize().into()
    }
    let raw = include_bytes!("../../../../../benchmarks/gemma-e2b-context-v1/tokens.u32le");
    let ids: Vec<u32> = raw
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    assert_eq!(ids.len(), 512);
    let mut second = vec![1];
    second.extend([31; 16]);
    second.push(15);
    let schedules = [vec![32; 16], second];
    let mut capture = capture::Capture::bounded(DIAGNOSTIC);
    let (source, file, path, original) = load_original();
    let owners = resident_snapshot(&source);
    let norms = hybrid::Norms::new(&source, PROCESS - DIAGNOSTIC).unwrap();
    let model = norms.model_for_context_probe(512).unwrap();
    assert!(model.content_identity().is_none());
    let context = gemma_affine::Context::new(gemma_affine::CONTEXT_ALLOWANCE_BYTES).unwrap();
    let mut expected = None;
    let mut prefix_logits = None;
    let mut prefix_caches: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
    for schedule in schedules {
        emit("CONTEXT512_SCHEDULE", json!(schedule));
        let mut session = norms
            .new_session(&model, &context, PROCESS - DIAGNOSTIC)
            .unwrap();
        assert_eq!(session.capacity(), 512);
        assert_eq!(session.chunk_capacity(), 32);
        assert_eq!(session.caches.len(), 15);
        let pointers_before = pointers(&session);
        let allocated = session.allocated_bytes().unwrap();
        assert_eq!(allocated, 263761128);
        let mut offset = 0;
        for (step, &count) in schedule.iter().enumerate() {
            let rows = session.forward(&ids[offset..offset + count]).unwrap();
            if prefix_logits.is_none() {
                assert_eq!((offset, count), (0, 32));
                prefix_logits = Some(hash(rows));
                capture.record("context512.initial_logits", rows, 0, 32);
            }
            for position in [0, 31, 255, 511] {
                if (offset..offset + count).contains(&position) {
                    let start = (position - offset) * 262144;
                    capture.record(
                        "context512.logits",
                        &rows[start..start + 262144],
                        position,
                        1,
                    );
                }
            }
            offset += count;
            if prefix_caches.is_empty() {
                assert_eq!(offset, 32);
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
            assert_eq!(session.tokens(), &ids[..offset]);
            for (owner, cache) in session.caches.iter().enumerate() {
                let dim = model.layout.layers[owner].head_dim;
                let (k, v) = cache.committed();
                assert_eq!((k.len(), v.len()), (512 * dim, 512 * dim));
                assert_eq!(cache.offset(), offset);
                assert_eq!(cache.generation(), step as u64 + 1);
                for slot in 0..512 {
                    assert_eq!(
                        cache.position_in_slot(slot),
                        (slot < offset).then_some(slot)
                    );
                }
            }
            assert_eq!(pointers(&session), pointers_before);
            assert_eq!(session.allocated_bytes().unwrap(), allocated);
        }
        assert_eq!(offset, 512);
        let before = snapshot(&session);
        assert!(matches!(
            session.forward(&[ids[0]]),
            Err(hybrid::Error::Model(Error::Capacity {
                offset: 512,
                tokens: 1,
                limit: 512
            }))
        ));
        assert_eq!(snapshot(&session), before);
        assert_eq!(pointers(&session), pointers_before);
        assert_eq!(session.allocated_bytes().unwrap(), allocated);
        let state = canonical(before, 262144);
        if let Some(expected) = &expected {
            assert_eq!(&state, expected);
        } else {
            expected = Some(state);
        }
        session.reset().unwrap();
        assert!(session.tokens().is_empty() && session.logits().is_empty());
        for cache in &session.caches {
            assert_eq!(cache.offset(), 0);
            assert!((0..512).all(|slot| cache.position_in_slot(slot).is_none()));
        }
        let rows = session.forward(&ids[..32]).unwrap();
        assert_eq!(Some(hash(rows)), prefix_logits);
        capture.record("context512.reset_logits", rows, 0, 32);
        assert_eq!(session.tokens(), &ids[..32]);
        for (owner, cache) in session.caches.iter().enumerate() {
            let count = 32 * model.layout.layers[owner].head_dim;
            let (k, v) = cache.committed();
            let previous = &prefix_caches[owner];
            assert!(
                k[..count]
                    .iter()
                    .zip(&previous.0)
                    .all(|(value, bits)| value.to_bits() == *bits)
            );
            assert!(
                v[..count]
                    .iter()
                    .zip(&previous.1)
                    .all(|(value, bits)| value.to_bits() == *bits)
            );
            assert_eq!(cache.offset(), 32);
            assert_eq!(cache.generation(), schedule.len() as u64 + 2);
            for slot in 0..512 {
                assert_eq!(cache.position_in_slot(slot), (slot < 32).then_some(slot));
            }
        }
        assert_eq!(pointers(&session), pointers_before);
        assert_eq!(session.allocated_bytes().unwrap(), allocated);
        emit(
            "CONTEXT512_STATE",
            json!({"schedule":schedule,"committed_before_rejection":512,"rejected_absolute_position":512,
            "overflow_preserved":true,"reset_prefix_tokens":32,"reset_all32_rows_exact":true,
            "all15_valid_kv_prefixes_exact":true,"allocated_bytes":allocated}),
        );
    }
    assert_eq!(resident_snapshot(&source), owners);
    assert_eq!(original, identity(&file.metadata().unwrap()));
    assert_eq!(
        original,
        identity(&std::fs::symlink_metadata(path).unwrap())
    );
    capture.finish();
    emit(
        "CONTEXT512_COMPLETE",
        json!({"state_passed":true,"positions":512,"selected_rows":[0,31,255,511],
        "prior_precision_failures_waived":false,"app_qualified":false,"phone_qualified":false}),
    );
}
