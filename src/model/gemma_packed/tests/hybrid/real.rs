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
#[test]
#[ignore = "explicit original Gemma E2B hybrid numerical graph capture"]
fn original_e2b_hybrid_graph_preserves_frozen_states_and_resident_owners() {
    let mut capture = capture::Capture::bounded(DIAGNOSTIC);
    let path =
        std::env::var_os("TOKLET_GEMMA_RESIDENT_MODEL").expect("explicit original model path");
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
