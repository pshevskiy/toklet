//! Explicit real official BF16 diagnostic and private-scratch publication checks.
use super::*;
use crate::metal::gemma_affine::{Bf16Context, Bf16GraphFault, Bf16Weights};

const VOCAB: usize = 262144;
const BUDGET: usize = PROCESS - DIAGNOSTIC;

pub(super) fn load_official() -> (Bf16Weights, Bf16Context, std::path::PathBuf, [u64; 7]) {
    let root = std::path::PathBuf::from(std::env::var_os("TOKLET_GEMMA_BF16_MODEL").unwrap());
    let path = root.join("model.safetensors");
    let config = std::fs::read(root.join("config.json")).unwrap();
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(&path)
        .unwrap();
    let original = identity(&file.metadata().unwrap());
    let context = Bf16Context::new(gemma_affine::CONTEXT_ALLOWANCE_BYTES).unwrap();
    let weights = Bf16Weights::load_text_only(&context, file, &config, BUDGET).unwrap();
    (weights, context, path, original)
}

#[test]
#[ignore = "explicit official BF16 short Metal graph and rollback witness"]
fn official_bf16_short_graph_preserves_state_and_causal_rows() {
    let (weights, context, path, original) = load_official();
    assert!(weights.model(0).is_err());
    assert!(weights.model(33).is_err());
    let model = weights.model(32).unwrap();
    assert!(model.content_identity().is_none());
    let mut capture = capture::Capture::bounded(16 * 1024 * 1024);
    let mut causal: Option<Vec<f32>> = None;
    for (label, ids) in [("original", IDS), ("future", [2, 100, 257, 101])] {
        let mut expected = None;
        for chunk in [32, 3, 1] {
            let warm = weights
                .new_session(&model, &context, chunk, BUDGET)
                .unwrap();
            let owned = warm.allocated_bytes().unwrap();
            let required = model.model_bytes() + owned;
            drop(warm);
            assert!(matches!(
                weights.new_session(&model, &context, chunk, required - 1),
                Err(hybrid::Error::Metal(crate::metal::Error::Budget { required: r, limit }))
                    if r == required && limit == required - 1
            ));
            let mut session = weights
                .new_session(&model, &context, chunk, required)
                .unwrap();
            let pointers_before = pointers(&session);
            let mut rows = Vec::with_capacity(ids.len() * VOCAB);
            for input in ids.chunks(chunk) {
                rows.extend_from_slice(session.forward(input).unwrap());
            }
            assert_eq!(session.tokens(), ids);
            assert_eq!(session.allocated_bytes().unwrap(), owned);
            assert_eq!(pointers(&session), pointers_before);
            let state = canonical(snapshot(&session), VOCAB);
            if let Some((want_rows, want_state)) = &expected {
                assert!(&rows == want_rows, "full logits changed by chunk schedule");
                assert!(
                    &state == want_state,
                    "committed state changed by chunk schedule"
                );
            } else {
                capture.record(&format!("official.{label}.logits"), &rows, 0, ids.len());
                for (owner, cache) in session.caches.iter().enumerate() {
                    let count = ids.len() * model.layout.layers[owner].head_dim;
                    let (keys, values) = cache.committed();
                    for (tag, part) in [
                        ("k_rotary", &keys[..count]),
                        ("self_attn.v_norm", &values[..count]),
                    ] {
                        capture.record(
                            &format!("official.{label}.layer.{owner}.{tag}"),
                            part,
                            0,
                            ids.len(),
                        );
                    }
                }
                expected = Some((rows, state));
            }
            session.reset().unwrap();
            assert!(session.tokens().is_empty() && session.logits().is_empty());
            assert_eq!(pointers(&session), pointers_before);
            emit(
                "BF16_GRAPH_CHUNK",
                json!({"sequence":label,"chunk":chunk,"required_bytes":required,"session_bytes":owned}),
            );
        }
        let (rows, _) = expected.unwrap();
        if let Some(prior) = &causal {
            assert!(
                &rows[..3 * VOCAB] == prior.as_slice(),
                "future input changed causal rows"
            );
        } else {
            causal = Some(rows[..3 * VOCAB].to_vec());
        }
    }
    for (fault, label) in [
        (Bf16GraphFault::AfterFirstHeadTile, "head_tile"),
        (Bf16GraphFault::AfterPle, "ple"),
    ] {
        let mut session = weights.new_session(&model, &context, 3, BUDGET).unwrap();
        session.forward(&[2]).unwrap();
        let before = snapshot(&session);
        session.operations.set_fault(fault);
        assert!(session.forward(&[100]).is_err());
        assert_eq!(snapshot(&session), before);
        assert!(session.reset().is_err());
        assert_eq!(snapshot(&session), before);
        emit(
            "BF16_GRAPH_FAILURE",
            json!({"after":label,"committed_state_preserved":true,"recreate":true}),
        );
    }
    assert_eq!(
        identity(&std::fs::symlink_metadata(&path).unwrap()),
        original
    );
    capture.finish();
    emit(
        "BF16_GRAPH_COMPLETE",
        json!({"committed_state_passed":true,"positions":4,"sequences":2,"app_qualified":false}),
    );
}
