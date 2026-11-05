use crate::{
    generation::{self, Input, Request, StopReason},
    metal::{self, AttentionMode, NormalizationMode, ProjectionMode, ReadbackMode},
    session::Options,
    weights::LoadOptions,
};
use sha2::{Digest, Sha256};
use std::{path::Path, sync::atomic::AtomicBool};

const BUDGET: usize = 2560 * 1024 * 1024;

fn row_hash(session: &metal::Session<'_>) -> String {
    let logits = session.logits().unwrap();
    let mut hash = Sha256::new();
    for value in logits {
        assert!(value.is_finite());
        hash.update(value.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

#[test]
#[ignore = "explicit audited Llama staged Mac512 session, admission and lifecycle"]
fn staged_llama_mac512_session_replays_isolated_rows() {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("models/MLX-Llama-3.2-1B-Instruct-8bit-d48cdf0");
    let model = metal::Model::load_mlx_llama3_audited(
        &path,
        LoadOptions {
            context_limit: 512,
            memory_limit_bytes: BUDGET,
        },
    )
    .unwrap();
    let options = Options {
        capacity: 512,
        position_offset: 0,
        prefill_chunk_size: 32,
        memory_limit_bytes: BUDGET,
    };
    let open = |options| {
        metal::Session::with_attention_mode(
            &model,
            options,
            ProjectionMode::MlxAffine,
            NormalizationMode::Compensated,
            ReadbackMode::FullLogits,
            AttentionMode::Reference,
        )
        .unwrap()
    };
    let mut session = open(options);
    let baseline = session.memory();
    assert!(!session.llama3_staged_simd_enabled());
    session.enable_llama3_staged_simd().unwrap();
    assert!(session.llama3_staged_simd_enabled());
    assert!(session.native_owned_memory().is_err());
    assert_eq!(session.scratch.scores.len, 32 * 32 * 512 + 7);
    assert_eq!(session.scratch.probs.len, session.scratch.scores.len);
    let required = session.memory().admitted_bytes;
    assert_eq!(required - baseline.admitted_bytes, 64 * 1024 * 1024);
    assert_eq!(session.memory().scratch_bytes, baseline.scratch_bytes);
    assert_eq!(session.memory().kv_bytes, baseline.kv_bytes);
    assert!(session.enable_llama3_staged_simd().is_err());
    assert!(session.enable_affine_prefill_tile4().is_err());
    drop(session);

    let mut exact = open(Options {
        memory_limit_bytes: required,
        ..options
    });
    exact.enable_llama3_staged_simd().unwrap();
    assert_eq!(exact.memory().admitted_bytes, required);
    drop(exact);
    let mut low = open(Options {
        memory_limit_bytes: required - 1,
        ..options
    });
    assert!(
        matches!(low.enable_llama3_staged_simd(), Err(metal::Error::Budget { required: bytes, limit })
        if bytes == required && limit + 1 == required)
    );
    assert_eq!(low.memory().admitted_bytes, baseline.admitted_bytes);
    drop(low);
    let mut smaller = open(Options {
        capacity: 511,
        ..options
    });
    assert!(smaller.enable_llama3_staged_simd().is_err());
    drop(smaller);

    let tokens: Vec<_> =
        include_bytes!("../../../benchmarks/llama3-original-quality-v1/tokens.u32le")
            .chunks_exact(4)
            .take(512)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect();
    let mut session = open(options);
    session.enable_llama3_staged_simd().unwrap();
    session.enable_profile().unwrap();
    session.set_tokens(&tokens[..128]).unwrap();
    session.evaluate().unwrap();
    let initial = row_hash(&session);
    let next = session.greedy_token().unwrap().unwrap();
    let cancel = AtomicBool::new(true);
    let stopped = generation::generate(
        &mut session,
        Request {
            input: Input::Continue,
            max_new_tokens: 1,
            stop_tokens: &[],
        },
        &cancel,
        |_| panic!("cancelled generation emitted"),
    )
    .unwrap();
    assert_eq!(stopped.reason, StopReason::Cancelled);
    assert_eq!(row_hash(&session), initial);
    let stop = generation::generate(
        &mut session,
        Request {
            input: Input::Continue,
            max_new_tokens: 1,
            stop_tokens: &[next],
        },
        &AtomicBool::new(false),
        |_| panic!("hidden stop emitted"),
    )
    .unwrap();
    assert_eq!(stop.reason, StopReason::StopToken { token_id: next });
    assert_eq!(session.pending_len(), 1);
    session.discard_pending();
    assert_eq!(row_hash(&session), initial);
    for &token in &tokens[128..256] {
        session.append(&[token]).unwrap();
        assert_eq!(session.evaluate_next_chunk().unwrap(), 1);
    }
    let staged = "6fa5561a1b2b617f09db0b19c3d1ff6606d85a86c3212e495b91bd7e82708197";
    assert_eq!(row_hash(&session), staged);
    assert_eq!(session.greedy_token().unwrap().unwrap(), 432);
    let profile = session.profile().unwrap();
    assert_eq!(
        profile
            .events
            .iter()
            .filter(|event| event.name == "attention_staged_simd")
            .count(),
        16
    );
    assert!(!profile.events.iter().any(|event| event.name == "attention"));
    // The staged route uses the same failure state as the ordinary graph.
    // Seed a checked status without unsafe operands or invalid GPU commands.
    session
        .scratch
        .tokens
        .with_mut(|words| words[0] = tokens[256]);
    session.scratch.status.with_mut(|words| words[0] = 1);
    let result = objc2::rc::autoreleasepool(|_| session.run_chunk(1, 256, None));
    assert!(matches!(result, Err(metal::Error::Nonfinite)));
    assert!(session.finish_chunk(1, result).is_err());
    assert_eq!(session.evaluated_len(), 256);
    assert_eq!(row_hash(&session), staged);
    assert!(
        session
            .finish_chunk(1, Err(metal::Error::Completion))
            .is_err()
    );
    assert!(matches!(
        session.evaluate_next_chunk(),
        Err(metal::Error::ReplayRequired)
    ));
    session.set_tokens(&tokens[..256]).unwrap();
    session.evaluate().unwrap();
    assert_eq!(row_hash(&session), staged);
    session.reset();
    session.set_tokens(&tokens[..256]).unwrap();
    session.evaluate().unwrap();
    assert_eq!(row_hash(&session), staged);
    session.set_tokens(&tokens).unwrap();
    session.evaluate().unwrap();
    assert_eq!(session.evaluated_len(), 512);
    let at_limit = row_hash(&session);
    assert!(session.append(&[1]).is_err());
    assert_eq!(row_hash(&session), at_limit);
    assert!(session.enable_llama3_staged_simd().is_err());
    assert_eq!(session.evaluated_len(), 512);
    println!(
        "STAGED_SESSION {{\"admitted_bytes\":{required},\"isolated_hash_matched\":true,\"boundary_hash\":\"{at_limit}\"}}"
    );
}
