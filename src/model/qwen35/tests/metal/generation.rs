use super::*;
use crate::session::InferenceSession;

fn step_options() -> Options {
    Options {
        chunk: 3,
        ..options()
    }
}

#[test]
#[ignore = "explicit tiny Metal generation-prefix, poisoned execution repair and wrapper budget controls"]
fn prefix_retry_and_repair_preserve_pending_history_and_wrapper_budget() {
    let fixture = Fixture::new();
    let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
    let gpu = upload(&context, &fixture);
    let model = bind(&fixture, &gpu);
    let ids = tokens();
    let mut direct = model.session(&context, step_options()).unwrap();
    for part in ids.chunks(3) {
        direct.forward(part).unwrap();
    }
    let expected = snapshot(&direct);
    drop(direct);
    let mut session = GenerationSession::new(
        model.session(&context, step_options()).unwrap(),
        MEMORY_LIMIT,
    )
    .unwrap();
    session.set_tokens(&ids[..3]).unwrap();
    assert_eq!(session.evaluate_next_chunk().unwrap(), 3);
    let prefix_logits = session.logits().unwrap().to_vec();
    let memory = session.memory_report();
    session.append(&ids[3..6]).unwrap();
    let result = with_projection_failure(1, || session.evaluate_next_chunk());
    assert!(result.is_err());
    assert!(session.requires_execution_repair());
    assert_eq!((session.evaluated_len(), session.pending_len()), (3, 3));
    assert_eq!(session.state().last_numerical_work.tokens_committed, 0);
    session.discard_pending();
    assert_float_bits(session.logits().unwrap(), &prefix_logits);
    session.append(&ids[3..6]).unwrap();
    session.repair_execution(&context).unwrap();
    assert!(!session.requires_execution_repair());
    assert_eq!(session.history(), &ids[..6]);
    let measure = Measurement::start();
    assert_eq!(session.evaluate_next_chunk().unwrap(), 3);
    assert_eq!(measure.finish(), 0);
    session.append(&ids[6..]).unwrap();
    while session.evaluate_next_chunk().unwrap() != 0 {}
    assert_eq!(session.memory_report(), memory);
    assert_eq!(snapshot(&session.into_numerical()), expected);

    let raw = model.session(&context, step_options()).unwrap();
    let full = GenerationSession::new(raw, MEMORY_LIMIT).unwrap();
    let tight_budget = full.memory_report().admitted_bytes;
    drop(full);
    let mut tight = GenerationSession::new(
        model.session(&context, step_options()).unwrap(),
        tight_budget,
    )
    .unwrap();
    tight.append(&ids[..3]).unwrap();
    assert!(with_projection_failure(1, || tight.evaluate_next_chunk()).is_err());
    assert!(tight.requires_execution_repair());
    let state = tight.state();
    assert!(
        matches!(tight.repair_execution(&context), Err(Error::MemoryBudget { required, limit }) if required > tight_budget - (memory.pending_history_bytes + memory.wrapper_metadata_bytes) && limit == tight_budget - (memory.pending_history_bytes + memory.wrapper_metadata_bytes))
    );
    assert_eq!(tight.state(), state);
    assert!(tight.requires_execution_repair());
    tight.reset();
    assert!(tight.requires_execution_repair());
    assert!(tight.history().is_empty() && tight.logits().is_none());
    println!(
        "QWEN35_GENERATION_TINY {}",
        serde_json::json!({"prefix_retry_bits_exact":true,"pending_history_preserved":true,"repair_overlap_budget_rejected":true,"warmed_rust_allocations":0,"memory":memory})
    );
}
