//! Long frozen prompt versus sampled-history consistency are distinct checks.
use super::*;
use crate::model::gemma_packed::tests::long_context as frozen;

#[test]
#[ignore = "explicit long scalar generation state qualification"]
fn pending_last_position_failure_and_sampled_history_replay() {
    let data = fixture::Fixture::new();
    let oracle = frozen::oracle();
    let ids = frozen::ids(&oracle);
    let model = Model::new_witness_with_context(data.weights(), 1025).unwrap();
    let mut selected = options();
    selected.capacity = 1025;
    selected.prefill_chunk_size = 32;
    let mut session = model.new_generation_session(selected).unwrap();
    let memory = session.memory();
    assert_eq!(sum(memory), memory.admitted_bytes);
    assert_eq!(memory.history_bytes, 1025 * 8);
    let mut bad = selected;
    bad.prefill_chunk_size = 33;
    let measured = Measurement::start();
    assert!(model.new_generation_session(bad).is_err());
    assert_eq!(measured.finish(), 0);
    let mut exact = selected;
    exact.memory_limit_bytes = memory.admitted_bytes - 1;
    let measured = Measurement::start();
    assert!(matches!(
        model.new_generation_session(exact),
        Err(crate::session::Error::MemoryBudget { .. })
    ));
    assert_eq!(measured.finish(), 0);
    exact.memory_limit_bytes += 1;
    assert_eq!(
        model.new_generation_session(exact).unwrap().memory(),
        memory
    );
    let addresses = pointers(&session.inner);
    let history = session.history.as_ptr();
    session.set_tokens(&ids[..1024]).unwrap();
    evaluated(&mut session);
    frozen::check(
        &oracle,
        "generation-prompt",
        "logits",
        session.logits().unwrap(),
        1023,
        1,
    );
    let mut winner = 0;
    let logits = session.logits().unwrap();
    for i in 1..logits.len() {
        if logits[i] > logits[winner] {
            winner = i;
        }
    }
    let cancel = AtomicBool::new(false);
    let mut emitted = Vec::with_capacity(1);
    let report = generation::generate(
        &mut session,
        Request {
            input: Input::Prompt(&ids[..1024]),
            max_new_tokens: 1,
            stop_tokens: &[],
        },
        &cancel,
        |e| {
            emitted.push(e.token_id);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(report.reason, StopReason::OutputLimit);
    assert_eq!(emitted, [winner as u32]);
    assert_eq!(
        (
            session.history().len(),
            session.evaluated_len(),
            session.pending_len()
        ),
        (1025, 1024, 1)
    );
    assert!(session.logits().is_none());
    let prefix = snapshot(&session.inner);
    let full_history = session.history().to_vec();
    assert!(session.append(&[1]).is_err());
    assert_eq!(snapshot(&session.inner), prefix);
    assert_eq!(session.history(), full_history);
    assert_eq!((session.evaluated_len(), session.pending_len()), (1024, 1));
    assert!(session.logits().is_none());
    session.inner.fault = Some(model.layout.layers.len() + 2);
    let measured = Measurement::start();
    let failed = session.evaluate_next_chunk();
    let allocations = measured.finish();
    assert!(matches!(
        failed,
        Err(crate::session::Error::Model(Error::InvalidConfig(
            "injected Gemma final failure"
        )))
    ));
    assert_eq!(allocations, 0);
    assert_eq!(snapshot(&session.inner), prefix);
    assert_eq!(session.history(), full_history);
    assert_eq!((session.evaluated_len(), session.pending_len()), (1024, 1));
    assert!(session.logits().is_none());
    session.inner.fault = None;
    let measured = Measurement::start();
    assert_eq!(session.evaluate_next_chunk().unwrap(), 1);
    assert_eq!(measured.finish(), 0);
    let mut replay = model.new_session_with_chunk(32, usize::MAX).unwrap();
    for chunk in session.history().chunks(32) {
        replay.forward(chunk).unwrap();
    }
    assert_eq!(
        session.logits().unwrap(),
        &replay.logits()[replay.logits().len() - 17..]
    );
    assert_eq!(snapshot(&session.inner), snapshot(&replay));
    session.set_tokens(&ids).unwrap();
    evaluated(&mut session);
    frozen::check(
        &oracle,
        "generation-restored",
        "logits",
        session.logits().unwrap(),
        1024,
        1,
    );
    assert_eq!(pointers(&session.inner), addresses);
    assert_eq!(session.history.as_ptr(), history);
    println!(
        "GEMMA_LONG_GENERATION_COMPLETE {}",
        serde_json::json!({"capacity":1025,"chunk":32,"memory":memory,"sampled_token":emitted[0],"frozen_prompt_position":1023,"pending_before_retry":1,"committed_before_retry":1024,"final_committed":session.evaluated_len(),"failure_atomic":true,"sampled_history_replay_bitwise":true,"frozen_prompt_restored":true,"pointers_stable":true})
    );
}
