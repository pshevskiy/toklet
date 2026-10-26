//! Final pending tokens, cache/history capacity and full-prefix replay stay distinct.
use super::*;

#[test]
fn context_sized_histories_preserve_final_pending_failure_and_replay() {
    let data = fixture::Fixture::new();
    let model = Model::new(data.weights()).unwrap();
    for capacity in [1, 2, 3, 4, 5, 17, 32] {
        let mut selected = options();
        selected.capacity = capacity;
        selected.prefill_chunk_size = capacity.min(3);
        let mut session = model.new_generation_session(selected).unwrap();
        let memory = session.memory();
        assert_eq!(sum(memory), memory.admitted_bytes);
        assert_eq!(memory.history_bytes, capacity * 2 * 4);
        assert_eq!(
            (
                session.inner.capacity(),
                session.inner.history.len(),
                session.history.len()
            ),
            (capacity, capacity, capacity)
        );
        let mut exact = selected;
        exact.memory_limit_bytes = memory.admitted_bytes - 1;
        let measured = Measurement::start();
        let rejected = model.new_generation_session(exact);
        let allocations = measured.finish();
        assert!(
            matches!(rejected,Err(crate::session::Error::MemoryBudget{required,limit})if required==memory.admitted_bytes&&limit==memory.admitted_bytes-1)
        );
        assert_eq!(allocations, 0);
        exact.memory_limit_bytes += 1;
        assert_eq!(
            model.new_generation_session(exact).unwrap().memory(),
            memory
        );
        let history_address = session.history.as_ptr();
        let addresses = pointers(&session.inner);
        let mut sampled = None;
        if capacity == 1 {
            let request = Request {
                input: Input::Prompt(&data.ids[..1]),
                max_new_tokens: 1,
                stop_tokens: &[],
            };
            let old = snapshot(&session.inner);
            assert!(matches!(
                generation::validate_request(&session, &request),
                Err(generation::Error::InvalidRequest(
                    "prompt plus requested output exceeds session capacity"
                ))
            ));
            assert_eq!(snapshot(&session.inner), old);
            assert!(session.history().is_empty());
            session.append(&data.ids[..1]).unwrap();
        } else {
            let prefix = &data.ids[..capacity - 1];
            session.set_tokens(prefix).unwrap();
            evaluated(&mut session);
            both(&data, "logits", session.logits().unwrap(), capacity - 2, 1);
            let mut emitted = Vec::with_capacity(1);
            let cancelled = AtomicBool::new(false);
            let report = generation::generate(
                &mut session,
                Request {
                    input: Input::Prompt(prefix),
                    max_new_tokens: 1,
                    stop_tokens: &[],
                },
                &cancelled,
                |event| {
                    emitted.push(event.token_id);
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(report.reason, StopReason::OutputLimit);
            assert_eq!(emitted.len(), 1);
            for contract in ["math", "staged_fp32"] {
                let all = data.values(contract, "logits");
                let values = &all[(capacity - 2) * 17..(capacity - 1) * 17];
                let mut winner = 0;
                for index in 1..values.len() {
                    if values[index] > values[winner] {
                        winner = index;
                    }
                }
                assert_eq!(emitted[0], winner as u32);
            }
            sampled = Some(emitted[0]);
        }
        assert_eq!(session.history().len(), capacity);
        assert_eq!(session.pending_len(), 1);
        assert_eq!(session.evaluated_len(), capacity - 1);
        assert!(session.logits().is_none());
        let pending_history = session.history().to_vec();
        let evaluated_before_retry = session.evaluated_len();
        let pending_before_retry = session.pending_len();
        assert!(session.append(&[1]).is_err());
        let prefix = snapshot(&session.inner);
        session.inner.fault = Some(model.layout.layers.len() + 2);
        let failed = session.evaluate_next_chunk();
        assert!(
            matches!(
                &failed,
                Err(crate::session::Error::Model(Error::InvalidConfig(
                    "injected Gemma final failure"
                )))
            ),
            "unexpected pending failure: {failed:?}"
        );
        assert_eq!(snapshot(&session.inner), prefix);
        assert_eq!(session.pending_len(), 1);
        assert!(session.logits().is_none());
        session.inner.fault = None;
        assert_eq!(session.evaluate_next_chunk().unwrap(), 1);
        assert_eq!(session.pending_len(), 0);
        assert_eq!(session.evaluated_len(), capacity);
        let mut full = model.new_session(usize::MAX).unwrap();
        let rows = full.forward(session.history()).unwrap();
        assert_eq!(session.logits().unwrap(), &rows[(capacity - 1) * 17..]);
        session.set_tokens(&data.ids[..capacity]).unwrap();
        evaluated(&mut session);
        both(&data, "logits", session.logits().unwrap(), capacity - 1, 1);
        assert_eq!(session.history.as_ptr(), history_address);
        assert_eq!(pointers(&session.inner), addresses);
        println!(
            "GEMMA_SESSION_CAPACITY_GENERATION {}",
            serde_json::json!({"capacity":capacity,
            "chunk_rows":selected.prefill_chunk_size,"memory":memory,"sampled_token":sampled,
            "pending_history":pending_history,"evaluated_before_retry":evaluated_before_retry,
            "pending_before_retry":pending_before_retry,"final_evaluated":session.evaluated_len(),
            "single_position_shared_budget_rejected":capacity==1,"final_pending_preserved":true,"late_failure_retry_passed":true,
            "full_replay_agreed":true,"frozen_prompt_restored":true,"budget_minus_one_allocations":allocations})
        );
    }
}
