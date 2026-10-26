//! Same frozen numerical graph, independent per-call scratch bound.
use super::*;
use crate::model::gemma_reference::tests::Measurement;

#[test]
fn bounded_chunk_memory_full_prefix_oracle_and_transaction_recovery() {
    let data = fixture::Fixture::new();
    let model = Model::new(data.weights()).unwrap();
    assert_eq!(model.new_session(usize::MAX).unwrap().chunk_capacity(), 32);
    for invalid in [0, 33] {
        let measured = Measurement::start();
        let planned = model.planned_session_bytes_with_chunk(invalid);
        let constructed = model.new_session_with_chunk(invalid, usize::MAX);
        let allocations = measured.finish();
        assert!(planned.is_err() && constructed.is_err());
        assert_eq!(allocations, 0);
    }
    for chunk in [1, 3, 8, 32] {
        let planned = model.planned_session_bytes_with_chunk(chunk).unwrap();
        let budget = model.model_bytes() + planned;
        let measured = Measurement::start();
        let rejected = model.new_session_with_chunk(chunk, budget - 1);
        let allocations = measured.finish();
        assert!(rejected.is_err());
        assert_eq!(allocations, 0);
        let mut session = model.new_session_with_chunk(chunk, budget).unwrap();
        assert_eq!(session.chunk_capacity(), chunk);
        assert_eq!(session.allocated_bytes().unwrap(), planned);
        let addresses = pointers(&session);
        println!(
            "GEMMA_CHUNK_SCRATCH_MEMORY {}",
            serde_json::json!({"chunk_rows":chunk,"context_positions":32,
                "model_bytes":model.model_bytes(),"budget":budget,"budget_minus_one_rejected":true,
                "planned_session_bytes":planned,"actual_session_bytes":session.allocated_bytes().unwrap(),
                "session_header_bytes":std::mem::size_of_val(&session),
                "route_heap_bytes":session.plan.allocated_metadata_bytes()-std::mem::size_of::<SharedKvPlan>(),
                "history_bytes":session.history.capacity()*4,
                "cache_owner_capacity":session.caches.capacity(),
                "cache_owner_bytes":std::mem::size_of::<WindowKvCache>(),
                "staging_owner_capacity":session.staging.capacity(),
                "staging_owner_bytes":std::mem::size_of::<Staging>(),
                "cache_payload_bytes":session.caches.iter().map(|cache| {
                    let (k,v)=cache.committed();(k.len()+v.len())*4
                }).sum::<usize>(),
                "staging_payload_bytes":session.staging.iter().map(|s|(s.k.capacity()+s.v.capacity())*4).sum::<usize>(),
                "scratch_elements":session.scratch.capacities(),"budget_minus_one_allocations":allocations})
        );
        let mut before = 0;
        for ids in data.ids.chunks(chunk) {
            let measured = Measurement::start();
            let rows = session.forward(ids).unwrap();
            let allocations = measured.finish();
            assert_eq!(allocations, 0);
            both(&data, "logits", rows, before, ids.len());
            println!(
                "GEMMA_CHUNK_SCRATCH_ROWS {}",
                serde_json::json!({"chunk_rows":chunk,"before":before,"rows":ids.len(),
                    "values":rows,"forward_allocations":allocations})
            );
            before += ids.len();
        }
        assert_eq!(session.tokens(), data.ids);
        assert_eq!(pointers(&session), addresses);
        let complete = snapshot(&session);
        assert!(matches!(
            session.forward(&[1]),
            Err(Error::Capacity { limit: 32, .. })
        ));
        assert_eq!(snapshot(&session), complete);
        let measured = Measurement::start();
        session.reset().unwrap();
        assert_eq!(measured.finish(), 0);
        let count = chunk.min(2);
        session.forward(&data.ids[..count]).unwrap();
        let stable = snapshot(&session);
        let oversized = vec![1; chunk + 1];
        let measured = Measurement::start();
        let rejected = session.forward(&oversized);
        let allocations = measured.finish();
        assert!(matches!(
            rejected,
            Err(Error::InvalidConfig("Gemma chunk exceeds scratch capacity"))
        ));
        assert_eq!(allocations, 0);
        assert_eq!(snapshot(&session), stable);
        session.fault = Some(model.layout.layers.len() + 2);
        let measured = Measurement::start();
        let failed = session.forward(&data.ids[count..count + 1]);
        let allocations = measured.finish();
        assert!(
            matches!(
                &failed,
                Err(Error::InvalidConfig("injected Gemma final failure"))
            ),
            "unexpected late-failure result: {failed:?}"
        );
        assert_eq!(allocations, 0);
        assert_eq!(snapshot(&session), stable);
        session.fault = None;
        both(
            &data,
            "logits",
            session.forward(&data.ids[count..count + 1]).unwrap(),
            count,
            1,
        );
        assert_eq!(pointers(&session), addresses);
        println!(
            "GEMMA_CHUNK_SCRATCH_RECOVERY {}",
            serde_json::json!({"chunk_rows":chunk,"context_overflow_rejected":true,
                "oversized_chunk_rejected":true,"prior_prefix_preserved":true,
                "final_failure_retry_passed":true,"reset_allocations":0,"failure_allocations":allocations})
        );
    }
}
