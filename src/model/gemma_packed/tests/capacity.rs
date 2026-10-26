//! Actual context/cache/history ownership, without increasing model qualification.
use super::*;
use crate::model::gemma_reference::tests::Measurement;

#[test]
fn admitted_contexts_match_frozen_prefixes_budgets_and_late_failure_recovery() {
    let data = fixture::Fixture::new();
    let model = Model::new(data.weights()).unwrap();
    for (capacity, chunk) in [
        (0, 1),
        (33, 1),
        (usize::MAX, 1),
        (3, 0),
        (3, 4),
        (32, usize::MAX),
    ] {
        let measured = Measurement::start();
        let planned = model.planned_session_bytes_with_capacity(capacity, chunk);
        let constructed = model.new_session_with_capacity(capacity, chunk, usize::MAX);
        let allocations = measured.finish();
        assert!(planned.is_err() && constructed.is_err());
        assert_eq!(allocations, 0);
    }
    let oracle: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../../../tests/fixtures/gemma-session-capacity-v1/manifest.json"
    ))
    .unwrap();
    for (case_index, case) in oracle["cases"].as_array().unwrap().iter().enumerate() {
        let capacity = case["capacity"].as_u64().unwrap() as usize;
        let chunk = case["chunk_rows"].as_u64().unwrap() as usize;
        let planned = model
            .planned_session_bytes_with_capacity(capacity, chunk)
            .unwrap();
        let budget = model.model_bytes() + planned;
        let measured = Measurement::start();
        let rejected = model
            .new_session_with_capacity(capacity, chunk, budget - 1)
            .is_err();
        let allocations = measured.finish();
        assert!(rejected);
        assert_eq!(allocations, 0);
        let mut session = model
            .new_session_with_capacity(capacity, chunk, budget)
            .unwrap();
        assert_eq!(
            (session.capacity(), session.chunk_capacity()),
            (capacity, chunk)
        );
        assert_eq!(session.history.len(), capacity);
        assert_eq!(session.allocated_bytes().unwrap(), planned);
        let addresses = pointers(&session);
        println!(
            "GEMMA_SESSION_CAPACITY_MEMORY {}",
            serde_json::json!({
            "case":case_index,"capacity":capacity,"chunk_rows":chunk,"model_bytes":model.model_bytes(),"budget":budget,
            "planned_session_bytes":planned,"actual_session_bytes":session.allocated_bytes().unwrap(),
            "session_header_bytes":std::mem::size_of_val(&session),
            "route_heap_bytes":session.plan.allocated_metadata_bytes()-std::mem::size_of::<SharedKvPlan>(),
            "history_bytes":session.history.capacity()*4,
            "cache_owner_capacity":session.caches.capacity(),"cache_owner_bytes":std::mem::size_of::<WindowKvCache>(),
            "staging_owner_capacity":session.staging.capacity(),"staging_owner_bytes":std::mem::size_of::<Staging>(),
            "cache_payload_bytes":session.caches.iter().map(|cache|{let(k,v)=cache.committed();(k.len()+v.len())*4}).sum::<usize>(),
            "staging_payload_bytes":session.staging.iter().map(|s|(s.k.capacity()+s.v.capacity())*4).sum::<usize>(),
            "scratch_elements":session.scratch.capacities(),"budget_minus_one_allocations":allocations})
        );
        let mut before = 0;
        for ids in data.ids[..capacity].chunks(chunk) {
            let measured = Measurement::start();
            let rows = session.forward(ids).unwrap();
            let allocations = measured.finish();
            assert_eq!(allocations, 0);
            both(&data, "logits", rows, before, ids.len());
            println!(
                "GEMMA_SESSION_CAPACITY_ROWS {}",
                serde_json::json!({"case":case_index,"capacity":capacity,
                "chunk_rows":chunk,"before":before,"rows":ids.len(),"values":rows,"forward_allocations":allocations})
            );
            before += ids.len();
        }
        let full = snapshot(&session);
        assert!(
            matches!(session.forward(&[1]),Err(Error::Capacity{offset,tokens:1,limit})if offset==capacity&&limit==capacity)
        );
        assert_eq!(snapshot(&session), full);
        let measured = Measurement::start();
        session.reset().unwrap();
        assert_eq!(measured.finish(), 0);
        let warm = chunk.min(capacity - 1);
        if warm > 0 {
            session.forward(&data.ids[..warm]).unwrap();
        }
        let prefix = snapshot(&session);
        session.fault = Some(model.layout.layers.len() + 2);
        let measured = Measurement::start();
        let failed = session.forward(&data.ids[warm..warm + 1]);
        let allocations = measured.finish();
        assert!(
            matches!(
                &failed,
                Err(Error::InvalidConfig("injected Gemma final failure"))
            ),
            "unexpected late failure: {failed:?}"
        );
        assert_eq!(allocations, 0);
        assert_eq!(snapshot(&session), prefix);
        session.fault = None;
        both(
            &data,
            "logits",
            session.forward(&data.ids[warm..warm + 1]).unwrap(),
            warm,
            1,
        );
        assert_eq!(pointers(&session), addresses);
        println!(
            "GEMMA_SESSION_CAPACITY_RAW_COMPLETE {}",
            serde_json::json!({"case":case_index,"capacity":capacity,
            "chunk_rows":chunk,"overflow_rejected":true,"late_failure_prefix_preserved":true,"retry_passed":true,
            "reset_allocations":0,"late_failure_allocations":allocations})
        );
    }
}
