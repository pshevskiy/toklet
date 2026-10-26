//! Explicit synthetic controls, separate from numerical trace qualification.
use super::*;
use crate::model::gemma_reference::tests::Measurement;
use crate::session::{Memory, Options};

const BUDGET: usize = 512 * 1024 * 1024;
fn options() -> Options {
    Options {
        capacity: 32,
        prefill_chunk_size: 3,
        position_offset: 0,
        memory_limit_bytes: BUDGET,
    }
}
fn sum(m: Memory) -> usize {
    m.weights_bytes
        + m.kv_bytes
        + m.scratch_bytes
        + m.history_bytes
        + m.logits_bytes
        + m.metadata_bytes
        + m.metadata_allowance_bytes
}
fn injected(error: hybrid::Error, gather: bool) {
    let expected = if gather {
        "injected Gemma hybrid gather failure"
    } else {
        "injected Gemma hybrid projection failure"
    };
    assert!(
        matches!(error, hybrid::Error::Metal(crate::metal::Error::Options(v)) if v == expected),
        "unexpected failure: {error:?}"
    );
}

#[test]
#[ignore = "explicit hybrid Gemma memory, failure and generation controls"]
fn admission_failure_replay_generation_and_source_ownership() {
    let data = fixture::Fixture::new();
    let uploaded = autoreleasepool(|_| {
        let context = gemma_affine::Context::new(gemma_affine::CONTEXT_ALLOWANCE_BYTES).unwrap();
        Uploaded::new(&context, data.weights())
    });
    let before = uploaded.snapshot();
    for (index, matrix) in before.as_array().unwrap().iter().enumerate() {
        println!(
            "GEMMA_HYBRID_CONTROL_WEIGHT_BEFORE {}",
            serde_json::json!({"index":index,"matrix":matrix})
        );
    }
    let norms = uploaded.norms();
    let model = Model::with_storage(
        uploaded.graph_weights(&norms),
        Layout::witness(),
        Some(uploaded.backing(&norms)),
    )
    .unwrap();
    let context = gemma_affine::Context::new(gemma_affine::CONTEXT_ALLOWANCE_BYTES).unwrap();
    let warm = norms.new_session(&model, &context, BUDGET).unwrap();
    let raw_admitted = model.model_bytes() + warm.allocated_bytes().unwrap();
    drop(warm);
    let measured = Measurement::start();
    let rejected = norms.new_session(&model, &context, raw_admitted - 1);
    let allocations = measured.finish();
    assert!(rejected.is_err());
    assert_eq!(allocations, 0);
    let exact = norms.new_session(&model, &context, raw_admitted).unwrap();
    assert_eq!(
        model.model_bytes() + exact.allocated_bytes().unwrap(),
        raw_admitted
    );
    drop(exact);

    let warm = norms
        .new_generation_session(&model, &context, options())
        .unwrap();
    let memory = warm.memory();
    assert_eq!(sum(memory), memory.admitted_bytes);
    assert_eq!(memory.weights_bytes, model.tensor_bytes);
    assert_eq!(memory.history_bytes, 2 * 32 * 4);
    assert_eq!(
        memory.logits_bytes,
        2 * options().prefill_chunk_size * 17 * 4
    );
    assert_eq!(
        memory.metadata_allowance_bytes,
        gemma_affine::CONTEXT_ALLOWANCE_BYTES + 2 * 16 * 1024 * 1024
    );
    drop(warm);
    let mut requested = options();
    requested.memory_limit_bytes = memory.admitted_bytes - 1;
    let measured = Measurement::start();
    let rejected = norms.new_generation_session(&model, &context, requested);
    let allocations = measured.finish();
    assert!(
        matches!(rejected, Err(crate::session::Error::MemoryBudget {required,limit})
        if required == memory.admitted_bytes && limit == memory.admitted_bytes - 1)
    );
    assert_eq!(allocations, 0);
    requested.memory_limit_bytes = memory.admitted_bytes;
    let exact = norms
        .new_generation_session(&model, &context, requested)
        .unwrap();
    assert_eq!(exact.memory(), memory);
    drop(exact);
    requested.capacity = 5;
    requested.prefill_chunk_size = 2;
    let mut smaller = norms
        .new_generation_session(&model, &context, requested)
        .unwrap();
    assert_eq!(smaller.memory().history_bytes, memory.history_bytes);
    assert_eq!(smaller.memory().weights_bytes, memory.weights_bytes);
    assert_eq!(smaller.memory().logits_bytes, 2 * 2 * 17 * 4);
    assert!(smaller.memory().scratch_bytes < memory.scratch_bytes);
    assert!(smaller.memory().kv_bytes < memory.kv_bytes);
    assert!(smaller.memory().admitted_bytes < memory.admitted_bytes);
    smaller.append(&data.ids[..5]).unwrap();
    assert!(smaller.append(&[1]).is_err());
    while smaller.evaluate_next_chunk().unwrap() != 0 {}
    assert_eq!(smaller.evaluated_len(), 5);
    assert!(smaller.set_tokens(&data.ids[..6]).is_err());
    drop(smaller);
    println!(
        "GEMMA_HYBRID_CONTROL_MEMORY {}",
        serde_json::json!({
        "raw_admitted_bytes":raw_admitted,"model_bytes":model.model_bytes(),
        "generation_wrapper_bytes":size_of::<hybrid::GenerationSession<'_ ,'_ ,'_ ,'_>>() - size_of::<hybrid::Session<'_ ,'_ ,'_ ,'_>>(),"generation":memory,
        "raw_minus_one_rust_allocations":0,"generation_minus_one_rust_allocations":0,
        "smaller_request_capacity":5,"physical_capacity":32,
        "physical_chunk_rows":options().prefill_chunk_size,"smaller_chunk_rows":2})
    );

    // A separate norm owner cannot authorize a model bound to the first owner.
    let foreign = uploaded.norms();
    let measured = Measurement::start();
    let rejected = foreign.new_session(&model, &context, BUDGET);
    let allocations = measured.finish();
    assert!(matches!(
        rejected,
        Err(hybrid::Error::Metal(crate::metal::Error::Options(
            "Gemma hybrid model/source mismatch"
        )))
    ));
    assert_eq!(allocations, 0);

    for (gather, successful_calls) in [(false, 12), (true, 1)] {
        let mut failed = norms.new_session(&model, &context, BUDGET).unwrap();
        failed.forward(&data.ids[..3]).unwrap();
        let prefix = snapshot(&failed);
        let addresses = pointers(&failed);
        failed.operations.inject_failure(gather, successful_calls);
        injected(failed.forward(&data.ids[3..5]).unwrap_err(), gather);
        assert_eq!(snapshot(&failed), prefix);
        assert!(matches!(
            failed.reset(),
            Err(hybrid::Error::Metal(crate::metal::Error::Options(
                "Gemma hybrid scratch failed; recreate session and replay"
            )))
        ));
        assert_eq!(snapshot(&failed), prefix);
        assert!(matches!(
            failed.forward(&data.ids[3..5]),
            Err(hybrid::Error::Metal(crate::metal::Error::Options(
                "Gemma hybrid scratch failed; recreate session and replay"
            )))
        ));
        assert_eq!(snapshot(&failed), prefix);
        assert_eq!(pointers(&failed), addresses);
        // Keep the failed owner alive while constructing fresh scratch. Charge
        // its external ownership, excluding the shared Context policy once.
        let extra_live = failed.allocated_bytes().unwrap() - gemma_affine::CONTEXT_ALLOWANCE_BYTES;
        let mut fresh = norms
            .new_session(&model, &context, BUDGET - extra_live)
            .unwrap();
        let logits = fresh.forward(&data.ids[..5]).unwrap();
        both(&data, "logits", logits, 0, 5);
        assert_eq!(snapshot(&failed), prefix);
        assert_eq!(uploaded.snapshot(), before);
        println!(
            "GEMMA_HYBRID_CONTROL_FAILURE {}",
            serde_json::json!({
            "gather":gather,"successful_calls_before_injection":successful_calls,
            "committed_prefix":3,"prefix_unchanged":true,"reset_rejected":true,
            "reuse_rejected":true,"fresh_replay_tokens":5,
            "extra_live_session_bytes":extra_live,"in_flight_failure_qualified":false})
        );
    }
    let mut session = norms
        .new_generation_session(&model, &context, options())
        .unwrap();
    // The second raw session exists simultaneously with the generator in the
    // shared helper. Count its additional allocation; Context and weights share.
    let extra_live =
        memory.admitted_bytes - model.model_bytes() - gemma_affine::CONTEXT_ALLOWANCE_BYTES;
    crate::model::gemma_packed::generation::tests::generator_checks(&data, &mut session, || {
        norms
            .new_session(&model, &context, BUDGET - extra_live)
            .unwrap()
    });
    drop(session);
    assert_eq!(uploaded.snapshot(), before);
    for (index, matrix) in uploaded.snapshot().as_array().unwrap().iter().enumerate() {
        println!(
            "GEMMA_HYBRID_CONTROL_WEIGHT_AFTER {}",
            serde_json::json!({"index":index,"matrix":matrix})
        );
    }
    println!(
        "GEMMA_HYBRID_CONTROL_COMPLETE {}",
        serde_json::json!({
        "generation_stop_cancel_output_failure_continue":true,"foreign_source_rejected":true,
        "weight_buffers_unchanged":true,"different_device_tested":false,
        "whole_e2b_qualified":false,"performance_qualified":false,"phone_qualified":false})
    );
}
