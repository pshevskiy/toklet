use super::*;
use crate::session::InferenceSession;

fn allocator_environment() -> Value {
    serde_json::json!({"MallocLargeCache":std::env::var("MallocLargeCache").ok(),
        "MallocAggressiveMadvise":std::env::var("MallocAggressiveMadvise").ok(),
        "MallocNanoZone":std::env::var("MallocNanoZone").ok()})
}
fn planned_combined(model: &MetalModel<'_>, options: Options) -> usize {
    let raw = model.planned_session_memory(options).unwrap();
    let extra = size_of::<GenerationSession<'_, '_, MetalMatrix<'_>, MetalExecution<'_>>>()
        - size_of::<MetalSession<'_, '_, '_>>()
        + options.capacity * 4;
    let bytes = raw.admitted_bytes
        + (raw.admitted_bytes - raw.weights_and_model_bytes)
        + extra
        + DIAGNOSTIC;
    assert!(bytes <= MEMORY_LIMIT);
    bytes
}
fn actual_combined(direct: Memory, wrapper: GenerationMemory) -> usize {
    assert_eq!(
        direct.weights_and_model_bytes,
        wrapper.numerical.weights_and_model_bytes
    );
    let bytes = direct.admitted_bytes + wrapper.admitted_bytes - direct.weights_and_model_bytes
        + DIAGNOSTIC;
    assert!(bytes <= MEMORY_LIMIT);
    bytes
}

#[test]
#[ignore = "explicit authenticated Qwen17 generation state and recovery qualification"]
fn actual_source_generation_state() {
    println!();
    let loaded = load();
    let model = MetalModel::from_metal_store(&loaded.store).unwrap();
    let options = opts(37);
    let planned = planned_combined(&model, options);
    let mut direct = model.session(&loaded.context, options).unwrap();
    direct.forward(&IDS).unwrap();
    let expected = snapshot(&direct);
    let expected_logits = direct.logits().unwrap().to_vec();
    let wrapper_limit = options.memory_limit_bytes
        - (direct.memory().admitted_bytes - direct.memory().weights_and_model_bytes);
    let mut session = GenerationSession::new(
        model.session(&loaded.context, options).unwrap(),
        wrapper_limit,
    )
    .unwrap();
    let memory = session.memory_report();
    let combined = actual_combined(direct.memory(), memory);
    assert_eq!(combined, planned);
    session.set_tokens(&IDS).unwrap();
    let measure = Measurement::start();
    assert_eq!(session.evaluate_next_chunk().unwrap(), 17);
    assert_eq!(measure.finish(), 0);
    assert_eq!(snapshot(session.numerical_for_tests()), expected);
    assert_float_bits(session.logits().unwrap(), &expected_logits);
    session.set_tokens(&IDS).unwrap();
    assert_eq!(session.state().prefix_action, PrefixAction::FullyCached);
    let measure = Measurement::start();
    assert_eq!(session.evaluate_next_chunk().unwrap(), 0);
    assert_eq!(measure.finish(), 0);
    assert_eq!(session.state().last_numerical_work, work(17));
    let full_state = session.state();

    session.set_tokens(&IDS[..5]).unwrap();
    assert_eq!(session.state().prefix_action, PrefixAction::ShortenedReplay);
    session.evaluate_next_chunk().unwrap();
    let prefix = snapshot(session.numerical_for_tests());
    session.append(&IDS[5..]).unwrap();
    assert!(with_projection_failure(1, || session.evaluate_next_chunk()).is_err());
    let failed_state = session.state();
    assert_eq!(
        (failed_state.committed_tokens, failed_state.pending_tokens),
        (5, 12)
    );
    assert_eq!(failed_state.last_numerical_work.tokens_committed, 0);
    assert_eq!(snapshot(session.numerical_for_tests()), prefix);
    assert!(session.requires_execution_repair());
    session.repair_execution(&loaded.context).unwrap();
    assert_eq!(session.state(), failed_state);
    let measure = Measurement::start();
    assert_eq!(session.evaluate_next_chunk().unwrap(), 12);
    assert_eq!(measure.finish(), 0);
    assert_eq!(snapshot(session.numerical_for_tests()), expected);
    assert_float_bits(session.logits().unwrap(), &expected_logits);
    let retry_state = session.state();
    let mut changed = IDS;
    changed[2] = 14;
    session.set_tokens(&changed).unwrap();
    assert_eq!(session.state().prefix_action, PrefixAction::ChangedReplay);
    session.evaluate_next_chunk().unwrap();
    direct.reset();
    direct.forward(&changed).unwrap();
    assert_eq!(snapshot(session.numerical_for_tests()), snapshot(&direct));
    let changed_snapshot = snapshot(&direct);
    let measure = Measurement::start();
    session.reset();
    session.append(&IDS).unwrap();
    session.discard_pending();
    assert_eq!(measure.finish(), 0);
    assert_eq!(session.memory_report(), memory);
    assert!(session.history().is_empty() && session.logits().is_none());
    println!(
        "QWEN35_GENERATION_STATE {}",
        serde_json::json!({
        "source_sha256":source::CONTAINER_SHA256,"tokens":IDS,"position_offset":37,"capacity":512,"chunk":32,
        "memory":memory,"combined_admitted_bytes":combined,"diagnostic_allowance_bytes":DIAGNOSTIC,
        "load_report":loaded.store.report(),"full_state":full_state,"failed_state":failed_state,"retry_state":retry_state,
        "snapshot":expected,"changed_snapshot":changed_snapshot,"full_logits_bits_exact":true,
        "warmed_rust_allocations":0,"prefix_failure_repair_retry_exact":true,"reset_discard_exact":true,
        "allocator_environment":allocator_environment(),"deeper_context_qualified":false,"request_metrics_qualified":false,
        "instrumentation_overhead_qualified":false,"quality_qualified":false,"managed_adoption_qualified":false})
    );
}

#[test]
#[ignore = "explicit Qwen17 numerical and generation-step release phase baseline"]
fn matched_generation_step_baseline() {
    println!();
    let loaded = load();
    let model = MetalModel::from_metal_store(&loaded.store).unwrap();
    let options = opts(37);
    let planned = planned_combined(&model, options);
    let mut direct = model.session(&loaded.context, options).unwrap();
    let wrapper_limit = options.memory_limit_bytes
        - (direct.memory().admitted_bytes - direct.memory().weights_and_model_bytes);
    let mut session = GenerationSession::new(
        model.session(&loaded.context, options).unwrap(),
        wrapper_limit,
    )
    .unwrap();
    let combined = actual_combined(direct.memory(), session.memory_report());
    assert_eq!(combined, planned);
    let cohort: usize = std::env::var("TOKLET_QWEN35_GENERATION_COHORT")
        .unwrap()
        .parse()
        .unwrap();
    assert!(cohort < 2);
    for iteration in 0..13 {
        direct.reset();
        session.reset();
        let mut numerical_ns = 0;
        let mut generation_ns = 0;
        for turn in 0..2 {
            let wrapper = (iteration + turn + cohort) % 2 == 1;
            let measure = Measurement::start();
            let start = Instant::now();
            if wrapper {
                session.set_tokens(&IDS).unwrap();
                assert_eq!(session.evaluate_next_chunk().unwrap(), 17);
                assert_eq!(session.evaluate_next_chunk().unwrap(), 0);
                generation_ns = start.elapsed().as_nanos();
            } else {
                direct.forward(&IDS).unwrap();
                numerical_ns = start.elapsed().as_nanos();
            }
            assert_eq!(measure.finish(), 0);
        }
        assert_float_bits(session.logits().unwrap(), direct.logits().unwrap());
        assert_eq!(snapshot(session.numerical_for_tests()), snapshot(&direct));
        println!(
            "QWEN35_GENERATION_SAMPLE {}",
            serde_json::json!({"iteration":iteration,"warmup":iteration<3,"cohort":cohort,
            "first_role":if (iteration+cohort)%2==1 {"generation"} else {"numerical"},
            "numerical_ns":numerical_ns,"generation_ns":generation_ns,"inputs":17,"position_offset":37,
            "warmed_rust_allocations":0,"all_state_logits_bits_exact":true})
        );
    }
    println!(
        "QWEN35_GENERATION_BASELINE {}",
        serde_json::json!({"source_sha256":source::CONTAINER_SHA256,
        "tokens":IDS,"position_offset":37,"capacity":512,"chunk":32,"cohort":cohort,"warmups":3,"measurements":10,
        "memory":session.memory_report(),"direct_memory":direct.memory(),"allocator_environment":allocator_environment(),
        "combined_admitted_bytes":combined,"diagnostic_allowance_bytes":DIAGNOSTIC,
        "scope":"Completed17-input forward with resident weights; resets outside timer, wrapper timer includes set_tokens and pending-drain loop; diagnostic compare/output outside; no request or instrumentation-overhead qualification",
        "request_ttft_qualified":false,"instrumentation_overhead_qualified":false,"quality_qualified":false,"managed_adoption_qualified":false})
    );
}
