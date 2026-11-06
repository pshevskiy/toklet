use super::*;

unsafe extern "C" fn record(context: *mut c_void, event: TokenEvent) -> u32 {
    // SAFETY: the test passes an exclusive live Vec for the synchronous call.
    unsafe { (*context.cast::<Vec<u32>>()).push(event.token_id) };
    0
}

#[test]
#[ignore = "explicit Mac official BF16 C ABI lifetime and raw generation witness"]
fn official_bf16_native_raw_owner_survives_model_release() {
    let directory = std::env::var("TOKLET_GEMMA_OFFICIAL_DIRECTORY").unwrap();
    let options = Options {
        abi_version: ABI_VERSION,
        struct_size: size_of::<Options>() as u32,
        capacity: 512,
        chunk: 32,
        memory_limit_bytes: weights::INITIAL_MEMORY_LIMIT_BYTES as u64,
    };
    let mut model = ptr::null_mut();
    let mut session = ptr::null_mut();
    let mut cancel = ptr::null_mut();
    let mut context = ContextInfo::default();
    let mut memory = GemmaBf16MemoryInfo::default();
    let error = ptr::null_mut();
    // SAFETY: the test keeps all owned handles and disjoint output storage live
    // on the creating thread, except the explicit wrong-thread query below.
    unsafe {
        assert_eq!(
            toklet_model_load_gemma_bf16_audited(
                directory.as_ptr(),
                directory.len(),
                &options,
                &mut model,
                error,
            ),
            OK
        );
        assert_eq!(
            toklet_session_create(model, &options, &mut session, error),
            OK
        );
        assert_eq!(toklet_session_context(session, &mut context, error), OK);
        assert_eq!(
            toklet_session_gemma_bf16_memory(session, &mut memory, error),
            OK
        );
        assert_eq!(context.declared_total_positions, 131_072);
        assert_eq!(
            (context.loaded_position_limit, context.session_capacity),
            (512, 512)
        );
        assert_eq!(
            memory.session_admitted_bytes,
            context.numerical_and_bridge_admitted_bytes + memory.session_reserved_bytes
        );
        assert_eq!(memory.codec_allowance_bytes, 1 << 30);
        assert!(memory.session_admitted_bytes <= options.memory_limit_bytes);
        let mut candidate = ptr::null_mut();
        let below = Options {
            memory_limit_bytes: memory.session_admitted_bytes - 1,
            ..options
        };
        assert_eq!(
            toklet_session_create(model, &below, &mut candidate, error),
            ENGINE_ERROR
        );
        assert!(candidate.is_null());
        let NativeModel::OfficialGemma(loaded) = (&*model).model.as_ref() else {
            panic!("expected the audited BF16 owner");
        };
        let prompt = loaded
            .owner
            .codec()
            .encode_chat(
                br#"[{"role":"user","content":"Name the capital of France."}]"#,
                crate::app_chat::gemma4::RenderOptions {
                    enable_thinking: false,
                    add_generation_prompt: true,
                },
            )
            .unwrap();
        assert_eq!(toklet_cancel_create(&mut cancel, error), OK);
        let stops = [1_u32, 106, 50];
        let generate = |maximum, output: &mut Vec<u32>, report: &mut Report| {
            toklet_session_generate(
                session,
                3,
                prompt.as_ptr(),
                prompt.len(),
                maximum,
                stops.as_ptr(),
                stops.len(),
                cancel,
                Some(record),
                std::ptr::from_mut(output).cast(),
                report,
                error,
            )
        };
        let mut output = Vec::new();
        let mut report = Report::default();
        assert_eq!(generate(0, &mut output, &mut report), OK);
        assert!(output.is_empty());
        assert_eq!(report.evaluated_tokens as usize, prompt.len());
        let transition = TransitionPlanningOptions {
            abi_version: ABI_VERSION,
            struct_size: size_of::<TransitionPlanningOptions>() as u32,
            expected_model_instance: context.model_instance,
            expected_session_instance: context.session_instance,
            transition_budget_bytes: options.memory_limit_bytes,
            replay_reserve_bytes: 0,
            text_reserve_bytes: 0,
            application_reserve_bytes: 0,
            reserved: 0,
        };
        let mut plan = ptr::null_mut();
        assert_eq!(
            toklet_session_transition_plan_create(
                model,
                session,
                &options,
                &transition,
                &mut plan,
                error
            ),
            UNSUPPORTED
        );
        assert!(plan.is_null());
        assert_eq!(toklet_session_context(session, &mut context, error), OK);
        assert_eq!(context.history_tokens, prompt.len() as u64);
        assert_eq!(context.committed_tokens, context.history_tokens);
        let mut chat = ptr::null_mut();
        assert_eq!(
            toklet_chat_create(model, 1 << 30, &mut chat, error),
            UNSUPPORTED
        );
        assert!(chat.is_null());
        let address = session as usize;
        assert_eq!(
            thread::spawn(move || {
                let mut output = GemmaBf16MemoryInfo::default();
                let code = toklet_session_gemma_bf16_memory(
                    address as *const SessionHandle,
                    &mut output,
                    ptr::null_mut(),
                );
                assert_eq!(output.struct_size, 0);
                code
            })
            .join()
            .unwrap(),
            WRONG_THREAD
        );
        assert_eq!(toklet_model_release(model, error), OK);
        let mut score = Score::default();
        assert_eq!(toklet_session_score(session, 100, &mut score, error), OK);
        assert!(score.negative_log_likelihood.is_finite());
        assert_eq!(generate(12, &mut output, &mut report), OK);
        assert_eq!(output, [818, 5279, 529, 7001, 563, 5213, 50429, 84750]);
        let prior = output.clone();
        assert_eq!(toklet_session_reset(session, error), OK);
        output.clear();
        assert_eq!(toklet_cancel_set(cancel, 1, error), OK);
        assert_eq!(generate(12, &mut output, &mut report), OK);
        assert!(output.is_empty() && report.reason == 3 && report.history_tokens == 0);
        assert_eq!(toklet_cancel_set(cancel, 0, error), OK);
        assert_eq!(generate(12, &mut output, &mut report), OK);
        assert_eq!(output, prior);
        assert_eq!(toklet_session_context(session, &mut context, error), OK);
        assert_eq!(context.model_instance, transition.expected_model_instance);
        assert_eq!(
            toklet_session_gemma_bf16_memory(session, &mut memory, error),
            OK
        );
        assert_eq!(toklet_cancel_release(cancel, error), OK);
        assert_eq!(toklet_session_release(session, error), OK);
        println!(
            "OFFICIAL_BF16_NATIVE_RAW {}",
            serde_json::json!({"load_admitted_bytes":memory.load_admitted_bytes,
                "session_admitted_bytes":memory.session_admitted_bytes,"kv_owned_bytes":memory.kv_owned_bytes,
                "visible_ids":prior,"survived_external_model_release":true})
        );
    }
}
