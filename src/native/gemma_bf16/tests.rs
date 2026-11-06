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
        let mut chat = ptr::null_mut();
        assert_eq!(
            toklet_chat_create(model, 1 << 30, &mut chat, error),
            UNSUPPORTED
        );
        assert!(chat.is_null());
        assert_eq!(
            toklet_chat_create_complete_audited(model, 1 << 30, &mut chat, error),
            INVALID
        );
        assert!(chat.is_null());
        assert_eq!(
            toklet_chat_create_complete_audited(model, 0, &mut chat, error),
            OK
        );
        let mut identity = ptr::null_mut();
        assert_eq!(
            toklet_model_chat_identity_create(model, chat, &mut identity, error),
            OK
        );
        let (other, other_session, other_cancel) = super::super::tests::handles();
        let mut unrelated = ptr::dangling_mut();
        assert_eq!(
            toklet_model_chat_identity_create(other, chat, &mut unrelated, error),
            INVALID
        );
        assert!(unrelated.is_null());
        assert_eq!(toklet_session_release(other_session, error), OK);
        assert_eq!(toklet_model_release(other, error), OK);
        assert_eq!(toklet_cancel_release(other_cancel, error), OK);

        let messages = br#"[{"role":"user","content":"Name the capital of France."}]"#;
        let mut prompt = [0; 15];
        let mut count = 0;
        let oversized = format!(
            r#"[{{"role":"user","content":"{}"}}]"#,
            "\u{f0000}".repeat(9_000)
        );
        assert_eq!(
            toklet_chat_encode(
                chat,
                oversized.as_ptr(),
                oversized.len(),
                ptr::null_mut(),
                0,
                &mut count,
                error,
            ),
            INVALID
        );
        assert_eq!(count, 0);
        assert_eq!(
            toklet_chat_encode(
                chat,
                messages.as_ptr(),
                messages.len(),
                prompt.as_mut_ptr(),
                prompt.len(),
                &mut count,
                error
            ),
            OK
        );
        assert_eq!(count, prompt.len());
        let mut stops = [0; 3];
        assert_eq!(
            toklet_chat_stop_ids(chat, stops.as_mut_ptr(), stops.len(), &mut count, error),
            OK
        );
        assert_eq!((count, stops), (3, [1, 106, 50]));
        assert_eq!(toklet_cancel_create(&mut cancel, error), OK);
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
        let mut visible = [0; 256];
        assert_eq!(
            toklet_chat_decode(
                chat,
                prior.as_ptr(),
                prior.len(),
                visible.as_mut_ptr(),
                visible.len(),
                &mut count,
                error
            ),
            OK
        );
        let reply = std::str::from_utf8(&visible[..count]).unwrap();
        assert_eq!(reply, "The capital of France is **Paris**.");
        assert_eq!(toklet_chat_release(chat, error), OK);
        assert_eq!(
            toklet_mlx_inspect_json(identity, ptr::null_mut(), 0, &mut count, error),
            OK
        );
        let mut json = vec![0; count];
        assert_eq!(
            toklet_mlx_inspect_json(identity, json.as_mut_ptr(), json.len(), &mut count, error),
            OK
        );
        assert_eq!(toklet_mlx_inspect_release(identity, error), OK);
        let identity: serde_json::Value = serde_json::from_slice(&json[..count]).unwrap();
        assert_eq!(
            identity["report_kind"],
            "accepted_official_gemma_bf16_identity"
        );
        assert_eq!(identity["loaded_position_limit"], 512);
        assert_eq!(identity["model_instance"], context.model_instance);
        assert_eq!(identity["whole_source_sha256_verified"], true);
        assert_eq!(identity["file_backed_ple"], true);
        assert_eq!(identity["current_disk_identity_evaluated"], false);
        assert_eq!(identity["qualification_evaluated"], false);
        assert_eq!(identity["whole_app_device_fit_evaluated"], false);
        assert_eq!(identity["assets"].as_array().unwrap().len(), 6);
        assert_eq!(
            identity["preference_namespace"]["manifest_sha256"],
            "654dc6dfd66fd893a24f0faa39355b71d5e7ffc3f53c2dac394dfcfd9ed95298"
        );
        assert_eq!(
            identity["preference_namespace"]["text"],
            "gemma4-basic-nonthinking-v1"
        );
        println!(
            "OFFICIAL_BF16_NATIVE_CHAT {}",
            serde_json::json!({"load_admitted_bytes":memory.load_admitted_bytes,
                "session_admitted_bytes":memory.session_admitted_bytes,"kv_owned_bytes":memory.kv_owned_bytes,
                "visible_ids":prior,"reply":reply,"accepted_identity":identity,"survived_external_model_release":true})
        );
    }
}
