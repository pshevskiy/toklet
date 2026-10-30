use super::*;
use serde_json::Value;

fn state_bytes(session: &NativeSession) -> (Vec<u32>, Option<Vec<f32>>) {
    match session {
        NativeSession::Cpu(s) => (s.history().to_vec(), s.logits().map(<[f32]>::to_vec)),
        #[cfg(all(
            feature = "metal",
            any(target_os = "macos", target_os = "ios"),
            target_arch = "aarch64"
        ))]
        NativeSession::Metal(s) => (s.history().to_vec(), s.logits().map(<[f32]>::to_vec)),
    }
}

// An intentional test-only pending-token discard exposes the last committed
// distribution, so repeated reporting is also checked against actual logits.
fn check_committed_snapshot(m: *mut ModelHandle, s: *mut SessionHandle, options: Options) {
    unsafe {
        (*s).access(|session| {
            match session {
                NativeSession::Cpu(s) => s.discard_pending(),
                #[cfg(all(
                    feature = "metal",
                    any(target_os = "macos", target_os = "ios"),
                    target_arch = "aarch64"
                ))]
                NativeSession::Metal(s) => s.discard_pending(),
            }
            Ok(())
        })
    }
    .unwrap();
    let before = unsafe { (*s).access(|session| Ok(state_bytes(session))) }.unwrap();
    assert!(before.1.is_some());
    let first = query(m, s, options, transition(m, s));
    assert_eq!(first["pending_tokens"], 0);
    assert_eq!(query(m, s, options, transition(m, s)), first);
    assert_eq!(
        unsafe { (*s).access(|session| Ok(state_bytes(session))) }.unwrap(),
        before
    );
}

fn transition(m: *mut ModelHandle, s: *mut SessionHandle) -> TransitionPlanningOptions {
    TransitionPlanningOptions {
        abi_version: ABI_VERSION,
        struct_size: size_of::<TransitionPlanningOptions>() as u32,
        expected_model_instance: unsafe { (*m).instance },
        expected_session_instance: unsafe { (*s).instance },
        transition_budget_bytes: 256 * 1024 * 1024,
        replay_reserve_bytes: 128,
        text_reserve_bytes: 4096,
        application_reserve_bytes: 8192,
        reserved: 0,
    }
}
fn capture(
    m: *const ModelHandle,
    s: *const SessionHandle,
    options: Options,
    request: TransitionPlanningOptions,
) -> (u32, *mut CompatibilityHandle) {
    let mut out = ptr::dangling_mut();
    let status = unsafe {
        toklet_session_transition_plan_create(m, s, &options, &request, &mut out, ptr::null_mut())
    };
    if status != OK {
        assert!(out.is_null());
    }
    (status, out)
}
fn read_json(handle: *mut CompatibilityHandle) -> Value {
    let mut count = 0;
    unsafe {
        assert_eq!(
            toklet_mlx_inspect_json(handle, ptr::null_mut(), 0, &mut count, ptr::null_mut()),
            OK
        );
        let mut bytes = vec![0; count];
        assert_eq!(
            toklet_mlx_inspect_json(
                handle,
                bytes.as_mut_ptr(),
                bytes.len(),
                &mut count,
                ptr::null_mut(),
            ),
            OK
        );
        assert_eq!(toklet_mlx_inspect_release(handle, ptr::null_mut()), OK);
        serde_json::from_slice(&bytes).unwrap()
    }
}
fn query(
    m: *mut ModelHandle,
    s: *mut SessionHandle,
    o: Options,
    t: TransitionPlanningOptions,
) -> Value {
    let (code, h) = capture(m, s, o, t);
    assert_eq!(code, OK);
    read_json(h)
}

#[test]
fn transition_snapshot_counts_shared_model_once_and_two_sessions_without_mutation() {
    let (m, s, cancel) = handles();
    let (status, before) =
        unsafe { generate(s, cancel, &[1, 7, 13, 3], 0, 2, None, ptr::null_mut()) };
    assert_eq!(status, OK);
    let initial_state = unsafe { (*s).access(|session| Ok(state_bytes(session))) }.unwrap();
    let t = transition(m, s);
    let value = query(m, s, opts(), t);
    assert_eq!(value["report_kind"], "same_model_session_transition");
    assert_eq!(value["model_instance"], t.expected_model_instance);
    assert_eq!(value["session_instance"], t.expected_session_instance);
    assert_eq!(value["history_tokens"], before.history_tokens);
    assert_eq!(value["pending_tokens"], before.pending_tokens);
    assert_eq!(value["candidate_allocated"], false);
    assert_eq!(value["creation_revalidation_required"], true);
    assert_eq!(value["reserves_verified"], false);
    assert_eq!(value["whole_app_device_fit_evaluated"], false);
    let a = value["current_allocated"]["total_bytes"].as_u64().unwrap();
    let b = value["candidate_planned"]["total_bytes"].as_u64().unwrap();
    let shared = value["shared_model"]["total_bytes"].as_u64().unwrap();
    let reserve = value["reserves"]["total_bytes"].as_u64().unwrap();
    let total = value["aggregate_required_bytes"].as_u64().unwrap();
    assert_eq!(total, shared + a + b + reserve);
    assert!(a > 0 && b > 0 && shared > 0);
    assert_eq!(
        value["reserves"]["current_metadata_driver_bytes"],
        64 * 1024 * 1024
    );
    assert_eq!(
        value["reserves"]["candidate_metadata_driver_bytes"],
        64 * 1024 * 1024
    );
    assert_eq!(value["reserves"]["current_bridge_bytes"], BRIDGE_BYTES);
    assert_eq!(value["reserves"]["candidate_bridge_bytes"], BRIDGE_BYTES);
    assert_eq!(
        value["reserves"]["json_snapshot_bytes"],
        COMPATIBILITY_JSON_ALLOWANCE_BYTES
    );
    let exact = TransitionPlanningOptions {
        transition_budget_bytes: total,
        ..t
    };
    assert_eq!(query(m, s, opts(), exact)["fits_transition_budget"], true);
    assert_eq!(
        query(
            m,
            s,
            opts(),
            TransitionPlanningOptions {
                transition_budget_bytes: total - 1,
                ..t
            }
        )["fits_transition_budget"],
        false
    );
    assert_eq!(
        query(
            m,
            s,
            opts(),
            TransitionPlanningOptions {
                transition_budget_bytes: 0,
                ..t
            }
        )["fits_transition_budget"],
        false
    );
    let required = value["candidate_constructor_required_bytes"]
        .as_u64()
        .unwrap();
    assert_eq!(
        query(
            m,
            s,
            Options {
                memory_limit_bytes: required,
                ..opts()
            },
            t
        )["fits_candidate_individual_budget"],
        true
    );
    assert_eq!(
        query(
            m,
            s,
            Options {
                memory_limit_bytes: required - 1,
                ..opts()
            },
            t
        )["fits_candidate_individual_budget"],
        false
    );
    let mut candidate = ptr::null_mut();
    unsafe {
        assert_eq!(
            toklet_session_create(m, &opts(), &mut candidate, ptr::null_mut()),
            OK
        );
    }
    let candidate_report = query(m, candidate, opts(), transition(m, candidate));
    assert_eq!(candidate_report["shared_model"], value["shared_model"]);
    assert!(
        candidate_report["current_allocated"]["total_bytes"]
            .as_u64()
            .unwrap()
            >= b
    );
    unsafe {
        assert_eq!(toklet_session_release(candidate, ptr::null_mut()), OK);
    }
    let again = query(m, s, opts(), t);
    assert_eq!(again, value);
    assert_eq!(
        unsafe { (*s).access(|session| Ok(state_bytes(session))) }.unwrap(),
        initial_state
    );
    let mut after = Report::default();
    unsafe {
        assert_eq!(toklet_session_state(s, &mut after, ptr::null_mut()), OK);
        assert_eq!(after.history_tokens, before.history_tokens);
        assert_eq!(after.evaluated_tokens, before.evaluated_tokens);
        assert_eq!(after.pending_tokens, before.pending_tokens);
        check_committed_snapshot(m, s, opts());
        cleanup(m, s, cancel);
    }
}

#[test]
fn transition_identity_boundaries_and_invalid_options_clear_outputs() {
    let (m, s, cancel) = handles();
    let (other, other_session, other_cancel) = handles();
    let t = transition(m, s);
    assert_eq!(capture(other, s, opts(), t).0, INVALID);
    // Even forged matching model ID does not make two identical models one Rc.
    unsafe { (*other).instance = (*m).instance };
    assert_eq!(capture(other, s, opts(), t).0, INVALID);
    for request in [
        TransitionPlanningOptions {
            expected_model_instance: t.expected_model_instance + 1,
            ..t
        },
        TransitionPlanningOptions {
            expected_session_instance: t.expected_session_instance + 1,
            ..t
        },
        TransitionPlanningOptions {
            expected_session_instance: 0,
            ..t
        },
        TransitionPlanningOptions { reserved: 1, ..t },
        TransitionPlanningOptions {
            abi_version: 99,
            ..t
        },
        TransitionPlanningOptions {
            struct_size: 0,
            ..t
        },
        TransitionPlanningOptions {
            replay_reserve_bytes: u64::MAX,
            ..t
        },
        TransitionPlanningOptions {
            transition_budget_bytes: u64::MAX,
            ..t
        },
    ] {
        assert_eq!(capture(m, s, opts(), request).0, INVALID);
    }
    for options in [
        Options {
            capacity: 0,
            ..opts()
        },
        Options {
            capacity: 33,
            ..opts()
        },
        Options { chunk: 0, ..opts() },
        Options {
            chunk: 33,
            ..opts()
        },
        Options {
            memory_limit_bytes: BRIDGE_BYTES as u64,
            ..opts()
        },
    ] {
        assert_eq!(capture(m, s, options, t).0, INVALID);
    }
    assert_eq!(
        query(
            m,
            s,
            Options {
                capacity: 1,
                chunk: 1,
                ..opts()
            },
            t
        )["candidate"]["capacity"],
        1
    );
    let borrowed = unsafe { (*s).state.borrow_mut() };
    assert_eq!(capture(m, s, opts(), t).0, BUSY);
    drop(borrowed);
    unsafe { (*s).poisoned.set(true) };
    assert_eq!(capture(m, s, opts(), t).0, POISONED);
    unsafe { (*s).poisoned.set(false) };
    let (mp, sp) = (m as usize, s as usize);
    std::thread::spawn(move || {
        assert_eq!(
            capture(
                mp as *const ModelHandle,
                sp as *const SessionHandle,
                opts(),
                t
            )
            .0,
            WRONG_THREAD
        );
    })
    .join()
    .unwrap();
    assert_eq!(size_of::<TransitionPlanningOptions>(), 64);
    assert_eq!(std::mem::align_of::<TransitionPlanningOptions>(), 8);
    assert_eq!(
        std::mem::offset_of!(TransitionPlanningOptions, expected_model_instance),
        8
    );
    assert_eq!(
        std::mem::offset_of!(TransitionPlanningOptions, reserved),
        56
    );
    unsafe {
        cleanup(other, other_session, other_cancel);
        cleanup(m, s, cancel);
    }
}

#[cfg(all(
    feature = "metal",
    any(target_os = "macos", target_os = "ios"),
    target_arch = "aarch64"
))]
#[test]
#[ignore = "Explicit tiny Mac same-model current/candidate accounting, post-step state and diagnostic exclusions"]
fn native_transition_plan_matches_tiny_metal_owners_and_preserves_used_session() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/mlx-qwen3-v1/tied"
    );
    let mut model = ptr::null_mut();
    let mut current = ptr::null_mut();
    let mut cancel = ptr::null_mut();
    unsafe {
        assert_eq!(
            toklet_model_load_mlx(
                path.as_ptr(),
                path.len(),
                &opts(),
                &mut model,
                ptr::null_mut()
            ),
            OK
        );
        assert_eq!(
            toklet_session_create(model, &opts(), &mut current, ptr::null_mut()),
            OK
        );
        assert_eq!(toklet_cancel_create(&mut cancel, ptr::null_mut()), OK);
    }
    let (status, before) =
        unsafe { generate(current, cancel, &[1, 7, 13, 3], 0, 2, None, ptr::null_mut()) };
    assert_eq!(status, OK);
    assert!(before.evaluated_tokens > 0);
    let request = transition(model, current);
    let candidate_options = Options {
        capacity: 16,
        chunk: 3,
        // This test later enables profiling on the second session, which adds
        // its own 64 MiB allowance. Ordinary fit/aggregate checks stay intact.
        memory_limit_bytes: 256 * 1024 * 1024,
        ..opts()
    };
    let value = query(model, current, candidate_options, request);
    assert_eq!(value["backend"], "metal");
    assert_eq!(value["execution_modes"]["projection"], "MlxAffine");
    assert_eq!(value["history_tokens"], before.history_tokens);
    assert_eq!(value["committed_tokens"], before.evaluated_tokens);
    let required = value["aggregate_required_bytes"].as_u64().unwrap();
    assert_eq!(
        query(
            model,
            current,
            candidate_options,
            TransitionPlanningOptions {
                transition_budget_bytes: required,
                ..request
            }
        )["fits_transition_budget"],
        true
    );
    assert_eq!(
        query(
            model,
            current,
            candidate_options,
            TransitionPlanningOptions {
                transition_budget_bytes: required - 1,
                ..request
            }
        )["fits_transition_budget"],
        false
    );
    assert!(
        value["candidate_largest_device_buffer_bytes"]
            .as_u64()
            .unwrap()
            <= value["device_maximum_buffer_bytes"].as_u64().unwrap()
    );
    let state = unsafe { (*current).access(|session| Ok(state_bytes(session))) }.unwrap();
    let mut second = ptr::null_mut();
    unsafe {
        assert_eq!(
            toklet_session_create(model, &candidate_options, &mut second, ptr::null_mut()),
            OK
        );
    }
    let actual = query(model, second, candidate_options, transition(model, second));
    assert_eq!(actual["shared_model"], value["shared_model"]);
    for field in ["kv_bytes", "scratch_bytes"] {
        assert_eq!(
            actual["current_allocated"][field],
            value["candidate_planned"][field]
        );
    }
    for field in [
        "history_bytes",
        "logits_bytes",
        "session_table_bytes",
        "total_bytes",
    ] {
        assert!(
            actual["current_allocated"][field].as_u64().unwrap()
                >= value["candidate_planned"][field].as_u64().unwrap()
        );
    }
    assert_eq!(query(model, current, candidate_options, request), value);
    assert_eq!(
        unsafe { (*current).access(|session| Ok(state_bytes(session))) }.unwrap(),
        state
    );
    check_committed_snapshot(model, current, candidate_options);
    // Retained diagnostic owners and profile state are unsupported, whereas the
    // last-chunk diagnostic_rows marker in the successfully used session is not.
    unsafe {
        assert_eq!(toklet_session_reset(current, ptr::null_mut()), OK);
    }
    unsafe {
        (*current).access(|session| match session {
            NativeSession::Metal(s) => s
                .enable_diagnostics(&[])
                .map_err(|e| fail(ENGINE_ERROR, e.to_string())),
            _ => unreachable!(),
        })
    }
    .unwrap();
    assert_eq!(
        capture(model, current, candidate_options, request).0,
        UNSUPPORTED
    );
    unsafe {
        (*second).access(|session| match session {
            NativeSession::Metal(s) => s
                .enable_profile()
                .map_err(|e| fail(ENGINE_ERROR, e.to_string())),
            _ => unreachable!(),
        })
    }
    .unwrap();
    assert_eq!(
        capture(model, second, candidate_options, transition(model, second)).0,
        UNSUPPORTED
    );
    println!(
        "{}",
        serde_json::json!({"native_transition_plan":true,"report":value,"actual_candidate":actual,"current_state_preserved":true,"diagnostic_and_trace_rejected":true})
    );
    unsafe {
        assert_eq!(toklet_session_release(second, ptr::null_mut()), OK);
        cleanup(model, current, cancel);
    }
}
