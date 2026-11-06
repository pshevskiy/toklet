use super::super::transition_planning::{Fault, inject};
use super::*;

fn request(m: *mut ModelHandle, s: *mut SessionHandle) -> TransitionPlanningOptions {
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
fn prepare(
    m: *mut ModelHandle,
    s: *mut SessionHandle,
    c: *mut CancelHandle,
    o: Options,
    t: TransitionPlanningOptions,
    fault: Fault,
) -> (u32, *mut SessionHandle) {
    inject(fault);
    let mut out = ptr::dangling_mut();
    let code = unsafe { toklet_session_prepare_empty(m, s, &o, &t, c, &mut out, ptr::null_mut()) };
    inject(Fault::None);
    if code != OK {
        assert!(out.is_null());
    }
    (code, out)
}
#[derive(Debug, PartialEq)]
struct Snapshot {
    instance: u64,
    model: u64,
    poison: bool,
    committed: u32,
    pending: u32,
    history: Vec<u32>,
    logits: Option<Vec<u32>>,
}
fn snapshot(s: *mut SessionHandle) -> Snapshot {
    let handle = unsafe { &*s };
    let state = handle.state.borrow();
    let r = report(&state.session);
    let (history, logits) = match &state.session {
        NativeSession::Cpu(s) => (s.history(), s.logits()),
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        NativeSession::OfficialGemma(s) => (s.inference.history(), s.inference.logits()),
        #[cfg(all(
            feature = "metal",
            any(target_os = "macos", target_os = "ios"),
            target_arch = "aarch64"
        ))]
        NativeSession::Metal(s) => (s.history(), s.logits()),
    };
    Snapshot {
        instance: handle.instance,
        model: state.model_instance,
        poison: handle.poisoned.get(),
        committed: r.evaluated_tokens,
        pending: r.pending_tokens,
        history: history.to_vec(),
        logits: logits.map(|v| v.iter().map(|x| x.to_bits()).collect()),
    }
}
fn planned(
    m: *mut ModelHandle,
    s: *mut SessionHandle,
    o: Options,
    t: TransitionPlanningOptions,
) -> serde_json::Value {
    let mut out = ptr::null_mut();
    unsafe {
        assert_eq!(
            toklet_session_transition_plan_create(m, s, &o, &t, &mut out, ptr::null_mut()),
            OK
        );
        let mut n = 0;
        assert_eq!(
            toklet_mlx_inspect_json(out, ptr::null_mut(), 0, &mut n, ptr::null_mut()),
            OK
        );
        let mut bytes = vec![0; n];
        assert_eq!(
            toklet_mlx_inspect_json(out, bytes.as_mut_ptr(), n, &mut n, ptr::null_mut()),
            OK
        );
        assert_eq!(toklet_mlx_inspect_release(out, ptr::null_mut()), OK);
        serde_json::from_slice(&bytes).unwrap()
    }
}
fn exercise_rollback(m: *mut ModelHandle, s: *mut SessionHandle, c: *mut CancelHandle) {
    let (code, _) = unsafe { generate(s, c, &[1, 7, 13, 3], 0, 2, None, ptr::null_mut()) };
    assert_eq!(code, OK);
    let before = snapshot(s);
    assert_eq!(
        (before.history.len(), before.committed, before.pending),
        (6, 5, 1)
    );
    let refs = unsafe { Rc::strong_count(&(*m).model) };
    let o = Options {
        capacity: 3,
        chunk: 3,
        ..opts()
    }; // Explicit empty shrink below old history.
    let t = request(m, s);
    for (fault, expected) in [
        (Fault::Allocation, ENGINE_ERROR),
        (Fault::PanicConstruction, PANIC),
        (Fault::FailureAfter, ENGINE_ERROR),
        (Fault::PanicAfter, PANIC),
        (Fault::CancelAfter, CANCELLED),
    ] {
        assert_eq!(prepare(m, s, c, o, t, fault).0, expected);
        assert_eq!(snapshot(s), before);
        assert_eq!(unsafe { Rc::strong_count(&(*m).model) }, refs);
        // Cancellation is observable and not silently cleared by preparation.
        assert_eq!(
            unsafe { (*c).flag.load(Ordering::Relaxed) },
            fault == Fault::CancelAfter
        );
        unsafe {
            assert_eq!(toklet_cancel_set(c, 0, ptr::null_mut()), OK);
        }
    }
    unsafe {
        assert_eq!(toklet_cancel_set(c, 1, ptr::null_mut()), OK);
    }
    assert_eq!(prepare(m, s, c, o, t, Fault::None).0, CANCELLED);
    unsafe {
        assert_eq!(toklet_cancel_set(c, 0, ptr::null_mut()), OK);
    }
    assert_eq!(snapshot(s), before);
    let (code, candidate) = prepare(m, s, c, o, t, Fault::None);
    assert_eq!(code, OK);
    let empty = snapshot(candidate);
    assert_ne!(empty.instance, before.instance);
    assert_eq!(empty.model, before.model);
    assert_eq!(
        (empty.history.len(), empty.committed, empty.pending),
        (0, 0, 0)
    );
    assert!(empty.logits.is_none());
    assert_eq!(snapshot(s), before);
    assert_eq!(unsafe { Rc::strong_count(&(*m).model) }, refs + 1);
    unsafe {
        assert_eq!(toklet_session_release(candidate, ptr::null_mut()), OK);
    }
    assert_eq!(unsafe { Rc::strong_count(&(*m).model) }, refs);
    // The original pending session remains usable after every rejected candidate.
    let (code, continued) = unsafe { generate(s, c, &[], 2, 1, None, ptr::null_mut()) };
    assert_eq!(code, OK);
    assert_eq!(continued.history_tokens, 7);
    // Also test a stable committed distribution with visible logits, rather than
    // treating pending logits=None as evidence about their numerical contents.
    unsafe {
        (*s).access(|session| {
            match session {
                NativeSession::Cpu(s) => s.discard_pending(),
                #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
                NativeSession::OfficialGemma(s) => s.inference.discard_pending(),
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
    let committed = snapshot(s);
    assert!(committed.logits.is_some());
    for (fault, expected) in [
        (Fault::FailureAfter, ENGINE_ERROR),
        (Fault::PanicAfter, PANIC),
        (Fault::CancelAfter, CANCELLED),
    ] {
        assert_eq!(prepare(m, s, c, o, t, fault).0, expected);
        assert_eq!(snapshot(s), committed);
        assert_eq!(unsafe { Rc::strong_count(&(*m).model) }, refs);
        unsafe {
            assert_eq!(toklet_cancel_set(c, 0, ptr::null_mut()), OK);
        }
    }
}
#[test]
fn empty_candidate_rollback_isolates_errors_panics_cancellation_and_shrink() {
    let (m, s, c) = handles();
    exercise_rollback(m, s, c);
    unsafe {
        cleanup(m, s, c);
    }
}
#[test]
fn empty_candidate_rechecks_actual_individual_and_coexistence_capacity() {
    let (m, s, c) = handles();
    let o = Options {
        capacity: 4,
        ..opts()
    };
    let t = request(m, s);
    let plan = planned(m, s, o, t);
    let (code, baseline) = prepare(m, s, c, o, t, Fault::None);
    assert_eq!(code, OK);
    let observed = planned(m, baseline, o, request(m, baseline));
    let aggregate = plan["shared_model"]["total_bytes"].as_u64().unwrap()
        + plan["current_allocated"]["total_bytes"].as_u64().unwrap()
        + observed["current_allocated"]["total_bytes"]
            .as_u64()
            .unwrap()
        + plan["reserves"]["total_bytes"].as_u64().unwrap();
    let individual = {
        let state = unsafe { (*baseline).state.borrow() };
        match &state.session {
            NativeSession::Cpu(s) => {
                (s.owned_memory().unwrap().admitted_bytes + BRIDGE_BYTES) as u64
            }
            #[cfg(all(
                feature = "metal",
                any(target_os = "macos", target_os = "ios"),
                target_arch = "aarch64"
            ))]
            NativeSession::Metal(_) => unreachable!(),
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            NativeSession::OfficialGemma(_) => unreachable!(),
        }
    };
    unsafe {
        assert_eq!(toklet_session_release(baseline, ptr::null_mut()), OK);
    }
    let before = snapshot(s);
    let refs = unsafe { Rc::strong_count(&(*m).model) };
    let exact = TransitionPlanningOptions {
        transition_budget_bytes: aggregate,
        ..t
    };
    assert_eq!(
        prepare(
            m,
            s,
            c,
            o,
            TransitionPlanningOptions {
                transition_budget_bytes: aggregate - 1,
                ..t
            },
            Fault::None
        )
        .0,
        ENGINE_ERROR
    );
    let (code, new) = prepare(m, s, c, o, exact, Fault::None);
    assert_eq!(code, OK);
    unsafe {
        assert_eq!(toklet_session_release(new, ptr::null_mut()), OK);
    }
    // Force real excess Vec capacity only in the candidate, after its constructor.
    assert_eq!(
        prepare(m, s, c, o, exact, Fault::ExcessHistory).0,
        ENGINE_ERROR
    );
    let exact_individual = Options {
        memory_limit_bytes: individual,
        ..o
    };
    let (code, new) = prepare(m, s, c, exact_individual, t, Fault::None);
    assert_eq!(code, OK);
    unsafe {
        assert_eq!(toklet_session_release(new, ptr::null_mut()), OK);
    }
    assert_eq!(
        prepare(
            m,
            s,
            c,
            Options {
                memory_limit_bytes: individual - 1,
                ..o
            },
            t,
            Fault::None
        )
        .0,
        ENGINE_ERROR
    );
    assert_eq!(
        prepare(m, s, c, exact_individual, t, Fault::ExcessHistory).0,
        ENGINE_ERROR
    );
    assert_eq!(snapshot(s), before);
    assert_eq!(unsafe { Rc::strong_count(&(*m).model) }, refs);
    unsafe {
        cleanup(m, s, c);
    }
}
#[test]
fn empty_candidate_rejects_identity_busy_poison_options_and_wrong_owner() {
    let (m, s, c) = handles();
    let (other, os, oc) = handles();
    let t = request(m, s);
    let before = snapshot(s);
    assert_eq!(
        prepare(
            m,
            s,
            c,
            opts(),
            TransitionPlanningOptions {
                expected_session_instance: t.expected_session_instance + 1,
                ..t
            },
            Fault::None
        )
        .0,
        INVALID
    );
    assert_eq!(
        prepare(
            other,
            s,
            c,
            opts(),
            TransitionPlanningOptions {
                expected_model_instance: unsafe { (*other).instance },
                ..t
            },
            Fault::None
        )
        .0,
        INVALID
    );
    assert_eq!(
        prepare(
            m,
            s,
            c,
            Options {
                capacity: 33,
                ..opts()
            },
            t,
            Fault::None
        )
        .0,
        INVALID
    );
    assert_eq!(
        prepare(
            m,
            s,
            c,
            opts(),
            TransitionPlanningOptions { reserved: 1, ..t },
            Fault::None
        )
        .0,
        INVALID
    );
    let hold = unsafe { (*s).state.borrow_mut() };
    assert_eq!(prepare(m, s, c, opts(), t, Fault::None).0, BUSY);
    drop(hold);
    unsafe {
        (*s).poisoned.set(true);
    }
    assert_eq!(prepare(m, s, c, opts(), t, Fault::None).0, POISONED);
    unsafe {
        (*s).poisoned.set(false);
    }
    let owner = unsafe { (*s).owner };
    let other_owner = thread::spawn(|| thread::current().id()).join().unwrap();
    unsafe {
        (*s).owner = other_owner;
    }
    assert_eq!(prepare(m, s, c, opts(), t, Fault::None).0, WRONG_THREAD);
    unsafe {
        (*s).owner = owner;
    }
    assert_eq!(snapshot(s), before);
    unsafe {
        cleanup(other, os, oc);
        cleanup(m, s, c);
    }
}
#[cfg(all(
    feature = "metal",
    any(target_os = "macos", target_os = "ios"),
    target_arch = "aarch64"
))]
#[test]
#[ignore = "Explicit tiny Metal empty-candidate rollback, pending-state preservation and actual coexistence admission"]
fn native_empty_candidate_preserves_used_tiny_metal_session() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/mlx-qwen3-v1/tied"
    );
    let mut m = ptr::null_mut();
    let mut s = ptr::null_mut();
    let mut c = ptr::null_mut();
    unsafe {
        assert_eq!(
            toklet_model_load_mlx(path.as_ptr(), path.len(), &opts(), &mut m, ptr::null_mut()),
            OK
        );
        assert_eq!(
            toklet_session_create(m, &opts(), &mut s, ptr::null_mut()),
            OK
        );
        assert_eq!(toklet_cancel_create(&mut c, ptr::null_mut()), OK);
    }
    exercise_rollback(m, s, c);
    let o = Options {
        capacity: 16,
        ..opts()
    };
    let t = request(m, s);
    let plan = planned(m, s, o, t);
    let exact = TransitionPlanningOptions {
        transition_budget_bytes: plan["aggregate_required_bytes"].as_u64().unwrap(),
        ..t
    };
    assert_eq!(
        prepare(
            m,
            s,
            c,
            o,
            TransitionPlanningOptions {
                transition_budget_bytes: exact.transition_budget_bytes - 1,
                ..t
            },
            Fault::None
        )
        .0,
        ENGINE_ERROR
    );
    let (code, new) = prepare(m, s, c, o, exact, Fault::None);
    assert_eq!(code, OK);
    let actual = planned(m, new, o, request(m, new));
    println!(
        "{}",
        serde_json::json!({"empty_candidate":true,"plan":plan,"actual_candidate":actual,"rollback_and_pending_state_preserved":true})
    );
    unsafe {
        assert_eq!(toklet_session_release(new, ptr::null_mut()), OK);
        cleanup(m, s, c);
    }
}
