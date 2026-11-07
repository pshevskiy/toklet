use super::*;

#[allow(clippy::too_many_arguments)]
unsafe fn run(
    s: *mut SessionHandle,
    c: *mut CancelHandle,
    operation: u32,
    input: &[u32],
    maximum: u32,
    stops: &[u32],
    emit: Emit,
    context: *mut c_void,
) -> (u32, Report, GenerationMetrics) {
    let mut report = Report::default();
    let mut metrics = GenerationMetrics {
        known_fields: u32::MAX,
        ..GenerationMetrics::default()
    };
    let code = unsafe {
        toklet_session_generate_observed(
            s,
            operation,
            input.as_ptr(),
            input.len(),
            maximum,
            stops.as_ptr(),
            stops.len(),
            c,
            emit,
            context,
            &mut report,
            &mut metrics,
            ptr::null_mut(),
        )
    };
    (code, report, metrics)
}
unsafe fn cleanup(m: *mut ModelHandle, s: *mut SessionHandle, c: *mut CancelHandle) {
    unsafe {
        assert_eq!(toklet_session_release(s, ptr::null_mut()), OK);
        assert_eq!(toklet_model_release(m, ptr::null_mut()), OK);
        assert_eq!(toklet_cancel_release(c, ptr::null_mut()), OK);
    }
}
#[test]
fn completed_prefill_pending_continuation_and_hidden_stop_are_exact() {
    let (m, s, c) = super::super::tests::handles();
    let (code, report, metrics) =
        unsafe { run(s, c, 0, &[1, 7, 13, 3], 2, &[], None, ptr::null_mut()) };
    assert_eq!(code, OK);
    assert_eq!(size_of::<GenerationMetrics>(), 96);
    assert_eq!(
        metrics.known_fields,
        GENERATION_REPORT_KNOWN | GENERATION_FIRST_TOKEN_KNOWN
    );
    assert_eq!(
        (
            metrics.prefill_tokens,
            metrics.reused_tokens,
            metrics.sampled_tokens,
            metrics.emitted_tokens
        ),
        (4, 0, 2, 2)
    );
    assert_eq!(metrics.session_instance, unsafe { (*s).instance });
    assert_eq!(
        (
            report.history_tokens,
            report.evaluated_tokens,
            report.pending_tokens
        ),
        (6, 5, 1)
    );
    assert!(
        metrics.total_seconds
            >= metrics.prefill_seconds + metrics.decode_seconds + metrics.sampling_seconds
    );
    assert!(metrics.first_token_seconds <= metrics.total_seconds);
    let (code, _, next) = unsafe { run(s, c, 2, &[], 1, &[], None, ptr::null_mut()) };
    assert_eq!(code, OK);
    assert_eq!(
        (next.reused_tokens, next.prefill_tokens, next.sampled_tokens),
        (5, 1, 1)
    );
    let (code, report, stop) =
        unsafe { run(s, c, 3, &[1, 7, 13, 3], 2, &[47], None, ptr::null_mut()) };
    assert_eq!((code, report.reason), (OK, 2));
    assert_eq!(
        (stop.sampled_tokens, stop.emitted_tokens, stop.known_fields),
        (1, 0, GENERATION_REPORT_KNOWN)
    );
    assert_eq!(stop.first_token_seconds, 0.0);
    unsafe { cleanup(m, s, c) };
}
#[test]
fn cancellation_zero_output_and_error_do_not_reuse_measurements() {
    let (m, s, c) = super::super::tests::handles();
    let (code, _, prefill) = unsafe { run(s, c, 0, &[1, 7, 13, 3], 0, &[], None, ptr::null_mut()) };
    assert_eq!(code, OK);
    assert_eq!(
        (
            prefill.prefill_tokens,
            prefill.sampled_tokens,
            prefill.known_fields
        ),
        (4, 0, GENERATION_REPORT_KNOWN)
    );
    unsafe {
        assert_eq!(toklet_cancel_set(c, 1, ptr::null_mut()), OK);
    }
    let (code, report, cancelled) = unsafe { run(s, c, 2, &[], 1, &[], None, ptr::null_mut()) };
    assert_eq!((code, report.reason), (OK, 3));
    assert_eq!((cancelled.prefill_tokens, cancelled.sampled_tokens), (0, 0));
    assert_eq!(cancelled.known_fields, GENERATION_REPORT_KNOWN);
    let (code, _, invalid) = unsafe { run(s, c, 99, &[], 1, &[], None, ptr::null_mut()) };
    assert_eq!(code, INVALID);
    assert_eq!(invalid.known_fields, 0);
    assert_eq!(invalid.prefill_seconds, 0.0);
    assert_eq!(invalid.abi_version, ABI_VERSION);
    assert_eq!(invalid.struct_size, 96);
    unsafe { cleanup(m, s, c) };
}
unsafe extern "C" fn fail_output(_: *mut c_void, _: TokenEvent) -> u32 {
    1
}
#[test]
fn callback_error_keeps_recoverable_state_but_phase_report_unknown() {
    let (m, s, c) = super::super::tests::handles();
    let (code, report, metrics) = unsafe {
        run(
            s,
            c,
            0,
            &[1, 7, 13, 3],
            2,
            &[],
            Some(fail_output),
            ptr::null_mut(),
        )
    };
    assert_eq!(code, CALLBACK_ERROR);
    assert_eq!(metrics.known_fields, 0);
    assert_eq!(
        (
            report.history_tokens,
            report.evaluated_tokens,
            report.pending_tokens,
            report.emitted_tokens
        ),
        (5, 4, 1, 0)
    );
    unsafe { cleanup(m, s, c) };
}

unsafe extern "C" fn reenter_observed(context: *mut c_void, _: TokenEvent) -> u32 {
    let session = context.cast::<SessionHandle>();
    let mut report = Report::default();
    let mut metrics = GenerationMetrics::default();
    let code = unsafe {
        toklet_session_generate_observed(
            session,
            2,
            ptr::null(),
            0,
            1,
            ptr::null(),
            0,
            ptr::null(),
            None,
            ptr::null_mut(),
            &mut report,
            &mut metrics,
            ptr::null_mut(),
        )
    };
    // Use the same live cancellation handle in the actual test below; null
    // cancellation itself must fail validation without touching the active owner.
    if code == INVALID && metrics.known_fields == 0 {
        0
    } else {
        1
    }
}
#[test]
fn invalid_observed_boundary_cannot_mutate_an_active_or_idle_session() {
    let (m, s, c) = super::super::tests::handles();
    let (code, report, metrics) = unsafe {
        run(
            s,
            c,
            0,
            &[1, 7, 13, 3],
            1,
            &[],
            Some(reenter_observed),
            s.cast(),
        )
    };
    assert_eq!(code, OK);
    assert_eq!(metrics.emitted_tokens, 1);
    assert_eq!(
        (
            report.history_tokens,
            report.evaluated_tokens,
            report.pending_tokens
        ),
        (5, 4, 1)
    );
    let before = unsafe { (*s).access(|s| Ok(super::super::report(s))).unwrap() };
    let mut report = Report::default();
    let code = unsafe {
        toklet_session_generate_observed(
            s,
            2,
            ptr::null(),
            0,
            1,
            ptr::null(),
            0,
            c,
            None,
            ptr::null_mut(),
            &mut report,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    assert_eq!(code, INVALID);
    let after = unsafe { (*s).access(|s| Ok(super::super::report(s))).unwrap() };
    assert_eq!(
        (
            before.history_tokens,
            before.evaluated_tokens,
            before.pending_tokens
        ),
        (
            after.history_tokens,
            after.evaluated_tokens,
            after.pending_tokens
        )
    );
    unsafe { cleanup(m, s, c) };
}

struct ReentryProbe {
    session: *mut SessionHandle,
    cancel: *mut CancelHandle,
    code: u32,
    known: u32,
}
unsafe extern "C" fn probe_busy(context: *mut c_void, _: TokenEvent) -> u32 {
    let probe = unsafe { &mut *context.cast::<ReentryProbe>() };
    let mut report = Report::default();
    let mut metrics = GenerationMetrics::default();
    probe.code = unsafe {
        toklet_session_generate_observed(
            probe.session,
            2,
            ptr::null(),
            0,
            1,
            ptr::null(),
            0,
            probe.cancel,
            None,
            ptr::null_mut(),
            &mut report,
            &mut metrics,
            ptr::null_mut(),
        )
    };
    probe.known = metrics.known_fields;
    0
}
#[test]
fn observed_reentry_is_busy_and_wrong_thread_stays_unknown() {
    let (m, s, c) = super::super::tests::handles();
    let mut probe = ReentryProbe {
        session: s,
        cancel: c,
        code: u32::MAX,
        known: u32::MAX,
    };
    let (code, report, _) = unsafe {
        run(
            s,
            c,
            0,
            &[1, 7, 13, 3],
            1,
            &[],
            Some(probe_busy),
            (&mut probe as *mut ReentryProbe).cast(),
        )
    };
    assert_eq!((code, probe.code, probe.known), (OK, BUSY, 0));
    assert_eq!(
        (
            report.history_tokens,
            report.evaluated_tokens,
            report.pending_tokens
        ),
        (5, 4, 1)
    );
    let addresses = (s as usize, c as usize);
    let (code, _, metrics) = std::thread::spawn(move || unsafe {
        run(
            addresses.0 as *mut SessionHandle,
            addresses.1 as *mut CancelHandle,
            2,
            &[],
            1,
            &[],
            None,
            ptr::null_mut(),
        )
    })
    .join()
    .unwrap();
    assert_eq!(code, WRONG_THREAD);
    assert_eq!(metrics.known_fields, 0);
    unsafe { cleanup(m, s, c) };
}
