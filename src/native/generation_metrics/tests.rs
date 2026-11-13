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

// V2 uses the same independent tiny CPU model as the legacy boundary tests.
// Expected rows below follow completed evaluations, not callback counts.
#[allow(clippy::too_many_arguments)]
unsafe fn run_v2(
    s: *mut SessionHandle,
    c: *mut CancelHandle,
    operation: u32,
    input: &[u32],
    maximum: u32,
    stops: &[u32],
    emit: Emit,
    context: *mut c_void,
) -> (u32, Report, GenerationMetricsV2) {
    let mut report = Report::default();
    let mut metrics = GenerationMetricsV2 {
        base: GenerationMetrics {
            known_fields: u32::MAX,
            session_instance: u64::MAX,
            ..GenerationMetrics::default()
        },
        decode_evaluations: u64::MAX,
    };
    let code = unsafe {
        toklet_session_generate_observed_v2(
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

fn assert_v2_counts(metrics: GenerationMetricsV2, report: Report, expected: [u64; 8]) {
    let base = metrics.base;
    assert_eq!(
        (base.abi_version, base.struct_size, base.reserved),
        (1, 96, 0)
    );
    assert_ne!(base.session_instance, 0);
    assert_eq!(
        base.known_fields,
        GENERATION_REPORT_KNOWN
            | if expected[3] == 0 {
                0
            } else {
                GENERATION_FIRST_TOKEN_KNOWN
            }
    );
    assert_eq!(
        [
            base.reused_tokens,
            base.prefill_tokens,
            base.sampled_tokens,
            base.emitted_tokens,
            metrics.decode_evaluations,
            u64::from(report.history_tokens),
            u64::from(report.evaluated_tokens),
            u64::from(report.pending_tokens),
        ],
        expected
    );
}

fn assert_v2_unknown(metrics: GenerationMetricsV2) {
    let base = metrics.base;
    assert_eq!(
        (base.abi_version, base.struct_size, base.reserved),
        (1, 96, 0)
    );
    assert_eq!(base.known_fields, 0);
    assert_eq!(
        [
            base.session_instance,
            base.reused_tokens,
            base.prefill_tokens,
            base.sampled_tokens,
            base.emitted_tokens,
            metrics.decode_evaluations,
        ],
        [0; 6]
    );
    assert_eq!(
        [
            base.prefill_seconds,
            base.decode_seconds,
            base.sampling_seconds,
            base.total_seconds,
            base.first_token_seconds,
        ],
        [0.0; 5]
    );
}

#[test]
fn v2_layout_preserves_legacy_layout_and_exact_write_extents() {
    use std::mem::offset_of;
    assert_eq!(
        (
            size_of::<GenerationMetrics>(),
            align_of::<GenerationMetrics>()
        ),
        (96, 8)
    );
    assert_eq!(
        (
            size_of::<GenerationMetricsV2>(),
            align_of::<GenerationMetricsV2>()
        ),
        (104, 8)
    );
    assert_eq!(offset_of!(GenerationMetricsV2, base), 0);
    assert_eq!(offset_of!(GenerationMetricsV2, decode_evaluations), 96);
    assert_eq!(
        [
            offset_of!(GenerationMetrics, abi_version),
            offset_of!(GenerationMetrics, struct_size),
            offset_of!(GenerationMetrics, known_fields),
            offset_of!(GenerationMetrics, reserved),
            offset_of!(GenerationMetrics, session_instance),
            offset_of!(GenerationMetrics, reused_tokens),
            offset_of!(GenerationMetrics, prefill_tokens),
            offset_of!(GenerationMetrics, sampled_tokens),
            offset_of!(GenerationMetrics, emitted_tokens),
            offset_of!(GenerationMetrics, prefill_seconds),
            offset_of!(GenerationMetrics, decode_seconds),
            offset_of!(GenerationMetrics, sampling_seconds),
            offset_of!(GenerationMetrics, total_seconds),
            offset_of!(GenerationMetrics, first_token_seconds),
        ],
        [0, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88]
    );
    #[repr(C)]
    struct Guarded<T> {
        value: T,
        canary: u64,
    }
    let (m, s, c) = super::super::tests::handles();
    let mut report = Report::default();
    let mut old = Guarded {
        value: GenerationMetrics::default(),
        canary: u64::MAX,
    };
    let mut new = Guarded {
        value: GenerationMetricsV2::default(),
        canary: u64::MAX,
    };
    unsafe {
        assert_eq!(
            toklet_session_generate_observed(
                s,
                0,
                [1, 7, 13, 3].as_ptr(),
                4,
                0,
                ptr::null(),
                0,
                c,
                None,
                ptr::null_mut(),
                &mut report,
                &mut old.value,
                ptr::null_mut()
            ),
            OK
        );
        assert_eq!(
            toklet_session_generate_observed_v2(
                s,
                3,
                [1, 7, 13, 3].as_ptr(),
                4,
                2,
                ptr::null(),
                0,
                c,
                None,
                ptr::null_mut(),
                &mut report,
                &mut new.value,
                ptr::null_mut()
            ),
            OK
        );
        cleanup(m, s, c);
    }
    assert_eq!((old.canary, new.canary), (u64::MAX, u64::MAX));
    assert_eq!(
        (old.value.struct_size, new.value.base.struct_size),
        (96, 96)
    );
    assert_eq!(new.value.decode_evaluations, 1);
}

#[test]
fn v2_rows_distinguish_cached_input_pending_prefill_reset_and_distribution_replay() {
    let (m, s, c) = super::super::tests::handles();
    unsafe {
        let (code, report, metrics) =
            run_v2(s, c, 0, &[1, 7, 13, 3], 0, &[], None, ptr::null_mut());
        assert_eq!(code, OK);
        assert_v2_counts(metrics, report, [0, 4, 0, 0, 0, 4, 4, 0]);
        let (code, report, metrics) =
            run_v2(s, c, 0, &[1, 7, 13, 3], 2, &[], None, ptr::null_mut());
        assert_eq!(code, OK);
        assert_v2_counts(metrics, report, [4, 0, 2, 2, 1, 6, 5, 1]);
        let (code, report, metrics) = run_v2(s, c, 2, &[], 1, &[], None, ptr::null_mut());
        assert_eq!(code, OK);
        assert_v2_counts(metrics, report, [5, 1, 1, 1, 0, 7, 6, 1]);
        let (code, report, metrics) =
            run_v2(s, c, 3, &[1, 7, 13, 3], 1, &[], None, ptr::null_mut());
        assert_eq!(code, OK);
        assert_v2_counts(metrics, report, [0, 4, 1, 1, 0, 5, 4, 1]);
        // Exact sampled history is still only four committed rows.
        let (code, report, metrics) =
            run_v2(s, c, 0, &[1, 7, 13, 3, 47], 1, &[], None, ptr::null_mut());
        assert_eq!(code, OK);
        assert_v2_counts(metrics, report, [4, 1, 1, 1, 0, 6, 5, 1]);
        // Shortening a committed prompt needs its last distribution replayed.
        let (code, report, metrics) = run_v2(s, c, 0, &[1, 7, 13], 1, &[], None, ptr::null_mut());
        assert_eq!(code, OK);
        assert_v2_counts(metrics, report, [2, 1, 1, 1, 0, 4, 3, 1]);
        cleanup(m, s, c);
    }
}

unsafe extern "C" fn cancel_v2_output(context: *mut c_void, _: TokenEvent) -> u32 {
    unsafe { toklet_cancel_set(context.cast::<CancelHandle>(), 1, ptr::null_mut()) }
}

#[test]
fn v2_hidden_stop_and_successful_cancellations_have_known_zero_rows() {
    let (m, s, c) = super::super::tests::handles();
    unsafe {
        let (code, report, metrics) =
            run_v2(s, c, 0, &[1, 7, 13, 3], 2, &[47], None, ptr::null_mut());
        assert_eq!((code, report.reason, report.stop_token), (OK, 2, 47));
        assert_v2_counts(metrics, report, [0, 4, 1, 0, 0, 5, 4, 1]);
        assert_eq!(metrics.base.first_token_seconds, 0.0);
        assert_eq!(toklet_cancel_set(c, 1, ptr::null_mut()), OK);
        let (code, report, metrics) = run_v2(s, c, 2, &[], 1, &[], None, ptr::null_mut());
        assert_eq!((code, report.reason), (OK, 3));
        assert_v2_counts(metrics, report, [0, 0, 0, 0, 0, 5, 4, 1]);
        assert_eq!(toklet_cancel_set(c, 0, ptr::null_mut()), OK);
        let (code, report, metrics) = run_v2(
            s,
            c,
            3,
            &[1, 7, 13, 3],
            2,
            &[],
            Some(cancel_v2_output),
            c.cast(),
        );
        assert_eq!((code, report.reason), (OK, 3));
        assert_v2_counts(metrics, report, [0, 4, 1, 1, 0, 5, 4, 1]);
        cleanup(m, s, c);
    }
}

// Only this test wrapper changes cancellation timing. Every model operation is
// delegated to the real tiny CPU session; no Report/counter is fabricated.
struct CancelAfterDecode<'a, 'model> {
    inner: session::Session<'model>,
    cancelled: &'a AtomicBool,
    armed: &'a Cell<bool>,
}
impl session::InferenceSession for CancelAfterDecode<'_, '_> {
    fn history(&self) -> &[u32] {
        self.inner.history()
    }
    fn evaluated_len(&self) -> usize {
        self.inner.evaluated_len()
    }
    fn pending_len(&self) -> usize {
        self.inner.pending_len()
    }
    fn capacity(&self) -> usize {
        self.inner.capacity()
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn memory(&self) -> session::Memory {
        self.inner.memory()
    }
    fn logits(&self) -> Option<&[f32]> {
        self.inner.logits()
    }
    fn append(&mut self, tokens: &[u32]) -> std::result::Result<(), session::Error> {
        self.inner.append(tokens)
    }
    fn set_tokens(&mut self, tokens: &[u32]) -> std::result::Result<(), session::Error> {
        self.inner.set_tokens(tokens)
    }
    fn evaluate_next_chunk(&mut self) -> std::result::Result<usize, session::Error> {
        let rows = self.inner.evaluate_next_chunk()?;
        if self.armed.get() {
            self.cancelled.store(true, Ordering::Relaxed);
        }
        Ok(rows)
    }
    fn discard_pending(&mut self) {
        self.inner.discard_pending();
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
}

#[test]
fn v2_counts_successful_decode_cancelled_before_the_next_sample() {
    let model = super::super::tests::model();
    let cancelled = AtomicBool::new(false);
    let armed = Cell::new(false);
    let mut session = CancelAfterDecode {
        inner: session::Session::new(
            &model,
            session::Options {
                capacity: 32,
                prefill_chunk_size: 3,
                ..session::Options::default()
            },
        )
        .unwrap(),
        cancelled: &cancelled,
        armed: &armed,
    };
    let report = generation::generate(
        &mut session,
        generation::Request {
            input: generation::Input::Prompt(&[1, 7, 13, 3]),
            max_new_tokens: 2,
            stop_tokens: &[],
        },
        &cancelled,
        |event| {
            assert_eq!(event.token_id, 47);
            armed.set(true);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(report.reason, generation::StopReason::Cancelled);
    assert_eq!(
        (
            report.decode_evaluations,
            report.sampled_tokens,
            report.emitted_tokens
        ),
        (1, 1, 1)
    );
    assert_eq!(
        (
            report.history_tokens,
            report.evaluated_tokens,
            report.pending_tokens
        ),
        (5, 5, 0)
    );
    assert!(session.inner.logits().is_some());
    let metrics = GenerationMetricsV2::from_report(7, &report);
    assert_eq!(metrics.decode_evaluations, 1);
    assert_eq!(
        metrics.base.known_fields,
        GENERATION_REPORT_KNOWN | GENERATION_FIRST_TOKEN_KNOWN
    );
    assert_eq!(
        (metrics.base.sampled_tokens, metrics.base.emitted_tokens),
        (1, 1)
    );
}

unsafe extern "C" fn fail_second_v2_output(context: *mut c_void, _: TokenEvent) -> u32 {
    let calls = unsafe { &mut *context.cast::<usize>() };
    *calls += 1;
    u32::from(*calls == 2)
}

#[test]
fn v2_callback_failure_keeps_completed_decode_unavailable() {
    let (m, s, c) = super::super::tests::handles();
    unsafe {
        let (code, report, metrics) = run_v2(
            s,
            c,
            0,
            &[1, 7, 13, 3],
            2,
            &[],
            Some(fail_output),
            ptr::null_mut(),
        );
        assert_eq!(code, CALLBACK_ERROR);
        assert_eq!(
            (
                report.history_tokens,
                report.evaluated_tokens,
                report.pending_tokens,
                report.emitted_tokens
            ),
            (5, 4, 1, 0)
        );
        assert_v2_unknown(metrics);
        let mut calls = 0_usize;
        let (code, report, metrics) = run_v2(
            s,
            c,
            3,
            &[1, 7, 13, 3],
            2,
            &[],
            Some(fail_second_v2_output),
            (&mut calls as *mut usize).cast(),
        );
        assert_eq!((code, calls), (CALLBACK_ERROR, 2));
        assert_eq!(
            (
                report.history_tokens,
                report.evaluated_tokens,
                report.pending_tokens,
                report.emitted_tokens
            ),
            (6, 5, 1, 1)
        );
        assert_v2_unknown(metrics);
        cleanup(m, s, c);
    }
}

unsafe extern "C" fn collect_v2_output(context: *mut c_void, event: TokenEvent) -> u32 {
    unsafe { &mut *context.cast::<Vec<u32>>() }.push(event.token_id);
    0
}

unsafe fn cpu_snapshot(s: *mut SessionHandle, discard: bool) -> (Vec<u32>, Option<Vec<u32>>) {
    unsafe {
        (*s).access(|state| match state {
            NativeSession::Cpu(cpu) => {
                if discard {
                    cpu.discard_pending();
                }
                Ok((
                    cpu.history().to_vec(),
                    cpu.logits()
                        .map(|v| v.iter().map(|x| x.to_bits()).collect()),
                ))
            }
            #[cfg(all(
                feature = "metal",
                any(target_os = "macos", target_os = "ios"),
                target_arch = "aarch64"
            ))]
            _ => panic!("handles() must retain the tiny CPU fixture"),
        })
        .unwrap()
    }
}

#[test]
fn v2_legacy_and_observed_paths_preserve_tokens_state_and_logit_bits() {
    let (m, s, c) = super::super::tests::handles();
    let mut outcomes = Vec::new();
    for path in 0..3 {
        let mut ids = Vec::<u32>::new();
        let context = (&mut ids as *mut Vec<u32>).cast();
        let (code, report) = unsafe {
            match path {
                0 => {
                    let mut report = Report::default();
                    let code = toklet_session_generate(
                        s,
                        3,
                        [1, 7, 13, 3].as_ptr(),
                        4,
                        2,
                        ptr::null(),
                        0,
                        c,
                        Some(collect_v2_output),
                        context,
                        &mut report,
                        ptr::null_mut(),
                    );
                    (code, report)
                }
                1 => {
                    let (code, report, metrics) = run(
                        s,
                        c,
                        3,
                        &[1, 7, 13, 3],
                        2,
                        &[],
                        Some(collect_v2_output),
                        context,
                    );
                    assert_eq!(metrics.session_instance, (*s).instance);
                    (code, report)
                }
                _ => {
                    let (code, report, metrics) = run_v2(
                        s,
                        c,
                        3,
                        &[1, 7, 13, 3],
                        2,
                        &[],
                        Some(collect_v2_output),
                        context,
                    );
                    assert_eq!(metrics.base.session_instance, (*s).instance);
                    assert_v2_counts(metrics, report, [0, 4, 2, 2, 1, 6, 5, 1]);
                    (code, report)
                }
            }
        };
        assert_eq!((code, report.reason), (OK, 1));
        assert_eq!(ids, [47, 47]);
        let pending = unsafe { cpu_snapshot(s, false) };
        assert!(pending.1.is_none());
        let committed = unsafe { cpu_snapshot(s, true) };
        assert!(committed.1.is_some());
        outcomes.push((
            (
                report.history_tokens,
                report.evaluated_tokens,
                report.pending_tokens,
                report.emitted_tokens,
                report.admitted_bytes,
            ),
            ids,
            pending,
            committed,
        ));
    }
    assert_eq!(outcomes[0], outcomes[1]);
    assert_eq!(outcomes[1], outcomes[2]);
    unsafe { cleanup(m, s, c) };
}

#[test]
fn v2_invalid_boundaries_clear_stale_metrics_without_mutating_the_session() {
    let (m, s, c) = super::super::tests::handles();
    unsafe {
        assert_eq!(
            run_v2(s, c, 0, &[1, 7, 13, 3], 0, &[], None, ptr::null_mut()).0,
            OK
        );
        let before = cpu_snapshot(s, false);
        for (session, cancel, operation, input, maximum, stops) in [
            (s, c, 99, &[][..], 1, &[][..]),
            (s, c, 0, &[64][..], 1, &[][..]),
            (s, c, 2, &[][..], 1, &[64][..]),
            (s, c, 2, &[][..], 33, &[][..]),
            (ptr::null_mut(), c, 2, &[][..], 1, &[][..]),
            (s, ptr::null_mut(), 2, &[][..], 1, &[][..]),
        ] {
            let (code, _, metrics) = run_v2(
                session,
                cancel,
                operation,
                input,
                maximum,
                stops,
                None,
                ptr::null_mut(),
            );
            assert_eq!(code, INVALID);
            assert_v2_unknown(metrics);
            assert_eq!(cpu_snapshot(s, false), before);
        }
        let mut metrics = GenerationMetricsV2 {
            base: GenerationMetrics {
                known_fields: u32::MAX,
                ..GenerationMetrics::default()
            },
            decode_evaluations: u64::MAX,
        };
        assert_eq!(
            toklet_session_generate_observed_v2(
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
                ptr::null_mut(),
                &mut metrics,
                ptr::null_mut()
            ),
            INVALID
        );
        assert_v2_unknown(metrics);
        let mut report = Report::default();
        assert_eq!(
            toklet_session_generate_observed_v2(
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
                ptr::null_mut()
            ),
            INVALID
        );
        let mut bytes = [0xA5_u8; 120];
        let offset = (8 - bytes.as_ptr() as usize % 8) % 8 + 1;
        assert_eq!(
            toklet_session_generate_observed_v2(
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
                bytes.as_mut_ptr().add(offset).cast(),
                ptr::null_mut()
            ),
            INVALID
        );
        assert_eq!(bytes, [0xA5; 120]);
        assert_eq!(cpu_snapshot(s, false), before);
        cleanup(m, s, c);
    }
}

struct V2ReentryProbe {
    session: *mut SessionHandle,
    cancel: *mut CancelHandle,
    code: u32,
    metrics: GenerationMetricsV2,
}
unsafe extern "C" fn probe_v2_busy(context: *mut c_void, _: TokenEvent) -> u32 {
    let probe = unsafe { &mut *context.cast::<V2ReentryProbe>() };
    let (code, _, metrics) = unsafe {
        run_v2(
            probe.session,
            probe.cancel,
            2,
            &[],
            1,
            &[],
            None,
            ptr::null_mut(),
        )
    };
    probe.code = code;
    probe.metrics = metrics;
    0
}

#[test]
fn v2_reentry_wrong_thread_and_poisoned_sessions_remain_unknown() {
    let (m, s, c) = super::super::tests::handles();
    let mut probe = V2ReentryProbe {
        session: s,
        cancel: c,
        code: u32::MAX,
        metrics: GenerationMetricsV2::default(),
    };
    unsafe {
        let (code, report, metrics) = run_v2(
            s,
            c,
            0,
            &[1, 7, 13, 3],
            1,
            &[],
            Some(probe_v2_busy),
            (&mut probe as *mut V2ReentryProbe).cast(),
        );
        assert_eq!((code, probe.code), (OK, BUSY));
        assert_v2_unknown(probe.metrics);
        assert_v2_counts(metrics, report, [0, 4, 1, 1, 0, 5, 4, 1]);
    }
    let addresses = (s as usize, c as usize);
    let (code, _, metrics) = thread::spawn(move || unsafe {
        run_v2(
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
    assert_v2_unknown(metrics);
    unsafe {
        let result: Result<()> = (*s).access(|_| panic!("V2 metric poison control"));
        assert!(matches!(result, Err(Failure(PANIC, _))));
        let (code, _, metrics) = run_v2(s, c, 2, &[], 1, &[], None, ptr::null_mut());
        assert_eq!(code, POISONED);
        assert_v2_unknown(metrics);
        cleanup(m, s, c);
    }
}
