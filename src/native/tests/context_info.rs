use super::*;
use crate::model::gemma_reference::tests::Measurement;

unsafe fn context(s: *mut SessionHandle) -> ContextInfo {
    let mut info = ContextInfo::default();
    assert_eq!(
        unsafe { toklet_session_context(s, &mut info, ptr::null_mut()) },
        OK
    );
    info
}
unsafe fn preflight(
    s: *mut SessionHandle,
    op: u32,
    tokens: &[u32],
    maximum: u32,
) -> (u32, RequestAdmission) {
    let mut info = RequestAdmission::default();
    let code = unsafe {
        toklet_session_preflight(
            s,
            op,
            tokens.as_ptr(),
            tokens.len(),
            maximum,
            ptr::null(),
            0,
            &mut info,
            ptr::null_mut(),
        )
    };
    (code, info)
}

#[test]
fn context_identity_unknown_legacy_and_retained_model_lifetime() {
    assert_eq!(size_of::<ContextInfo>(), 136);
    assert_eq!(size_of::<RequestAdmission>(), 192);
    let (m, s, c) = handles();
    let mut other = ptr::null_mut();
    unsafe {
        assert_eq!(
            toklet_session_create(m, &opts(), &mut other, ptr::null_mut()),
            OK
        );
        let first = context(s);
        let second = context(other);
        assert!(first.model_instance > 0 && first.session_instance > 0);
        assert_eq!(first.model_instance, (*m).instance);
        assert_eq!(second.model_instance, first.model_instance);
        assert_ne!(second.session_instance, first.session_instance);
        assert_eq!((first.abi_version, first.struct_size), (ABI_VERSION, 136));
        assert_eq!(
            (
                first.backend,
                first.known_fields,
                first.declared_total_positions,
                first.declared_input_tokens,
                first.declared_output_tokens
            ),
            (0, 0, 0, 0, 0)
        );
        assert_eq!(
            (
                first.engine_global_position_ceiling,
                first.loaded_position_limit,
                first.session_capacity,
                first.absolute_position_offset
            ),
            (32768, 32, 32, 0)
        );
        assert_eq!(
            (
                first.context_qualification,
                first.whole_app_device_fit,
                first.reserved
            ),
            (CONTEXT_NOT_EVALUATED, CONTEXT_NOT_EVALUATED, [0; 2])
        );
        let weak = Rc::downgrade(&(*m).model);
        assert_eq!(toklet_model_release(m, ptr::null_mut()), OK);
        let measured = Measurement::start();
        let again = context(s);
        let allocations = measured.finish();
        assert_eq!(allocations, 0);
        assert_eq!(again, first);
        assert_eq!(toklet_session_reset(s, ptr::null_mut()), OK);
        assert_eq!(context(s), first);
        assert_eq!(toklet_session_release(s, ptr::null_mut()), OK);
        assert!(weak.upgrade().is_some());
        assert_eq!(toklet_session_release(other, ptr::null_mut()), OK);
        assert!(weak.upgrade().is_none());
        assert_eq!(toklet_cancel_release(c, ptr::null_mut()), OK);
        println!(
            "CONTEXT_NATIVE_LIFETIME {{\"passed\":true,\"model_instance\":{},\"session_instance\":{},\"second_session_instance\":{},\"declared_known\":false,\"loaded\":32,\"selected\":32,\"query_allocations\":0}}",
            first.model_instance, first.session_instance, second.session_instance
        );
    }
}

#[test]
fn shared_preflight_counts_pending_and_preserves_state_without_reservation() {
    let (m, s, c) = handles();
    unsafe {
        assert_eq!(
            generate(s, c, &[1, 7, 13, 3], 0, 1, None, ptr::null_mut()).0,
            OK
        );
        let old = context(s);
        assert_eq!(
            (old.history_tokens, old.committed_tokens, old.pending_tokens),
            (5, 4, 1)
        );
        for (op, tokens, maximum, base, after) in [
            (0, &[1, 7][..], 30, 0, 2),
            (3, &[1, 7][..], 30, 0, 2),
            (1, &[1, 7][..], 25, 5, 7),
            (2, &[][..], 27, 5, 5),
            (2, &[][..], 0, 5, 5),
        ] {
            let measured = Measurement::start();
            let (code, admission) = preflight(s, op, tokens, maximum);
            let allocations = measured.finish();
            assert_eq!(allocations, 0);
            assert_eq!(code, OK);
            assert_eq!(admission.context, old);
            assert_eq!(admission.operation, op);
            assert_eq!(admission.reserved, 0);
            assert_eq!(
                (
                    admission.base_history_tokens,
                    admission.input_tokens,
                    admission.history_after_input_tokens
                ),
                (base, tokens.len() as u64, after)
            );
            assert_eq!(admission.required_capacity, after + u64::from(maximum));
            assert_eq!(admission.remaining_tokens, 32 - admission.required_capacity);
            assert_eq!(context(s), old);
            println!(
                "CONTEXT_NATIVE_ADMISSION {}",
                serde_json::json!({"operation":op,"base":base,"new_input":tokens.len(),"history":after,"maximum":maximum,"required":admission.required_capacity,"remaining":admission.remaining_tokens,"allocations":allocations})
            );
        }
        for (op, tokens, maximum) in [
            (0, &[1, 7][..], 31),
            (3, &[1, 7][..], 31),
            (1, &[1, 7][..], 26),
            (2, &[][..], 28),
            (2, &[1][..], 1),
            (4, &[][..], 1),
            (0, &[][..], 1),
            (0, &[64][..], 1),
        ] {
            let mut out = RequestAdmission {
                required_capacity: 777,
                ..RequestAdmission::default()
            };
            let before = out;
            assert_eq!(
                toklet_session_preflight(
                    s,
                    op,
                    tokens.as_ptr(),
                    tokens.len(),
                    maximum,
                    ptr::null(),
                    0,
                    &mut out,
                    ptr::null_mut()
                ),
                INVALID
            );
            assert_eq!(out, before);
            assert_eq!(context(s), old);
        }
        let (code, admission) = preflight(s, 1, &[1], 26);
        assert_eq!(code, OK);
        assert_eq!(admission.required_capacity, 32);
        // Intervening generation invalidates the old arithmetic snapshot.
        assert_eq!(generate(s, c, &[], 2, 1, None, ptr::null_mut()).0, OK);
        let changed = context(s);
        assert_eq!(
            generate(s, c, &[1], 1, 26, None, ptr::null_mut()).0,
            INVALID
        );
        assert_eq!(context(s), changed);
        cleanup(m, s, c);
    }
}

#[test]
fn context_queries_reject_null_wrong_thread_busy_and_poison_without_writes() {
    let (m, s, c) = handles();
    unsafe {
        let mut out = ContextInfo {
            session_capacity: 777,
            ..ContextInfo::default()
        };
        let old = out;
        assert_eq!(
            toklet_session_context(ptr::null(), &mut out, ptr::null_mut()),
            INVALID
        );
        assert_eq!(out, old);
        assert_eq!(
            toklet_session_context(s, ptr::null_mut(), ptr::null_mut()),
            INVALID
        );
        let address = s as usize;
        thread::spawn(move || {
            let s = address as *mut SessionHandle;
            let mut out = ContextInfo {
                session_capacity: 777,
                ..ContextInfo::default()
            };
            let old = out;
            assert_eq!(
                toklet_session_context(s, &mut out, ptr::null_mut()),
                WRONG_THREAD
            );
            assert_eq!(out, old);
            let mut admission = RequestAdmission {
                required_capacity: 777,
                ..RequestAdmission::default()
            };
            let old = admission;
            assert_eq!(
                toklet_session_preflight(
                    s,
                    0,
                    [1].as_ptr(),
                    1,
                    1,
                    ptr::null(),
                    0,
                    &mut admission,
                    ptr::null_mut()
                ),
                WRONG_THREAD
            );
            assert_eq!(admission, old);
        })
        .join()
        .unwrap();
        (*s).access(|_| {
            assert_eq!(toklet_session_context(s, &mut out, ptr::null_mut()), BUSY);
            assert_eq!(out, old);
            let mut admission = RequestAdmission::default();
            assert_eq!(
                toklet_session_preflight(
                    s,
                    0,
                    [1].as_ptr(),
                    1,
                    1,
                    ptr::null(),
                    0,
                    &mut admission,
                    ptr::null_mut()
                ),
                BUSY
            );
            Ok(())
        })
        .unwrap();
        assert_eq!(
            (*s).access::<()>(|_| panic!("context poison witness"))
                .unwrap_err()
                .0,
            PANIC
        );
        assert_eq!(
            toklet_session_context(s, &mut out, ptr::null_mut()),
            POISONED
        );
        assert_eq!(out, old);
        cleanup(m, s, c);
    }
}
