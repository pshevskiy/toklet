use super::*;
use crate::native::tests::handles;
#[test]
fn complete_factory_cpu_budget_and_null_failures_add_no_owner() {
    unsafe {
        let (model, session, cancel) = handles();
        let before = Rc::strong_count(&(*model).model);
        for (budget, status) in [
            (0, INVALID),
            (64 * 1024 * 1024 - 1, INVALID),
            (6 * 1024 * 1024 * 1024 + 1, INVALID),
            (64 * 1024 * 1024, UNSUPPORTED),
            (128 * 1024 * 1024, UNSUPPORTED),
            (256 * 1024 * 1024, UNSUPPORTED),
        ] {
            let mut out = ptr::dangling_mut();
            assert_eq!(
                toklet_chat_create_complete_audited(model, budget, &mut out, ptr::null_mut()),
                status
            );
            assert!(out.is_null());
            assert_eq!(Rc::strong_count(&(*model).model), before);
        }
        let mut out = ptr::dangling_mut();
        assert_eq!(
            toklet_chat_create_complete_audited(ptr::null(), 0, &mut out, ptr::null_mut()),
            INVALID
        );
        assert!(out.is_null());
        assert_eq!(
            toklet_chat_create_complete_audited(model, 0, ptr::null_mut(), ptr::null_mut()),
            INVALID
        );
        assert_eq!(toklet_session_release(session, ptr::null_mut()), OK);
        assert_eq!(toklet_model_release(model, ptr::null_mut()), OK);
        assert_eq!(toklet_cancel_release(cancel, ptr::null_mut()), OK);
    }
}
#[test]
fn complete_factory_foreign_thread_does_not_open_or_retain() {
    unsafe {
        let (model, session, cancel) = handles();
        let before = Rc::strong_count(&(*model).model);
        let address = model as usize;
        thread::spawn(move || {
            let mut out = ptr::dangling_mut();
            assert_eq!(
                toklet_chat_create_complete_audited(
                    address as *const ModelHandle,
                    64 * 1024 * 1024,
                    &mut out,
                    ptr::null_mut()
                ),
                WRONG_THREAD
            );
            assert!(out.is_null());
        })
        .join()
        .unwrap();
        assert_eq!(Rc::strong_count(&(*model).model), before);
        assert_eq!(toklet_session_release(session, ptr::null_mut()), OK);
        assert_eq!(toklet_model_release(model, ptr::null_mut()), OK);
        assert_eq!(toklet_cancel_release(cancel, ptr::null_mut()), OK);
    }
}
