// Reuse the existing tiny CPU model/session fixture; no audited model setter.
use super::*;
use crate::native::tests::{handles, model};
fn chat(retained: Option<Rc<NativeModel>>) -> *mut ChatHandle {
    Box::into_raw(Box::new(ChatHandle {
        owner: thread::current().id(),
        codec: Some(crate::app_chat::decode_fixture()),
        _model: retained,
    }))
}
#[test]
fn identity_getter_rejects_unpaired_cpu_and_distinct_equal_content_owners() {
    unsafe {
        let (m, s, c) = handles();
        let unpaired = chat(None);
        let paired = chat(Some(Rc::clone(&(*m).model)));
        let other = chat(Some(Rc::new(NativeModel::Cpu(Box::new(model())))));
        for (chat, status) in [
            (unpaired, UNSUPPORTED),
            (paired, UNSUPPORTED),
            (other, INVALID),
        ] {
            let before = Rc::strong_count(&(*m).model);
            let mut out = ptr::dangling_mut();
            assert_eq!(
                toklet_model_chat_identity_create(m, chat, &mut out, ptr::null_mut()),
                status
            );
            assert!(out.is_null());
            assert_eq!(Rc::strong_count(&(*m).model), before);
        }
        assert_eq!(toklet_chat_release(unpaired, ptr::null_mut()), OK);
        assert_eq!(toklet_chat_release(paired, ptr::null_mut()), OK);
        assert_eq!(toklet_chat_release(other, ptr::null_mut()), OK);
        assert_eq!(toklet_session_release(s, ptr::null_mut()), OK);
        assert_eq!(toklet_model_release(m, ptr::null_mut()), OK);
        assert_eq!(toklet_cancel_release(c, ptr::null_mut()), OK);
    }
}
#[test]
fn identity_getter_null_and_foreign_thread_leave_outputs_and_owners_valid() {
    unsafe {
        let (m, s, c) = handles();
        let chat = chat(Some(Rc::clone(&(*m).model)));
        let before = Rc::strong_count(&(*m).model);
        let mut out = ptr::dangling_mut();
        assert_eq!(
            toklet_model_chat_identity_create(ptr::null(), chat, &mut out, ptr::null_mut()),
            INVALID
        );
        assert!(out.is_null());
        out = ptr::dangling_mut();
        assert_eq!(
            toklet_model_chat_identity_create(m, ptr::null(), &mut out, ptr::null_mut()),
            INVALID
        );
        assert!(out.is_null());
        assert_eq!(
            toklet_model_chat_identity_create(m, chat, ptr::null_mut(), ptr::null_mut()),
            INVALID
        );
        let addresses = (m as usize, chat as usize);
        thread::spawn(move || {
            let mut out = ptr::dangling_mut();
            assert_eq!(
                toklet_model_chat_identity_create(
                    addresses.0 as *const ModelHandle,
                    addresses.1 as *const ChatHandle,
                    &mut out,
                    ptr::null_mut()
                ),
                WRONG_THREAD
            );
            assert!(out.is_null());
        })
        .join()
        .unwrap();
        // Owner-check chat independently rather than relying only on model's owner.
        let foreign = thread::spawn(|| thread::current().id()).join().unwrap();
        let original = (*chat).owner;
        (*chat).owner = foreign;
        out = ptr::dangling_mut();
        assert_eq!(
            toklet_model_chat_identity_create(m, chat, &mut out, ptr::null_mut()),
            WRONG_THREAD
        );
        assert!(out.is_null());
        (*chat).owner = original;
        assert_eq!(Rc::strong_count(&(*m).model), before);
        assert_eq!(toklet_chat_release(chat, ptr::null_mut()), OK);
        assert_eq!(toklet_session_release(s, ptr::null_mut()), OK);
        assert_eq!(toklet_model_release(m, ptr::null_mut()), OK);
        assert_eq!(toklet_cancel_release(c, ptr::null_mut()), OK);
    }
}
#[test]
fn identity_getter_does_not_borrow_or_poison_an_unrelated_busy_session() {
    unsafe {
        let (m, s, c) = handles();
        let chat = chat(Some(Rc::clone(&(*m).model)));
        let before = Rc::strong_count(&(*m).model);
        let guard = (*s).state.borrow_mut();
        let mut out = ptr::dangling_mut();
        assert_eq!(
            toklet_model_chat_identity_create(m, chat, &mut out, ptr::null_mut()),
            UNSUPPORTED
        );
        assert!(out.is_null());
        assert!(!(*s).poisoned.get());
        assert_eq!(Rc::strong_count(&(*m).model), before);
        drop(guard);
        let mut report = Report::default();
        assert_eq!(toklet_session_state(s, &mut report, ptr::null_mut()), OK);
        assert_eq!(report.history_tokens, 0);
        assert_eq!(toklet_chat_release(chat, ptr::null_mut()), OK);
        assert_eq!(toklet_session_release(s, ptr::null_mut()), OK);
        assert_eq!(toklet_model_release(m, ptr::null_mut()), OK);
        assert_eq!(toklet_cancel_release(c, ptr::null_mut()), OK);
    }
}
