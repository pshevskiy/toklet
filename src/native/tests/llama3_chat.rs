//! Explicit cached Mac codec + native-session composition, not app admission.
use super::*;
#[path = "../../../examples/support/llama3_artifact.rs"]
mod artifact;

fn message(text: &str) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!([{"role":"user","content":text}])).unwrap()
}
unsafe fn history(s: *mut SessionHandle) -> Vec<u32> {
    let mut ids = vec![0; 512];
    let mut count = 0;
    assert_eq!(
        unsafe {
            toklet_session_history(s, ids.as_mut_ptr(), ids.len(), &mut count, ptr::null_mut())
        },
        OK
    );
    ids.truncate(count);
    ids
}
struct TextCopy(std::path::PathBuf);
impl TextCopy {
    fn new(source: &std::path::Path, instance: u64) -> Self {
        let directory = std::env::temp_dir().join(format!(
            "toklet-llama3-text-{}-{instance}",
            std::process::id()
        ));
        std::fs::create_dir(&directory).unwrap();
        let copy = Self(directory);
        for name in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ] {
            std::fs::copy(source.join(name), copy.0.join(name)).unwrap();
        }
        let path = copy.0.join("tokenizer.json");
        let mut bytes = std::fs::read(&path).unwrap();
        let at = bytes.iter().position(|&b| b == b' ').unwrap();
        bytes[at] = b'\t';
        std::fs::write(path, bytes).unwrap();
        copy
    }
}
impl Drop for TextCopy {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
struct Output {
    ids: Vec<u32>,
    cancel: *mut CancelHandle,
    cancel_status: u32,
}
unsafe extern "C" fn emit(context: *mut c_void, token: TokenEvent) -> u32 {
    // SAFETY: synchronous callback uses the live stack Output; no session calls.
    let output = unsafe { &mut *context.cast::<Output>() };
    output.ids.push(token.token_id);
    if !output.cancel.is_null() {
        output.cancel_status = unsafe { toklet_cancel_set(output.cancel, 1, ptr::null_mut()) };
        if output.cancel_status != OK {
            return 1;
        }
    }
    0
}
unsafe fn request(
    s: *mut SessionHandle,
    c: *mut CancelHandle,
    ids: &[u32],
    maximum: u32,
    stops: &[u32],
    output: &mut Output,
) -> (u32, Report) {
    let mut r = Report::default();
    let code = unsafe {
        toklet_session_generate(
            s,
            3,
            ids.as_ptr(),
            ids.len(),
            maximum,
            stops.as_ptr(),
            stops.len(),
            c,
            Some(emit),
            (output as *mut Output).cast(),
            &mut r,
            ptr::null_mut(),
        )
    };
    (code, r)
}
fn check_output(report: &Report, output: &Output, stops: &[u32]) {
    assert_eq!(output.ids.len(), report.emitted_tokens as usize);
    assert!(output.ids.iter().all(|id| !stops.contains(id)));
    match report.reason {
        1 => assert_eq!(report.stop_token, u32::MAX),
        2 => assert!(stops.contains(&report.stop_token)),
        other => panic!("unexpected reason {other}"),
    }
}
#[test]
#[ignore = "explicit pinned Llama Mac512 codec/session boundary and cancellation; no app admission"]
fn pinned_codec_native_session_boundary_and_recovery() {
    run(false);
}
#[test]
#[ignore = "explicit audited Llama Mac512 model/complete codec owner and generation"]
fn pinned_audited_pair_boundary_and_recovery() {
    run(true);
}
fn run(audited: bool) {
    let folder = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("models/MLX-Llama-3.2-1B-Instruct-8bit-d48cdf0");
    for &(name, size, digest) in artifact::FILES {
        artifact::hash_file(&folder.join(name), size, digest).unwrap();
    }
    let codec = crate::app_chat::llama3::Codec::load(&folder).unwrap();
    let mut prompt = codec.encode(&message("Count from one to twenty.")).unwrap();
    let mut stops = codec.stop_ids().to_vec();
    let boundary = (1..600)
        .find_map(|n| {
            let ids = codec.encode(&message(&" x".repeat(n))).unwrap();
            (ids.len() == 511).then_some(ids)
        })
        .expect("real renderer must provide a measured511-token prompt");
    assert_eq!(boundary.len(), 511);
    assert!(boundary.iter().all(|&id| (id as usize) < artifact::VOCAB));
    let options = Options {
        capacity: 512,
        chunk: 32,
        memory_limit_bytes: 6 * 1024 * 1024 * 1024 - 256 * 1024 * 1024,
        ..opts()
    };
    let path = folder.to_str().unwrap();
    let (mut m, mut s, mut c) = (ptr::null_mut(), ptr::null_mut(), ptr::null_mut());
    let mut error = NativeError {
        code: 0,
        message: [0; 256],
    };
    let mut chat = ptr::null_mut();
    unsafe {
        let status = if audited {
            toklet_model_load_mlx_llama3_audited(
                path.as_ptr(),
                path.len(),
                &options,
                &mut m,
                &mut error,
            )
        } else {
            toklet_model_load_mlx(path.as_ptr(), path.len(), &options, &mut m, &mut error)
        };
        assert_eq!(status, OK, "{}", String::from_utf8_lossy(&error.message));
        assert_eq!(toklet_session_create(m, &options, &mut s, &mut error), OK);
        assert_eq!(toklet_cancel_create(&mut c, &mut error), OK);
    }
    unsafe {
        // Legacy config selection stays closed even for the exact Llama directory.
        assert_ne!(
            text::toklet_chat_create(m, 256 * 1024 * 1024, &mut chat, &mut error),
            OK
        );
        assert!(chat.is_null());
        let status =
            text::toklet_chat_create_complete_audited(m, 256 * 1024 * 1024, &mut chat, &mut error);
        if audited {
            assert_eq!(status, OK, "{}", String::from_utf8_lossy(&error.message));
            let mut ids = vec![0; 32768];
            let mut count = 0;
            let json = message("Count from one to twenty.");
            assert_eq!(
                text::toklet_chat_encode(
                    chat,
                    json.as_ptr(),
                    json.len(),
                    ids.as_mut_ptr(),
                    ids.len(),
                    &mut count,
                    &mut error
                ),
                OK
            );
            ids.truncate(count);
            assert_eq!(ids, prompt);
            prompt = ids;
            let mut native_stops = [0; 8];
            assert_eq!(
                text::toklet_chat_stop_ids(
                    chat,
                    native_stops.as_mut_ptr(),
                    native_stops.len(),
                    &mut count,
                    &mut error
                ),
                OK
            );
            assert_eq!(&native_stops[..count], codec.stop_ids());
            stops = native_stops[..count].to_vec();
            let mut identity = ptr::null_mut();
            assert_eq!(
                text::toklet_model_chat_identity_create(m, chat, &mut identity, &mut error),
                OK
            );
            let mut bytes = vec![0; 8192];
            assert_eq!(
                toklet_mlx_inspect_json(
                    identity,
                    bytes.as_mut_ptr(),
                    bytes.len(),
                    &mut count,
                    &mut error
                ),
                OK
            );
            bytes.truncate(count);
            let report: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(report["repository"], artifact::REPOSITORY);
            assert_eq!(report["revision"], artifact::REVISION);
            assert_eq!(report["loaded_position_limit"], 512);
            assert_eq!(
                report["preference_namespace"]["text"],
                "llama3-plain-20240726-v1"
            );
            assert_eq!(report["preference_namespace"]["load"], "llama3_audited_v1");
            assert_eq!(report["assets"].as_array().unwrap().len(), 6);
            assert_eq!(report["exact_model_pair"], true);
            assert_eq!(report["qualification_evaluated"], false);
            println!("LLAMA3_IDENTITY {}", report);
            assert_eq!(toklet_mlx_inspect_release(identity, &mut error), OK);
            let before = Rc::strong_count(&(*m).model);
            let mut rejected = ptr::dangling_mut();
            assert_eq!(
                text::toklet_chat_create_complete_audited(
                    m,
                    256 * 1024 * 1024 - 1,
                    &mut rejected,
                    &mut error
                ),
                INVALID
            );
            assert!(rejected.is_null());
            // Private text-only copy: no cached artifact or model weight is modified.
            let copy = TextCopy::new(&folder, (*m).instance);
            let original = std::mem::replace(&mut (*m).directory, copy.0.clone());
            let status = text::toklet_chat_create_complete_audited(
                m,
                256 * 1024 * 1024,
                &mut rejected,
                &mut error,
            );
            (*m).directory = original;
            assert_eq!(status, LOAD_ERROR);
            assert!(rejected.is_null());
            assert_eq!(Rc::strong_count(&(*m).model), before);
            // The exact allocation is retained; another model instance is rejected.
            let other = ModelHandle {
                owner: thread::current().id(),
                instance: context::next_instance().unwrap(),
                model: Rc::new(NativeModel::Cpu(Box::new(model()))),
                directory: folder.clone(),
            };
            assert_eq!(
                text::toklet_model_chat_identity_create(&other, chat, &mut identity, &mut error),
                INVALID
            );
            assert!(identity.is_null());
        } else {
            assert_eq!(status, UNSUPPORTED);
            assert!(chat.is_null());
        }
    }
    let mut output = Output {
        ids: Vec::new(),
        cancel: ptr::null_mut(),
        cancel_status: OK,
    };
    let mut admission = RequestAdmission::default();
    unsafe {
        assert_eq!(
            toklet_session_preflight(
                s,
                3,
                boundary.as_ptr(),
                boundary.len(),
                1,
                stops.as_ptr(),
                stops.len(),
                &mut admission,
                &mut error
            ),
            OK
        );
    }
    assert_eq!(admission.required_capacity, 512);
    let (code, exact) = unsafe { request(s, c, &boundary, 1, &stops, &mut output) };
    assert_eq!(code, OK);
    check_output(&exact, &output, &stops);
    let boundary_stop = (exact.reason == 2).then_some(exact.stop_token);
    let before = unsafe { history(s) };
    let emitted_before = output.ids.len();
    let (code, rejected) = unsafe { request(s, c, &boundary, 2, &stops, &mut output) };
    assert_eq!(code, INVALID);
    assert_eq!(output.ids.len(), emitted_before);
    assert_eq!(rejected.emitted_tokens, 0);
    assert_eq!(unsafe { history(s) }, before);
    assert_eq!(
        (
            rejected.history_tokens,
            rejected.evaluated_tokens,
            rejected.pending_tokens
        ),
        (
            exact.history_tokens,
            exact.evaluated_tokens,
            exact.pending_tokens
        )
    );
    output.ids.clear();
    output.cancel = c;
    let (code, cancelled) = unsafe { request(s, c, &prompt, 4, &stops, &mut output) };
    assert_eq!(code, OK);
    assert_eq!(output.cancel_status, OK);
    assert_eq!(
        (
            cancelled.reason,
            cancelled.emitted_tokens,
            cancelled.pending_tokens
        ),
        (3, 1, 1)
    );
    assert_eq!(cancelled.evaluated_tokens as usize, prompt.len());
    let selected = output.ids.clone();
    assert!(!codec.decode(&selected).unwrap().is_empty());
    let mut expected = prompt.clone();
    expected.extend_from_slice(&selected);
    assert_eq!(unsafe { history(s) }, expected);
    unsafe {
        assert_eq!(toklet_cancel_set(c, 0, &mut error), OK);
    }
    output.ids.clear();
    output.cancel = ptr::null_mut();
    let (code, recovered) = unsafe { request(s, c, &prompt, 2, &stops, &mut output) };
    assert_eq!(code, OK);
    assert_eq!(output.ids.first(), selected.first());
    assert!(output.ids.iter().all(|id| !stops.contains(id)));
    assert!(!codec.decode(&output.ids).unwrap().is_empty());
    check_output(&recovered, &output, &stops);
    println!(
        "LLAMA3_CHAT {}",
        serde_json::json!({"repository":artifact::REPOSITORY,"revision":artifact::REVISION,"audited_pair":audited,"capacity":512,"boundary_prompt":boundary.len(),"exact_required":admission.required_capacity,"overflow_rejected":true,"cancelled_pending":cancelled.pending_tokens,"recovered_ids":output.ids,"boundary_stop":boundary_stop,"recovery_stop":if recovered.reason==2 {Some(recovered.stop_token)}else {None},"app_qualified":false,"quality_qualified":false,"phone_qualified":false})
    );
    unsafe {
        if audited {
            let mut bytes = vec![0; 65536];
            let mut count = 0;
            assert_eq!(
                text::toklet_chat_decode(
                    chat,
                    output.ids.as_ptr(),
                    output.ids.len(),
                    bytes.as_mut_ptr(),
                    bytes.len(),
                    &mut count,
                    &mut error
                ),
                OK
            );
            bytes.truncate(count);
            assert_eq!(
                String::from_utf8(bytes).unwrap(),
                codec.decode(&output.ids).unwrap()
            );
        }
        let weak = Rc::downgrade(&(*m).model);
        cleanup(m, s, c);
        assert_eq!(weak.upgrade().is_some(), audited);
        assert_eq!(text::toklet_chat_release(chat, &mut error), OK);
        assert!(weak.upgrade().is_none());
    }
    for &(name, size, digest) in artifact::FILES {
        artifact::hash_file(&folder.join(name), size, digest).unwrap();
    }
}
