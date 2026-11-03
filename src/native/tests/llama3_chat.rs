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
    let folder = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("models/MLX-Llama-3.2-1B-Instruct-8bit-d48cdf0");
    for &(name, size, digest) in artifact::FILES {
        artifact::hash_file(&folder.join(name), size, digest).unwrap();
    }
    let codec = crate::app_chat::llama3::Codec::load(&folder).unwrap();
    let prompt = codec.encode(&message("Count from one to twenty.")).unwrap();
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
    unsafe {
        assert_eq!(
            toklet_model_load_mlx(path.as_ptr(), path.len(), &options, &mut m, &mut error),
            OK,
            "{}",
            String::from_utf8_lossy(&error.message)
        );
        assert_eq!(toklet_session_create(m, &options, &mut s, &mut error), OK);
        assert_eq!(toklet_cancel_create(&mut c, &mut error), OK);
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
                codec.stop_ids().as_ptr(),
                codec.stop_ids().len(),
                &mut admission,
                &mut error
            ),
            OK
        );
    }
    assert_eq!(admission.required_capacity, 512);
    let (code, exact) = unsafe { request(s, c, &boundary, 1, codec.stop_ids(), &mut output) };
    assert_eq!(code, OK);
    check_output(&exact, &output, codec.stop_ids());
    let boundary_stop = (exact.reason == 2).then_some(exact.stop_token);
    let before = unsafe { history(s) };
    let emitted_before = output.ids.len();
    let (code, rejected) = unsafe { request(s, c, &boundary, 2, codec.stop_ids(), &mut output) };
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
    let (code, cancelled) = unsafe { request(s, c, &prompt, 4, codec.stop_ids(), &mut output) };
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
    let (code, recovered) = unsafe { request(s, c, &prompt, 2, codec.stop_ids(), &mut output) };
    assert_eq!(code, OK);
    assert_eq!(output.ids.first(), selected.first());
    assert!(output.ids.iter().all(|id| !codec.stop_ids().contains(id)));
    assert!(!codec.decode(&output.ids).unwrap().is_empty());
    check_output(&recovered, &output, codec.stop_ids());
    println!(
        "LLAMA3_CHAT {}",
        serde_json::json!({"repository":artifact::REPOSITORY,"revision":artifact::REVISION,"capacity":512,"boundary_prompt":boundary.len(),"exact_required":admission.required_capacity,"overflow_rejected":true,"cancelled_pending":cancelled.pending_tokens,"recovered_ids":output.ids,"boundary_stop":boundary_stop,"recovery_stop":if recovered.reason==2 {Some(recovered.stop_token)}else {None},"app_qualified":false,"quality_qualified":false,"phone_qualified":false})
    );
    unsafe {
        cleanup(m, s, c);
    }
    for &(name, size, digest) in artifact::FILES {
        artifact::hash_file(&folder.join(name), size, digest).unwrap();
    }
}
