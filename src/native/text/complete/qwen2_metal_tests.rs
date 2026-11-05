//! One explicitly selected cached assembly; no generation or numerical gate.
use super::*;
use std::io::Write;
struct Owners {
    model: *mut ModelHandle,
    chat: *mut ChatHandle,
    identity: *mut CompatibilityHandle,
}
impl Drop for Owners {
    fn drop(&mut self) {
        unsafe {
            let _ = toklet_mlx_inspect_release(self.identity, ptr::null_mut());
            let _ = toklet_chat_release(self.chat, ptr::null_mut());
            let _ = toklet_model_release(self.model, ptr::null_mut());
        }
    }
}
fn record(event: &str, value: serde_json::Value) {
    let bytes = serde_json::to_vec(&serde_json::json!({"event":event,"value":value})).unwrap();
    assert!(bytes.len() <= 64 * 1024);
    let mut out = std::io::stdout().lock();
    out.write_all(&bytes).unwrap();
    out.write_all(b"\n").unwrap();
    out.flush().unwrap();
}
unsafe fn captured(handle: *const CompatibilityHandle) -> Vec<u8> {
    unsafe {
        let mut count = 0;
        assert_eq!(
            toklet_mlx_inspect_json(handle, ptr::null_mut(), 0, &mut count, ptr::null_mut()),
            OK
        );
        assert!((1..8192).contains(&count));
        let mut bytes = vec![0; count];
        assert_eq!(
            toklet_mlx_inspect_json(
                handle,
                bytes.as_mut_ptr(),
                bytes.len(),
                &mut count,
                ptr::null_mut()
            ),
            OK
        );
        assert_eq!(count, bytes.len());
        bytes
    }
}
#[test]
#[ignore = "Explicit cached Qwen2 audited assembly, complete codec and identity; no generation"]
fn pinned_qwen2_complete_pair_identity_and_release() {
    const MODEL: u64 = 1024 * 1024 * 1024;
    const TEXT: u64 = 128 * 1024 * 1024;
    const _: () = assert!(MODEL + TEXT + 256 * 1024 * 1024 < 6 * 1024 * 1024 * 1024);
    let path = b"models/MLX-Qwen2.5-0.5B-Instruct-8bit-6474d4c";
    let options = Options {
        abi_version: ABI_VERSION,
        struct_size: size_of::<Options>() as u32,
        capacity: 512,
        chunk: 32,
        memory_limit_bytes: MODEL,
    };
    let mut owners = Owners {
        model: ptr::null_mut(),
        chat: ptr::null_mut(),
        identity: ptr::null_mut(),
    };
    let mut error = NativeError {
        code: 0,
        message: [0; 256],
    };
    unsafe {
        let status = toklet_model_load_mlx_qwen2_audited(
            path.as_ptr(),
            path.len(),
            &options,
            &mut owners.model,
            &mut error,
        );
        record(
            "load",
            serde_json::json!({"status":status,"nonnull":!owners.model.is_null(),"error_code":error.code}),
        );
        assert_eq!(status, OK);
        let instance = (*owners.model).instance;
        let weak = Rc::downgrade(&(*owners.model).model);
        assert_eq!(weak.strong_count(), 1);
        let source = {
            let NativeModel::Metal(model) = (*owners.model).model.as_ref() else {
                panic!("wrong backend")
            };
            assert!(model.mlx_source_content_identity().is_none());
            assert!(model.mlx_qwen3_source_content_identity().is_none());
            assert_eq!(
                model.native_attention_mode(),
                crate::metal::AttentionMode::Online
            );
            assert_eq!(
                model.native_session_modes(),
                (
                    crate::metal::ProjectionMode::MlxAffine,
                    crate::metal::NormalizationMode::Compensated
                )
            );
            *model.mlx_qwen2_source_content_identity().unwrap()
        };
        let missing =
            std::env::temp_dir().join(format!("toklet-no-complete-qwen2-{}", std::process::id()));
        assert!(!missing.exists());
        let directory = std::mem::replace(&mut (*owners.model).directory, missing);
        let mut rejected = ptr::dangling_mut();
        let status =
            toklet_chat_create_complete_audited(owners.model, TEXT, &mut rejected, &mut error);
        record(
            "missing-codec-directory",
            serde_json::json!({"status":status,"null":rejected.is_null(),"strong_count":weak.strong_count()}),
        );
        (*owners.model).directory = directory;
        assert_eq!(status, LOAD_ERROR);
        assert!(rejected.is_null());
        assert_eq!(weak.strong_count(), 1);
        // A Qwen2 assembled receipt cannot enter the legacy Qwen3-only factory.
        let mut rejected = ptr::dangling_mut();
        let status = toklet_chat_create_qwen3_audited(
            owners.model,
            256 * 1024 * 1024,
            &mut rejected,
            &mut error,
        );
        record(
            "wrong-profile",
            serde_json::json!({"status":status,"null":rejected.is_null(),"strong_count":weak.strong_count()}),
        );
        assert_eq!(status, UNSUPPORTED);
        assert!(rejected.is_null());
        assert_eq!(weak.strong_count(), 1);
        let status =
            toklet_chat_create_complete_audited(owners.model, TEXT, &mut owners.chat, &mut error);
        record(
            "complete",
            serde_json::json!({"status":status,"nonnull":!owners.chat.is_null(),"strong_count":weak.strong_count()}),
        );
        assert_eq!(status, OK);
        assert_eq!(weak.strong_count(), 2);
        assert!((*owners.chat).codec.qwen3_content_identity().is_none());
        assert_eq!(
            (*owners.chat)
                .codec
                .qwen2_content_identity()
                .unwrap()
                .source(),
            &source
        );
        assert_eq!(
            (*owners.chat)
                .codec
                .qwen2_content_identity()
                .unwrap()
                .text_policy(),
            "qwen-chatml-v1"
        );
        assert!(Rc::ptr_eq(
            (*owners.chat)._model.as_ref().unwrap(),
            &(*owners.model).model
        ));
        let status = toklet_model_chat_identity_create(
            owners.model,
            owners.chat,
            &mut owners.identity,
            &mut error,
        );
        record(
            "getter",
            serde_json::json!({"status":status,"nonnull":!owners.identity.is_null(),"strong_count":weak.strong_count()}),
        );
        assert_eq!(status, OK);
        let bytes = captured(owners.identity);
        let report: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        record("identity", report.clone());
        assert_eq!(report["model_instance"], instance);
        assert_eq!(report["loaded_position_limit"], 512);
        assert_eq!(report["preference_namespace"]["load"], "qwen2_audited_v1");
        assert_eq!(
            report["assets"][1]["roles"],
            serde_json::json!(["source_bytes_delivered_to_assembled_model"])
        );
        assert_eq!(report["source_container_physical_ranges"], 630);
        let model = std::mem::replace(&mut owners.model, ptr::null_mut());
        assert_eq!(toklet_model_release(model, &mut error), OK);
        assert_eq!(weak.strong_count(), 1);
        let mut stops = [0; 8];
        let mut count = 0;
        assert_eq!(
            toklet_chat_stop_ids(
                owners.chat,
                stops.as_mut_ptr(),
                stops.len(),
                &mut count,
                &mut error
            ),
            OK
        );
        assert_eq!(&stops[..count], &[151643, 151645]);
        let chat = std::mem::replace(&mut owners.chat, ptr::null_mut());
        assert_eq!(toklet_chat_release(chat, &mut error), OK);
        assert!(weak.upgrade().is_none());
        assert_eq!(captured(owners.identity), bytes);
        let identity = std::mem::replace(&mut owners.identity, ptr::null_mut());
        assert_eq!(toklet_mlx_inspect_release(identity, &mut error), OK);
        record(
            "closed",
            serde_json::json!({"passed":true,"model_loads":1,"all_owners_released":true,"snapshot_survived_owners":true,"generation":false,"app_or_phone_qualified":false}),
        );
    }
}

#[test]
#[ignore = "Explicit cached larger Qwen2.5 audited Reference and plain ChatML"]
fn pinned_qwen15_reference_pair_and_session() {
    const TEXT: u64 = 128 * 1024 * 1024;
    let path = b"models/MLX-Qwen2.5-1.5B-Instruct-8bit-c7e1ff2";
    let mut options = Options {
        abi_version: ABI_VERSION,
        struct_size: size_of::<Options>() as u32,
        capacity: 512,
        chunk: 32,
        memory_limit_bytes: 2 * 1024 * 1024 * 1024,
    };
    let mut owners = Owners {
        model: ptr::null_mut(),
        chat: ptr::null_mut(),
        identity: ptr::null_mut(),
    };
    let mut error = NativeError {
        code: 0,
        message: [0; 256],
    };
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/app-chat-qwen-v1/reference.json"
    ))
    .unwrap();
    unsafe {
        let mut excessive = options;
        excessive.capacity = 513;
        assert_eq!(
            toklet_model_load_mlx_qwen15_audited(
                path.as_ptr(),
                path.len(),
                &excessive,
                &mut owners.model,
                &mut error
            ),
            LOAD_ERROR
        );
        assert!(owners.model.is_null());
        assert_eq!(
            toklet_model_load_mlx_qwen15_audited(
                path.as_ptr(),
                path.len(),
                &options,
                &mut owners.model,
                &mut error
            ),
            OK
        );
        let NativeModel::Metal(model) = (*owners.model).model.as_ref() else {
            panic!("expected Metal");
        };
        assert!(model.mlx_qwen2_source_content_identity().is_none());
        assert_eq!(
            model.native_attention_mode(),
            crate::metal::AttentionMode::Reference
        );
        let (memory, _, _) = model
            .native_session_plan(session::Options {
                capacity: 512,
                prefill_chunk_size: 32,
                position_offset: 0,
                memory_limit_bytes: options.memory_limit_bytes as usize - BRIDGE_BYTES,
            })
            .unwrap();
        assert_eq!(
            toklet_chat_create_complete_audited(
                owners.model,
                TEXT - 1,
                &mut owners.chat,
                &mut error
            ),
            INVALID
        );
        assert!(owners.chat.is_null());
        assert_eq!(
            toklet_chat_create_complete_audited(owners.model, TEXT, &mut owners.chat, &mut error),
            OK
        );
        for case in fixture["cases"].as_array().unwrap() {
            let bytes = serde_json::to_vec(&case["messages"]).unwrap();
            let expected: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
            let mut actual = vec![0; expected.len()];
            let mut count = 0;
            assert_eq!(
                toklet_chat_encode(
                    owners.chat,
                    bytes.as_ptr(),
                    bytes.len(),
                    actual.as_mut_ptr(),
                    actual.len(),
                    &mut count,
                    &mut error
                ),
                OK
            );
            assert_eq!(count, expected.len());
            assert_eq!(actual, expected);
        }
        let mut stops = [0; 2];
        let mut count = 0;
        assert_eq!(
            toklet_chat_stop_ids(
                owners.chat,
                stops.as_mut_ptr(),
                stops.len(),
                &mut count,
                &mut error
            ),
            OK
        );
        assert_eq!((count, stops), (2, [151643, 151645]));
        assert_eq!(
            toklet_model_chat_identity_create(
                owners.model,
                owners.chat,
                &mut owners.identity,
                &mut error
            ),
            OK
        );
        let identity: serde_json::Value =
            serde_json::from_slice(&captured(owners.identity)).unwrap();
        assert_eq!(
            identity["preference_namespace"]["load"],
            "qwen15_audited_v1"
        );
        assert_eq!(
            identity["preference_namespace"]["manifest_sha256"],
            "775a61717b5785783a75a19a6a472fb8eb9ea1e5a8efbbf4c919330e7c6c209c"
        );
        assert_eq!(identity["loaded_position_limit"], 512);

        let mut native = ptr::null_mut();
        options.memory_limit_bytes = (memory.admitted_bytes + BRIDGE_BYTES - 1) as u64;
        assert_eq!(
            toklet_session_create(owners.model, &options, &mut native, &mut error),
            ENGINE_ERROR
        );
        assert!(native.is_null());
        options.memory_limit_bytes += 1;
        assert_eq!(
            toklet_session_create(owners.model, &options, &mut native, &mut error),
            OK
        );
        let mut state = Report::default();
        assert_eq!(toklet_session_state(native, &mut state, &mut error), OK);
        assert_eq!(state.admitted_bytes, options.memory_limit_bytes);
        assert_eq!(toklet_session_release(native, &mut error), OK);
        record("qwen15_identity", identity);
        record(
            "qwen15_admission",
            serde_json::json!({"attention":"Reference",
            "admitted_bytes":state.admitted_bytes,"text_allowance":TEXT,"loaded":512,"chunk":32}),
        );
    }
}
