//! Explicit one-model ownership/codec probe, never an ordinary test dependency.
use super::*;
use std::io::Write;
struct Handles {
    model: *mut ModelHandle,
    chat: *mut ChatHandle,
}
impl Drop for Handles {
    fn drop(&mut self) {
        unsafe {
            let _ = toklet_chat_release(self.chat, ptr::null_mut());
            let _ = toklet_model_release(self.model, ptr::null_mut());
        }
    }
}
fn record(label: &str, value: serde_json::Value) {
    let bytes = serde_json::to_vec(&value).unwrap();
    assert!(bytes.len() <= 64 * 1024);
    let mut out = std::io::stdout().lock();
    out.write_all(label.as_bytes()).unwrap();
    out.write_all(b" ").unwrap();
    out.write_all(&bytes).unwrap();
    out.write_all(b"\n").unwrap();
    out.flush().unwrap();
}
#[test]
#[ignore = "Explicit cached audited Qwen3 Metal assembly and paired codec; no generation or context expansion"]
fn pinned_qwen3_audited_model_codec_retains_exact_owner() {
    const NUMERICAL: u64 = 1536 * 1024 * 1024;
    const TEXT: u64 = 256 * 1024 * 1024;
    // Explicit numerical + text + bounded probe/app reserve, below 6 GiB process
    // policy; each native admission remains checked. Not a device-fit proof.
    const _: () = assert!(NUMERICAL + TEXT + 256 * 1024 * 1024 < 6 * 1024 * 1024 * 1024);
    let directory = Path::new("models/MLX-Qwen3-0.6B-8bit-11de968");
    let path = directory.to_str().unwrap().as_bytes();
    let opts = Options {
        abi_version: ABI_VERSION,
        struct_size: size_of::<Options>() as u32,
        capacity: 512,
        chunk: 32,
        memory_limit_bytes: NUMERICAL,
    };
    let mut handles = Handles {
        model: ptr::null_mut(),
        chat: ptr::null_mut(),
    };
    let mut error = NativeError {
        code: 0,
        message: [0; 256],
    };
    let status = unsafe {
        toklet_model_load_mlx_qwen3_audited(
            path.as_ptr(),
            path.len(),
            &opts,
            &mut handles.model,
            &mut error,
        )
    };
    record(
        "QWEN_AUDITED_LOAD",
        serde_json::json!({"status":status,"nonnull":!handles.model.is_null(),"error_code":error.code}),
    );
    assert_eq!(status, OK);
    assert!(!handles.model.is_null());
    unsafe {
        let instance = (*handles.model).instance;
        let weak = Rc::downgrade(&(*handles.model).model);
        let source = {
            let NativeModel::Metal(m) = (*handles.model).model.as_ref() else {
                panic!("wrong backend")
            };
            assert!(m.mlx_source_content_identity().is_none());
            assert_eq!(m.config().max_positions, 512);
            *m.mlx_qwen3_source_content_identity().unwrap()
        };
        let initial = Rc::strong_count(&(*handles.model).model);
        assert_eq!(initial, 1);
        // Correct live audited owner but missing codec location must fail without
        // changing the source receipt or retaining a partial chat/model owner.
        let missing = std::env::temp_dir().join(format!(
            "toklet-no-qwen-codec-assets-{}",
            std::process::id()
        ));
        assert!(!missing.exists());
        let original = std::mem::replace(&mut (*handles.model).directory, missing);
        let mut rejected = ptr::dangling_mut();
        let status =
            toklet_chat_create_qwen3_audited(handles.model, TEXT, &mut rejected, &mut error);
        record(
            "QWEN_AUDITED_REJECT",
            serde_json::json!({"status":status,"null":rejected.is_null(),"strong_count":Rc::strong_count(&(*handles.model).model)}),
        );
        assert_eq!(status, LOAD_ERROR);
        assert!(rejected.is_null());
        assert_eq!(Rc::strong_count(&(*handles.model).model), 1);
        (*handles.model).directory = original;
        let status =
            toklet_chat_create_smol_audited(handles.model, TEXT, &mut rejected, &mut error);
        assert_eq!(status, UNSUPPORTED);
        assert!(rejected.is_null());
        assert_eq!(Rc::strong_count(&(*handles.model).model), 1);
        let status =
            toklet_chat_create_qwen3_audited(handles.model, TEXT, &mut handles.chat, &mut error);
        assert_eq!(status, OK);
        assert!(!handles.chat.is_null());
        let paired = &*handles.chat;
        assert!(Rc::ptr_eq(
            paired._model.as_ref().unwrap(),
            &(*handles.model).model
        ));
        let content = *paired.codec.qwen3_content_identity().unwrap();
        assert_eq!(content.source(), &source);
        assert!(paired.codec.content_identity().is_none());
        record(
            "QWEN_AUDITED_PAIR",
            serde_json::json!({
                "model_instance":instance,"strong_before":initial,"strong_paired":Rc::strong_count(&(*handles.model).model),"exact_rc":true,
                "source_expected_manifest":source.expected_manifest_sha256(),"source_config":source.config_sha256(),"source_index":source.index_sha256(),"source_container":source.container_sha256(),
                "source_container_bytes":source.container_bytes(),"source_ranges":source.container_physical_ranges(),"source_reads":source.verified_container_reads(),
                "complete_manifest":content.complete_manifest_sha256(),"tokenizer_consumed":content.tokenizer_sha256(),"tokenizer_config_verified_policy":content.tokenizer_config_policy_sha256(),"special_map_verified_ancillary":content.ancillary_special_map_sha256(),"text_policy":content.text_policy()
            }),
        );
        assert_eq!(Rc::strong_count(&(*handles.model).model), 2);
        let model_pointer = std::mem::replace(&mut handles.model, ptr::null_mut());
        assert_eq!(toklet_model_release(model_pointer, &mut error), OK);
        assert_eq!(weak.strong_count(), 1);
        let json = br#"[{"role":"user","content":"hello"}]"#;
        let mut ids = [0_u32; MAX_TOKENS];
        let mut count = 0;
        assert_eq!(
            toklet_chat_encode(
                handles.chat,
                json.as_ptr(),
                json.len(),
                ids.as_mut_ptr(),
                ids.len(),
                &mut count,
                &mut error
            ),
            OK
        );
        assert!(count > 0 && count <= MAX_RESPONSE_TOKENS);
        let mut decoded = [0_u8; MAX_TEXT];
        let mut text_count = 0;
        assert_eq!(
            toklet_chat_decode(
                handles.chat,
                ids.as_ptr(),
                count,
                decoded.as_mut_ptr(),
                decoded.len(),
                &mut text_count,
                &mut error
            ),
            OK
        );
        let text = std::str::from_utf8(&decoded[..text_count]).unwrap();
        assert!(text.contains("hello"));
        let mut stops = [0_u32; 8];
        let mut stop_count = 0;
        assert_eq!(
            toklet_chat_stop_ids(
                handles.chat,
                stops.as_mut_ptr(),
                stops.len(),
                &mut stop_count,
                &mut error
            ),
            OK
        );
        assert_eq!(&stops[..stop_count], &[151645]);
        record(
            "QWEN_AUDITED_TEXT",
            serde_json::json!({"model_handle_released":true,"retained_strong":weak.strong_count(),"ids":&ids[..count],"decoded":text,"stops":&stops[..stop_count],"generation":false}),
        );
        let chat_pointer = std::mem::replace(&mut handles.chat, ptr::null_mut());
        assert_eq!(toklet_chat_release(chat_pointer, &mut error), OK);
        assert!(weak.upgrade().is_none());
        record(
            "QWEN_AUDITED_DONE",
            serde_json::json!({"passed":true,"all_owners_released":true,"model_loads":1,"model_assembled":true,"codec_paired":true,"generation":false,"app_or_phone_qualified":false,"performance_qualified":false}),
        );
    }
}
