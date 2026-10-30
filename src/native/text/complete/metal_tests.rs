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
#[ignore = "Explicit cached Smol audited assembly, complete codec and identity; no generation"]
fn pinned_smol_complete_pair_identity_and_release() {
    const MODEL: u64 = 512 * 1024 * 1024;
    const TEXT: u64 = 64 * 1024 * 1024;
    const _: () = assert!(MODEL + TEXT + 256 * 1024 * 1024 < 6 * 1024 * 1024 * 1024);
    let path = b"models/MLX-SmolLM2-135M-Instruct-8bit-0f0d9b8";
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
        let status = toklet_model_load_mlx_smol_audited(
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
        let missing =
            std::env::temp_dir().join(format!("toklet-no-complete-smol-{}", std::process::id()));
        assert!(!missing.exists());
        let directory = std::mem::replace(&mut (*owners.model).directory, missing);
        let mut rejected = ptr::dangling_mut();
        let status =
            toklet_chat_create_complete_audited(owners.model, TEXT, &mut rejected, &mut error);
        record(
            "missing-ancillary",
            serde_json::json!({"status":status,"null":rejected.is_null(),"strong_count":weak.strong_count()}),
        );
        (*owners.model).directory = directory;
        assert_eq!(status, LOAD_ERROR);
        assert!(rejected.is_null());
        assert_eq!(weak.strong_count(), 1);
        // Existing three-asset factory stays accepted but cannot fabricate full
        // manifest completion. Drop it before constructing the complete codec.
        assert_eq!(
            toklet_chat_create_smol_audited(owners.model, TEXT, &mut owners.chat, &mut error),
            OK
        );
        let status = toklet_model_chat_identity_create(
            owners.model,
            owners.chat,
            &mut owners.identity,
            &mut error,
        );
        record(
            "legacy-incomplete",
            serde_json::json!({"status":status,"null":owners.identity.is_null(),"strong_count":weak.strong_count()}),
        );
        assert_eq!(status, UNSUPPORTED);
        assert!(owners.identity.is_null());
        assert_eq!(weak.strong_count(), 2);
        let legacy = std::mem::replace(&mut owners.chat, ptr::null_mut());
        assert_eq!(toklet_chat_release(legacy, &mut error), OK);
        assert_eq!(weak.strong_count(), 1);
        let status =
            toklet_chat_create_complete_audited(owners.model, TEXT, &mut owners.chat, &mut error);
        record(
            "complete",
            serde_json::json!({"status":status,"nonnull":!owners.chat.is_null(),"strong_count":weak.strong_count()}),
        );
        assert_eq!(status, OK);
        assert_eq!(weak.strong_count(), 2);
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
        assert_eq!(
            report["preference_namespace"]["load"],
            "smol_witness_audited_v1"
        );
        assert_eq!(
            report["assets"][1]["roles"],
            serde_json::json!([
                "source_bytes_delivered_to_assembled_model",
                "reference_container_digest_anchor"
            ])
        );
        assert_eq!(report["source_container_physical_ranges"], 694);
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
        assert_eq!(&stops[..count], &[0, 2]);
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
