//! Explicit real-model ownership probe; ordinary tests never require assets/GPU.
use super::audited_capture_tests::{compare, observe};
use super::*;
use crate::app_chat::{TOKENIZER_ALLOWANCE, content::tests::capture::Capture};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{error::Error, io::Write, path::PathBuf};
const POLICY: usize = 1024 * 1024 * 1024;
const MODEL_POLICY: usize = 512 * 1024 * 1024;
const CAPTURE_LIMIT: usize = 16 * 1024 * 1024;
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
fn reference<T: DeserializeOwned>(value: &Value) -> T {
    serde_json::from_value(value.clone()).unwrap()
}
fn record(label: &str, value: Value) -> std::result::Result<(), Box<dyn Error>> {
    let bytes = serde_json::to_vec(&value)?;
    if bytes.len() > 1024 * 1024 {
        return Err("native observation exceeds bound".into());
    }
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(label.as_bytes())?;
    stdout.write_all(b" ")?;
    stdout.write_all(&bytes)?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    Ok(())
}
fn run() -> std::result::Result<(), Box<dyn Error>> {
    // Reserve model + independent text + both capture bounds + 16MiB overhead
    // before allocations. All fit this explicit Mac process policy; no phone fit.
    const _: () = assert!(
        MODEL_POLICY + TOKENIZER_ALLOWANCE + 2 * CAPTURE_LIMIT + 16 * 1024 * 1024 <= POLICY
    );
    let path = PathBuf::from(
        std::env::var_os("TOKLET_SMOL_PAIR_MODEL_DIRECTORY").ok_or("missing model directory")?,
    );
    let output = PathBuf::from(
        std::env::var_os("TOKLET_SMOL_PAIR_CAPTURE_DIRECTORY")
            .ok_or("missing capture directory")?,
    );
    if !path.is_absolute() || !output.is_absolute() {
        return Err("diagnostic paths must be absolute".into());
    }
    let path = path.to_str().ok_or("path must be UTF8")?.as_bytes();
    let mut captures = [
        Capture::new(&output.join("native-actual.jsonl"), CAPTURE_LIMIT)?,
        Capture::new(&output.join("native-expected.jsonl"), CAPTURE_LIMIT)?,
    ];
    let options = Options {
        abi_version: ABI_VERSION,
        struct_size: size_of::<Options>() as u32,
        capacity: 8,
        chunk: 3,
        memory_limit_bytes: MODEL_POLICY as u64,
    };
    let mut handles = Handles {
        model: ptr::null_mut(),
        chat: ptr::null_mut(),
    };
    unsafe {
        let legacy = toklet_model_load_mlx(
            path.as_ptr(),
            path.len(),
            &options,
            &mut handles.model,
            ptr::null_mut(),
        );
        compare(
            &mut captures,
            "legacy-load",
            0,
            None,
            [legacy, (!handles.model.is_null()) as u32],
            [OK, 1],
        )?;
        let count = Rc::strong_count(&(*handles.model).model);
        let mut rejected = ptr::dangling_mut();
        let status = toklet_chat_create_smol_audited(
            handles.model,
            TOKENIZER_ALLOWANCE as u64,
            &mut rejected,
            ptr::null_mut(),
        );
        compare(
            &mut captures,
            "legacy-rejection",
            0,
            None,
            vec![
                status,
                rejected.is_null() as u32,
                Rc::strong_count(&(*handles.model).model) as u32,
            ],
            vec![UNSUPPORTED, 1, count as u32],
        )?;
        assert_eq!(toklet_model_release(handles.model, ptr::null_mut()), OK);
        handles.model = ptr::null_mut();
        let status = toklet_model_load_mlx_smol_audited(
            path.as_ptr(),
            path.len(),
            &options,
            &mut handles.model,
            ptr::null_mut(),
        );
        compare(
            &mut captures,
            "audited-load",
            0,
            None,
            [status, (!handles.model.is_null()) as u32],
            [OK, 1],
        )?;
        let weak = Rc::downgrade(&(*handles.model).model);
        for allowance in [1, u64::MAX] {
            let mut rejected = ptr::dangling_mut();
            let status = toklet_chat_create_smol_audited(
                handles.model,
                allowance,
                &mut rejected,
                ptr::null_mut(),
            );
            compare(
                &mut captures,
                "allowance-rejection",
                if allowance == 1 { 0 } else { 1 },
                None,
                vec![
                    status,
                    rejected.is_null() as u32,
                    Rc::strong_count(&(*handles.model).model) as u32,
                ],
                vec![INVALID, 1, 1],
            )?;
        }
        let original = (*handles.model).directory.clone();
        (*handles.model).directory = output.join("missing-assets");
        let mut rejected = ptr::dangling_mut();
        let status = toklet_chat_create_smol_audited(
            handles.model,
            TOKENIZER_ALLOWANCE as u64,
            &mut rejected,
            ptr::null_mut(),
        );
        (*handles.model).directory = original;
        compare(
            &mut captures,
            "asset-rejection",
            0,
            None,
            vec![
                status,
                rejected.is_null() as u32,
                Rc::strong_count(&(*handles.model).model) as u32,
            ],
            vec![LOAD_ERROR, 1, 1],
        )?;
        let status = toklet_chat_create_smol_audited(
            handles.model,
            TOKENIZER_ALLOWANCE as u64,
            &mut handles.chat,
            ptr::null_mut(),
        );
        compare(
            &mut captures,
            "audited-chat-create",
            0,
            None,
            [status, (!handles.chat.is_null()) as u32],
            [OK, 1],
        )?;
        let same = (*handles.chat)
            ._model
            .as_ref()
            .is_some_and(|m| Rc::ptr_eq(m, &(*handles.model).model));
        compare(
            &mut captures,
            "owner-association",
            0,
            None,
            vec![
                same as u32,
                Rc::strong_count(&(*handles.model).model) as u32,
            ],
            vec![1, 2],
        )?;
        let model_address = handles.model as usize;
        let chat_address = handles.chat as usize;
        let statuses = thread::spawn(move || {
            let mut output = ptr::dangling_mut();
            [
                toklet_chat_create_smol_audited(
                    model_address as *const ModelHandle,
                    TOKENIZER_ALLOWANCE as u64,
                    &mut output,
                    ptr::null_mut(),
                ),
                toklet_chat_release(chat_address as *mut ChatHandle, ptr::null_mut()),
                toklet_model_release(model_address as *mut ModelHandle, ptr::null_mut()),
                output.is_null() as u32,
            ]
        })
        .join()
        .unwrap();
        compare(
            &mut captures,
            "wrong-thread",
            0,
            None,
            statuses,
            [WRONG_THREAD, WRONG_THREAD, WRONG_THREAD, 1],
        )?;
        compare(
            &mut captures,
            "owners-after-wrong-thread",
            0,
            None,
            Rc::strong_count(&(*handles.model).model),
            2usize,
        )?;
        assert_eq!(toklet_model_release(handles.model, ptr::null_mut()), OK);
        handles.model = ptr::null_mut();
        compare(
            &mut captures,
            "live-after-external-release",
            0,
            None,
            weak.strong_count(),
            1usize,
        )?;
        let bytes = include_bytes!("../../../tests/fixtures/app-chat-v1/reference.json");
        if format!("{:x}", Sha256::digest(bytes))
            != "3b962a75fb9e84fc0d098ef4cf350520ebb620e0d1d68f22a4462a9375b83577"
        {
            return Err("frozen reference changed".into());
        }
        let fixture: Value = serde_json::from_slice(bytes)?;
        let cases = fixture["cases"].as_array().ok_or("missing cases")?;
        if cases.len() != 2 {
            return Err("expected two cases".into());
        }
        for (i, case) in cases.iter().enumerate() {
            let input = serde_json::to_vec(&case["messages"])?;
            let expected_ids: Vec<u32> = reference(&case["ids"]);
            let mut count = 0;
            let status = toklet_chat_encode(
                handles.chat,
                input.as_ptr(),
                input.len(),
                ptr::null_mut(),
                0,
                &mut count,
                ptr::null_mut(),
            );
            compare(
                &mut captures,
                "encode-query",
                i,
                None,
                vec![status as usize, count],
                vec![OK as usize, expected_ids.len()],
            )?;
            let mut ids = [99u32; 512];
            let status = toklet_chat_encode(
                handles.chat,
                input.as_ptr(),
                input.len(),
                ids.as_mut_ptr(),
                1,
                &mut count,
                ptr::null_mut(),
            );
            compare(
                &mut captures,
                "encode-short-buffer",
                i,
                None,
                observe(status, count, &ids),
                observe(INVALID, expected_ids.len(), &[99u32; 512]),
            )?;
            compare(
                &mut captures,
                "encode-short",
                i,
                None,
                vec![
                    status as usize,
                    count,
                    ids.iter().all(|&n| n == 99) as usize,
                ],
                vec![INVALID as usize, expected_ids.len(), 1],
            )?;
            let status = toklet_chat_encode(
                handles.chat,
                input.as_ptr(),
                input.len(),
                ids.as_mut_ptr(),
                512,
                &mut count,
                ptr::null_mut(),
            );
            let mut expected_storage = [99u32; 512];
            expected_storage[..expected_ids.len()].copy_from_slice(&expected_ids);
            compare(
                &mut captures,
                "encode-buffer",
                i,
                None,
                observe(status, count, &ids),
                observe(OK, expected_ids.len(), &expected_storage),
            )?;
            compare(
                &mut captures,
                "encode-status",
                i,
                None,
                vec![status as usize, count],
                vec![OK as usize, expected_ids.len()],
            )?;
            compare(
                &mut captures,
                "encode",
                i,
                None,
                ids[..count].to_vec(),
                expected_ids.clone(),
            )?;
            let expected_text = case["decoded"].as_str().ok_or("missing decoded text")?;
            let mut count = 0;
            let status = toklet_chat_decode(
                handles.chat,
                ids.as_ptr(),
                expected_ids.len(),
                ptr::null_mut(),
                0,
                &mut count,
                ptr::null_mut(),
            );
            compare(
                &mut captures,
                "decode-query",
                i,
                Some(&expected_ids),
                vec![status as usize, count],
                vec![OK as usize, expected_text.len()],
            )?;
            let mut text = [99u8; 65536];
            let status = toklet_chat_decode(
                handles.chat,
                ids.as_ptr(),
                expected_ids.len(),
                text.as_mut_ptr(),
                1,
                &mut count,
                ptr::null_mut(),
            );
            compare(
                &mut captures,
                "decode-short-buffer",
                i,
                Some(&expected_ids),
                observe(status, count, &text),
                observe(INVALID, expected_text.len(), &[99u8; 65536]),
            )?;
            compare(
                &mut captures,
                "decode-short",
                i,
                Some(&expected_ids),
                vec![
                    status as usize,
                    count,
                    text.iter().all(|&n| n == 99) as usize,
                ],
                vec![INVALID as usize, expected_text.len(), 1],
            )?;
            let status = toklet_chat_decode(
                handles.chat,
                ids.as_ptr(),
                expected_ids.len(),
                text.as_mut_ptr(),
                text.len(),
                &mut count,
                ptr::null_mut(),
            );
            let mut expected_storage = [99u8; 65536];
            expected_storage[..expected_text.len()].copy_from_slice(expected_text.as_bytes());
            compare(
                &mut captures,
                "decode-buffer",
                i,
                Some(&expected_ids),
                observe(status, count, &text),
                observe(OK, expected_text.len(), &expected_storage),
            )?;
            compare(
                &mut captures,
                "decode-status",
                i,
                Some(&expected_ids),
                vec![status as usize, count],
                vec![OK as usize, expected_text.len()],
            )?;
            compare(
                &mut captures,
                "decode",
                i,
                Some(&expected_ids),
                std::str::from_utf8(&text[..count])?.to_owned(),
                expected_text.to_owned(),
            )?;
        }
        let mut stops = [99u32; 8];
        let mut count = 0;
        let status = toklet_chat_stop_ids(
            handles.chat,
            stops.as_mut_ptr(),
            stops.len(),
            &mut count,
            ptr::null_mut(),
        );
        compare(
            &mut captures,
            "stop-buffer",
            0,
            None,
            observe(status, count, &stops),
            observe(OK, 2, &[0, 2, 99, 99, 99, 99, 99, 99]),
        )?;
        compare(
            &mut captures,
            "stop-status",
            0,
            None,
            vec![status as usize, count],
            vec![OK as usize, 2],
        )?;
        compare(
            &mut captures,
            "stops",
            0,
            None,
            stops[..count].to_vec(),
            vec![0, 2],
        )?;
        assert_eq!(toklet_chat_release(handles.chat, ptr::null_mut()), OK);
        handles.chat = ptr::null_mut();
        compare(
            &mut captures,
            "destroyed-after-chat-release",
            0,
            None,
            weak.upgrade().is_none(),
            true,
        )?;
        for capture in &mut captures {
            capture.flush()?;
        }
        record(
            "NATIVE_SMOL_PAIR",
            json!({"format":"toklet-smol-native-pair-v1","cases":2,"source_ranges":694,"legacy_rejected":true,"same_rc_allocation":true,"retained_after_external_release":true,"destroyed_after_chat_release":true,"wrong_thread_preserves_owners":true,"model_policy_bytes":MODEL_POLICY,"text_allowance_bytes":TOKENIZER_ALLOWANCE,"capture_bound_each_bytes":CAPTURE_LIMIT,"process_policy_bytes":POLICY,"capture_bytes":[captures[0].bytes(),captures[1].bytes()],"chat_handle_bytes":size_of::<ChatHandle>(),"retained_option_bytes":size_of::<Option<Rc<NativeModel>>>(),"new_numerical_forward":false,"app_capability_granted":false,"phone_fit_qualified":false,"performance_qualified":false}),
        )?;
    }
    Ok(())
}
#[test]
#[ignore = "requires explicit local Smol model, GPU and empty capture directory"]
fn audited_native_chat_retains_exact_model_until_final_release() {
    run().unwrap();
}
