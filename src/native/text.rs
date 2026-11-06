//! Additive, immutable text handle. Numerical handles remain raw-token APIs.
use super::*;
mod complete;
mod identity;
use crate::app_chat::{Codec, MAX_RESPONSE_TOKENS, MAX_TEXT, MAX_TOKENS, Profile};
pub use complete::toklet_chat_create_complete_audited;
pub use identity::toklet_model_chat_identity_create;

pub struct ChatHandle {
    owner: ThreadId,
    codec: Codec,
    // Exactly the supplied loaded allocation, retained only by audited factory.
    // Covered by separate text allowance; count the shared physical model once.
    _model: Option<Rc<NativeModel>>,
}

/// Load the pinned chat assets from the loaded model's own directory.
/// Reserve 64MiB (SmolLM2/Mistral), 128MiB (Qwen), or 256MiB (Qwen3) beyond numerical memory.
/// # Safety
/// Follow the parent module's foreign pointer/thread contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn toklet_chat_create(
    model: *const ModelHandle,
    allowance: u64,
    out: *mut *mut ChatHandle,
    error: *mut NativeError,
) -> u32 {
    unsafe {
        boundary(error, || {
            pointer(out)?;
            out.write(ptr::null_mut());
            pointer(model)?;
            let model = &*model;
            owner(model.owner)?;
            let config = match model.model.as_ref() {
                NativeModel::Cpu(model) => model.config(),
                #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
                NativeModel::OfficialGemma(_) => {
                    return Err(fail(
                        UNSUPPORTED,
                        "Gemma BF16 requires its audited text route",
                    ));
                }
                #[cfg(all(
                    feature = "metal",
                    any(target_os = "macos", target_os = "ios"),
                    target_arch = "aarch64"
                ))]
                NativeModel::Metal(model) => model.config(),
            };
            let profile =
                Profile::load(&model.directory).map_err(|e| fail(LOAD_ERROR, e.to_string()))?;
            if allowance < profile.allowance() as u64
                || allowance > weights::INITIAL_MEMORY_LIMIT_BYTES as u64
            {
                return Err(fail(
                    INVALID,
                    "reserve text memory separately: SmolLM2/Mistral 64MiB, Qwen 128MiB, Qwen3 256MiB",
                ));
            }
            if !profile.supports(config) {
                return Err(fail(
                    UNSUPPORTED,
                    "loaded model geometry does not match its pinned text profile",
                ));
            }
            let codec = Codec::load_profile(&model.directory, profile)
                .map_err(|e| fail(LOAD_ERROR, e.to_string()))?;
            out.write(Box::into_raw(Box::new(ChatHandle {
                owner: thread::current().id(),
                codec,
                _model: None,
            })));
            Ok(())
        })
    }
}

/// Bind validated Smol assets to the exact live audited model allocation.
/// Reserve a separate 64 MiB text allowance BEFORE loading the model. Keep the
/// model's numerical memory admitted until both external model and all retaining
/// sessions/audited chats are released. Individual allowance maxima do not prove
/// combined process/device fit. This does not grant native app capabilities.
/// # Safety
/// Follow the parent module's foreign pointer/thread/disjoint-output contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn toklet_chat_create_smol_audited(
    model: *const ModelHandle,
    allowance: u64,
    out: *mut *mut ChatHandle,
    error: *mut NativeError,
) -> u32 {
    unsafe {
        boundary(error, || {
            pointer(out)?;
            out.write(ptr::null_mut());
            pointer(model)?;
            let model = &*model;
            owner(model.owner)?;
            if allowance < crate::app_chat::TOKENIZER_ALLOWANCE as u64
                || allowance > weights::INITIAL_MEMORY_LIMIT_BYTES as u64
            {
                return Err(fail(
                    INVALID,
                    "reserve audited Smol text memory separately: >=64MiB and <=6GiB",
                ));
            }
            // CPU and legacy Metal imports have no eligible source evidence.
            // Reject them BEFORE inspecting their directory or any text asset.
            let source = match model.model.as_ref() {
                NativeModel::Cpu(_) => None,
                #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
                NativeModel::OfficialGemma(_) => None,
                #[cfg(all(
                    feature = "metal",
                    any(target_os = "macos", target_os = "ios"),
                    target_arch = "aarch64"
                ))]
                NativeModel::Metal(model) => model.mlx_source_content_identity().copied(),
            }
            .ok_or_else(|| {
                fail(
                    UNSUPPORTED,
                    "audited Smol chat requires an audited loaded Smol model",
                )
            })?;
            let codec = Codec::load_smol_audited(&model.directory, &source, allowance as usize)
                .map_err(|e| fail(LOAD_ERROR, e.to_string()))?;
            // Clone only after every fallible validation; failed factories add
            // no owner. The codec and exact model association publish atomically.
            out.write(Box::into_raw(Box::new(ChatHandle {
                owner: thread::current().id(),
                codec,
                _model: Some(Rc::clone(&model.model)),
            })));
            Ok(())
        })
    }
}

/// Bind verified Qwen3 codec/policy/ancillary assets to the exact supplied
/// audited model allocation. Requires a separately reserved 256 MiB text allowance
/// (up to 6 GiB), in addition to model/session memory; no combined-fit claim.
/// Legacy/CPU/wrong-profile models reject before file access. The chat retains
/// the same Rc only after successful codec construction. Token APIs still return
/// ordinary raw IDs; callers remain responsible for using them with the right model.
/// # Safety
/// Follow the parent module's foreign pointer, output and thread contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn toklet_chat_create_qwen3_audited(
    model: *const ModelHandle,
    allowance: u64,
    out: *mut *mut ChatHandle,
    error: *mut NativeError,
) -> u32 {
    unsafe {
        boundary(error, || {
            pointer(out)?;
            out.write(ptr::null_mut());
            pointer(model)?;
            let model = &*model;
            owner(model.owner)?;
            if allowance < Profile::Qwen3.allowance() as u64
                || allowance > weights::INITIAL_MEMORY_LIMIT_BYTES as u64
            {
                return Err(fail(
                    INVALID,
                    "reserve audited Qwen3 text memory separately: >=256MiB and <=6GiB",
                ));
            }
            let source = match model.model.as_ref() {
                NativeModel::Cpu(_) => None,
                #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
                NativeModel::OfficialGemma(_) => None,
                #[cfg(all(
                    feature = "metal",
                    any(target_os = "macos", target_os = "ios"),
                    target_arch = "aarch64"
                ))]
                NativeModel::Metal(value) => {
                    if Profile::Qwen3.supports(value.config()) {
                        value.mlx_qwen3_source_content_identity().copied()
                    } else {
                        None
                    }
                }
            }
            .ok_or_else(|| {
                fail(
                    UNSUPPORTED,
                    "audited Qwen3 chat requires an audited loaded Qwen3 model",
                )
            })?;
            let codec = Codec::load_qwen3_audited(&model.directory, &source, allowance as usize)
                .map_err(|e| fail(LOAD_ERROR, e.to_string()))?;
            out.write(Box::into_raw(Box::new(ChatHandle {
                owner: thread::current().id(),
                codec,
                _model: Some(Rc::clone(&model.model)),
            })));
            Ok(())
        })
    }
}

/// Encode a bounded JSON user/assistant conversation using the pinned template.
/// NULL+0 queries required count; undersized buffers copy no elements.
/// # Safety
/// Follow the parent module's foreign pointer/thread contract. count is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn toklet_chat_encode(
    chat: *const ChatHandle,
    json: *const u8,
    len: usize,
    output: *mut u32,
    capacity: usize,
    count: *mut usize,
    error: *mut NativeError,
) -> u32 {
    unsafe {
        boundary(error, || {
            pointer(chat)?;
            let chat = &*chat;
            owner(chat.owner)?;
            if capacity > MAX_TOKENS {
                return Err(fail(INVALID, "text token capacity exceeds engine ceiling"));
            }
            let ids = chat
                .codec
                .encode(slice(json, len, MAX_TEXT)?)
                .map_err(|e| fail(INVALID, e.to_string()))?;
            copy_output(&ids, output, capacity, count)
        })
    }
}

/// Decode the complete visible prefix, skipping special IDs. Replace displayed
/// text with this UTF-8 result; do not concatenate single-token decodes. A later
/// prefix repairs any incomplete multi-token UTF-8 sequence. No trailing NUL.
/// # Safety
/// Follow the parent module's foreign pointer/thread contract. count is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn toklet_chat_decode(
    chat: *const ChatHandle,
    ids: *const u32,
    len: usize,
    output: *mut u8,
    capacity: usize,
    count: *mut usize,
    error: *mut NativeError,
) -> u32 {
    unsafe {
        boundary(error, || {
            pointer(chat)?;
            let chat = &*chat;
            owner(chat.owner)?;
            if capacity > MAX_TEXT {
                return Err(fail(INVALID, "decoded text capacity exceeds 64 KiB"));
            }
            let text = chat
                .codec
                .decode(slice(ids, len, MAX_RESPONSE_TOKENS)?)
                .map_err(|e| fail(INVALID, e.to_string()))?;
            copy_output(text.as_bytes(), output, capacity, count)
        })
    }
}

/// Query the pinned profile's complete stop-ID set. No implicit EOS is added.
/// NULL+0 and undersized-buffer behavior matches the other text queries.
/// # Safety
/// Follow the parent module's pointer/thread/disjoint-output contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn toklet_chat_stop_ids(
    chat: *const ChatHandle,
    output: *mut u32,
    capacity: usize,
    count: *mut usize,
    error: *mut NativeError,
) -> u32 {
    unsafe {
        boundary(error, || {
            pointer(chat)?;
            let chat = &*chat;
            owner(chat.owner)?;
            if capacity > 8 {
                return Err(fail(INVALID, "stop-ID capacity exceeds eight"));
            }
            copy_output(chat.codec.stop_ids(), output, capacity, count)
        })
    }
}

/// Release an immutable text handle. NULL is a no-op. An audited chat releases
/// its retained model owner; a legacy chat has no model lifetime dependency.
/// # Safety
/// Follow the parent module's foreign pointer/thread contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn toklet_chat_release(
    chat: *mut ChatHandle,
    error: *mut NativeError,
) -> u32 {
    unsafe {
        boundary(error, || {
            if !chat.is_null() {
                pointer(chat)?;
                owner((*chat).owner)?;
                drop(Box::from_raw(chat));
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rendered_encode_ceiling_preserves_query_copy_and_rejection_contracts() {
        let chat = Box::into_raw(Box::new(ChatHandle {
            owner: thread::current().id(),
            codec: crate::app_chat::encode_boundary_fixture(),
            _model: None,
        }));
        unsafe {
            for total in [513, 1024, 32768] {
                let (json, expected) = crate::app_chat::encode_boundary_case(total);
                let mut count = 0;
                assert_eq!(
                    toklet_chat_encode(
                        chat,
                        json.as_ptr(),
                        json.len(),
                        ptr::null_mut(),
                        0,
                        &mut count,
                        ptr::null_mut()
                    ),
                    OK
                );
                assert_eq!(count, total);
                let mut output = vec![u32::MAX; total];
                assert_eq!(
                    toklet_chat_encode(
                        chat,
                        json.as_ptr(),
                        json.len(),
                        output.as_mut_ptr(),
                        total - 1,
                        &mut count,
                        ptr::null_mut()
                    ),
                    INVALID
                );
                assert_eq!(count, total);
                assert!(output.iter().all(|&id| id == u32::MAX));
                assert_eq!(
                    toklet_chat_encode(
                        chat,
                        json.as_ptr(),
                        json.len(),
                        output.as_mut_ptr(),
                        total,
                        &mut count,
                        ptr::null_mut()
                    ),
                    OK
                );
                assert_eq!(output, expected);
            }
            let (json, _) = crate::app_chat::encode_boundary_case(32769);
            let mut output = vec![u32::MAX; 32769];
            let mut count = 777;
            assert_eq!(
                toklet_chat_encode(
                    chat,
                    json.as_ptr(),
                    json.len(),
                    output.as_mut_ptr(),
                    32768,
                    &mut count,
                    ptr::null_mut()
                ),
                INVALID
            );
            assert_eq!(count, 777);
            assert!(output.iter().all(|&id| id == u32::MAX));
            let (json, _) = crate::app_chat::encode_boundary_case(513);
            assert_eq!(
                toklet_chat_encode(
                    chat,
                    json.as_ptr(),
                    json.len(),
                    output.as_mut_ptr(),
                    32769,
                    &mut count,
                    ptr::null_mut()
                ),
                INVALID
            );
            assert_eq!(count, 777);
            assert!(output.iter().all(|&id| id == u32::MAX));
            assert_eq!(toklet_chat_release(chat, ptr::null_mut()), OK);
        }
    }
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    use crate::app_chat::TOKENIZER_ALLOWANCE;
    #[test]
    fn visible_decode_boundary_has_exact_query_copy_and_257_rejection() {
        let chat = Box::into_raw(Box::new(ChatHandle {
            owner: thread::current().id(),
            codec: crate::app_chat::decode_fixture(),
            _model: None,
        }));
        let ids = [32; 257];
        let mut count = 0;
        let mut output = [0xa5; 257];
        unsafe {
            assert_eq!(
                toklet_chat_decode(
                    chat,
                    ids.as_ptr(),
                    256,
                    ptr::null_mut(),
                    0,
                    &mut count,
                    ptr::null_mut()
                ),
                OK
            );
            assert_eq!(count, 256);
            assert_eq!(
                toklet_chat_decode(
                    chat,
                    ids.as_ptr(),
                    256,
                    output.as_mut_ptr(),
                    255,
                    &mut count,
                    ptr::null_mut()
                ),
                INVALID
            );
            assert_eq!(count, 256);
            assert_eq!(output, [0xa5; 257]);
            assert_eq!(
                toklet_chat_decode(
                    chat,
                    ids.as_ptr(),
                    256,
                    output.as_mut_ptr(),
                    256,
                    &mut count,
                    ptr::null_mut()
                ),
                OK
            );
            assert_eq!(&output[..256], &[b'A'; 256]);
            assert_eq!(output[256], 0xa5);
            output.fill(0xa5);
            assert_eq!(
                toklet_chat_decode(
                    chat,
                    ids.as_ptr(),
                    257,
                    output.as_mut_ptr(),
                    output.len(),
                    &mut count,
                    ptr::null_mut()
                ),
                INVALID
            );
            assert_eq!(output, [0xa5; 257]);
            assert_eq!(toklet_chat_release(chat, ptr::null_mut()), OK);
        }
    }
    #[test]
    fn model_directory_freezes_relative_origin() {
        let path = origin_path(Path::new("models/local")).unwrap();
        assert!(path.is_absolute());
        assert_eq!(path, std::env::current_dir().unwrap().join("models/local"));
    }

    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "explicit real local model and Apple GPU C bridge check"]
    fn public_chat_create_and_release_independence() {
        let path = b"models/MLX-SmolLM2-135M-Instruct-8bit-0f0d9b8";
        let options = Options {
            abi_version: ABI_VERSION,
            struct_size: size_of::<Options>() as u32,
            capacity: 128,
            chunk: 32,
            memory_limit_bytes: 512 * 1024 * 1024,
        };
        let mut model = ptr::null_mut();
        let mut chat = ptr::dangling_mut();
        unsafe {
            assert_eq!(
                toklet_model_load_mlx(
                    path.as_ptr(),
                    path.len(),
                    &options,
                    &mut model,
                    ptr::null_mut()
                ),
                OK
            );
            assert!((*model).directory.is_absolute());
            assert_eq!(
                toklet_chat_create(model, 1, &mut chat, ptr::null_mut()),
                INVALID
            );
            assert!(chat.is_null());
            assert_eq!(
                toklet_chat_create(
                    model,
                    TOKENIZER_ALLOWANCE as u64,
                    &mut chat,
                    ptr::null_mut()
                ),
                OK
            );
            let saved = (*model).directory.clone();
            (*model).directory = Path::new("tests/fixtures/app-chat-v1").to_path_buf();
            let mut rejected = ptr::dangling_mut();
            assert_eq!(
                toklet_chat_create(
                    model,
                    TOKENIZER_ALLOWANCE as u64,
                    &mut rejected,
                    ptr::null_mut()
                ),
                LOAD_ERROR
            );
            assert!(rejected.is_null());
            (*model).directory = saved;
            assert_eq!(toklet_model_release(model, ptr::null_mut()), OK);
            // Immutable tokenizer owns no model borrow; it still encodes after release.
            let input = br#"[{"role":"user","content":"Hello!"}]"#;
            let mut count = 0;
            let mut ids = [99u32; 512];
            assert_eq!(
                toklet_chat_encode(
                    chat,
                    input.as_ptr(),
                    input.len(),
                    ptr::null_mut(),
                    0,
                    &mut count,
                    ptr::null_mut()
                ),
                OK
            );
            assert_eq!(count, 32);
            assert_eq!(
                toklet_chat_encode(
                    chat,
                    input.as_ptr(),
                    input.len(),
                    ids.as_mut_ptr(),
                    1,
                    &mut count,
                    ptr::null_mut()
                ),
                INVALID
            );
            assert_eq!(ids, [99; 512]);
            assert_eq!(
                toklet_chat_encode(
                    chat,
                    input.as_ptr(),
                    input.len(),
                    ids.as_mut_ptr(),
                    512,
                    &mut count,
                    ptr::null_mut()
                ),
                OK
            );
            let mut bytes = 0;
            assert_eq!(
                toklet_chat_decode(
                    chat,
                    ids.as_ptr(),
                    count,
                    ptr::null_mut(),
                    0,
                    &mut bytes,
                    ptr::null_mut()
                ),
                OK
            );
            assert!(bytes > 0);
            let mut text = [99u8; 65536];
            assert_eq!(
                toklet_chat_decode(
                    chat,
                    ids.as_ptr(),
                    count,
                    text.as_mut_ptr(),
                    1,
                    &mut bytes,
                    ptr::null_mut()
                ),
                INVALID
            );
            assert_eq!(text, [99; 65536]);
            assert_eq!(
                toklet_chat_decode(
                    chat,
                    ids.as_ptr(),
                    count,
                    text.as_mut_ptr(),
                    65536,
                    &mut bytes,
                    ptr::null_mut()
                ),
                OK
            );
            assert!(
                std::str::from_utf8(&text[..bytes])
                    .unwrap()
                    .contains("Hello!")
            );
            let address = chat as usize;
            assert_eq!(
                thread::spawn(move || {
                    let chat = address as *mut ChatHandle;
                    let mut count = 0;
                    [
                        toklet_chat_encode(
                            chat,
                            ptr::null(),
                            0,
                            ptr::null_mut(),
                            0,
                            &mut count,
                            ptr::null_mut(),
                        ),
                        toklet_chat_decode(
                            chat,
                            ptr::null(),
                            0,
                            ptr::null_mut(),
                            0,
                            &mut count,
                            ptr::null_mut(),
                        ),
                        toklet_chat_release(chat, ptr::null_mut()),
                    ]
                })
                .join()
                .unwrap(),
                [WRONG_THREAD; 3]
            );
            assert_eq!(toklet_chat_release(chat, ptr::null_mut()), OK);
        }
    }
    #[test]
    fn caller_buffer_query_and_failure_are_transactional() {
        let values = [11u32, 22, 33];
        let mut output = [99u32; 3];
        let mut count = 0;
        unsafe {
            assert!(copy_output(&values, ptr::null_mut(), 0, &mut count).is_ok());
            assert_eq!(count, 3);
            assert!(copy_output(&values, output.as_mut_ptr(), 2, &mut count).is_err());
            assert_eq!(output, [99; 3]);
            assert!(copy_output(&values, output.as_mut_ptr(), 3, &mut count).is_ok());
            assert_eq!(output, values);
            assert!(copy_output(&values, output.as_mut_ptr(), 3, ptr::null_mut()).is_err());
            assert!(copy_output::<u32>(&[], ptr::null_mut(), 1, &mut count).is_err());
            assert_eq!(toklet_chat_release(ptr::null_mut(), ptr::null_mut()), OK);
            assert_eq!(
                toklet_chat_create(ptr::null(), 0, &mut ptr::null_mut(), ptr::null_mut()),
                INVALID
            );
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "explicit real local Qwen GPU text check"]
    fn public_qwen_chat_generation_and_stop_query() {
        public_chat_case(
            b"models/MLX-Qwen2.5-0.5B-Instruct-8bit-6474d4c",
            Profile::Qwen,
            1024 * 1024 * 1024,
        );
    }

    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "explicit real local Qwen3 GPU text check"]
    fn public_qwen3_chat_generation_and_stop_query() {
        public_chat_case(
            b"models/MLX-Qwen3-0.6B-8bit-11de968",
            Profile::Qwen3,
            1536 * 1024 * 1024,
        );
    }

    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "explicit real local Mistral GPU text check"]
    fn public_mistral_chat_generation_and_stop_query() {
        public_chat_case(
            b"models/MLX-Mistral-7B-Instruct-v0.3-4bit-a4b8f87",
            Profile::Mistral,
            5 * 1024 * 1024 * 1024,
        );
    }

    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    fn public_chat_case(path: &[u8], profile: Profile, memory_limit_bytes: u64) {
        unsafe extern "C" fn collect(ctx: *mut c_void, event: TokenEvent) -> u32 {
            // SAFETY: this synchronous test lends its unique Vec for the callback.
            unsafe { (&mut *ctx.cast::<Vec<u32>>()).push(event.token_id) };
            0
        }
        let opts = Options {
            abi_version: ABI_VERSION,
            struct_size: size_of::<Options>() as u32,
            capacity: 128,
            chunk: 32,
            memory_limit_bytes,
        };
        unsafe {
            let mut model = ptr::null_mut();
            let mut chat = ptr::null_mut();
            assert_eq!(
                toklet_model_load_mlx(
                    path.as_ptr(),
                    path.len(),
                    &opts,
                    &mut model,
                    ptr::null_mut()
                ),
                OK
            );
            assert_eq!(
                toklet_chat_create(
                    model,
                    (profile.allowance() - 1) as u64,
                    &mut chat,
                    ptr::null_mut()
                ),
                INVALID
            );
            assert!(chat.is_null());
            assert_eq!(
                toklet_chat_create(
                    model,
                    profile.allowance() as u64,
                    &mut chat,
                    ptr::null_mut()
                ),
                OK
            );
            let mut stops = vec![0; profile.stop_ids().len()];
            let mut count = 0;
            assert_eq!(
                toklet_chat_stop_ids(
                    chat,
                    stops.as_mut_ptr(),
                    stops.len(),
                    &mut count,
                    ptr::null_mut()
                ),
                OK
            );
            assert_eq!(stops, profile.stop_ids());
            assert_eq!(count, stops.len());
            let mut session = ptr::null_mut();
            let mut cancel = ptr::null_mut();
            assert_eq!(
                toklet_session_create(model, &opts, &mut session, ptr::null_mut()),
                OK
            );
            assert_eq!(toklet_cancel_create(&mut cancel, ptr::null_mut()), OK);
            assert_eq!(toklet_model_release(model, ptr::null_mut()), OK);
            let input = br#"[{"role":"user","content":"Hello!"}]"#;
            let mut prompt = [0; 512];
            assert_eq!(
                toklet_chat_encode(
                    chat,
                    input.as_ptr(),
                    input.len(),
                    prompt.as_mut_ptr(),
                    prompt.len(),
                    &mut count,
                    ptr::null_mut()
                ),
                OK
            );
            let mut ids = Vec::<u32>::new();
            let mut report = Report::default();
            assert_eq!(
                toklet_session_generate(
                    session,
                    3,
                    prompt.as_ptr(),
                    count,
                    8,
                    stops.as_ptr(),
                    stops.len(),
                    cancel,
                    Some(collect),
                    (&mut ids as *mut Vec<u32>).cast(),
                    &mut report,
                    ptr::null_mut()
                ),
                OK
            );
            assert!(!ids.is_empty());
            assert!(ids.iter().all(|id| !stops.contains(id)));
            let mut text = [0u8; MAX_TEXT];
            let mut bytes = 0;
            assert_eq!(
                toklet_chat_decode(
                    chat,
                    ids.as_ptr(),
                    ids.len(),
                    text.as_mut_ptr(),
                    text.len(),
                    &mut bytes,
                    ptr::null_mut()
                ),
                OK
            );
            let decoded = std::str::from_utf8(&text[..bytes]).unwrap();
            assert!(!decoded.is_empty());
            println!(
                "NATIVE_CHAT_JSON={}",
                serde_json::json!({"profile": format!("{profile:?}"), "ids": ids, "text": decoded, "reason": report.reason})
            );
            // A stop on the known first token is hidden and leaves no visible event.
            let forced = [ids[0]];
            ids.clear();
            assert_eq!(
                toklet_session_generate(
                    session,
                    3,
                    prompt.as_ptr(),
                    count,
                    8,
                    forced.as_ptr(),
                    1,
                    cancel,
                    Some(collect),
                    (&mut ids as *mut Vec<u32>).cast(),
                    &mut report,
                    ptr::null_mut()
                ),
                OK
            );
            assert_eq!(report.reason, 2);
            assert_eq!(report.stop_token, forced[0]);
            assert!(ids.is_empty());
            assert_eq!(toklet_session_release(session, ptr::null_mut()), OK);
            assert_eq!(toklet_cancel_release(cancel, ptr::null_mut()), OK);
            assert_eq!(toklet_chat_release(chat, ptr::null_mut()), OK);
        }
    }

    #[test]
    #[ignore = "explicit local tokenizer bridge check; no model download"]
    fn real_text_handle_bounds_thread_and_utf8() {
        let chat = Box::into_raw(Box::new(ChatHandle {
            owner: thread::current().id(),
            codec: Codec::load(Path::new("models/MLX-SmolLM2-135M-Instruct-8bit-0f0d9b8")).unwrap(),
            _model: None,
        }));
        let input = br#"[{"role":"user","content":"Hello!"}]"#;
        let mut count = 0;
        let mut output = [99u32; 512];
        unsafe {
            assert_eq!(
                toklet_chat_encode(
                    chat,
                    input.as_ptr(),
                    input.len(),
                    ptr::null_mut(),
                    0,
                    &mut count,
                    ptr::null_mut()
                ),
                OK
            );
            assert_eq!(count, 32);
            assert_eq!(
                toklet_chat_encode(
                    chat,
                    input.as_ptr(),
                    input.len(),
                    output.as_mut_ptr(),
                    2,
                    &mut count,
                    ptr::null_mut()
                ),
                INVALID
            );
            assert_eq!(output, [99; 512]);
            assert_eq!(
                toklet_chat_encode(
                    chat,
                    input.as_ptr(),
                    input.len(),
                    output.as_mut_ptr(),
                    512,
                    &mut count,
                    ptr::null_mut()
                ),
                OK
            );
            let mut text = [0u8; 65536];
            let mut bytes = 0;
            assert_eq!(
                toklet_chat_decode(
                    chat,
                    output.as_ptr(),
                    count,
                    text.as_mut_ptr(),
                    text.len(),
                    &mut bytes,
                    ptr::null_mut()
                ),
                OK
            );
            assert!(
                std::str::from_utf8(&text[..bytes])
                    .unwrap()
                    .contains("Hello!")
            );
            assert_eq!(
                toklet_chat_encode(
                    chat,
                    [255u8].as_ptr(),
                    1,
                    output.as_mut_ptr(),
                    512,
                    &mut count,
                    ptr::null_mut()
                ),
                INVALID
            );
            assert_eq!(
                toklet_chat_decode(
                    chat,
                    [49152u32].as_ptr(),
                    1,
                    text.as_mut_ptr(),
                    text.len(),
                    &mut bytes,
                    ptr::null_mut()
                ),
                INVALID
            );
        }
        unsafe {
            assert_eq!(
                toklet_chat_stop_ids(chat, ptr::null_mut(), 1, &mut count, ptr::null_mut()),
                INVALID
            );
            assert_eq!(
                toklet_chat_stop_ids(chat, ptr::null_mut(), 0, ptr::null_mut(), ptr::null_mut()),
                INVALID
            );
            let mut stops = [99u32; 2];
            assert_eq!(
                toklet_chat_stop_ids(chat, ptr::null_mut(), 0, &mut count, ptr::null_mut()),
                OK
            );
            assert_eq!(count, 2);
            assert_eq!(
                toklet_chat_stop_ids(chat, stops.as_mut_ptr(), 1, &mut count, ptr::null_mut()),
                INVALID
            );
            assert_eq!(stops, [99, 99]);
            assert_eq!(
                toklet_chat_stop_ids(chat, stops.as_mut_ptr(), 2, &mut count, ptr::null_mut()),
                OK
            );
            assert_eq!(stops, [0, 2]);
            assert_eq!(
                toklet_chat_stop_ids(chat, stops.as_mut_ptr(), 9, &mut count, ptr::null_mut()),
                INVALID
            );
        }
        let address = chat as usize;
        assert_eq!(
            thread::spawn(move || unsafe {
                let mut count = 0;
                assert_eq!(
                    toklet_chat_stop_ids(
                        address as *mut ChatHandle,
                        ptr::null_mut(),
                        0,
                        &mut count,
                        ptr::null_mut()
                    ),
                    WRONG_THREAD
                );

                toklet_chat_release(address as *mut ChatHandle, ptr::null_mut())
            })
            .join()
            .unwrap(),
            WRONG_THREAD
        );
        unsafe {
            assert_eq!(toklet_chat_release(chat, ptr::null_mut()), OK);
        }
    }
}

#[cfg(all(test, feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod audited_tests;

#[cfg(test)]
mod audited_capture_tests;

#[cfg(all(test, feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod qwen3_audited_tests;
