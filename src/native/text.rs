//! Additive, immutable text handle. Numerical handles remain raw-token APIs.
use super::*;
use crate::app_chat::{Codec, MAX_TEXT, MAX_TOKENS, TOKENIZER_ALLOWANCE};

pub struct ChatHandle {
    owner: ThreadId,
    codec: Codec,
}

unsafe fn copy_output<T: Copy>(
    values: &[T],
    output: *mut T,
    capacity: usize,
    count: *mut usize,
) -> Result<()> {
    pointer(count)?;
    if output.is_null() && capacity != 0 {
        return Err(fail(INVALID, "NULL output requires zero capacity"));
    }
    // SAFETY: count is a caller-owned, writable disjoint usize record.
    unsafe { count.write(values.len()) };
    if output.is_null() && capacity == 0 {
        return Ok(());
    }
    if capacity < values.len() {
        return Err(fail(INVALID, "text output buffer is too small"));
    }
    if !values.is_empty() {
        pointer(output)?;
        // SAFETY: caller owns capacity initialized writable elements; no overlap
        // with input, handle or count. Only the checked values.len() is written.
        unsafe { ptr::copy_nonoverlapping(values.as_ptr(), output, values.len()) };
    }
    Ok(())
}

/// Load the exact SmolLM2 chat assets from the loaded model's own directory.
/// The caller reserves 64MiB in ADDITION to its numerical budget.
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
            if allowance < TOKENIZER_ALLOWANCE as u64
                || allowance > weights::INITIAL_MEMORY_LIMIT_BYTES as u64
            {
                return Err(fail(
                    INVALID,
                    "reserve at least 64MiB separately for text assets and buffers",
                ));
            }
            let config = match model.model.as_ref() {
                NativeModel::Cpu(model) => model.config(),
                #[cfg(all(
                    feature = "metal",
                    any(target_os = "macos", target_os = "ios"),
                    target_arch = "aarch64"
                ))]
                NativeModel::Metal(model) => model.config(),
            };
            if (
                config.vocab_size,
                config.hidden_size,
                config.intermediate_size,
                config.num_layers,
                config.query_heads,
                config.kv_heads,
                config.head_dim,
            ) != (49152, 576, 1536, 30, 9, 3, 64)
                || config.rope_theta != 100000.0
                || config.rms_norm_eps != 1e-5
            {
                return Err(fail(
                    UNSUPPORTED,
                    "text chat currently supports the selected SmolLM2-135M profile",
                ));
            }
            let codec =
                Codec::load(&model.directory).map_err(|e| fail(LOAD_ERROR, e.to_string()))?;
            out.write(Box::into_raw(Box::new(ChatHandle {
                owner: thread::current().id(),
                codec,
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
                return Err(fail(INVALID, "text token capacity exceeds 512"));
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
                .decode(slice(ids, len, 128)?)
                .map_err(|e| fail(INVALID, e.to_string()))?;
            copy_output(text.as_bytes(), output, capacity, count)
        })
    }
}

/// Release an immutable text handle. NULL is a no-op; no model lifetime retained.
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

    #[test]
    #[ignore = "explicit local tokenizer bridge check; no model download"]
    fn real_text_handle_bounds_thread_and_utf8() {
        let chat = Box::into_raw(Box::new(ChatHandle {
            owner: thread::current().id(),
            codec: Codec::load(Path::new("models/MLX-SmolLM2-135M-Instruct-8bit-0f0d9b8")).unwrap(),
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
        let address = chat as usize;
        assert_eq!(
            thread::spawn(move || unsafe {
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
