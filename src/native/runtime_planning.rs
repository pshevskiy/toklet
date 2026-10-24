//! Explicit runtime planning on the caller's dedicated inference thread.
use super::*;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RuntimePlanningOptions {
    pub abi_version: u32,
    pub struct_size: u32,
    pub context_limit: u32,
    pub capacity: u32,
    pub chunk: u32,
    pub position_offset: u32,
    /// 0 Original, 1 Compensated. Projection is always MlxAffine.
    pub normalization: u32,
    /// 0 FullLogits, 1 GreedyToken.
    pub readback: u32,
    /// 0 Reference, 1 Online; unqualified family/mode combinations reject.
    pub attention: u32,
    pub reserved: u32,
    /// Exact numerical budgets, without the ordinary bridge deduction.
    pub load_memory_limit_bytes: u64,
    pub session_memory_limit_bytes: u64,
}
impl RuntimePlanningOptions {
    fn validate(self) -> Result<(weights::LoadOptions, session::Options)> {
        if self.abi_version != ABI_VERSION
            || self.struct_size as usize != size_of::<Self>()
            || self.reserved != 0
            || self.normalization > 1
            || self.readback > 1
            || self.attention > 1
            || self.context_limit == 0
            || self.context_limit as usize > weights::MAX_CONTEXT_LIMIT
            || self.capacity == 0
            || self.capacity > self.context_limit
            || self.chunk == 0
            || self.chunk > 1024
            || self.chunk > self.capacity
            || self
                .position_offset
                .checked_add(self.capacity)
                .is_none_or(|n| n > self.context_limit)
            || self.load_memory_limit_bytes == 0
            || self.load_memory_limit_bytes > weights::INITIAL_MEMORY_LIMIT_BYTES as u64
            || self.session_memory_limit_bytes > weights::INITIAL_MEMORY_LIMIT_BYTES as u64
        {
            return Err(fail(
                INVALID,
                "runtime planning options outside ABI/engine bounds",
            ));
        }
        let load = weights::LoadOptions {
            context_limit: self.context_limit as usize,
            memory_limit_bytes: usize::try_from(self.load_memory_limit_bytes)
                .map_err(|_| fail(INVALID, "load budget exceeds address width"))?,
        };
        let options = session::Options {
            capacity: self.capacity as usize,
            prefill_chunk_size: self.chunk as usize,
            position_offset: self.position_offset as usize,
            memory_limit_bytes: usize::try_from(self.session_memory_limit_bytes)
                .map_err(|_| fail(INVALID, "session budget exceeds address width"))?,
        };
        Ok((load, options))
    }
}

/// Initialize Metal/MPS pipelines and capture shared runtime resource planning.
/// No numerical weights/KV/scratch/logits are allocated. Insufficient requested
/// budgets produce an OK JSON report with false fit flags. This snapshot never
/// authorizes loading without revalidation. No fallback when Metal is unavailable.
/// # Safety
/// Follow the parent pointer/creating-thread contract. A valid out is cleared on
/// error. Copy/release use toklet_mlx_inspect_json/release on the creating thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn toklet_mlx_plan_create(
    path: *const u8,
    len: usize,
    opts: *const RuntimePlanningOptions,
    out: *mut *mut CompatibilityHandle,
    error: *mut NativeError,
) -> u32 {
    unsafe {
        boundary(error, || {
            pointer(out)?;
            out.write(ptr::null_mut());
            pointer(opts)?;
            let raw = *opts;
            let (load, options) = raw.validate()?;
            let path = local_path(path, len)?;
            #[cfg(all(
                feature = "metal",
                any(target_os = "macos", target_os = "ios"),
                target_arch = "aarch64"
            ))]
            {
                use crate::metal::{self, AttentionMode, NormalizationMode, ReadbackMode};
                let report = metal::planning::inspect_mlx(
                    path,
                    load,
                    options,
                    if raw.normalization == 0 {
                        NormalizationMode::Original
                    } else {
                        NormalizationMode::Compensated
                    },
                    if raw.readback == 0 {
                        ReadbackMode::FullLogits
                    } else {
                        ReadbackMode::GreedyToken
                    },
                    if raw.attention == 0 {
                        AttentionMode::Reference
                    } else {
                        AttentionMode::Online
                    },
                )
                .map_err(|error| {
                    fail(
                        match error {
                            metal::Error::Options(_) => INVALID,
                            metal::Error::Unsupported(_) => UNSUPPORTED,
                            metal::Error::Weights(_) | metal::Error::Budget { .. } => LOAD_ERROR,
                            _ => ENGINE_ERROR,
                        },
                        error.to_string(),
                    )
                })?;
                out.write(compatibility::capture(&report)?);
                Ok(())
            }
            #[cfg(not(all(
                feature = "metal",
                any(target_os = "macos", target_os = "ios"),
                target_arch = "aarch64"
            )))]
            {
                let _ = (load, options, path);
                Err(fail(
                    UNSUPPORTED,
                    "Runtime Metal planning unavailable in this build/platform",
                ))
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn options() -> RuntimePlanningOptions {
        RuntimePlanningOptions {
            abi_version: ABI_VERSION,
            struct_size: size_of::<RuntimePlanningOptions>() as u32,
            context_limit: 32,
            capacity: 31,
            chunk: 3,
            position_offset: 1,
            normalization: 1,
            readback: 0,
            attention: 0,
            reserved: 0,
            load_memory_limit_bytes: 256 * 1024 * 1024,
            session_memory_limit_bytes: 256 * 1024 * 1024,
        }
    }
    #[test]
    fn runtime_planning_abi_and_invalid_arguments_clear_output_before_device() {
        assert_eq!(size_of::<RuntimePlanningOptions>(), 56);
        assert_eq!(
            std::mem::offset_of!(RuntimePlanningOptions, load_memory_limit_bytes),
            40
        );
        assert_eq!(
            std::mem::offset_of!(RuntimePlanningOptions, session_memory_limit_bytes),
            48
        );
        let valid = options();
        assert!(valid.validate().is_ok());
        assert!(
            RuntimePlanningOptions {
                session_memory_limit_bytes: 0,
                ..valid
            }
            .validate()
            .is_ok()
        );
        let invalid = [
            RuntimePlanningOptions {
                abi_version: 2,
                ..valid
            },
            RuntimePlanningOptions {
                struct_size: 0,
                ..valid
            },
            RuntimePlanningOptions {
                reserved: 1,
                ..valid
            },
            RuntimePlanningOptions {
                normalization: 2,
                ..valid
            },
            RuntimePlanningOptions {
                readback: 2,
                ..valid
            },
            RuntimePlanningOptions {
                attention: 2,
                ..valid
            },
            RuntimePlanningOptions {
                context_limit: 0,
                ..valid
            },
            RuntimePlanningOptions {
                context_limit: 32769,
                ..valid
            },
            RuntimePlanningOptions {
                capacity: 0,
                ..valid
            },
            RuntimePlanningOptions {
                capacity: 33,
                ..valid
            },
            RuntimePlanningOptions { chunk: 0, ..valid },
            RuntimePlanningOptions { chunk: 32, ..valid },
            RuntimePlanningOptions {
                chunk: 1025,
                ..valid
            },
            RuntimePlanningOptions {
                position_offset: u32::MAX,
                ..valid
            },
            RuntimePlanningOptions {
                load_memory_limit_bytes: 0,
                ..valid
            },
            RuntimePlanningOptions {
                load_memory_limit_bytes: weights::INITIAL_MEMORY_LIMIT_BYTES as u64 + 1,
                ..valid
            },
            RuntimePlanningOptions {
                session_memory_limit_bytes: u64::MAX,
                ..valid
            },
        ];
        for options in invalid {
            let mut out = ptr::dangling_mut::<CompatibilityHandle>();
            assert_eq!(
                unsafe {
                    toklet_mlx_plan_create(b"x".as_ptr(), 1, &options, &mut out, ptr::null_mut())
                },
                INVALID
            );
            assert!(out.is_null());
        }
        let oversized = vec![b'x'; 16385];
        let nul_path = [b'x', 0];
        for (path, length) in [
            (ptr::null(), 1),
            (b"".as_ptr(), 0),
            (b"\xff".as_ptr(), 1),
            (nul_path.as_ptr(), nul_path.len()),
            (oversized.as_ptr(), oversized.len()),
        ] {
            let mut out = ptr::dangling_mut::<CompatibilityHandle>();
            assert_eq!(
                unsafe { toklet_mlx_plan_create(path, length, &valid, &mut out, ptr::null_mut()) },
                INVALID
            );
            assert!(out.is_null());
        }
        let mut out = ptr::dangling_mut::<CompatibilityHandle>();
        assert_eq!(
            unsafe {
                toklet_mlx_plan_create(b"x".as_ptr(), 1, ptr::null(), &mut out, ptr::null_mut())
            },
            INVALID
        );
        assert!(out.is_null());
        assert_eq!(
            unsafe {
                toklet_mlx_plan_create(b"x".as_ptr(), 1, &valid, ptr::null_mut(), ptr::null_mut())
            },
            INVALID
        );
    }
    #[test]
    #[cfg(not(all(
        feature = "metal",
        any(target_os = "macos", target_os = "ios"),
        target_arch = "aarch64"
    )))]
    fn runtime_planning_unavailable_build_is_explicit_without_fallback() {
        let mut out = ptr::dangling_mut::<CompatibilityHandle>();
        assert_eq!(
            unsafe {
                toklet_mlx_plan_create(
                    b"missing".as_ptr(),
                    7,
                    &options(),
                    &mut out,
                    ptr::null_mut(),
                )
            },
            UNSUPPORTED
        );
        assert!(out.is_null());
    }
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    mod gpu {
        use super::*;
        use crate::metal::{self, AttentionMode, NormalizationMode, ReadbackMode};
        use serde_json::Value;
        use std::{fs, path::PathBuf};
        fn path(name: &str) -> PathBuf {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures")
                .join(name)
        }
        fn create(path: &Path, options: RuntimePlanningOptions) -> *mut CompatibilityHandle {
            let bytes = path.to_str().unwrap().as_bytes();
            let mut out = ptr::null_mut();
            assert_eq!(
                unsafe {
                    toklet_mlx_plan_create(
                        bytes.as_ptr(),
                        bytes.len(),
                        &options,
                        &mut out,
                        ptr::null_mut(),
                    )
                },
                OK
            );
            assert!(!out.is_null());
            out
        }
        fn bytes(handle: *const CompatibilityHandle) -> Vec<u8> {
            let mut count = 0;
            assert_eq!(
                unsafe {
                    toklet_mlx_inspect_json(handle, ptr::null_mut(), 0, &mut count, ptr::null_mut())
                },
                OK
            );
            assert!(count > 0 && count <= MAX_COMPATIBILITY_JSON_BYTES);
            let mut bytes = vec![0xa5; count + 1];
            let length = count;
            assert_eq!(
                unsafe {
                    toklet_mlx_inspect_json(
                        handle,
                        bytes.as_mut_ptr(),
                        length - 1,
                        &mut count,
                        ptr::null_mut(),
                    )
                },
                INVALID
            );
            assert_eq!(count, length);
            assert!(bytes.iter().all(|&v| v == 0xa5));
            assert_eq!(
                unsafe {
                    toklet_mlx_inspect_json(
                        handle,
                        bytes.as_mut_ptr(),
                        length,
                        &mut count,
                        ptr::null_mut(),
                    )
                },
                OK
            );
            assert_eq!(count, length);
            assert_eq!(bytes[length], 0xa5);
            bytes.truncate(length);
            bytes
        }
        #[test]
        #[ignore = "Explicit Apple GPU runtime C/direct equality, budgets and owner contract"]
        fn native_runtime_planning_matches_direct_reports_and_keeps_snapshot_ownership() {
            let mut reports = 0;
            for name in [
                "mlx-qwen-v1/model",
                "mlx-llama-v1/tied",
                "mlx-mistral-v1/untied",
                "mlx-qwen3-v1/tied",
            ] {
                for normalization in 0..=1 {
                    for readback in 0..=1 {
                        for attention in 0..=1 {
                            if attention == 1 && name != "mlx-qwen-v1/model" {
                                continue;
                            }
                            let options = RuntimePlanningOptions {
                                normalization,
                                readback,
                                attention,
                                ..options()
                            };
                            let p = path(name);
                            let handle = create(&p, options);
                            let output = bytes(handle);
                            let (load, session) = options.validate().unwrap();
                            let direct = metal::planning::inspect_mlx(
                                &p,
                                load,
                                session,
                                if normalization == 0 {
                                    NormalizationMode::Original
                                } else {
                                    NormalizationMode::Compensated
                                },
                                if readback == 0 {
                                    ReadbackMode::FullLogits
                                } else {
                                    ReadbackMode::GreedyToken
                                },
                                if attention == 0 {
                                    AttentionMode::Reference
                                } else {
                                    AttentionMode::Online
                                },
                            )
                            .unwrap();
                            assert_eq!(
                                serde_json::from_slice::<Value>(&output).unwrap(),
                                serde_json::to_value(direct).unwrap()
                            );
                            assert_eq!(
                                unsafe { toklet_mlx_inspect_release(handle, ptr::null_mut()) },
                                OK
                            );
                            reports += 1;
                        }
                    }
                }
            }
            assert_eq!(reports, 20);
            let handle = create(
                &path("mlx-qwen-v1/model"),
                RuntimePlanningOptions {
                    load_memory_limit_bytes: 1,
                    session_memory_limit_bytes: 0,
                    ..options()
                },
            );
            let value: Value = serde_json::from_slice(&bytes(handle)).unwrap();
            assert_eq!(value["fits_requested_budgets"], false);
            assert_eq!(value["requested_load_budget_bytes"], 1);
            assert_eq!(value["requested_session_budget_bytes"], 0);
            let address = handle as usize;
            std::thread::spawn(move || unsafe {
                let handle = address as *mut CompatibilityHandle;
                let mut count = usize::MAX;
                let mut output = [0xa5; 8];
                assert_eq!(
                    toklet_mlx_inspect_json(
                        handle,
                        output.as_mut_ptr(),
                        output.len(),
                        &mut count,
                        ptr::null_mut()
                    ),
                    WRONG_THREAD
                );
                assert_eq!(count, 0);
                assert_eq!(output, [0xa5; 8]);
                assert_eq!(
                    toklet_mlx_inspect_release(handle, ptr::null_mut()),
                    WRONG_THREAD
                );
            })
            .join()
            .unwrap();
            assert_eq!(
                serde_json::from_slice::<Value>(&bytes(handle)).unwrap(),
                value
            );
            assert_eq!(
                unsafe { toklet_mlx_inspect_release(handle, ptr::null_mut()) },
                OK
            );
            for name in [
                "mlx-llama-v1/tied",
                "mlx-mistral-v1/untied",
                "mlx-qwen3-v1/tied",
            ] {
                let p = path(name);
                let raw = p.to_str().unwrap().as_bytes();
                let mut out = ptr::dangling_mut::<CompatibilityHandle>();
                let options = RuntimePlanningOptions {
                    attention: 1,
                    ..options()
                };
                assert_eq!(
                    unsafe {
                        toklet_mlx_plan_create(
                            raw.as_ptr(),
                            raw.len(),
                            &options,
                            &mut out,
                            ptr::null_mut(),
                        )
                    },
                    UNSUPPORTED
                );
                assert!(out.is_null());
            }
            struct Temp(PathBuf);
            impl Drop for Temp {
                fn drop(&mut self) {
                    fs::remove_dir_all(&self.0).unwrap();
                }
            }
            let temp_path = std::env::temp_dir()
                .join(format!("toklet-runtime-snapshot-{}", std::process::id()));
            fs::create_dir(&temp_path).unwrap();
            let temp = Temp(temp_path);
            for name in ["config.json", "model.safetensors"] {
                fs::copy(path("mlx-qwen-v1/model").join(name), temp.0.join(name)).unwrap();
            }
            let handle = create(&temp.0, options());
            let before = bytes(handle);
            fs::write(temp.0.join("config.json"), b"{}").unwrap();
            assert_eq!(bytes(handle), before);
            let raw = temp.0.to_str().unwrap().as_bytes();
            let mut out = ptr::dangling_mut::<CompatibilityHandle>();
            assert_eq!(
                unsafe {
                    toklet_mlx_plan_create(
                        raw.as_ptr(),
                        raw.len(),
                        &options(),
                        &mut out,
                        ptr::null_mut(),
                    )
                },
                LOAD_ERROR
            );
            assert!(out.is_null());
            assert_eq!(
                unsafe { toklet_mlx_inspect_release(handle, ptr::null_mut()) },
                OK
            );
            println!(
                "NATIVE_RUNTIME_PLANNING {}",
                serde_json::json!({"direct_reports_equal":reports,"fits_false":true,"snapshot_immutable":true,"wrong_thread_rejected":true,"passed":true})
            );
        }
    }
}
