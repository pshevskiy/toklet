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
    /// 0 Reference, 1 Online, 2 native model factory policy. Reports resolve2.
    /// Unqualified explicit family/mode combinations reject.
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
            || self.attention > 2
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
    unsafe { plan_create(path, len, opts, out, error, ReportKind::Resources) }
}

/// Search numerical capacity within the requested upper bound, using the same
/// explicit modes or native factory policy as toklet_mlx_plan_create. Returns
/// {resources, capacity}; neither the selected capacity nor an OK status grants
/// loading, context qualification or application/device fit. Check load flags
/// separately. The requested chunk is the search minimum; modes/offset stay fixed.
/// # Safety
/// Same pointer, disjointness, creating-thread and release contract as
/// toklet_mlx_plan_create. A valid out is cleared on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn toklet_mlx_capacity_plan_create(
    path: *const u8,
    len: usize,
    opts: *const RuntimePlanningOptions,
    out: *mut *mut CompatibilityHandle,
    error: *mut NativeError,
) -> u32 {
    unsafe { plan_create(path, len, opts, out, error, ReportKind::Capacity) }
}

#[derive(Clone, Copy)]
enum ReportKind {
    Resources,
    Capacity,
}

#[cfg(all(
    feature = "metal",
    any(target_os = "macos", target_os = "ios"),
    target_arch = "aarch64"
))]
fn planning_error(error: crate::metal::Error) -> Failure {
    use crate::metal::Error;
    fail(
        match error {
            Error::Options(_) => INVALID,
            Error::Unsupported(_) => UNSUPPORTED,
            Error::Weights(_) | Error::Budget { .. } => LOAD_ERROR,
            _ => ENGINE_ERROR,
        },
        error.to_string(),
    )
}

// One cold ABI path keeps validation, error codes and handle ownership identical.
unsafe fn plan_create(
    path: *const u8,
    len: usize,
    opts: *const RuntimePlanningOptions,
    out: *mut *mut CompatibilityHandle,
    error: *mut NativeError,
    kind: ReportKind,
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
                let normalization = if raw.normalization == 0 {
                    NormalizationMode::Original
                } else {
                    NormalizationMode::Compensated
                };
                let readback = if raw.readback == 0 {
                    ReadbackMode::FullLogits
                } else {
                    ReadbackMode::GreedyToken
                };
                let attention = if raw.attention == 0 {
                    AttentionMode::Reference
                } else {
                    AttentionMode::Online
                };
                let handle = match kind {
                    ReportKind::Resources => {
                        let report = if raw.attention == 2 {
                            metal::planning::inspect_mlx_with_native_attention(
                                path,
                                load,
                                options,
                                normalization,
                                readback,
                            )
                        } else {
                            metal::planning::inspect_mlx(
                                path,
                                load,
                                options,
                                normalization,
                                readback,
                                attention,
                            )
                        }
                        .map_err(planning_error)?;
                        compatibility::capture(&report)?
                    }
                    ReportKind::Capacity => {
                        let report = if raw.attention == 2 {
                            metal::planning::inspect_mlx_capacity_with_native_attention(
                                path,
                                load,
                                options,
                                normalization,
                                readback,
                            )
                        } else {
                            metal::planning::inspect_mlx_capacity(
                                path,
                                load,
                                options,
                                normalization,
                                readback,
                                attention,
                            )
                        }
                        .map_err(planning_error)?;
                        compatibility::capture(&report)?
                    }
                };
                out.write(handle);
                Ok(())
            }
            #[cfg(not(all(
                feature = "metal",
                any(target_os = "macos", target_os = "ios"),
                target_arch = "aarch64"
            )))]
            {
                let _ = (load, options, path, kind);
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
    type Create = unsafe extern "C" fn(
        *const u8,
        usize,
        *const RuntimePlanningOptions,
        *mut *mut CompatibilityHandle,
        *mut NativeError,
    ) -> u32;

    #[test]
    fn runtime_planning_abi_and_invalid_arguments_clear_output_before_device() {
        for create in [
            toklet_mlx_plan_create as Create,
            toklet_mlx_capacity_plan_create,
        ] {
            invalid_arguments(create);
        }
    }

    fn invalid_arguments(create: Create) {
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
                attention: 3,
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
                unsafe { create(b"x".as_ptr(), 1, &options, &mut out, ptr::null_mut()) },
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
                unsafe { create(path, length, &valid, &mut out, ptr::null_mut()) },
                INVALID
            );
            assert!(out.is_null());
        }
        let mut out = ptr::dangling_mut::<CompatibilityHandle>();
        assert_eq!(
            unsafe { create(b"x".as_ptr(), 1, ptr::null(), &mut out, ptr::null_mut()) },
            INVALID
        );
        assert!(out.is_null());
        assert_eq!(
            unsafe { create(b"x".as_ptr(), 1, &valid, ptr::null_mut(), ptr::null_mut()) },
            INVALID
        );
    }
    #[test]
    fn native_model_allowance_subtraction_is_checked_at_boundaries() {
        assert_eq!(MODEL_SESSION_BRIDGE_BYTES, 20480);
        for budget in [MODEL_SESSION_BRIDGE_BYTES - 1, MODEL_SESSION_BRIDGE_BYTES] {
            let raw = Options {
                abi_version: ABI_VERSION,
                struct_size: size_of::<Options>() as u32,
                capacity: 32,
                chunk: 3,
                memory_limit_bytes: budget as u64,
            };
            assert!(matches!(
                unsafe { super::super::options(&raw) },
                Err(Failure(INVALID, _))
            ));
            let mut out = ptr::dangling_mut::<ModelHandle>();
            assert_eq!(
                unsafe { toklet_model_load_mlx(b"x".as_ptr(), 1, &raw, &mut out, ptr::null_mut()) },
                INVALID
            );
            assert!(out.is_null());
        }
        let raw = Options {
            abi_version: ABI_VERSION,
            struct_size: size_of::<Options>() as u32,
            capacity: 32,
            chunk: 3,
            memory_limit_bytes: (MODEL_SESSION_BRIDGE_BYTES + 1) as u64,
        };
        assert_eq!(
            unsafe { super::super::options(&raw) }
                .unwrap()
                .memory_limit_bytes,
            1
        );
        assert!(
            RuntimePlanningOptions {
                attention: 2,
                ..options()
            }
            .validate()
            .is_ok()
        );
    }
    #[test]
    #[cfg(not(all(
        feature = "metal",
        any(target_os = "macos", target_os = "ios"),
        target_arch = "aarch64"
    )))]
    fn runtime_planning_unavailable_build_is_explicit_without_fallback() {
        for create in [
            toklet_mlx_plan_create as Create,
            toklet_mlx_capacity_plan_create,
        ] {
            let mut out = ptr::dangling_mut::<CompatibilityHandle>();
            assert_eq!(
                unsafe {
                    create(
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
            create_with(toklet_mlx_plan_create, path, options)
        }
        fn create_with(
            create: Create,
            path: &Path,
            options: RuntimePlanningOptions,
        ) -> *mut CompatibilityHandle {
            let bytes = path.to_str().unwrap().as_bytes();
            let mut out = ptr::null_mut();
            assert_eq!(
                unsafe {
                    create(
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
        fn report(create: Create, path: &Path, request: RuntimePlanningOptions) -> Value {
            let handle = create_with(create, path, request);
            let first = bytes(handle);
            assert_eq!(bytes(handle), first);
            let value = serde_json::from_slice(&first).unwrap();
            assert_eq!(
                unsafe { toklet_mlx_inspect_release(handle, ptr::null_mut()) },
                OK
            );
            value
        }

        #[test]
        #[ignore = "Explicit Apple GPU capacity C/direct policy equality and independent load/session fit"]
        fn native_capacity_planning_preserves_policy_bounds_and_separate_load_fit() {
            for (directory, attention) in [
                ("mlx-qwen-v1/model", AttentionMode::Online),
                ("mlx-llama-v1/tied", AttentionMode::Reference),
                ("mlx-mistral-v1/untied", AttentionMode::Reference),
                ("mlx-qwen3-v1/tied", AttentionMode::Reference),
            ] {
                let p = path(directory);
                let request = RuntimePlanningOptions {
                    attention: 2,
                    ..options()
                };
                let capacity = report(toklet_mlx_capacity_plan_create, &p, request);
                let single = report(toklet_mlx_plan_create, &p, request);
                // The old JSON does not gain an envelope or change its fields.
                assert_eq!(capacity["resources"], single);
                assert!(single.get("resources").is_none());
                assert_eq!(capacity["capacity"]["minimum"], 3);
                assert_eq!(capacity["capacity"]["maximum"], 31);
                assert_eq!(capacity["capacity"]["selected"]["capacity"], 31);
                assert!(capacity["capacity"]["first_excluded"].is_null());
                assert_eq!(capacity["resources"]["numerical_buffers_allocated"], false);
                assert_eq!(capacity["resources"]["qualification_evaluated"], false);
                let (load, session) = request.validate().unwrap();
                let direct = metal::planning::inspect_mlx_capacity(
                    &p,
                    load,
                    session,
                    NormalizationMode::Compensated,
                    ReadbackMode::FullLogits,
                    attention,
                )
                .unwrap();
                assert_eq!(capacity, serde_json::to_value(direct).unwrap());
                let explicit = RuntimePlanningOptions {
                    attention: if attention == AttentionMode::Online {
                        1
                    } else {
                        0
                    },
                    ..request
                };
                assert_eq!(
                    capacity,
                    report(toklet_mlx_capacity_plan_create, &p, explicit)
                );
                if attention == AttentionMode::Reference {
                    let unsupported = RuntimePlanningOptions {
                        attention: 1,
                        ..request
                    };
                    let raw = p.to_str().unwrap().as_bytes();
                    let mut handle = ptr::dangling_mut();
                    assert_eq!(
                        unsafe {
                            toklet_mlx_capacity_plan_create(
                                raw.as_ptr(),
                                raw.len(),
                                &unsupported,
                                &mut handle,
                                ptr::null_mut(),
                            )
                        },
                        UNSUPPORTED
                    );
                    assert!(handle.is_null());
                }
            }
            let p = path("mlx-qwen-v1/model");
            let request = RuntimePlanningOptions {
                attention: 2,
                ..options()
            };
            let at17 = report(
                toklet_mlx_plan_create,
                &p,
                RuntimePlanningOptions {
                    capacity: 17,
                    ..request
                },
            );
            let bytes17 = at17["session_memory"]["admitted_bytes"].as_u64().unwrap();
            for (budget, expected) in [(bytes17, 17), (bytes17 - 1, 16)] {
                let value = report(
                    toklet_mlx_capacity_plan_create,
                    &p,
                    RuntimePlanningOptions {
                        session_memory_limit_bytes: budget,
                        ..request
                    },
                );
                assert_eq!(value["capacity"]["selected"]["capacity"], expected);
                assert_eq!(
                    value["capacity"]["first_excluded"]["capacity"],
                    expected + 1
                );
                assert_eq!(value["capacity"]["first_excluded"]["fits_budget"], false);
                assert_eq!(value["resources"]["fits_session_budget"], false);
            }
            let empty = report(
                toklet_mlx_capacity_plan_create,
                &p,
                RuntimePlanningOptions {
                    session_memory_limit_bytes: 0,
                    ..request
                },
            );
            assert!(empty["capacity"]["selected"].is_null());
            assert_eq!(empty["capacity"]["first_excluded"]["capacity"], 3);
            // A fitting session range cannot authorize loading under an insufficient load budget.
            let load_rejected = report(
                toklet_mlx_capacity_plan_create,
                &p,
                RuntimePlanningOptions {
                    load_memory_limit_bytes: 1,
                    ..request
                },
            );
            assert_eq!(load_rejected["capacity"]["selected"]["capacity"], 31);
            assert_eq!(load_rejected["resources"]["fits_source_load_budget"], false);
            assert_eq!(
                load_rejected["resources"]["fits_resident_load_budget"],
                false
            );
            assert_eq!(load_rejected["resources"]["fits_requested_budgets"], false);
            println!(
                "NATIVE_CAPACITY_PLANNING {}",
                serde_json::json!({
                    "factory_cases":4,"bounds":[3,31],"position_offset":1,
                    "exact_and_below_session_budget":true,"load_fit_separate":true,
                    "numerical_buffers_allocated":false,"passed":true
                })
            );
        }

        #[test]
        #[ignore = "Explicit Apple GPU native-policy plans match actual model/session factory budgets"]
        fn native_policy_matches_factory_and_frozen_parent_memory() {
            let frozen: Value = serde_json::from_slice(include_bytes!(
                "../../tests/fixtures/metal-resource-v1/baseline.json"
            ))
            .unwrap();
            let mut cases = 0;
            for (directory, attention) in [
                ("mlx-qwen-v1/model", "Online"),
                ("mlx-llama-v1/tied", "Reference"),
                ("mlx-mistral-v1/untied", "Reference"),
                ("mlx-qwen3-v1/tied", "Reference"),
            ] {
                let p = path(directory);
                let raw_path = p.to_str().unwrap().as_bytes();
                let request = RuntimePlanningOptions {
                    capacity: 32,
                    position_offset: 0,
                    attention: 2,
                    load_memory_limit_bytes: 256 * 1024 * 1024 - MODEL_SESSION_BRIDGE_BYTES as u64,
                    session_memory_limit_bytes: 256 * 1024 * 1024
                        - MODEL_SESSION_BRIDGE_BYTES as u64,
                    ..options()
                };
                let handle = create(&p, request);
                let report: Value = serde_json::from_slice(&bytes(handle)).unwrap();
                assert_eq!(report["attention"], attention);
                let baseline = frozen["cases"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|row| {
                        row["directory"] == directory
                            && row["chunk"] == 3
                            && row["readback"] == "FullLogits"
                            && row["attention"] == attention
                    })
                    .unwrap();
                assert_eq!(report["session_memory"], baseline["memory"]);
                let load_minimum = report["source_admitted_load_bytes"]
                    .as_u64()
                    .unwrap()
                    .max(report["resident_admitted_load_bytes"].as_u64().unwrap());
                let session_minimum = report["session_memory"]["admitted_bytes"].as_u64().unwrap();
                let mut load_options = Options {
                    abi_version: ABI_VERSION,
                    struct_size: size_of::<Options>() as u32,
                    capacity: 32,
                    chunk: 3,
                    memory_limit_bytes: load_minimum + MODEL_SESSION_BRIDGE_BYTES as u64 - 1,
                };
                let mut model = ptr::dangling_mut::<ModelHandle>();
                assert_eq!(
                    unsafe {
                        toklet_model_load_mlx(
                            raw_path.as_ptr(),
                            raw_path.len(),
                            &load_options,
                            &mut model,
                            ptr::null_mut(),
                        )
                    },
                    LOAD_ERROR
                );
                assert!(model.is_null());
                load_options.memory_limit_bytes += 1;
                assert_eq!(
                    unsafe {
                        toklet_model_load_mlx(
                            raw_path.as_ptr(),
                            raw_path.len(),
                            &load_options,
                            &mut model,
                            ptr::null_mut(),
                        )
                    },
                    OK
                );
                assert!(!model.is_null());
                let weights = match unsafe { &*model }.model.as_ref() {
                    NativeModel::Metal(m) => m.weights_bytes(),
                    NativeModel::Cpu(_) => panic!("Expected explicit Metal model"),
                };
                assert_eq!(report["resident_weights_bytes"], weights);
                let mut session_options = Options {
                    memory_limit_bytes: session_minimum + MODEL_SESSION_BRIDGE_BYTES as u64 - 1,
                    ..load_options
                };
                let mut session = ptr::dangling_mut::<SessionHandle>();
                assert_eq!(
                    unsafe {
                        toklet_session_create(
                            model,
                            &session_options,
                            &mut session,
                            ptr::null_mut(),
                        )
                    },
                    ENGINE_ERROR
                );
                assert!(session.is_null());
                session_options.memory_limit_bytes += 1;
                assert_eq!(
                    unsafe {
                        toklet_session_create(
                            model,
                            &session_options,
                            &mut session,
                            ptr::null_mut(),
                        )
                    },
                    OK
                );
                assert!(!session.is_null());
                let mut state = Report::default();
                assert_eq!(
                    unsafe { toklet_session_state(session, &mut state, ptr::null_mut()) },
                    OK
                );
                assert_eq!(
                    state.admitted_bytes,
                    session_minimum + MODEL_SESSION_BRIDGE_BYTES as u64
                );
                assert_eq!(
                    unsafe { toklet_session_release(session, ptr::null_mut()) },
                    OK
                );
                assert_eq!(unsafe { toklet_model_release(model, ptr::null_mut()) }, OK);
                assert_eq!(
                    unsafe { toklet_mlx_inspect_release(handle, ptr::null_mut()) },
                    OK
                );
                cases += 1;
            }
            println!(
                "NATIVE_FACTORY_POLICY {}",
                serde_json::json!({"cases":cases,"frozen_parent_memory_equal":true,"exact_and_below_load_and_session_budgets":true,"passed":true})
            );
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
