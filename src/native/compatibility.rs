//! Immutable metadata snapshots; never retained as authority to load a model.
use super::*;
use crate::mlx::compatibility::{self, Backend};
use std::io::Write;

/// Maximum JSON length, without a terminating NUL. Metadata inspection uses
/// bounded parsers; JSON ownership has a separate explicit logical allowance.
pub const MAX_COMPATIBILITY_JSON_BYTES: usize = 64 * 1024;
/// Covers two maximum JSON buffers during conversion and the handle record.
/// This is logical ownership, excluding allocator overhead and parser temporaries.
pub const COMPATIBILITY_JSON_ALLOWANCE_BYTES: usize = 2 * MAX_COMPATIBILITY_JSON_BYTES + 256;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CompatibilityOptions {
    pub abi_version: u32,
    pub struct_size: u32,
    /// 0 = scalar reference; 1 = Metal. Never selects a runtime GPU.
    pub backend: u32,
    pub context_limit: u32,
    /// Exact budget reported to the engine, without numerical bridge deduction.
    pub memory_limit_bytes: u64,
}
pub struct CompatibilityHandle {
    owner: ThreadId,
    json: Box<[u8]>,
}
const _: () = assert!(size_of::<CompatibilityHandle>() <= 256);
struct JsonBuffer(Vec<u8>);
impl Write for JsonBuffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > MAX_COMPATIBILITY_JSON_BYTES - self.0.len() {
            return Err(io::Error::other("compatibility JSON exceeds 64 KiB"));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(super) fn capture(report: &impl serde::Serialize) -> Result<*mut CompatibilityHandle> {
    let mut json = JsonBuffer(Vec::with_capacity(MAX_COMPATIBILITY_JSON_BYTES));
    serde_json::to_writer(&mut json, report)
        .map_err(|error| fail(ENGINE_ERROR, error.to_string()))?;
    Ok(Box::into_raw(Box::new(CompatibilityHandle {
        owner: thread::current().id(),
        json: json.0.into_boxed_slice(),
    })))
}

/// Capture one local compatibility report, including unsupported/unknown states.
/// A report is historical metadata, not permission to load without revalidation.
/// # Safety
/// Obey the parent foreign pointer/thread contract; out is cleared on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn toklet_mlx_inspect_create(
    path: *const u8,
    len: usize,
    opts: *const CompatibilityOptions,
    out: *mut *mut CompatibilityHandle,
    error: *mut NativeError,
) -> u32 {
    unsafe {
        boundary(error, || {
            pointer(out)?;
            out.write(ptr::null_mut());
            pointer(opts)?;
            let options = *opts;
            if options.abi_version != ABI_VERSION
                || options.struct_size as usize != size_of::<CompatibilityOptions>()
            {
                return Err(fail(
                    INVALID,
                    "unsupported inspection options version or size",
                ));
            }
            let backend = match options.backend {
                0 => Backend::ScalarReference,
                1 => Backend::Metal,
                _ => return Err(fail(INVALID, "unknown inspection backend")),
            };
            let memory_limit_bytes = usize::try_from(options.memory_limit_bytes)
                .map_err(|_| fail(INVALID, "inspection budget exceeds address width"))?;
            let path = local_path(path, len)?;
            let report = compatibility::inspect(
                path,
                backend,
                weights::LoadOptions {
                    context_limit: options.context_limit as usize,
                    memory_limit_bytes,
                },
            );
            out.write(capture(&report)?);
            Ok(())
        })
    }
}

/// Copy the captured JSON as UTF-8 bytes, no NUL. NULL+0 queries byte count.
/// A short buffer sets required count and copies nothing; other errors set count
/// to zero if count is valid. Source changes never alter this captured report.
/// # Safety
/// count/output/error/handle must be disjoint, live and initialized. Same thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn toklet_mlx_inspect_json(
    handle: *const CompatibilityHandle,
    output: *mut u8,
    capacity: usize,
    count: *mut usize,
    error: *mut NativeError,
) -> u32 {
    unsafe {
        boundary(error, || {
            pointer(count)?;
            count.write(0);
            pointer(handle)?;
            let handle = &*handle;
            owner(handle.owner)?;
            if capacity > MAX_COMPATIBILITY_JSON_BYTES {
                return Err(fail(INVALID, "inspection output capacity exceeds 64 KiB"));
            }
            copy_output(&handle.json, output, capacity, count)
        })
    }
}

/// Release exactly once on the creating thread; NULL is a no-op.
/// # Safety
/// Obey the parent foreign pointer/thread contract; no callers retain the handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn toklet_mlx_inspect_release(
    handle: *mut CompatibilityHandle,
    error: *mut NativeError,
) -> u32 {
    unsafe {
        boundary(error, || {
            if !handle.is_null() {
                pointer(handle)?;
                owner((*handle).owner)?;
                drop(Box::from_raw(handle));
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use std::{fs, path::PathBuf, sync::atomic::AtomicUsize};
    fn options() -> CompatibilityOptions {
        CompatibilityOptions {
            abi_version: ABI_VERSION,
            struct_size: size_of::<CompatibilityOptions>() as u32,
            backend: 1,
            context_limit: 32,
            memory_limit_bytes: 128 * 1024 * 1024,
        }
    }
    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    }
    fn create(path: &Path, options: CompatibilityOptions) -> *mut CompatibilityHandle {
        let path = path.to_str().unwrap().as_bytes();
        let mut handle = ptr::null_mut();
        assert_eq!(
            unsafe {
                toklet_mlx_inspect_create(
                    path.as_ptr(),
                    path.len(),
                    &options,
                    &mut handle,
                    ptr::null_mut(),
                )
            },
            OK
        );
        assert!(!handle.is_null());
        handle
    }
    fn bytes(handle: *const CompatibilityHandle) -> Vec<u8> {
        let mut count = usize::MAX;
        assert_eq!(
            unsafe {
                toklet_mlx_inspect_json(handle, ptr::null_mut(), 0, &mut count, ptr::null_mut())
            },
            OK
        );
        assert!((1..=MAX_COMPATIBILITY_JSON_BYTES).contains(&count));
        let mut out = vec![0; count];
        assert_eq!(
            unsafe {
                toklet_mlx_inspect_json(
                    handle,
                    out.as_mut_ptr(),
                    out.len(),
                    &mut count,
                    ptr::null_mut(),
                )
            },
            OK
        );
        out
    }
    fn release(handle: *mut CompatibilityHandle) {
        assert_eq!(
            unsafe { toklet_mlx_inspect_release(handle, ptr::null_mut()) },
            OK
        );
    }
    struct Temporary(PathBuf);
    impl Temporary {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "toklet-inspect-native-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn config(&self, value: Value) {
            fs::write(
                self.0.join("config.json"),
                serde_json::to_vec(&value).unwrap(),
            )
            .unwrap();
        }
    }
    impl Drop for Temporary {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }
    #[test]
    fn native_inspection_reports_each_family_without_bridge_budget_deduction() {
        assert_eq!(size_of::<CompatibilityOptions>(), 24);
        for (path, family) in [
            ("mlx-qwen-v1/model", "qwen2"),
            ("mlx-llama-v1/tied", "llama"),
            ("mlx-mistral-v1/untied", "mistral"),
            ("mlx-qwen3-v1/tied", "qwen3"),
        ] {
            for backend in [0, 1] {
                let options = CompatibilityOptions {
                    backend,
                    ..options()
                };
                let handle = create(&fixture(path), options);
                let value: Value = serde_json::from_slice(&bytes(handle)).unwrap();
                assert_eq!(value["architecture"], family);
                assert_eq!(value["status"], "compatible_untested");
                assert_eq!(
                    value["requested_memory_limit_bytes"],
                    options.memory_limit_bytes
                );
                assert_eq!(value["qualification_evaluated"], false);
                release(handle);
            }
        }
    }
    #[test]
    fn native_snapshot_survives_source_changes_and_preserves_failure_atomicity() {
        let temp = Temporary::new();
        temp.config(json!({"model_type":"future_graph"}));
        let handle = create(&temp.0, options());
        let original = bytes(handle);
        assert_eq!(
            serde_json::from_slice::<Value>(&original).unwrap()["status"],
            "unknown"
        );
        temp.config(json!({"model_type":"qwen3"}));
        assert_eq!(bytes(handle), original);
        let mut out = vec![0xa5; original.len()];
        let mut count = usize::MAX;
        assert_eq!(
            unsafe {
                toklet_mlx_inspect_json(
                    handle,
                    out.as_mut_ptr(),
                    out.len() - 1,
                    &mut count,
                    ptr::null_mut(),
                )
            },
            INVALID
        );
        assert_eq!(count, original.len());
        assert!(out.iter().all(|&b| b == 0xa5));
        for capacity in [1, MAX_COMPATIBILITY_JSON_BYTES + 1] {
            count = usize::MAX;
            assert_eq!(
                unsafe {
                    toklet_mlx_inspect_json(
                        handle,
                        ptr::null_mut(),
                        capacity,
                        &mut count,
                        ptr::null_mut(),
                    )
                },
                INVALID
            );
            assert_eq!(count, 0);
        }
        assert_eq!(
            unsafe {
                toklet_mlx_inspect_json(
                    handle,
                    out.as_mut_ptr(),
                    out.len(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                )
            },
            INVALID
        );
        assert!(out.iter().all(|&b| b == 0xa5));
        let address = handle as usize;
        std::thread::spawn(move || unsafe {
            let handle = address as *mut CompatibilityHandle;
            let mut count = usize::MAX;
            let mut out = [0xa5; 8];
            assert_eq!(
                toklet_mlx_inspect_json(
                    handle,
                    out.as_mut_ptr(),
                    out.len(),
                    &mut count,
                    ptr::null_mut()
                ),
                WRONG_THREAD
            );
            assert_eq!(count, 0);
            assert_eq!(out, [0xa5; 8]);
            assert_eq!(
                toklet_mlx_inspect_release(handle, ptr::null_mut()),
                WRONG_THREAD
            );
        })
        .join()
        .unwrap();
        assert_eq!(bytes(handle), original);
        release(handle);
        let changed = create(&temp.0, options());
        assert_eq!(
            serde_json::from_slice::<Value>(&bytes(changed)).unwrap()["reason"],
            "artifact_rejected"
        );
        release(changed);
    }
    #[test]
    fn native_invalid_boundary_arguments_clear_valid_outputs() {
        let valid = options();
        let path = b"missing";
        for options in [
            CompatibilityOptions {
                abi_version: 99,
                ..valid
            },
            CompatibilityOptions {
                struct_size: 0,
                ..valid
            },
            CompatibilityOptions {
                backend: 2,
                ..valid
            },
        ] {
            let mut handle = ptr::dangling_mut();
            assert_eq!(
                unsafe {
                    toklet_mlx_inspect_create(
                        path.as_ptr(),
                        path.len(),
                        &options,
                        &mut handle,
                        ptr::null_mut(),
                    )
                },
                INVALID
            );
            assert!(handle.is_null());
        }
        for path in [b"".as_slice(), b"a\0b", b"\xff"] {
            let mut handle = ptr::dangling_mut();
            assert_eq!(
                unsafe {
                    toklet_mlx_inspect_create(
                        path.as_ptr(),
                        path.len(),
                        &valid,
                        &mut handle,
                        ptr::null_mut(),
                    )
                },
                INVALID
            );
            assert!(handle.is_null());
        }
        let mut handle = ptr::dangling_mut();
        assert_eq!(
            unsafe {
                toklet_mlx_inspect_create(ptr::null(), 16385, &valid, &mut handle, ptr::null_mut())
            },
            INVALID
        );
        assert!(handle.is_null());
        assert_eq!(
            unsafe {
                toklet_mlx_inspect_create(
                    path.as_ptr(),
                    path.len(),
                    ptr::null(),
                    &mut handle,
                    ptr::null_mut(),
                )
            },
            INVALID
        );
        assert!(handle.is_null());
        assert_eq!(
            unsafe {
                toklet_mlx_inspect_create(
                    path.as_ptr(),
                    path.len(),
                    &valid,
                    ptr::null_mut(),
                    ptr::null_mut(),
                )
            },
            INVALID
        );
        let mut count = usize::MAX;
        assert_eq!(
            unsafe {
                toklet_mlx_inspect_json(
                    ptr::null(),
                    ptr::null_mut(),
                    0,
                    &mut count,
                    ptr::null_mut(),
                )
            },
            INVALID
        );
        assert_eq!(count, 0);
        release(ptr::null_mut());
        for options in [
            CompatibilityOptions {
                context_limit: 0,
                ..valid
            },
            CompatibilityOptions {
                memory_limit_bytes: 0,
                ..valid
            },
        ] {
            let handle = create(Path::new("missing"), options);
            assert_eq!(
                serde_json::from_slice::<Value>(&bytes(handle)).unwrap()["reason"],
                "invalid_request"
            );
            release(handle);
        }
        let handle = create(Path::new("missing"), valid);
        assert_eq!(
            serde_json::from_slice::<Value>(&bytes(handle)).unwrap()["reason"],
            "invalid_configuration"
        );
        release(handle);
    }
    #[test]
    fn json_writer_rejects_overflow_without_extending_retained_storage() {
        let mut buffer = JsonBuffer(Vec::with_capacity(MAX_COMPATIBILITY_JSON_BYTES));
        buffer
            .write_all(&vec![b'x'; MAX_COMPATIBILITY_JSON_BYTES])
            .unwrap();
        assert!(buffer.write_all(b"y").is_err());
        assert_eq!(buffer.0.len(), MAX_COMPATIBILITY_JSON_BYTES);
    }
}
