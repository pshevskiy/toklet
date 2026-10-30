use super::*;
#[test]
fn audited_qwen2_model_boundary_clears_output_before_any_load() {
    for (path, options) in [
        (&b""[..], opts()),
        (&b"bad\0path"[..], opts()),
        (&b"\xff"[..], opts()),
        (
            &b"missing"[..],
            Options {
                abi_version: 99,
                ..opts()
            },
        ),
        (
            &b"missing"[..],
            Options {
                struct_size: 0,
                ..opts()
            },
        ),
        (
            &b"missing"[..],
            Options {
                memory_limit_bytes: MODEL_SESSION_BRIDGE_BYTES as u64,
                ..opts()
            },
        ),
    ] {
        let mut out = ptr::dangling_mut();
        assert_eq!(
            unsafe {
                toklet_model_load_mlx_qwen2_audited(
                    path.as_ptr(),
                    path.len(),
                    &options,
                    &mut out,
                    ptr::null_mut(),
                )
            },
            INVALID
        );
        assert!(out.is_null());
    }
    let mut out = ptr::dangling_mut();
    unsafe {
        assert_eq!(
            toklet_model_load_mlx_qwen2_audited(
                b"missing".as_ptr(),
                7,
                ptr::null(),
                &mut out,
                ptr::null_mut()
            ),
            INVALID
        );
        assert!(out.is_null());
        assert_eq!(
            toklet_model_load_mlx_qwen2_audited(
                b"missing".as_ptr(),
                7,
                &opts(),
                ptr::null_mut(),
                ptr::null_mut()
            ),
            INVALID
        );
    }
}
#[test]
fn audited_qwen2_valid_missing_source_never_creates_a_model() {
    let mut out = ptr::dangling_mut();
    let path = b"/toklet-absent-qwen2-audit-source";
    let options = Options {
        capacity: 16,
        chunk: 2,
        memory_limit_bytes: 256 * 1024 * 1024,
        ..opts()
    };
    let code = unsafe {
        toklet_model_load_mlx_qwen2_audited(
            path.as_ptr(),
            path.len(),
            &options,
            &mut out,
            ptr::null_mut(),
        )
    };
    if cfg!(all(
        feature = "metal",
        any(target_os = "macos", target_os = "ios"),
        target_arch = "aarch64"
    )) {
        assert_eq!(code, LOAD_ERROR);
    } else {
        assert_eq!(code, UNSUPPORTED);
    }
    assert!(out.is_null());
}
