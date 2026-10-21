use super::*;
use crate::{
    model::{Config, LayerWeights, Projection, Weights},
    sampling,
};
#[path = "../../tests/support/mod.rs"]
mod fixtures;

fn model() -> model::Model {
    let f = fixtures::Fixture::tiny();
    let m = &f.manifest["model"];
    let n = |key: &str| m[key].as_u64().unwrap() as usize;
    let c = Config {
        vocab_size: n("vocab_size"),
        hidden_size: n("hidden_size"),
        intermediate_size: n("intermediate_size"),
        num_layers: n("num_layers"),
        query_heads: n("num_query_heads"),
        kv_heads: n("num_kv_heads"),
        head_dim: n("head_dim"),
        max_positions: n("capacity"),
        rms_norm_eps: m["rms_norm_eps"].as_f64().unwrap() as f32,
        rope_theta: m["rope_theta"].as_f64().unwrap() as f32,
    };
    let layers = (0..c.num_layers)
        .map(|i| {
            let p = format!("weights.layers.{i}");
            let proj = |name| Projection {
                weight: f.f32(&format!("{p}.{name}_proj.weight")),
                bias: f.f32(&format!("{p}.{name}_proj.bias")),
            };
            LayerWeights {
                input_norm: f.f32(&format!("{p}.input_layernorm")),
                post_attention_norm: f.f32(&format!("{p}.post_attention_layernorm")),
                q: proj("q"),
                k: proj("k"),
                v: proj("v"),
                o: f.f32(&format!("{p}.o_proj.weight")),
                gate: f.f32(&format!("{p}.gate_proj.weight")),
                up: f.f32(&format!("{p}.up_proj.weight")),
                down: f.f32(&format!("{p}.down_proj.weight")),
            }
        })
        .collect();
    model::Model::new(
        c,
        Weights {
            embedding: f.f32("weights.embed_tokens"),
            layers,
            final_norm: f.f32("weights.norm"),
        },
    )
    .unwrap()
}
fn opts() -> Options {
    Options {
        abi_version: ABI_VERSION,
        struct_size: size_of::<Options>() as u32,
        capacity: 32,
        chunk: 3,
        memory_limit_bytes: 128 * 1024 * 1024,
    }
}
fn handles() -> (*mut ModelHandle, *mut SessionHandle, *mut CancelHandle) {
    let m = Box::into_raw(Box::new(ModelHandle {
        owner: thread::current().id(),
        model: Rc::new(NativeModel::Cpu(Box::new(model()))),
    }));
    let mut s = ptr::null_mut();
    let mut c = ptr::null_mut();
    unsafe {
        assert_eq!(
            toklet_session_create(m, &opts(), &mut s, ptr::null_mut()),
            OK
        );
        assert_eq!(toklet_cancel_create(&mut c, ptr::null_mut()), OK);
    }
    (m, s, c)
}
unsafe fn generate(
    s: *mut SessionHandle,
    c: *mut CancelHandle,
    input: &[u32],
    op: u32,
    count: u32,
    emit: Emit,
    ctx: *mut c_void,
) -> (u32, Report) {
    let mut r = Report::default();
    let code = unsafe {
        toklet_session_generate(
            s,
            op,
            input.as_ptr(),
            input.len(),
            count,
            ptr::null(),
            0,
            c,
            emit,
            ctx,
            &mut r,
            ptr::null_mut(),
        )
    };
    (code, r)
}
unsafe fn cleanup(m: *mut ModelHandle, s: *mut SessionHandle, c: *mut CancelHandle) {
    unsafe {
        assert_eq!(toklet_session_release(s, ptr::null_mut()), OK);
        assert_eq!(toklet_model_release(m, ptr::null_mut()), OK);
        assert_eq!(toklet_cancel_release(c, ptr::null_mut()), OK);
    }
}
unsafe extern "C" fn collect(ctx: *mut c_void, e: TokenEvent) -> u32 {
    unsafe {
        (*(ctx.cast::<Vec<u32>>())).push(e.token_id);
    }
    0
}
#[test]
fn retained_model_and_pending_continuation_match_independent_greedy_fixture() {
    let f = fixtures::Fixture::tiny();
    let expected = f.u32("greedy.tokens");
    let (m, s, c) = handles();
    let weak = unsafe { Rc::downgrade(&(*m).model) };
    unsafe {
        assert_eq!(toklet_model_release(m, ptr::null_mut()), OK);
    }
    assert!(weak.upgrade().is_some());
    let mut ids = Vec::<u32>::new();
    let (code, r) = unsafe {
        generate(
            s,
            c,
            &[1, 7, 13, 3],
            0,
            2,
            Some(collect),
            (&mut ids as *mut Vec<u32>).cast(),
        )
    };
    assert_eq!(code, OK);
    assert_eq!(
        (r.history_tokens, r.evaluated_tokens, r.pending_tokens),
        (6, 5, 1)
    );
    let (code, r) = unsafe {
        generate(
            s,
            c,
            &[],
            2,
            2,
            Some(collect),
            (&mut ids as *mut Vec<u32>).cast(),
        )
    };
    assert_eq!(code, OK);
    assert_eq!(r.pending_tokens, 1);
    assert_eq!(ids, expected);
    unsafe {
        cleanup(ptr::null_mut(), s, c);
    }
    assert!(weak.upgrade().is_none());
}
#[test]
fn callback_reentry_is_busy_and_failure_preserves_the_sampled_token() {
    unsafe extern "C" fn reenter(ctx: *mut c_void, _: TokenEvent) -> u32 {
        let s = ctx.cast::<SessionHandle>();
        unsafe {
            assert_eq!(toklet_session_reset(s, ptr::null_mut()), BUSY);
            assert_eq!(toklet_session_release(s, ptr::null_mut()), BUSY);
            let mut count = 0;
            assert_eq!(
                toklet_session_history(s, ptr::null_mut(), 0, &mut count, ptr::null_mut()),
                BUSY
            );
        }
        1
    }
    let (m, s, c) = handles();
    let (code, r) = unsafe { generate(s, c, &[1, 7, 13, 3], 0, 3, Some(reenter), s.cast()) };
    assert_eq!(code, CALLBACK_ERROR);
    assert_eq!(
        (r.history_tokens, r.evaluated_tokens, r.pending_tokens),
        (5, 4, 1)
    );
    assert_eq!(
        unsafe { generate(s, c, &[], 2, 1, None, ptr::null_mut()).0 },
        OK
    );
    unsafe {
        cleanup(m, s, c);
    }
}
#[test]
fn cancellation_from_another_thread_stops_at_valid_pending_prefix() {
    unsafe extern "C" fn cancel(ctx: *mut c_void, _: TokenEvent) -> u32 {
        let address = ctx as usize;
        thread::spawn(move || unsafe {
            assert_eq!(
                toklet_cancel_set(address as *const CancelHandle, 1, ptr::null_mut()),
                OK
            );
        })
        .join()
        .unwrap();
        0
    }
    let (m, s, c) = handles();
    let (code, r) = unsafe { generate(s, c, &[1, 7, 13, 3], 0, 3, Some(cancel), c.cast()) };
    assert_eq!(code, OK);
    assert_eq!((r.reason, r.emitted_tokens, r.pending_tokens), (3, 1, 1));
    let (_, before) = unsafe { generate(s, c, &[], 2, 1, None, ptr::null_mut()) };
    assert_eq!(before.history_tokens, 5);
    unsafe {
        assert_eq!(toklet_cancel_set(c, 0, ptr::null_mut()), OK);
    }
    assert_eq!(
        unsafe { generate(s, c, &[], 2, 1, None, ptr::null_mut()).0 },
        OK
    );
    unsafe {
        cleanup(m, s, c);
    }
}
#[test]
fn hidden_stop_and_admission_errors_preserve_history() {
    let (m, s, c) = handles();
    let first =
        sampling::greedy(&fixtures::Fixture::tiny().f32("cases.prompt.logits")[3 * 64..]).unwrap();
    let mut r = Report::default();
    let input = [1, 7, 13, 3];
    let code = unsafe {
        toklet_session_generate(
            s,
            0,
            input.as_ptr(),
            4,
            3,
            &first,
            1,
            c,
            None,
            ptr::null_mut(),
            &mut r,
            ptr::null_mut(),
        )
    };
    assert_eq!(code, OK);
    assert_eq!(
        (r.reason, r.stop_token, r.emitted_tokens, r.pending_tokens),
        (2, first, 0, 1)
    );
    assert_eq!(
        unsafe { generate(s, c, &[64], 0, 1, None, ptr::null_mut()).0 },
        INVALID
    );
    assert_eq!(
        unsafe { generate(s, c, &[], 2, 32, None, ptr::null_mut()).0 },
        INVALID
    );
    let mut state = Report::default();
    unsafe {
        assert_eq!(toklet_session_state(s, &mut state, ptr::null_mut()), OK);
    }
    assert_eq!(state.history_tokens, 5);
    unsafe {
        cleanup(m, s, c);
    }
}
#[test]
fn invalid_abi_pointer_lengths_and_failed_load_do_not_publish_handles() {
    let (m, s, c) = handles();
    let mut out = ptr::dangling_mut::<SessionHandle>();
    let mut o = opts();
    o.abi_version += 1;
    unsafe {
        assert_eq!(
            toklet_session_create(m, &o, &mut out, ptr::null_mut()),
            INVALID
        );
    }
    assert!(out.is_null());
    let mut r = Report::default();
    unsafe {
        assert_eq!(
            toklet_session_generate(
                s,
                0,
                ptr::null(),
                usize::MAX,
                1,
                ptr::null(),
                0,
                c,
                None,
                ptr::null_mut(),
                &mut r,
                ptr::null_mut()
            ),
            INVALID
        );
        assert_eq!(
            toklet_session_generate(
                s,
                0,
                ptr::without_provenance(1),
                1,
                1,
                ptr::null(),
                0,
                c,
                None,
                ptr::null_mut(),
                &mut r,
                ptr::null_mut()
            ),
            INVALID
        );
        assert_eq!(
            toklet_session_state(ptr::null(), &mut r, ptr::null_mut()),
            INVALID
        );
        let mut loaded = ptr::dangling_mut::<ModelHandle>();
        assert_eq!(
            toklet_model_load(
                b"/missing-toklet-native-model".as_ptr(),
                27,
                &opts(),
                &mut loaded,
                ptr::null_mut()
            ),
            LOAD_ERROR
        );
        assert!(loaded.is_null());
        cleanup(m, s, c);
    }
}
#[test]
fn panic_is_contained_and_session_requires_recreation() {
    let (m, s, c) = handles();
    let result: Result<()> = unsafe { (*s).access(|_| panic!("injected session panic")) };
    assert!(matches!(result, Err(Failure(PANIC, _))));
    unsafe {
        assert_eq!(toklet_session_reset(s, ptr::null_mut()), POISONED);
        cleanup(m, s, c);
    }
    assert_eq!(
        unsafe { boundary(ptr::null_mut(), || panic!("injected boundary panic")) },
        PANIC
    );
}
#[test]
fn wrong_thread_calls_reject_without_dropping_the_handle() {
    let (m, s, c) = handles();
    let address = m as usize;
    let code = thread::spawn(move || unsafe {
        toklet_model_release(address as *mut ModelHandle, ptr::null_mut())
    })
    .join()
    .unwrap();
    assert_eq!(code, WRONG_THREAD);
    let address = s as usize;
    let code = thread::spawn(move || unsafe {
        let mut count = usize::MAX;
        let code = toklet_session_history(
            address as *const SessionHandle,
            ptr::null_mut(),
            0,
            &mut count,
            ptr::null_mut(),
        );
        assert_eq!(count, usize::MAX);
        code
    })
    .join()
    .unwrap();
    assert_eq!(code, WRONG_THREAD);
    unsafe {
        cleanup(m, s, c);
    }
}

#[test]
fn later_callback_failure_reports_already_delivered_events_and_can_continue() {
    unsafe extern "C" fn third(ctx: *mut c_void, _: TokenEvent) -> u32 {
        let count = unsafe { &mut *ctx.cast::<u32>() };
        *count += 1;
        u32::from(*count == 3)
    }
    let (m, s, c) = handles();
    let mut attempted = 0_u32;
    let (code, r) = unsafe {
        generate(
            s,
            c,
            &[1, 7, 13, 3],
            0,
            4,
            Some(third),
            (&mut attempted as *mut u32).cast(),
        )
    };
    assert_eq!(code, CALLBACK_ERROR);
    assert_eq!(
        (
            attempted,
            r.emitted_tokens,
            r.history_tokens,
            r.evaluated_tokens,
            r.pending_tokens
        ),
        (3, 2, 7, 6, 1)
    );
    assert_eq!(
        unsafe { generate(s, c, &[], 2, 1, None, ptr::null_mut()).0 },
        OK
    );
    unsafe {
        cleanup(m, s, c);
    }
}

#[test]
fn bridge_reservation_covers_owned_handle_and_wrapper_allocations() {
    let mut model_bytes = size_of::<model::Model>();
    let mut session_bytes = size_of::<session::Session<'static>>();
    #[cfg(all(
        feature = "metal",
        any(target_os = "macos", target_os = "ios"),
        target_arch = "aarch64"
    ))]
    {
        model_bytes = model_bytes.max(size_of::<crate::metal::Model>());
        session_bytes = session_bytes.max(size_of::<crate::metal::Session<'static>>());
    }
    let required = size_of::<ModelHandle>()
        + size_of::<NativeModel>()
        + 2 * size_of::<usize>()
        + size_of::<SessionHandle>()
        + size_of::<CancelHandle>()
        + model_bytes
        + session_bytes;
    assert!(
        required <= BRIDGE_BYTES,
        "bridge allocations {required} exceed reserved {BRIDGE_BYTES}"
    );
    // Keep both feature configurations warning-free without hiding layout checks.
    let _ = (&mut model_bytes, &mut session_bytes);
}
#[cfg(not(all(
    feature = "metal",
    any(target_os = "macos", target_os = "ios"),
    target_arch = "aarch64"
)))]
#[test]
fn unavailable_metal_returns_explicit_error_without_cpu_fallback() {
    let mut m = ptr::dangling_mut::<ModelHandle>();
    let path = b"missing";
    let code = unsafe {
        toklet_model_load_metal(
            path.as_ptr(),
            path.len(),
            ptr::null(),
            0,
            &opts(),
            &mut m,
            ptr::null_mut(),
        )
    };
    assert_eq!(code, UNSUPPORTED);
    assert!(m.is_null());
}
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "explicit Apple GPU and pinned selected local model/artifact check"]
fn metal_native_selected_models_match_direct_sessions() {
    use crate::metal::{self, AttentionMode, NormalizationMode, ProjectionMode, ReadbackMode};
    for (name, packed) in [
        ("Qwen2.5-0.5B-Instruct", None),
        ("Qwen2.5-1.5B-Instruct", Some("artifacts/m5-larger-q4-v2")),
    ] {
        let path = format!("{}/models/{name}", env!("CARGO_MANIFEST_DIR"));
        let o = Options {
            capacity: 16,
            chunk: 3,
            memory_limit_bytes: weights::INITIAL_MEMORY_LIMIT_BYTES as u64,
            ..opts()
        };
        let mut m = ptr::null_mut();
        let mut s = ptr::null_mut();
        let mut c = ptr::null_mut();
        let mut error = NativeError {
            code: 0,
            message: [0; 256],
        };
        let packed = packed.map(str::as_bytes).unwrap_or(&[]);
        unsafe {
            let code = toklet_model_load_metal(
                path.as_ptr(),
                path.len(),
                packed.as_ptr(),
                packed.len(),
                &o,
                &mut m,
                &mut error,
            );
            assert_eq!(code, OK, "{}", String::from_utf8_lossy(&error.message));
            assert_eq!(toklet_session_create(m, &o, &mut s, &mut error), OK);
            assert_eq!(toklet_cancel_create(&mut c, &mut error), OK);
        }
        let NativeModel::Metal(model) = (unsafe { (*m).model.as_ref() }) else {
            panic!("Metal request silently changed backend")
        };
        let (p, n) = if packed.is_empty() {
            (ProjectionMode::HalfCompensated, NormalizationMode::Original)
        } else {
            (
                ProjectionMode::Q4CompensatedTiled,
                NormalizationMode::Compensated,
            )
        };
        let mut direct = metal::Session::with_attention_mode(
            model,
            unsafe { options(&o).unwrap() },
            p,
            n,
            ReadbackMode::FullLogits,
            AttentionMode::Online,
        )
        .unwrap();
        let mut native_memory = Report::default();
        unsafe {
            assert_eq!(toklet_session_state(s, &mut native_memory, &mut error), OK);
        }
        let combined = native_memory.admitted_bytes + direct.memory().admitted_bytes as u64
            - direct.memory().weights_bytes as u64;
        assert!(combined <= weights::INITIAL_MEMORY_LIMIT_BYTES as u64);
        let prompt = [9707, 11, 358, 498];
        let mut expected = Vec::new();
        let expected_report = generation::generate(
            &mut direct,
            generation::Request {
                input: generation::Input::Prompt(&prompt),
                max_new_tokens: 3,
                stop_tokens: &[],
            },
            &AtomicBool::new(false),
            |e| {
                expected.push(e.token_id);
                Ok(())
            },
        )
        .unwrap();
        let mut ids = Vec::<u32>::new();
        let (code, r) = unsafe {
            generate(
                s,
                c,
                &prompt,
                0,
                3,
                Some(collect),
                (&mut ids as *mut Vec<u32>).cast(),
            )
        };
        assert_eq!(code, OK);
        assert_eq!(ids, expected);
        assert_eq!(r.history_tokens as usize, expected_report.history_tokens);
        assert_eq!(
            (r.history_tokens, r.evaluated_tokens, r.pending_tokens),
            (7, 6, 1)
        );
        assert_eq!(unsafe { toklet_session_reset(s, ptr::null_mut()) }, OK);
        drop(direct);
        unsafe {
            cleanup(m, s, c);
        }
        println!(
            "native_metal_model={name} ids={ids:?} state=7/6/1 combined_admission_bytes={combined} passed"
        );
    }
}

#[test]
fn history_copy_includes_pending_and_replays_without_guessing() {
    let (m, s, c) = handles();
    unsafe {
        let (code, original) = generate(s, c, &[1, 7, 13, 3], 0, 2, None, ptr::null_mut());
        assert_eq!(code, OK);
        assert_eq!(original.pending_tokens, 1);
        let mut count = usize::MAX;
        assert_eq!(
            toklet_session_history(s, ptr::null_mut(), 0, &mut count, ptr::null_mut()),
            OK
        );
        assert_eq!(count, 6);
        let mut too_small = [u32::MAX; 5];
        assert_eq!(
            toklet_session_history(s, too_small.as_mut_ptr(), 5, &mut count, ptr::null_mut()),
            INVALID
        );
        assert_eq!(count, 6);
        assert_eq!(too_small, [u32::MAX; 5]);
        let mut history = [u32::MAX; 32];
        assert_eq!(
            toklet_session_history(s, history.as_mut_ptr(), 32, &mut count, ptr::null_mut()),
            OK
        );
        assert_eq!(&history[..count], &[1, 7, 13, 3, 47, 47]);
        assert_eq!(history[count], u32::MAX);
        // A precancelled replacement prompt must leave the old exact IDs intact.
        assert_eq!(toklet_cancel_set(c, 1, ptr::null_mut()), OK);
        assert_eq!(
            generate(s, c, &[3, 13, 7, 1], 0, 2, None, ptr::null_mut()).0,
            OK
        );
        assert_eq!(
            toklet_session_history(s, history.as_mut_ptr(), 32, &mut count, ptr::null_mut()),
            OK
        );
        assert_eq!(&history[..count], &[1, 7, 13, 3, 47, 47]);
        assert_eq!(toklet_cancel_set(c, 0, ptr::null_mut()), OK);
        assert_eq!(toklet_session_release(s, ptr::null_mut()), OK);
        let mut replay = ptr::null_mut();
        assert_eq!(
            toklet_session_create(m, &opts(), &mut replay, ptr::null_mut()),
            OK
        );
        // Replaying the full sampled history eagerly commits pending IDs. The
        // exact token prefix stays unchanged and continuation has the same oracle.
        let (code, report) = generate(replay, c, &history[..count], 0, 0, None, ptr::null_mut());
        assert_eq!(code, OK);
        assert_eq!(
            (
                report.history_tokens,
                report.evaluated_tokens,
                report.pending_tokens
            ),
            (6, 6, 0)
        );
        let mut ids = Vec::new();
        assert_eq!(
            generate(
                replay,
                c,
                &[],
                2,
                2,
                Some(collect),
                (&mut ids as *mut Vec<u32>).cast()
            )
            .0,
            OK
        );
        assert_eq!(ids, [47_u32, 47]);
        assert_eq!(
            toklet_session_history(replay, ptr::null_mut(), 1, &mut count, ptr::null_mut()),
            INVALID
        );
        assert_eq!(
            toklet_session_history(
                replay,
                history.as_mut_ptr(),
                32769,
                &mut count,
                ptr::null_mut()
            ),
            INVALID
        );
        assert_eq!(
            toklet_session_history(
                replay,
                history.as_mut_ptr(),
                32,
                ptr::null_mut(),
                ptr::null_mut()
            ),
            INVALID
        );
        cleanup(m, replay, c);
    }
}
