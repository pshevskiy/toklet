//! Full request boundary diagnostic overhead, with constructor/export separate.
use crate::{
    app_chat::gemma4,
    generation::{self, Input, Request, StopReason},
    metal::gemma_affine::{Bf16CaptureState, Bf16ProfileOptions, OfficialBf16},
    session::Options,
};
use std::{path::PathBuf, sync::atomic::AtomicBool, time::Instant};
const LIMIT: usize = crate::weights::INITIAL_MEMORY_LIMIT_BYTES;
const CALLER: usize = 16 * 1024 * 1024;
fn thermal() -> isize {
    objc2_foundation::NSProcessInfo::processInfo()
        .thermalState()
        .0
}
#[test]
#[ignore = "Explicit two-cohort plain/profile paired official BF16/G2 console measurement"]
fn official_bf16_profile_paired_request_overhead() {
    let directory = PathBuf::from(std::env::var_os("TOKLET_GEMMA_OFFICIAL_DIRECTORY").unwrap());
    let cohort: usize = std::env::var("TOKLET_GEMMA_PROFILE_COHORT")
        .unwrap()
        .parse()
        .unwrap();
    assert!(matches!(cohort, 1 | 2));
    let owner = OfficialBf16::load(&directory, 512, LIMIT - CALLER).unwrap();
    let model = owner.model().unwrap();
    let options = Options {
        capacity: 512,
        prefill_chunk_size: 32,
        position_offset: 0,
        memory_limit_bytes: LIMIT - CALLER,
    };
    let prompt = owner
        .codec()
        .encode_chat(
            br#"[{"role":"user","content":"Name the capital of France."}]"#,
            gemma4::RenderOptions {
                enable_thinking: false,
                add_generation_prompt: true,
            },
        )
        .unwrap();
    assert_eq!(
        prompt,
        [
            2, 105, 2364, 107, 1567, 506, 5279, 529, 7001, 236761, 106, 107, 105, 4368, 107
        ]
    );
    let mut samples = Vec::with_capacity(26);
    for pair in 0..13 {
        let order = if (pair + cohort) % 2 == 1 {
            [false, true]
        } else {
            [true, false]
        };
        for (within_pair, enabled) in order.into_iter().enumerate() {
            let before = thermal();
            assert_eq!(before, 0, "nominal thermal state required");
            let mut request_id = [0u8; 16];
            request_id[0] = cohort as u8;
            request_id[1] = pair as u8;
            request_id[2] = u8::from(enabled);
            let clock = Instant::now();
            let mut session = if enabled {
                owner
                    .new_profiled_session(
                        &model,
                        options,
                        Bf16ProfileOptions {
                            request_id,
                            evaluation_ordinal: 2,
                        },
                    )
                    .unwrap()
            } else {
                owner.new_session(&model, options).unwrap()
            };
            session.use_last_row_prefill();
            let setup_seconds = clock.elapsed().as_secs_f64();
            assert!(
                session.memory().admitted_bytes + owner.memory().session_reserved_bytes + CALLER
                    <= LIMIT
            );
            let mut ids = Vec::with_capacity(12);
            let clock = Instant::now();
            let report = generation::generate(
                &mut session,
                Request {
                    input: Input::Reset(&prompt),
                    max_new_tokens: 12,
                    stop_tokens: owner.codec().stop_ids(),
                },
                &AtomicBool::new(false),
                |event| {
                    ids.push(event.token_id);
                    Ok(())
                },
            )
            .unwrap();
            let caller_seconds = clock.elapsed().as_secs_f64();
            let after = thermal();
            assert_eq!(after, 0, "nominal thermal state required");
            assert_eq!(ids, [818, 5279, 529, 7001, 563, 5213, 50429, 84750]);
            assert_eq!(report.reason, StopReason::StopToken { token_id: 106 });
            assert_eq!(
                (
                    report.reused_tokens,
                    report.prefill_tokens,
                    report.sampled_tokens,
                    report.emitted_tokens
                ),
                (0, 15, 9, 8)
            );
            assert_eq!(
                (
                    session.history().len(),
                    session.evaluated_len(),
                    session.pending_len()
                ),
                (24, 23, 1)
            );
            let text = owner.codec().decode(&ids, true).unwrap();
            assert_eq!(text, "The capital of France is **Paris**.");
            // The sampled stop remains pending, so the evaluated distribution
            // must not be exposed as if it included that token. Full bitwise
            // prefill/decode parity is checked in the dedicated profile control.
            assert!(session.logits().is_none());
            let clock = Instant::now();
            let capture = if enabled {
                let frame = session.bf16_profile().unwrap();
                assert_eq!(frame.request_id, request_id);
                assert_eq!(frame.evaluation_ordinal, 2);
                assert_eq!(frame.status.state, Bf16CaptureState::Complete);
                assert_eq!(
                    (
                        frame.tiles.len(),
                        frame.clocks.events.len(),
                        frame.status.gathers_recorded
                    ),
                    (349, 349, 2)
                );
                assert_eq!(
                    (
                        frame.evaluation.position,
                        frame.evaluation.committed_positions
                    ),
                    (15, 16)
                );
                Some(
                    serde_json::json!({"tiles":frame.tiles.len(),"gathers":frame.status.gathers_recorded,"frame_wall_seconds":frame.frame_wall_seconds}),
                )
            } else {
                assert!(session.bf16_profile_status().is_none() && session.bf16_profile().is_err());
                None
            };
            let export_seconds = clock.elapsed().as_secs_f64();
            samples.push(serde_json::json!({"pair":pair,"within_pair":within_pair,"warmup":pair<3,
                "profile_enabled":enabled,"request_id":request_id,"setup_seconds":setup_seconds,
                "caller_seconds":caller_seconds,"report":report,"export_seconds":export_seconds,
                "capture":capture,"emitted_ids":ids,"text":text,"thermal_before":before,"thermal_after":after}));
        }
    }
    let mut footprint = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
    // SAFETY: flavor4 matches the owned zeroed writable Apple structure; the
    // successful FFI copy initializes fields without aliases during that call.
    assert_eq!(
        unsafe {
            libc::proc_pid_rusage(
                libc::getpid(),
                libc::RUSAGE_INFO_V4,
                footprint.as_mut_ptr().cast(),
            )
        },
        0
    );
    // SAFETY: successful flavor4 call populated the zeroed structure above.
    let footprint = unsafe { footprint.assume_init() };
    assert!(
        footprint.ri_phys_footprint > 0
            && footprint.ri_phys_footprint <= footprint.ri_lifetime_max_phys_footprint
            && footprint.ri_lifetime_max_phys_footprint <= LIMIT as u64
    );
    println!(
        "BF16_PROFILE_OVERHEAD {}",
        serde_json::json!({"passed":true,"cohort":cohort,"warm_pairs":3,"measured_pairs":10,
        "samples":samples,"pending_logits_unavailable":true,"caller_allowance_bytes":CALLER,
        "physical_bytes":footprint.ri_phys_footprint,"process_lifetime_peak_bytes":footprint.ri_lifetime_max_phys_footprint,
        "scope":"Diagnostic capture overhead only. Constructor/export separate; no optimization or summary overhead qualification."})
    );
}
