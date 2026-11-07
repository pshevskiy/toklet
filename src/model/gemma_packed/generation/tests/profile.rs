//! Explicit GPU correctness/ownership witness. Download activity and these
//! diagnostic clocks do not qualify performance or instrumentation overhead.
use crate::{
    app_chat::gemma4,
    generation::{self, Input, Request, StopReason},
    metal::gemma_affine::{
        Bf16CaptureState, Bf16FrameProfile, Bf16GraphFault, Bf16ProfileOptions, OfficialBf16,
        RuntimeInventory,
    },
    session::Options,
};
use std::{collections::BTreeMap, path::PathBuf, sync::atomic::AtomicBool};
const LIMIT: usize = crate::weights::INITIAL_MEMORY_LIMIT_BYTES;
const CALLER: usize = 16 * 1024 * 1024;
fn bits(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "logit bit mismatch at {i}");
    }
}
fn validate(frame: &Bf16FrameProfile, inventory: &RuntimeInventory, ordinal: usize) {
    assert_eq!(frame.evaluation_ordinal, ordinal);
    assert_eq!(frame.request_id, [ordinal as u8; 16]);
    assert_eq!(frame.status.state, Bf16CaptureState::Complete);
    assert_eq!(
        (
            frame.tiles.len(),
            frame.clocks.events.len(),
            frame.status.tiles_recorded
        ),
        (349, 349, 349)
    );
    assert_eq!(frame.status.gathers_recorded, 2);
    assert!(!frame.status.overflowed);
    assert_eq!(frame.diagnostic_allowance_bytes, 64 * 1024 * 1024);
    let (position, rows, committed) = if ordinal == 1 {
        (0, 15, 15)
    } else {
        (15, 1, 16)
    };
    assert_eq!(
        (
            frame.evaluation.position,
            frame.evaluation.input_rows,
            frame.evaluation.output_rows,
            frame.evaluation.committed_positions
        ),
        (position, rows, 1, committed)
    );
    assert!(
        frame.frame_wall_seconds > 0.
            && frame.exclusive_host_seconds >= 0.
            && frame.unattributed_host_seconds >= 0.
    );
    assert!(
        (frame.frame_wall_seconds
            - frame.exclusive_host_seconds
            - frame.clock_bracket_seconds
            - frame.unattributed_host_seconds)
            .abs()
            <= 1e-6
    );
    let mut counts = BTreeMap::new();
    for (slot, tile) in frame.tiles.iter().enumerate() {
        let record = &tile.record;
        let a = record.attribution;
        assert_eq!(record.counter_slot, slot);
        let selected = inventory
            .operations
            .iter()
            .find(|p| p.logical.id == a.operation)
            .expect("cold role exists");
        let expected_rows = if ordinal == 2 || selected.logical.prefill_rows == 1 {
            1
        } else {
            15
        };
        assert_eq!(
            (a.source_matrix_slot, a.input_features, a.output_features),
            (
                selected.logical.source_matrix_slot,
                selected.logical.source_columns,
                selected.logical.source_rows
            )
        );
        assert_eq!(a.input_rows, expected_rows);
        assert_eq!(a.source_first_row, a.tile_index * 8192);
        assert_eq!(
            a.source_rows,
            (a.output_features - a.source_first_row).min(8192)
        );
        assert_eq!(
            (
                record.dispatches_encoded,
                record.commands_submitted,
                record.host_wait_calls
            ),
            (1, 1, 1)
        );
        assert_eq!(record.input_copy_bytes, a.input_rows * a.input_features * 4);
        assert_eq!(record.output_copy_bytes, a.input_rows * a.source_rows * 4);
        assert!(
            tile.event.gpu_end_ticks >= tile.event.gpu_begin_ticks
                && tile.event.gpu_begin_ticks > 0
        );
        *counts.entry(a.operation).or_insert(0usize) += 1;
    }
    for operation in &inventory.operations {
        if operation.projection_tiles > 0 {
            assert_eq!(counts[&operation.logical.id], operation.projection_tiles);
        }
    }
    for gather in frame.cpu_gathers {
        let selected = inventory
            .operations
            .iter()
            .find(|p| p.logical.id == gather.operation)
            .unwrap();
        assert_eq!(
            (gather.source_matrix_slot, gather.rows, gather.columns),
            (
                selected.logical.source_matrix_slot,
                rows,
                selected.logical.source_columns
            )
        );
        assert_eq!(gather.active_source_bytes, rows * gather.columns * 2);
        assert!(gather.host_end_seconds >= gather.host_begin_seconds);
    }
}
#[test]
#[ignore = "Explicit official BF16 calibrated profile/logit/state/fault/overflow Mac check"]
fn official_bf16_profile_matches_plain_frames_and_rejects_invalid_captures() {
    let directory = PathBuf::from(
        std::env::var_os("TOKLET_GEMMA_OFFICIAL_DIRECTORY").expect("pinned offline source"),
    );
    let output = PathBuf::from(
        std::env::var_os("TOKLET_GEMMA_PROFILE_OUTPUT").expect("fresh output directory"),
    );
    assert!(output.is_dir());
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
    assert_eq!(prompt.len(), 15);
    let inventory = owner.runtime_inventory(32).unwrap();
    let (prefill, decode, plain_memory) = {
        let mut plain = owner.new_session(&model, options).unwrap();
        plain.use_last_row_prefill();
        assert!(plain.bf16_profile_status().is_none() && plain.bf16_profile().is_err());
        plain.append(&prompt).unwrap();
        assert_eq!(plain.evaluate_next_chunk().unwrap(), 15);
        let prefill = plain.logits().unwrap().to_vec();
        plain.append(&[818]).unwrap();
        assert_eq!(plain.evaluate_next_chunk().unwrap(), 1);
        (prefill, plain.logits().unwrap().to_vec(), plain.memory())
    };
    for ordinal in [1, 2] {
        let mut session = owner
            .new_profiled_session(
                &model,
                options,
                Bf16ProfileOptions {
                    request_id: [ordinal as u8; 16],
                    evaluation_ordinal: ordinal,
                },
            )
            .unwrap();
        assert_eq!(
            session.memory().admitted_bytes,
            plain_memory.admitted_bytes + 64 * 1024 * 1024
        );
        assert!(
            session.memory().admitted_bytes + owner.memory().session_reserved_bytes + CALLER
                <= LIMIT
        );
        assert!(session.bf16_profile().is_err());
        session.append(&prompt).unwrap();
        assert_eq!(session.evaluate_next_chunk().unwrap(), 15);
        bits(session.logits().unwrap(), &prefill);
        if ordinal == 2 {
            assert!(session.bf16_profile().is_err());
        }
        session.append(&[818]).unwrap();
        assert_eq!(session.evaluate_next_chunk().unwrap(), 1);
        bits(session.logits().unwrap(), &decode);
        let frame = session.bf16_profile().unwrap();
        validate(&frame, &inventory, ordinal);
        let bytes = serde_json::to_vec_pretty(&frame).unwrap();
        assert!(bytes.len() <= 1024 * 1024);
        std::fs::write(output.join(format!("frame-{ordinal}.json")), bytes).unwrap();
        session.reset();
        assert_eq!(
            session.bf16_profile_status().unwrap().state,
            Bf16CaptureState::Invalid
        );
        assert!(session.bf16_profile().is_err());
    }
    for fault in [Bf16GraphFault::AfterPle, Bf16GraphFault::AfterFirstHeadTile] {
        let mut session = owner
            .new_profiled_session(
                &model,
                options,
                Bf16ProfileOptions {
                    request_id: [3; 16],
                    evaluation_ordinal: 1,
                },
            )
            .unwrap();
        session.inner.operations.set_fault(fault);
        session.append(&prompt).unwrap();
        assert!(session.evaluate_next_chunk().is_err());
        assert_eq!((session.evaluated_len(), session.pending_len()), (0, 15));
        assert!(session.inner.caches.iter().all(|c| c.offset() == 0));
        assert!(session.logits().is_none() && session.bf16_profile().is_err());
        assert_eq!(
            session.bf16_profile_status().unwrap().state,
            Bf16CaptureState::Invalid
        );
    }
    {
        let mut session = owner
            .new_profiled_session(
                &model,
                options,
                Bf16ProfileOptions {
                    request_id: [4; 16],
                    evaluation_ordinal: 1,
                },
            )
            .unwrap();
        session.inner.operations.limit_profile_events_for_test(1);
        session.append(&prompt).unwrap();
        assert!(session.evaluate_next_chunk().is_err());
        let state = session.bf16_profile_status().unwrap();
        assert!(state.overflowed);
        assert_eq!(state.tiles_recorded, 1);
        assert_eq!((session.evaluated_len(), session.pending_len()), (0, 15));
        assert!(session.bf16_profile().is_err());
    }
    {
        let mut session = owner
            .new_profiled_session(
                &model,
                options,
                Bf16ProfileOptions {
                    request_id: [5; 16],
                    evaluation_ordinal: 1,
                },
            )
            .unwrap();
        let cancelled = generation::generate(
            &mut session,
            Request {
                input: Input::Reset(&prompt),
                max_new_tokens: 12,
                stop_tokens: owner.codec().stop_ids(),
            },
            &AtomicBool::new(true),
            |_| panic!("cancelled request emitted"),
        )
        .unwrap();
        assert_eq!(cancelled.reason, StopReason::Cancelled);
        assert_eq!(session.bf16_profile_status().unwrap().evaluations_seen, 0);
        assert!(session.history().is_empty() && session.bf16_profile().is_err());
        session.append(&prompt).unwrap();
        session.evaluate_next_chunk().unwrap();
        assert!(session.bf16_profile().is_ok());
        session
            .inner
            .operations
            .set_fault(Bf16GraphFault::AfterFirstHeadTile);
        session.append(&[818]).unwrap();
        assert!(session.evaluate_next_chunk().is_err());
        assert_eq!((session.evaluated_len(), session.pending_len()), (15, 1));
        assert!(session.bf16_profile().is_err());
    }
    let mut footprint = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
    // SAFETY: flavor4 matches the owned zeroed writable libc/Apple structure;
    // status0 initializes its fields, with no alias during the FFI copy.
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
        "BF16_PROFILE_CORRECTNESS {}",
        serde_json::json!({"passed":true,"frames":2,"tiles_per_frame":349,"cpu_gathers_per_frame":2,
        "logits":"bitwise plain/profile prefill and first decode","faults":"latePLE/head,overflow,postcapturefailure","reset_invalidation":true,
        "prestart_cancel_unavailable":true,"caller_allowance_bytes":CALLER,"timing_qualified":false,
        "physical_bytes":footprint.ri_phys_footprint,"process_lifetime_peak_bytes":footprint.ri_lifetime_max_phys_footprint})
    );
}
