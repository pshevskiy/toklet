//! Actual-model readiness controls; original full numerical/quality gates remain separate.
use crate::{
    app_chat::gemma4,
    generation::{self, Input, Request, StopReason},
    metal::gemma_affine::{
        Bf16CaptureState, Bf16FrameProfile, Bf16GenerationSession, Bf16GraphFault,
        Bf16ProfileOptions, OfficialBf16,
    },
    model::{gemma_packed::tests::pointers, gemma_reference::tests::Measurement},
    session::{self, Options},
};
use std::{
    collections::BTreeSet,
    mem::size_of,
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
};

const LIMIT: usize = crate::weights::INITIAL_MEMORY_LIMIT_BYTES;
const CALLER: usize = 16 * 1024 * 1024;
const HELPER: usize = 32 * 1024 * 1024;
const OWNER_LIMIT: usize = LIMIT - CALLER - HELPER;
const AUXILIARY: usize = 8 * 1024 * 1024;
const VOCAB: usize = 262144;

fn load() -> OfficialBf16 {
    let directory = PathBuf::from(
        std::env::var_os("TOKLET_GEMMA_OFFICIAL_DIRECTORY").expect("authenticated offline source"),
    );
    let owner = OfficialBf16::load(&directory, 512, OWNER_LIMIT).unwrap();
    assert!(owner.memory().load_admitted_bytes + CALLER + HELPER <= LIMIT);
    owner
}

fn options(chunk: usize) -> Options {
    Options {
        capacity: 512,
        prefill_chunk_size: chunk,
        position_offset: 0,
        memory_limit_bytes: OWNER_LIMIT,
    }
}

fn admitted(owner: &OfficialBf16, session: &Bf16GenerationSession<'_, '_, '_>) {
    // Numerical memory already includes the shared weights. Count the codec/
    // owner reserve once, alongside the fixed caller and external helper bounds.
    assert!(
        session.memory().admitted_bytes + owner.memory().session_reserved_bytes + CALLER + HELPER
            <= LIMIT
    );
}

fn prompt(owner: &OfficialBf16) -> Vec<u32> {
    let ids = owner
        .codec()
        .encode_chat(
            br#"[{"role":"user","content":"Name the capital of France."}]"#,
            gemma4::RenderOptions {
                enable_thinking: false,
                add_generation_prompt: true,
            },
        )
        .unwrap();
    assert_eq!(ids.len(), 15);
    ids
}

fn bitwise(actual: &[f32], expected: &[u32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual.to_bits(), *expected, "FP32 bit mismatch at {index}");
    }
}

struct CachePrefix {
    offset: usize,
    generation: u64,
    keys: Vec<u32>,
    values: Vec<u32>,
}

struct Checkpoint {
    history: Vec<u32>,
    committed: Vec<u32>,
    logits: Option<Vec<u32>>,
    caches: Vec<CachePrefix>,
}

impl Checkpoint {
    fn capture(session: &Bf16GenerationSession<'_, '_, '_>) -> Self {
        assert!(session.evaluated_len() <= 33);
        assert_eq!(session.inner.caches.len(), 15);
        let caches = session
            .inner
            .caches
            .iter()
            .map(|cache| {
                // This graph has one KV head. Before local eviction these
                // segments contain just the valid prefix, never512-slot tails.
                let segments = cache.committed_segments();
                assert!(cache.offset() <= 33);
                assert!(segments.iter().map(|s| s.k.len()).sum::<usize>() <= 33 * 512);
                assert!(segments.iter().all(|s| s.k.len() == s.v.len()));
                assert_eq!(
                    segments.iter().map(|s| s.keys).sum::<usize>(),
                    cache.offset()
                );
                let collect = |keys: bool| {
                    segments
                        .iter()
                        .flat_map(|s| if keys { s.k } else { s.v })
                        .map(|value| value.to_bits())
                        .collect()
                };
                CachePrefix {
                    offset: cache.offset(),
                    generation: cache.generation(),
                    keys: collect(true),
                    values: collect(false),
                }
            })
            .collect();
        let logits = session.logits().map(|row| {
            assert_eq!(row.len(), VOCAB);
            row.iter().map(|value| value.to_bits()).collect()
        });
        let checkpoint = Self {
            history: session.history().to_vec(),
            committed: session.inner.tokens().to_vec(),
            logits,
            caches,
        };
        assert!(checkpoint.owned_bytes() + AUXILIARY <= CALLER);
        checkpoint
    }

    fn owned_bytes(&self) -> usize {
        size_of::<Self>()
            + (self.history.capacity() + self.committed.capacity()) * 4
            + self.logits.as_ref().map_or(0, |row| row.capacity() * 4)
            + self.caches.capacity() * size_of::<CachePrefix>()
            + self
                .caches
                .iter()
                .map(|c| (c.keys.capacity() + c.values.capacity()) * 4)
                .sum::<usize>()
    }

    fn assert_kv(&self, session: &Bf16GenerationSession<'_, '_, '_>) {
        assert_eq!(session.inner.tokens(), self.committed);
        assert_eq!(session.inner.caches.len(), self.caches.len());
        for (cache, expected) in session.inner.caches.iter().zip(&self.caches) {
            assert_eq!(
                (cache.offset(), cache.generation()),
                (expected.offset, expected.generation)
            );
            let segments = cache.committed_segments();
            assert!(
                segments
                    .iter()
                    .flat_map(|s| s.k)
                    .map(|v| v.to_bits())
                    .eq(expected.keys.iter().copied())
            );
            assert!(
                segments
                    .iter()
                    .flat_map(|s| s.v)
                    .map(|v| v.to_bits())
                    .eq(expected.values.iter().copied())
            );
        }
    }

    fn assert_matches(&self, session: &Bf16GenerationSession<'_, '_, '_>) {
        assert_eq!(session.history(), self.history);
        self.assert_kv(session);
        match (&self.logits, session.logits()) {
            (Some(expected), Some(actual)) => bitwise(actual, expected),
            (None, None) => {}
            _ => panic!("distribution availability changed"),
        }
    }
}

fn profile(ordinal: usize) -> Bf16ProfileOptions {
    Bf16ProfileOptions {
        request_id: [ordinal as u8; 16],
        evaluation_ordinal: ordinal,
    }
}

fn validate_frame(frame: &Bf16FrameProfile, chunk: usize, groups: usize, commands: usize) {
    assert_eq!(frame.schema_version, 2);
    assert_eq!(frame.status.state, Bf16CaptureState::Complete);
    assert!(!frame.status.overflowed);
    assert_eq!(
        (
            frame.tiles.len(),
            frame.clocks.events.len(),
            frame.status.tiles_recorded
        ),
        (349, 349, 349)
    );
    assert_eq!(
        (frame.cpu_gathers.len(), frame.status.gathers_recorded),
        (2, 2)
    );
    assert_eq!(frame.diagnostic_allowance_bytes, 64 * 1024 * 1024);
    assert!(
        (frame.frame_wall_seconds
            - frame.exclusive_host_seconds
            - frame.clock_bracket_seconds
            - frame.unattributed_host_seconds)
            .abs()
            <= 1e-6
    );
    let submissions = frame
        .submission_groups
        .as_ref()
        .expect("explicit candidate groups");
    assert_eq!(submissions.len(), groups);
    let mut member = vec![false; frame.tiles.len()];
    for (id, group) in submissions.iter().enumerate() {
        assert_eq!(
            (group.group_id, group.owner_counter_slot),
            (id, group.first_counter_slot)
        );
        assert_eq!(group.input_rows, 1);
        assert!(group.tile_count > 1 && group.output_features <= chunk * 8192);
        assert_eq!(group.tile_count, group.output_features.div_ceil(8192));
        for relative in 0..group.tile_count {
            let slot = group.first_counter_slot + relative;
            assert!(!member[slot], "overlapping submission groups");
            member[slot] = true;
            let record = &frame.tiles[slot].record;
            let a = record.attribution;
            assert_eq!(
                (
                    a.operation,
                    a.source_matrix_slot,
                    a.input_rows,
                    a.input_features,
                    a.output_features
                ),
                (
                    group.operation,
                    group.source_matrix_slot,
                    1,
                    group.input_features,
                    group.output_features
                )
            );
            assert_eq!(
                (a.tile_index, a.source_first_row),
                (relative, relative * 8192)
            );
            if relative == 0 {
                assert_eq!((record.commands_submitted, record.host_wait_calls), (1, 1));
                assert_eq!(
                    (record.input_copy_bytes, record.output_copy_bytes),
                    (group.input_features * 4, group.output_features * 4)
                );
            } else {
                assert_eq!(
                    (
                        record.commands_submitted,
                        record.host_wait_calls,
                        record.input_copy_bytes,
                        record.output_copy_bytes
                    ),
                    (0, 0, 0, 0)
                );
                assert_eq!(
                    [
                        record.host_staging_seconds,
                        record.host_commit_seconds,
                        record.host_wait_seconds,
                        record.host_validation_and_copy_seconds
                    ],
                    [0.0; 4]
                );
            }
        }
    }
    for (slot, tile) in frame.tiles.iter().enumerate() {
        let r = &tile.record;
        let a = r.attribution;
        assert_eq!((r.counter_slot, r.dispatches_encoded), (slot, 1));
        assert!(
            tile.event.gpu_begin_ticks > 0
                && tile.event.gpu_end_ticks >= tile.event.gpu_begin_ticks
        );
        let eligible =
            a.input_rows == 1 && a.output_features > 8192 && a.output_features <= chunk * 8192;
        assert_eq!(
            member[slot], eligible,
            "missing/extra eligible group at {slot}"
        );
        if !member[slot] {
            assert_eq!((r.commands_submitted, r.host_wait_calls), (1, 1));
            assert_eq!(
                (r.input_copy_bytes, r.output_copy_bytes),
                (
                    a.input_rows * a.input_features * 4,
                    a.input_rows * a.source_rows * 4
                )
            );
        }
    }
    assert_eq!(
        frame
            .tiles
            .iter()
            .map(|t| t.record.commands_submitted)
            .sum::<usize>(),
        commands
    );
    assert_eq!(
        frame
            .tiles
            .iter()
            .map(|t| t.record.host_wait_calls)
            .sum::<usize>(),
        commands
    );
}

fn assert_legacy_wire(frame: &Bf16FrameProfile) {
    assert_eq!(frame.schema_version, 1);
    assert!(frame.submission_groups.is_none());
    let wire = serde_json::to_value(frame).unwrap();
    let keys: BTreeSet<_> = wire
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        keys,
        BTreeSet::from([
            "schema_version",
            "request_id",
            "evaluation_ordinal",
            "evaluation",
            "status",
            "clocks",
            "tiles",
            "cpu_gathers",
            "frame_wall_seconds",
            "exclusive_host_seconds",
            "clock_bracket_seconds",
            "unattributed_host_seconds",
            "diagnostic_allowance_bytes",
        ])
    );
    for tile in &frame.tiles {
        assert_eq!(
            (
                tile.record.dispatches_encoded,
                tile.record.commands_submitted,
                tile.record.host_wait_calls
            ),
            (1, 1, 1)
        );
    }
    assert!(serde_json::to_vec(frame).unwrap().len() <= 1024 * 1024);
}

#[test]
#[ignore = "explicit official BF16 batched G2 France prefill/decode full-bit and committed-KV/profile readiness"]
fn batched_submission_france_matches_legacy_logits_kv_and_sparse_groups() {
    let owner = load();
    let model = owner.model().unwrap();
    let ids = prompt(&owner);
    let (prefill, decode) = {
        let mut plain = owner.new_g2_session(&model, options(32)).unwrap();
        admitted(&owner, &plain);
        assert!(plain.bf16_profile_status().is_none());
        plain.append(&ids).unwrap();
        assert_eq!(plain.evaluate_next_chunk().unwrap(), 15);
        let prefill = Checkpoint::capture(&plain);
        plain.append(&[818]).unwrap();
        assert_eq!(plain.evaluate_next_chunk().unwrap(), 1);
        (prefill, Checkpoint::capture(&plain))
    };
    assert!(prefill.owned_bytes() + decode.owned_bytes() + AUXILIARY <= CALLER);
    for ordinal in [1, 2] {
        let mut candidate = owner
            .new_batched_profiled_session(&model, options(32), profile(ordinal))
            .unwrap();
        admitted(&owner, &candidate);
        candidate.append(&ids).unwrap();
        assert_eq!(candidate.evaluate_next_chunk().unwrap(), 15);
        prefill.assert_matches(&candidate);
        candidate.append(&[818]).unwrap();
        assert_eq!(candidate.evaluate_next_chunk().unwrap(), 1);
        decode.assert_matches(&candidate);
        let frame = candidate.bf16_profile().unwrap();
        assert_eq!(frame.evaluation_ordinal, ordinal);
        assert_eq!(frame.request_id, [ordinal as u8; 16]);
        assert_eq!(
            (
                frame.evaluation.position,
                frame.evaluation.input_rows,
                frame.evaluation.output_rows,
                frame.evaluation.committed_positions
            ),
            if ordinal == 1 {
                (0, 15, 1, 15)
            } else {
                (15, 1, 1, 16)
            }
        );
        validate_frame(
            &frame,
            32,
            if ordinal == 1 { 41 } else { 42 },
            if ordinal == 1 { 278 } else { 277 },
        );
    }
}

#[test]
#[ignore = "explicit official BF16 chunks1/3/32 initialized-fit/fallback, full-chunk and tail profiles, legacy wire"]
fn batched_submission_chunk_geometry_matches_legacy_and_keeps_v1_wire() {
    let owner = load();
    let model = owner.model().unwrap();
    let tokens: Vec<_> =
        include_bytes!("../../../../../benchmarks/gemma-quality-v1/held_out_loss.u32le")
            .chunks_exact(4)
            .take(33)
            .map(|raw| u32::from_le_bytes(raw.try_into().unwrap()))
            .collect();
    for (chunk, full_groups, full_commands, tail_groups, tail_commands) in [
        (1, 0, 349, 0, 349),
        (3, 40, 309, 41, 308),
        (32, 41, 278, 42, 277),
    ] {
        let inventory = owner.runtime_inventory_batched_single_row(chunk).unwrap();
        assert_eq!(inventory.schema_version, Some(2));
        assert_eq!(inventory.submission_policy, Some("batched_single_row"));
        assert_eq!(
            inventory
                .operations
                .iter()
                .map(|op| op.predicted_commands_per_chunk)
                .sum::<usize>(),
            full_commands
        );
        assert_eq!(
            inventory
                .operations
                .iter()
                .map(|op| op.predicted_decode_commands_per_chunk.unwrap())
                .sum::<usize>(),
            tail_commands
        );
        assert_eq!(
            inventory
                .operations
                .iter()
                .map(|op| op.predicted_decode_host_waits_per_chunk.unwrap())
                .sum::<usize>(),
            tail_commands
        );
        let (full, tail) = {
            let mut legacy = owner.new_g2_session(&model, options(chunk)).unwrap();
            admitted(&owner, &legacy);
            legacy.append(&tokens[..chunk]).unwrap();
            assert_eq!(legacy.evaluate_next_chunk().unwrap(), chunk);
            let full = Checkpoint::capture(&legacy);
            legacy.append(&tokens[chunk..chunk + 1]).unwrap();
            assert_eq!(legacy.evaluate_next_chunk().unwrap(), 1);
            (full, Checkpoint::capture(&legacy))
        };
        assert!(full.owned_bytes() + tail.owned_bytes() + AUXILIARY <= CALLER);
        for ordinal in [1, 2] {
            let mut candidate = owner
                .new_batched_profiled_session(&model, options(chunk), profile(ordinal))
                .unwrap();
            admitted(&owner, &candidate);
            candidate.append(&tokens[..chunk]).unwrap();
            assert_eq!(candidate.evaluate_next_chunk().unwrap(), chunk);
            full.assert_matches(&candidate);
            candidate.append(&tokens[chunk..chunk + 1]).unwrap();
            assert_eq!(candidate.evaluate_next_chunk().unwrap(), 1);
            tail.assert_matches(&candidate);
            let frame = candidate.bf16_profile().unwrap();
            validate_frame(
                &frame,
                chunk,
                if ordinal == 1 {
                    full_groups
                } else {
                    tail_groups
                },
                if ordinal == 1 {
                    full_commands
                } else {
                    tail_commands
                },
            );
            assert_eq!(
                frame.evaluation.input_rows,
                if ordinal == 1 { chunk } else { 1 }
            );
        }
        if chunk == 32 {
            let mut legacy = owner
                .new_profiled_session(&model, options(chunk), profile(1))
                .unwrap();
            admitted(&owner, &legacy);
            legacy.append(&tokens[..chunk]).unwrap();
            legacy.evaluate_next_chunk().unwrap();
            full.assert_matches(&legacy);
            assert_legacy_wire(&legacy.bf16_profile().unwrap());
            let inventory_wire =
                serde_json::to_value(owner.runtime_inventory(chunk).unwrap()).unwrap();
            assert_eq!(inventory_wire.as_object().unwrap().len(), 6);
            assert!(
                inventory_wire.get("schema_version").is_none()
                    && inventory_wire.get("submission_policy").is_none()
            );
        }
    }
}

#[test]
#[ignore = "explicit official BF16 batch late-PLE/head failure preserves prefix, invalidates capture and requires recreation"]
fn batched_submission_late_faults_preserve_prefix_and_never_publish_stale_capture() {
    let owner = load();
    let model = owner.model().unwrap();
    let ids = prompt(&owner);
    let before = {
        let mut plain = owner.new_g2_session(&model, options(32)).unwrap();
        admitted(&owner, &plain);
        plain.append(&ids).unwrap();
        plain.evaluate_next_chunk().unwrap();
        Checkpoint::capture(&plain)
    };
    for fault in [Bf16GraphFault::AfterPle, Bf16GraphFault::AfterFirstHeadTile] {
        {
            let mut candidate = owner
                .new_batched_profiled_session(&model, options(32), profile(1))
                .unwrap();
            admitted(&owner, &candidate);
            candidate.append(&ids).unwrap();
            candidate.evaluate_next_chunk().unwrap();
            before.assert_matches(&candidate);
            validate_frame(&candidate.bf16_profile().unwrap(), 32, 41, 278);
            candidate.inner.operations.set_fault(fault);
            candidate.append(&[818]).unwrap();
            assert!(candidate.evaluate_next_chunk().is_err());
            assert_eq!(
                (candidate.evaluated_len(), candidate.pending_len()),
                (15, 1)
            );
            assert_eq!(candidate.history()[15], 818);
            before.assert_kv(&candidate);
            assert!(candidate.logits().is_none() && candidate.bf16_profile().is_err());
            assert_eq!(
                candidate.bf16_profile_status().unwrap().state,
                Bf16CaptureState::Invalid
            );
            candidate.reset();
            candidate.append(&ids).unwrap();
            assert!(
                candidate.evaluate_next_chunk().is_err(),
                "logical reset must not heal poisoned backend"
            );
            assert!(candidate.bf16_profile().is_err());
        }
        let mut recreated = owner.new_batched_g2_session(&model, options(32)).unwrap();
        admitted(&owner, &recreated);
        recreated.append(&ids).unwrap();
        recreated.evaluate_next_chunk().unwrap();
        before.assert_matches(&recreated);
    }
}

#[test]
#[ignore = "explicit official BF16 pre/mid cancellation, pending continuation, capacity and reset semantics"]
fn batched_submission_generation_cancellation_pending_and_reset_match_plain() {
    let owner = load();
    let model = owner.model().unwrap();
    let ids = prompt(&owner);
    let (prefix, final_state, expected) = {
        let mut plain = owner.new_g2_session(&model, options(32)).unwrap();
        admitted(&owner, &plain);
        // Match Input::Reset's physical cache generation before comparing the
        // interrupted candidate with this uninterrupted reference.
        plain.reset();
        plain.append(&ids).unwrap();
        plain.evaluate_next_chunk().unwrap();
        let prefix = Checkpoint::capture(&plain);
        let mut emitted = Vec::with_capacity(4);
        generation::generate(
            &mut plain,
            Request {
                input: Input::Continue,
                max_new_tokens: 4,
                stop_tokens: &[],
            },
            &AtomicBool::new(false),
            |event| {
                emitted.push(event.token_id);
                Ok(())
            },
        )
        .unwrap();
        (prefix, Checkpoint::capture(&plain), emitted)
    };
    assert!(prefix.owned_bytes() + final_state.owned_bytes() + AUXILIARY <= CALLER);
    assert_eq!(expected, [818, 5279, 529, 7001]);
    let mut candidate = owner
        .new_batched_profiled_session(&model, options(32), profile(1))
        .unwrap();
    admitted(&owner, &candidate);
    let cancelled = AtomicBool::new(true);
    let request = || Request {
        input: Input::Reset(&ids),
        max_new_tokens: 4,
        stop_tokens: &[],
    };
    let pre = generation::generate(&mut candidate, request(), &cancelled, |_| {
        panic!("pre-cancel emitted")
    })
    .unwrap();
    assert_eq!(pre.reason, StopReason::Cancelled);
    assert!(candidate.history().is_empty() && candidate.bf16_profile().is_err());
    assert_eq!(candidate.bf16_profile_status().unwrap().evaluations_seen, 0);
    cancelled.store(false, Ordering::Relaxed);
    let mut emitted = Vec::with_capacity(4);
    let stopped = generation::generate(&mut candidate, request(), &cancelled, |event| {
        emitted.push(event.token_id);
        cancelled.store(true, Ordering::Relaxed);
        Ok(())
    })
    .unwrap();
    assert_eq!(stopped.reason, StopReason::Cancelled);
    assert_eq!(
        (candidate.evaluated_len(), candidate.pending_len()),
        (15, 1)
    );
    assert_eq!((stopped.sampled_tokens, stopped.emitted_tokens), (1, 1));
    prefix.assert_kv(&candidate);
    cancelled.store(false, Ordering::Relaxed);
    generation::generate(
        &mut candidate,
        Request {
            input: Input::Continue,
            max_new_tokens: 3,
            stop_tokens: &[],
        },
        &cancelled,
        |event| {
            emitted.push(event.token_id);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(emitted, expected);
    final_state.assert_matches(&candidate);
    assert_eq!(
        (
            candidate.history().len(),
            candidate.evaluated_len(),
            candidate.pending_len()
        ),
        (19, 18, 1)
    );
    let boundary = [2; 385];
    assert!(
        generation::generate(
            &mut candidate,
            Request {
                input: Input::Reset(&boundary),
                max_new_tokens: 128,
                stop_tokens: &[]
            },
            &cancelled,
            |_| panic!("capacity reject emitted")
        )
        .is_err()
    );
    final_state.assert_matches(&candidate);
    candidate.reset();
    assert!(candidate.history().is_empty() && candidate.logits().is_none());
    assert!(candidate.bf16_profile().is_err());
    candidate.append(&ids).unwrap();
    candidate.evaluate_next_chunk().unwrap();
    bitwise(candidate.logits().unwrap(), prefix.logits.as_ref().unwrap());
    for (cache, want) in candidate.inner.caches.iter().zip(&prefix.caches) {
        assert!(cache.generation() > want.generation);
        let segments = cache.committed_segments();
        assert!(
            segments
                .iter()
                .flat_map(|s| s.k)
                .map(|v| v.to_bits())
                .eq(want.keys.iter().copied())
        );
        assert!(
            segments
                .iter()
                .flat_map(|s| s.v)
                .map(|v| v.to_bits())
                .eq(want.values.iter().copied())
        );
    }
}

#[test]
#[ignore = "explicit official BF16 batched exact admission and warmed unprofiled numerical Rust allocation control"]
fn batched_submission_exact_admission_and_warmed_decode_allocate_no_rust_storage() {
    let owner = load();
    let model = owner.model().unwrap();
    let ids = prompt(&owner);
    let memory = {
        let warm = owner.new_batched_g2_session(&model, options(32)).unwrap();
        admitted(&owner, &warm);
        warm.memory()
    };
    let required = memory.admitted_bytes + owner.memory().session_reserved_bytes;
    let mut selected = options(32);
    selected.memory_limit_bytes = required - 1;
    assert!(
        matches!(owner.new_batched_g2_session(&model,selected),Err(session::Error::MemoryBudget{required:actual,limit}) if actual==required && limit==required-1)
    );
    selected.memory_limit_bytes = required;
    let mut candidate = owner.new_batched_g2_session(&model, selected).unwrap();
    admitted(&owner, &candidate);
    assert_eq!(candidate.memory(), memory);
    assert!(candidate.bf16_profile_status().is_none());
    candidate.append(&ids).unwrap();
    candidate.evaluate_next_chunk().unwrap();
    candidate.append(&[818]).unwrap();
    candidate.evaluate_next_chunk().unwrap();
    let addresses = pointers(&candidate.inner);
    let measurement = Measurement::start();
    let append = candidate.append(&[5279]);
    let evaluate = candidate.evaluate_next_chunk();
    let allocations = measurement.finish();
    append.unwrap();
    assert_eq!(evaluate.unwrap(), 1);
    println!(
        "BF16_BATCH_DECODE_ALLOCATION {}",
        serde_json::json!({"rust_allocations":allocations,
            "release_zero_required":!cfg!(debug_assertions),
            "allocation_gate_qualified":!cfg!(debug_assertions) && allocations==0,
            "scope":"Warmed append5279 plus completed unprofiled decode; debug objc2 encoding checks included; native driver allocations excluded"})
    );
    // objc2 debug signature verification parses owned encoding trees per
    // message. Keep these counts visible without treating debug as release.
    if !cfg!(debug_assertions) {
        assert_eq!(allocations, 0);
    }
    assert_eq!(
        (
            candidate.history().len(),
            candidate.evaluated_len(),
            candidate.pending_len()
        ),
        (17, 17, 0)
    );
    assert_eq!(candidate.memory(), memory);
    assert_eq!(pointers(&candidate.inner), addresses);
    assert_eq!(candidate.logits().unwrap().len(), VOCAB);
    assert!(candidate.logits().unwrap().iter().all(|v| v.is_finite()));
}
