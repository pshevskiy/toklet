use super::*;
use crate::generation::{self, Input, Request, StopReason};
use crate::model::gemma_packed::tests::{both, fixture, pointers, snapshot};
use crate::model::gemma_reference::tests::Measurement;
use std::{
    io,
    sync::atomic::{AtomicBool, Ordering},
};

mod capacity;
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod g2;
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod g3;
mod verification;

fn options() -> Options {
    Options {
        capacity: 32,
        prefill_chunk_size: 3,
        position_offset: 0,
        memory_limit_bytes: crate::weights::INITIAL_MEMORY_LIMIT_BYTES,
    }
}
fn evaluated(session: &mut GenerationSession<'_, '_>) {
    while session.evaluate_next_chunk().unwrap() != 0 {}
}
fn sum(memory: Memory) -> usize {
    memory.weights_bytes
        + memory.kv_bytes
        + memory.scratch_bytes
        + memory.history_bytes
        + memory.logits_bytes
        + memory.metadata_bytes
        + memory.metadata_allowance_bytes
}

#[test]
fn admission_categories_actual_capacities_and_smaller_request_limits() {
    let data = fixture::Fixture::new();
    let plain = Model::new(data.weights()).unwrap();
    let report = crate::mlx::gemma4::payload::LoadReport {
        container_bytes: 0,
        container_sha256: [0; 32],
        text_source_bytes: 0,
        excluded_source_bytes: 0,
        packed_matrices: 0,
        fp32_vectors: 0,
        tensor_capacity_bytes: 200_000,
        metadata_reservation_bytes: 64 * 1024 * 1024,
        admitted_load_bytes: usize::MAX,
    };
    let backed = Model::with_storage(
        data.weights(),
        Layout::witness(),
        Some(Backing::cpu(&report)),
    )
    .unwrap();
    for model in [&plain, &backed] {
        let session = model.new_generation_session(options()).unwrap();
        let memory = session.memory();
        let extra = std::mem::size_of::<GenerationSession<'_, '_>>()
            - std::mem::size_of::<Session<'_, '_>>()
            + session.history.capacity() * 4;
        assert_eq!(
            memory.admitted_bytes,
            model.model_bytes() + session.inner.allocated_bytes().unwrap() + extra
        );
        assert_eq!(sum(memory), memory.admitted_bytes);
        assert_eq!(memory.history_bytes, 256);
        assert_eq!(
            memory.logits_bytes,
            2 * options().prefill_chunk_size * 17 * 4
        );
        assert_eq!(memory.weights_bytes, model.tensor_bytes);
        assert_eq!(
            memory.metadata_allowance_bytes,
            model.metadata_allowance_bytes
        );
        for budget in [memory.admitted_bytes - 1, memory.admitted_bytes] {
            let mut requested = options();
            requested.memory_limit_bytes = budget;
            let measured = Measurement::start();
            let result = model.new_generation_session(requested);
            let allocations = measured.finish();
            if budget < memory.admitted_bytes {
                assert!(result.is_err());
                assert_eq!(allocations, 0);
            } else {
                assert_eq!(result.unwrap().memory(), memory);
            }
        }
        let mut smaller = options();
        smaller.capacity = 5;
        smaller.prefill_chunk_size = 2;
        let mut small = model.new_generation_session(smaller).unwrap();
        assert_eq!(small.inner.chunk_capacity(), 2);
        assert_eq!(small.memory().history_bytes, 2 * smaller.capacity * 4);
        assert_eq!(small.inner.capacity(), smaller.capacity);
        assert_eq!(small.memory().weights_bytes, memory.weights_bytes);
        assert_eq!(small.memory().logits_bytes, 2 * 2 * 17 * 4);
        assert!(small.memory().scratch_bytes < memory.scratch_bytes);
        assert!(small.memory().kv_bytes < memory.kv_bytes); // smaller proposals, same caches
        assert!(small.memory().admitted_bytes < memory.admitted_bytes);
        small.append(&data.ids[..5]).unwrap();
        assert!(small.append(&[1]).is_err());
        evaluated(&mut small);
        assert_eq!(small.evaluated_len(), 5);
        assert!(small.set_tokens(&data.ids[..6]).is_err());
    }
    for change in 0..7 {
        let mut invalid = options();
        match change {
            0 => invalid.capacity = 0,
            1 => invalid.capacity = 33,
            2 => invalid.prefill_chunk_size = 0,
            3 => invalid.prefill_chunk_size = 33,
            4 => invalid.position_offset = 1,
            5 => invalid.memory_limit_bytes = 0,
            _ => invalid.memory_limit_bytes = usize::MAX,
        }
        let measured = Measurement::start();
        let rejected = plain.new_generation_session(invalid).is_err();
        assert_eq!(measured.finish(), 0);
        assert!(rejected);
    }
}

#[test]
fn committed_prefix_pending_and_fork_replay_match_frozen_rows() {
    let data = fixture::Fixture::new();
    let model = Model::new(data.weights()).unwrap();
    let mut session = model.new_generation_session(options()).unwrap();
    session.append(&data.ids[..5]).unwrap();
    assert_eq!(session.evaluated_len(), 0);
    assert!(session.logits().is_none());
    assert_eq!(session.evaluate_next_chunk().unwrap(), 3);
    assert_eq!(session.pending_len(), 2);
    assert!(session.logits().is_none());
    assert_eq!(session.evaluate_next_chunk().unwrap(), 2);
    both(&data, "logits", session.logits().unwrap(), 4, 1);
    let prior = snapshot(&session.inner);
    let addresses = pointers(&session.inner);
    session.append(&[1, 2]).unwrap();
    assert_eq!(session.evaluated_len(), 5);
    assert!(session.logits().is_none());
    assert_eq!(snapshot(&session.inner), prior);
    session.set_tokens(&data.ids[..5]).unwrap();
    assert_eq!(session.evaluate_next_chunk().unwrap(), 0);
    both(&data, "logits", session.logits().unwrap(), 4, 1);
    session.set_tokens(&data.ids[..8]).unwrap();
    assert_eq!(session.evaluated_len(), 5);
    evaluated(&mut session);
    both(&data, "logits", session.logits().unwrap(), 7, 1);
    let before = session.history().to_vec();
    let raw_before = snapshot(&session.inner);
    assert!(session.set_tokens(&[17]).is_err());
    assert!(session.append(&[17]).is_err());
    assert_eq!(session.history(), before);
    assert_eq!(snapshot(&session.inner), raw_before);
    // Shortening invalidates immediately; matching stale raw bytes cannot undo it.
    session.set_tokens(&data.ids[..3]).unwrap();
    assert_eq!(session.evaluated_len(), 0);
    assert!(session.logits().is_none());
    assert_eq!(session.pending_len(), 3);
    session.set_tokens(&data.ids[..8]).unwrap();
    assert_eq!(session.evaluated_len(), 0);
    evaluated(&mut session);
    both(&data, "logits", session.logits().unwrap(), 7, 1);
    let mut changed = data.ids[..6].to_vec();
    changed[1] = (changed[1] + 1) % 17;
    session.set_tokens(&changed).unwrap();
    assert_eq!(session.evaluated_len(), 0);
    evaluated(&mut session);
    let mut control = model.new_session(usize::MAX).unwrap();
    let all = control.forward(&changed).unwrap();
    assert_eq!(session.logits().unwrap(), &all[5 * 17..]);
    assert_eq!(pointers(&session.inner), addresses);
    session.reset();
    assert!(session.history().is_empty());
    assert!(session.logits().is_none());
    assert_eq!(session.evaluate_next_chunk().unwrap(), 0);
    session.append(&data.ids[..3]).unwrap();
    session.discard_pending();
    assert!(session.history().is_empty());
    assert_eq!(session.evaluated_len(), 0);
    session.append(&data.ids[..3]).unwrap();
    evaluated(&mut session);
    both(&data, "logits", session.logits().unwrap(), 2, 1);
}

#[test]
fn real_reset_error_and_late_forward_error_keep_pending_state_retryable() {
    let data = fixture::Fixture::new();
    let model = Model::new(data.weights()).unwrap();
    let mut session = model.new_generation_session(options()).unwrap();
    session.append(&data.ids[..3]).unwrap();
    evaluated(&mut session);
    session.inner.caches[1].set_generation_for_test(u64::MAX);
    let old = snapshot(&session.inner);
    session.reset();
    session.append(&data.ids[..3]).unwrap();
    assert!(session.evaluate_next_chunk().is_err());
    assert_eq!(snapshot(&session.inner), old);
    assert!(session.needs_replay);
    assert_eq!(session.pending_len(), 3);
    assert_eq!(session.evaluated_len(), 0);
    assert!(session.logits().is_none());
    let generation = session.inner.caches[0].generation();
    session.inner.caches[1].set_generation_for_test(generation);
    session.inner.fault = Some(6); // final capped-logit boundary in the four-layer graph
    assert!(session.evaluate_next_chunk().is_err());
    assert!(!session.needs_replay);
    assert_eq!(session.evaluated_len(), 0);
    assert_eq!(session.pending_len(), 3);
    assert!(session.inner.tokens().is_empty());
    assert!(session.logits().is_none());
    session.inner.fault = None;
    evaluated(&mut session);
    both(&data, "logits", session.logits().unwrap(), 2, 1);
    let before = snapshot(&session.inner);
    session.append(&data.ids[3..5]).unwrap();
    session.inner.fault = Some(6);
    assert!(session.evaluate_next_chunk().is_err());
    assert_eq!(snapshot(&session.inner), before);
    assert_eq!(session.evaluated_len(), 3);
    assert_eq!(session.pending_len(), 2);
    session.discard_pending();
    assert_eq!(session.history(), &data.ids[..3]);
    both(&data, "logits", session.logits().unwrap(), 2, 1);
    session.inner.fault = None;
}

#[test]
fn existing_generator_stop_cancel_output_failure_and_continuation_keep_pending_separate() {
    let data = fixture::Fixture::new();
    let model = Model::new(data.weights()).unwrap();
    let mut session = model.new_generation_session(options()).unwrap();
    generator_checks(&data, &mut session, || {
        model.new_session(usize::MAX).unwrap()
    });
}
pub(in crate::model::gemma_packed) fn generator_checks<
    'model,
    'weights: 'model,
    M: MatrixSource + 'model,
    B: Operations<M>,
>(
    data: &fixture::Fixture,
    session: &mut GenerationSession<'model, 'weights, M, B>,
    mut control: impl FnMut() -> Session<'model, 'weights, M, B>,
) where
    B::Error: std::fmt::Debug,
{
    let cancelled = AtomicBool::new(false);
    let expected = crate::sampling::greedy(
        &data.values("staged_fp32", "logits")[2 * 17..3 * 17]
            .iter()
            .map(|v| *v as f32)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let report = generation::generate(
        session,
        Request {
            input: Input::Prompt(&data.ids[..3]),
            max_new_tokens: 2,
            stop_tokens: &[expected],
        },
        &cancelled,
        |_| panic!("Stop token must not emit"),
    )
    .unwrap();
    assert_eq!(report.reason, StopReason::StopToken { token_id: expected });
    assert_eq!(
        (
            report.sampled_tokens,
            report.emitted_tokens,
            report.evaluated_tokens,
            report.pending_tokens
        ),
        (1, 0, 3, 1)
    );
    assert_eq!(session.history().last(), Some(&expected));
    let before = snapshot(&session.inner);
    let history = session.history().to_vec();
    let rejected = generation::generate(
        session,
        Request {
            input: Input::Continue,
            max_new_tokens: 32,
            stop_tokens: &[],
        },
        &cancelled,
        |_| panic!("Rejected request must not emit"),
    );
    assert!(rejected.is_err());
    assert_eq!(snapshot(&session.inner), before);
    assert_eq!(session.history(), history);
    cancelled.store(true, Ordering::Relaxed);
    let report = generation::generate(
        session,
        Request {
            input: Input::Reset(&data.ids[..2]),
            max_new_tokens: 1,
            stop_tokens: &[],
        },
        &cancelled,
        |_| panic!(),
    )
    .unwrap();
    assert_eq!(report.reason, StopReason::Cancelled);
    assert_eq!(snapshot(&session.inner), before);
    assert_eq!(session.history(), history);
    cancelled.store(false, Ordering::Relaxed);
    let error = generation::generate(
        session,
        Request {
            input: Input::Reset(&data.ids[..3]),
            max_new_tokens: 2,
            stop_tokens: &[],
        },
        &cancelled,
        |_| Err(io::Error::other("injected output failure")),
    )
    .unwrap_err();
    assert!(matches!(error, generation::Error::Output(_)));
    assert_eq!(session.evaluated_len(), 3);
    assert_eq!(session.pending_len(), 1);
    let history = session.history().to_vec();
    let report = generation::generate(
        session,
        Request {
            input: Input::Continue,
            max_new_tokens: 0,
            stop_tokens: &[],
        },
        &cancelled,
        |_| panic!(),
    )
    .unwrap();
    assert_eq!(report.pending_tokens, 0);
    assert_eq!(session.evaluated_len(), 4);
    let mut control = control();
    let raw = control.forward(&history).unwrap();
    assert_eq!(session.logits().unwrap(), &raw[3 * 17..]);
    let report = generation::generate(
        session,
        Request {
            input: Input::Prompt(&data.ids[..3]),
            max_new_tokens: 2,
            stop_tokens: &[],
        },
        &cancelled,
        |_| {
            cancelled.store(true, Ordering::Relaxed);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(report.reason, StopReason::Cancelled);
    assert_eq!(report.emitted_tokens, 1);
    assert_eq!(session.pending_len(), 1);
    session.discard_pending();
    assert_eq!(session.evaluated_len(), 3);
    assert_eq!(session.pending_len(), 0);
    both(data, "logits", session.logits().unwrap(), 2, 1);
}

#[test]
fn warmed_history_evaluation_replay_reset_and_discard_allocate_nothing() {
    let data = fixture::Fixture::new();
    let model = Model::new(data.weights()).unwrap();
    let mut session = model.new_generation_session(options()).unwrap();
    session.append(&data.ids[..3]).unwrap();
    evaluated(&mut session);
    session.reset();
    let memory = session.memory();
    let addresses = pointers(&session.inner);
    let measured = Measurement::start();
    for _ in 0..3 {
        for token in &data.ids {
            session.append(std::slice::from_ref(token)).unwrap();
            session.evaluate_next_chunk().unwrap();
        }
        session.set_tokens(&data.ids).unwrap();
        assert_eq!(session.evaluate_next_chunk().unwrap(), 0);
        session.set_tokens(&data.ids[..3]).unwrap();
        session.evaluate_next_chunk().unwrap();
        session.append(&[1]).unwrap();
        session.discard_pending();
        session.reset();
    }
    assert_eq!(measured.finish(), 0);
    assert_eq!(session.memory(), memory);
    assert_eq!(pointers(&session.inner), addresses);
    println!(
        "Gemma generation: 96 warmed decode steps,3 fork replays,resets,discards; zero Rust allocations"
    );
}

mod long_context;

fn assert_same_last_row(full: &GenerationSession<'_, '_>, last: &GenerationSession<'_, '_>) {
    assert_eq!(last.inner.tokens(), full.inner.tokens());
    assert_eq!(last.inner.logits().len(), VOCAB_SIZE);
    let expected = full.inner.logits();
    assert!(
        expected[expected.len() - VOCAB_SIZE..]
            .iter()
            .map(|x| x.to_bits())
            .eq(last.inner.logits().iter().map(|x| x.to_bits()))
    );
    for (a, b) in full.inner.caches.iter().zip(&last.inner.caches) {
        assert_eq!((a.offset(), a.generation()), (b.offset(), b.generation()));
        let (ak, av) = a.committed();
        let (bk, bv) = b.committed();
        assert!(
            ak.iter()
                .chain(av)
                .map(|x| x.to_bits())
                .eq(bk.iter().chain(bv).map(|x| x.to_bits()))
        );
    }
}

#[test]
fn last_row_prefill_matches_full_tail_and_preserves_transactional_cache() {
    let data = fixture::Fixture::new();
    let model = Model::new(data.weights()).unwrap();
    for chunk in [1, 3, 32] {
        let options = Options {
            prefill_chunk_size: chunk,
            ..options()
        };
        let mut full = model.new_generation_session(options).unwrap();
        let mut last = model.new_generation_session(options).unwrap();
        let retry_rows = chunk.min(3);
        let ids: Vec<_> = data.ids.iter().copied().cycle().take(32).collect();
        let (prefix, suffix) = ids.split_at(ids.len() - retry_rows);
        for session in [&mut full, &mut last] {
            session.append(prefix).unwrap();
        }
        while full.pending_len() > 0 {
            let rows = full.evaluate_next_chunk().unwrap();
            let measured = Measurement::start();
            let actual = last.evaluate_next_chunk_last_row().unwrap();
            assert_eq!(measured.finish(), 0);
            assert_eq!(rows, actual);
            assert_same_last_row(&full, &last);
        }
        let before = snapshot(&last.inner);
        last.append(suffix).unwrap();
        assert!(last.append(&ids[..1]).is_err());
        for fault in [model.layout.producers, model.layout.layers.len() + 1] {
            last.inner.fault = Some(fault);
            assert!(last.evaluate_next_chunk_last_row().is_err());
            assert_eq!(snapshot(&last.inner), before);
            assert_eq!(last.pending_len(), retry_rows);
            assert_eq!(last.history(), ids.as_slice());
        }
        last.inner.fault = None;
        full.append(suffix).unwrap();
        assert_eq!(full.evaluate_next_chunk().unwrap(), retry_rows);
        assert_eq!(last.evaluate_next_chunk_last_row().unwrap(), retry_rows);
        assert_eq!(last.history(), full.history());
        assert_same_last_row(&full, &last);
        assert_eq!(last.evaluated_len(), 32);
    }
}
