use super::*;
use crate::{
    generation::{self, Input, Request, StopReason},
    model::{
        gemma_reference::tests::Measurement,
        qwen35::tests::{Fixture, assert_float_bits, snapshot, tokens},
    },
};
use std::sync::atomic::{AtomicBool, Ordering};

fn options() -> Options {
    Options {
        capacity: 17,
        chunk: 3,
        position_offset: 0,
        memory_limit_bytes: MEMORY_LIMIT,
    }
}
fn finish(session: &mut impl InferenceSession) {
    while session.evaluate_next_chunk().unwrap() != 0 {}
}
fn raw<'m, 'w>(model: &'m Model<'w>, ids: &[u32]) -> Session<'m, 'w> {
    let mut s = model.session(options()).unwrap();
    for part in ids.chunks(3) {
        s.forward(part).unwrap();
    }
    s
}

#[test]
fn admission_counts_distinct_owners_and_preserves_committed_distribution() {
    let fixture = Fixture::new();
    let model = fixture.model();
    let ids = tokens();
    let numerical = raw(&model, &ids[..3]);
    let before = snapshot(&numerical);
    let mut session = GenerationSession::new(numerical, MEMORY_LIMIT).unwrap();
    let memory = session.memory_report();
    let extra = std::mem::size_of_val(&session) - std::mem::size_of_val(&session.inner)
        + session.history.capacity() * 4;
    assert_eq!(memory.numerical, session.inner.memory());
    assert_eq!(memory.pending_history_bytes, session.capacity() * 4);
    assert_eq!(
        memory.admitted_bytes,
        memory.numerical.admitted_bytes + extra
    );
    assert_eq!(snapshot(session.numerical_for_tests()), before);
    assert_eq!(session.history(), &ids[..3]);
    assert_eq!(session.state().prefix_action, PrefixAction::FullyCached);
    assert!(session.logits().is_some());
    let c = session.memory();
    assert_eq!(
        c.admitted_bytes,
        c.weights_bytes
            + c.kv_bytes
            + c.scratch_bytes
            + c.history_bytes
            + c.logits_bytes
            + c.metadata_bytes
            + c.metadata_allowance_bytes
    );
    assert_eq!(
        c.scratch_bytes,
        memory.numerical.scratch_bytes
            + memory.numerical.recurrent_bank_bytes
            + memory.numerical.convolution_bank_bytes
    );
    session.append(&ids[3..5]).unwrap();
    assert_eq!(session.memory_report(), memory);
    assert_eq!(session.into_numerical().history(), &ids[..3]);
    for limit in [0, memory.admitted_bytes - 1, MEMORY_LIMIT + 1] {
        let numerical = model.session(options()).unwrap();
        let measure = Measurement::start();
        let error = GenerationSession::new(numerical, limit).err().unwrap();
        assert_eq!(measure.finish(), 0);
        if limit == memory.admitted_bytes - 1 {
            assert!(
                matches!(error, Error::MemoryBudget { required, .. } if required == memory.admitted_bytes)
            );
        }
    }
    let exact =
        GenerationSession::new(model.session(options()).unwrap(), memory.admitted_bytes).unwrap();
    assert_eq!(exact.memory_report(), memory);
    assert_eq!(
        exact.inner.options.memory_limit_bytes,
        memory.numerical.admitted_bytes
    );
}

#[test]
fn exact_prefix_pending_replacement_and_fully_cached_logits() {
    let fixture = Fixture::new();
    let model = fixture.model();
    let ids = tokens();
    let mut s = GenerationSession::new(model.session(options()).unwrap(), MEMORY_LIMIT).unwrap();
    s.set_tokens(&ids).unwrap();
    assert_eq!(s.state().prefix_action, PrefixAction::Cold);
    assert!(s.logits().is_none());
    assert_eq!(s.evaluate_next_chunk().unwrap(), 3);
    assert_eq!(s.evaluate_next_chunk().unwrap(), 3);
    assert_eq!(s.evaluated_len(), 6);
    assert_eq!(s.pending_len(), 11);
    let committed = snapshot(&s.inner);
    for invalid in [vec![32], vec![1; 18]] {
        let state = s.state();
        let history = s.history().to_vec();
        assert!(s.set_tokens(&invalid).is_err());
        assert!(s.append(&invalid).is_err());
        assert_eq!(s.state(), state);
        assert_eq!(s.history(), history);
        assert_eq!(snapshot(&s.inner), committed);
    }
    s.set_tokens(&ids[..6]).unwrap();
    assert_eq!(s.state().prefix_action, PrefixAction::FullyCached);
    assert_eq!(s.evaluate_next_chunk().unwrap(), 0);
    assert_float_bits(
        s.logits().unwrap(),
        raw(&model, &ids[..6]).logits().unwrap(),
    );
    s.set_tokens(&ids[..10]).unwrap();
    assert_eq!(s.state().prefix_action, PrefixAction::ExactPrefix);
    let mut edited = ids[..10].to_vec();
    edited[8] = (edited[8] + 1) % 32;
    s.set_tokens(&edited).unwrap();
    assert_eq!(s.state().prefix_action, PrefixAction::ExactPrefix);
    assert_eq!(s.evaluated_len(), 6);
    finish(&mut s);
    assert_eq!(snapshot(&s.inner), snapshot(&raw(&model, &edited)));
    assert_eq!(s.pending_len(), 0);
    s.append(&[1]).unwrap();
    assert!(s.logits().is_none());
    s.discard_pending();
    assert_eq!(s.history(), edited);
    assert!(s.logits().is_some());
}

#[test]
fn changed_shortened_reset_and_discard_replay_without_warmed_allocations() {
    let fixture = Fixture::new();
    let model = fixture.model();
    let ids = tokens();
    let mut s = GenerationSession::new(model.session(options()).unwrap(), MEMORY_LIMIT).unwrap();
    s.set_tokens(&ids[..10]).unwrap();
    finish(&mut s);
    let mut changed = ids[..10].to_vec();
    changed[2] = (changed[2] + 1) % 32;
    let measure = Measurement::start();
    s.set_tokens(&changed).unwrap();
    assert_eq!(s.evaluated_len(), 0);
    assert!(s.logits().is_none());
    finish(&mut s);
    assert_eq!(measure.finish(), 0);
    assert_eq!(s.state().prefix_action, PrefixAction::ChangedReplay);
    assert_eq!(snapshot(&s.inner), snapshot(&raw(&model, &changed)));
    s.set_tokens(&changed[..5]).unwrap();
    assert_eq!(s.state().prefix_action, PrefixAction::ShortenedReplay);
    assert_eq!(s.evaluated_len(), 0);
    finish(&mut s);
    assert_eq!(snapshot(&s.inner), snapshot(&raw(&model, &changed[..5])));
    let measure = Measurement::start();
    s.reset();
    assert_eq!(s.state().prefix_action, PrefixAction::ExplicitReset);
    assert!(s.history().is_empty() && s.logits().is_none());
    s.append(&changed).unwrap();
    s.discard_pending();
    assert!(s.history().is_empty());
    assert_eq!(measure.finish(), 0);
}

#[test]
fn late_chunk_failure_and_cancellation_preserve_prefix_and_pending_for_retry() {
    let fixture = Fixture::new();
    let model = fixture.model();
    let ids = tokens();
    let mut s = GenerationSession::new(raw(&model, &ids[..3]), MEMORY_LIMIT).unwrap();
    s.append(&ids[3..6]).unwrap();
    let before = snapshot(&s.inner);
    let cancel = AtomicBool::new(false);
    let measure = Measurement::start();
    let result = s.evaluate_with(|inner, ids| {
        inner
            .forward_observed(ids, &cancel, &mut |o| {
                if o.stage == "before_commit" {
                    return Err(Error::Observation("late-generation-test"));
                }
                Ok(())
            })
            .map(|_| ())
    });
    assert!(matches!(result, Err(Error::Observation(_))));
    assert_eq!(measure.finish(), 0);
    assert_eq!(snapshot(&s.inner), before);
    assert_eq!(s.state().last_numerical_work.outcome, Outcome::Error);
    assert_eq!(s.state().last_numerical_work.tokens_completed, 3);
    assert_eq!(s.state().last_numerical_work.tokens_committed, 0);
    assert_eq!(s.evaluated_len(), 3);
    assert_eq!(s.pending_len(), 3);
    let result = s.evaluate_with(|inner, ids| {
        inner
            .forward_observed(ids, &cancel, &mut |o| {
                if o.stage == "before_commit" {
                    cancel.store(true, Ordering::Relaxed);
                }
                Ok(())
            })
            .map(|_| ())
    });
    assert!(matches!(result, Err(Error::Cancelled)));
    assert_eq!(s.state().last_numerical_work.outcome, Outcome::Cancelled);
    assert_eq!(snapshot(&s.inner), before);
    let measure = Measurement::start();
    assert_eq!(s.evaluate_next_chunk().unwrap(), 3);
    assert_eq!(measure.finish(), 0);
    assert_eq!(snapshot(&s.inner), snapshot(&raw(&model, &ids[..6])));
}

#[test]
fn controller_stop_continuation_output_failure_and_whole_output_admission() {
    let fixture = Fixture::new();
    let model = fixture.model();
    let ids = tokens();
    let mut s = GenerationSession::new(model.session(options()).unwrap(), MEMORY_LIMIT).unwrap();
    let stop = crate::sampling::greedy(raw(&model, &ids[..3]).logits().unwrap()).unwrap();
    let cancel = AtomicBool::new(false);
    let report = generation::generate(
        &mut s,
        Request {
            input: Input::Prompt(&ids[..3]),
            max_new_tokens: 4,
            stop_tokens: &[stop],
        },
        &cancel,
        |_| panic!("stop must not emit"),
    )
    .unwrap();
    assert_eq!(report.reason, StopReason::StopToken { token_id: stop });
    assert_eq!(
        (
            report.sampled_tokens,
            report.emitted_tokens,
            report.pending_tokens
        ),
        (1, 0, 1)
    );
    assert_eq!((s.history().len(), s.evaluated_len()), (4, 3));
    assert!(s.logits().is_none());
    let report = generation::generate(
        &mut s,
        Request {
            input: Input::Continue,
            max_new_tokens: 2,
            stop_tokens: &[],
        },
        &cancel,
        |_| Ok(()),
    )
    .unwrap();
    assert_eq!(report.reason, StopReason::OutputLimit);
    assert_eq!(
        (
            report.prefill_tokens,
            report.sampled_tokens,
            report.emitted_tokens
        ),
        (1, 2, 2)
    );
    assert_eq!(
        (s.history().len(), s.evaluated_len(), s.pending_len()),
        (6, 5, 1)
    );
    let state = s.state();
    let history = s.history().to_vec();
    assert!(
        generation::generate(
            &mut s,
            Request {
                input: Input::Continue,
                max_new_tokens: 12,
                stop_tokens: &[]
            },
            &cancel,
            |_| Ok(())
        )
        .is_err()
    );
    assert_eq!(s.state(), state);
    assert_eq!(s.history(), history);
    let error = generation::generate(
        &mut s,
        Request {
            input: Input::Reset(&ids[..3]),
            max_new_tokens: 2,
            stop_tokens: &[],
        },
        &cancel,
        |_| Err(std::io::Error::other("output-test")),
    );
    assert!(matches!(error, Err(generation::Error::Output(_))));
    assert_eq!(
        (s.history().len(), s.evaluated_len(), s.pending_len()),
        (4, 3, 1)
    );
    let expected = raw(&model, &s.history()[..3]);
    assert_eq!(snapshot(&s.inner), snapshot(&expected));
}

#[test]
fn controller_cancellation_empty_and_single_token_observations() {
    let fixture = Fixture::new();
    let model = fixture.model();
    let ids = tokens();
    let mut s = GenerationSession::new(model.session(options()).unwrap(), MEMORY_LIMIT).unwrap();
    let cancel = AtomicBool::new(true);
    let report = generation::generate(
        &mut s,
        Request {
            input: Input::Prompt(&ids[..3]),
            max_new_tokens: 4,
            stop_tokens: &[],
        },
        &cancel,
        |_| panic!("precancel"),
    )
    .unwrap();
    assert_eq!(report.reason, StopReason::Cancelled);
    assert!(s.history().is_empty());
    assert_eq!((report.sampled_tokens, report.emitted_tokens), (0, 0));
    assert!(report.first_token_seconds.is_none());
    cancel.store(false, Ordering::Relaxed);
    let report = generation::generate(
        &mut s,
        Request {
            input: Input::Prompt(&ids[..3]),
            max_new_tokens: 0,
            stop_tokens: &[],
        },
        &cancel,
        |_| panic!("no output"),
    )
    .unwrap();
    assert_eq!(
        (
            report.prefill_tokens,
            report.sampled_tokens,
            report.pending_tokens
        ),
        (3, 0, 0)
    );
    assert!(report.first_token_seconds.is_none());
    let report = generation::generate(
        &mut s,
        Request {
            input: Input::Prompt(&ids[..3]),
            max_new_tokens: 4,
            stop_tokens: &[],
        },
        &cancel,
        |_| {
            cancel.store(true, Ordering::Relaxed);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(report.reason, StopReason::Cancelled);
    assert_eq!(
        (
            report.reused_tokens,
            report.prefill_tokens,
            report.emitted_tokens,
            report.pending_tokens
        ),
        (3, 0, 1, 1)
    );
    assert!(report.first_token_seconds.is_some());
    assert_eq!(s.state().prefix_action, PrefixAction::FullyCached);
}
