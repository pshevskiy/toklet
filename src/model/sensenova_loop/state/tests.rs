use super::*;
use std::sync::atomic::Ordering;

fn plan() -> Plan {
    Plan::from_mlx_config(include_bytes!(
        "../../../../tests/fixtures/sensenova_loop/mlx-config.json"
    ))
    .unwrap()
}

fn options() -> Options {
    Options {
        capacity: 4,
        chunk: 2,
        position_offset: 7,
        memory_limit_bytes: 6 << 30,
    }
}

fn tiny() -> State {
    State::with_layout(
        *plan().visits(),
        Layout {
            rnn: 2,
            conv: 3,
            kv_heads: 2,
            head_dim: 2,
            position_limit: 128,
            vocab: 5,
        },
        options(),
        8,
        16,
    )
    .unwrap()
}

fn one(token: Token, frame: &mut Frame<'_>, logits: &mut [f32], fail: Option<usize>) -> Result<()> {
    assert_eq!(token.absolute, token.position + 7);
    let mut sum = 0.0;
    while let Some((visit, state)) = frame.next_visit()? {
        let value = (visit.state_slot + 1) as f32 + token.id as f32;
        match state {
            VisitState::Linear {
                recurrent,
                convolution,
            } => {
                for (i, (out, past)) in recurrent.next.iter_mut().zip(recurrent.prior).enumerate() {
                    *out = past + value + i as f32;
                    sum += *out;
                }
                for (i, (out, past)) in convolution
                    .next
                    .iter_mut()
                    .zip(convolution.prior)
                    .enumerate()
                {
                    *out = past + value + i as f32;
                    sum += *out;
                }
            }
            VisitState::Full(mut kv) => {
                for i in 0..2 {
                    let head = kv.head(i).unwrap();
                    assert_eq!(head.past_key.len(), token.position * 2);
                    assert_eq!(head.past_value.len(), token.position * 2);
                    sum += head.past_key.iter().chain(head.past_value).sum::<f32>();
                    head.key.fill(value + i as f32);
                    head.value.fill(-value - i as f32);
                }
                assert!(kv.head(2).is_none());
            }
        }
        if fail == Some(visit.state_slot) {
            return Err(invalid("SenseNova test second-pass failure"));
        }
    }
    logits.fill(sum);
    Ok(())
}

#[test]
fn fixed_visit_storage_has_original_head_major_capacity_and_bounded_reserves() {
    let mut o = options();
    o.capacity = 512;
    o.chunk = 32;
    let p = plan();
    let info = State::planned(&p, o, 5_050_626_048, 1_048_576).unwrap();
    assert_eq!(
        info.recurrent_bank_bytes + info.convolution_bank_bytes,
        276_627_456
    );
    assert_eq!(info.kv_bytes, 58_720_256);
    assert_eq!(info.logits_bytes, 63_569_920);
    assert_eq!(info.scratch_bytes, 1_048_576);
    o.memory_limit_bytes = info.admitted_bytes - 1;
    assert!(matches!(
        State::planned(&p, o, 5_050_626_048, 1_048_576),
        Err(Error::MemoryBudget { .. })
    ));
    o.capacity = 513;
    assert!(State::planned(&p, o, 0, 0).is_err());
    assert!(State::planned(&p, options(), usize::MAX, 0).is_err());
}

#[test]
fn second_pass_error_leaves_committed_banks_logits_and_kv_prefix_retriable() {
    let mut state = tiny();
    let mut reference = tiny();
    let cancel = AtomicBool::new(false);
    for owner in [&mut state, &mut reference] {
        owner
            .evaluate(&[2], &cancel, |t, f, l| one(t, f, l, None))
            .unwrap();
    }
    // Physical layer 4 and 7 each execute twice but own different slots.
    assert_eq!(
        state.visits[4].physical_layer,
        state.visits[28].physical_layer
    );
    let [Owner::Linear(first), Owner::Linear(second)] = [state.owners[4], state.owners[28]] else {
        panic!("linear owner");
    };
    assert_ne!(
        state.delta[first].rnn.committed(),
        state.delta[second].rnn.committed()
    );
    assert_ne!(
        state.delta[first].history.committed(),
        state.delta[second].history.committed()
    );
    let [Owner::Full(first), Owner::Full(second)] = [state.owners[7], state.owners[31]] else {
        panic!("full owner");
    };
    assert_ne!(&state.kv[first].k[..2], &state.kv[second].k[..2]);
    assert_ne!(&state.kv[first].v[..2], &state.kv[second].v[..2]);

    let saved = state.logits().unwrap().to_vec();
    assert!(
        state
            .evaluate(&[3, 4], &cancel, |t, f, l| one(
                t,
                f,
                l,
                (t.position == 2).then_some(35)
            ))
            .is_err()
    );
    assert_eq!(state.history(), &[2]);
    assert_eq!(state.logits().unwrap(), saved);
    for (x, y) in state.delta.iter().zip(&reference.delta) {
        assert_eq!(x.rnn.committed(), y.rnn.committed());
        assert_eq!(x.history.committed(), y.history.committed());
    }
    for (x, y) in state.kv.iter().zip(&reference.kv) {
        for head in 0..2 {
            let prefix = head * 8..head * 8 + 2;
            assert_eq!(x.k[prefix.clone()], y.k[prefix.clone()]);
            assert_eq!(x.v[prefix.clone()], y.v[prefix]);
        }
    }

    let chunk = state
        .evaluate(&[3, 4], &cancel, |t, f, l| one(t, f, l, None))
        .unwrap()
        .to_vec();
    let first = reference
        .evaluate(&[3], &cancel, |t, f, l| one(t, f, l, None))
        .unwrap()
        .to_vec();
    let second = reference
        .evaluate(&[4], &cancel, |t, f, l| one(t, f, l, None))
        .unwrap();
    assert_eq!(chunk, [first.as_slice(), second].concat());
    for (x, y) in state.delta.iter().zip(&reference.delta) {
        assert_eq!(x.rnn.committed(), y.rnn.committed());
        assert_eq!(x.history.committed(), y.history.committed());
    }
    assert_eq!(state.reuse_prefix(&[2, 3, 4, 0]).unwrap(), 3);
    assert_eq!(state.reuse_prefix(&[2, 0]).unwrap(), 0);
    assert!(state.logits().is_none());
    assert!(state.delta.iter().all(|x| {
        x.rnn
            .committed()
            .iter()
            .chain(x.history.committed())
            .all(|&v| v == 0.0)
    }));
    assert_eq!(
        state
            .evaluate(&[2], &cancel, |t, f, l| one(t, f, l, None))
            .unwrap(),
        saved
    );
}

#[test]
fn cancellation_incomplete_and_invalid_chunks_never_publish() {
    let mut state = tiny();
    let cancel = AtomicBool::new(false);
    assert!(state.evaluate(&[1], &cancel, |_, _, _| Ok(())).is_err());
    let outcome = state.evaluate(&[1], &cancel, |t, f, l| {
        assert!(f.next_visit()?.is_some());
        cancel.store(true, Ordering::Relaxed);
        one(t, f, l, None)
    });
    assert!(matches!(outcome, Err(Error::Cancelled)));
    cancel.store(false, Ordering::Relaxed);
    assert!(
        state
            .evaluate(&[5], &cancel, |_, _, _| panic!(
                "invalid token entered frame"
            ))
            .is_err()
    );
    assert!(state.history().is_empty() && state.logits().is_none());
    state
        .evaluate(&[1], &cancel, |t, f, l| one(t, f, l, None))
        .unwrap();
    assert!(state.reuse_prefix(&[5]).is_err());
    assert_eq!(state.history(), &[1]);
}
