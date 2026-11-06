use super::*;

#[test]
fn strict_draft_rounds_keep_the_accepted_hidden_and_wrapped_kv() {
    use crate::generation::{GreedyVerification, VerifiedNext};
    use crate::sampling::greedy;
    use std::cell::Cell;

    let data = fixture::Fixture::new();
    let model = Model::new(data.weights()).unwrap();
    let mut serial = model.new_generation_session(options()).unwrap();
    let mut draft = model.new_generation_session(options()).unwrap();
    for session in [&mut serial, &mut draft] {
        session.append(&data.ids[..5]).unwrap();
        evaluated(session);
    }
    let hidden = |s: &GenerationSession<'_, '_>| {
        let end = s.inner.published_rows * model.layout.hidden;
        s.inner.scratch.norm[end - model.layout.hidden..end].to_vec()
    };
    let mut anchor = hidden(&draft);
    let initial_anchor = anchor.clone();
    let initial = snapshot(&draft.inner);
    let first = greedy(serial.logits().unwrap()).unwrap();
    serial.append(&[first]).unwrap();
    evaluated(&mut serial);
    let correction = greedy(serial.logits().unwrap()).unwrap();
    let wrong = (correction + 1) % VOCAB_SIZE as u32;
    let proposals = [first, wrong];
    let checks = Cell::new(0);
    assert!(
        draft
            .verify_greedy_round(&proposals, &[], &mut anchor, || {
                let check = checks.get();
                checks.set(check + 1);
                check == 1
            })
            .unwrap()
            .is_none()
    );
    assert_eq!(snapshot(&draft.inner), initial);
    assert_eq!(draft.pending_len(), 0);
    assert_eq!(anchor, initial_anchor);
    assert_eq!(
        draft
            .verify_greedy_round(&proposals, &[], &mut anchor, || false)
            .unwrap(),
        Some(GreedyVerification {
            accepted: 1,
            next: VerifiedNext::Token(correction)
        })
    );
    assert_eq!(anchor, hidden(&serial));
    assert_eq!(snapshot(&draft.inner), snapshot(&serial.inner));
    assert_eq!(draft.pending_len(), 1);
    assert!(
        draft
            .verify_greedy_round(&[], &[], &mut anchor, || false)
            .is_err()
    );

    serial.append(&[correction]).unwrap();
    evaluated(&mut serial);
    evaluated(&mut draft);
    anchor.copy_from_slice(&hidden(&draft));
    assert_eq!(anchor, hidden(&serial));
    let stop = greedy(serial.logits().unwrap()).unwrap();
    let before_stop = snapshot(&draft.inner);
    let old_anchor = anchor.clone();
    assert_eq!(
        draft
            .verify_greedy_round(&[stop], &[stop], &mut anchor, || false)
            .unwrap(),
        Some(GreedyVerification {
            accepted: 0,
            next: VerifiedNext::Stop(stop)
        })
    );
    assert_eq!(draft.pending_len(), 1);
    assert_eq!(draft.history().last(), Some(&stop));
    assert_eq!(snapshot(&draft.inner), before_stop);
    assert_eq!(anchor, old_anchor);
    draft.discard_pending();

    serial.append(&[stop]).unwrap();
    evaluated(&mut serial);
    let bonus = greedy(serial.logits().unwrap()).unwrap();
    assert_eq!(
        draft
            .verify_greedy_round(&[stop], &[], &mut anchor, || false)
            .unwrap(),
        Some(GreedyVerification {
            accepted: 1,
            next: VerifiedNext::Token(bonus)
        })
    );
    assert_eq!(anchor, hidden(&serial));
    assert_eq!(snapshot(&draft.inner), snapshot(&serial.inner));
    assert_eq!(draft.pending_len(), 1);
}
