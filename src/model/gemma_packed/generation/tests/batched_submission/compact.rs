//! Original 384/128 replay, with counters and raw-state hashes outside timing.
use super::{Bf16GenerationSession, Bf16GraphFault, admitted, load, options};
use crate::{sampling::greedy, session::InferenceSession};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{cell::RefCell, fs, path::PathBuf};

const PROMPT: usize = 384;
const EVALUATIONS: usize = 127;
const HISTORY_SHA: &str = "3377caf4f5020d8b5419ea10bc38fa314ced2355764f1fb75a3c528198960984";

#[derive(Default)]
struct Attention {
    hash: Sha256,
    calls: usize,
    words: usize,
}

thread_local! {
    static ATTENTION: RefCell<Option<Attention>> = const { RefCell::new(None) };
}

fn words(hash: &mut Sha256, values: &[f32]) {
    let mut bytes = [0; 4096];
    for chunk in values.chunks(bytes.len() / 4) {
        for (to, word) in bytes.chunks_exact_mut(4).zip(chunk) {
            to.copy_from_slice(&word.to_bits().to_le_bytes());
        }
        hash.update(&bytes[..chunk.len() * 4]);
    }
}

pub(in crate::model::gemma_packed) fn observe_attention(
    layer: usize,
    start: usize,
    rows: usize,
    scores: &[f32],
    probabilities: &[f32],
    context: &[f32],
) {
    ATTENTION.with_borrow_mut(|option| {
        if let Some(plane) = option {
            for n in [
                layer,
                start,
                rows,
                scores.len(),
                probabilities.len(),
                context.len(),
            ] {
                plane.hash.update(n.to_le_bytes());
            }
            for values in [scores, probabilities, context] {
                words(&mut plane.hash, values);
                plane.words += values.len();
            }
            plane.calls += 1;
        }
    });
}

fn hex(hash: Sha256) -> String {
    format!("{:x}", hash.finalize())
}

fn physical(session: &Bf16GenerationSession<'_, '_, '_>) -> String {
    let mut hash = Sha256::new();
    assert_eq!(session.inner.caches.len(), 15);
    for cache in &session.inner.caches {
        assert_eq!(cache.offset(), session.evaluated_len());
        hash.update(cache.offset().to_le_bytes());
        hash.update(cache.generation().to_le_bytes());
        let (keys, values) = cache.physical_words();
        hash.update(keys.len().to_le_bytes());
        hash.update(values.len().to_le_bytes());
        words(&mut hash, keys);
        words(&mut hash, values);
    }
    hex(hash)
}

fn head(session: &Bf16GenerationSession<'_, '_, '_>) -> String {
    let mut hash = Sha256::new();
    let values = session.inner.logits();
    assert_eq!(values.len(), session.vocab_size());
    words(&mut hash, values);
    hex(hash)
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct Record {
    committed: usize,
    pending: usize,
    head: String,
    physical_kv: String,
    attention: String,
    attention_calls: usize,
    attention_words: usize,
    next_id: Option<u32>,
}

fn step(
    session: &mut Bf16GenerationSession<'_, '_, '_>,
    tape: &[u32],
    rows: usize,
    decode: bool,
) -> Record {
    let before = session.evaluated_len();
    ATTENTION.with_borrow_mut(|capture| *capture = Some(Attention::default()));
    let count = if decode {
        session.evaluate_next_decode_chunk()
    } else {
        session.evaluate_next_chunk()
    };
    assert_eq!(count.unwrap(), rows);
    let attention = ATTENTION.with_borrow_mut(|capture| capture.take().unwrap());
    let committed = session.evaluated_len();
    assert_eq!(committed, before + rows);
    assert_eq!(session.inner.tokens(), &tape[..committed]);
    assert_eq!(session.history(), &tape[..session.history().len()]);
    assert_eq!(attention.calls, 35);
    let next_id = session.logits().map(|row| greedy(row).unwrap());
    if committed >= PROMPT
        && let Some(next_id) = next_id
    {
        assert_eq!(next_id, tape[committed]);
    }
    Record {
        committed,
        pending: session.pending_len(),
        head: head(session),
        physical_kv: physical(session),
        attention: hex(attention.hash),
        attention_calls: attention.calls,
        attention_words: attention.words,
        next_id,
    }
}

fn assert_fault_prefix(actual: &Record, full_prompt: &Record) {
    // The fault fixture queues only32 prompt tokens. It shares the exact
    // committed numerical state with the first cut of the 384-token request.
    assert_eq!((actual.committed, actual.pending), (32, 0));
    assert_eq!((full_prompt.committed, full_prompt.pending), (32, 352));
    assert!(actual.next_id.is_some());
    assert!(full_prompt.next_id.is_none());
    assert_eq!(actual.head, full_prompt.head);
    assert_eq!(actual.physical_kv, full_prompt.physical_kv);
    assert_eq!(actual.attention, full_prompt.attention);
    assert_eq!(actual.attention_calls, full_prompt.attention_calls);
    assert_eq!(actual.attention_words, full_prompt.attention_words);
}

#[test]
#[ignore = "requires original official Gemma BF16 model and exclusive admitted Apple GPU"]
fn real_compact_decode_384_128_exact_and_fault_retry() {
    let bytes = fs::read(std::env::var_os("TOKLET_GEMMA_T1_TAPE").unwrap()).unwrap();
    assert_eq!(format!("{:x}", Sha256::digest(&bytes)), HISTORY_SHA);
    let tape: Vec<_> = bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect();
    assert_eq!(tape.len(), 512);
    let owner = load();
    let model = owner.model().unwrap();
    let make_session = |compact, chunk| {
        if compact {
            owner.new_compact_tiled_g2_session(&model, options(chunk))
        } else {
            owner.new_tiled_g2_session(&model, options(chunk), true)
        }
        .unwrap()
    };
    let mut records = Vec::new();
    let mut work = Vec::new();
    for compact in [false, true] {
        let mut session = make_session(compact, 32);
        admitted(&owner, &session);
        session.append(&tape[..PROMPT]).unwrap();
        for cut in 0..12 {
            let record = step(&mut session, &tape, 32, false);
            if compact {
                assert_eq!(record, records[cut], "prefill cut {cut}");
            } else {
                records.push(record);
            }
        }
        let prefill = session.operations().compact_test_work();
        assert_eq!(&prefill[2..], &[0, 0]);
        for index in 0..EVALUATIONS {
            let sampled = greedy(session.logits().unwrap()).unwrap();
            assert_eq!(sampled, tape[PROMPT + index]);
            session.append(&[sampled]).unwrap();
            let before = session.operations().compact_test_work();
            let record = step(&mut session, &tape, 1, true);
            let after = session.operations().compact_test_work();
            let delta = std::array::from_fn(|i| after[i] - before[i]);
            assert_eq!(
                delta,
                if compact {
                    [349, 277, 276, 276]
                } else {
                    [349, 277, 0, 0]
                }
            );
            if compact {
                assert_eq!(record, records[index + 12], "decode target {index}");
            } else {
                records.push(record);
            }
        }
        let sampled = greedy(session.logits().unwrap()).unwrap();
        assert_eq!(sampled, tape[511]);
        session.append(&[sampled]).unwrap();
        assert_eq!(session.history(), tape);
        assert_eq!((session.evaluated_len(), session.pending_len()), (511, 1));
        assert!(session.logits().is_none());
        work.push(serde_json::json!({
            "compact":compact, "prefill_work":prefill,
            "terminal_work":session.operations().compact_test_work(),
            "admitted_bytes":session.memory().admitted_bytes,
            "owned_with_caller_and_helper_bytes":
                session.memory().admitted_bytes + owner.memory().session_reserved_bytes
                + super::CALLER + super::HELPER,
        }));
    }
    assert_eq!(work[0]["prefill_work"], work[1]["prefill_work"]);
    assert_eq!(
        work[1]["admitted_bytes"].as_u64().unwrap() - work[0]["admitted_bytes"].as_u64().unwrap(),
        98_360
    );
    eprintln!("COMPACT_EXACT 139 checkpoints 128 IDs; 35052 selections per compact buffer");

    let mut small_chunk = Vec::new();
    let mut first_prefill = None;
    let mut first_decode = None;
    for compact in [false, true] {
        let mut session = make_session(compact, 1);
        admitted(&owner, &session);
        session.append(&tape[..1]).unwrap();
        let prefill = step(&mut session, &tape, 1, false);
        let before = session.operations().compact_test_work();
        assert_eq!(&before[2..], &[0, 0]);
        session.append(&tape[1..2]).unwrap();
        let decode = step(&mut session, &tape, 1, true);
        let after = session.operations().compact_test_work();
        let delta: [usize; 4] = std::array::from_fn(|i| after[i] - before[i]);
        if compact {
            assert_eq!(first_prefill.as_ref(), Some(&prefill));
            assert_eq!(first_decode.as_ref(), Some(&decode));
            assert_eq!(delta, [349, 349, 317, 317]);
        } else {
            first_prefill = Some(prefill);
            first_decode = Some(decode);
            assert_eq!(delta, [349, 349, 0, 0]);
        }
        small_chunk.push(delta);
    }

    let mut control = make_session(false, 32);
    admitted(&owner, &control);
    control.append(&tape[..32]).unwrap();
    assert_fault_prefix(&step(&mut control, &tape, 32, false), &records[0]);
    control.append(&tape[32..33]).unwrap();
    assert_eq!(control.evaluate_next_decode_chunk().unwrap(), 1);
    let clean = (physical(&control), head(&control));
    drop(control);
    let mut faults = Vec::new();
    for (fault, name, expected) in [
        (
            Bf16GraphFault::CompactInputGuard,
            "compact-input-guard",
            "guard/inactive capacity changed",
        ),
        (
            Bf16GraphFault::CompactInactiveOutput,
            "compact-inactive-output",
            "guard/inactive capacity changed",
        ),
        (Bf16GraphFault::CompactStatus, "compact-status", "nonfinite"),
        (
            Bf16GraphFault::AfterFirstHeadTile,
            "original-late-head",
            "private-copy failure",
        ),
    ] {
        {
            let mut session = make_session(true, 32);
            admitted(&owner, &session);
            session.append(&tape[..32]).unwrap();
            let cut = step(&mut session, &tape, 32, false);
            assert_fault_prefix(&cut, &records[0]);
            session.append(&tape[32..33]).unwrap();
            session.operations_mut().set_fault(fault);
            let cause = session
                .evaluate_next_decode_chunk()
                .unwrap_err()
                .to_string();
            assert!(cause.contains(expected), "{name}: {cause}");
            assert_eq!(session.history(), &tape[..33]);
            assert_eq!(session.inner.tokens(), &tape[..32]);
            assert_eq!((session.evaluated_len(), session.pending_len()), (32, 1));
            assert_eq!(physical(&session), cut.physical_kv);
            assert_eq!(head(&session), cut.head);
            assert!(session.logits().is_none());
            session.reset();
            session.append(&tape[..32]).unwrap();
            assert!(
                session
                    .evaluate_next_chunk()
                    .unwrap_err()
                    .to_string()
                    .contains("backend poisoned; recreate")
            );
            faults.push(serde_json::json!({"fault":name,"error":cause}));
        }
        let mut session = make_session(true, 32);
        admitted(&owner, &session);
        session.append(&tape[..32]).unwrap();
        assert_fault_prefix(&step(&mut session, &tape, 32, false), &records[0]);
        session.append(&tape[32..33]).unwrap();
        assert_eq!(session.evaluate_next_decode_chunk().unwrap(), 1);
        assert_eq!(session.inner.tokens(), &tape[..33]);
        assert_eq!(clean, (physical(&session), head(&session)));
    }
    let receipt = serde_json::json!({
        "format":"toklet-gemma-t1-compact-exact-v1", "tape_sha256":HISTORY_SHA,
        "prompt":PROMPT, "ids":&tape[PROMPT..], "evaluate_count":EVALUATIONS,
        "terminal":[511,1], "cut_and_decode_rows":records, "work":work,
        "faults":faults,"clean_recreation":clean,"chunk_one_work":small_chunk,
    });
    fs::write(
        PathBuf::from(std::env::var_os("TOKLET_GEMMA_T1_OUT").unwrap()),
        serde_json::to_vec_pretty(&receipt).unwrap(),
    )
    .unwrap();
}
