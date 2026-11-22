// Included as a test-only child of the exact Nanbeige session module.
use super::*;
use crate::metal::memory::system_device;
use objc2_metal::{MTLDevice, MTLResource};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

fn options() -> Options {
    Options {
        capacity: 130,
        prefill_chunk_size: 128,
        ..Options::default()
    }
}

fn input() -> (PathBuf, Vec<u32>) {
    let model = std::env::var_os("TOKLET_NANBEIGE_MODEL").expect("pinned local model path");
    let tape = std::fs::read(std::env::var_os("TOKLET_NANBEIGE_TAPE").expect("pinned tape path"))
        .expect("sealed tape bytes");
    assert_eq!(
        format!("{:x}", Sha256::digest(&tape)),
        "0964b3b9856d0df51f6684010aa3bf24c0db37b048755365911201bfc66c0c28"
    );
    let tokens = tape[..130 * 4]
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect();
    (PathBuf::from(model), tokens)
}

#[derive(Debug, PartialEq, Eq)]
struct Published {
    rows: usize,
    greedy: u32,
    k: Vec<[u8; 32]>,
    v: Vec<[u8; 32]>,
    logits: [u8; 32],
}

fn published(session: &Session<'_>) -> Published {
    let rows = session.evaluated_len();
    let mut key = vec![0; rows * KV_WIDTH];
    let mut value = vec![0; rows * KV_WIDTH];
    let mut k = Vec::with_capacity(plan::VISITS);
    let mut v = Vec::with_capacity(plan::VISITS);
    for slot in 0..plan::VISITS {
        session.copy_kv_bf16(slot, &mut key, &mut value).unwrap();
        for (panel, receipts) in [(&key, &mut k), (&value, &mut v)] {
            let mut hash = Sha256::new();
            for word in panel {
                hash.update(word.to_le_bytes());
            }
            receipts.push(hash.finalize().into());
        }
    }
    let logits = session.logits().expect("committed diagnostic logits");
    assert_eq!(logits.len(), LOGIT_ROWS);
    let mut hash = Sha256::new();
    for word in logits {
        assert!(word.is_finite());
        hash.update(word.to_bits().to_le_bytes());
    }
    Published {
        rows,
        greedy: session.greedy_token().unwrap().unwrap(),
        k,
        v,
        logits: hash.finalize().into(),
    }
}

fn commit_prefix(session: &mut Session<'_>, tape: &[u32]) {
    session.set_tokens(&tape[..1]).unwrap();
    assert_eq!(session.evaluate_next_chunk().unwrap(), 1);
    assert_eq!(session.last_chunk_used_dense(), Some(false));
    session.append(&tape[1..]).unwrap();
    assert_eq!((session.evaluated_len(), session.pending_len()), (1, 129));
}

fn waited_dense_without_publication(session: &mut Session<'_>) {
    let start = session.evaluated;
    assert_eq!(start, 1);
    assert_eq!(session.pending_len(), 129);
    assert!(session.fused_gate_up_ready);
    session.scratch.tokens.with_mut(|words| {
        words[..128].copy_from_slice(&session.history[start..start + 128]);
    });
    session.scratch.status.with_mut(|words| words[0] = 0);
    autoreleasepool(|_| session.submit(128, start)).unwrap();
    assert_eq!(session.scratch.status.with_slice(|words| words[0]), 0);
    assert!(
        session
            .diagnostic
            .as_ref()
            .expect("actual route witness")
            .staged_dense
            .get()
    );
}

#[test]
#[ignore = "Explicit local Nanbeige model and Apple M1 Pro GPU"]
fn dense_waited_cancellation_and_status_keep_published_prefix() {
    let (directory, tape) = input();
    let cancelled = AtomicBool::new(false);
    let model = Model::load_dense(&directory, options(), true).unwrap();
    assert_eq!(model.dense_prefill_receipt().unwrap().physical_pairs, 22);

    let mut session = Session::new(&model, options(), true, &cancelled).unwrap();
    commit_prefix(&mut session, &tape);
    let prefix = published(&session);
    waited_dense_without_publication(&mut session);
    cancelled.store(true, Ordering::Relaxed);
    assert!(matches!(
        session.finish_chunk(128, Ok(())),
        Err(SessionError::Model(crate::model::Error::Cancelled))
    ));
    cancelled.store(false, Ordering::Relaxed);
    assert_eq!(session.last_chunk_used_dense(), Some(false));
    assert_eq!(session.pending_len(), 129);
    assert_eq!(session.history(), tape);
    assert_eq!(published(&session), prefix);

    waited_dense_without_publication(&mut session);
    session.scratch.status.with_mut(|words| words[0] = 1);
    assert!(matches!(
        session.finish_chunk(128, Ok(())),
        Err(SessionError::Backend(message)) if message == Error::Nonfinite.to_string()
    ));
    assert_eq!(session.last_chunk_used_dense(), Some(false));
    assert_eq!(session.pending_len(), 129);
    assert_eq!(session.history(), tape);
    assert_eq!(published(&session), prefix);

    assert_eq!(session.evaluate_next_chunk().unwrap(), 128);
    assert_eq!(session.last_chunk_used_dense(), Some(true));
    let resumed = published(&session);
    assert_eq!(session.evaluate_next_chunk().unwrap(), 1);
    assert_eq!(session.last_chunk_used_dense(), Some(false));
    let terminal = published(&session);
    drop(session);

    let mut fresh = Session::new(&model, options(), true, &cancelled).unwrap();
    commit_prefix(&mut fresh, &tape);
    assert_eq!(published(&fresh), prefix);
    assert_eq!(fresh.evaluate_next_chunk().unwrap(), 128);
    assert_eq!(fresh.last_chunk_used_dense(), Some(true));
    assert_eq!(published(&fresh), resumed);
    assert_eq!(fresh.evaluate_next_chunk().unwrap(), 1);
    assert_eq!(fresh.last_chunk_used_dense(), Some(false));
    assert_eq!(published(&fresh), terminal);
    println!(
        "dense_fault_routes packed,dense_attempt_cancel,dense_attempt_status,dense_retry,packed; exact 44 K/V and logits at 1,129,130"
    );
}

#[test]
#[ignore = "Explicit local Nanbeige model and Apple M1 Pro GPU"]
fn poisoned_signed_startup_releases_temporary_owners_and_stays_packed() {
    let (directory, tape) = input();
    let cancelled = AtomicBool::new(false);
    let device = system_device().unwrap();
    let before = device.currentAllocatedSize();
    super::super::dense_basis::poison_next_startup();
    let model = Model::load_dense(&directory, options(), true).unwrap();
    assert!(!super::super::dense_basis::startup_poison_pending());
    assert_eq!(
        model.dense_prefill_denial(),
        Some("Nanbeige dense signed startup parity")
    );
    assert!(model.dense_prefill_receipt().is_none() && model.dense.is_none());
    let packed_bytes: usize = model
        .weights
        .iter()
        .map(|bank| MTLResource::allocatedSize(&*bank.raw))
        .sum();
    let after = device.currentAllocatedSize();
    let overhead = after.saturating_sub(before + packed_bytes);
    assert!(overhead < HIDDEN * INTERMEDIATE * 4);
    let mut packed = Session::new(&model, options(), true, &cancelled).unwrap();
    assert_eq!(
        packed.memory().weights_bytes,
        crate::mlx::nanbeige::WEIGHT_BYTES
    );
    assert!(packed.dense_gate_up_prefill_rows().is_empty());
    commit_prefix(&mut packed, &tape);
    assert_eq!(packed.evaluate_next_chunk().unwrap(), 128);
    assert_eq!(packed.last_chunk_used_dense(), Some(false));
    assert_eq!(packed.evaluate_next_chunk().unwrap(), 1);
    let terminal = published(&packed);
    drop(packed);
    drop(model);

    let release = Model::load(&directory, options(), true).unwrap();
    let mut fresh = Session::new(&release, options(), true, &cancelled).unwrap();
    commit_prefix(&mut fresh, &tape);
    assert_eq!(fresh.evaluate_next_chunk().unwrap(), 128);
    assert_eq!(fresh.evaluate_next_chunk().unwrap(), 1);
    assert_eq!(published(&fresh), terminal);
    println!(
        "startup_poison consumed; pairs=0; packed resource bytes={packed_bytes}; unexplained_device_bytes={overhead}; exact packed terminal"
    );
}
