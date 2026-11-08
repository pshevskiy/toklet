use super::*;
use crate::{
    generation::{self, Input, Request, StopReason},
    session::{InferenceSession, Options},
};
use sha2::{Digest, Sha256};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
};
const LIMIT: usize = crate::weights::INITIAL_MEMORY_LIMIT_BYTES;
const CALLER: usize = 64 * 1024 * 1024;
const PROMPT: [u32; 15] = [
    2, 105, 2364, 107, 1567, 506, 5279, 529, 7001, 236761, 106, 107, 105, 4368, 107,
];
mod bench;
fn opts() -> Options {
    Options {
        capacity: 512,
        prefill_chunk_size: 32,
        position_offset: 0,
        memory_limit_bytes: LIMIT - CALLER,
    }
}
fn digest(row: &[f32]) -> String {
    let mut h = Sha256::new();
    for &v in row {
        assert!(v.is_finite());
        h.update(v.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn greedy(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 { (i, v) } else { best }
        })
        .0 as u32
}
fn footprint() -> serde_json::Value {
    let mut info = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
    // SAFETY: flavor4 matches this owned zeroed writable Apple structure, with
    // no concurrent references. Success populates the fields before reading.
    assert_eq!(
        unsafe {
            libc::proc_pid_rusage(
                libc::getpid(),
                libc::RUSAGE_INFO_V4,
                info.as_mut_ptr().cast(),
            )
        },
        0
    );
    // SAFETY: the successful flavor4 copy populated the zeroed structure.
    let info = unsafe { info.assume_init() };
    assert!(
        info.ri_phys_footprint > 0
            && info.ri_phys_footprint <= info.ri_lifetime_max_phys_footprint
            && info.ri_lifetime_max_phys_footprint <= LIMIT as u64
    );
    serde_json::json!({"physical_bytes":info.ri_phys_footprint,"lifetime_peak_bytes":info.ri_lifetime_max_phys_footprint})
}
fn load(context: &HereticContext) -> HereticWeights {
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
    let dir = PathBuf::from(std::env::var_os("TOKLET_GEMMA_HERETIC_DIRECTORY").unwrap());
    let path = dir.join("model.safetensors");
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(&path)
        .unwrap();
    let a = file.metadata().unwrap();
    assert_eq!(a.len(), uncensored::CONTAINER_BYTES);
    let weights = HereticWeights::load(
        context,
        &std::fs::read(dir.join("config.json")).unwrap(),
        &mut file,
        LIMIT - CALLER,
    )
    .unwrap();
    let b = file.metadata().unwrap();
    let c = std::fs::symlink_metadata(path).unwrap();
    for d in [&b, &c] {
        assert_eq!(
            (a.dev(), a.ino(), a.len(), a.mtime(), a.mtime_nsec()),
            (d.dev(), d.ino(), d.len(), d.mtime(), d.mtime_nsec())
        );
    }
    weights
}

#[test]
fn whole_source_digest_literal_and_six_gib_preflight_are_exact() {
    assert_eq!(
        SHA256
            .iter()
            .map(|v| format!("{v:02x}"))
            .collect::<String>(),
        uncensored::CONTAINER_SHA256
    );
    for budget in [0, LIMIT + 1, usize::MAX] {
        assert!(process_budget(budget).is_err());
        assert!(HereticContext::new(budget).is_err());
    }
}

#[test]
#[ignore = "explicit pinned Heretic MLX4 native load/logit/state/admission controls"]
fn actual_mixed_source_loads_and_preserves_prefix_and_generation_state() {
    let context = HereticContext::new(CONTEXT_BYTES).unwrap();
    let source = load(&context);
    assert_eq!(
        (
            source.report.affine_matrices,
            source.report.bf16_matrices,
            source.report.vectors
        ),
        (277, 1, 262)
    );
    assert_eq!(source.identity.matched_text_components, 1094);
    assert!(source.report.admitted_load_policy_bytes + CALLER <= LIMIT);
    let norms = Norms::new(&source, LIMIT - CALLER).unwrap();
    assert!(norms.model(0).is_err());
    assert!(norms.model(513).is_err());
    let model = norms.model(512).unwrap();
    let mut hashes = Vec::new();
    let first;
    let raw_admission;
    {
        let mut raw = norms
            .new_session(&model, &context, 32, LIMIT - CALLER)
            .unwrap();
        raw_admission = model.model_bytes() + raw.allocated_bytes().unwrap();
        assert!(raw_admission + CALLER <= LIMIT);
        let owned = raw.allocated_bytes().unwrap();
        let all = raw.forward(&PROMPT).unwrap();
        assert_eq!(all.len(), 15 * 262144);
        for row in all.chunks_exact(262144) {
            hashes.push(digest(row));
        }
        first = greedy(&all[14 * 262144..]);
        if let Some(out) = std::env::var_os("TOKLET_HERETIC_CAPTURE_DIRECTORY") {
            use std::io::Write;
            let out = PathBuf::from(out);
            std::fs::create_dir(&out).unwrap();
            let file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(out.join("prompt-last.f32le"))
                .unwrap();
            let mut f = std::io::BufWriter::new(file);
            for v in &all[14 * 262144..] {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
            f.flush().unwrap();
        }
        assert_eq!(raw.tokens(), PROMPT);
        raw.reset().unwrap();
        let mut position = 0;
        for tokens in [&PROMPT[..3], &PROMPT[3..8], &PROMPT[8..]] {
            for row in raw.forward(tokens).unwrap().chunks_exact(262144) {
                assert_eq!(digest(row), hashes[position]);
                position += 1;
            }
            assert_eq!(raw.tokens(), &PROMPT[..position]);
        }
        assert_eq!(raw.allocated_bytes().unwrap(), owned);
        let old = raw.tokens().to_vec();
        assert!(raw.forward(&[262144]).is_err());
        assert_eq!(raw.tokens(), old);
        assert_eq!(raw.allocated_bytes().unwrap(), owned);
    }
    {
        // One evaluation uses277 projections (the two embeddings gather, while
        // the tied embedding also supplies the head). Fail the next frame after
        // its BF16 root and first two affine projections have completed.
        let mut fault = norms
            .faulted_generation_session(&model, &context, opts(), 277 + 3)
            .unwrap();
        fault.set_tokens(&PROMPT[..3]).unwrap();
        assert_eq!(fault.evaluate_next_chunk().unwrap(), 3);
        assert_eq!(digest(fault.logits().unwrap()), hashes[2]);
        fault.append(&PROMPT[3..7]).unwrap();
        assert!(fault.evaluate_next_chunk().is_err());
        assert_eq!(
            (
                fault.history().len(),
                fault.evaluated_len(),
                fault.pending_len()
            ),
            (7, 3, 4)
        );
        assert!(fault.logits().is_none());
        assert!(fault.evaluate_next_chunk().is_err());
        let mut replay = norms
            .new_generation_session(&model, &context, opts())
            .unwrap();
        replay.set_tokens(fault.history()).unwrap();
        assert_eq!(replay.evaluate_next_chunk().unwrap(), 7);
        assert_eq!(digest(replay.logits().unwrap()), hashes[6]);
    }
    let mut session = norms
        .new_generation_session(&model, &context, opts())
        .unwrap();
    let admission = session.memory().admitted_bytes;
    assert!(admission + CALLER <= LIMIT);
    let mut too_small = opts();
    too_small.memory_limit_bytes = admission - 1;
    assert!(
        norms
            .new_generation_session(&model, &context, too_small)
            .is_err()
    );
    let cancelled = AtomicBool::new(true);
    let r = generation::generate(
        &mut session,
        Request {
            input: Input::Reset(&PROMPT),
            max_new_tokens: 1,
            stop_tokens: &[],
        },
        &cancelled,
        |_| panic!("cancelled callback"),
    )
    .unwrap();
    assert_eq!(r.reason, StopReason::Cancelled);
    assert!(session.history().is_empty());
    assert_eq!(session.evaluated_len(), 0);
    session.set_tokens(&PROMPT).unwrap();
    assert_eq!(session.evaluate_next_chunk().unwrap(), 15);
    assert_eq!(digest(session.logits().unwrap()), hashes[14]);
    assert_eq!(session.greedy_token().unwrap().unwrap(), first);
    let mut cached_ids = Vec::new();
    let cached = generation::generate(
        &mut session,
        Request {
            input: Input::Prompt(&PROMPT),
            max_new_tokens: 1,
            stop_tokens: &[],
        },
        &AtomicBool::new(false),
        |e| {
            cached_ids.push(e.token_id);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(cached_ids, [first]);
    assert_eq!(
        (
            cached.reused_tokens,
            cached.prefill_tokens,
            cached.sampled_tokens,
            cached.emitted_tokens
        ),
        (15, 0, 1, 1)
    );
    assert_eq!(
        (
            session.history().len(),
            session.evaluated_len(),
            session.pending_len()
        ),
        (16, 15, 1)
    );
    assert!(session.logits().is_none());
    let failed = generation::generate(
        &mut session,
        Request {
            input: Input::Reset(&PROMPT),
            max_new_tokens: 1,
            stop_tokens: &[],
        },
        &AtomicBool::new(false),
        |_| Err(std::io::Error::other("public output sink fixture")),
    );
    assert!(matches!(failed, Err(generation::Error::Output(_))));
    assert_eq!(
        (
            session.history().len(),
            session.evaluated_len(),
            session.pending_len()
        ),
        (16, 15, 1)
    );
    let stop = generation::generate(
        &mut session,
        Request {
            input: Input::Reset(&PROMPT),
            max_new_tokens: 1,
            stop_tokens: &[first],
        },
        &AtomicBool::new(false),
        |_| panic!("hidden stop emitted"),
    )
    .unwrap();
    assert_eq!(stop.reason, StopReason::StopToken { token_id: first });
    assert_eq!((stop.sampled_tokens, stop.emitted_tokens), (1, 0));
    let cancelled = AtomicBool::new(false);
    let partial = generation::generate(
        &mut session,
        Request {
            input: Input::Reset(&PROMPT),
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
    assert_eq!(partial.reason, StopReason::Cancelled);
    assert_eq!((partial.sampled_tokens, partial.emitted_tokens), (1, 1));
    assert_eq!(
        (
            session.history().len(),
            session.evaluated_len(),
            session.pending_len()
        ),
        (16, 15, 1)
    );
    let before = session.history().to_vec();
    let mut full = opts();
    full.capacity = 513;
    assert!(
        norms
            .new_generation_session(&model, &context, full)
            .is_err()
    );
    assert!(session.append(&vec![2; 513]).is_err());
    assert_eq!(session.history(), before);
    println!(
        "HERETIC_CONTROL {}",
        serde_json::json!({"passed":true,"source":source.identity(),"load":source.report(),"context_creation_seconds":context.creation_seconds(),"model_bytes":model.model_bytes(),"norms_bytes":norms.owned_bytes(),"raw_admission_bytes":raw_admission,"generation_memory":session.memory(),"caller_reserve_bytes":CALLER,"prompt_ids":PROMPT,"prompt_row_sha256":hashes,"first_greedy_id":first,"full_quality_qualified":false,"memory":footprint()})
    );
}
