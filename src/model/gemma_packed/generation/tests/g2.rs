//! Exact Mac official BF16 parity and paired timing for the last-row candidate.
use crate::{
    generation::{self, Input, Request},
    metal::gemma_affine::{Bf16GenerationSession, OfficialBf16},
    session::Options,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{sync::atomic::AtomicBool, time::Instant};

const PROCESS: usize = crate::weights::INITIAL_MEMORY_LIMIT_BYTES;

fn emit(event: &str, data: serde_json::Value) {
    println!("GEMMA_{event} {data}");
}

fn inputs() -> Vec<u32> {
    let path = std::env::var_os("TOKLET_GEMMA_G2_INPUTS").expect("explicit frozen loss IDs");
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(bytes.len(), 40_004);
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        "877cbe18574b47e4b6a8d1b58fb79621c37e0531bd1f9504e50ce63d186f8c46"
    );
    bytes[..512 * 4]
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

fn check_bits(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (index, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "bit mismatch at {index}");
    }
}

fn options(chunk: usize) -> Options {
    Options {
        capacity: 512,
        prefill_chunk_size: chunk,
        position_offset: 0,
        memory_limit_bytes: PROCESS,
    }
}

fn load() -> OfficialBf16 {
    let directory = std::env::var_os("TOKLET_GEMMA_OFFICIAL_DIRECTORY").expect("verified source");
    OfficialBf16::load(std::path::Path::new(&directory), 512, PROCESS).unwrap()
}

#[test]
#[ignore = "explicit real Mac official BF16 last-row parity"]
fn official_bf16_last_row_matches_complete_prefill() {
    let tokens = inputs();
    let owner = load();
    let model = owner.model().unwrap();
    for (chunk, length) in [(1, 33), (3, 65), (32, 512)] {
        let mut full = owner.new_session(&model, options(chunk)).unwrap();
        let mut last = owner.new_session(&model, options(chunk)).unwrap();
        full.append(&tokens[..length]).unwrap();
        last.append(&tokens[..length]).unwrap();
        while full.pending_len() > 0 {
            assert_eq!(
                full.evaluate_next_chunk().unwrap(),
                last.evaluate_next_chunk_last_row().unwrap()
            );
            assert_eq!(full.inner.tokens(), last.inner.tokens());
            let width = full.vocab_size();
            let previous = full.inner.logits();
            check_bits(&previous[previous.len() - width..], last.inner.logits());
            for (old, new) in full.inner.caches.iter().zip(&last.inner.caches) {
                assert_eq!(
                    (old.offset(), old.generation()),
                    (new.offset(), new.generation())
                );
                let (ok, ov) = old.committed();
                let (nk, nv) = new.committed();
                check_bits(ok, nk);
                check_bits(ov, nv);
            }
        }
        assert_eq!(full.history(), last.history());
        emit(
            "G2_PARITY",
            json!({"chunk":chunk,"positions":length,"kv_bytes":last.memory().kv_bytes}),
        );
    }
}

fn run(
    session: &mut Bf16GenerationSession<'_, '_, '_>,
    prompt: &[u32],
    output: usize,
    last: bool,
) -> ([u64; 3], Vec<u32>) {
    let mut ids = Vec::with_capacity(output);
    session.reset();
    let start = Instant::now();
    session.set_tokens(prompt).unwrap();
    while session.pending_len() > 0 {
        if last {
            session.evaluate_next_chunk_last_row().unwrap();
        } else {
            session.evaluate_next_chunk().unwrap();
        }
    }
    let prefill = start.elapsed();
    let report = generation::generate(
        session,
        Request {
            input: Input::Continue,
            max_new_tokens: output,
            stop_tokens: &[],
        },
        &AtomicBool::new(false),
        |event| {
            ids.push(event.token_id);
            Ok(())
        },
    )
    .unwrap();
    let elapsed = start.elapsed();
    assert_eq!(ids.len(), output);
    assert_eq!(report.history_tokens, prompt.len() + output);
    (
        [
            prefill.as_nanos() as u64,
            (report.decode_seconds * 1e9) as u64,
            elapsed.as_nanos() as u64,
        ],
        ids,
    )
}

#[test]
#[ignore = "explicit paired release Mac timing; not a default test"]
fn official_bf16_last_row_paired_prefill_measurement() {
    let tokens = inputs();
    let owner = load();
    let model = owner.model().unwrap();
    let mut session = owner.new_session(&model, options(32)).unwrap();
    let reversed = std::env::var("TOKLET_GEMMA_G2_ORDER").is_ok_and(|value| value == "last-first");
    for (prompt, output) in [(384, 128), (128, 128), (511, 1)] {
        for sample in 0..13 {
            let last_first = reversed ^ (sample % 2 == 1);
            let (first, first_ids) = run(&mut session, &tokens[..prompt], output, last_first);
            let (second, second_ids) = run(&mut session, &tokens[..prompt], output, !last_first);
            assert_eq!(first_ids, second_ids);
            let digest = Sha256::digest(
                first_ids
                    .iter()
                    .flat_map(|id| id.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            let (full, last) = if last_first {
                (second, first)
            } else {
                (first, second)
            };
            emit(
                "G2_SAMPLE",
                json!({"prompt":prompt,"output":output,"sample":sample,"warmup":sample < 3,
                "first":if last_first {"last"} else {"full"},"full_ns":full,"last_ns":last,"ids_sha256":format!("{digest:x}")}),
            );
        }
    }
}
