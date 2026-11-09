//! Completed 9B hybrid input-forward baselines; checkpoint/request gates are separate.
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};
use toklet::{
    metal::gemma_affine::{CONTEXT_ALLOWANCE_BYTES, Context},
    mlx::qwen35::{
        ModalityPolicy,
        nine_b::{self as source, payload},
    },
    model::qwen35::{MetalModel, MetalSession, Options, Outcome, Work},
};
const LIMIT: usize = 6 * 1024 * 1024 * 1024;
const DIAGNOSTIC: usize = 320 * 1024 * 1024;
#[cfg(toklet_context256)]
const TOKENS: usize = 256;
#[cfg(not(toklet_context256))]
const TOKENS: usize = 128;
const IDS: [u32; 256] = {
    let pattern = [1, 7, 13, 3, 9, 5, 11, 2, 17, 4, 6, 10, 19, 8, 21, 23, 29];
    let mut ids = [0; 256];
    let mut i = 0;
    while i < ids.len() {
        ids[i] = pattern[i % pattern.len()];
        i += 1;
    }
    ids
};
fn config(folder: &Path) -> Vec<u8> {
    let f = File::open(folder.join("config.json")).unwrap();
    assert!(f.metadata().unwrap().is_file());
    assert_eq!(f.metadata().unwrap().len(), source::CONFIG_BYTES as u64);
    let mut bytes = Vec::new();
    f.take(source::CONFIG_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .unwrap();
    bytes
}
fn digest(values: &[f32]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        assert!(value.is_finite());
        hash.update(value.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}
fn expected_work(tokens: usize) -> Work {
    Work {
        outcome: Outcome::Success,
        tokens_completed: tokens,
        tokens_committed: tokens,
        projections_completed: tokens * 249,
        gathers_completed: tokens,
        convolution_steps_completed: tokens * 24,
        recurrence_steps_completed: tokens * 24,
        attention_heads_completed: tokens * 128,
    }
}
fn measure(session: &mut MetalSession<'_, '_, '_>, vocab: usize, output: &Path) {
    // One bounded diagnostic owner. Full rows are checked on every replay;
    // complete digests and the final raw row are persisted outside all timers.
    let mut expected = Vec::with_capacity(TOKENS.checked_mul(vocab).unwrap());
    assert_eq!(expected.capacity() * 4, TOKENS * vocab * 4);
    assert!(expected.capacity() * 4 + 65536 < DIAGNOSTIC);
    for ordinal in 0..13 {
        session.reset();
        let mut ns = 0u128;
        let mut token = 0;
        let mut calls = Vec::with_capacity(8);
        let mut total = Work {
            outcome: Outcome::Success,
            ..Work::default()
        };
        for input in IDS[..TOKENS].chunks(32) {
            let start = Instant::now();
            let values = session.forward(input).unwrap();
            ns += start.elapsed().as_nanos();
            assert_eq!(values.len(), input.len() * vocab);
            if ordinal == 0 {
                expected.extend_from_slice(values);
            } else {
                assert!(
                    values
                        .iter()
                        .zip(&expected[token * vocab..(token + input.len()) * vocab])
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            }
            let w = session.work();
            assert_eq!(w, expected_work(input.len()));
            total.tokens_completed += w.tokens_completed;
            total.tokens_committed += w.tokens_committed;
            total.projections_completed += w.projections_completed;
            total.gathers_completed += w.gathers_completed;
            total.convolution_steps_completed += w.convolution_steps_completed;
            total.recurrence_steps_completed += w.recurrence_steps_completed;
            total.attention_heads_completed += w.attention_heads_completed;
            calls.push(input.len());
            token += input.len();
        }
        assert_eq!(token, TOKENS);
        assert_eq!(expected.len(), TOKENS * vocab);
        assert_eq!(session.history(), &IDS[..TOKENS]);
        assert_eq!(total, expected_work(TOKENS));
        let memory = session.memory();
        assert!(memory.admitted_bytes + DIAGNOSTIC <= LIMIT);
        println!(
            "NINE_B_CONTEXT_SAMPLE {}",
            serde_json::json!({"ordinal":ordinal,"warmup":ordinal<3,
            "forward_ns":ns,"forward_calls":calls,"output_values":TOKENS*vocab,"memory":memory,"work":total,
            "reference_created":ordinal==0,"repeated_logits_bits_equal":ordinal>0,
            "diagnostic_allowance_bytes":DIAGNOSTIC,"numerical_adoption_qualified":false,"managed_adoption_qualified":false})
        );
    }
    let row = &expected[(TOKENS - 1) * vocab..];
    let mut writer = BufWriter::with_capacity(65536, File::create(output).unwrap());
    for value in row {
        writer.write_all(&value.to_le_bytes()).unwrap();
    }
    writer.flush().unwrap();
    writer.get_ref().sync_all().unwrap();
    println!(
        "NINE_B_CONTEXT_OUTPUT {}",
        serde_json::json!({"input_tokens":TOKENS,"output_values":expected.len(),
        "all_logits_sha256":digest(&expected),"prefix128_logits_sha256":digest(&expected[..128*vocab]),
        "final_row_sha256":digest(row),"persisted_final_row_bytes":row.len()*4,
        "full_output_replays_bits_equal":true,"full_output_reference_bytes":expected.capacity()*4,
        "timing_scope":"Sum of completed chunk forwards; reset, setup, comparisons, hashing and writes excluded; full logits for every input; no request TTFT or optimization claim"})
    );
}
fn run(folder: &Path, output: &Path) {
    let config = config(folder);
    let start = Instant::now();
    let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
    let context_ns = start.elapsed().as_nanos();
    let start = Instant::now();
    let cpu = payload::load(
        &config,
        File::open(folder.join("model.safetensors")).unwrap(),
        payload::LoadOptions {
            memory_limit_bytes: LIMIT - DIAGNOSTIC - CONTEXT_ALLOWANCE_BYTES,
            modality: ModalityPolicy::TextOnly,
        },
    )
    .unwrap();
    let load_ns = start.elapsed().as_nanos();
    assert_eq!(cpu.identity().container_bytes(), source::CONTAINER_BYTES);
    let start = Instant::now();
    let store = cpu.into_metal(&context, LIMIT - DIAGNOSTIC).unwrap();
    let upload_ns = start.elapsed().as_nanos();
    let report = store.report();
    assert_eq!(report.matrices, 250);
    assert_eq!(report.retained_cpu_matrix_bytes, 0);
    let start = Instant::now();
    let model = MetalModel::from_nine_b_metal_store(&store).unwrap();
    let bind_ns = start.elapsed().as_nanos();
    let start = Instant::now();
    let mut session = model
        .session(
            &context,
            Options {
                capacity: 512,
                chunk: 32,
                position_offset: 37,
                memory_limit_bytes: LIMIT - DIAGNOSTIC,
            },
        )
        .unwrap();
    let session_ns = start.elapsed().as_nanos();
    println!(
        "NINE_B_CONTEXT_INITIAL {}",
        serde_json::json!({"backend":"metal_hybrid","tokens":&IDS[..TOKENS],
        "input_tokens":TOKENS,"position_offset":37,"capacity":512,"chunk":32,"source_sha256":source::CONTAINER_SHA256,
        "config_sha256":source::CONFIG_SHA256,"header_sha256":source::HEADER_SHA256,"source_bytes":source::CONTAINER_BYTES,
        "context_ns":context_ns,"load_ns":load_ns,"upload_ns":upload_ns,"bind_ns":bind_ns,"session_ns":session_ns,"load_report":report,
        "cache_state":"resident weights, empty numerical session after reset each sample",
        "warmups":3,"measured":10,"diagnostic_allowance_bytes":DIAGNOSTIC})
    );
    measure(&mut session, model.config().vocab, output);
}
fn main() {
    let folder = PathBuf::from(std::env::args_os().nth(1).expect("model directory"));
    let output = PathBuf::from(std::env::args_os().nth(2).expect("fresh final-row output"));
    assert!(!output.exists());
    run(&folder, &output);
}
