//! Raw task IDs through the existing generation API; grading stays external.
use super::*;
use crate::{generation, session};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{io::Read, sync::atomic::AtomicBool};

#[derive(Deserialize)]
struct Task {
    id: String,
    category: String,
    prompt_tokens: Vec<u32>,
    max_output_tokens: usize,
    required_capacity: usize,
    stop_ids: Vec<u32>,
}

fn read_tasks() -> (Vec<Task>, std::fs::File, std::path::PathBuf, [u64; 7]) {
    let path = std::path::PathBuf::from(
        std::env::var_os("TOKLET_GEMMA_TASK_INPUTS").expect("explicit frozen task payload"),
    );
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(&path)
        .unwrap();
    let metadata = file.metadata().unwrap();
    let original = identity(&metadata);
    assert!(metadata.is_file());
    assert_eq!(metadata.len(), 188152);
    assert_eq!(
        identity(&std::fs::symlink_metadata(&path).unwrap()),
        original
    );
    let mut body = Vec::new();
    (&mut file).take(188153).read_to_end(&mut body).unwrap();
    assert_eq!(body.len(), 188152);
    assert_eq!(identity(&file.metadata().unwrap()), original);
    let manifest: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../../../../../benchmarks/gemma-quality-v1/manifest.json"
    ))
    .unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&body)),
        manifest["assets"]["tasks.jsonl"]["sha256"]
            .as_str()
            .unwrap()
    );
    let tasks: Vec<Task> = body
        .split(|&b| b == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect();
    assert_eq!(tasks.len(), 100);
    for (index, task) in tasks.iter().enumerate() {
        let (id, category, cap) = if index < 40 {
            (format!("HumanEval/{index}"), "code", 256)
        } else if index < 70 {
            (format!("truthfulqa/row-{}", index - 40), "prose", 32)
        } else {
            (format!("lambada/row-{}", index - 70), "continuation", 16)
        };
        assert_eq!(task.id, id);
        assert_eq!(task.category, category);
        assert_eq!(task.max_output_tokens, cap);
        assert_eq!(task.required_capacity, task.prompt_tokens.len() + cap);
        assert!(!task.prompt_tokens.is_empty() && task.required_capacity <= 567);
        assert!(task.prompt_tokens.iter().all(|&id| id < 262144));
        assert_eq!(task.stop_ids, [1, 106, 50]);
    }
    (tasks, file, path, original)
}

#[test]
#[ignore = "explicit real Gemma three-category generation controller pilot"]
fn existing_generation_pilot() {
    run(true);
}

#[test]
#[ignore = "explicit real Gemma remaining97 generation cases"]
fn existing_generation_remaining_tasks() {
    run(false);
}

fn run(pilot: bool) {
    println!(); // Separate libtest's test-name prefix from structured records.
    let (tasks, inputs, input_path, input_identity) = read_tasks();
    let (source, file, path, original) = load_original();
    let owners = resident_snapshot(&source);
    let norms = hybrid::Norms::new(&source, PROCESS - DIAGNOSTIC).unwrap();
    let model = norms.model_for_context_probe(567).unwrap();
    assert!(model.content_identity().is_none());
    let context = gemma_affine::Context::new(gemma_affine::CONTEXT_ALLOWANCE_BYTES).unwrap();
    let mut session = norms
        .new_generation_session(
            &model,
            &context,
            session::Options {
                capacity: 567,
                prefill_chunk_size: 32,
                memory_limit_bytes: PROCESS - DIAGNOSTIC,
                position_offset: 0,
            },
        )
        .unwrap();
    let admitted = session.memory().admitted_bytes;
    assert!(admitted + DIAGNOSTIC <= PROCESS);
    let cancelled = AtomicBool::new(false);
    let mut count = 0;
    for (index, task) in tasks.iter().enumerate() {
        if [0, 40, 70].contains(&index) != pilot {
            continue;
        }
        session.reset();
        assert!(session.history().is_empty());
        assert_eq!((session.evaluated_len(), session.pending_len()), (0, 0));
        assert!(session.logits().is_none());
        let mut visible = Vec::with_capacity(task.max_output_tokens);
        let report = generation::generate(
            &mut session,
            generation::Request {
                input: generation::Input::Reset(&task.prompt_tokens),
                max_new_tokens: task.max_output_tokens,
                stop_tokens: &task.stop_ids,
            },
            &cancelled,
            |event| {
                assert_eq!(event.index, visible.len());
                assert!(!task.stop_ids.contains(&event.token_id));
                visible.push(event.token_id);
                Ok(())
            },
        )
        .unwrap();
        let mut expected = task.prompt_tokens.clone();
        expected.extend_from_slice(&visible);
        let stop = match report.reason {
            generation::StopReason::StopToken { token_id } => {
                assert!(task.stop_ids.contains(&token_id));
                expected.push(token_id);
                true
            }
            generation::StopReason::OutputLimit => {
                assert_eq!(visible.len(), task.max_output_tokens);
                false
            }
            generation::StopReason::Cancelled => panic!("unexpected task cancellation"),
        };
        assert_eq!(session.history(), expected);
        assert_eq!(report.reused_tokens, 0);
        assert_eq!(report.prefill_tokens, task.prompt_tokens.len());
        assert_eq!(report.emitted_tokens, visible.len());
        assert_eq!(report.sampled_tokens, visible.len() + usize::from(stop));
        assert_eq!(report.history_tokens, expected.len());
        assert_eq!(report.pending_tokens, 1);
        assert_eq!(report.evaluated_tokens + 1, expected.len());
        assert_eq!(session.memory().admitted_bytes, admitted);
        emit(
            "TASK_RESULT",
            json!({"index":index,"id":task.id,"category":task.category,
                "prompt_tokens":task.prompt_tokens.len(),"output_reserve":task.max_output_tokens,
                "visible_token_ids":visible,"stop_reason":report.reason,"generation":report,
                "memory":session.memory(),"full_quality_qualified":false}),
        );
        session.reset();
        assert!(session.history().is_empty());
        assert_eq!((session.evaluated_len(), session.pending_len()), (0, 0));
        assert!(session.logits().is_none());
        assert_eq!(resident_snapshot(&source), owners);
        count += 1;
    }
    assert_eq!(count, if pilot { 3 } else { 97 });
    assert_eq!(identity(&file.metadata().unwrap()), original);
    assert_eq!(
        identity(&std::fs::symlink_metadata(&path).unwrap()),
        original
    );
    assert_eq!(identity(&inputs.metadata().unwrap()), input_identity);
    assert_eq!(
        identity(&std::fs::symlink_metadata(&input_path).unwrap()),
        input_identity
    );
    emit(
        "TASK_SUMMARY",
        json!({"cases":count,"pilot":pilot,"capacity":567,"chunk_capacity":32,
            "admitted_bytes_including_diagnostic":admitted+DIAGNOSTIC,
            "reset_checked":true,"quality_qualified":false}),
    );
}
