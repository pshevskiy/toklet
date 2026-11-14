//! Quality-only owner and independent teacher-forced replay for the official BF16 profile.
use super::replay::verify_descriptor;
use super::*;

const CAPACITY: usize = 567;
const CHUNK: usize = 32;
const BUDGET: usize = PROCESS - DIAGNOSTIC;

#[test]
#[ignore = "explicit official BF16 three-category generation pilot"]
fn official_bf16_generation_pilot() {
    generate(true);
}

#[test]
#[ignore = "explicit official BF16 remaining97 generation cases"]
fn official_bf16_generation_remaining() {
    generate(false);
}

fn generate(pilot: bool) {
    println!();
    let submission = super::super::bf16::Submission::selected();
    let (tasks, inputs, input_path, input_identity) = read_tasks();
    let (source, context, path, original) = super::super::bf16::load_official();
    let owner = source.test_retained_digest();
    let model = source.model_for_quality_probe(CAPACITY).unwrap();
    let mut options = session::Options {
        capacity: CAPACITY,
        prefill_chunk_size: CHUNK,
        memory_limit_bytes: BUDGET,
        position_offset: 0,
    };
    let warm = submission
        .new_generation_session(&source, &model, &context, options)
        .unwrap();
    let memory = warm.memory();
    drop(warm);
    options.memory_limit_bytes = memory.admitted_bytes - 1;
    assert!(matches!(
        submission.new_generation_session(&source, &model, &context, options),
        Err(session::Error::MemoryBudget { required, limit })
        if required == memory.admitted_bytes && limit == required - 1
    ));
    options.memory_limit_bytes = memory.admitted_bytes;
    let mut session = submission
        .new_generation_session(&source, &model, &context, options)
        .unwrap();
    assert_eq!(session.memory(), memory);
    if pilot {
        let cancelled = generation::generate(
            &mut session,
            generation::Request {
                input: generation::Input::Reset(&tasks[0].prompt_tokens),
                max_new_tokens: 1,
                stop_tokens: &tasks[0].stop_ids,
            },
            &AtomicBool::new(true),
            |_| panic!("cancelled request published a token"),
        )
        .unwrap();
        assert_eq!(cancelled.reason, generation::StopReason::Cancelled);
        assert!(session.history().is_empty());
        assert!(session.append(&[2; CAPACITY + 1]).is_err());
        assert!(session.history().is_empty() && session.logits().is_none());
    }
    let count = run_tasks(&tasks, &mut session, pilot, || {});
    assert_eq!(source.test_retained_digest(), owner);
    assert_eq!(
        identity(&std::fs::symlink_metadata(&path).unwrap()),
        original
    );
    verify_descriptor(&inputs, &input_path, input_identity);
    emit_summary(count, pilot, memory.admitted_bytes);
}

#[test]
#[ignore = "explicit fresh raw-session BF16 pilot trajectory verification"]
fn official_bf16_replay_pilot() {
    replay(true);
}

#[test]
#[ignore = "explicit fresh raw-session full100 BF16 trajectory verification"]
fn official_bf16_replay_all() {
    replay(false);
}

fn replay(pilot: bool) {
    println!();
    let submission = super::super::bf16::Submission::selected();
    let (tasks, inputs, input_path, input_identity) = read_tasks();
    let trace = super::replay::Trace::read("TOKLET_GEMMA_BF16_TRAJECTORY", pilot);
    let (source, context, path, original) = super::super::bf16::load_official();
    let owner = source.test_retained_digest();
    let model = source.model_for_quality_probe(CAPACITY).unwrap();
    let mut session = submission
        .new_session(&source, &model, &context, CHUNK, BUDGET)
        .unwrap();
    super::replay::run(&tasks, &trace, &mut session, CHUNK, "BF16_REPLAY", || {
        verify_descriptor(&inputs, &input_path, input_identity);
        assert_eq!(
            identity(&std::fs::symlink_metadata(&path).unwrap()),
            original
        );
        assert_eq!(source.test_retained_digest(), owner);
    });
}
