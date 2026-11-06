//! Quality-only owner and independent teacher-forced replay for the pinned QAT profile.
use super::replay::verify_descriptor;
use super::*;
use crate::metal::gemma_affine::MixedContext;
use hybrid::test_support::qat_resident_snapshot;

const CAPACITY: usize = 567;
const CHUNK: usize = 32;
const BUDGET: usize = PROCESS - DIAGNOSTIC;

#[test]
#[ignore = "explicit pinned QAT three-category generation pilot"]
fn mixed_qat_generation_pilot() {
    generate(true);
}

#[test]
#[ignore = "explicit pinned QAT remaining97 generation cases"]
fn mixed_qat_generation_remaining() {
    generate(false);
}

fn generate(pilot: bool) {
    println!();
    let (tasks, inputs, input_path, input_identity) = read_tasks();
    let (source, file, path, original) = super::super::qat::load_qat();
    let owners = qat_resident_snapshot(&source);
    let norms = hybrid::QatNorms::new(&source, BUDGET).unwrap();
    let model = norms.model_for_quality_probe(CAPACITY).unwrap();
    let context = MixedContext::new(super::super::qat::MIXED_ALLOWANCE).unwrap();
    let mut options = session::Options {
        capacity: CAPACITY,
        prefill_chunk_size: CHUNK,
        memory_limit_bytes: BUDGET,
        position_offset: 0,
    };
    let warm = norms
        .new_quality_generation_session(&model, &context, options)
        .unwrap();
    let memory = warm.memory();
    drop(warm);
    options.memory_limit_bytes = memory.admitted_bytes - 1;
    assert!(matches!(
        norms.new_quality_generation_session(&model, &context, options),
        Err(session::Error::MemoryBudget { required, limit })
        if required == memory.admitted_bytes && limit == required - 1
    ));
    options.memory_limit_bytes = memory.admitted_bytes;
    let mut session = norms
        .new_quality_generation_session(&model, &context, options)
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
    assert_eq!(qat_resident_snapshot(&source), owners);
    verify_descriptor(&file, &path, original);
    verify_descriptor(&inputs, &input_path, input_identity);
    emit_summary(count, pilot, memory.admitted_bytes);
}

#[test]
#[ignore = "explicit fresh raw-session QAT pilot trajectory verification"]
fn mixed_qat_replay_pilot() {
    replay(true);
}

#[test]
#[ignore = "explicit fresh raw-session full100 QAT trajectory verification"]
fn mixed_qat_replay_all() {
    replay(false);
}

fn replay(pilot: bool) {
    println!();
    let (tasks, inputs, input_path, input_identity) = read_tasks();
    let trace = super::replay::Trace::read("TOKLET_GEMMA_QAT_TRAJECTORY", pilot);
    let (source, file, path, original) = super::super::qat::load_qat();
    let owners = qat_resident_snapshot(&source);
    let norms = hybrid::QatNorms::new(&source, BUDGET).unwrap();
    let model = norms.model_for_quality_probe(CAPACITY).unwrap();
    let context = MixedContext::new(super::super::qat::MIXED_ALLOWANCE).unwrap();
    let mut session = norms.new_session(&model, &context, CHUNK, BUDGET).unwrap();
    super::replay::run(&tasks, &trace, &mut session, CHUNK, "QAT_REPLAY", || {
        verify_descriptor(&inputs, &input_path, input_identity);
        verify_descriptor(&file, &path, original);
        assert_eq!(qat_resident_snapshot(&source), owners);
    });
}
