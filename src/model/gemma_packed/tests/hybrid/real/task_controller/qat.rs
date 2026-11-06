//! Quality-only owner and independent teacher-forced replay for the pinned QAT profile.
use super::*;
use crate::metal::gemma_affine::MixedContext;
use hybrid::test_support::qat_resident_snapshot;

const CAPACITY: usize = 567;
const CHUNK: usize = 32;
const VOCAB: usize = 262144;
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

fn verify_descriptor(file: &std::fs::File, path: &std::path::Path, before: [u64; 7]) {
    assert_eq!(identity(&file.metadata().unwrap()), before);
    assert_eq!(identity(&std::fs::symlink_metadata(path).unwrap()), before);
}

#[derive(Deserialize)]
struct Trajectory {
    index: usize,
    id: String,
    sampled_ids: Vec<u32>,
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
    let trajectory_path =
        std::path::PathBuf::from(std::env::var_os("TOKLET_GEMMA_QAT_TRAJECTORY").unwrap());
    let mut trajectory_file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(&trajectory_path)
        .unwrap();
    let metadata = trajectory_file.metadata().unwrap();
    assert!(metadata.is_file() && metadata.len() <= 512 * 1024);
    let trajectory_identity = identity(&metadata);
    let mut raw = Vec::new();
    (&mut trajectory_file)
        .take(512 * 1024 + 1)
        .read_to_end(&mut raw)
        .unwrap();
    assert_eq!(raw.len() as u64, metadata.len());
    let trajectories: Vec<Trajectory> = raw
        .split(|&b| b == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect();
    let expected: Vec<_> = if pilot {
        vec![0, 40, 70]
    } else {
        (0..100).collect()
    };
    assert!(trajectories.iter().map(|x| x.index).eq(expected));
    let (source, file, path, original) = super::super::qat::load_qat();
    let owners = qat_resident_snapshot(&source);
    let norms = hybrid::QatNorms::new(&source, BUDGET).unwrap();
    let model = norms.model_for_quality_probe(CAPACITY).unwrap();
    let context = MixedContext::new(super::super::qat::MIXED_ALLOWANCE).unwrap();
    let mut session = norms.new_session(&model, &context, CHUNK, BUDGET).unwrap();
    let allocated = session.allocated_bytes().unwrap();
    let mut decisions = 0;
    let mut mismatches = 0;
    for trajectory in &trajectories {
        let task = &tasks[trajectory.index];
        assert_eq!(task.id, trajectory.id);
        let sampled = &trajectory.sampled_ids;
        assert!((1..=task.max_output_tokens).contains(&sampled.len()));
        assert!(sampled.iter().all(|&id| (id as usize) < VOCAB));
        assert!(
            sampled[..sampled.len() - 1]
                .iter()
                .all(|id| !task.stop_ids.contains(id))
        );
        assert!(
            sampled.len() == task.max_output_tokens
                || task.stop_ids.contains(sampled.last().unwrap())
        );
        let mut input = task.prompt_tokens.clone();
        input.extend_from_slice(&sampled[..sampled.len() - 1]);
        session.reset().unwrap();
        assert!(session.tokens().is_empty() && session.logits().is_empty());
        let first = task.prompt_tokens.len() - 1;
        let mut case_mismatches = 0;
        for (block, tokens) in input.chunks(CHUNK).enumerate() {
            let offset = block * CHUNK;
            for (i, row) in session
                .forward(tokens)
                .unwrap()
                .chunks_exact(VOCAB)
                .enumerate()
            {
                let position = offset + i;
                if position >= first {
                    let step = position - first;
                    let actual = crate::sampling::greedy(row).unwrap();
                    if actual != sampled[step] {
                        case_mismatches += 1;
                        emit(
                            "QAT_REPLAY_MISMATCH",
                            json!({
                                "index":trajectory.index,"step":step,"actual":actual,"expected":sampled[step]
                            }),
                        );
                    }
                }
            }
            assert_eq!(session.tokens(), &input[..offset + tokens.len()]);
        }
        assert_eq!(session.allocated_bytes().unwrap(), allocated);
        decisions += sampled.len();
        mismatches += case_mismatches;
        emit(
            "QAT_REPLAY_CASE",
            json!({
                "index":trajectory.index,"id":task.id,"decisions":sampled.len(),
                "mismatches":case_mismatches,"hidden_stop":task.stop_ids.contains(sampled.last().unwrap())
            }),
        );
    }
    verify_descriptor(&trajectory_file, &trajectory_path, trajectory_identity);
    verify_descriptor(&inputs, &input_path, input_identity);
    verify_descriptor(&file, &path, original);
    assert_eq!(qat_resident_snapshot(&source), owners);
    emit(
        "QAT_REPLAY_SUMMARY",
        json!({
            "cases":trajectories.len(),"decisions":decisions,"mismatches":mismatches,
            "session_allocated_bytes":allocated,"trajectory_sha256":format!("{:x}",Sha256::digest(&raw)),
            "independent_raw_replay":true
        }),
    );
    assert_eq!(mismatches, 0, "fresh raw replay changed greedy decisions");
}
