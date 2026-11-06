//! Common frozen task replay with an independent raw graph session.
use super::*;
use sha2::{Digest, Sha256};

const VOCAB: usize = 262144;

pub(super) fn verify_descriptor(file: &std::fs::File, path: &std::path::Path, before: [u64; 7]) {
    assert_eq!(identity(&file.metadata().unwrap()), before);
    assert_eq!(identity(&std::fs::symlink_metadata(path).unwrap()), before);
}

#[derive(Deserialize)]
struct Trajectory {
    index: usize,
    id: String,
    sampled_ids: Vec<u32>,
}

pub(super) struct Trace {
    path: std::path::PathBuf,
    file: std::fs::File,
    identity: [u64; 7],
    trajectories: Vec<Trajectory>,
    sha256: String,
}

impl Trace {
    pub(super) fn read(environment: &str, pilot: bool) -> Self {
        let trajectory_path = std::path::PathBuf::from(std::env::var_os(environment).unwrap());
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
        Self {
            path: trajectory_path,
            file: trajectory_file,
            identity: trajectory_identity,
            trajectories,
            sha256: format!("{:x}", Sha256::digest(&raw)),
        }
    }
}

pub(super) fn run<M, B>(
    tasks: &[Task],
    trace: &Trace,
    session: &mut Session<'_, '_, M, B>,
    chunk: usize,
    label: &str,
    verify_owners: impl FnOnce(),
) where
    M: MatrixSource,
    B: Operations<M, Error = hybrid::Error>,
{
    let mismatch_label = format!("{label}_MISMATCH");
    let case_label = format!("{label}_CASE");
    let allocated = session.allocated_bytes().unwrap();
    let mut decisions = 0;
    let mut mismatches = 0;
    for trajectory in &trace.trajectories {
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
        for (block, tokens) in input.chunks(chunk).enumerate() {
            let offset = block * chunk;
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
                            &mismatch_label,
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
            &case_label,
            json!({
                "index":trajectory.index,"id":task.id,"decisions":sampled.len(),
                "mismatches":case_mismatches,"hidden_stop":task.stop_ids.contains(sampled.last().unwrap())
            }),
        );
    }
    verify_descriptor(&trace.file, &trace.path, trace.identity);
    verify_owners();
    emit(
        &format!("{label}_SUMMARY"),
        json!({
            "cases":trace.trajectories.len(),"decisions":decisions,"mismatches":mismatches,
            "session_allocated_bytes":allocated,"trajectory_sha256":trace.sha256,
            "independent_raw_replay":true
        }),
    );
    assert_eq!(mismatches, 0, "fresh raw replay changed greedy decisions");
}
