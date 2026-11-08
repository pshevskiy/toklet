//! Same frozen task controller/replay for each separately authenticated source.
use super::super::heretic as owner;
use super::*;
use crate::metal::gemma_affine::resident::heretic::Norms;

const CAPACITY: usize = 567;

#[test]
#[ignore = "explicit pinned Heretic three-category generation pilot"]
fn quality_generation_pilot() {
    generate(true);
}
#[test]
#[ignore = "explicit pinned Heretic remaining97 generation cases"]
fn quality_generation_remaining() {
    generate(false);
}
#[test]
#[ignore = "explicit pinned Heretic fresh raw pilot trajectory verification"]
fn quality_replay_pilot() {
    replay(true);
}
#[test]
#[ignore = "explicit pinned Heretic fresh raw full100 trajectory verification"]
fn quality_replay_all() {
    replay(false);
}

fn admitted<S: session::InferenceSession>(
    new_session: impl Fn(session::Options) -> Result<S, session::Error>,
) -> S {
    let mut options = session::Options {
        capacity: CAPACITY,
        prefill_chunk_size: 32,
        memory_limit_bytes: owner::BUDGET,
        position_offset: 0,
    };
    let warm = new_session(options).unwrap();
    let memory = warm.memory();
    drop(warm);
    assert!(memory.admitted_bytes + DIAGNOSTIC <= PROCESS);
    options.memory_limit_bytes = memory.admitted_bytes - 1;
    assert!(
        matches!(new_session(options), Err(session::Error::MemoryBudget { required, limit })
        if required == memory.admitted_bytes && limit == required - 1)
    );
    options.memory_limit_bytes = memory.admitted_bytes;
    let active = new_session(options).unwrap();
    assert_eq!(active.memory(), memory);
    active
}

fn tasks(tasks: &[Task], active: &mut impl session::InferenceSession, pilot: bool) {
    if pilot {
        let report = generation::generate(
            active,
            generation::Request {
                input: generation::Input::Reset(&tasks[0].prompt_tokens),
                max_new_tokens: 1,
                stop_tokens: &tasks[0].stop_ids,
            },
            &AtomicBool::new(true),
            |_| panic!("cancelled pilot published a token"),
        )
        .unwrap();
        assert_eq!(report.reason, generation::StopReason::Cancelled);
        assert!(active.history().is_empty());
        assert!(active.append(&[2; CAPACITY + 1]).is_err());
        assert!(active.history().is_empty() && active.logits().is_none());
    }
    let count = run_tasks(tasks, active, pilot, || {});
    emit_summary(count, pilot, active.memory().admitted_bytes);
}

fn generate(pilot: bool) {
    println!();
    let (cases, inputs, path, identity) = read_tasks();
    if owner::modified() {
        let (source, context, held) = owner::load_parent();
        let before = source.test_retained_digest();
        let model = source.model_for_quality_probe(CAPACITY).unwrap();
        let mut active =
            admitted(|options| source.new_generation_session(&model, &context, options));
        tasks(&cases, &mut active, pilot);
        held.verify();
        assert_eq!(source.test_retained_digest(), before);
    } else {
        let (source, context, held) = owner::load_mlx();
        let norms = Norms::new(&source, owner::BUDGET).unwrap();
        let model = norms.model_for_quality_probe(CAPACITY).unwrap();
        let mut active =
            admitted(|options| norms.new_generation_session(&model, &context, options));
        tasks(&cases, &mut active, pilot);
        held.verify();
    }
    super::replay::verify_descriptor(&inputs, &path, identity);
    owner::footprint();
}

fn replay(pilot: bool) {
    println!();
    let (cases, inputs, path, identity) = read_tasks();
    let trace = super::replay::Trace::read("TOKLET_HERETIC_QUALITY_TRAJECTORY", pilot);
    if owner::modified() {
        let (source, context, held) = owner::load_parent();
        let before = source.test_retained_digest();
        let model = source.model_for_quality_probe(CAPACITY).unwrap();
        let mut active = source
            .new_session(&model, &context, 32, owner::BUDGET)
            .unwrap();
        super::replay::run(&cases, &trace, &mut active, 32, "HERETIC_REPLAY", || {
            held.verify();
            assert_eq!(source.test_retained_digest(), before);
            super::replay::verify_descriptor(&inputs, &path, identity);
        });
    } else {
        let (source, context, held) = owner::load_mlx();
        let norms = Norms::new(&source, owner::BUDGET).unwrap();
        let model = norms.model_for_quality_probe(CAPACITY).unwrap();
        let mut active = norms
            .new_session(&model, &context, 32, owner::BUDGET)
            .unwrap();
        super::replay::run(&cases, &trace, &mut active, 32, "HERETIC_REPLAY", || {
            held.verify();
            super::replay::verify_descriptor(&inputs, &path, identity);
        });
    }
    owner::footprint();
}
