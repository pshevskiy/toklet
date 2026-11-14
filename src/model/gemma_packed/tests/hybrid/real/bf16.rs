//! Explicit real official BF16 diagnostic and private-scratch publication checks.
use super::*;
use crate::metal::gemma_affine::{
    Bf16Context, Bf16GenerationSession, Bf16GraphFault, Bf16Model, Bf16Session, Bf16Weights,
};
use std::{ffi::OsStr, io::Write};

const VOCAB: usize = 262144;
const BUDGET: usize = PROCESS - DIAGNOSTIC;

/// Cold qualification selection shared by the existing original-BF16 gates.
/// It changes submission only; output-row policy and numerical checks stay put.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Submission {
    PerTile,
    BatchedSingleRow,
}

impl Submission {
    fn parse(value: Option<&OsStr>) -> std::result::Result<Self, &'static str> {
        match value {
            None => Ok(Self::PerTile),
            Some(value) if value == "per-tile" => Ok(Self::PerTile),
            Some(value) if value == "batched-single-row" => Ok(Self::BatchedSingleRow),
            Some(_) => Err("TOKLET_GEMMA_BF16_SUBMISSION must be per-tile or batched-single-row"),
        }
    }

    pub(super) fn selected() -> Self {
        let value = std::env::var_os("TOKLET_GEMMA_BF16_SUBMISSION");
        let selected = Self::parse(value.as_deref()).expect("valid explicit BF16 test submission");
        let label = match selected {
            Self::PerTile => "per_tile",
            Self::BatchedSingleRow => "batched_single_row",
        };
        // Separate from capture/journal prefixes, so existing numerical readers
        // keep their exact schemas while the worker's route remains explicit.
        println!(
            "\nGEMMA_BF16_SUBMISSION {}",
            json!({"submission_policy":label,"scope":"original_bf16_qualification"})
        );
        std::io::stdout().flush().unwrap();
        selected
    }

    pub(super) fn new_session<'model, 'weights, 'context>(
        self,
        source: &'weights Bf16Weights,
        model: &'model Bf16Model<'weights>,
        context: &'context Bf16Context,
        chunk: usize,
        budget: usize,
    ) -> std::result::Result<Bf16Session<'model, 'weights, 'context>, hybrid::Error> {
        match self {
            Self::PerTile => source.new_session(model, context, chunk, budget),
            Self::BatchedSingleRow => source.new_batched_session(model, context, chunk, budget),
        }
    }

    pub(super) fn new_generation_session<'model, 'weights, 'context>(
        self,
        source: &'weights Bf16Weights,
        model: &'model Bf16Model<'weights>,
        context: &'context Bf16Context,
        options: crate::session::Options,
    ) -> std::result::Result<Bf16GenerationSession<'model, 'weights, 'context>, crate::session::Error>
    {
        match self {
            Self::PerTile => source.new_generation_session(model, context, options),
            Self::BatchedSingleRow => {
                source.new_generation_session_with_policy(model, context, options, None, true)
            }
        }
    }
}

#[test]
fn official_bf16_submission_selector_requires_a_known_explicit_route() {
    assert_eq!(Submission::parse(None), Ok(Submission::PerTile));
    assert_eq!(
        Submission::parse(Some(OsStr::new("per-tile"))),
        Ok(Submission::PerTile)
    );
    assert_eq!(
        Submission::parse(Some(OsStr::new("batched-single-row"))),
        Ok(Submission::BatchedSingleRow)
    );
    for unknown in ["", "unknown", "PER-TILE", "batched_single_row"] {
        assert!(Submission::parse(Some(OsStr::new(unknown))).is_err());
    }
    use std::os::unix::ffi::OsStrExt;
    assert!(Submission::parse(Some(OsStr::from_bytes(&[0xff]))).is_err());
}

pub(super) fn load_official() -> (Bf16Weights, Bf16Context, std::path::PathBuf, [u64; 7]) {
    let root = std::path::PathBuf::from(std::env::var_os("TOKLET_GEMMA_BF16_MODEL").unwrap());
    let path = root.join("model.safetensors");
    let config = std::fs::read(root.join("config.json")).unwrap();
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(&path)
        .unwrap();
    let original = identity(&file.metadata().unwrap());
    let context = Bf16Context::new(gemma_affine::CONTEXT_ALLOWANCE_BYTES).unwrap();
    let weights = Bf16Weights::load_text_only(&context, file, &config, BUDGET).unwrap();
    (weights, context, path, original)
}

#[test]
#[ignore = "explicit official BF16 short Metal graph and rollback witness"]
fn official_bf16_short_graph_preserves_state_and_causal_rows() {
    let submission = Submission::selected();
    let (weights, context, path, original) = load_official();
    assert!(weights.model(0).is_err());
    assert!(weights.model(33).is_err());
    let model = weights.model(32).unwrap();
    assert!(model.content_identity().is_none());
    let mut capture = capture::Capture::bounded(16 * 1024 * 1024);
    let mut causal: Option<Vec<f32>> = None;
    for (label, ids) in [("original", IDS), ("future", [2, 100, 257, 101])] {
        let mut expected = None;
        for chunk in [32, 3, 1] {
            let warm = submission
                .new_session(&weights, &model, &context, chunk, BUDGET)
                .unwrap();
            let owned = warm.allocated_bytes().unwrap();
            let required = model.model_bytes() + owned;
            drop(warm);
            assert!(matches!(
                submission.new_session(&weights, &model, &context, chunk, required - 1),
                Err(hybrid::Error::Metal(crate::metal::Error::Budget { required: r, limit }))
                    if r == required && limit == required - 1
            ));
            let mut session = submission
                .new_session(&weights, &model, &context, chunk, required)
                .unwrap();
            let pointers_before = pointers(&session);
            let mut rows = Vec::with_capacity(ids.len() * VOCAB);
            for input in ids.chunks(chunk) {
                rows.extend_from_slice(session.forward(input).unwrap());
            }
            assert_eq!(session.tokens(), ids);
            assert_eq!(session.allocated_bytes().unwrap(), owned);
            assert_eq!(pointers(&session), pointers_before);
            let state = canonical(snapshot(&session), VOCAB);
            if let Some((want_rows, want_state)) = &expected {
                assert!(&rows == want_rows, "full logits changed by chunk schedule");
                assert!(
                    &state == want_state,
                    "committed state changed by chunk schedule"
                );
            } else {
                capture.record(&format!("official.{label}.logits"), &rows, 0, ids.len());
                for (owner, cache) in session.caches.iter().enumerate() {
                    let count = ids.len() * model.layout.layers[owner].head_dim;
                    let (keys, values) = cache.committed();
                    for (tag, part) in [
                        ("k_rotary", &keys[..count]),
                        ("self_attn.v_norm", &values[..count]),
                    ] {
                        capture.record(
                            &format!("official.{label}.layer.{owner}.{tag}"),
                            part,
                            0,
                            ids.len(),
                        );
                    }
                }
                expected = Some((rows, state));
            }
            session.reset().unwrap();
            assert!(session.tokens().is_empty() && session.logits().is_empty());
            assert_eq!(pointers(&session), pointers_before);
            emit(
                "BF16_GRAPH_CHUNK",
                json!({"sequence":label,"chunk":chunk,"required_bytes":required,"session_bytes":owned}),
            );
        }
        let (rows, _) = expected.unwrap();
        if let Some(prior) = &causal {
            assert!(
                &rows[..3 * VOCAB] == prior.as_slice(),
                "future input changed causal rows"
            );
        } else {
            causal = Some(rows[..3 * VOCAB].to_vec());
        }
    }
    for (fault, label) in [
        (Bf16GraphFault::AfterFirstHeadTile, "head_tile"),
        (Bf16GraphFault::AfterPle, "ple"),
    ] {
        let mut session = submission
            .new_session(&weights, &model, &context, 3, BUDGET)
            .unwrap();
        session.forward(&[2]).unwrap();
        let before = snapshot(&session);
        session.operations.set_fault(fault);
        assert!(session.forward(&[100]).is_err());
        assert_eq!(snapshot(&session), before);
        assert!(session.reset().is_err());
        assert_eq!(snapshot(&session), before);
        emit(
            "BF16_GRAPH_FAILURE",
            json!({"after":label,"committed_state_preserved":true,"recreate":true}),
        );
    }
    assert_eq!(
        identity(&std::fs::symlink_metadata(&path).unwrap()),
        original
    );
    capture.finish();
    emit(
        "BF16_GRAPH_COMPLETE",
        json!({"committed_state_passed":true,"positions":4,"sequences":2,"app_qualified":false}),
    );
}

#[test]
#[ignore = "explicit official BF16 producer trace for the frozen 96-token prefix"]
fn official_bf16_producer_trace() {
    trace_producers(&[31, 38, 94, 95], 96, 0..15, false);
}

#[test]
#[ignore = "explicit official BF16 local V trace at the staged cache deviations"]
fn official_bf16_late_v_trace() {
    trace_producers(&[507, 508, 825, 826, 827], 832, 12..15, false);
}

#[test]
#[ignore = "explicit official BF16 two-producer trace before the first changed cache row"]
fn official_bf16_first_drift_trace() {
    trace_producers(&[69, 70, 71], 96, 0..2, true);
}

fn trace_producers(
    positions: &[usize],
    prefix: usize,
    producers: std::ops::Range<usize>,
    attention_details: bool,
) {
    const CHUNK: usize = 32;
    let submission = Submission::selected();
    let ids: Vec<u32> =
        include_bytes!("../../../../../../benchmarks/gemma-e2b-wrap-v1/tokens.u32le")
            .chunks_exact(4)
            .take(prefix)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect();
    assert_eq!(ids.len(), prefix);
    assert_eq!(prefix % CHUNK, 0);
    assert!(positions.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(positions.iter().all(|&position| position < prefix));
    let (source, context, path, original) = load_official();
    let owner = source.test_retained_digest();
    let model = source.model_for_quality_probe(ids.len()).unwrap();
    let mut session = submission
        .new_session(&source, &model, &context, CHUNK, BUDGET)
        .unwrap();
    let allocated = session.allocated_bytes().unwrap();
    let owners = pointers(&session);
    let mut capture = capture::Capture::bounded(8 * 1024 * 1024);
    for (chunk, input) in ids.chunks_exact(CHUNK).enumerate() {
        let before = chunk * CHUNK;
        let first = positions.partition_point(|&position| position < before);
        let end = positions.partition_point(|&position| position < before + CHUNK);
        let selected = &positions[first..end];
        let committed = (!selected.is_empty()).then(|| snapshot(&session));
        for index in producers.clone().filter(|_| committed.is_some()) {
            session.fault = Some(index);
            assert!(matches!(
                session.forward(input),
                Err(hybrid::Error::Model(Error::InvalidConfig(
                    "injected Gemma layer failure"
                )))
            ));
            assert_eq!(Some(snapshot(&session)), committed);
            let scratch = &session.scratch;
            let dim = model.layout.layers[index].head_dim;
            for &position in selected {
                let row = position - before;
                for (stage, values, width) in [
                    ("input_layernorm", &scratch.norm[..], H),
                    ("self_attn.v_proj", &scratch.kv[..], dim),
                    ("self_attn.v_norm", &session.staging[index].v[..], dim),
                    ("k_rotary", &session.staging[index].k[..], dim),
                    ("context", &scratch.context[..], model.layout.heads * dim),
                    ("output", &scratch.hidden[..], H),
                ] {
                    capture.record(
                        &format!("layer.{index}.{stage}"),
                        &values[row * width..(row + 1) * width],
                        position,
                        1,
                    );
                }
                if attention_details {
                    for (stage, values, width) in [
                        ("q_rotary", &scratch.q[..], model.layout.heads * dim),
                        ("self_attn.o_proj", &scratch.attention[..], H),
                        ("ple_input", &scratch.layer_ple[..], P),
                    ] {
                        capture.record(
                            &format!("layer.{index}.{stage}"),
                            &values[row * width..(row + 1) * width],
                            position,
                            1,
                        );
                    }
                    // Both layers are local and this prefix precedes eviction.
                    // The proposal is chronological through the full chunk.
                    let key_count = before + CHUNK;
                    assert!(key_count <= model.layout.window);
                    for (stage, values) in [
                        ("attention_scores", &scratch.scores[..]),
                        ("attention_probabilities", &scratch.probabilities[..]),
                    ] {
                        let mut causal = Vec::with_capacity(model.layout.heads * (position + 1));
                        for head in 0..model.layout.heads {
                            let start = (head * CHUNK + row) * key_count;
                            causal.extend_from_slice(&values[start..start + position + 1]);
                        }
                        capture.record(&format!("layer.{index}.{stage}"), &causal, position, 1);
                    }
                }
            }
        }
        session.fault = None;
        let logits = session.forward(input).unwrap();
        for &position in selected {
            let row = position - before;
            capture.record(
                "logits",
                &logits[row * VOCAB..(row + 1) * VOCAB],
                position,
                1,
            );
        }
        assert_eq!(session.tokens(), &ids[..before + CHUNK]);
        assert_eq!(pointers(&session), owners);
        assert_eq!(session.allocated_bytes().unwrap(), allocated);
    }
    assert_eq!(source.test_retained_digest(), owner);
    assert_eq!(
        identity(&std::fs::symlink_metadata(&path).unwrap()),
        original
    );
    capture.finish();
    emit(
        "BF16_PRODUCER_TRACE",
        json!({"positions":positions,"prefix":ids.len(),"session_bytes":allocated,
            "faults_preserved_committed_state":true,"diagnostic_only":true}),
    );
}
