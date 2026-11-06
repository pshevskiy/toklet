//! Real target state delivery to the separately authenticated scalar assistant.
use super::*;
use crate::{
    app_chat::gemma4::RenderOptions,
    metal::gemma_affine::OfficialBf16,
    model::gemma_assistant::{
        Assistant, LOAD_BYTES, Scratch as DraftScratch, TARGET_HIDDEN, TargetKv,
    },
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
};

fn path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var_os(name).expect("explicit official fixture"))
}

fn write_floats(output: &mut impl Write, values: &[f32]) {
    for value in values {
        output.write_all(&value.to_le_bytes()).unwrap();
    }
}

#[test]
#[ignore = "explicit real Mac target and official scalar assistant capture"]
fn official_target_assistant_proposal_snapshot() {
    const CAPACITY: usize = 128;
    let output = path("TOKLET_GEMMA_G3_OUTPUT");
    let mut capture = BufWriter::new(File::create_new(output.join("target-input.bin")).unwrap());
    let mut results =
        BufWriter::new(File::create_new(output.join("assistant-output.bin")).unwrap());
    let scratch_bytes = DraftScratch::planned_bytes(CAPACITY).unwrap();
    let budget = crate::weights::INITIAL_MEMORY_LIMIT_BYTES - LOAD_BYTES - scratch_bytes;
    let owner =
        OfficialBf16::load(&path("TOKLET_GEMMA_OFFICIAL_DIRECTORY"), CAPACITY, budget).unwrap();
    let graph = owner.model().unwrap();
    let mut session = owner
        .new_session(
            &graph,
            Options {
                capacity: CAPACITY,
                prefill_chunk_size: 32,
                position_offset: 0,
                memory_limit_bytes: budget,
            },
        )
        .unwrap();
    let assistant = Assistant::open(&path("TOKLET_GEMMA_ASSISTANT_DIRECTORY"), LOAD_BYTES).unwrap();
    let mut scratch = DraftScratch::new(CAPACITY, scratch_bytes).unwrap();
    let prompt = owner
        .codec()
        .encode_chat(
            br#"[{"role":"user","content":"Name the capital of France."}]"#,
            RenderOptions {
                enable_thinking: false,
                add_generation_prompt: true,
            },
        )
        .unwrap();
    assert!(prompt.len() > 1 && prompt.len() <= CAPACITY);
    session.set_tokens(&prompt).unwrap();
    while session.pending_len() != 0 {
        session.evaluate_next_chunk().unwrap();
    }
    assert_eq!(session.evaluated_len(), prompt.len());
    let before = snapshot(&session.inner);
    let inner = &mut session.inner;
    let token = *inner.tokens().last().unwrap();
    let position = prompt.len() - 1;
    let mut input = [0.; 2 * TARGET_HIDDEN];
    let (embedding, hidden) = input.split_at_mut(TARGET_HIDDEN);
    let mut raw = [0.; TARGET_HIDDEN];
    inner
        .operations
        .gather(&graph.weights.embedding, &[token], &mut raw)
        .unwrap();
    cpu::prepare_per_layer_projection_input(&raw, embedding, 1, TARGET_HIDDEN).unwrap();
    let end = inner.published_rows * TARGET_HIDDEN;
    hidden.copy_from_slice(&inner.scratch.norm[end - TARGET_HIDDEN..end]);
    assert_eq!(inner.plan.producer_for(33).unwrap(), 13);
    assert_eq!(inner.plan.producer_for(34).unwrap(), 14);
    let kv = TargetKv {
        position,
        local: inner.caches[13].committed_segments(),
        full: inner.caches[14].committed_segments(),
    };
    write_floats(&mut capture, &input);
    for spans in [kv.local, kv.full] {
        for span in spans {
            write_floats(&mut capture, span.k);
        }
        for span in spans {
            write_floats(&mut capture, span.v);
        }
    }
    capture.flush().unwrap();
    let prediction = assistant.propose(&input, &kv, &mut scratch).unwrap();
    write_floats(&mut results, prediction.hidden);
    for &id in prediction.ids {
        results.write_all(&id.to_le_bytes()).unwrap();
    }
    write_floats(&mut results, prediction.logits);
    results.flush().unwrap();
    assert_eq!(snapshot(&session.inner), before);
    println!(
        "OFFICIAL_GEMMA_ASSISTANT_PROPOSAL {}",
        json!({
            "position":position,"token":token,"assistant_winner":prediction.winner,
            "target_winner":crate::sampling::greedy(session.logits().unwrap()).unwrap(),
            "load_allowance_bytes":LOAD_BYTES,"scratch_bytes":scratch_bytes,
            "target_snapshot_sha256":format!("{:x}",Sha256::digest(std::fs::read(output.join("target-input.bin")).unwrap())),
            "assistant_result_sha256":format!("{:x}",Sha256::digest(std::fs::read(output.join("assistant-output.bin")).unwrap()))
        })
    );
}

#[test]
#[ignore = "explicit Mac official BF16 strict target publication"]
fn official_target_verifies_prefix_before_publishing() {
    use crate::generation::{GreedyVerification, VerifiedNext};
    use crate::sampling::greedy;
    use std::cell::Cell;

    let capacity = 128;
    let budget = crate::weights::INITIAL_MEMORY_LIMIT_BYTES;
    let owner =
        OfficialBf16::load(&path("TOKLET_GEMMA_OFFICIAL_DIRECTORY"), capacity, budget).unwrap();
    let graph = owner.model().unwrap();
    let options = Options {
        capacity,
        prefill_chunk_size: 4,
        position_offset: 0,
        memory_limit_bytes: budget,
    };
    let mut expected = owner.new_session(&graph, options).unwrap();
    let mut actual = owner.new_session(&graph, options).unwrap();
    let prompt = owner
        .codec()
        .encode_chat(
            br#"[{"role":"user","content":"Name the capital of France."}]"#,
            RenderOptions {
                enable_thinking: false,
                add_generation_prompt: true,
            },
        )
        .unwrap();
    for state in [&mut actual, &mut expected] {
        state.use_last_row_prefill();
        state.append(&prompt).unwrap();
        while state.pending_len() != 0 {
            state.evaluate_next_chunk().unwrap();
        }
    }
    let last_hidden = |state: &crate::metal::gemma_affine::Bf16GenerationSession<'_, '_, '_>| {
        let end = state.inner.published_rows * TARGET_HIDDEN;
        state.inner.scratch.norm[end - TARGET_HIDDEN..end].to_vec()
    };
    let before = snapshot(&actual.inner);
    let mut anchor = last_hidden(&actual);
    let old_anchor = anchor.clone();
    let first = greedy(expected.logits().unwrap()).unwrap();
    expected.append(&[first]).unwrap();
    expected.evaluate_next_chunk().unwrap();
    let correction = greedy(expected.logits().unwrap()).unwrap();
    let wrong = (correction + 1) % actual.vocab_size() as u32;
    let proposals = [first, wrong];
    let checks = Cell::new(0);
    assert!(
        actual
            .verify_greedy_round(&proposals, &[], &mut anchor, || {
                let current = checks.get();
                checks.set(current + 1);
                current == 1
            })
            .unwrap()
            .is_none()
    );
    assert_eq!(anchor, old_anchor);
    assert_eq!(snapshot(&actual.inner), before);
    assert_eq!(
        actual
            .verify_greedy_round(&proposals, &[], &mut anchor, || false)
            .unwrap(),
        Some(GreedyVerification {
            accepted: 1,
            next: VerifiedNext::Token(correction)
        })
    );
    assert_eq!(anchor, last_hidden(&expected));
    assert_eq!(snapshot(&actual.inner), snapshot(&expected.inner));
    assert_eq!(actual.pending_len(), 1);
    assert_eq!(actual.history().last(), Some(&correction));
    expected.append(&[correction]).unwrap();
    expected.evaluate_next_chunk().unwrap();
    actual.evaluate_next_chunk().unwrap();
    assert_eq!(snapshot(&actual.inner), snapshot(&expected.inner));
    assert_eq!(last_hidden(&actual), last_hidden(&expected));
    println!(
        "OFFICIAL_GEMMA_STRICT_PREFIX {}",
        json!({
            "prompt":prompt.len(),"accepted":1,"corrected":correction,"cancellation_preserved":true,
            "final_history":actual.history(),"exact_cache_and_hidden":true
        })
    );
}
