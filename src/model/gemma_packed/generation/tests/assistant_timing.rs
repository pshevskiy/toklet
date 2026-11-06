use super::*;
use crate::{
    app_chat::gemma4::RenderOptions,
    generation::{self, Input, Request},
    metal::gemma_affine::{Bf16GenerationSession, OfficialBf16},
    model::gemma_assistant::{Assistant, LOAD_BYTES, Scratch},
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{path::PathBuf, sync::atomic::AtomicBool, time::Instant};

fn asset(name: &str) -> PathBuf {
    PathBuf::from(std::env::var_os(name).expect("explicit exact Gemma artifact"))
}

fn equal_state(a: &Bf16GenerationSession<'_, '_, '_>, b: &Bf16GenerationSession<'_, '_, '_>) {
    assert_eq!(a.history(), b.history());
    assert_eq!(a.evaluated_len(), b.evaluated_len());
    assert_eq!(a.pending_len(), b.pending_len());
    let width = a.vocab_size();
    let last = |size| size - width;
    let (al, bl) = (a.inner.logits(), b.inner.logits());
    assert!(
        al[last(al.len())..]
            .iter()
            .map(|x| x.to_bits())
            .eq(bl[last(bl.len())..].iter().map(|x| x.to_bits()))
    );
    for (old, new) in a.inner.caches.iter().zip(&b.inner.caches) {
        assert_eq!(old.offset(), new.offset());
        let (ak, av) = old.committed();
        let (bk, bv) = new.committed();
        assert!(
            ak.iter()
                .chain(av)
                .map(|x| x.to_bits())
                .eq(bk.iter().chain(bv).map(|x| x.to_bits()))
        );
    }
}

#[test]
#[ignore = "explicit real Mac assistant verification and paired speed feasibility"]
fn official_gemma_assistant_paired_pilot() {
    paired(false, false);
}

#[test]
#[ignore = "explicit real Mac paired assistant timing and guard workloads"]
fn official_gemma_assistant_paired_confirmation() {
    let order = std::env::var("TOKLET_GEMMA_ASSISTANT_ORDER").expect("explicit timing order");
    assert!(matches!(
        order.as_str(),
        "ordinary-first" | "assistant-first"
    ));
    paired(true, order == "assistant-first");
}

fn paired(confirmation: bool, reverse: bool) {
    const CAPACITY: usize = 512;
    let mut workspace = Scratch::new(CAPACITY, Scratch::planned_bytes(CAPACITY).unwrap()).unwrap();
    let budget = crate::weights::INITIAL_MEMORY_LIMIT_BYTES
        - LOAD_BYTES
        - Scratch::planned_bytes(CAPACITY).unwrap();
    let owner =
        OfficialBf16::load(&asset("TOKLET_GEMMA_OFFICIAL_DIRECTORY"), CAPACITY, budget).unwrap();
    let assistant =
        Assistant::open(&asset("TOKLET_GEMMA_ASSISTANT_DIRECTORY"), LOAD_BYTES).unwrap();
    let graph = owner.model().unwrap();
    let options = Options {
        capacity: CAPACITY,
        prefill_chunk_size: 32,
        position_offset: 0,
        memory_limit_bytes: budget,
    };
    let mut ordinary = owner.new_session(&graph, options).unwrap();
    let mut assisted = owner.new_session(&graph, options).unwrap();
    ordinary.use_last_row_prefill();
    assisted.use_last_row_prefill();
    let canonical = std::fs::read(asset("TOKLET_GEMMA_G2_INPUTS")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&canonical)),
        "877cbe18574b47e4b6a8d1b58fb79621c37e0531bd1f9504e50ce63d186f8c46"
    );
    let ids: Vec<u32> = canonical[..384 * 4]
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    let chat = owner
        .codec()
        .encode_chat(
            br#"[{"role":"user","content":"Name the capital of France."}]"#,
            RenderOptions {
                enable_thinking: false,
                add_generation_prompt: true,
            },
        )
        .unwrap();
    let cancel = AtomicBool::new(false);
    let pilot: &[(&[u32], usize, &[u32], usize)] = &[
        (chat.as_slice(), 32, owner.codec().stop_ids(), 2),
        (&ids[..128], 32, &[], 4),
        (&ids[..384], 32, &[], 4),
    ];
    let frozen: &[(&[u32], usize, &[u32], usize)] =
        &[(&ids[..128], 128, &[], 13), (&ids[..384], 128, &[], 13)];
    for &(prompt, maximum, stops, samples) in if confirmation { frozen } else { pilot } {
        for sample in 0..samples {
            let request = || Request {
                input: Input::Reset(prompt),
                max_new_tokens: maximum,
                stop_tokens: stops,
            };
            let mut basic_ids = Vec::with_capacity(maximum);
            let mut draft_ids = Vec::with_capacity(maximum);
            let mut basic = || {
                let time = Instant::now();
                let report = generation::generate(&mut ordinary, request(), &cancel, |event| {
                    basic_ids.push(event.token_id);
                    Ok(())
                })
                .unwrap();
                (report, time.elapsed().as_nanos())
            };
            let mut draft = || {
                let time = Instant::now();
                let report = super::super::assistant_experiment::generate_assisted(
                    &mut assisted,
                    &assistant,
                    &mut workspace,
                    request(),
                    &mut draft_ids,
                    &cancel,
                )
                .unwrap();
                (report, time.elapsed().as_nanos())
            };
            let assistant_first = reverse ^ (sample % 2 == 1);
            let ((basic_report, basic_ns), (draft_report, draft_ns)) = if !assistant_first {
                let before = basic();
                let after = draft();
                (before, after)
            } else {
                let after = draft();
                let before = basic();
                (before, after)
            };
            assert_eq!(basic_ids, draft_ids);
            assert_eq!(basic_report.reason, draft_report.reason);
            equal_state(&ordinary, &assisted);
            println!(
                "OFFICIAL_GEMMA_ASSISTED_PILOT {}",
                json!({
                    "prompt":prompt.len(),"maximum":maximum,"emitted":basic_ids.len(),"sample":sample,
                    "warmup":sample < if confirmation { 3 } else { 1 },
                "first":if assistant_first { "assistant" } else { "ordinary" },"ordinary_ns":basic_ns,"assisted_ns":draft_ns,
                    "ordinary_decode_seconds":basic_report.decode_seconds,
                    "proposed":draft_report.proposed,"accepted":draft_report.accepted,"rounds":draft_report.verification_rounds,
                    "ids_sha256":format!("{:x}",Sha256::digest(basic_ids.into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>()))
                })
            );
        }
    }
}
