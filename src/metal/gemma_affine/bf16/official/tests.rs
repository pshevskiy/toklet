use super::*;
use crate::generation::{self, Input, Request, StopReason};
use std::sync::atomic::AtomicBool;

#[test]
fn official_owner_rejects_context_and_budget_before_opening_assets() {
    let absent = Path::new("missing-official-gemma-source");
    for capacity in [0, OfficialBf16::MAX_POSITIONS + 1] {
        assert!(matches!(
            OfficialBf16::load(absent, capacity, INITIAL_MEMORY_LIMIT_BYTES),
            Err(Error::Options("Gemma BF16 Mac position limit"))
        ));
    }
    assert!(matches!(
        OfficialBf16::load(absent, 1, INITIAL_MEMORY_LIMIT_BYTES + 1),
        Err(Error::Options("Gemma BF16 Mac memory limit exceeds 6 GiB"))
    ));
    assert!(matches!(
        OfficialBf16::load(absent, 1, gemma4::ALLOWANCE_BYTES),
        Err(Error::Budget { required, limit }) if required > limit
    ));
}

#[test]
#[ignore = "explicit official Gemma BF16 combined owner, codec and generation witness"]
fn official_bf16_owner_admits_combined_memory_and_generates() {
    let directory = std::path::PathBuf::from(
        std::env::var_os("TOKLET_GEMMA_OFFICIAL_DIRECTORY").expect("pinned offline source"),
    );
    let budget = INITIAL_MEMORY_LIMIT_BYTES;
    let owner = OfficialBf16::load(&directory, OfficialBf16::MAX_POSITIONS, budget).unwrap();
    assert_eq!(owner.position_limit(), 512);
    assert_eq!(OfficialBf16::DECLARED_POSITIONS, 131_072);
    let memory = owner.memory();
    assert!(memory.load_admitted_bytes <= budget);
    assert_eq!(
        memory.load_admitted_bytes,
        memory.numerical_load.admitted_load_bytes + memory.session_reserved_bytes + CONFIG_BYTES
    );
    assert!(memory.session_reserved_bytes >= owner.codec().allowance_bytes());
    let graph = owner.model().unwrap();
    let mut options = Options {
        capacity: owner.position_limit(),
        prefill_chunk_size: 32,
        memory_limit_bytes: budget,
        position_offset: 0,
    };
    let warm = owner.new_session(&graph, options).unwrap();
    let numerical = warm.memory();
    let required = numerical.admitted_bytes + memory.session_reserved_bytes;
    drop(warm);
    options.memory_limit_bytes = required - 1;
    assert!(matches!(
        owner.new_session(&graph, options),
        Err(session::Error::MemoryBudget { required: actual, limit })
            if actual == required && limit == required - 1
    ));
    options.memory_limit_bytes = required;
    let mut session = owner.new_session(&graph, options).unwrap();
    assert_eq!(session.memory(), numerical);
    let prompt = owner
        .codec()
        .encode_chat(
            br#"[{"role":"user","content":"Name the capital of France."}]"#,
            gemma4::RenderOptions {
                enable_thinking: false,
                add_generation_prompt: true,
            },
        )
        .unwrap();
    assert_eq!(prompt.first(), Some(&2));
    let request = || Request {
        input: Input::Reset(&prompt),
        max_new_tokens: 12,
        stop_tokens: owner.codec().stop_ids(),
    };
    let boundary = vec![2; 511];
    let admitted = generation::request_admission(
        &session,
        &Request {
            input: Input::Reset(&boundary),
            max_new_tokens: 1,
            ..request()
        },
    );
    assert!(admitted.is_ok());
    assert!(
        generation::request_admission(
            &session,
            &Request {
                input: Input::Reset(&boundary),
                max_new_tokens: 2,
                ..request()
            },
        )
        .is_err()
    );
    let stopped = generation::generate(&mut session, request(), &AtomicBool::new(true), |_| {
        panic!("cancelled request emitted a token")
    })
    .unwrap();
    assert_eq!(stopped.reason, StopReason::Cancelled);
    assert!(session.history().is_empty());

    let mut reply = Vec::new();
    for _ in 0..2 {
        let mut ids = Vec::new();
        generation::generate(&mut session, request(), &AtomicBool::new(false), |event| {
            ids.push(event.token_id);
            Ok(())
        })
        .unwrap();
        if reply.is_empty() {
            assert!(!ids.is_empty());
            reply = ids;
        } else {
            assert_eq!(ids, reply);
        }
        session.reset();
        assert!(session.history().is_empty() && session.logits().is_none());
    }
    let text = owner.codec().decode(&reply, true).unwrap();
    assert!(!text.trim().is_empty());
    drop(session);
    drop(graph);
    drop(owner);

    let load_budget = memory.load_admitted_bytes;
    assert!(matches!(
        OfficialBf16::load(&directory, 512, load_budget - 1),
        Err(Error::Budget { required, limit })
            if required == load_budget && limit == load_budget - 1
    ));
    let exact = OfficialBf16::load(&directory, 512, load_budget).unwrap();
    assert_eq!(exact.memory().load_admitted_bytes, load_budget);
    println!(
        "OFFICIAL_BF16_OWNER {}",
        serde_json::json!({"load_admitted_bytes":load_budget,"session_admitted_bytes":required,
            "codec_allowance_bytes":gemma4::ALLOWANCE_BYTES,"kv_owned_bytes":numerical.kv_bytes,
            "declared_positions":OfficialBf16::DECLARED_POSITIONS,"position_limit":512,
            "prompt_tokens":prompt.len(),"visible_ids":reply,"stop_ids":gemma4::STOP_IDS})
    );
}

mod console;
