//! Explicit small probe harness; ordinary CPU/Metal test targets do not run it.
#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use sha2::{Digest, Sha256};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};
use toklet::metal::gemma_affine::OfficialBf16;

struct Counting;
thread_local! {
    static TRACK: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}
fn allocated() {
    let _ = TRACK.try_with(|track| {
        if track.get() {
            ALLOCATIONS.with(|count| count.set(count.get() + 1));
        }
    });
}
// SAFETY: Same test-only accounting boundary as the existing numerical fixture
// allocator: forward pointer/layout/size to System without touching payloads.
// Thread-local counters exclude concurrent harness/driver-thread allocations.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        allocated();
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        allocated();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        allocated();
        unsafe { System.realloc(pointer, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn start() {
    ALLOCATIONS.with(|count| count.set(0));
    TRACK.with(|track| track.set(true));
}
fn finish() -> usize {
    TRACK.with(|track| track.set(false));
    ALLOCATIONS.with(Cell::get)
}

#[test]
#[ignore = "explicit GPU component controls, no model or timing cohort"]
fn head_batch_component_controls() {
    OfficialBf16::run_head_batch_component_controls(start, finish);
    println!("HEAD_BATCH_CONTROLS passed=9; zero-Rust-allocation control included");
}

#[test]
#[ignore = "explicit authenticated real-head shape screen, not whole-engine retention"]
fn head_batch_fixed_screen() {
    OfficialBf16::run_head_batch_component_screen();
}

#[test]
#[ignore = "explicit optimized authenticated BF16 session: warmed ordinary decode must allocate zero Rust storage"]
fn head_batch_ordinary_session_release_allocations_are_zero() {
    assert!(
        !cfg!(debug_assertions),
        "release session allocation qualification requires debug assertions disabled"
    );
    const LIMIT: usize = 6 * 1024 * 1024 * 1024;
    const CALLER: usize = 16 * 1024 * 1024;
    const HELPER: usize = 32 * 1024 * 1024;
    const BUDGET: usize = LIMIT - CALLER - HELPER;
    let directory = std::env::var_os("TOKLET_GEMMA_OFFICIAL_DIRECTORY")
        .expect("authenticated offline official source");
    let owner = OfficialBf16::load(std::path::Path::new(&directory), 512, BUDGET).unwrap();
    assert!(owner.memory().load_admitted_bytes + CALLER + HELPER <= LIMIT);
    let model = owner.model().unwrap();
    let ids = owner
        .codec()
        .encode_chat(
            br#"[{"role":"user","content":"Name the capital of France."}]"#,
            toklet::app_chat::gemma4::RenderOptions {
                enable_thinking: false,
                add_generation_prompt: true,
            },
        )
        .unwrap();
    assert_eq!(
        ids,
        [
            2, 105, 2364, 107, 1567, 506, 5279, 529, 7001, 236761, 106, 107, 105, 4368, 107
        ]
    );
    let input_sha256 = format!(
        "{:x}",
        Sha256::digest(
            ids.iter()
                .flat_map(|id| id.to_le_bytes())
                .collect::<Vec<_>>()
        )
    );
    let mut options = toklet::session::Options {
        capacity: 512,
        prefill_chunk_size: 32,
        position_offset: 0,
        memory_limit_bytes: BUDGET,
    };
    let memory = {
        let session = owner.new_batched_g2_session(&model, options).unwrap();
        session.memory()
    };
    let required = memory.admitted_bytes + owner.memory().session_reserved_bytes;
    assert!(required + CALLER + HELPER <= LIMIT);
    options.memory_limit_bytes = required - 1;
    assert!(matches!(
        owner.new_batched_g2_session(&model,options),
        Err(toklet::session::Error::MemoryBudget{required:actual,limit})
        if actual==required && limit==required-1
    ));
    options.memory_limit_bytes = required;
    let mut session = owner.new_batched_g2_session(&model, options).unwrap();
    assert_eq!(session.memory(), memory);
    assert!(session.bf16_profile_status().is_none());
    session.append(&ids).unwrap();
    assert_eq!(session.evaluate_next_chunk().unwrap(), 15);
    session.append(&[818]).unwrap();
    assert_eq!(session.evaluate_next_chunk().unwrap(), 1);
    assert_eq!(
        (
            session.history().len(),
            session.evaluated_len(),
            session.pending_len()
        ),
        (16, 16, 0)
    );
    start();
    let append = session.append(&[5279]);
    let evaluate = session.evaluate_next_chunk();
    let allocations = finish();
    append.unwrap();
    assert_eq!(evaluate.unwrap(), 1);
    assert_eq!(allocations, 0, "release ordinary decode allocation gate");
    assert_eq!(
        (
            session.history().len(),
            session.evaluated_len(),
            session.pending_len()
        ),
        (17, 17, 0)
    );
    assert_eq!(session.memory(), memory);
    assert_eq!(session.logits().unwrap().len(), 262144);
    assert!(
        session
            .logits()
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );
    println!(
        "HEAD_BATCH_SESSION_ALLOCATION {}",
        serde_json::json!({"passed":true,"release_zero_required":true,
            "rust_allocations":allocations,"input_tokens":ids.len(),"input_tokens_sha256":input_sha256,
            "history":17,"committed":17,"pending":0,"profile_enabled":false,
            "whole_owner_session_admitted_bytes":required,"caller_allowance_bytes":CALLER,
            "helper_allowance_bytes":HELPER,"limit_bytes":LIMIT,
            "scope":"Warmed append5279 plus completed ordinary batched G2 decode; current-thread Rust allocations; native driver allocations excluded"})
    );
}
