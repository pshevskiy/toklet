//! Authenticated raw-token LFM3 graph verification and release baseline.
use std::{collections::BTreeMap, hint::black_box, path::Path, time::Instant};

use half::bf16;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use toklet::{
    model::{Error, lfm3::Model},
    weights::{INITIAL_MEMORY_LIMIT_BYTES, load_lfm3},
};

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

struct Counting;
thread_local! {
    static TRACK:Cell<bool>=const {Cell::new(false)};
    static ALLOCATIONS:Cell<usize>=const {Cell::new(0)};
}
fn count_allocation() {
    let _ = TRACK.try_with(|track| {
        if track.get() {
            ALLOCATIONS.with(|count| count.set(count.get() + 1));
        }
    });
}
// SAFETY: The test-only shim forwards every pointer/layout/size unchanged to
// System. TLS counters inspect no allocated payload and only measure this test
// thread, excluding concurrent harness allocations. Numerical code stays safe.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count_allocation();
        unsafe { System.realloc(pointer, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;

struct Measurement;
impl Measurement {
    fn start() -> Self {
        ALLOCATIONS.with(|v| v.set(0));
        TRACK.with(|v| v.set(true));
        Self
    }
    fn finish(self) -> usize {
        TRACK.with(|v| v.set(false));
        ALLOCATIONS.with(Cell::get)
    }
}
impl Drop for Measurement {
    fn drop(&mut self) {
        TRACK.with(|v| v.set(false));
    }
}

const REFERENCE: &[u8] = include_bytes!("../../tests/fixtures/lfm3-bf16-graph-v1/fixture.json");
const CONTRACT: &[u8] = include_bytes!("../../tests/fixtures/lfm3-bf16-graph-v1/contract.md");
const PRODUCER: &[u8] = include_bytes!("../reference/lfm3_graph_bf16_fixture.py");

#[derive(Deserialize)]
struct Fixture {
    tokens: Vec<u32>,
    schedules: Vec<Schedule>,
    contract_sha256: String,
    producer_sha256: String,
}
#[derive(Deserialize)]
struct Schedule {
    chunks: Vec<usize>,
    layers: Vec<Vec<BTreeMap<String, String>>>,
    final_norm: Vec<String>,
    logits: Vec<String>,
}

fn hash(values: &[bf16]) -> String {
    let mut sha = Sha256::new();
    for v in values {
        sha.update(v.to_le_bytes());
    }
    format!("{:x}", sha.finalize())
}

fn verify(model: &mut Model) -> Result<(), Box<dyn std::error::Error>> {
    let fixture: Fixture = serde_json::from_slice(REFERENCE)?;
    assert_eq!(
        format!("{:x}", Sha256::digest(REFERENCE)),
        "c1593a06e671e1914f03897fce3299341806e17361b3dbac907cf517328934b5"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(CONTRACT)),
        fixture.contract_sha256
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(PRODUCER)),
        fixture.producer_sha256
    );
    let mut session = model.session()?;
    let mut comparisons = 0;
    let mut failures = 0;
    let mut values = 0;
    for (s, schedule) in fixture.schedules.iter().enumerate() {
        session.reset()?;
        let mut offset = 0;
        for (chunk, count) in schedule.chunks.iter().enumerate() {
            let end = offset + count;
            let tokens = &fixture.tokens[offset..end];
            if s == 1 && chunk == 1 {
                let before = hash(session.logits().unwrap());
                let generation = session.generation();
                let aborted = session.forward_observed(tokens, |event| {
                    if event.stage == "logits" {
                        Err(Error::InvalidConfig("fixture abort"))
                    } else {
                        Ok(())
                    }
                });
                assert!(matches!(
                    aborted,
                    Err(Error::InvalidConfig("fixture abort"))
                ));
                assert_eq!(hash(session.logits().unwrap()), before);
                assert_eq!(session.history(), &fixture.tokens[..offset]);
                assert_eq!(session.generation(), generation);
            }
            let mut observed = 0;
            let logits = session.forward_observed(tokens, |event| {
                let expected = match event.layer {
                    Some(layer) => &schedule.layers[layer][chunk][event.stage],
                    None if event.stage == "final_norm" => &schedule.final_norm[chunk],
                    None => &schedule.logits[chunk],
                };
                let actual = hash(event.values);
                if &actual != expected {
                    eprintln!("schedule={s} chunk={chunk} layer={:?} stage={} actual={actual} expected={expected}",
                              event.layer, event.stage);
                    failures += 1;
                }
                observed += 1;
                values += event.values.len();
                Ok(())
            })?;
            assert_eq!(observed, 13 * 2 + 11 * 3 + 2);
            assert_eq!(logits.len(), 128_000);
            comparisons += observed;
            assert_eq!(session.history(), &fixture.tokens[..end]);
            offset = end;
        }
        assert_eq!(offset, fixture.tokens.len());
    }
    session.reset()?;
    let tokens = &fixture.tokens[..1];
    let measure = Measurement::start();
    let logits = session.forward(tokens);
    let allocations = measure.finish();
    let singleton = fixture.schedules.iter().find(|s| s.chunks[0] == 1).unwrap();
    assert_eq!(hash(logits?), singleton.logits[0]);
    let generation = session.generation();
    assert!(session.forward(&[128_000]).is_err());
    assert_eq!(session.history(), tokens);
    assert_eq!(session.generation(), generation);
    assert_eq!(hash(session.logits().unwrap()), singleton.logits[0]);
    assert_eq!(allocations, 0);
    println!(
        "{}",
        serde_json::json!({
            "schedules": fixture.schedules.len(), "comparisons": comparisons, "failed": failures,
            "words": values, "late_abort_and_retry": true, "prepared_forward_allocations": allocations
        })
    );
    if failures != 0 {
        return Err("LFM3 source graph differs".into());
    }
    Ok(())
}

fn benchmark(model: &mut Model, reverse: bool) -> Result<(), Box<dyn std::error::Error>> {
    let prepare = Instant::now();
    let mut session = model.session()?;
    let prepare_ns = prepare.elapsed().as_nanos();
    let fixture: Fixture = serde_json::from_slice(REFERENCE)?;
    let mut cohorts = [(1, false), (1, true), (4, false), (16, false)];
    if reverse {
        cohorts.reverse();
    }
    for (count, continuation) in cohorts {
        let tokens: Vec<_> = fixture.tokens.iter().copied().cycle().take(count).collect();
        let mut samples = [0; 14];
        for duration in &mut samples {
            session.reset()?;
            if continuation {
                session.forward(&fixture.tokens[..2])?;
            }
            let started = Instant::now();
            let logits = session.forward(black_box(&tokens))?;
            *duration = started.elapsed().as_nanos();
            black_box(logits);
        }
        println!(
            "{}",
            serde_json::json!({
                "queries": count, "prefix": if continuation { 2 } else { 0 }, "prepare_ns": prepare_ns,
                "first_ns": samples[0], "warmup_ns": &samples[1..4], "samples_ns": &samples[4..],
                "logits_sha256": hash(session.logits().unwrap()),
            })
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(debug_assertions) {
        return Err("release build required".into());
    }
    let args: Vec<_> = std::env::args().skip(1).collect();
    let [directory, mode] = args.as_slice() else {
        return Err("usage: lfm3_graph_bf16_probe MODEL_DIRECTORY verify|forward|reverse".into());
    };
    if !matches!(mode.as_str(), "verify" | "forward" | "reverse") {
        return Err("unknown probe mode".into());
    }
    let begin = Instant::now();
    let loaded = load_lfm3(Path::new(directory), 2048, 16, INITIAL_MEMORY_LIMIT_BYTES)?;
    let mut model = loaded.model;
    let report = loaded.report;
    let m = model.memory();
    println!(
        "{}",
        serde_json::json!({
            "load_ns": begin.elapsed().as_nanos(), "weights_bytes": m.weights_bytes,
            "authenticated_ns": report.authenticated_ns, "layout_ns": report.layout_ns,
            "delivery_ns": report.delivery_ns, "rotary_ns": report.rotary_ns,
            "source_bytes": report.source_bytes, "temporary_allowance_bytes": report.temporary_allowance_bytes,
            "source_sha256": format!("{:x}", sha2::digest::Output::<Sha256>::from(report.source_sha256)),
            "state_bytes": m.state_bytes, "positional_bytes": m.positional_bytes, "scratch_bytes": m.scratch_bytes,
            "history_bytes": m.history_bytes, "published_logits_bytes": m.published_logits_bytes,
            "owner_bytes": m.owner_bytes, "total_bytes": m.total_bytes,
            "capacity": 2048, "chunk": 16,
        })
    );
    if mode == "verify" {
        verify(&mut model)
    } else {
        benchmark(&mut model, mode == "reverse")
    }
}
