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

use half::bf16;
use toklet::cpu::lfm3_linear_bf16;

struct Work {
    input: Vec<bf16>,
    weights: Vec<bf16>,
    output: Vec<bf16>,
    columns: usize,
}
impl Work {
    fn new(columns: usize) -> Self {
        let fixture: serde_json::Value = serde_json::from_slice(include_bytes!(
            "fixtures/lfm3-projection-bf16-v1/fixture.json"
        ))
        .unwrap();
        let item = fixture["matrices"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["source_shape"][1].as_u64().unwrap() == columns as u64)
            .unwrap();
        let weights = &item["weights"];
        let begin = weights["byte_offset"].as_u64().unwrap() as usize;
        let bytes = weights["byte_count"].as_u64().unwrap() as usize;
        let raw = &include_bytes!("fixtures/lfm3-projection-bf16-v1/weights.bf16le")
            [begin..begin + bytes];
        Self {
            input: vec![bf16::from_f32(0.25); 128 * columns],
            weights: raw
                .chunks_exact(2)
                .map(|b| bf16::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            output: vec![bf16::ZERO; 128 * 4],
            columns,
        }
    }
    fn apply(&mut self, tokens: usize) {
        lfm3_linear_bf16(
            &self.input[..tokens * self.columns],
            &self.weights,
            &mut self.output[..tokens * 4],
            tokens,
            self.columns,
            4,
        )
        .unwrap();
    }
}

#[test]
fn prepared_source_projection_and_rejections_allocate_nothing() {
    let mut work = [Work::new(2048), Work::new(7680)];
    for profile in &mut work {
        for tokens in [1, 17, 128] {
            profile.apply(tokens);
        }
    }
    let measurement = Measurement::start();
    for _ in 0..4 {
        for profile in &mut work {
            for tokens in [1, 17, 128] {
                profile.apply(tokens);
                assert!(
                    lfm3_linear_bf16(
                        &[],
                        &profile.weights,
                        &mut profile.output[..tokens * 4],
                        tokens,
                        profile.columns,
                        4
                    )
                    .is_err()
                );
                assert!(
                    lfm3_linear_bf16(
                        &profile.input[..tokens * profile.columns],
                        &profile.weights[..profile.weights.len() - 1],
                        &mut profile.output[..tokens * 4],
                        tokens,
                        profile.columns,
                        4
                    )
                    .is_err()
                );
                profile.input[0] = bf16::NAN;
                assert!(
                    lfm3_linear_bf16(
                        &profile.input[..tokens * profile.columns],
                        &profile.weights,
                        &mut profile.output[..tokens * 4],
                        tokens,
                        profile.columns,
                        4
                    )
                    .is_err()
                );
                profile.input[0] = bf16::from_f32(0.25);
                assert!(
                    lfm3_linear_bf16(
                        &profile.input,
                        &profile.weights,
                        &mut profile.output,
                        usize::MAX,
                        2,
                        4
                    )
                    .is_err()
                );
            }
        }
    }
    let allocations = measurement.finish();
    assert_eq!(allocations, 0);
    println!("LFM3_PROJECTION_ALLOC projections=24 rejected=96 Rust_allocations={allocations}");
}
