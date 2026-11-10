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
use smallvec::SmallVec;
use toklet::model::{Error, lfm3_state::State, short_conv7_bf16, window_cache_bf16};

struct Buffers {
    input: Vec<bf16>,
    keys: Vec<bf16>,
    values: Vec<bf16>,
    weights: Vec<bf16>,
    candidate: Vec<Vec<bf16>>,
    output: Vec<bf16>,
}
impl Buffers {
    fn new() -> Self {
        let mut weights = vec![bf16::ZERO; 2048 * 7];
        for row in weights.chunks_exact_mut(7) {
            row[6] = bf16::ONE;
        }
        Self {
            input: vec![bf16::from_f32(0.25); 128 * 2048],
            keys: vec![bf16::from_f32(0.5); 4 * 128 * 128],
            values: vec![bf16::from_f32(-0.5); 4 * 128 * 128],
            weights,
            candidate: (0..13).map(|_| vec![bf16::ZERO; 6 * 2048]).collect(),
            output: vec![bf16::ZERO; 128 * 2048],
        }
    }
    fn step(&mut self, state: &mut State, tokens: usize) -> Result<(), Error> {
        let mut histories: SmallVec<[short_conv7_bf16::Prepared<'_>; 13]> = SmallVec::new();
        let mut caches: SmallVec<[window_cache_bf16::PreparedKv<'_>; 11]> = SmallVec::new();
        for (owner, candidate) in state.histories().iter().zip(&mut self.candidate) {
            histories.push(owner.prepare_chunk(
                &self.input[..tokens * 2048],
                &self.weights,
                candidate,
                &mut self.output[..tokens * 2048],
                tokens,
            )?);
        }
        for cache in state.caches() {
            let frame = cache.prepare(
                &self.keys[..4 * tokens * 128],
                &self.values[..4 * tokens * 128],
                tokens,
            )?;
            let view = cache.view(&frame)?;
            std::hint::black_box(view.segments());
            for head in 0..4 {
                std::hint::black_box(view.head(head)?);
            }
            std::hint::black_box(view.visible_range(state.offset() + tokens - 1)?);
            caches.push(frame);
        }
        state.commit(&histories, &caches, tokens)
    }
}

#[test]
fn admitted_bf16_state_transitions_and_rejections_allocate_nothing() {
    let mut state = State::new(2048).unwrap();
    let mut buffers = Buffers::new();
    for _ in 0..4 {
        buffers.step(&mut state, 128).unwrap();
    }
    let owned = state.allocated_bytes();
    let measurement = Measurement::start();
    for _ in 0..8 {
        for tokens in [1, 17, 128] {
            buffers.step(&mut state, tokens).unwrap();
            let offset = state.offset();
            assert!(state.commit(&[], &[], 1).is_err());
            assert!(
                state.histories()[0]
                    .prepare_chunk(
                        &buffers.input[..2048],
                        &[],
                        &mut buffers.candidate[0],
                        &mut buffers.output[..2048],
                        1
                    )
                    .is_err()
            );
            buffers.keys[0] = bf16::NAN;
            assert!(
                state.caches()[0]
                    .prepare(&buffers.keys[..512], &buffers.values[..512], 1)
                    .is_err()
            );
            buffers.keys[0] = bf16::from_f32(0.5);
            assert!(state.commit(&[], &[], usize::MAX).is_err());
            assert_eq!(state.offset(), offset);
            assert_eq!(state.allocated_bytes(), owned);
        }
    }
    let allocations = measurement.finish();
    assert_eq!(allocations, 0);
    assert_eq!(state.offset(), 1680);
    println!(
        "LFM3_BF16_STATE_ALLOC transitions=24 histories=312 KV=264 rejected=96 Rust_allocations={}",
        allocations
    );
}
