use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};
use toklet::model::short_conv3::{History, commit_batch, reset_batch};

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

#[test]
fn fixed_state_and_shared_work_prepare_commit_reset_allocate_nothing() {
    let mut states = [
        History::new(1024, 64).unwrap(),
        History::new(1024, 64).unwrap(),
    ];
    let mut candidates = [vec![0.; 2048], vec![0.; 2048]];
    let mut work = vec![0.; 2048];
    let input = vec![0.25; 7 * 1024];
    let weight = vec![0.5; 3 * 1024];
    let mut output = vec![0.; input.len()];
    let pointers = [
        states[0].committed().as_ptr(),
        states[1].committed().as_ptr(),
    ];
    let bytes = [
        states[0].allocated_bytes().unwrap(),
        states[1].allocated_bytes().unwrap(),
    ];
    // Warm TLS and numerical path before measurement.
    let [a, b] = &mut candidates;
    let f0 = states[0]
        .prepare_chunk(&input, &weight, a, &mut work, &mut output, 7)
        .unwrap();
    let f1 = states[1]
        .prepare_chunk(&input, &weight, b, &mut work, &mut output, 7)
        .unwrap();
    commit_batch(&mut states, &[f0, f1]).unwrap();
    reset_batch(&mut states).unwrap();
    let measurement = Measurement::start();
    for _ in 0..8 {
        for n in [1, 2, 3, 7] {
            let [a, b] = &mut candidates;
            let f0 = states[0]
                .prepare_chunk(
                    &input[..n * 1024],
                    &weight,
                    a,
                    &mut work,
                    &mut output[..n * 1024],
                    n,
                )
                .unwrap();
            let f1 = states[1]
                .prepare_chunk(
                    &input[..n * 1024],
                    &weight,
                    b,
                    &mut work,
                    &mut output[..n * 1024],
                    n,
                )
                .unwrap();
            commit_batch(&mut states, &[f0, f1]).unwrap();
        }
        reset_batch(&mut states).unwrap();
        assert!(
            states[0]
                .prepare_chunk(
                    &input,
                    &weight,
                    &mut candidates[0],
                    &mut work,
                    &mut output,
                    0
                )
                .is_err()
        );
    }
    let allocations = measurement.finish();
    assert_eq!(allocations, 0);
    for i in 0..2 {
        assert_eq!(states[i].committed().as_ptr(), pointers[i]);
        assert_eq!(states[i].allocated_bytes().unwrap(), bytes[i]);
        assert_eq!(states[i].offset(), 0);
        assert!(states[i].committed().iter().all(|x| x.to_bits() == 0));
    }
    println!(
        "SHORT_CONV3_ALLOC prepare=64 commit_batches=32 reset_batches=8 rejected_zero=8 Rust_allocations={allocations}"
    );
}
