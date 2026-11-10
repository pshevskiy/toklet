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

#[test]
fn repeated_source_rotary_calls_and_rejections_allocate_nothing() {
    use half::bf16;
    use toklet::model::lfm3_rotary::{Profile, Rotary};
    for profile in [Profile::FullAttention, Profile::SlidingAttention] {
        let bytes = Rotary::planned_bytes(2048).unwrap();
        let table = Rotary::new(profile, 2048, bytes).unwrap();
        let input = vec![bf16::from_f32(0.25); 128 * 16 * 128];
        let mut output = vec![bf16::ZERO; input.len()];
        table.apply(&input, &mut output, 128, 16, 1023).unwrap();
        let before = table.allocated_bytes().unwrap();
        let pointer = table.coefficient_row(0).unwrap().as_ptr();
        let measurement = Measurement::start();
        for _ in 0..8 {
            for heads in [4, 16] {
                for tokens in [1, 32, 128] {
                    let length = tokens * heads * 128;
                    table
                        .apply(&input[..length], &mut output[..length], tokens, heads, 511)
                        .unwrap();
                }
            }
            assert!(table.apply(&input, &mut output, 0, 16, 0).is_err());
            assert!(table.apply(&input, &mut output, 128, 16, 2048).is_err());
        }
        let allocations = measurement.finish();
        assert_eq!(allocations, 0);
        assert_eq!(table.allocated_bytes().unwrap(), before);
        assert_eq!(table.coefficient_row(0).unwrap().as_ptr(), pointer);
        println!(
            "LFM3_ROTARY_ALLOC profile={:?} apply=48 rejected=16 Rust_allocations={}",
            profile, allocations
        );
    }
}
