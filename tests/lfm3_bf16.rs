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
fn canonical_bf16_norm_and_unary_calls_allocate_nothing() {
    use half::bf16;
    use toklet::cpu::{lfm3_rmsnorm_bf16, lfm3_sigmoid_bf16, lfm3_silu_bf16};
    for width in [128, 2048] {
        let input = vec![bf16::from_f32(0.25); 17 * width];
        let gain = vec![bf16::ONE; width];
        let mut norm_output = vec![bf16::ZERO; input.len()];
        let mut activation = vec![bf16::from_f32(-0.375); 17 * 7680];
        let mut unary_output = vec![bf16::ZERO; activation.len()];
        lfm3_rmsnorm_bf16(&input, &gain, &mut norm_output, 17, width).unwrap();
        lfm3_sigmoid_bf16(&activation, &mut unary_output).unwrap();
        lfm3_silu_bf16(&activation, &mut unary_output).unwrap();
        let measurement = Measurement::start();
        for _ in 0..8 {
            for rows in [1, 17] {
                lfm3_rmsnorm_bf16(
                    &input[..rows * width],
                    &gain,
                    &mut norm_output[..rows * width],
                    rows,
                    width,
                )
                .unwrap();
                lfm3_sigmoid_bf16(&activation[..rows * 7680], &mut unary_output[..rows * 7680])
                    .unwrap();
                lfm3_silu_bf16(&activation[..rows * 7680], &mut unary_output[..rows * 7680])
                    .unwrap();
            }
            assert!(lfm3_rmsnorm_bf16(&input, &gain, &mut norm_output, 0, width).is_err());
            assert!(lfm3_sigmoid_bf16(&[], &mut []).is_err());
            activation[0] = bf16::NAN;
            assert!(lfm3_silu_bf16(&activation, &mut unary_output).is_err());
            activation[0] = bf16::from_f32(-0.375);
        }
        let allocations = measurement.finish();
        assert_eq!(allocations, 0);
        println!(
            "LFM3_BF16_ALLOC width={} norm=16 sigmoid=16 silu=16 rejected=24 Rust_allocations={}",
            width, allocations
        );
    }
}
