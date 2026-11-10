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
fn canonical_bf16_convolution_and_binary_calls_allocate_nothing() {
    use half::bf16;
    use toklet::cpu::{lfm3_add_bf16, lfm3_conv7_bf16, lfm3_head_gate_bf16, lfm3_mul_bf16};
    let mut input = vec![bf16::from_f32(0.25); 128 * 7680];
    let right = vec![bf16::from_f32(0.5); input.len()];
    let mut binary_output = vec![bf16::ZERO; input.len()];
    let history = vec![bf16::ZERO; 6 * 2048];
    let weights = vec![bf16::from_f32(0.125); 7 * 2048];
    let gates = vec![bf16::from_f32(0.5); 128 * 16];
    let mut conv_output = vec![bf16::ZERO; 128 * 2048];
    lfm3_conv7_bf16(
        &input[..128 * 2048],
        &weights,
        &history,
        &mut conv_output,
        128,
        2048,
    )
    .unwrap();
    lfm3_head_gate_bf16(&input[..128 * 2048], &gates, &mut conv_output, 128).unwrap();
    lfm3_add_bf16(&input, &right, &mut binary_output).unwrap();
    lfm3_mul_bf16(&input, &right, &mut binary_output).unwrap();
    let measurement = Measurement::start();
    for _ in 0..8 {
        for rows in [1, 17, 128] {
            lfm3_conv7_bf16(
                &input[..rows * 2048],
                &weights,
                &history,
                &mut conv_output[..rows * 2048],
                rows,
                2048,
            )
            .unwrap();
            lfm3_head_gate_bf16(
                &input[..rows * 2048],
                &gates[..rows * 16],
                &mut conv_output[..rows * 2048],
                rows,
            )
            .unwrap();
            lfm3_add_bf16(
                &input[..rows * 7680],
                &right[..rows * 7680],
                &mut binary_output[..rows * 7680],
            )
            .unwrap();
            lfm3_mul_bf16(
                &input[..rows * 7680],
                &right[..rows * 7680],
                &mut binary_output[..rows * 7680],
            )
            .unwrap();
            assert!(
                lfm3_conv7_bf16(&input, &weights, &history, &mut conv_output, 0, 2048).is_err()
            );
            assert!(lfm3_add_bf16(&input, &right[..1], &mut binary_output).is_err());
            input[0] = bf16::NAN;
            assert!(lfm3_mul_bf16(&input, &right, &mut binary_output).is_err());
            input[0] = bf16::from_f32(0.25);
            assert!(
                lfm3_head_gate_bf16(&input[..2048], &gates[..15], &mut conv_output[..2048], 1)
                    .is_err()
            );
        }
    }
    let allocations = measurement.finish();
    assert_eq!(allocations, 0);
    println!(
        "LFM3_CONV_BF16_ALLOC tokens=1,17,128 channels=2048 convolution=24 add=24 mul=24 head_gate=24 rejected=96 Rust_allocations={}",
        allocations
    );
}
