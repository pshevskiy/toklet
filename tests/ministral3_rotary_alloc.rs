use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

struct Counting;
thread_local! {
    static TRACK: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn count_allocation() {
    let _ = TRACK.try_with(|track| {
        if track.get() {
            ALLOCATIONS.with(|count| count.set(count.get() + 1));
        }
    });
}

// SAFETY: This test-only shim forwards every pointer/layout/size unchanged to
// System. TLS counts inspect no payload and measure only this test thread,
// excluding concurrent harness allocations. Numerical implementation is safe.
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
        ALLOCATIONS.with(|count| count.set(0));
        TRACK.with(|track| track.set(true));
        Self
    }

    fn finish(self) -> usize {
        TRACK.with(|track| track.set(false));
        ALLOCATIONS.with(Cell::get)
    }
}

impl Drop for Measurement {
    fn drop(&mut self) {
        TRACK.with(|track| track.set(false));
    }
}

#[test]
fn warmed_keys_queries_and_rejected_calls_allocate_nothing() {
    use half::f16;
    use toklet::model::ministral3_rotary::{HEAD_DIM, ORIGINAL_MAX_POSITIONS, Rotary};

    let (base, capacity, query_heads, kv_heads) = (ORIGINAL_MAX_POSITIONS - 1, 17, 32, 8);
    let table = Rotary::new(base, capacity, Rotary::planned_bytes(capacity).unwrap()).unwrap();
    let mut queries = vec![f16::from_f32(-0.375); capacity * query_heads * HEAD_DIM];
    let keys = vec![f16::from_f32(0.25); capacity * kv_heads * HEAD_DIM];
    let mut query_output = vec![f16::ZERO; queries.len()];
    let mut key_output = vec![f16::ZERO; keys.len()];
    table
        .apply_queries(&queries, &mut query_output, capacity, query_heads, base)
        .unwrap();
    table
        .apply_keys(&keys, &mut key_output, capacity, kv_heads, base)
        .unwrap();

    let measurement = Measurement::start();
    for _ in 0..8 {
        for tokens in [1, 3, capacity] {
            let query_len = tokens * query_heads * HEAD_DIM;
            let key_len = tokens * kv_heads * HEAD_DIM;
            table
                .apply_queries(
                    &queries[..query_len],
                    &mut query_output[..query_len],
                    tokens,
                    query_heads,
                    base,
                )
                .unwrap();
            table
                .apply_keys(
                    &keys[..key_len],
                    &mut key_output[..key_len],
                    tokens,
                    kv_heads,
                    base,
                )
                .unwrap();
        }
        assert!(
            table
                .apply_queries(&[], &mut [], 0, query_heads, base)
                .is_err()
        );
        assert!(
            table
                .apply_keys(&keys, &mut key_output, capacity, kv_heads, base - 1)
                .is_err()
        );
        assert!(
            table
                .apply_queries(
                    &queries[..queries.len() - 1],
                    &mut query_output,
                    capacity,
                    query_heads,
                    base
                )
                .is_err()
        );
        assert!(
            table
                .apply_keys(&keys, &mut key_output, capacity, kv_heads, usize::MAX)
                .is_err()
        );
        let last = queries.len() - 1;
        queries[last] = f16::NAN;
        assert!(
            table
                .apply_queries(&queries, &mut query_output, capacity, query_heads, base)
                .is_err()
        );
        queries[last] = f16::from_f32(-0.375);
    }
    let allocations = measurement.finish();
    assert_eq!(allocations, 0);
    println!(
        "MINISTRAL3_ROTARY_ALLOC keys=24 queries=24 rejected=40 Rust_allocations={allocations}"
    );
}
