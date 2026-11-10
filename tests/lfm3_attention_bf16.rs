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
use toklet::cpu::{KvSegment, LFM3_ATTENTION_SCALE, SpanAttentionShape, lfm3_attention_bf16};

#[test]
fn prepared_attention_and_admission_rejections_allocate_nothing() {
    for (queries, keys) in [(1, 3), (1, 4), (32, 287), (128, 383)] {
        let q = vec![bf16::ONE; queries * 16 * 128];
        let kv = vec![bf16::ONE; 4 * keys * 128];
        let mut output = vec![bf16::ZERO; q.len()];
        let mut scratch = vec![0.0; keys];
        let window = (keys > 255).then_some(256);
        let segments = [
            KvSegment {
                k: &[],
                v: &[],
                keys: 0,
                head_stride: 0,
                key_start: 0,
            },
            KvSegment {
                k: &[],
                v: &[],
                keys: 0,
                head_stride: 0,
                key_start: 0,
            },
            KvSegment {
                k: &kv,
                v: &kv,
                keys,
                head_stride: keys * 128,
                key_start: 0,
            },
        ];
        let shape = SpanAttentionShape {
            queries,
            keys,
            query_heads: 16,
            kv_heads: 4,
            head_dim: 128,
            query_start: keys - queries,
            key_start: 0,
            window,
            scale: LFM3_ATTENTION_SCALE,
        };
        lfm3_attention_bf16(&q, &segments, &mut output, &mut scratch, &shape).unwrap();
        let measure = Measurement::start();
        let result = lfm3_attention_bf16(&q, &segments, &mut output, &mut scratch, &shape);
        let invalid = lfm3_attention_bf16(&q, &segments, &mut output, &mut scratch[..0], &shape);
        assert_eq!(measure.finish(), 0);
        result.unwrap();
        assert!(invalid.is_err());
    }
}
