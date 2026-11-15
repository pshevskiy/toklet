#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};
use toklet::{
    cpu::{AttentionKvSegment, SpanAttentionShape},
    metal::gemma_attention::{CONTEXT_ALLOWANCE_BYTES, Context, Options},
};
struct Counting;
thread_local! {static ACTIVE:Cell<bool>=const{Cell::new(false)};static COUNT:Cell<usize>=const{Cell::new(0)};}
fn count() {
    let _ = ACTIVE.try_with(|v| {
        if v.get() {
            COUNT.with(|n| n.set(n.get() + 1))
        }
    });
}
// SAFETY: test-only shim forwards each pointer/layout/size to System unchanged.
// TLS counts only this thread's Rust allocations and never accesses payloads.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        count();
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        count();
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        count();
        unsafe { System.realloc(p, l, n) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;
struct Measure;
impl Measure {
    fn start() -> Self {
        COUNT.with(|v| v.set(0));
        ACTIVE.with(|v| v.set(true));
        Self
    }
    fn finish(self) -> usize {
        ACTIVE.with(|v| v.set(false));
        COUNT.with(Cell::get)
    }
}
impl Drop for Measure {
    fn drop(&mut self) {
        ACTIVE.with(|v| v.set(false));
    }
}
#[test]
#[ignore = "explicit Apple GPU release workspace allocation/reuse check"]
fn warmed_changing_shapes_and_rejected_admission_allocate_no_rust_memory() {
    let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
    let options = Options {
        max_queries: 3,
        max_keys: 17,
        query_heads: 8,
        head_dim: 512,
    };
    let budget = options.planned_bytes().unwrap();
    let measured = Measure::start();
    let rejected = context.workspace(options, budget - 1).is_err();
    let allocations = measured.finish();
    assert!(rejected);
    assert_eq!(allocations, 0);
    let mut owner = context.workspace(options, budget).unwrap();
    let q = vec![0.; 3 * 8 * 512];
    let k = vec![0.; 17 * 512];
    let v = vec![1.; 17 * 512];
    let mut output = vec![42.; q.len()];
    let mut scores = vec![0.; 8 * 3 * 17];
    let mut probabilities = scores.clone();
    let empty = AttentionKvSegment {
        k: &[],
        v: &[],
        keys: 0,
        head_stride: 0,
        key_start: 0,
    };
    let spans = [
        AttentionKvSegment {
            k: &k,
            v: &v,
            keys: 17,
            head_stride: 17 * 512,
            key_start: 0,
        },
        empty,
        empty,
    ];
    let mut s = SpanAttentionShape {
        queries: 3,
        keys: 17,
        query_heads: 8,
        kv_heads: 1,
        head_dim: 512,
        query_start: 14,
        key_start: 0,
        window: Some(16),
        scale: 1.,
    };
    for _ in 0..3 {
        owner.apply(&q, &spans, &mut output, &s).unwrap();
    }
    let measured = Measure::start();
    let mut passed = true;
    for i in 0..32 {
        s.queries = 1 + i % 3;
        s.keys = [17, 3, 9, 17][i % 4];
        s.query_start = s.keys - s.queries;
        let parts = if i % 2 == 0 {
            [s.keys, 0, 0]
        } else {
            [s.keys / 3, s.keys / 3, s.keys - 2 * (s.keys / 3)]
        };
        let mut cursor = 0;
        let spans = parts.map(|keys| {
            let start = cursor;
            cursor += keys;
            AttentionKvSegment {
                k: &k[start * 512..cursor * 512],
                v: &v[start * 512..cursor * 512],
                keys,
                head_stride: keys * 512,
                key_start: start,
            }
        });
        let n = s.queries * 8 * 512;
        passed &= owner.apply(&q[..n], &spans, &mut output[..n], &s).is_ok();
        passed &= output[..n].iter().all(|&x| (x - 1.).abs() < 1e-6);
        let n_scores = 8 * s.queries * s.keys;
        passed &= owner
            .copy_diagnostics(&mut scores[..n_scores], &mut probabilities[..n_scores])
            .is_ok();
        passed &= owner
            .apply(&q[..n - 1], &spans, &mut output[..n], &s)
            .is_err();
        passed &= owner.owned_bytes() == budget;
    }
    let allocations = measured.finish();
    assert!(passed);
    assert_eq!(allocations, 0);
    assert!(!owner.is_poisoned());
    println!(
        "Gemma attention: 32 changing-shape warmed operations, {allocations} Rust allocations, {budget} admitted owned bytes; no driver-allocation or request-performance claim"
    );
}
