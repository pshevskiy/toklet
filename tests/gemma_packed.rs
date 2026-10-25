use toklet as engine;
use toklet::model::gemma_packed::Model;
#[path = "common/gemma_packed_fixture.rs"]
mod fixture;
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
fn warmed_decode_reset_accounting_and_rejected_admission_allocate_nothing() {
    let fixture = fixture::Fixture::new();
    let ids = &fixture.ids;
    assert_eq!(fixture.values("math", "logits").len(), 32 * 17);
    let model = Model::new(fixture.weights()).unwrap();
    let budget = model.model_bytes() + model.planned_session_bytes().unwrap();
    let measured = Measurement::start();
    let rejected = model.new_session(budget - 1).is_err();
    let allocations = measured.finish();
    assert!(rejected);
    assert_eq!(allocations, 0);
    let mut session = model.new_session(budget).unwrap();
    session.forward(&ids[..3]).unwrap();
    session.reset().unwrap();
    let bytes = session.allocated_bytes().unwrap();
    let measured = Measurement::start();
    for _ in 0..3 {
        for id in ids {
            session.forward(std::slice::from_ref(id)).unwrap();
        }
        assert_eq!(session.allocated_bytes().unwrap(), bytes);
        session.reset().unwrap();
    }
    let allocations = measured.finish();
    assert_eq!(allocations, 0);
    println!(
        "Gemma packed synthetic reference: 96 warmed decode steps, 3 resets, {allocations} Rust allocations; model={} session={bytes} bytes",
        model.model_bytes()
    );
}
