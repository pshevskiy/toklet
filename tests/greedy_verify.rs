use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    hint::black_box,
    time::Instant,
};
use toklet::generation::{GreedyVerification, VerifiedNext, verify_greedy};

struct CountAllocations;
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

// SAFETY: This test-only shim delegates all allocation/deallocation and layout
// contracts unchanged to System. TLS counting touches no allocated memory and
// records only the measured test thread, excluding concurrent harness activity.
unsafe impl GlobalAlloc for CountAllocations {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count_allocation();
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountAllocations = CountAllocations;

const ROWS: [f32; 12] = [0., 5., 5., 0., 0., 0., 3., 0., 4., 0., 0., 0.];

#[test]
fn selects_the_exact_correction_or_bonus_with_lowest_id_ties() {
    for (proposals, accepted, token) in [([3, 2], 0, 1), ([1, 3], 1, 2), ([1, 2], 2, 0)] {
        assert_eq!(
            verify_greedy(&proposals, &ROWS, 4, &[]).unwrap(),
            GreedyVerification {
                accepted,
                next: VerifiedNext::Token(token),
            },
        );
    }
    assert_eq!(
        verify_greedy(&[], &ROWS[..4], 4, &[]).unwrap(),
        GreedyVerification {
            accepted: 0,
            next: VerifiedNext::Token(1),
        },
    );
}

#[test]
fn a_target_stop_stays_hidden_even_if_the_proposal_matches() {
    for (proposals, stops, accepted, token) in [
        ([1, 2], [2], 1, 2),
        ([1, 3], [2], 1, 2),
        ([1, 2], [0], 2, 0),
        ([1, 2], [1], 0, 1),
    ] {
        assert_eq!(
            verify_greedy(&proposals, &ROWS, 4, &stops).unwrap(),
            GreedyVerification {
                accepted,
                next: VerifiedNext::Stop(token),
            },
        );
    }
}

#[test]
fn rejects_malformed_inputs_even_after_the_first_decision() {
    for (proposals, rows, vocabulary, stops) in [
        ([1, 2], &ROWS[..11], 4, &[][..]),
        ([1, 2], &ROWS[..], 0, &[][..]),
        ([1, 2], &ROWS[..], usize::MAX, &[][..]),
        ([3, 4], &ROWS[..], 4, &[][..]),
        ([3, 2], &ROWS[..], 4, &[4][..]),
    ] {
        assert!(verify_greedy(&proposals, rows, vocabulary, stops).is_err());
    }
    for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut bad = ROWS;
        bad[11] = nonfinite;
        assert!(verify_greedy(&[3, 2], &bad, 4, &[]).is_err());
    }
}

#[test]
fn verification_does_not_allocate() {
    ALLOCATIONS.with(|count| count.set(0));
    TRACK.with(|track| track.set(true));
    for _ in 0..48 {
        black_box(verify_greedy(black_box(&[1, 2]), black_box(&ROWS), 4, &[2]).unwrap());
    }
    TRACK.with(|track| track.set(false));
    assert_eq!(ALLOCATIONS.with(Cell::get), 0);
}

#[test]
#[ignore = "Explicit release verifier throughput; excludes model execution"]
fn verifier_release_screen() {
    const VOCABULARY: usize = 151_936;
    const PROPOSALS: [u32; 4] = [100, 200, 300, 400];
    const REPEATS: usize = 64;
    let mut rows = vec![0.0; VOCABULARY * (PROPOSALS.len() + 1)];
    for (row, token) in PROPOSALS.into_iter().chain([500]).enumerate() {
        rows[row * VOCABULARY + token as usize] = 1.0;
    }
    let mut samples = Vec::with_capacity(10);
    for trial in 0..13 {
        let start = Instant::now();
        for _ in 0..REPEATS {
            assert_eq!(
                verify_greedy(black_box(&PROPOSALS), black_box(&rows), VOCABULARY, &[]).unwrap(),
                GreedyVerification {
                    accepted: 4,
                    next: VerifiedNext::Token(500)
                }
            );
        }
        if trial >= 3 {
            samples.push(start.elapsed().as_secs_f64() / REPEATS as f64);
        }
    }
    println!(
        "GREEDY_VERIFY {{\"vocabulary\":{VOCABULARY},\"rows\":5,\"seconds_per_batch\":{samples:?}}}"
    );
}
