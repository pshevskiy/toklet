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

#[test]
fn prompt_lookup_prefers_longest_then_latest_disjoint_continuation() {
    use toklet::generation::prompt_lookup;

    for (history, expected) in [
        (&[1, 2, 7, 8, 1, 2][..], &[7, 8][..]),
        (&[1, 2, 7, 1, 2, 8, 1, 2], &[8]),
        (&[7, 8, 42, 8, 13, 7, 8], &[42, 8, 13]),
        (&[1, 2, 1, 2], &[1]),
        (&[1, 1, 1, 1, 1], &[1]),
    ] {
        let mut proposal = [u32::MAX; 10];
        let count = prompt_lookup(history, 8, &mut proposal).unwrap();
        assert_eq!(&proposal[..count], expected);
        assert!(proposal[count..].iter().all(|&token| token == u32::MAX));
    }
    let mut proposal = [u32::MAX; 10];
    assert_eq!(
        prompt_lookup(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2], 8, &mut proposal).unwrap(),
        8
    );
    assert_eq!(proposal, [3, 4, 5, 6, 7, 8, 9, 10, u32::MAX, u32::MAX]);
}

#[test]
fn prompt_lookup_handles_short_or_missing_matches_without_writing() {
    use toklet::generation::prompt_lookup;

    let mut output = [999; 2];
    for (history, length) in [
        (&[][..], 8),
        (&[1], 8),
        (&[1, 1], 8),
        (&[1, 2, 3, 4, 5, 6], 8),
        (&[1, 2, 7, 1, 2], 0),
    ] {
        assert_eq!(prompt_lookup(history, length, &mut output).unwrap(), 0);
        assert_eq!(output, [999; 2]);
    }
    assert!(prompt_lookup(&[1, 9, 1], 9, &mut output).is_err());
    assert_eq!(output, [999; 2]);
    assert_eq!(prompt_lookup(&[1, 9, 1], 8, &mut []).unwrap(), 0);
    assert_eq!(
        prompt_lookup(&[1, 2, 7, 8, 1, 2], 8, &mut output[..1]).unwrap(),
        1
    );
    assert_eq!(output, [7, 999]);
}

#[test]
#[ignore = "Explicit release prompt lookup throughput and allocations"]
fn prompt_lookup_release_screen() {
    use toklet::generation::prompt_lookup;

    const REPEATS: usize = 2_048;
    for length in [128usize, 512, 1024] {
        for found in [false, true] {
            let mut history: Vec<_> = (0..u32::try_from(length).unwrap()).collect();
            if found {
                history[length - 8..].copy_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7]);
            }
            let mut output = [0; 8];
            let mut samples = Vec::with_capacity(10);
            for trial in 0..13 {
                ALLOCATIONS.with(|count| count.set(0));
                let start = Instant::now();
                TRACK.with(|track| track.set(true));
                for _ in 0..REPEATS {
                    let count =
                        prompt_lookup(black_box(&history), 8, black_box(&mut output)).unwrap();
                    assert_eq!(count, usize::from(found) * 8);
                    black_box(&output);
                }
                TRACK.with(|track| track.set(false));
                let elapsed = start.elapsed().as_secs_f64();
                assert_eq!(ALLOCATIONS.with(Cell::get), 0);
                if trial >= 3 {
                    samples.push(elapsed / REPEATS as f64);
                }
            }
            let scanned = if found {
                length - 16
            } else {
                (1..=8).map(|n| length - 2 * n).sum()
            };
            println!(
                "PROMPT_LOOKUP {{\"history\":{length},\"found\":{found},\"scanned\":{scanned},\"seconds_per_call\":{samples:?}}}"
            );
        }
    }
}
