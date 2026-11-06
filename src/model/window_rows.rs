//! Allocation-free row descriptors shared by cache views and commit copies.
//!
//! Callers validate nonzero capacity, local window==capacity, admitted positions
//! and nonempty chunks before calling. These are internal geometry operations,
//! not replacements for constructor/artifact or all-owner commit validation.
//! Descriptors address rows; each backend separately checks element/byte strides.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OldSpan {
    pub position: usize,
    pub slot: usize,
    pub rows: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CopyRun {
    pub source_row: usize,
    pub slot: usize,
    pub rows: usize,
}

/// At most W-1 old rows can affect any query in the new chunk, although the
/// physical local owner retains W rows. Full attention begins at absolute zero.
pub(crate) fn view_start(offset: usize, window: Option<usize>) -> usize {
    debug_assert!(window != Some(0));
    window.map_or(0, |window| offset.saturating_sub(window - 1))
}

/// Increasing absolute order, without linearizing the physical ring. Full
/// retention yields one old interval. Empty descriptors remain well-defined.
pub(crate) fn old_spans(capacity: usize, offset: usize, window: Option<usize>) -> [OldSpan; 2] {
    debug_assert!(capacity > 0);
    debug_assert!(window.map_or(offset <= capacity, |w| w == capacity));
    spans(capacity, view_start(offset, window), offset)
}

/// Physical intervals for an admitted absolute range of at most one ring.
pub(crate) fn spans(capacity: usize, start: usize, end: usize) -> [OldSpan; 2] {
    debug_assert!(capacity > 0 && start <= end && end - start <= capacity);
    let old = end - start;
    let slot = start % capacity;
    let first = old.min(capacity - slot);
    [
        OldSpan {
            position: start,
            slot,
            rows: first,
        },
        OldSpan {
            position: start + first,
            slot: 0,
            rows: old - first,
        },
    ]
}

/// Copy only the final min(tokens,W) local rows. Earlier new rows stay available
/// in staging during attention but are not part of the final committed cache.
/// The caller admits offset+tokens before this nonfallible copy phase.
pub(crate) fn copy_runs(
    capacity: usize,
    offset: usize,
    tokens: usize,
    sliding: bool,
) -> [CopyRun; 2] {
    debug_assert!(capacity > 0 && tokens > 0);
    debug_assert!(offset.checked_add(tokens).is_some());
    debug_assert!(sliding || offset + tokens <= capacity);
    let first_new = if sliding {
        tokens.saturating_sub(capacity)
    } else {
        0
    };
    let slot = if sliding {
        (offset + first_new) % capacity
    } else {
        offset
    };
    let count = tokens - first_new;
    let first = count.min(capacity - slot);
    [
        CopyRun {
            source_row: first_new,
            slot,
            rows: first,
        },
        CopyRun {
            source_row: first_new + first,
            slot: 0,
            rows: count - first,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_intervals_match_independent_absolute_rows() {
        for capacity in 1..=12 {
            for sliding in [false, true] {
                let window = sliding.then_some(capacity);
                for offset in 0..=48 {
                    if !sliding && offset > capacity {
                        continue;
                    }
                    let spans = old_spans(capacity, offset, window);
                    let actual: Vec<_> = spans
                        .into_iter()
                        .flat_map(|s| (0..s.rows).map(move |i| (s.position + i, s.slot + i)))
                        .collect();
                    let expected: Vec<_> = (0..offset)
                        .filter(|&p| !sliding || offset - p < capacity)
                        .map(|p| (p, if sliding { p % capacity } else { p }))
                        .collect();
                    assert_eq!(
                        actual, expected,
                        "capacity{capacity}, offset{offset}, sliding{sliding}"
                    );
                    assert_eq!(
                        view_start(offset, window),
                        expected.first().map_or(offset, |r| r.0)
                    );
                }
            }
        }
    }

    #[test]
    fn contiguous_copy_runs_match_tokenwise_oracle_and_preserve_untouched_slots() {
        for capacity in 1..=12 {
            for sliding in [false, true] {
                for offset in 0..=48 {
                    for tokens in 1..=16 {
                        if offset + tokens > 64 || (!sliding && offset + tokens > capacity) {
                            continue;
                        }
                        let mut actual = vec![None; capacity];
                        let mut expected = actual.clone();
                        for position in 0..offset {
                            let slot = if sliding {
                                position % capacity
                            } else {
                                position
                            };
                            actual[slot] = Some(position);
                            expected[slot] = Some(position);
                        }
                        let before = actual.clone();
                        for position in offset..offset + tokens {
                            expected[if sliding {
                                position % capacity
                            } else {
                                position
                            }] = Some(position);
                        }
                        let runs = copy_runs(capacity, offset, tokens, sliding);
                        let mut copied = vec![false; capacity];
                        for run in runs {
                            for i in 0..run.rows {
                                assert!(!copied[run.slot + i]);
                                copied[run.slot + i] = true;
                                actual[run.slot + i] = Some(offset + run.source_row + i);
                            }
                        }
                        assert_eq!(
                            actual, expected,
                            "capacity{capacity}, offset{offset}, tokens{tokens}, sliding{sliding}"
                        );
                        assert_eq!(copied.iter().filter(|&&x| x).count(), tokens.min(capacity));
                        for slot in 0..capacity {
                            if !copied[slot] {
                                assert_eq!(actual[slot], before[slot]);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn near_usize_boundary_keeps_absolute_and_physical_indices_separate() {
        let offset = usize::MAX - 1;
        let spans = old_spans(512, offset, Some(512));
        assert_eq!(spans[0].rows + spans[1].rows, 511);
        assert_eq!(spans[0].position, offset - 511);
        assert_eq!(spans[1].position + spans[1].rows, offset);
        let runs = copy_runs(512, offset, 1, true);
        assert_eq!(
            runs[0],
            CopyRun {
                source_row: 0,
                slot: offset % 512,
                rows: 1
            }
        );
        assert_eq!(runs[1].rows, 0);
        let singleton = old_spans(1, offset, Some(1));
        assert_eq!(singleton[0].rows + singleton[1].rows, 0);
        assert_eq!(view_start(offset, Some(1)), offset);
    }
}
