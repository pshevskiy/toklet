use super::*;

#[test]
fn projection_plan_separates_all_scratch_from_weights_and_admits_actual_capacities() {
    assert_eq!(size_of::<Projection<'_>>(), 240);
    let layout = Layout::new(19, 128, 4, 64, MetadataDtype::BF16).unwrap();
    // GPU input384 + output24 + status1, each7 guards; CPU full57 + chunk24.
    let required = (384 + 24 + 1 + 21 + 57 + 24) * 4 + size_of::<Projection<'_>>();
    let plan = Plan::new(layout, 8, 3, required).unwrap();
    assert_eq!(plan.minimum_owned, required);
    assert_eq!(plan.buffers(), [384, 24, 1]);
    assert_eq!(plan.owned(60, 25).unwrap(), required + 16);
    assert!(
        matches!(Plan::new(layout,8,3,required-1),Err(Error::Budget{required:r,limit}) if r==required && limit==required-1)
    );
    for (full, chunk) in [(56, 24), (57, 23), (usize::MAX, 24), (57, usize::MAX)] {
        assert!(plan.owned(full, chunk).is_err());
    }
    for (maximum, capacity) in [(0, 3), (16385, 3), (8, 0), (8, 33), (usize::MAX, 3)] {
        assert!(Plan::new(layout, maximum, capacity, usize::MAX).is_err());
    }
    let wrong = Layout::new(19, 128, 8, 64, MetadataDtype::BF16).unwrap();
    assert!(Plan::new(wrong, 8, 3, usize::MAX).is_err());
    let head = Layout::new(262144, 1536, 4, 64, MetadataDtype::BF16).unwrap();
    let head = Plan::new(head, 16384, 4, usize::MAX).unwrap();
    assert_eq!(head.input_values, 6144);
    assert_eq!(head.chunk_values, 65536);
    assert_eq!(head.full_values, 1048576);
}
#[test]
fn every_child_including_late_oversized_chunk_is_checked_before_staging() {
    let layout = |rows, columns| Layout::new(rows, columns, 4, 64, MetadataDtype::BF16).unwrap();
    let plan = Plan::new(layout(15, 64), 7, 3, usize::MAX).unwrap();
    assert!(plan.accepts([layout(7, 64), layout(7, 64), layout(1, 64)].into_iter()));
    assert!(!plan.accepts([layout(7, 64), layout(8, 64)].into_iter()));
    assert!(!plan.accepts([layout(7, 64), layout(7, 128), layout(1, 64)].into_iter()));
    assert!(!plan.accepts([layout(7, 64), layout(7, 64)].into_iter()));
    assert!(!plan.accepts([layout(7, 64), layout(7, 64), layout(2, 64)].into_iter()));
    assert!(!plan.accepts(std::iter::empty()));
}

use super::super::partitioned::tests::fixture;
mod gpu;

#[test]
fn maximum_child_bound_controls_whole_projection_without_weakening_l1_admission() {
    for columns in [64, 128] {
        let mut sparse = vec![0.; columns];
        sparse[0] = 1.;
        let hot = 16777216.;
        assert!(bounds(columns, hot, 1.).1 > ORDERED_ERROR_LIMIT);
        assert!(admit(columns, hot, &sparse).is_ok());
        assert!(admit(columns, hot, &vec![1.; columns]).is_err());
        for input in [sparse, vec![1.; columns]] {
            let children = [0., 0.25, 1., hot];
            let all = children
                .iter()
                .all(|&maximum| admit(columns, maximum, &input).is_ok());
            assert_eq!(admit(columns, hot, &input).is_ok(), all);
        }
    }
}
