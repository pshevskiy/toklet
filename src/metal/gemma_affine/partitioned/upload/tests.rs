use super::*;

#[test]
fn peak_plan_counts_all_loading_buffers_descriptors_and_resident_capacity_without_gpu() {
    let layout = Layout::new(3, 64, 4, 64, MetadataDtype::BF16).unwrap();
    // Independent device bytes: two rows508 + tail row472. Resident CPU
    // scratch is (3+2)*4. Loading is64+4+4 with three Vec descriptors.
    let resident = 1000 + 2 * size_of::<Affine<'_>>() + size_of::<PartitionedAffine<'_>>();
    let required = resident + 72 + size_of::<SourceScratch>();
    let plan = UploadPlan::new(layout, 2, 1, required).unwrap();
    assert_eq!(plan.source_bytes, [64, 4, 4]);
    assert_eq!(plan.resident.minimum_owned, resident);
    assert_eq!(plan.minimum_working, required);
    assert!(
        matches!(UploadPlan::new(layout,2,1,required-1),Err(Error::Budget{required:r,limit})
        if r==required && limit==required-1)
    );
    assert_eq!(
        plan.working(3, 4, 4, [65, 6, 7]).unwrap(),
        required + size_of::<Affine<'_>>() + 18
    );
    assert!(matches!(
        plan.working(2, 3, 2, [usize::MAX, 4, 4]),
        Err(Error::Options(OVERFLOW))
    ));
    for (maximum, capacity) in [(0, 1), (16385, 1), (2, 0), (2, 33)] {
        assert!(UploadPlan::new(layout, maximum, capacity, usize::MAX).is_err());
    }
    let wrong_format = Layout::new(3, 64, 8, 64, MetadataDtype::BF16).unwrap();
    assert!(UploadPlan::new(wrong_format, 2, 1, usize::MAX).is_err());
}

#[test]
fn real_ple_peak_loads_one_sixteenth_of_source_bytes_without_gpu() {
    let layout = Layout::new(262144, 8960, 4, 64, MetadataDtype::BF16).unwrap();
    let plan = UploadPlan::new(layout, 16384, 1, usize::MAX).unwrap();
    assert_eq!(layout.total_bytes(), 1321205760);
    assert_eq!(plan.source_bytes, [73400320, 4587520, 4587520]);
    assert_eq!(plan.source_bytes.into_iter().sum::<usize>(), 82575360);
    assert_eq!(
        plan.minimum_working - plan.resident.minimum_owned,
        82575360 + size_of::<SourceScratch>()
    );
    // A plan does not reserve/load these large allocations or qualify a device.
}

mod gpu;
