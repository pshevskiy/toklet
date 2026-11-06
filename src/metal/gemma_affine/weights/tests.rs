use super::*;

#[test]
#[ignore = "explicit Metal immutable affine8 tile boundary checks"]
fn affine8_storage_preserves_original_bytes_across_tiles() {
    let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
    let layout = Layout::new(3, 64, 8, 64, MetadataDtype::BF16).unwrap();
    assert!(
        context
            .weights_from_fill(layout, 2, usize::MAX, |_, _, _, _, _| {
                panic!("public affine4 upload must reject affine8 before callback")
            })
            .is_err()
    );
    let codes: Vec<_> = (0..192).map(|i| ((i * 7) % 256) as u8).collect();
    let scales = [128, 59, 0, 189, 128, 60];
    let biases = [128, 63, 0, 192, 128, 62];
    let fill = |first: usize, child: Layout, a: &mut [u8], b: &mut [u8], c: &mut [u8]| {
        let end = first + child.rows();
        a.copy_from_slice(&codes[first * 64..end * 64]);
        b.copy_from_slice(&scales[first * 2..end * 2]);
        c.copy_from_slice(&biases[first * 2..end * 2]);
        Ok(())
    };
    let plan = Plan::storage(layout, 2, usize::MAX).unwrap();
    let budget = plan.minimum_working;
    assert!(
        matches!(context.weights_from_fill_reported(layout,2,budget-1,fill),
        Err(Error::Budget { required, limit }) if required==budget && limit==budget-1)
    );
    let (storage, stats) = context
        .weights_from_fill_reported(layout, 2, budget, fill)
        .unwrap();
    assert_eq!(stats.working_bytes, budget);
    assert_eq!(storage.layout(), layout);
    assert_eq!(storage.views().count(), 2);
    for (first, view) in [0, 2].into_iter().zip(storage.views()) {
        let end = first + view.layout.rows();
        for (buffer, original) in [
            (view.codes, &codes[first * 64..end * 64]),
            (view.scales, &scales[first * 2..end * 2]),
            (view.biases, &biases[first * 2..end * 2]),
        ] {
            buffer.with_slice(|words| {
                assert!(
                    words
                        .iter()
                        .flat_map(|word| word.to_le_bytes())
                        .take(original.len())
                        .eq(original.iter().copied())
                );
                assert!(
                    words[original.len().div_ceil(4)..]
                        .iter()
                        .all(|&word| word == SENTINEL)
                );
            });
        }
    }
}

#[test]
fn immutable_plan_counts_only_original_buffers_device_handle_slots_and_source_without_gpu() {
    let layout = Layout::new(3, 64, 4, 64, MetadataDtype::BF16).unwrap();
    // Chunk2: codes64+28, scales4+28, biases4+28 =156.
    // Chunk1: codes32+28, odd BF16 scales/biases each2+2pad+28=124.
    let resident = 280 + 2 * size_of::<Chunk>() + size_of::<Weights>();
    let working = resident + 72 + size_of::<SourceScratch>();
    let plan = Plan::new(layout, 2, working).unwrap();
    assert_eq!(plan.source_bytes, [64, 4, 4]);
    assert_eq!(plan.owned(2).unwrap(), resident);
    assert_eq!(plan.minimum_working, working);
    assert_eq!(
        plan.working(3, [65, 6, 7]).unwrap(),
        working + size_of::<Chunk>() + 6
    );
    assert!(
        matches!(Plan::new(layout,2,working-1),Err(Error::Budget{required,limit}) if required==working && limit==working-1)
    );
    assert!(plan.owned(1).is_err());
    assert!(plan.working(2, [usize::MAX, 4, 4]).is_err());
    for maximum in [0, 16385, usize::MAX] {
        assert!(Plan::new(layout, maximum, usize::MAX).is_err());
    }
    let too_many = Layout::new(65, 64, 4, 64, MetadataDtype::BF16).unwrap();
    assert!(Plan::new(too_many, 1, usize::MAX).is_err());
    let wrong = Layout::new(3, 64, 8, 64, MetadataDtype::BF16).unwrap();
    assert!(Plan::new(wrong, 2, usize::MAX).is_err());
}
#[test]
fn complete_table_ownership_excludes_all_affine_session_buffers_without_gpu() {
    for k in [1536, 8960] {
        let layout = Layout::new(262144, k, 4, 64, MetadataDtype::BF16).unwrap();
        let plan = Plan::new(layout, 16384, usize::MAX).unwrap();
        assert_eq!(plan.chunks, 16);
        assert_eq!(plan.buffer_bytes, layout.total_bytes() + 16 * 3 * GUARD * 4);
        assert_eq!(
            plan.owned(16).unwrap(),
            layout.total_bytes()
                + 16 * 3 * GUARD * 4
                + 16 * size_of::<Chunk>()
                + size_of::<Weights>()
        );
        assert_eq!(
            plan.minimum_working - plan.owned(16).unwrap(),
            layout.total_bytes() / 16 + size_of::<SourceScratch>()
        );
    }
}
