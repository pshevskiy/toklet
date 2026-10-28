use super::*;

fn coefficient(value: f32, dtype: MetadataDtype) -> Vec<u8> {
    match dtype {
        MetadataDtype::F16 => f16::from_f32(value).to_le_bytes().to_vec(),
        MetadataDtype::BF16 => half::bf16::from_f32(value).to_le_bytes().to_vec(),
        MetadataDtype::F32 => value.to_le_bytes().to_vec(),
    }
}

// A leading byte deliberately makes all three original spans unaligned. The
// row-specific bytes and coefficients are independently prescribed here;
// neither row_span nor original_tensors constructs the expected slices below.
fn source(layout: Layout) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    assert_eq!(layout.rows(), 5);
    assert_eq!(layout.columns(), layout.group_size());
    let mut weights = vec![0x91];
    let mut scales = vec![0x92];
    let mut biases = vec![0x93];
    let row_bytes = layout.columns() * usize::from(layout.bits()) / 8;
    for row in 0..5 {
        weights.extend((0..row_bytes).map(|i| (row * 37 + i * 11) as u8));
        scales.extend(coefficient(
            [0.5, -0.25, 0.125, 1., -0.][row],
            layout.metadata_dtype(),
        ));
        biases.extend(coefficient(
            [-1., 0.5, 0.25, -0.5, 2.][row],
            layout.metadata_dtype(),
        ));
    }
    (weights, scales, biases)
}

#[test]
fn original_matrix_copy_preserves_all_formats_unaligned_and_nested_rows() {
    for bits in [2, 3, 4, 5, 6, 8] {
        for dtype in [MetadataDtype::F16, MetadataDtype::BF16, MetadataDtype::F32] {
            for columns in [32, 64, 128] {
                let layout = Layout::new(5, columns, bits, columns, dtype).unwrap();
                let (w, s, b) = source(layout);
                let original = (w.clone(), s.clone(), b.clone());
                let parent = Matrix::from_le_bytes(layout, &w[1..], &s[1..], &b[1..]).unwrap();
                let selected = parent.row_range(1..4).unwrap();
                let nested = selected.row_range(1..2).unwrap();
                for (matrix, first, count) in [(&parent, 0, 5), (&selected, 1, 3), (&nested, 2, 1)]
                {
                    let layout = matrix.layout();
                    let metadata = layout.metadata_bytes().div_ceil(4) * 4;
                    let mut out_w = vec![0xa1; layout.weight_bytes() + 2];
                    let mut out_s = vec![0xa2; metadata + 2];
                    let mut out_b = vec![0xa3; metadata + 2];
                    copy_matrix(
                        layout,
                        matrix,
                        &mut out_w[1..=layout.weight_bytes()],
                        &mut out_s[1..=metadata],
                        &mut out_b[1..=metadata],
                    )
                    .unwrap();
                    let row_bytes = columns * usize::from(bits) / 8;
                    let start = 1 + first * row_bytes;
                    assert_eq!(
                        &out_w[1..=layout.weight_bytes()],
                        &w[start..start + count * row_bytes]
                    );
                    let start = 1 + first * dtype.bytes();
                    assert_eq!(
                        &out_s[1..1 + count * dtype.bytes()],
                        &s[start..start + count * dtype.bytes()]
                    );
                    assert_eq!(
                        &out_b[1..1 + count * dtype.bytes()],
                        &b[start..start + count * dtype.bytes()]
                    );
                    assert!(
                        out_s[1 + count * dtype.bytes()..=metadata]
                            .iter()
                            .all(|&x| x == 0)
                    );
                    assert!(
                        out_b[1 + count * dtype.bytes()..=metadata]
                            .iter()
                            .all(|&x| x == 0)
                    );
                    assert_eq!((out_w[0], *out_w.last().unwrap()), (0xa1, 0xa1));
                    assert_eq!((out_s[0], *out_s.last().unwrap()), (0xa2, 0xa2));
                    assert_eq!((out_b[0], *out_b.last().unwrap()), (0xa3, 0xa3));
                }
                assert_eq!((w, s, b), original);
            }
        }
    }
}

#[test]
fn matrix_upload_layout_and_every_destination_extent_reject_before_writes() {
    let layout = Layout::new(5, 32, 4, 32, MetadataDtype::BF16).unwrap();
    let (w, s, b) = source(layout);
    let parent = Matrix::from_le_bytes(layout, &w[1..], &s[1..], &b[1..]).unwrap();
    let matrix = parent.row_range(1..4).unwrap();
    let expected = matrix.layout();
    for owner in 0..3 {
        for adjustment in [-1isize, 1] {
            let mut lengths = [48usize, 8, 8];
            lengths[owner] = lengths[owner].checked_add_signed(adjustment).unwrap();
            let mut w = vec![0xa1; lengths[0]];
            let mut s = vec![0xa2; lengths[1]];
            let mut b = vec![0xa3; lengths[2]];
            assert!(copy_matrix(expected, &matrix, &mut w, &mut s, &mut b).is_err());
            assert!(w.iter().all(|&x| x == 0xa1));
            assert!(s.iter().all(|&x| x == 0xa2));
            assert!(b.iter().all(|&x| x == 0xa3));
        }
    }
    // Equal byte lengths do not make a different rows/columns contract valid.
    let different = Layout::new(1, 96, 4, 32, MetadataDtype::BF16).unwrap();
    let (mut w, mut s, mut b) = ([0xa1; 48], [0xa2; 8], [0xa3; 8]);
    assert!(copy_matrix(different, &matrix, &mut w, &mut s, &mut b).is_err());
    assert_eq!((w, s, b), ([0xa1; 48], [0xa2; 8], [0xa3; 8]));
}

#[test]
fn matrix_upload_preserves_destinations_when_scalar_valid_domain_is_not_gpu_valid() {
    let layout = Layout::new(1, 32, 4, 32, MetadataDtype::F32).unwrap();
    for (scale, bias) in [
        (0., f32::from_bits(1)),
        (2f32.powi(-80), 0.),
        (f32::MAX, 0.),
    ] {
        let raw = [0u8; 16];
        let scales = scale.to_le_bytes();
        let biases = bias.to_le_bytes();
        let matrix = Matrix::from_le_bytes(layout, &raw, &scales, &biases).unwrap();
        let (mut w, mut s, mut b) = ([0xa1; 16], [0xa2; 4], [0xa3; 4]);
        assert!(matches!(
            copy_matrix(layout, &matrix, &mut w, &mut s, &mut b),
            Err(Error::Unsupported(_))
        ));
        assert_eq!((w, s, b), ([0xa1; 16], [0xa2; 4], [0xa3; 4]));
    }
}

#[test]
#[ignore = "Explicit Apple GPU original-byte Matrix upload; no graph/model execution"]
fn resident_matrix_upload_keeps_original_rows_and_padding() {
    autoreleasepool(|_| {
        let gpu = Gpu::new().unwrap();
        let pipelines = compile(&gpu).unwrap();
        for bits in [2, 3, 4, 5, 6, 8] {
            for dtype in [MetadataDtype::F16, MetadataDtype::BF16, MetadataDtype::F32] {
                let layout = Layout::new(5, 64, bits, 64, dtype).unwrap();
                let (w, s, b) = source(layout);
                let parent = Matrix::from_le_bytes(layout, &w[1..], &s[1..], &b[1..]).unwrap();
                let matrix = parent.row_range(1..4).unwrap();
                let layout = matrix.layout();
                let mut packed = Packed::allocate(&gpu, layout, &pipelines).unwrap();
                let mut weights = Buffer::<u16>::new(&gpu, layout.weight_bytes() / 2).unwrap();
                weights.with_bytes_mut(|b| b.fill(0xa1));
                packed.scales.with_bytes_mut(|b| b.fill(0xa2));
                packed.biases.with_bytes_mut(|b| b.fill(0xa3));
                let uploaded = packed.upload_matrix(&matrix, &mut weights);
                let actual_w = weights.with_bytes_mut(|b| b.to_vec());
                let actual_s = packed.scales.with_bytes_mut(|b| b.to_vec());
                let actual_b = packed.biases.with_bytes_mut(|b| b.to_vec());
                // Cold mapped upload submits no GPU work. Retain physical
                // destination bytes and original source before any assertion.
                println!(
                    "matrix_upload bits={bits} dtype={dtype:?} result={uploaded:?} source_w={w:?} source_s={s:?} source_b={b:?} weights={actual_w:?} scales={actual_s:?} biases={actual_b:?}"
                );
                uploaded.unwrap();
                let row_bytes = 64 * usize::from(bits) / 8;
                assert_eq!(actual_w, w[1 + row_bytes..1 + 4 * row_bytes]);
                let metadata = 3 * dtype.bytes();
                assert_eq!(
                    actual_s[..metadata],
                    s[1 + dtype.bytes()..1 + 4 * dtype.bytes()]
                );
                assert_eq!(
                    actual_b[..metadata],
                    b[1 + dtype.bytes()..1 + 4 * dtype.bytes()]
                );
                assert!(actual_s[metadata..].iter().all(|&b| b == 0));
                assert!(actual_b[metadata..].iter().all(|&b| b == 0));
                assert_eq!(
                    packed_bytes(layout).unwrap(),
                    actual_w.len() + actual_s.len() + actual_b.len()
                );
                let wrong = parent.row_range(0..2).unwrap();
                let rejected = packed.upload_matrix(&wrong, &mut weights);
                let after_w = weights.with_bytes_mut(|b| b.to_vec());
                let after_s = packed.scales.with_bytes_mut(|b| b.to_vec());
                let after_b = packed.biases.with_bytes_mut(|b| b.to_vec());
                println!(
                    "matrix_upload_rejected bits={bits} dtype={dtype:?} result={rejected:?} weights={after_w:?} scales={after_s:?} biases={after_b:?}"
                );
                assert!(rejected.is_err());
                assert_eq!(after_w, actual_w);
                assert_eq!(after_s, actual_s);
                assert_eq!(after_b, actual_b);
            }
        }
    });
}
