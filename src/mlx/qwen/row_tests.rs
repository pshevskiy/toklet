use super::*;
use crate::weights::range_tests::Fixture;

pub(super) fn wrapper(source: weights::Source, storage: Storage, shape: Vec<usize>) -> Source {
    let payload_bytes = match storage {
        Storage::Affine(layout) => layout.total_bytes(),
        Storage::Float(dtype) => shape.iter().product::<usize>() * dtype.bytes(),
    };
    let tensors = BTreeMap::from([(
        "x.weight".into(),
        TensorSpec {
            shape,
            storage,
            payload_bytes,
        },
    )]);
    Source {
        declared_positions: 8,
        declared_sliding_window: None,
        config: Config {
            vocab_size: 1,
            hidden_size: 32,
            intermediate_size: 32,
            num_layers: 1,
            query_heads: 1,
            kv_heads: 1,
            head_dim: 32,
            max_positions: 8,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.,
        },
        architecture: Architecture::Llama,
        tied_output: true,
        position_plan: crate::model::PositionPlan::Ordinary,
        rotary_layers: None,
        tensors: tensors.clone(),
        logical: logical::plan(&tensors, &[]).unwrap(),
        source,
        tensor_payload_bytes: payload_bytes,
        admitted_load_bytes: METADATA_ALLOWANCE + STREAM_BYTES + payload_bytes,
    }
}
fn encode(dtype: MetadataDtype, value: f32) -> Vec<u8> {
    match dtype {
        MetadataDtype::F16 => half::f16::from_f32(value).to_le_bytes().to_vec(),
        MetadataDtype::BF16 => half::bf16::from_f32(value).to_le_bytes().to_vec(),
        MetadataDtype::F32 => value.to_le_bytes().to_vec(),
    }
}
fn dtype(value: MetadataDtype) -> Dtype {
    match value {
        MetadataDtype::F16 => Dtype::F16,
        MetadataDtype::BF16 => Dtype::BF16,
        MetadataDtype::F32 => Dtype::F32,
    }
}
fn affine_fixture(layout: Layout, bad_row: Option<usize>) -> (Fixture, [Vec<u8>; 3]) {
    let codes: Vec<_> = (0..layout.weight_bytes())
        .map(|i| (i % 251) as u8)
        .collect();
    let groups = layout.columns() / layout.group_size();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    for row in 0..layout.rows() {
        for _ in 0..groups {
            scales.extend(encode(
                layout.metadata_dtype(),
                if bad_row == Some(row) {
                    f32::NAN
                } else {
                    (row + 1) as f32
                },
            ));
            biases.extend(encode(layout.metadata_dtype(), -(row as f32)));
        }
    }
    let f = Fixture::new(vec![
        (
            "x.weight",
            Dtype::U32,
            vec![layout.rows(), layout.row_bytes() / 4],
            codes.clone(),
        ),
        (
            "x.scales",
            dtype(layout.metadata_dtype()),
            vec![layout.rows(), groups],
            scales.clone(),
        ),
        (
            "x.biases",
            dtype(layout.metadata_dtype()),
            vec![layout.rows(), groups],
            biases.clone(),
        ),
    ]);
    (f, [codes, scales, biases])
}

#[test]
fn selected_affine_rows_match_exact_original_bytes_all_formats() {
    for bits in [2, 3, 4, 5, 6, 8] {
        for dtype in [MetadataDtype::F16, MetadataDtype::BF16, MetadataDtype::F32] {
            let layout = Layout::new(5, 128, bits, 32, dtype).unwrap();
            let (f, original) = affine_fixture(layout, None);
            assert_eq!(f.data_start % 4, 1);
            let mut source = wrapper(f.source(), Storage::Affine(layout), vec![5, 128]);
            for rows in [0..1, 1..4, 4..5, 0..5] {
                let count = rows.len();
                let mut w = vec![0xee; count * 128 * usize::from(bits) / 8];
                let mut s = vec![0xee; count * 4 * dtype.bytes()];
                let mut b = s.clone();
                let selected = source
                    .read_affine_rows("x.weight", rows.clone(), &mut w, &mut s, &mut b)
                    .unwrap();
                assert_eq!(selected, Layout::new(count, 128, bits, 32, dtype).unwrap());
                let ws = 128 * usize::from(bits) / 8;
                let ms = 4 * dtype.bytes();
                assert_eq!(w, original[0][rows.start * ws..rows.end * ws]);
                assert_eq!(s, original[1][rows.start * ms..rows.end * ms]);
                assert_eq!(b, original[2][rows.start * ms..rows.end * ms]);
                let full = Matrix::from_le_bytes(layout, &original[0], &original[1], &original[2])
                    .unwrap();
                let selected = Matrix::from_le_bytes(selected, &w, &s, &b).unwrap();
                let mut expected = [0.; 128];
                let mut actual = [0.; 128];
                for row in 0..count {
                    full.copy_row(rows.start + row, &mut expected).unwrap();
                    selected.copy_row(row, &mut actual).unwrap();
                    assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
                }
            }
        }
    }
}

#[test]
fn affine_all_destination_sizes_and_ranges_validate_before_any_writes() {
    let layout = Layout::new(5, 128, 4, 32, MetadataDtype::F16).unwrap();
    let (f, _) = affine_fixture(layout, None);
    let mut source = wrapper(f.source(), Storage::Affine(layout), vec![5, 128]);
    for bad in 0..3 {
        let mut w = vec![0xee; 128 - usize::from(bad == 0)];
        let mut s = vec![0xee; 16 - usize::from(bad == 1)];
        let mut b = vec![0xee; 16 - usize::from(bad == 2)];
        let before = [w.clone(), s.clone(), b.clone()];
        assert!(
            source
                .read_affine_rows("x.weight", 1..3, &mut w, &mut s, &mut b)
                .is_err()
        );
        assert_eq!([w, s, b], before);
    }
    for rows in [
        0..0,
        std::ops::Range { start: 3, end: 2 },
        4..6,
        usize::MAX..usize::MAX,
    ] {
        let mut w = [0xee; 128];
        let mut s = [0xee; 16];
        let mut b = [0xee; 16];
        assert!(
            source
                .read_affine_rows("x.weight", rows, &mut w, &mut s, &mut b)
                .is_err()
        );
        assert_eq!(w, [0xee; 128]);
        assert_eq!(s, [0xee; 16]);
        assert_eq!(b, [0xee; 16]);
    }
    let mut w = [0xee; 128];
    let mut s = [0xee; 16];
    let mut b = [0xee; 16];
    assert!(
        source
            .read_affine_rows("missing", 1..3, &mut w, &mut s, &mut b)
            .is_err()
    );
    assert_eq!(w, [0xee; 128]);
    assert_eq!(s, [0xee; 16]);
    assert_eq!(b, [0xee; 16]);
    source
        .read_affine_rows("x.weight", 1..3, &mut w, &mut s, &mut b)
        .unwrap();
}

#[test]
fn selected_affine_metadata_only_is_validated_and_whole_read_still_checks_all() {
    for dtype in [MetadataDtype::F16, MetadataDtype::BF16, MetadataDtype::F32] {
        let layout = Layout::new(5, 128, 4, 32, dtype).unwrap();
        let (f, _) = affine_fixture(layout, Some(0));
        let mut source = wrapper(f.source(), Storage::Affine(layout), vec![5, 128]);
        let mut w = vec![0; 128];
        let mut s = vec![0; 8 * dtype.bytes()];
        let mut b = s.clone();
        source
            .read_affine_rows("x.weight", 1..3, &mut w, &mut s, &mut b)
            .unwrap();
        assert!(
            source
                .read_affine_rows("x.weight", 0..2, &mut w, &mut s, &mut b)
                .is_err()
        );
        assert!(
            source
                .read_affine(
                    "x.weight",
                    &mut vec![0; layout.weight_bytes()],
                    &mut vec![0; layout.metadata_bytes()],
                    &mut vec![0; layout.metadata_bytes()]
                )
                .is_err()
        );
        source
            .read_affine_rows("x.weight", 1..3, &mut w, &mut s, &mut b)
            .unwrap();
        assert!(source.content_identity().is_none());
    }
}

#[test]
fn floating_rows_decode_exact_nonsuffix_values_and_recover_after_consumer_failure() {
    for dtype_value in [MetadataDtype::F16, MetadataDtype::BF16, MetadataDtype::F32] {
        let values: Vec<_> = (0..40)
            .map(|n| if n == 8 { -0.0 } else { n as f32 - 20.0 })
            .collect();
        let bytes: Vec<_> = values
            .iter()
            .flat_map(|&v| encode(dtype_value, v))
            .collect();
        let f = Fixture::new(vec![("x.weight", dtype(dtype_value), vec![5, 8], bytes)]);
        let mut source = wrapper(f.source(), Storage::Float(dtype_value), vec![5, 8]);
        for rows in [0..1, 1..3, 4..5, 0..5] {
            let mut actual = Vec::new();
            source
                .visit_float_rows("x.weight", rows.clone(), |v| {
                    actual.push(v.to_bits());
                    Ok(())
                })
                .unwrap();
            assert_eq!(
                actual,
                values[rows.start * 8..rows.end * 8]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        let mut count = 0;
        assert!(
            source
                .visit_float_rows("x.weight", 1..3, |_| {
                    count += 1;
                    Err(invalid("consumer"))
                })
                .is_err()
        );
        assert_eq!(count, 1);
        count = 0;
        source
            .visit_float_rows("x.weight", 1..3, |_| {
                count += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(count, 16);
        for rows in [
            0..0,
            std::ops::Range { start: 3, end: 2 },
            4..6,
            usize::MAX..usize::MAX,
        ] {
            assert!(
                source
                    .visit_float_rows("x.weight", rows, |_| {
                        panic!("invalid range delivered value")
                    })
                    .is_err()
            );
        }
        source.tensors.get_mut("x.weight").unwrap().shape = vec![40];
        assert!(
            source
                .visit_float_rows("x.weight", 0..1, |_| panic!("rank-one delivered value"))
                .is_err()
        );
        source.tensors.get_mut("x.weight").unwrap().shape = vec![2, usize::MAX];
        assert!(
            source
                .visit_float_rows("x.weight", 0..1, |_| panic!("overflow delivered value"))
                .is_err()
        );
    }
}

#[test]
fn floating_unselected_nan_is_not_a_whole_tensor_validation() {
    for dtype_value in [MetadataDtype::F16, MetadataDtype::BF16, MetadataDtype::F32] {
        let bytes: Vec<_> = [f32::NAN, 2., 3., 4.]
            .into_iter()
            .flat_map(|v| encode(dtype_value, v))
            .collect();
        let f = Fixture::new(vec![("x.weight", dtype(dtype_value), vec![2, 2], bytes)]);
        let mut source = wrapper(f.source(), Storage::Float(dtype_value), vec![2, 2]);
        let mut values = Vec::new();
        source
            .visit_float_rows("x.weight", 1..2, |v| {
                values.push(v);
                Ok(())
            })
            .unwrap();
        assert_eq!(values, [3., 4.]);
        assert!(
            source
                .visit_float_rows("x.weight", 0..1, |_| panic!("nonfinite delivered"))
                .is_err()
        );
        assert!(source.visit_float("x.weight", |_| Ok(())).is_err());
        source
            .visit_float_rows("x.weight", 1..2, |_| Ok(()))
            .unwrap();
    }
}

#[test]
fn audited_public_row_read_rejects_before_delivery_and_poison_is_sticky() {
    use crate::weights::audit::tests::Fixture as Audited;
    let mut f = Audited::new(vec![("x.weight", Dtype::F32, vec![2, 2], vec![0; 16])]);
    let raw = std::mem::replace(&mut f.source, weights::Source::open(&f.path).unwrap());
    let mut source = wrapper(raw, Storage::Float(MetadataDtype::F32), vec![2, 2]);
    source.visit_float("x.weight", |_| Ok(())).unwrap();
    assert!(source.content_identity().is_some());
    assert!(
        source
            .visit_float_rows("x.weight", 0..2, |_| panic!("audited range delivered"))
            .is_err()
    );
    assert!(source.content_identity().is_none());
    assert!(
        source
            .visit_float("x.weight", |_| panic!("poisoned whole read delivered"))
            .is_err()
    );
    let mut f = Audited::new(vec![
        ("x.weight", Dtype::U32, vec![1, 8], vec![0; 32]),
        ("x.scales", Dtype::F16, vec![1, 1], vec![0; 2]),
        ("x.biases", Dtype::F16, vec![1, 1], vec![0; 2]),
    ]);
    let raw = std::mem::replace(&mut f.source, weights::Source::open(&f.path).unwrap());
    let layout = Layout::new(1, 32, 8, 32, MetadataDtype::F16).unwrap();
    let mut source = wrapper(raw, Storage::Affine(layout), vec![1, 32]);
    source
        .read_affine("x.weight", &mut [0; 32], &mut [0; 2], &mut [0; 2])
        .unwrap();
    assert!(source.content_identity().is_some());
    let mut w = [0xee; 32];
    let mut s = [0xee; 2];
    let mut b = [0xee; 2];
    assert!(
        source
            .read_affine_rows("x.weight", 0..1, &mut w, &mut s, &mut b)
            .is_err()
    );
    assert_eq!(w, [0xee; 32]);
    assert_eq!(s, [0xee; 2]);
    assert_eq!(b, [0xee; 2]);
    assert!(source.content_identity().is_none());
    assert!(
        source
            .read_affine("x.weight", &mut w, &mut s, &mut b)
            .is_err()
    );
}

#[test]
fn affine_partial_read_failure_requires_discard_and_can_be_retried() {
    let layout = Layout::new(5, 128, 4, 32, MetadataDtype::F16).unwrap();
    let (f, original) = affine_fixture(layout, None);
    let mut source = wrapper(f.source(), Storage::Affine(layout), vec![5, 128]);
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(f.path.join("model.safetensors"))
        .unwrap();
    file.set_len(f.original.len() as u64 - 1).unwrap();
    let mut w = [0xee; 64];
    let mut s = [0xee; 8];
    let mut b = [0xee; 8];
    assert!(
        source
            .read_affine_rows("x.weight", 4..5, &mut w, &mut s, &mut b)
            .is_err()
    );
    assert_eq!(w, original[0][256..320]);
    assert_eq!(s, original[1][32..40]);
    assert_eq!(b, [0xee; 8]);
    // All three buffers remain unpublished; the next complete read replaces them.
    std::fs::write(f.path.join("model.safetensors"), &f.original).unwrap();
    source
        .read_affine_rows("x.weight", 4..5, &mut w, &mut s, &mut b)
        .unwrap();
    assert_eq!(b, original[2][32..40]);
    assert!(
        source
            .visit_float_rows("x.weight", 0..1, |_| panic!("affine decoded as float"))
            .is_err()
    );
}

#[test]
fn floating_rows_cross_raw_buffer_boundaries_without_losing_values() {
    for dtype_value in [MetadataDtype::F16, MetadataDtype::BF16, MetadataDtype::F32] {
        let bytes: Vec<_> = (0..120_000)
            .flat_map(|i| encode(dtype_value, (i % 127) as f32 - 63.))
            .collect();
        let f = Fixture::new(vec![(
            "x.weight",
            dtype(dtype_value),
            vec![3, 40_000],
            bytes,
        )]);
        let mut source = wrapper(f.source(), Storage::Float(dtype_value), vec![3, 40_000]);
        let mut count = 0;
        source
            .visit_float_rows("x.weight", 1..2, |v| {
                assert_eq!(v, ((40_000 + count) % 127) as f32 - 63.);
                count += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(count, 40_000);
        let mut w = [0xee; 1];
        let mut s = [0xee; 1];
        let mut b = [0xee; 1];
        assert!(
            source
                .read_affine_rows("x.weight", 0..1, &mut w, &mut s, &mut b)
                .is_err()
        );
        assert_eq!([w, s, b], [[0xee]; 3]);
    }
}
