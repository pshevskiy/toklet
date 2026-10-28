use super::*;
use crate::weights::range_tests::Fixture;
use logical::{RowPartition, plan};

fn partition(parts: &[(&str, Range<usize>)]) -> RowPartition {
    RowPartition {
        physical_name: "x.weight".into(),
        parts: parts
            .iter()
            .map(|(name, rows)| ((*name).into(), rows.clone()))
            .collect(),
    }
}
fn coefficient(dtype: MetadataDtype, value: f32) -> Vec<u8> {
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
/// Synthetic GQA-like row partition: two Q rows, one K and one V. This tests
/// loading/accounting only; it is not a newly admitted model architecture.
pub(crate) fn split_source(bits: u8, kind: MetadataDtype) -> (Fixture, Source) {
    let layout = Layout::new(4, 32, bits, 32, kind).unwrap();
    let scales = (1..=4).flat_map(|v| coefficient(kind, v as f32)).collect();
    let biases = (0..4)
        .flat_map(|v| coefficient(kind, -(v as f32)))
        .collect();
    let f = Fixture::new(vec![
        (
            "x.weight",
            Dtype::U32,
            vec![4, layout.row_bytes() / 4],
            vec![0xff; layout.weight_bytes()],
        ),
        ("x.scales", dtype(kind), vec![4, 1], scales),
        ("x.biases", dtype(kind), vec![4, 1], biases),
    ]);
    let mut source = row_tests::wrapper(f.source(), Storage::Affine(layout), vec![4, 32]);
    source.logical = plan(
        &source.tensors,
        &[partition(&[
            ("q.weight", 0..2),
            ("k.weight", 2..3),
            ("v.weight", 3..4),
        ])],
    )
    .unwrap();
    (f, source)
}

#[test]
fn partitions_preserve_effective_specs_and_unique_payload_all_formats() {
    for bits in [2, 3, 4, 5, 6, 8] {
        for kind in [MetadataDtype::F16, MetadataDtype::BF16, MetadataDtype::F32] {
            let (_f, mut source) = split_source(bits, kind);
            assert_eq!(source.tensors().len(), 1);
            assert_eq!(source.logical_tensors().len(), 3);
            let total: usize = source
                .logical_tensors()
                .values()
                .map(|v| v.tensor().payload_bytes())
                .sum();
            assert_eq!(total, source.tensor_payload_bytes());
            for (name, start, rows) in [("q.weight", 0, 2), ("k.weight", 2, 1), ("v.weight", 3, 1)]
            {
                let spec = source.logical_tensors()[name].tensor();
                assert_eq!(spec.shape(), [rows, 32]);
                let Storage::Affine(layout) = spec.storage() else {
                    panic!()
                };
                assert_eq!(layout, Layout::new(rows, 32, bits, 32, kind).unwrap());
                let mut w = vec![0; layout.weight_bytes()];
                let mut s = vec![0; layout.metadata_bytes()];
                let mut b = s.clone();
                source
                    .read_logical_affine(name, &mut w, &mut s, &mut b)
                    .unwrap();
                assert_eq!(w, vec![0xff; layout.weight_bytes()]);
                let matrix = Matrix::from_le_bytes(layout, &w, &s, &b).unwrap();
                for row in 0..rows {
                    let original = start + row;
                    let expected = ((1usize << bits) - 1) * (original + 1) - original;
                    for column in 0..32 {
                        assert_eq!(matrix.value(row, column).unwrap(), expected as f32);
                    }
                }
            }
        }
    }
}

#[test]
fn plan_rejects_nonpartitions_collisions_and_invalid_metadata() {
    let (_f, source) = split_source(4, MetadataDtype::F16);
    for parts in [
        vec![],
        vec![("q", 0..1)],
        vec![("q", 1..4)],
        vec![("q", 0..2), ("k", 1..4)],
        vec![("q", 0..2), ("k", 3..4)],
        vec![("q", 0..0), ("k", 0..4)],
        vec![("q", 0..5)],
        vec![("q", 0..2), ("q", 2..4)],
        vec![("", 0..4)],
        vec![("q", 0..usize::MAX)],
    ] {
        assert!(plan(&source.tensors, &[partition(&parts)]).is_err());
    }
    let p = || partition(&[("q", 0..4)]);
    assert!(plan(&source.tensors, &[p(), p()]).is_err());
    let mut missing = p();
    missing.physical_name = "absent".into();
    assert!(plan(&source.tensors, &[missing]).is_err());
    let mut physical = source.tensors.clone();
    physical.insert("q".into(), physical["x.weight"].clone());
    assert!(plan(&physical, &[p()]).is_err());
    for change in 0..3 {
        let mut physical = source.tensors.clone();
        let spec = physical.get_mut("x.weight").unwrap();
        match change {
            0 => spec.payload_bytes += 1,
            1 => spec.shape = vec![128],
            _ => spec.shape = vec![usize::MAX, 32],
        }
        assert!(plan(&physical, &[]).is_err());
    }
    let norm = TensorSpec {
        shape: vec![4],
        storage: Storage::Float(MetadataDtype::F16),
        payload_bytes: 8,
    };
    assert!(plan(&BTreeMap::from([("x.weight".into(), norm)]), &[p()]).is_err());
    let too_many = (0..8193)
        .map(|i| {
            (
                i.to_string(),
                TensorSpec {
                    shape: vec![1],
                    storage: Storage::Float(MetadataDtype::F16),
                    payload_bytes: 2,
                },
            )
        })
        .collect();
    assert!(plan(&too_many, &[]).is_err());
}

#[test]
fn logical_float_selection_and_identity_have_exact_original_values() {
    for kind in [MetadataDtype::F16, MetadataDtype::BF16, MetadataDtype::F32] {
        let values: Vec<_> = (0..128).map(|v| v as f32 - 64.).collect();
        let raw = values.iter().flat_map(|&v| coefficient(kind, v)).collect();
        let f = Fixture::new(vec![("x.weight", dtype(kind), vec![4, 32], raw)]);
        let mut source = row_tests::wrapper(f.source(), Storage::Float(kind), vec![4, 32]);
        let mut full = Vec::new();
        source
            .visit_logical_float("x.weight", |v| {
                full.push(v);
                Ok(())
            })
            .unwrap();
        assert_eq!(full, values);
        source.logical = plan(
            &source.tensors,
            &[partition(&[("gate", 0..2), ("up", 2..4)])],
        )
        .unwrap();
        for (name, range) in [("gate", 0..64), ("up", 64..128)] {
            let mut actual = Vec::new();
            source
                .visit_logical_float(name, |v| {
                    actual.push(v);
                    Ok(())
                })
                .unwrap();
            assert_eq!(actual, values[range]);
        }
    }
}

#[test]
fn logical_whole_audit_is_preserved_but_rows_and_lookup_errors_poison() {
    use crate::weights::audit::tests::Fixture as Audited;
    for rows in [false, true] {
        let mut f = Audited::new(vec![("x.weight", Dtype::F32, vec![4, 32], vec![0; 512])]);
        let raw = std::mem::replace(&mut f.source, weights::Source::open(&f.path).unwrap());
        let mut source = row_tests::wrapper(raw, Storage::Float(MetadataDtype::F32), vec![4, 32]);
        source.visit_logical_float("x.weight", |_| Ok(())).unwrap();
        assert_eq!(source.content_identity().unwrap().verified_reads(), 1);
        source.visit_logical_float("x.weight", |_| Ok(())).unwrap();
        assert_eq!(source.content_identity().unwrap().verified_reads(), 2);
        if rows {
            source.logical = plan(&source.tensors, &[partition(&[("q", 0..4)])]).unwrap();
        }
        let name = if rows { "q" } else { "absent" };
        assert!(
            source
                .visit_logical_float(name, |_| panic!("rejected read delivered"))
                .is_err()
        );
        assert!(source.content_identity().is_none());
        assert!(
            source
                .visit_float("x.weight", |_| panic!("poisoned read delivered"))
                .is_err()
        );
    }
    let mut f = Audited::new(vec![
        ("x.weight", Dtype::U32, vec![4, 4], vec![0; 64]),
        ("x.scales", Dtype::F16, vec![4, 1], vec![0; 8]),
        ("x.biases", Dtype::F16, vec![4, 1], vec![0; 8]),
    ]);
    let raw = std::mem::replace(&mut f.source, weights::Source::open(&f.path).unwrap());
    let layout = Layout::new(4, 32, 4, 32, MetadataDtype::F16).unwrap();
    let mut source = row_tests::wrapper(raw, Storage::Affine(layout), vec![4, 32]);
    source
        .read_logical_affine("x.weight", &mut [0; 64], &mut [0; 8], &mut [0; 8])
        .unwrap();
    assert_eq!(source.content_identity().unwrap().verified_reads(), 3);
    source.logical = plan(&source.tensors, &[partition(&[("q", 0..4)])]).unwrap();
    let mut w = [0xee; 64];
    let mut s = [0xee; 8];
    let mut b = [0xee; 8];
    assert!(
        source
            .read_logical_affine("q", &mut w, &mut s, &mut b)
            .is_err()
    );
    assert_eq!(w, [0xee; 64]);
    assert_eq!(s, [0xee; 8]);
    assert_eq!(b, [0xee; 8]);
    assert!(source.content_identity().is_none());
}

#[test]
fn existing_tied_and_untied_sources_keep_identity_plans_and_owner_counts() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mlx-llama-v1");
    for variant in ["tied", "untied"] {
        let source = Source::open_dense(
            &root.join(variant),
            LoadOptions {
                context_limit: 8,
                memory_limit_bytes: weights::INITIAL_MEMORY_LIMIT_BYTES,
            },
        )
        .unwrap();
        assert_eq!(source.logical_tensors().len(), source.tensors().len());
        assert_eq!(
            source.logical_tensors().contains_key("lm_head.weight"),
            variant == "untied"
        );
        for (name, physical) in source.tensors() {
            assert_eq!(source.logical_tensors()[name].tensor(), physical);
        }
        let total: usize = source
            .logical_tensors()
            .values()
            .map(|v| v.tensor().payload_bytes())
            .sum();
        assert_eq!(total, source.tensor_payload_bytes());
    }
}
