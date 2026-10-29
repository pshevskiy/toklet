use super::*;
use crate::mlx::{
    affine::{Layout, MetadataDtype},
    gemma3::{Dtype, PhysicalTensor, TensorPlan},
};
use std::{
    collections::BTreeMap,
    io::{self, Cursor},
};

const PREFIX: &[u8] = b"private synthetic prefix; never a public 270M artifact";
const CONFIG: &[u8] =
    include_bytes!("../../../../tests/fixtures/gemma3-270m-metadata-v1/config.json");
const INDEX: &[u8] =
    include_bytes!("../../../../tests/fixtures/gemma3-270m-metadata-v1/index.json");
const HEADER: &[u8] =
    include_bytes!("../../../../tests/fixtures/gemma3-270m-metadata-v1/header.json");
fn hash(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}
fn start() -> Sha256 {
    let mut s = Sha256::new();
    s.update(PREFIX);
    s
}
fn options(raw: &[u8]) -> LoadOptions {
    let mut sha = start();
    sha.update(raw);
    LoadOptions {
        identity: ContentIdentity {
            config_sha256: hash(b"synthetic config"),
            index_sha256: hash(b"synthetic index"),
            header_sha256: hash(PREFIX),
            container_sha256: sha.finalize().into(),
            container_bytes: (PREFIX.len() + raw.len()) as u64,
        },
        memory_limit_bytes: usize::MAX,
    }
}
fn synthetic() -> (Plan, Vec<u8>) {
    let layout = Layout::new(2, 64, 4, 64, MetadataDtype::BF16).unwrap();
    let mut raw = Vec::new();
    let mut physical = BTreeMap::new();
    for (name, dtype, shape, bytes) in [
        ("table.weight", Dtype::U32, vec![2, 8], vec![0x71; 64]),
        (
            "table.scales",
            Dtype::Bf16,
            vec![2, 1],
            [
                bf16::from_f32(-0.25).to_le_bytes(),
                bf16::from_f32(0.5).to_le_bytes(),
            ]
            .concat(),
        ),
        (
            "table.biases",
            Dtype::Bf16,
            vec![2, 1],
            [
                bf16::from_f32(1.0).to_le_bytes(),
                bf16::from_f32(-2.0).to_le_bytes(),
            ]
            .concat(),
        ),
        (
            "norm",
            Dtype::Bf16,
            vec![2],
            [
                bf16::from_f32(0.75).to_le_bytes(),
                bf16::from_f32(-0.0).to_le_bytes(),
            ]
            .concat(),
        ),
    ] {
        let begin = raw.len();
        raw.extend(bytes);
        physical.insert(
            name.into(),
            PhysicalTensor {
                dtype,
                shape,
                range: begin..raw.len(),
            },
        );
    }
    let plan = Plan {
        layers: super::super::layers(),
        logical: vec![
            TensorPlan {
                name: "table.weight".into(),
                shape: Shape::Matrix {
                    rows: 2,
                    columns: 64,
                },
                storage: Storage::Affine(layout),
            },
            TensorPlan {
                name: "norm".into(),
                shape: Shape::Vector(2),
                storage: Storage::Bf16Vector,
            },
        ],
        physical,
        payload_bytes: raw.len(),
    };
    (plan, raw)
}
fn admitted(
    mut reader: impl Read,
    plan: Plan,
    options: LoadOptions,
) -> Result<PayloadStore, Error> {
    load_admitted(&mut reader, plan, start(), PREFIX.len() as u64, options)
}
struct Fragmented {
    bytes: Cursor<Vec<u8>>,
    width: usize,
    fail: Option<u64>,
}
impl Read for Fragmented {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.fail.is_some_and(|n| self.bytes.position() >= n) {
            return Err(io::Error::other("injected read failure"));
        }
        let n = out.len().min(self.width);
        self.bytes.read(&mut out[..n])
    }
}
#[test]
fn original_ranges_stream_identity_and_raw_gamma_bits_are_preserved() {
    for width in [1, 3, 65536] {
        let (plan, raw) = synthetic();
        let expected = options(&raw);
        let store = admitted(
            Fragmented {
                bytes: Cursor::new(raw.clone()),
                width,
                fail: None,
            },
            plan,
            expected,
        )
        .unwrap();
        assert_eq!(store.content_identity(), &expected.identity);
        assert_eq!(store.payload, raw);
        assert_eq!(store.report.packed_matrices, 1);
        assert_eq!(store.report.raw_bf16_norms, 1);
        assert_eq!(
            store.report.admitted_load_bytes,
            store.payload.capacity() + METADATA_RESERVATION + STREAM_BYTES
        );
        assert_eq!(
            store.bytes("table.weight").unwrap().as_ptr(),
            store.payload.as_ptr()
        );
        assert_eq!(
            store
                .raw_gamma("norm")
                .unwrap()
                .iter()
                .map(|g| g.to_bits())
                .collect::<Vec<_>>(),
            [0x3f40, 0x8000]
        );
        let matrix = store.matrix("table.weight").unwrap();
        assert_eq!(matrix.layout().total_bytes(), 72);
        assert!(store.matrix("norm").is_err());
        assert!(store.raw_gamma("table.weight").is_err());
        assert!(
            store.prepare_raw_gamma(12, 0, usize::MAX).is_err(),
            "private tiny plan must not reach the selected-profile adapter"
        );
    }
}
#[test]
fn truncated_failed_trailing_and_wrong_identity_streams_publish_no_store() {
    for mode in 0..6 {
        let (plan, mut raw) = synthetic();
        let mut expected = options(&raw);
        let fail = if mode == 0 { Some(3) } else { None };
        match mode {
            1 => {
                raw.pop();
            }
            2 => raw.push(0),
            3 => expected.identity.container_sha256[0] ^= 1,
            4 => expected.identity.container_bytes += 1,
            5 => raw[0] ^= 1,
            _ => {}
        }
        assert!(
            admitted(
                Fragmented {
                    bytes: Cursor::new(raw),
                    width: 1,
                    fail
                },
                plan,
                expected
            )
            .is_err()
        );
    }
}
#[test]
fn reauthenticated_nonfinite_and_overflowing_values_still_fail() {
    for (offset, bits) in [(64, 0x7fc0u16), (68, 0x7f80), (72, 0xff80), (64, 0x7f7f)] {
        let (plan, mut raw) = synthetic();
        raw[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
        let expected = options(&raw);
        assert!(admitted(Cursor::new(raw), plan, expected).is_err());
    }
    assert!(decode_gamma(&[0, 0, 0], 2).is_err());
    assert!(multiply(usize::MAX, 2).is_err());
    assert!(add(usize::MAX, 1).is_err());
}
#[test]
fn budget_refuses_payload_before_read_and_counts_temporary_gammas() {
    let (plan, raw) = synthetic();
    let mut expected = options(&raw);
    let required = raw.len() + METADATA_RESERVATION + STREAM_BYTES;
    expected.memory_limit_bytes = required - 1;
    struct Never;
    impl Read for Never {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            panic!("budget must fail before payload read")
        }
    }
    assert!(
        matches!(admitted(Never,plan,expected),Err(Error::Budget{required:r,..})if r==required)
    );
    let (plan, raw) = synthetic();
    let mut expected = options(&raw);
    expected.memory_limit_bytes = required;
    let store = admitted(Cursor::new(raw), plan, expected).unwrap();
    assert_eq!(
        temporary_bytes(store.plan(), 3, 4).unwrap(),
        3 * size_of::<LayerWeights<Matrix<'_>>>() + 4 * size_of::<LayerSpec>() + 4
    );
    assert!(check_budget(required, required).is_ok());
}
#[test]
fn public_metadata_identity_header_and_budget_checks_need_no_real_payload() {
    let mut identity = ContentIdentity {
        config_sha256: hash(CONFIG),
        index_sha256: hash(INDEX),
        header_sha256: hash(HEADER),
        container_sha256: [0; 32],
        container_bytes: 150939130,
    };
    let mut bytes = (HEADER.len() as u64).to_le_bytes().to_vec();
    bytes.extend(HEADER);
    let mut reader = Cursor::new(bytes.clone());
    identity.config_sha256[0] ^= 1;
    assert!(
        load(
            CONFIG,
            INDEX,
            &mut reader,
            LoadOptions {
                identity,
                memory_limit_bytes: usize::MAX
            }
        )
        .is_err()
    );
    assert_eq!(reader.position(), 0);
    identity.config_sha256 = hash(CONFIG);
    identity.index_sha256[0] ^= 1;
    assert!(
        load(
            CONFIG,
            INDEX,
            &mut reader,
            LoadOptions {
                identity,
                memory_limit_bytes: usize::MAX
            }
        )
        .is_err()
    );
    assert_eq!(reader.position(), 0);
    identity.index_sha256 = hash(INDEX);
    identity.header_sha256[0] ^= 1;
    assert!(
        load(
            CONFIG,
            INDEX,
            &mut reader,
            LoadOptions {
                identity,
                memory_limit_bytes: usize::MAX
            }
        )
        .is_err()
    );
    assert_eq!(reader.position(), bytes.len() as u64);
    identity.header_sha256 = hash(HEADER);
    let mut reader = Cursor::new(bytes);
    assert!(matches!(
        load(
            CONFIG,
            INDEX,
            &mut reader,
            LoadOptions {
                identity,
                memory_limit_bytes: METADATA_RESERVATION + STREAM_BYTES + 128
            }
        ),
        Err(Error::Budget { .. })
    ));
    assert_eq!(reader.position(), 8 + HEADER.len() as u64);
    for count in [0, MAX_HEADER_BYTES as u64 + 1, u64::MAX] {
        assert!(
            load(
                CONFIG,
                INDEX,
                Cursor::new(count.to_le_bytes()),
                LoadOptions {
                    identity,
                    memory_limit_bytes: usize::MAX
                }
            )
            .is_err()
        );
    }
}
#[test]
fn bounded_capacity_preserves_source_declarations_and_effective_window() {
    for capacity in [1, 12, 511, 512, 513, 544] {
        let config = execution_config(capacity).unwrap();
        assert_eq!(config.max_positions, capacity);
        assert_eq!(config.hidden_size, 640);
        assert_eq!(config.query_heads * config.head_dim, 1024);
        assert_eq!(
            512.min(capacity),
            if capacity < 512 { capacity } else { 512 }
        );
    }
    for capacity in [0, 545, usize::MAX] {
        assert!(execution_config(capacity).is_err());
    }
    let plan = inspect(CONFIG, INDEX, HEADER, 150939130).unwrap();
    assert_eq!(plan.declared_total_positions(), 32768);
    assert_eq!(plan.declared_sliding_window(), 512);
    assert_eq!(temporary_bytes(&plan, 0, 0).unwrap(), 55936 * 2);
}

// A private name-resolution fixture: all original names, but only 2x64 matrices
// and two-word norms (under 10 KiB total). It cannot enter public inspect/load
// and never claims to be a 270M payload or a numerically executable model.
fn tiny_names() -> PayloadStore {
    let mut logical = super::super::logical().unwrap();
    let layout = Layout::new(2, 64, 4, 64, MetadataDtype::BF16).unwrap();
    let mut raw = Vec::new();
    let mut physical = BTreeMap::new();
    for (ordinal, tensor) in logical.iter_mut().enumerate() {
        let marker = bf16::from_f32(ordinal as f32).to_le_bytes();
        let components = match tensor.storage {
            Storage::Affine(_) => {
                tensor.shape = Shape::Matrix {
                    rows: 2,
                    columns: 64,
                };
                tensor.storage = Storage::Affine(layout);
                let base = tensor.name.strip_suffix(".weight").unwrap();
                vec![
                    (tensor.name.clone(), Dtype::U32, vec![2, 8], vec![0; 64]),
                    (
                        format!("{base}.scales"),
                        Dtype::Bf16,
                        vec![2, 1],
                        [bf16::ONE.to_le_bytes(); 2].concat(),
                    ),
                    (
                        format!("{base}.biases"),
                        Dtype::Bf16,
                        vec![2, 1],
                        [marker; 2].concat(),
                    ),
                ]
            }
            Storage::Bf16Vector => {
                tensor.shape = Shape::Vector(2);
                vec![(
                    tensor.name.clone(),
                    Dtype::Bf16,
                    vec![2],
                    [marker; 2].concat(),
                )]
            }
        };
        for (name, dtype, shape, bytes) in components {
            let start = raw.len();
            raw.extend(bytes);
            physical.insert(
                name,
                PhysicalTensor {
                    dtype,
                    shape,
                    range: start..raw.len(),
                },
            );
        }
    }
    assert!(raw.len() < 10 * 1024);
    let plan = Plan {
        layers: super::super::layers(),
        logical,
        physical,
        payload_bytes: raw.len(),
    };
    let expected = options(&raw);
    admitted(Cursor::new(raw), plan, expected).unwrap()
}
#[test]
fn cold_adapter_resolves_all_names_once_and_accounts_actual_temporary_capacity() {
    let store = tiny_names();
    for capacity in [1, 511, 512, 544] {
        let input = store.prepare_raw_gamma(capacity, 777, usize::MAX).unwrap();
        assert_eq!(input.local_window, 512.min(capacity));
        assert_eq!(input.config.max_positions, capacity);
        assert_eq!(input.weights.embedding.value(0, 0).unwrap(), 0.0);
        assert_eq!(input.weights.final_gamma[0].to_f32(), 1.0);
        assert_eq!(input.specs.len(), 18);
        for (index, layer) in input.weights.layers.iter().enumerate() {
            for (field, offset) in [
                (&layer.q, 8),
                (&layer.k, 9),
                (&layer.v, 10),
                (&layer.o, 11),
                (&layer.gate, 12),
                (&layer.up, 13),
                (&layer.down, 14),
            ] {
                assert_eq!(field.value(0, 0).unwrap(), (13 * index + offset) as f32);
            }
            for (field, offset) in [
                (&layer.gamma.input, 2),
                (&layer.gamma.post_attention, 3),
                (&layer.gamma.pre_ff, 4),
                (&layer.gamma.post_ff, 5),
                (&layer.gamma.q, 6),
                (&layer.gamma.k, 7),
            ] {
                assert_eq!(field[0].to_f32(), (13 * index + offset) as f32);
            }
            assert_eq!(input.specs[index].kind, store.plan.layers()[index].kind);
            assert_eq!(
                input.specs[index].theta,
                store.plan.layers()[index].rope_theta
            );
        }
        let expected = store.payload.capacity()
            + METADATA_RESERVATION
            + 777
            + input_capacity(&input.weights, input.specs.capacity()).unwrap();
        assert_eq!(input.source_resident_bytes, expected);
        drop(input);
        assert!(store.prepare_raw_gamma(capacity, 777, expected).is_ok());
        assert!(matches!(
            store.prepare_raw_gamma(capacity, 777, expected - 1),
            Err(Error::Budget { .. })
        ));
    }
    assert!(
        store
            .prepare_raw_gamma(544, usize::MAX, usize::MAX)
            .is_err()
    );
}
