use super::*;
use crate::mlx::affine::MetadataDtype;
use crate::mlx::gemma4::{Dtype, PhysicalTensor, Shape, TensorPlan, layer_plans};
use std::io::{self, Cursor};

const PREFIX: &[u8] = b"synthetic header prefix, not real E2B metadata";
fn synthetic() -> (E2bPlan, Vec<u8>, LoadOptions) {
    let layout = Layout::new(2, 64, 4, 64, MetadataDtype::BF16).unwrap();
    let mut payload = Vec::new();
    let mut text = BTreeMap::new();
    for (name, dtype, shape, raw) in [
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
                bf16::from_f32(1.).to_le_bytes(),
                bf16::from_f32(-2.).to_le_bytes(),
            ]
            .concat(),
        ),
        (
            "norm",
            Dtype::Bf16,
            vec![2],
            [
                bf16::from_f32(0.75).to_le_bytes(),
                bf16::from_f32(-0.).to_le_bytes(),
            ]
            .concat(),
        ),
    ] {
        let start = payload.len();
        payload.extend(raw);
        text.insert(
            name.into(),
            PhysicalTensor {
                dtype,
                shape,
                range: start..payload.len(),
            },
        );
    }
    let text_bytes = payload.len();
    let excluded_raw = [0xff, 0xff, 0xff, 0x7f, 1, 2, 3, 4];
    payload.extend(excluded_raw);
    let excluded = BTreeMap::from([(
        "vision_tower.unexecuted".into(),
        PhysicalTensor {
            dtype: Dtype::U32,
            shape: vec![2],
            range: text_bytes..payload.len(),
        },
    )]);
    let plan = E2bPlan {
        layers: layer_plans(),
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
        text,
        excluded,
        text_payload_bytes: text_bytes,
        excluded_payload_bytes: 8,
        payload_bytes: payload.len(),
    };
    let mut sha = Sha256::new();
    sha.update(PREFIX);
    sha.update(&payload);
    let options = LoadOptions {
        container_bytes: (PREFIX.len() + payload.len()) as u64,
        container_sha256: sha.finalize().into(),
        memory_limit_bytes: usize::MAX,
        modality: ModalityPolicy::TextOnly,
    };
    (plan, payload, options)
}
fn identity_metadata() -> MetadataIdentity {
    MetadataIdentity::new(
        br#"{"synthetic":true}"#,
        br#"{"index":true}"#,
        PREFIX.len() as u64,
    )
}
fn start() -> Sha256 {
    let mut s = Sha256::new();
    s.update(PREFIX);
    s
}
fn rehash(options: &mut LoadOptions, payload: &[u8]) {
    let mut s = start();
    s.update(payload);
    options.container_sha256 = s.finalize().into();
}
struct Fragments {
    data: Cursor<Vec<u8>>,
    width: usize,
    fail_after: Option<u64>,
}
impl Read for Fragments {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.fail_after.is_some_and(|n| self.data.position() >= n) {
            return Err(io::Error::other("injected read failure"));
        }
        let n = out.len().min(self.width);
        self.data.read(&mut out[..n])
    }
}

#[test]
fn synthetic_stream_integrity_exclusion_borrowed_views_and_capacities() {
    for width in [1, 3, 65536] {
        let (plan, raw, options) = synthetic();
        let expected = reservation(&plan).unwrap().total;
        let mut reader = Fragments {
            data: Cursor::new(raw.clone()),
            width,
            fail_after: None,
        };
        let store =
            load_admitted(&mut reader, plan, start(), options, identity_metadata()).unwrap();
        assert_eq!(reader.data.position(), raw.len() as u64);
        let report = store.report();
        assert_eq!(report.container_sha256, options.container_sha256);
        assert_eq!((report.packed_matrices, report.fp32_vectors), (1, 1));
        assert_eq!(report.text_source_bytes, 76);
        assert_eq!(report.excluded_source_bytes, 8);
        assert_eq!(report.tensor_capacity_bytes, 80);
        assert_eq!(report.admitted_load_bytes, expected);
        assert_eq!(store.plan().text_tensors().len(), 4);
        assert_eq!(
            store
                .vector("norm")
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            [0.75_f32.to_bits(), (-0_f32).to_bits()]
        );
        assert!(store.vector("vision_tower.unexecuted").is_err());
        assert!(store.matrix("norm").is_err());
        let m = store.matrix("table.weight").unwrap();
        let mut row = [0.; 64];
        m.copy_row(0, &mut row).unwrap();
        for (i, &v) in row.iter().enumerate() {
            assert_eq!(v, if i % 2 == 0 { 0.75 } else { -0.75 });
        }
        let mut out = [0.; 2];
        m.matvec(&[1.; 64], &mut out).unwrap();
        assert_eq!(out, [0., 0.]);
    }
}

#[test]
fn digest_including_excluded_bytes_truncation_tail_and_io_fail_without_store() {
    for case in 0..5 {
        let (plan, mut raw, mut options) = synthetic();
        match case {
            0 => options.container_sha256[0] ^= 1,
            1 => {
                raw.pop();
            }
            2 => raw.push(0),
            3 => {
                let last = raw.len() - 1;
                raw[last] ^= 1;
            }
            _ => {}
        }
        let mut reader = Fragments {
            data: Cursor::new(raw),
            width: 1,
            fail_after: (case == 4).then_some(65),
        };
        assert!(load_admitted(&mut reader, plan, start(), options, identity_metadata()).is_err());
    }
}

#[test]
fn finite_vectors_metadata_and_both_affine_overflow_directions_are_required() {
    for case in 0..7 {
        let (plan, mut raw, mut options) = synthetic();
        let (offset, value) = match case {
            0 => (72, bf16::NAN),
            1 => (64, bf16::INFINITY),
            2 => (68, bf16::NEG_INFINITY),
            3 => (66, bf16::MAX),
            4 => (66, bf16::MIN),
            // Endpoints include code0; finite biases with scale of the same sign
            // can overflow even if the actual stored code is small.
            5 => (70, bf16::MAX),
            _ => (70, bf16::MIN),
        };
        raw[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
        if case >= 5 {
            raw[66..68]
                .copy_from_slice(&(if case == 5 { bf16::MAX } else { bf16::MIN }).to_le_bytes());
        }
        rehash(&mut options, &raw);
        assert!(
            load_admitted(
                &mut Cursor::new(raw),
                plan,
                start(),
                options,
                identity_metadata()
            )
            .is_err(),
            "case {case}"
        );
    }
    assert!(vector(&[0]).is_err());
}

#[test]
fn numerical_reservation_rejects_before_payload_reads_and_exact_budget_passes() {
    let (plan, raw, mut options) = synthetic();
    let required = reservation(&plan).unwrap().total;
    options.memory_limit_bytes = required - 1;
    let mut reader = Cursor::new(raw.clone());
    assert!(
        matches!(load_admitted(&mut reader,plan,start(),options,identity_metadata()),Err(Error::Budget{required:r,..}) if r==required)
    );
    assert_eq!(reader.position(), 0);
    let (plan, _, mut options) = synthetic();
    options.memory_limit_bytes = required;
    let store = load_admitted(
        &mut Cursor::new(raw),
        plan,
        start(),
        options,
        identity_metadata(),
    )
    .unwrap();
    assert_eq!(store.report().admitted_load_bytes, required);
    assert!(add(usize::MAX, 1).is_err());
    assert!(multiply(usize::MAX, 4).is_err());
}

#[test]
fn public_real_metadata_admits_header_then_rejects_budget_without_payload() {
    let config = include_bytes!("../../../../tests/fixtures/gemma4-e2b-metadata-v1/config.json");
    let index = include_bytes!("../../../../tests/fixtures/gemma4-e2b-metadata-v1/index.json");
    let header = include_bytes!("../../../../tests/fixtures/gemma4-e2b-metadata-v1/header.json");
    let mut prefix = (header.len() as u64).to_le_bytes().to_vec();
    prefix.extend_from_slice(header);
    let mut reader = Cursor::new(prefix);
    let options = LoadOptions {
        container_bytes: 3550670554,
        container_sha256: [0; 32],
        memory_limit_bytes: METADATA_RESERVATION + STREAM_BYTES,
        modality: ModalityPolicy::TextOnly,
    };
    let result = load(config, index, &mut reader, options);
    assert!(matches!(result,Err(Error::Budget{required,..}) if required>2_603_981_382));
    assert_eq!(reader.position(), 8 + header.len() as u64);
    reader.set_position(0);
    assert!(
        load(
            config,
            index,
            &mut reader,
            LoadOptions {
                modality: ModalityPolicy::Reject,
                ..options
            }
        )
        .is_err()
    );
    assert_eq!(reader.position(), 8 + header.len() as u64);
    for raw in [
        vec![],
        vec![0; 8],
        (MAX_HEADER_BYTES as u64 + 1).to_le_bytes().to_vec(),
    ] {
        assert!(load(config, index, Cursor::new(raw), options).is_err());
    }
}

#[test]
fn content_identity_uses_exact_metadata_and_completed_consumption() {
    let (plan, raw, options) = synthetic();
    let config = br#"{"synthetic":true}"#;
    let index = br#"{"index":true}"#;
    let store = load_admitted(
        &mut Cursor::new(raw),
        plan,
        start(),
        options,
        MetadataIdentity::new(config, index, PREFIX.len() as u64),
    )
    .unwrap();
    let id = *store.content_identity();
    assert_eq!(
        format!("{:x}", Sha256::digest(config)),
        "a8198524f58e72b56283ab71ebddada22840f108b46ef7165bb3fca9919c5558"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(index)),
        "2c5f4604b853198b10417ee859c33c4efaadefc4757adf596787b8115536a751"
    );
    assert_eq!(
        id.config_sha256(),
        &<[u8; 32]>::from(Sha256::digest(config))
    );
    assert_eq!(id.index_sha256(), &<[u8; 32]>::from(Sha256::digest(index)));
    assert_eq!(id.container_sha256(), &store.report().container_sha256);
    assert_eq!(id.container_bytes(), store.report().container_bytes);
    // Same JSON meaning has distinct byte identity. This is synthetic storage,
    // not a claim that the real E2B public config accepts arbitrary schemas.
    let config_spaced = br#"{ "synthetic" : true }"#;
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(config).unwrap(),
        serde_json::from_slice::<serde_json::Value>(config_spaced).unwrap()
    );
    let (plan, raw, options) = synthetic();
    let other = load_admitted(
        &mut Cursor::new(raw),
        plan,
        start(),
        options,
        MetadataIdentity::new(config_spaced, index, PREFIX.len() as u64),
    )
    .unwrap();
    assert_ne!(id.config_sha256(), other.content_identity().config_sha256());
    assert_eq!(id.index_sha256(), other.content_identity().index_sha256());
    assert_eq!(
        id.container_sha256(),
        other.content_identity().container_sha256()
    );
    drop(store); // Copied content evidence has no file/store lifetime dependency.
    assert_eq!(id.container_bytes(), options.container_bytes);
}

#[test]
fn consumed_count_mismatch_and_overflow_never_publish_identity() {
    for prefix in [PREFIX.len() as u64 - 1, PREFIX.len() as u64 + 1, u64::MAX] {
        let (plan, raw, options) = synthetic();
        let metadata = MetadataIdentity::new(b"config", b"index", prefix);
        assert!(load_admitted(&mut Cursor::new(raw), plan, start(), options, metadata).is_err());
    }
}
