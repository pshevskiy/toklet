use super::*;
use std::io::Cursor;

const EMBED: &str = "language_model.model.embed_tokens.weight";
const NORM: &str = "language_model.model.norm.weight";
const CONV: &str = "language_model.model.layers.0.linear_attn.conv1d.weight";
const CONFIG: &[u8] =
    include_bytes!("../../../../tests/fixtures/qwen35-4b-avan-metadata-v1/config.json");
const HEADER: &[u8] =
    include_bytes!("../../../../tests/fixtures/qwen35-4b-avan-metadata-v1/header.json");

fn fixture() -> (Plan, Vec<u8>) {
    let layout = Layout::new(2, 64, 4, 64, MetadataDtype::BF16).unwrap();
    let common = vec![
        TensorPlan {
            name: EMBED.into(),
            shape: Shape::Matrix {
                rows: 2,
                columns: 64,
            },
            storage: Storage::Affine(layout),
        },
        TensorPlan {
            name: NORM.into(),
            shape: Shape::Vector(3),
            storage: Storage::Bf16Vector,
        },
    ];
    let convolutions = vec![ConvolutionPlan {
        layer: 0,
        history_owner: 0,
        name: CONV.into(),
        channels: 2,
        taps: 4,
    }];
    let mut physical = BTreeMap::new();
    let mut payload = Vec::new();
    // Literal low-first nibble pattern, independently expected in scalar checks.
    let codes = [0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe].repeat(8);
    let mut append = |name: &str, dtype, shape: Vec<usize>, raw: &[u8]| {
        let start = payload.len();
        payload.extend_from_slice(raw);
        physical.insert(
            name.into(),
            PhysicalTensor {
                dtype,
                shape,
                range: start..payload.len(),
            },
        );
    };
    append(EMBED, Dtype::U32, vec![2, 8], &codes);
    append(
        CONV,
        Dtype::Bf16,
        vec![2, 4, 1],
        &[
            0x80, 0x3f, 0x00, 0xc0, 0x00, 0x3f, 0x00, 0x80, 0x00, 0, 0x80, 0xbf, 0x00, 0x40, 0x80,
            0x3e,
        ],
    );
    append(
        "language_model.model.embed_tokens.scales",
        Dtype::Bf16,
        vec![2, 1],
        &[0x80, 0x3f, 0x00, 0xbf],
    );
    append(
        NORM,
        Dtype::Bf16,
        vec![3],
        &[0x80, 0x3f, 0x80, 0xbf, 0x00, 0x3f],
    );
    append(
        "language_model.model.embed_tokens.biases",
        Dtype::Bf16,
        vec![2, 1],
        &[0x00, 0xc0, 0x40, 0x40],
    );
    let plan = Plan {
        layers: recipe::build().unwrap().0,
        common,
        convolutions,
        physical,
        bytes: PayloadBytes {
            affine_codes: 64,
            affine_metadata: 8,
            bf16_vectors: 6,
            bf16_convolutions: 16,
            total: 94,
        },
    };
    assert_eq!(payload.len(), 94);
    (plan, payload)
}
fn load_fixture(
    plan: Plan,
    payload: &[u8],
    expected: &str,
    limit: usize,
) -> Result<PayloadStore, Error> {
    let mut hash = Sha256::new();
    hash.update([0xabu8; 9]);
    let identity = ContentIdentity {
        config: [1; 32],
        header: [2; 32],
        container: [0; 32],
        bytes: 103,
    };
    load_admitted(
        &mut Cursor::new(payload),
        plan,
        hash,
        identity,
        9,
        expected,
        limit,
    )
}
fn digest(payload: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update([0xabu8; 9]);
    h.update(payload);
    format!("{:x}", h.finalize())
}

#[test]
fn independent_packed_bf16_and_four_tap_values_remain_original() {
    let (p, raw) = fixture();
    let required = reservation(&p).unwrap().total;
    let store = load_fixture(p, &raw, &digest(&raw), required).unwrap();
    let matrix = store.matrix(EMBED).unwrap();
    for row in 0..2 {
        for col in 0..64 {
            let code = (col % 16) as f32;
            let want = if row == 0 {
                code - 2.0
            } else {
                3.0 - 0.5 * code
            };
            assert_eq!(matrix.value(row, col).unwrap().to_bits(), want.to_bits());
        }
    }
    assert_eq!(store.vector(NORM).unwrap(), [1.0, -1.0, 0.5]);
    let conv = store.convolution(CONV).unwrap();
    assert_eq!(conv, [1.0, -2.0, 0.5, -0.0, 0.0, -1.0, 2.0, 0.25]);
    assert_eq!(conv[3].to_bits(), (-0.0f32).to_bits());
    assert_eq!(store.matrices[EMBED].codes.as_slice(), &raw[..64]);
    assert!(std::ptr::eq(
        store.plan().embedding(),
        store.plan().tied_head()
    ));
    assert_eq!(
        (
            store.report.packed_matrices,
            store.report.fp32_vectors,
            store.report.fp32_convolutions
        ),
        (1, 1, 1)
    );
    assert_eq!(store.report.retained_tensor_capacity_bytes, 116);
    assert_eq!(store.identity.config_sha256(), &[1; 32]);
    assert_eq!(store.identity.container_bytes(), 103);
    assert!(
        store.matrix(CONV).is_err()
            && store.vector(EMBED).is_err()
            && store.convolution(NORM).is_err()
    );
}

#[test]
fn exact_reservation_rejects_minus_one_before_payload_read() {
    let (p, raw) = fixture();
    let wanted = reservation(&p).unwrap().total;
    assert!(
        matches!(load_fixture(p,&raw,&digest(&raw),wanted-1),Err(Error::Budget{required,limit}) if required==wanted && limit==wanted-1)
    );
    struct NoRead;
    impl Read for NoRead {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            panic!("budget rejection must precede read")
        }
    }
    let (p, _) = fixture();
    let identity = ContentIdentity {
        config: [0; 32],
        header: [0; 32],
        container: [0; 32],
        bytes: 103,
    };
    assert!(
        load_admitted(
            &mut NoRead,
            p,
            Sha256::new(),
            identity,
            9,
            "unused",
            wanted - 1
        )
        .is_err()
    );
    assert!(matches!(
        budget(MAX_LOAD_BYTES + 1, usize::MAX),
        Err(Error::Budget {
            limit: MAX_LOAD_BYTES,
            ..
        })
    ));
}

#[test]
fn hash_truncation_tail_and_io_failures_publish_no_store() {
    let (_, raw) = fixture();
    for cut in [0, 1, 63, 64, 80, 84, 93] {
        let (p, _) = fixture();
        assert!(load_fixture(p, &raw[..cut], &digest(&raw), usize::MAX).is_err());
    }
    let mut tail = raw.clone();
    tail.push(0);
    let (p, _) = fixture();
    assert!(load_fixture(p, &tail, &digest(&tail), usize::MAX).is_err());
    let mut changed = raw.clone();
    changed[1] ^= 1;
    let (p, _) = fixture();
    assert!(load_fixture(p, &changed, &digest(&raw), usize::MAX).is_err());
    struct Bad;
    impl Read for Bad {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("injected"))
        }
    }
    let (p, _) = fixture();
    let id = ContentIdentity {
        config: [0; 32],
        header: [0; 32],
        container: [0; 32],
        bytes: 103,
    };
    assert!(load_admitted(&mut Bad, p, Sha256::new(), id, 9, "unused", usize::MAX).is_err());
}

#[test]
fn nonfinite_metadata_vectors_convolutions_and_reconstruction_reject() {
    let (_, original) = fixture();
    for offset in [64, 80, 84, 90] {
        let mut raw = original.clone();
        raw[offset..offset + 2].copy_from_slice(&[0x80, 0x7f]);
        let (p, _) = fixture();
        assert!(load_fixture(p, &raw, &digest(&raw), usize::MAX).is_err());
    }
    // Finite BF16 scale can overflow FP32 at the largest packed code.
    let mut raw = original;
    raw[80..82].copy_from_slice(&[0x7f, 0x7f]);
    let (p, _) = fixture();
    assert!(load_fixture(p, &raw, &digest(&raw), usize::MAX).is_err());
    assert!(floats(&[0]).is_err());
}

#[test]
fn zero_scale_is_valid_and_physical_ownership_is_exact() {
    let (p, mut raw) = fixture();
    raw[80..84].fill(0);
    let s = load_fixture(p, &raw, &digest(&raw), usize::MAX).unwrap();
    assert_eq!(s.matrix(EMBED).unwrap().value(0, 63).unwrap(), -2.0);
    let (mut p, raw) = fixture();
    p.physical.get_mut(EMBED).unwrap().range.start += 1;
    assert!(load_fixture(p, &raw, &digest(&raw), usize::MAX).is_err());
    let (mut p, raw) = fixture();
    p.common[0].name = "missing.weight".into();
    assert!(load_fixture(p, &raw, &digest(&raw), usize::MAX).is_err());
    let (mut p, raw) = fixture();
    p.convolutions.clear();
    assert!(load_fixture(p, &raw, &digest(&raw), usize::MAX).is_err());
}

#[test]
fn public_metadata_policy_and_admission_fail_before_large_allocation() {
    let options = LoadOptions {
        memory_limit_bytes: usize::MAX,
        modality: ModalityPolicy::TextOnly,
    };
    for length in [0u64, HEADER_BYTES as u64 - 1, u64::MAX] {
        assert!(load(CONFIG, Cursor::new(length.to_le_bytes()), options).is_err());
    }
    let mut prefix = (HEADER_BYTES as u64).to_le_bytes().to_vec();
    prefix.extend(HEADER);
    let mut reader = Cursor::new(prefix.clone());
    let p = inspect(CONFIG, HEADER, CONTAINER_BYTES, ModalityPolicy::TextOnly).unwrap();
    let required = reservation(&p).unwrap().total;
    assert!(matches!(
        load(
            CONFIG,
            &mut reader,
            LoadOptions {
                memory_limit_bytes: required - 1,
                ..options
            }
        ),
        Err(Error::Budget { .. })
    ));
    assert_eq!(reader.position(), 119837);
    assert!(
        load(
            CONFIG,
            Cursor::new(&prefix),
            LoadOptions {
                modality: ModalityPolicy::Reject,
                ..options
            }
        )
        .is_err()
    );
    let mut c = CONFIG.to_vec();
    c[0] ^= 1;
    assert!(load(&c, Cursor::new(&prefix), options).is_err());
    prefix[20] ^= 1;
    assert!(load(CONFIG, Cursor::new(prefix), options).is_err());
    assert_eq!(reservation(&p).unwrap().tensors, 2_369_040_384);
    assert_eq!(required, 2_436_280_320);
}

#[test]
#[ignore = "explicit authenticated Qwen3.5-4B CPU storage/capture, no graph or GPU"]
fn actual_payload_probe() {
    use std::{
        fs::File,
        io::{BufWriter, Seek, Write},
        path::PathBuf,
        time::Instant,
    };
    let folder = PathBuf::from(std::env::var_os("TOKLET_QWEN35_MODEL_DIRECTORY").unwrap());
    let output = PathBuf::from(std::env::var_os("TOKLET_QWEN35_PAYLOAD_CAPTURE").unwrap());
    std::fs::create_dir(&output).unwrap();
    let config_file = File::open(folder.join("config.json")).unwrap();
    assert_eq!(config_file.metadata().unwrap().len(), CONFIG_BYTES as u64);
    let mut config = Vec::with_capacity(CONFIG_BYTES + 1);
    config_file
        .take((CONFIG_BYTES + 1) as u64)
        .read_to_end(&mut config)
        .unwrap();
    assert_eq!(config, CONFIG);
    let p = inspect(CONFIG, HEADER, CONTAINER_BYTES, ModalityPolicy::TextOnly).unwrap();
    let required = reservation(&p).unwrap().total;
    let mut file = File::open(folder.join("model.safetensors")).unwrap();
    assert!(file.metadata().unwrap().is_file());
    assert_eq!(file.metadata().unwrap().len(), CONTAINER_BYTES);
    assert!(matches!(
        load(
            &config,
            &mut file,
            LoadOptions {
                memory_limit_bytes: required - 1,
                modality: ModalityPolicy::TextOnly
            }
        ),
        Err(Error::Budget { .. })
    ));
    assert_eq!(file.stream_position().unwrap(), 119837);
    drop(file);
    let file = File::open(folder.join("model.safetensors")).unwrap();
    let start = Instant::now();
    let s = load(
        &config,
        file,
        LoadOptions {
            memory_limit_bytes: required,
            modality: ModalityPolicy::TextOnly,
        },
    )
    .unwrap();
    let load_ns = start.elapsed().as_nanos();
    assert_eq!(s.report.retained_tensor_capacity_bytes, 2_369_040_384);
    assert_eq!(s.report.admitted_load_bytes, required);
    assert_eq!(
        (
            s.report.packed_matrices,
            s.report.fp32_vectors,
            s.report.fp32_convolutions
        ),
        (249, 153, 24)
    );
    let mut rows = BufWriter::with_capacity(
        STREAM_BYTES,
        File::create(output.join("rows.f32le")).unwrap(),
    );
    let mut offset = 0;
    let mut row_count = 0;
    let mut components = 0;
    let component = |name: &str, stored: &str, bytes: usize, sha: String| {
        println!(
            "QWEN35_PAYLOAD_COMPONENT {}",
            json!({"name":name,"stored":stored,"bytes":bytes,"sha256":sha})
        );
    };
    println!();
    for (name, p) in &s.matrices {
        let base = name.strip_suffix(".weight").unwrap();
        for (n, raw) in [
            (name.clone(), &p.codes),
            (format!("{base}.scales"), &p.scales),
            (format!("{base}.biases"), &p.biases),
        ] {
            component(
                &n,
                "original",
                raw.len(),
                format!("{:x}", Sha256::digest(raw)),
            );
            components += 1;
        }
        let matrix = s.matrix(name).unwrap();
        let mut values = vec![0.; p.layout.columns()];
        for row in [0, p.layout.rows() / 2, p.layout.rows() - 1] {
            matrix.copy_row(row, &mut values).unwrap();
            let mut digest = Sha256::new();
            for &v in &values {
                let raw = v.to_le_bytes();
                rows.write_all(&raw).unwrap();
                digest.update(raw);
            }
            let bytes = values.len() * 4;
            println!(
                "QWEN35_PAYLOAD_ROW {}",
                json!({"name":name,"row":row,"columns":values.len(),"offset":offset,"bytes":bytes,"sha256":format!("{:x}",digest.finalize())})
            );
            offset += bytes;
            row_count += 1;
        }
    }
    for (name, values) in s.vectors.iter().chain(s.convolutions.iter()) {
        let mut digest = Sha256::new();
        for &v in values {
            digest.update(v.to_le_bytes());
        }
        component(
            name,
            "fp32",
            values.len() * 4,
            format!("{:x}", digest.finalize()),
        );
        components += 1;
    }
    rows.flush().unwrap();
    rows.get_ref().sync_all().unwrap();
    assert_eq!(components, 924);
    assert_eq!(row_count, 747);
    println!(
        "QWEN35_PAYLOAD_REPORT {}",
        json!({"container_sha256":CONTAINER_SHA256,"container_bytes":s.identity.container_bytes(),"config_sha256":CONFIG_SHA256,"header_sha256":HEADER_SHA256,"source_payload_bytes":s.report.source_payload_bytes,"retained_tensor_capacity_bytes":s.report.retained_tensor_capacity_bytes,"metadata_reservation_bytes":s.report.metadata_reservation_bytes,"admitted_load_bytes":s.report.admitted_load_bytes,"load_ns":load_ns,"load_and_capture_ns":start.elapsed().as_nanos(),"components":components,"rows":row_count,"rows_bytes":offset,"graph_executed":false,"runtime_qualified":false})
    );
    assert_eq!(
        s.identity.container_sha256().as_slice(),
        &hex_digest(CONTAINER_SHA256)
    );
}

fn hex_digest(text: &str) -> [u8; 32] {
    std::array::from_fn(|i| u8::from_str_radix(&text[2 * i..2 * i + 2], 16).unwrap())
}

#[test]
#[ignore = "explicit full Qwen shard corruption test; no file mutation or graph"]
fn actual_payload_corruption_is_not_published() {
    use std::{fs::File, path::PathBuf};
    struct Corrupt {
        file: File,
        position: u64,
    }
    impl Read for Corrupt {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            let n = self.file.read(out)?;
            if self.position < CONTAINER_BYTES && self.position + n as u64 > CONTAINER_BYTES - 1 {
                out[(CONTAINER_BYTES - 1 - self.position) as usize] ^= 1;
            }
            self.position += n as u64;
            Ok(n)
        }
    }
    let path = PathBuf::from(std::env::var_os("TOKLET_QWEN35_MODEL_DIRECTORY").unwrap())
        .join("model.safetensors");
    let reader = Corrupt {
        file: File::open(path).unwrap(),
        position: 0,
    };
    let error = load(
        CONFIG,
        reader,
        LoadOptions {
            memory_limit_bytes: MAX_LOAD_BYTES,
            modality: ModalityPolicy::TextOnly,
        },
    )
    .err()
    .unwrap();
    assert!(matches!(error,Error::Invalid(ref s) if s=="Qwen3.5 complete source SHA256 differs"));
    println!(
        "QWEN35_PAYLOAD_CORRUPTION {}",
        json!({"last_byte_changed_in_reader":true,"source_file_mutated":false,"store_published":false,"hash_rejected":true})
    );
}
