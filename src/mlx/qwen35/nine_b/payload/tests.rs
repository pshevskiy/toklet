use super::*;
use crate::mlx::{
    affine::{Layout, MetadataDtype},
    qwen35::{ConvolutionPlan, Dtype, PayloadBytes, PhysicalTensor, Shape, Storage, TensorPlan},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, io::Cursor};

const EMBED: &str = "language_model.model.embed_tokens.weight";
const HEAD: &str = "language_model.lm_head.weight";
const NORM: &str = "language_model.model.norm.weight";
const CONV: &str = "language_model.model.layers.0.linear_attn.conv1d.weight";
const STREAM: &[u8] =
    include_bytes!("../../../../../tests/fixtures/qwen35-9b-storage-v1/stream.bin");
const CONFIG: &[u8] =
    include_bytes!("../../../../../tests/fixtures/qwen35-9b-lukey-metadata-v1/config.json");
const HEADER: &[u8] =
    include_bytes!("../../../../../tests/fixtures/qwen35-9b-lukey-metadata-v1/header.json");
fn inventory() -> Value {
    serde_json::from_slice(include_bytes!(
        "../../../../../tests/fixtures/qwen35-9b-storage-v1/inventory.json"
    ))
    .unwrap()
}
fn fixture() -> Plan {
    let matrix = |name: &str, rows| TensorPlan {
        name: name.into(),
        shape: Shape::Matrix { rows, columns: 64 },
        storage: Storage::Affine(Layout::new(rows, 64, 4, 64, MetadataDtype::BF16).unwrap()),
    };
    let mut physical = BTreeMap::new();
    for (name, row) in inventory()["physical"].as_object().unwrap() {
        physical.insert(
            name.clone(),
            PhysicalTensor {
                dtype: match row["dtype"].as_str().unwrap() {
                    "U32" => Dtype::U32,
                    "BF16" => Dtype::Bf16,
                    _ => panic!("fixture dtype"),
                },
                shape: row["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|n| n.as_u64().unwrap() as usize)
                    .collect(),
                range: row["data_offsets"][0].as_u64().unwrap() as usize
                    ..row["data_offsets"][1].as_u64().unwrap() as usize,
            },
        );
    }
    Plan {
        layers: crate::mlx::qwen35::recipe::build().unwrap().0,
        common: vec![
            matrix(EMBED, 2),
            matrix(HEAD, 3),
            TensorPlan {
                name: NORM.into(),
                shape: Shape::Vector(3),
                storage: Storage::Bf16Vector,
            },
        ],
        convolutions: vec![ConvolutionPlan {
            layer: 0,
            history_owner: 0,
            name: CONV.into(),
            channels: 2,
            taps: 4,
        }],
        physical,
        bytes: PayloadBytes {
            affine_codes: 160,
            affine_metadata: 20,
            bf16_vectors: 6,
            bf16_convolutions: 16,
            total: 202,
        },
    }
}
fn digest(payload: &[u8]) -> String {
    let mut sha = Sha256::new();
    sha.update(&STREAM[..9]);
    sha.update(payload);
    format!("{:x}", sha.finalize())
}
fn private_load<R: Read>(
    plan: Plan,
    reader: &mut R,
    expected: &str,
    limit: usize,
) -> Result<PayloadStore, Error> {
    let mut sha = Sha256::new();
    sha.update(&STREAM[..9]);
    let components = cold::read_components(
        reader,
        source_recipe(&plan),
        sha,
        ContentIdentity::pending(b"fixture-config", b"fixture-header", 211),
        9,
        expected,
        limit,
    )?;
    Ok(PayloadStore { plan, components })
}
fn tiny() -> PayloadStore {
    private_load(
        fixture(),
        &mut Cursor::new(&STREAM[9..]),
        inventory()["source_sha256"].as_str().unwrap(),
        usize::MAX,
    )
    .unwrap()
}

#[test]
fn independent_rows_original_components_and_distinct_head_owners() {
    let fixture = inventory();
    assert_eq!(STREAM.len(), 211);
    assert_eq!(
        format!("{:x}", Sha256::digest(STREAM)),
        fixture["source_sha256"]
    );
    let store = tiny();
    assert!(!std::ptr::eq(
        store.plan().embedding(),
        store.plan().output_head()
    ));
    assert_ne!(
        store.components.matrices[EMBED].codes.as_ptr(),
        store.components.matrices[HEAD].codes.as_ptr()
    );
    for (name, rows) in fixture["expected_rows"].as_object().unwrap() {
        let matrix = store.matrix(name).unwrap();
        for (row, expected) in rows.as_array().unwrap().iter().enumerate() {
            let mut actual = vec![0.0; 64];
            matrix.copy_row(row, &mut actual).unwrap();
            for (col, value) in actual.into_iter().enumerate() {
                assert_eq!(
                    value.to_bits(),
                    (expected[col].as_f64().unwrap() as f32).to_bits(),
                    "{name}/{row}/{col}"
                );
            }
        }
    }
    for (name, p) in &store.components.matrices {
        let base = name.strip_suffix(".weight").unwrap();
        for (key, raw) in [
            (name.clone(), &p.codes),
            (format!("{base}.scales"), &p.scales),
            (format!("{base}.biases"), &p.biases),
        ] {
            assert_eq!(
                format!("{:x}", Sha256::digest(raw)),
                fixture["physical"][&key]["sha256"]
            );
        }
    }
    assert!(
        store.matrix(CONV).is_err()
            && store.vector(EMBED).is_err()
            && store.convolution(NORM).is_err()
    );
}

#[test]
fn exact_small_bits_capacity_and_historical_identity() {
    let fixture = inventory();
    let store = tiny();
    for (name, bits) in fixture["expected_small_fp32_bits"].as_object().unwrap() {
        let values = if name == CONV {
            store.convolution(name).unwrap()
        } else {
            store.vector(name).unwrap()
        };
        assert_eq!(values.len(), bits.as_array().unwrap().len());
        for (&value, expected) in values.iter().zip(bits.as_array().unwrap()) {
            assert_eq!(value.to_bits(), expected.as_u64().unwrap() as u32);
        }
    }
    let report = store.report();
    assert_eq!(
        (
            report.packed_matrices,
            report.fp32_vectors,
            report.fp32_convolutions
        ),
        (2, 1, 1)
    );
    assert_eq!(report.retained_tensor_capacity_bytes, 224);
    assert_eq!(report.source_payload_bytes, 202);
    assert_eq!(
        report.admitted_load_bytes,
        fixture["requested_load_bytes"].as_u64().unwrap() as usize
    );
    assert_eq!(report.metadata_reservation_bytes, 64 * 1024 * 1024);
    assert_eq!(store.identity().container_bytes(), 211);
    let expected: [u8; 32] = Sha256::digest(b"fixture-config").into();
    assert_eq!(store.identity().config_sha256(), &expected);
    let expected: [u8; 32] = Sha256::digest(STREAM).into();
    assert_eq!(store.identity().container_sha256(), &expected);
}

struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
        panic!("admission must precede payload read")
    }
}
#[test]
fn exact_minus_one_and_overflow_fail_before_read() {
    let plan = fixture();
    let required = cold::reservation_for(&source_recipe(&plan)).unwrap().total;
    assert!(
        matches!(private_load(plan,&mut NoRead,"unused",required-1),Err(Error::Budget{required:r,limit}) if r==required && limit==required-1)
    );
    let mut p = fixture();
    p.convolutions[0].channels = usize::MAX;
    assert!(private_load(p, &mut NoRead, "unused", usize::MAX).is_err());
    let mut p = fixture();
    p.common[2].shape = Shape::Vector(usize::MAX);
    assert!(private_load(p, &mut NoRead, "unused", usize::MAX).is_err());
    let mut p = fixture();
    p.physical.clear();
    p.physical.insert(
        EMBED.into(),
        PhysicalTensor {
            dtype: Dtype::U32,
            shape: vec![1],
            range: 0..usize::MAX,
        },
    );
    assert!(private_load(p, &mut NoRead, "unused", usize::MAX).is_err());
}

#[test]
fn truncation_tail_hash_and_io_errors_never_return_storage() {
    for cut in [0, 1, 31, 63, 64, 79, 80, 175, 195, 201] {
        assert!(
            private_load(
                fixture(),
                &mut Cursor::new(&STREAM[9..9 + cut]),
                inventory()["source_sha256"].as_str().unwrap(),
                usize::MAX
            )
            .is_err()
        );
    }
    let mut trailing = STREAM[9..].to_vec();
    trailing.push(0);
    assert!(
        private_load(
            fixture(),
            &mut Cursor::new(trailing),
            inventory()["source_sha256"].as_str().unwrap(),
            usize::MAX
        )
        .is_err()
    );
    assert!(
        private_load(
            fixture(),
            &mut Cursor::new(&STREAM[9..]),
            &"00".repeat(32),
            usize::MAX
        )
        .is_err()
    );
    struct Failed;
    impl Read for Failed {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("fixture IO failure"))
        }
    }
    assert!(private_load(fixture(), &mut Failed, "unused", usize::MAX).is_err());
}

#[test]
fn authenticated_nonfinite_or_overflowing_original_components_reject() {
    let f = inventory();
    for name in [
        "language_model.lm_head.scales",
        "language_model.lm_head.biases",
        NORM,
        CONV,
    ] {
        for bits in [0x7f80u16, 0x7fc0, 0xff80] {
            let mut payload = STREAM[9..].to_vec();
            let offset = f["physical"][name]["data_offsets"][0].as_u64().unwrap() as usize;
            payload[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
            assert!(
                private_load(
                    fixture(),
                    &mut Cursor::new(&payload),
                    &digest(&payload),
                    usize::MAX
                )
                .is_err(),
                "{name}/{bits:x}"
            );
        }
    }
    let mut payload = STREAM[9..].to_vec();
    let offset = f["physical"]["language_model.lm_head.scales"]["data_offsets"][0]
        .as_u64()
        .unwrap() as usize;
    payload[offset..offset + 2].copy_from_slice(&0x7f7fu16.to_le_bytes());
    assert!(
        private_load(
            fixture(),
            &mut Cursor::new(&payload),
            &digest(&payload),
            usize::MAX
        )
        .is_err()
    );
}

#[test]
fn short_and_interrupted_reads_preserve_original_components() {
    struct Chunked<'a> {
        payload: &'a [u8],
        interrupted: bool,
    }
    impl Read for Chunked<'_> {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            if !self.interrupted {
                self.interrupted = true;
                return Err(std::io::ErrorKind::Interrupted.into());
            }
            let n = out.len().min(3).min(self.payload.len());
            out[..n].copy_from_slice(&self.payload[..n]);
            self.payload = &self.payload[n..];
            Ok(n)
        }
    }
    let store = private_load(
        fixture(),
        &mut Chunked {
            payload: &STREAM[9..],
            interrupted: false,
        },
        inventory()["source_sha256"].as_str().unwrap(),
        usize::MAX,
    )
    .unwrap();
    assert_eq!(store.matrix(HEAD).unwrap().value(0, 0).unwrap(), 4.75);
    assert_eq!(store.matrix(EMBED).unwrap().value(0, 0).unwrap(), -2.0);
}

#[test]
fn changed_physical_gaps_aliases_and_missing_owners_reject() {
    let mut p = fixture();
    p.physical.get_mut(CONV).unwrap().range.start += 1;
    assert!(
        private_load(
            p,
            &mut Cursor::new(&STREAM[9..]),
            inventory()["source_sha256"].as_str().unwrap(),
            usize::MAX
        )
        .is_err()
    );
    let mut p = fixture();
    let range = p.physical[EMBED].range.clone();
    p.physical.get_mut(HEAD).unwrap().range = range;
    assert!(
        private_load(
            p,
            &mut Cursor::new(&STREAM[9..]),
            inventory()["source_sha256"].as_str().unwrap(),
            usize::MAX
        )
        .is_err()
    );
    let mut p = fixture();
    p.common[1].name = "missing.weight".into();
    assert!(
        private_load(
            p,
            &mut Cursor::new(&STREAM[9..]),
            inventory()["source_sha256"].as_str().unwrap(),
            usize::MAX
        )
        .is_err()
    );
}

#[test]
fn actual_recipe_budget_rejects_before_tensor_payload() {
    let plan = inspect(CONFIG, HEADER, CONTAINER_BYTES, ModalityPolicy::TextOnly).unwrap();
    let required = cold::reservation_for(&source_recipe(&plan)).unwrap().total;
    assert_eq!(required, 5_107_402_752);
    let mut header = (HEADER_BYTES as u64).to_le_bytes().to_vec();
    header.extend_from_slice(HEADER);
    let mut reader = Cursor::new(header);
    assert!(
        matches!(load(CONFIG,&mut reader,LoadOptions{memory_limit_bytes:required-1,modality:ModalityPolicy::TextOnly}),Err(Error::Budget{required:r,limit}) if r==required && limit==required-1)
    );
    assert_eq!(reader.position(), 120889);
    for options in [
        LoadOptions {
            memory_limit_bytes: 0,
            modality: ModalityPolicy::TextOnly,
        },
        LoadOptions {
            memory_limit_bytes: usize::MAX,
            modality: ModalityPolicy::Reject,
        },
    ] {
        assert!(load(CONFIG, &mut NoRead, options).is_err());
    }
}

#[test]
fn public_source_pins_and_cross_source_eligibility_stay_separate() {
    let options = LoadOptions {
        memory_limit_bytes: usize::MAX,
        modality: ModalityPolicy::TextOnly,
    };
    assert!(load(b"fixture-config", &mut NoRead, options).is_err());
    assert!(load(CONFIG, Cursor::new(STREAM), options).is_err());
    let four =
        include_bytes!("../../../../../tests/fixtures/qwen35-4b-avan-metadata-v1/config.json");
    assert!(load(four, &mut NoRead, options).is_err());
    assert!(cold::load(CONFIG, &mut NoRead, options).is_err());
    let mut header = (HEADER_BYTES as u64).to_le_bytes().to_vec();
    header.extend_from_slice(HEADER);
    let mut config = CONFIG.to_vec();
    config[0] ^= 1;
    let mut reader = Cursor::new(header);
    assert!(load(&config, &mut reader, options).is_err());
    assert_eq!(reader.position(), 120889);
}

use crate::mlx::qwen35::nine_b::{CONFIG_SHA256, HEADER_SHA256};

#[test]
#[ignore = "explicit pinned9B CPU storage source/capture; no numerical graph or GPU"]
fn actual_payload_probe() {
    use std::{
        fs::File,
        io::{BufWriter, Seek, Write},
        path::PathBuf,
        time::Instant,
    };
    let folder = PathBuf::from(std::env::var_os("TOKLET_QWEN35_9B_MODEL_DIRECTORY").unwrap());
    let output = PathBuf::from(std::env::var_os("TOKLET_QWEN35_9B_PAYLOAD_CAPTURE").unwrap());
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
    let required = cold::reservation_for(&source_recipe(&p)).unwrap().total;
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
    assert_eq!(file.stream_position().unwrap(), 120889);
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
    assert_eq!(s.report().retained_tensor_capacity_bytes, 5_040_162_816);
    assert_eq!(s.report().admitted_load_bytes, required);
    assert_eq!(
        (
            s.report().packed_matrices,
            s.report().fp32_vectors,
            s.report().fp32_convolutions
        ),
        (250, 153, 24)
    );
    let mut rows =
        BufWriter::with_capacity(65536, File::create(output.join("rows.f32le")).unwrap());
    let mut offset = 0;
    let mut row_count = 0;
    let mut components = 0;
    let component = |name: &str, stored: &str, bytes: usize, sha: String| {
        println!(
            "QWEN35_9B_PAYLOAD_COMPONENT {}",
            json!({"name":name,"stored":stored,"bytes":bytes,"sha256":sha})
        );
    };
    println!();
    for (name, p) in &s.components.matrices {
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
                "QWEN35_9B_PAYLOAD_ROW {}",
                json!({"name":name,"row":row,"columns":values.len(),"offset":offset,"bytes":bytes,"sha256":format!("{:x}",digest.finalize())})
            );
            offset += bytes;
            row_count += 1;
        }
    }
    for (name, values) in s
        .components
        .vectors
        .iter()
        .chain(s.components.convolutions.iter())
    {
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
    assert_eq!(components, 927);
    assert_eq!(row_count, 750);
    println!(
        "QWEN35_9B_PAYLOAD_REPORT {}",
        json!({"container_sha256":CONTAINER_SHA256,"container_bytes":s.identity().container_bytes(),"config_sha256":CONFIG_SHA256,"header_sha256":HEADER_SHA256,"source_payload_bytes":s.report().source_payload_bytes,"retained_tensor_capacity_bytes":s.report().retained_tensor_capacity_bytes,"metadata_reservation_bytes":s.report().metadata_reservation_bytes,"admitted_load_bytes":s.report().admitted_load_bytes,"load_ns":load_ns,"load_and_capture_ns":start.elapsed().as_nanos(),"components":components,"rows":row_count,"rows_bytes":offset,"graph_executed":false,"runtime_qualified":false})
    );
    assert_eq!(
        s.identity().container_sha256().as_slice(),
        &hex_digest(CONTAINER_SHA256)
    );
}

fn hex_digest(text: &str) -> [u8; 32] {
    std::array::from_fn(|i| u8::from_str_radix(&text[2 * i..2 * i + 2], 16).unwrap())
}

#[test]
#[ignore = "explicit pinned9B corruption in reader; source file unchanged, no numerical graph"]
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
    let path = PathBuf::from(std::env::var_os("TOKLET_QWEN35_9B_MODEL_DIRECTORY").unwrap())
        .join("model.safetensors");
    let reader = Corrupt {
        file: File::open(path).unwrap(),
        position: 0,
    };
    let error = load(
        CONFIG,
        reader,
        LoadOptions {
            memory_limit_bytes: 6 * 1024 * 1024 * 1024,
            modality: ModalityPolicy::TextOnly,
        },
    )
    .err()
    .unwrap();
    assert!(matches!(error,Error::Invalid(ref s) if s=="Qwen3.5 complete source SHA256 differs"));
    println!(
        "QWEN35_9B_PAYLOAD_CORRUPTION {}",
        json!({"last_byte_changed_in_reader":true,"source_file_mutated":false,"store_published":false,"hash_rejected":true})
    );
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal;
