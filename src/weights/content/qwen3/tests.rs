use super::*;
use crate::weights::content::{ExpectedFile, tests::Fixture};
use serde_json::json;
use std::fs;

pub(crate) struct SourceFixture {
    owner: Fixture,
    expected: [ExpectedFile<'static>; 6],
    config: FileDelivery,
    data_start: u64,
}
impl SourceFixture {
    pub(crate) fn new() -> Self {
        let owner = Fixture::new(vec![0]);
        let config = b"{}";
        let mut header = serde_json::to_vec(&json!({
            "a": {"dtype":"F32","shape":[2],"data_offsets":[0,8]},
            "b": {"dtype":"F32","shape":[1],"data_offsets":[8,12]}
        }))
        .unwrap();
        while !(header.len() + 8).is_multiple_of(8) {
            header.push(b' ');
        }
        let data_start = (header.len() + 8) as u64;
        let mut model = (header.len() as u64).to_le_bytes().to_vec();
        model.extend_from_slice(&header);
        for x in [1_f32, 2., 3.] {
            model.extend_from_slice(&x.to_le_bytes());
        }
        let index = serde_json::to_vec(&json!({"metadata":{"total_size":12},"weight_map":{"a":"model.safetensors","b":"model.safetensors"}})).unwrap();
        let data: [(&str, &[u8]); 6] = [
            ("config.json", config),
            ("model.safetensors", &model),
            ("model.safetensors.index.json", &index),
            ("tokenizer.json", b"{}"),
            ("tokenizer_config.json", b"{}"),
            ("special_tokens_map.json", b"{}"),
        ];
        let expected = data.map(|(name, bytes)| {
            fs::write(owner.path.join(name), bytes).unwrap();
            ExpectedFile {
                name,
                bytes: bytes.len() as u64,
                sha256: sha2::Sha256::digest(bytes).into(),
            }
        });
        let manifest = Manifest::new("test/artifact", super::super::tests::REV, &expected).unwrap();
        let (_, receipt) = verified_small(&owner.path, &manifest, 0).unwrap();
        Self {
            owner,
            expected,
            config: receipt,
            data_start,
        }
    }
    pub(crate) fn config_receipt(&self) -> FileDelivery {
        self.config
    }
    pub(crate) fn source(&self) -> Source {
        let manifest =
            Manifest::new("test/artifact", super::super::tests::REV, &self.expected).unwrap();
        DenseAudit::open_source_expected(
            &self.owner.path,
            self.config,
            &manifest,
            1024 * 1024 * 1024,
        )
        .unwrap()
    }
}

#[test]
fn actual_source_consumer_requires_all_tensor_ranges_and_poison_on_public_raw_range() {
    let f = SourceFixture::new();
    let mut source = f.source();
    assert!(source.dense_content_identity().is_none());
    assert_eq!(source.shards.len(), 1);
    assert!(source.buffer.is_empty()); // Only audit's reusable64KiB buffer.
    let mut values = Vec::new();
    source
        .visit("a", |v| {
            values.push(v);
            Ok(())
        })
        .unwrap();
    assert!(source.dense_content_identity().is_none());
    source
        .visit("b", |v| {
            values.push(v);
            Ok(())
        })
        .unwrap();
    assert_eq!(values, [1., 2., 3.]);
    let identity = source.dense_content_identity().unwrap();
    assert_eq!(identity.config_sha256(), f.expected[0].sha256);
    assert_eq!(identity.index_sha256(), f.expected[2].sha256);
    assert_eq!(identity.container_sha256(), f.expected[1].sha256);
    assert_eq!(identity.container_bytes(), f.expected[1].bytes);
    assert_eq!(identity.container_physical_ranges(), 4);
    assert_eq!(identity.verified_container_reads(), 4);
    assert!(source.audit_owned_bytes().unwrap() < METADATA_ALLOWANCE_BYTES);
    assert!(source.visit_raw_range("a", 0..8, |_| Ok(())).is_err());
    assert!(source.dense_content_identity().is_none());
    assert!(source.visit_raw("a", |_| Ok(())).is_err());
}

#[test]
fn source_conversion_failure_and_bad_name_poison_complete_evidence() {
    let f = SourceFixture::new();
    let mut source = f.source();
    source.visit_raw("a", |_| Ok(())).unwrap();
    source.visit_raw("b", |_| Ok(())).unwrap();
    assert!(source.dense_content_identity().is_some());
    assert!(
        source
            .visit("a", |_| Err(invalid("synthetic decoded consumer failure")))
            .is_err()
    );
    assert!(source.dense_content_identity().is_none());
    let mut source = f.source();
    assert!(source.visit_raw("absent", |_| Ok(())).is_err());
    assert!(source.visit_raw("a", |_| Ok(())).is_err());
    assert!(source.dense_content_identity().is_none());
}

#[test]
fn source_mutation_and_index_budget_fail_before_receipt_publication() {
    use std::io::{Seek, SeekFrom, Write};
    let f = SourceFixture::new();
    let manifest = Manifest::new("test/artifact", super::super::tests::REV, &f.expected).unwrap();
    let exact = 12 + METADATA_ALLOWANCE_BYTES + super::super::BUFFER_BYTES;
    // Remove container: authenticated index admission must reject before it opens.
    fs::rename(
        f.owner.path.join("model.safetensors"),
        f.owner.path.join("saved"),
    )
    .unwrap();
    assert!(
        matches!(DenseAudit::open_source_expected(&f.owner.path,f.config,&manifest,exact-1),Err(Error::Budget{required,..}) if required==exact)
    );
    fs::rename(
        f.owner.path.join("saved"),
        f.owner.path.join("model.safetensors"),
    )
    .unwrap();
    let mut source =
        DenseAudit::open_source_expected(&f.owner.path, f.config, &manifest, exact).unwrap();
    let mut file = File::options()
        .write(true)
        .open(f.owner.path.join("model.safetensors"))
        .unwrap();
    file.seek(SeekFrom::Start(f.data_start + 8)).unwrap();
    file.write_all(&9_f32.to_le_bytes()).unwrap();
    source.visit_raw("a", |_| Ok(())).unwrap();
    assert!(source.visit_raw("b", |_| Ok(())).is_err());
    assert!(source.dense_content_identity().is_none());
    // A fresh prepass also fails expected full-file SHA, independently of size.
    assert!(DenseAudit::open_source_expected(&f.owner.path, f.config, &manifest, exact).is_err());
}

#[test]
fn full_manifest_anchor_does_not_claim_codec_or_ancillary_consumption() {
    let f = SourceFixture::new();
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ] {
        fs::remove_file(f.owner.path.join(name)).unwrap();
    }
    let mut source = f.source();
    source.visit_raw("a", |_| Ok(())).unwrap();
    source.visit_raw("b", |_| Ok(())).unwrap();
    let receipt = source.dense_content_identity().unwrap();
    let manifest = Manifest::new("test/artifact", super::super::tests::REV, &f.expected).unwrap();
    assert_eq!(receipt.expected_manifest_sha256(), manifest.identity());
    let mut changed = f.expected;
    changed[5].sha256[0] ^= 1;
    let different = Manifest::new("test/artifact", super::super::tests::REV, &changed).unwrap();
    assert_ne!(manifest.identity(), different.identity());
    assert!(
        DenseAudit::open_source_expected(&f.owner.path, f.config, &different, 1024 * 1024 * 1024)
            .is_err()
    );
}

#[test]
fn engine_qwen_manifests_keep_canonical_anchors_and_exact_catalog_bounds() {
    let qwen2 = QwenProfile::Qwen2.manifest().unwrap();
    let qwen3 = QwenProfile::Qwen3.manifest().unwrap();
    assert_eq!(
        qwen2
            .identity()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>(),
        "e92e6d359697d8ada4714ff4f5ff618ba8d9fc69d7abc24bfb4ec905eae6c59b"
    );
    assert_eq!(
        qwen3
            .identity()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>(),
        "5e9aafa305e0e48182a856cafe3ae2b597599033e61b16664f42b09a4af5ff51"
    );
    assert_ne!(qwen2.identity(), qwen3.identity());
    assert_eq!(qwen2.files()[1].bytes, 525045902);
    assert_eq!(qwen2.files()[4].bytes, 7308);
    assert_eq!(qwen2.files().len(), 6);
}

#[test]
fn selected_profile_rejects_cross_manifest_config_before_any_index_open() {
    let missing = Path::new("/toklet-absent-audited-profile");
    for (from, to) in [
        (QwenProfile::Qwen2, QwenProfile::Qwen3),
        (QwenProfile::Qwen3, QwenProfile::Qwen2),
    ] {
        let m = from.manifest().unwrap();
        let receipt = FileDelivery {
            manifest: m.identity(),
            file: m.files()[0].sha256,
            name: sha2::Sha256::digest(b"config.json").into(),
            bytes: m.files()[0].bytes,
            ranges: 1,
            reads: 1,
        };
        assert!(
            matches!(DenseAudit::open_source(DenseProfile::Qwen(to),missing,receipt,1024*1024*1024),
            Err(Error::Invalid(message)) if message == "Dense source config receipt mismatch")
        );
        // Matching manifest still cannot substitute the wrong config digest/name.
        for bad in [
            FileDelivery {
                file: [0; 32],
                ..receipt
            },
            FileDelivery {
                name: [0; 32],
                ..receipt
            },
        ] {
            assert!(
                matches!(DenseAudit::open_source(DenseProfile::Qwen(from),missing,bad,1024*1024*1024),
                Err(Error::Invalid(message)) if message == "Dense source config receipt mismatch")
            );
        }
    }
}

#[test]
fn shared_source_rejects_authenticated_malformed_header_and_index_plans() {
    for case in 0..7 {
        let mut f = SourceFixture::new();
        let (name, bytes) = if case < 3 {
            let index = match case {
                0 => {
                    json!({"metadata":{"total_size":12},"weight_map":{"a":"other.safetensors","b":"model.safetensors"}})
                }
                1 => {
                    json!({"metadata":{"total_size":12},"weight_map":{"a":"model.safetensors","absent":"model.safetensors"}})
                }
                _ => {
                    json!({"metadata":{"total_size":13},"weight_map":{"a":"model.safetensors","b":"model.safetensors"}})
                }
            };
            (
                "model.safetensors.index.json",
                serde_json::to_vec(&index).unwrap(),
            )
        } else {
            let header = match case {
                3 => r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"a":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}}"#.as_bytes().to_vec(),
                4 => serde_json::to_vec(&json!({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}})).unwrap(),
                5 => serde_json::to_vec(&json!({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}})).unwrap(),
                _ => serde_json::to_vec(&json!({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[0],"data_offsets":[8,8]}})).unwrap(),
            };
            let mut model = (header.len() as u64).to_le_bytes().to_vec();
            model.extend(header);
            model.extend([0; 12]);
            ("model.safetensors", model)
        };
        fs::write(f.owner.path.join(name), &bytes).unwrap();
        let entry = f.expected.iter_mut().find(|x| x.name == name).unwrap();
        entry.bytes = bytes.len() as u64;
        entry.sha256 = sha2::Sha256::digest(bytes).into();
        let manifest =
            Manifest::new("test/artifact", super::super::tests::REV, &f.expected).unwrap();
        // Rebind synthetic config to this intentionally malformed expected artifact.
        let receipt = FileDelivery {
            manifest: manifest.identity(),
            ..f.config
        };
        assert!(
            DenseAudit::open_source_expected(&f.owner.path, receipt, &manifest, 1024 * 1024 * 1024)
                .is_err(),
            "case{case}"
        );
    }
}
