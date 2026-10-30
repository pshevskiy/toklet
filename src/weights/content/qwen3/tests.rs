use super::*;
use crate::weights::content::tests::Fixture;
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
    pub(crate) fn source(&self) -> Source {
        let manifest =
            Manifest::new("test/artifact", super::super::tests::REV, &self.expected).unwrap();
        QwenAudit::open_source_expected(
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
    assert!(source.qwen_content_identity().is_none());
    assert_eq!(source.shards.len(), 1);
    assert!(source.buffer.is_empty()); // Only audit's reusable64KiB buffer.
    let mut values = Vec::new();
    source
        .visit("a", |v| {
            values.push(v);
            Ok(())
        })
        .unwrap();
    assert!(source.qwen_content_identity().is_none());
    source
        .visit("b", |v| {
            values.push(v);
            Ok(())
        })
        .unwrap();
    assert_eq!(values, [1., 2., 3.]);
    let identity = source.qwen_content_identity().unwrap();
    assert_eq!(identity.config_sha256(), f.expected[0].sha256);
    assert_eq!(identity.index_sha256(), f.expected[2].sha256);
    assert_eq!(identity.container_sha256(), f.expected[1].sha256);
    assert_eq!(identity.container_bytes(), f.expected[1].bytes);
    assert_eq!(identity.container_physical_ranges(), 4);
    assert_eq!(identity.verified_container_reads(), 4);
    assert!(source.audit_owned_bytes().unwrap() < METADATA_ALLOWANCE_BYTES);
    assert!(source.visit_raw_range("a", 0..8, |_| Ok(())).is_err());
    assert!(source.qwen_content_identity().is_none());
    assert!(source.visit_raw("a", |_| Ok(())).is_err());
}

#[test]
fn source_conversion_failure_and_bad_name_poison_complete_evidence() {
    let f = SourceFixture::new();
    let mut source = f.source();
    source.visit_raw("a", |_| Ok(())).unwrap();
    source.visit_raw("b", |_| Ok(())).unwrap();
    assert!(source.qwen_content_identity().is_some());
    assert!(
        source
            .visit("a", |_| Err(invalid("synthetic decoded consumer failure")))
            .is_err()
    );
    assert!(source.qwen_content_identity().is_none());
    let mut source = f.source();
    assert!(source.visit_raw("absent", |_| Ok(())).is_err());
    assert!(source.visit_raw("a", |_| Ok(())).is_err());
    assert!(source.qwen_content_identity().is_none());
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
        matches!(QwenAudit::open_source_expected(&f.owner.path,f.config,&manifest,exact-1),Err(Error::Budget{required,..}) if required==exact)
    );
    fs::rename(
        f.owner.path.join("saved"),
        f.owner.path.join("model.safetensors"),
    )
    .unwrap();
    let mut source =
        QwenAudit::open_source_expected(&f.owner.path, f.config, &manifest, exact).unwrap();
    let mut file = File::options()
        .write(true)
        .open(f.owner.path.join("model.safetensors"))
        .unwrap();
    file.seek(SeekFrom::Start(f.data_start + 8)).unwrap();
    file.write_all(&9_f32.to_le_bytes()).unwrap();
    source.visit_raw("a", |_| Ok(())).unwrap();
    assert!(source.visit_raw("b", |_| Ok(())).is_err());
    assert!(source.qwen_content_identity().is_none());
    // A fresh prepass also fails expected full-file SHA, independently of size.
    assert!(QwenAudit::open_source_expected(&f.owner.path, f.config, &manifest, exact).is_err());
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
    let receipt = source.qwen_content_identity().unwrap();
    let manifest = Manifest::new("test/artifact", super::super::tests::REV, &f.expected).unwrap();
    assert_eq!(receipt.expected_manifest_sha256(), manifest.identity());
    let mut changed = f.expected;
    changed[5].sha256[0] ^= 1;
    let different = Manifest::new("test/artifact", super::super::tests::REV, &changed).unwrap();
    assert_ne!(manifest.identity(), different.identity());
    assert!(
        QwenAudit::open_source_expected(&f.owner.path, f.config, &different, 1024 * 1024 * 1024)
            .is_err()
    );
}
