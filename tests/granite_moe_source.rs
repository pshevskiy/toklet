use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Seek, SeekFrom, Write},
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use toklet::{
    mlx::granite_moe::{ExpectedArtifact, Plan, Role, Source},
    weights::Error,
};
const LIMIT: usize = 6 * 1024 * 1024 * 1024;
fn sha(b: &[u8]) -> [u8; 32] {
    Sha256::digest(b).into()
}
fn fixture() -> Value {
    let raw = include_bytes!("fixtures/granite-moe-source-v1/manifest.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        "6408f2b6a6131105bd507b6376aa85ff85643a7f3b8531ab3660e410bb1d58c2"
    );
    let m: Value = serde_json::from_slice(raw).unwrap();
    assert_eq!(
        m["producer_sha256"],
        format!(
            "{:x}",
            Sha256::digest(include_bytes!(
                "../tools/reference/granite_moe_source_fixture.py"
            ))
        )
    );
    m
}
fn header(bytes: &[u8]) -> &[u8] {
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    &bytes[8..8 + n]
}
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "toklet-moe-source-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        fs::write(
            p.join("config.json"),
            include_bytes!("fixtures/granite-moe-source-v1/config.json"),
        )
        .unwrap();
        fs::write(
            p.join("model.safetensors"),
            include_bytes!("fixtures/granite-moe-source-v1/model.safetensors"),
        )
        .unwrap();
        Self(p)
    }
    fn expected(&self) -> ExpectedArtifact<'static> {
        let c = fs::read(self.0.join("config.json")).unwrap();
        let m = fs::read(self.0.join("model.safetensors")).unwrap();
        ExpectedArtifact {
            repository: "test/granite-moe",
            revision: "0000000000000000000000000000000000000000",
            config_bytes: c.len() as u64,
            config_sha256: sha(&c),
            model_bytes: m.len() as u64,
            model_sha256: sha(&m),
        }
    }
    fn open(&self) -> Source {
        Source::open(&self.0, self.expected(), LIMIT).unwrap()
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
#[test]
fn authenticated_import_matches_independent_half_bits_and_exact_slices() {
    let m = fixture();
    let t = Temp::new();
    let mut source = t.open();
    let plan = source.plan();
    assert_eq!(plan.tensors().len(), 11);
    let storage = plan.storage();
    assert_eq!(
        storage.half_weights_bytes,
        plan.tensors()
            .iter()
            .map(|t| t.storage_bytes())
            .sum::<usize>()
    );
    let mut destinations = Vec::new();
    for tensor in plan.tensors() {
        assert_eq!(tensor.alignment(), 2);
        if tensor.role() == Role::GateUp {
            assert_eq!(tensor.shape(), [3, 6, 2]);
            assert_eq!(tensor.gate_up_slices(2).unwrap(), (24..30, 30..36));
            assert!(tensor.gate_up_slices(3).is_err());
            assert!(tensor.down_slice(0).is_err());
        } else if tensor.role() == Role::ExpertDown {
            assert_eq!(tensor.down_slice(2).unwrap(), 12..18);
            assert!(tensor.down_slice(3).is_err());
        }
        destinations.push(vec![0x7777; tensor.elements()]);
    }
    for (i, out) in destinations.iter_mut().enumerate() {
        let expected = &m["tensors"][i];
        assert_eq!(
            source.plan().tensors()[i].name(),
            expected["name"].as_str().unwrap()
        );
        source.import_next(out).unwrap();
        let bits: Vec<_> = expected["half_bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u16)
            .collect();
        assert_eq!(*out, bits);
    }
    let receipt = source.finish().unwrap();
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(include_bytes!(
                "fixtures/granite-moe-source-v1/model.safetensors"
            ))
        ),
        m["model_sha256"]
    );
    assert_eq!(receipt.source_model_sha256(), t.expected().model_sha256);
    assert_eq!(
        receipt
            .half_payload_sha256()
            .iter()
            .fold(String::new(), |mut s, x| {
                use std::fmt::Write;
                write!(&mut s, "{x:02x}").unwrap();
                s
            }),
        m["half_sha256"]
    );
    assert_eq!(receipt.converted_values() * 2, storage.half_weights_bytes);
    assert_eq!(
        receipt.underflowed_values(),
        m["underflows"].as_u64().unwrap() as usize
    );
}
#[test]
fn incomplete_capacity_failure_and_bad_payload_never_publish() {
    let t = Temp::new();
    assert!(t.open().finish().is_err());
    let mut s = t.open();
    let mut short = vec![0; s.plan().tensors()[0].elements() - 1];
    assert!(s.import_next(&mut short).is_err());
    assert!(s.finish().is_err());
    for bad in [0x7f80u16, 0x7fc0, 0x4780] {
        let t = Temp::new();
        let path = t.0.join("model.safetensors");
        let mut raw = fs::read(&path).unwrap();
        let last = raw.len() - 2;
        raw[last..].copy_from_slice(&bad.to_le_bytes());
        fs::write(&path, &raw).unwrap();
        let mut s = t.open();
        let mut failed = false;
        for i in 0..s.plan().tensors().len() {
            let mut out = vec![0; s.plan().tensors()[i].elements()];
            if s.import_next(&mut out).is_err() {
                failed = true;
                break;
            }
        }
        assert!(failed);
        assert!(s.finish().is_err());
    }
}
#[test]
fn retained_descriptor_mutation_truncation_and_digest_guards() {
    let t = Temp::new();
    let expected = t.expected();
    fs::write(t.0.join("config.json"), b"{}").unwrap();
    assert!(Source::open(&t.0, expected, LIMIT).is_err());
    for truncate in [false, true] {
        let t = Temp::new();
        let mut s = t.open();
        let path = t.0.join("model.safetensors");
        let mut f = fs::OpenOptions::new().write(true).open(path).unwrap();
        let last = s.plan().tensors().len() - 1;
        for i in 0..last {
            let mut out = vec![0; s.plan().tensors()[i].elements()];
            s.import_next(&mut out).unwrap();
        }
        if truncate {
            f.set_len(8).unwrap();
        } else {
            f.seek(SeekFrom::End(-2)).unwrap();
            f.write_all(&0x4000u16.to_le_bytes()).unwrap();
            f.flush().unwrap();
        }
        let mut out = vec![0; s.plan().tensors()[last].elements()];
        assert!(s.import_next(&mut out).is_err());
        assert!(s.finish().is_err());
    }
    let t = Temp::new();
    let mut s = t.open();
    fs::rename(t.0.join("model.safetensors"), t.0.join("old.safetensors")).unwrap();
    fs::write(t.0.join("model.safetensors"), b"replaced").unwrap();
    for i in 0..s.plan().tensors().len() {
        let mut out = vec![0; s.plan().tensors()[i].elements()];
        s.import_next(&mut out).unwrap();
    }
    assert!(s.finish().is_ok());
}
fn inspect(c: &[u8], m: &[u8], budget: usize) -> Result<Plan, Error> {
    Plan::inspect(c, header(m), m.len() as u64, budget)
}
#[test]
fn strict_semantics_inventory_and_budget_minus_one() {
    let c = include_bytes!("fixtures/granite-moe-source-v1/config.json");
    let m = include_bytes!("fixtures/granite-moe-source-v1/model.safetensors");
    let p = inspect(c, m, LIMIT).unwrap();
    let required = p.storage().admitted_load_bytes;
    assert!(matches!(
        inspect(c, m, required - 1),
        Err(Error::Budget { .. })
    ));
    assert!(inspect(c, m, required).is_ok());
    let t = Temp::new();
    assert!(matches!(
        Source::open(&t.0, t.expected(), required - 1),
        Err(Error::Budget { .. })
    ));
    for (key, value) in [
        ("model_type", json!("granite")),
        ("tie_word_embeddings", json!(false)),
        ("attention_bias", json!(true)),
        ("torch_dtype", json!("float16")),
        ("num_local_experts", json!(33)),
        ("num_experts_per_tok", json!(4)),
        ("embedding_multiplier", json!(0)),
        ("residual_multiplier", json!(1e-20)),
        ("rope_scaling", json!({"type":"linear"})),
        ("quantization", json!({"bits":4})),
        ("shared_expert_intermediate_size", json!(3)),
    ] {
        let mut v: Value = serde_json::from_slice(c).unwrap();
        v[key] = value;
        assert!(
            inspect(&serde_json::to_vec(&v).unwrap(), m, LIMIT).is_err(),
            "{key}"
        );
    }
    for change in [0, 1, 2, 3] {
        let mut h: Value = serde_json::from_slice(header(m)).unwrap();
        let name = "model.layers.0.block_sparse_moe.input_linear.weight";
        match change {
            0 => h[name]["dtype"] = json!("F16"),
            1 => h[name]["shape"] = json!([3, 2, 6]),
            2 => {
                h["lm_head.weight"] = h["model.embed_tokens.weight"].clone();
            }
            _ => h[name]["data_offsets"] = json!([1, 3]),
        }
        let raw = serde_json::to_vec(&h).unwrap();
        assert!(
            Plan::inspect(
                c,
                &raw,
                (8 + raw.len() + m.len() - 8 - header(m).len()) as u64,
                LIMIT
            )
            .is_err()
        );
    }
    let t = Temp::new();
    fs::write(t.0.join("model.safetensors.index.json"), b"{}").unwrap();
    assert!(Source::open(&t.0, t.expected(), LIMIT).is_err());
}
#[test]
fn real_metadata_exact_storage_is_not_session_admission() {
    let f = fixture();
    let c = include_bytes!("fixtures/granite-moe-source-v1/real-config.json");
    let h = include_bytes!("fixtures/granite-moe-source-v1/real-header.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(c)),
        f["real_metadata"]["config_sha256"]
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(h)),
        f["real_metadata"]["header_sha256"]
    );
    let p = Plan::inspect(c, h, 2669283096, LIMIT).unwrap();
    let s = p.storage();
    assert_eq!(p.tensors().len(), 218);
    assert_eq!(s.half_weights_bytes, 2669256704);
    assert_eq!(s.largest_tensor_bytes, 100669440);
    assert_eq!(p.prospective_kv_bytes(512).unwrap(), 25165824);
    assert!(p.prospective_kv_bytes(0).is_err());
    assert!(p.prospective_kv_bytes(131073).is_err());
    assert_eq!(
        s.admitted_load_bytes,
        s.half_weights_bytes + s.import_owned_bytes + s.metadata_reservation_bytes
    );
    println!(
        "MOE_STORAGE {}",
        json!({"half_weights_bytes":s.half_weights_bytes,"largest_tensor_bytes":s.largest_tensor_bytes,"import_owned_bytes":s.import_owned_bytes,"metadata_reservation_bytes":s.metadata_reservation_bytes,"admitted_load_bytes":s.admitted_load_bytes,"prospective_kv512_bytes":p.prospective_kv_bytes(512).unwrap(),"session_or_payload_qualified":false})
    );
}

#[test]
fn multi_chunk_direct_import_keeps_owner_storage_bounded() {
    let t = Temp::new();
    let small = t.open().plan().storage().import_owned_bytes;
    let mut config: Value =
        serde_json::from_slice(&fs::read(t.0.join("config.json")).unwrap()).unwrap();
    config["intermediate_size"] = json!(6001);
    let original = fs::read(t.0.join("model.safetensors")).unwrap();
    let mut h: Value = serde_json::from_slice(header(&original)).unwrap();
    h["model.layers.0.block_sparse_moe.input_linear.weight"]["shape"] = json!([3, 12002, 2]);
    h["model.layers.0.block_sparse_moe.output_linear.weight"]["shape"] = json!([3, 2, 6001]);
    let mut payload = Vec::new();
    for spec in h.as_object_mut().unwrap().values_mut() {
        let count = spec["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_u64().unwrap() as usize)
            .product::<usize>();
        let begin = payload.len();
        for _ in 0..count {
            payload.extend_from_slice(&0x3f80u16.to_le_bytes());
        }
        spec["data_offsets"] = json!([begin, payload.len()]);
    }
    let mut raw = serde_json::to_vec(&h).unwrap();
    while !raw.len().is_multiple_of(8) {
        raw.push(b' ');
    }
    let mut model = (raw.len() as u64).to_le_bytes().to_vec();
    model.extend_from_slice(&raw);
    model.extend_from_slice(&payload);
    fs::write(
        t.0.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    fs::write(t.0.join("model.safetensors"), model).unwrap();
    let mut s = t.open();
    assert!(s.plan().storage().largest_tensor_bytes > 2 * 64 * 1024);
    assert_eq!(s.plan().storage().import_owned_bytes, small);
    for i in 0..s.plan().tensors().len() {
        let mut out = vec![0; s.plan().tensors()[i].elements()];
        s.import_next(&mut out).unwrap();
        assert!(out.iter().all(|&x| x == 0x3c00));
    }
    assert_eq!(s.finish().unwrap().underflowed_values(), 0);
}

#[test]
fn source_hash_and_regular_file_boundaries_reject_unsupported_inputs() {
    let t = Temp::new();
    let mut e = t.expected();
    e.model_sha256 = [0; 32];
    assert!(Source::open(&t.0, e, LIMIT).is_err());
    #[cfg(unix)]
    {
        fs::rename(t.0.join("model.safetensors"), t.0.join("held.safetensors")).unwrap();
        std::os::unix::fs::symlink("held.safetensors", t.0.join("model.safetensors")).unwrap();
        assert!(Source::open(&t.0, t.expected(), LIMIT).is_err());
    }
}
