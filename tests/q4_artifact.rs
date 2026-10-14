use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};
use toklet::{q4, weights};

static NEXT: AtomicUsize = AtomicUsize::new(0);

#[test]
fn matrix_inventory_rejects_unbounded_layers_before_allocating() {
    let mut config = toklet::model::Config {
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 64,
        num_layers: usize::MAX,
        query_heads: 4,
        kv_heads: 2,
        head_dim: 8,
        max_positions: 32,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
    };
    assert!(q4::matrix_shapes(&config).is_err());
    config.num_layers = 1025;
    assert!(q4::matrix_shapes(&config).is_err());
    config.num_layers = 2;
    let shapes = q4::matrix_shapes(&config).unwrap();
    assert_eq!(shapes.len(), 14);
    assert_eq!(shapes["model.layers.0.self_attn.k_proj.weight"], [16, 32]);
    assert_eq!(shapes["model.layers.1.mlp.down_proj.weight"], [32, 64]);
}
struct Fixture {
    root: PathBuf,
    manifest: Value,
}
fn sha(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "toklet-q4-artifact-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        fs::create_dir(root.join("source")).unwrap();
        fs::create_dir(root.join("packed")).unwrap();
        fs::write(root.join("source/config.json"), b"{}").unwrap();
        fs::write(root.join("source/model.safetensors"), b"source payload").unwrap();
        let data = include_bytes!("fixtures/q4/rows.q4");
        fs::write(root.join("packed/test.q4"), data).unwrap();
        let manifest = json!({"format":"toklet-q4-model-v1","quantizer":q4::QUANTIZER,
            "config_sha256":sha(b"{}"),"sources":[{"file":"model.safetensors","sha256":sha(b"source payload")}],
            "matrices":[{"name":"test","file":"test.q4","rows":2,"columns":65,"packed_bytes":data.len(),
                "packed_sha256":sha(data),"source_f32_sha256":"0".repeat(64)}],"packed_matrix_bytes":data.len(),
            "exclusions":["norms","qkv_biases","tied_embedding_output"],"excluded_dtype":"F16"});
        let fixture = Self { root, manifest };
        fixture.save();
        fixture
    }
    fn save(&self) {
        fs::write(
            self.root.join("packed/manifest.json"),
            serde_json::to_vec(&self.manifest).unwrap(),
        )
        .unwrap();
    }
    fn open(&self, limit: usize) -> Result<q4::Artifact, weights::Error> {
        q4::Artifact::open(
            &self.root.join("packed"),
            &self.root.join("source"),
            &BTreeMap::from([("test".into(), vec![2, 65])]),
            &["model.safetensors".into()],
            limit,
        )
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).unwrap();
    }
}

#[test]
fn validates_source_bound_inventory_and_exact_load_budget_without_gpu() {
    let f = Fixture::new();
    let artifact = f.open(weights::INITIAL_MEMORY_LIMIT_BYTES).unwrap();
    assert_eq!(
        (artifact.packed_matrix_bytes, artifact.largest_matrix_bytes),
        (320, 320)
    );
    let exact = artifact.admitted_packed_load_bytes;
    assert!(f.open(exact).is_ok());
    assert!(f.open(exact - 1).is_err());
    assert!(f.open(0).is_err());
    assert!(f.open(weights::INITIAL_MEMORY_LIMIT_BYTES + 1).is_err());
    assert_eq!(
        artifact.read_matrix("test").unwrap(),
        include_bytes!("fixtures/q4/rows.q4")
    );
    assert!(artifact.read_matrix("absent").is_err());
    fs::write(f.root.join("packed/test.q4"), vec![0; 320]).unwrap();
    assert!(artifact.read_matrix("test").is_err());
    assert!(f.open(exact).is_err());
}

#[test]
fn rejects_identity_mismatch_duplicate_names_paths_shapes_and_policy() {
    let f = Fixture::new();
    let original = f.manifest.clone();
    let artifact = f.open(weights::INITIAL_MEMORY_LIMIT_BYTES).unwrap();
    fs::write(f.root.join("source/config.json"), b"{ }").unwrap();
    assert!(artifact.verify_sources().is_err());
    assert!(f.open(weights::INITIAL_MEMORY_LIMIT_BYTES).is_err());
    fs::write(f.root.join("source/config.json"), b"{}").unwrap();
    fs::write(f.root.join("source/model.safetensors"), b"changed source").unwrap();
    assert!(artifact.verify_sources().is_err());
    assert!(f.open(weights::INITIAL_MEMORY_LIMIT_BYTES).is_err());
    fs::write(f.root.join("source/model.safetensors"), b"source payload").unwrap();
    for pointer in [
        "/format",
        "/quantizer",
        "/excluded_dtype",
        "/sources/0/file",
        "/sources/0/sha256",
        "/matrices/0/file",
        "/matrices/0/name",
        "/matrices/0/source_f32_sha256",
    ] {
        let mut changed = original.clone();
        *changed.pointer_mut(pointer).unwrap() = json!("../invalid");
        fs::write(
            f.root.join("packed/manifest.json"),
            serde_json::to_vec(&changed).unwrap(),
        )
        .unwrap();
        assert!(
            f.open(weights::INITIAL_MEMORY_LIMIT_BYTES).is_err(),
            "{pointer}"
        );
    }
    for pointer in [
        "/matrices/0/rows",
        "/matrices/0/columns",
        "/matrices/0/packed_bytes",
        "/packed_matrix_bytes",
    ] {
        let mut changed = original.clone();
        *changed.pointer_mut(pointer).unwrap() = json!(usize::MAX);
        fs::write(
            f.root.join("packed/manifest.json"),
            serde_json::to_vec(&changed).unwrap(),
        )
        .unwrap();
        assert!(f.open(weights::INITIAL_MEMORY_LIMIT_BYTES).is_err());
    }
    for field in ["sources", "matrices"] {
        let mut changed = original.clone();
        let entry = changed[field][0].clone();
        changed[field].as_array_mut().unwrap().push(entry);
        fs::write(
            f.root.join("packed/manifest.json"),
            serde_json::to_vec(&changed).unwrap(),
        )
        .unwrap();
        assert!(f.open(weights::INITIAL_MEMORY_LIMIT_BYTES).is_err());
    }
}

#[test]
fn rejects_oversized_metadata_and_hashed_but_invalid_matrix_bytes() {
    let mut f = Fixture::new();
    fs::write(
        f.root.join("packed/manifest.json"),
        vec![b' '; 1024 * 1024 + 1],
    )
    .unwrap();
    assert!(f.open(weights::INITIAL_MEMORY_LIMIT_BYTES).is_err());
    let mut data = include_bytes!("fixtures/q4/rows.q4").to_vec();
    data[104] = 1;
    f.manifest["matrices"][0]["packed_sha256"] = json!(sha(&data));
    f.save();
    fs::write(f.root.join("packed/test.q4"), data).unwrap();
    let a = f.open(weights::INITIAL_MEMORY_LIMIT_BYTES).unwrap();
    assert!(a.read_matrix("test").is_err());
}

#[test]
#[ignore = "explicit local source+packed artifact check; no GPU; set TOKLET_Q4_SOURCE and TOKLET_Q4_PACKED"]
fn packed_source_artifact_matches_independent_known_geometry() {
    let source = PathBuf::from(std::env::var_os("TOKLET_Q4_SOURCE").expect("source path"));
    let packed = PathBuf::from(std::env::var_os("TOKLET_Q4_PACKED").expect("packed path"));
    let kind = std::env::var("TOKLET_Q4_KIND").unwrap();
    let (vocab_size, hidden_size, intermediate_size, num_layers, query_heads, kv_heads, head_dim) =
        match kind.as_str() {
            "tiny" => (64, 32, 64, 2, 4, 2, 8),
            "qwen" => (151936, 896, 4864, 24, 14, 2, 64),
            _ => panic!("explicitly qualified geometry required"),
        };
    let config = toklet::model::Config {
        vocab_size,
        hidden_size,
        intermediate_size,
        num_layers,
        query_heads,
        kv_heads,
        head_dim,
        max_positions: 2048,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
    };
    let shapes = q4::matrix_shapes(&config).unwrap();
    let artifact = q4::Artifact::open(
        &packed,
        &source,
        &shapes,
        &["model.safetensors".into()],
        weights::INITIAL_MEMORY_LIMIT_BYTES,
    )
    .unwrap();
    for (name, shape) in &shapes {
        let data = artifact.read_matrix(name).unwrap();
        let matrix = q4::Matrix::parse(&data).unwrap();
        assert_eq!([matrix.rows(), matrix.columns()], [shape[0], shape[1]]);
        assert!(matrix.value(0, 0).unwrap().is_finite());
        assert!(
            matrix
                .value(shape[0] - 1, shape[1] - 1)
                .unwrap()
                .is_finite()
        );
    }
    artifact.verify_sources().unwrap();
    println!(
        "{}",
        json!({"matrices":shapes.len(), "packed_matrix_bytes":artifact.packed_matrix_bytes,
        "largest_matrix_bytes":artifact.largest_matrix_bytes,"admitted_packed_load_bytes":artifact.admitted_packed_load_bytes})
    );
}

#[test]
fn v2_inventory_binds_header_version_and_preserves_v1() {
    let mut f = Fixture::new();
    let data = include_bytes!("fixtures/q4-v2/rows.q4");
    fs::write(f.root.join("packed/test.q4"), data).unwrap();
    f.manifest["format"] = json!("toklet-q4-model-v2");
    f.manifest["quantizer"] = json!(q4::Format::V2.quantizer());
    f.manifest["matrices"][0]["columns"] = json!(17);
    f.manifest["matrices"][0]["packed_bytes"] = json!(data.len());
    f.manifest["matrices"][0]["packed_sha256"] = json!(sha(data));
    f.manifest["packed_matrix_bytes"] = json!(data.len());
    f.save();
    let open = || {
        q4::Artifact::open(
            &f.root.join("packed"),
            &f.root.join("source"),
            &BTreeMap::from([("test".into(), vec![2, 17])]),
            &["model.safetensors".into()],
            weights::INITIAL_MEMORY_LIMIT_BYTES,
        )
    };
    let artifact = open().unwrap();
    assert_eq!(artifact.format(), q4::Format::V2);
    assert_eq!(artifact.read_matrix("test").unwrap(), data);
    // Shape/byte count/hash can match while the authenticated matrix format differs.
    let mut mixed = data.to_vec();
    mixed[..8].copy_from_slice(q4::MAGIC);
    mixed[8..12].copy_from_slice(&1u32.to_le_bytes());
    mixed[12..16].copy_from_slice(&64u32.to_le_bytes());
    mixed[64..].fill(0);
    assert_eq!(q4::Matrix::parse(&mixed).unwrap().format(), q4::Format::V1);
    fs::write(f.root.join("packed/test.q4"), &mixed).unwrap();
    assert!(artifact.read_matrix("test").is_err());
    f.manifest["matrices"][0]["packed_sha256"] = json!(sha(&mixed));
    f.save();
    let mixed_artifact = q4::Artifact::open(
        &f.root.join("packed"),
        &f.root.join("source"),
        &BTreeMap::from([("test".into(), vec![2, 17])]),
        &["model.safetensors".into()],
        weights::INITIAL_MEMORY_LIMIT_BYTES,
    )
    .unwrap();
    assert!(mixed_artifact.read_matrix("test").is_err());
    f.manifest["quantizer"] = json!(q4::QUANTIZER);
    f.save();
    assert!(
        q4::Artifact::open(
            &f.root.join("packed"),
            &f.root.join("source"),
            &BTreeMap::from([("test".into(), vec![2, 17])]),
            &["model.safetensors".into()],
            weights::INITIAL_MEMORY_LIMIT_BYTES
        )
        .is_err()
    );
}
