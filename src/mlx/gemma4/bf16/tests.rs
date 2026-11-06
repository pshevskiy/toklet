use super::*;
use std::io::Read;

const CONFIG: &[u8] =
    include_bytes!("../../../../tests/fixtures/gemma-unquantized-header-v1/config.json");
const HEADER: &[u8] =
    include_bytes!("../../../../tests/fixtures/gemma-unquantized-header-v1/header.json");

#[test]
fn exact_official_plan_matches_independent_inventory_and_ple_partition() {
    let plan = inspect(CONFIG, HEADER, CONTAINER_BYTES, ModalityPolicy::TextOnly).unwrap();
    let layout: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../../../tests/fixtures/gemma-unquantized-header-v1/layout.json"
    ))
    .unwrap();
    let expected = layout["selected_text"].as_object().unwrap();
    assert_eq!(plan.text_tensors().len(), expected.len());
    for (name, tensor) in plan.text_tensors() {
        let reference = &expected[name];
        assert_eq!(json!(tensor.shape), reference["shape"], "{name}");
        assert_eq!(json!(tensor.range.len()), reference["bytes"], "{name}");
        assert_eq!(
            json!(plan.payload_offset() + tensor.range.start),
            reference["file_offset"],
            "{name}"
        );
    }
    assert_eq!(plan.per_layer_embedding().shape, [262144, 8960]);
    assert_eq!(plan.per_layer_embedding().range.len(), 4_697_620_480);
    assert_eq!(plan.resident_tensor_bytes(), 4_559_518_278);
    assert_eq!(plan.media_bytes(), 951_458_176);
    assert_eq!(plan.unused_shared_kv_bytes(), 37_761_024);
    assert_eq!(plan.layers()[13].producer, 13);
    assert_eq!(plan.layers()[34].producer, 14);
}

#[test]
fn source_pins_reject_mutations_truncation_and_implicit_modalities() {
    let mut config = CONFIG.to_vec();
    config[0] ^= 1;
    let mut header = HEADER.to_vec();
    header[100] ^= 1;
    for (config, header, bytes, policy) in [
        (
            config.as_slice(),
            HEADER,
            CONTAINER_BYTES,
            ModalityPolicy::TextOnly,
        ),
        (
            CONFIG,
            header.as_slice(),
            CONTAINER_BYTES,
            ModalityPolicy::TextOnly,
        ),
        (
            CONFIG,
            &HEADER[..HEADER.len() - 1],
            CONTAINER_BYTES,
            ModalityPolicy::TextOnly,
        ),
        (
            CONFIG,
            HEADER,
            CONTAINER_BYTES - 1,
            ModalityPolicy::TextOnly,
        ),
        (CONFIG, HEADER, CONTAINER_BYTES, ModalityPolicy::Reject),
    ] {
        assert!(inspect(config, header, bytes, policy).is_err());
    }
}

#[test]
#[ignore = "explicit official Gemma BF16 shard metadata probe"]
fn actual_official_bf16_config_and_header_match() {
    let root = std::path::PathBuf::from(std::env::var_os("TOKLET_GEMMA_BF16_MODEL").unwrap());
    let config = std::fs::read(root.join("config.json")).unwrap();
    let file = std::fs::File::open(root.join("model.safetensors")).unwrap();
    let extent = file.metadata().unwrap().len();
    let mut source = file.take((HEADER_BYTES + 8) as u64);
    let mut len = [0u8; 8];
    source.read_exact(&mut len).unwrap();
    assert_eq!(u64::from_le_bytes(len), HEADER_BYTES as u64);
    let mut header = Vec::new();
    source.read_to_end(&mut header).unwrap();
    let plan = inspect(&config, &header, extent, ModalityPolicy::TextOnly).unwrap();
    assert_eq!(source.limit(), 0);
    println!(
        "{}",
        json!({
            "header_bytes_read":header.len() + 8,
            "text_tensors":plan.text_tensors().len(),
            "resident_tensor_bytes":plan.resident_tensor_bytes(),
            "file_backed_ple_bytes":plan.per_layer_embedding().range.len(),
            "media_excluded_bytes":plan.media_bytes(),
            "unused_shared_kv_excluded_bytes":plan.unused_shared_kv_bytes(),
        })
    );
}
