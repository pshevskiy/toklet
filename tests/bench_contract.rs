use std::{fs, path::PathBuf};

use serde_json::Value;
use sha2::{Digest, Sha256};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks")
}

fn manifest() -> Value {
    serde_json::from_slice(&fs::read(root().join("contract-v1/manifest.json")).unwrap()).unwrap()
}

fn tokens(manifest: &Value, name: &str) -> Vec<u32> {
    let array = &manifest["arrays"][name];
    let bytes = fs::read(
        root()
            .join("contract-v1")
            .join(array["file"].as_str().unwrap()),
    )
    .unwrap();
    assert_eq!(array["dtype"], "u32le");
    assert_eq!(bytes.len() as u64, array["count"].as_u64().unwrap() * 4);
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        array["sha256"].as_str().unwrap()
    );
    bytes
        .chunks_exact(4)
        .map(|value| u32::from_le_bytes(value.try_into().unwrap()))
        .collect()
}

#[test]
fn offline_workload_arrays_have_frozen_identities_and_exact_admission_boundary() {
    let manifest = manifest();
    for name in manifest["arrays"].as_object().unwrap().keys() {
        assert!(tokens(&manifest, name).iter().all(|&token| token < 151936));
    }
    for count in [128, 512, 1024, 1920, 1921] {
        assert_eq!(tokens(&manifest, &format!("prompt_{count}")).len(), count);
    }
    let output = tokens(&manifest, "continuation_128");
    assert_eq!(output.len(), 128);
    assert_eq!(tokens(&manifest, "prompt_1920").len() + output.len(), 2048);
    assert!(tokens(&manifest, "prompt_1921").len() + output.len() > 2048);
    assert_eq!(tokens(&manifest, "held_out_loss").len(), 10001);
    let base = tokens(&manifest, "prompt_512");
    let changed = tokens(&manifest, "changed_prompt_512");
    let changed_positions: Vec<_> = base
        .iter()
        .zip(&changed)
        .enumerate()
        .filter_map(|(i, (a, b))| (a != b).then_some(i))
        .collect();
    assert_eq!(changed_positions, [128]);
}

#[test]
fn quality_policy_and_pinned_source_model_are_available_without_downloads() {
    let manifest = manifest();
    let quality = &manifest["quality"];
    assert_eq!(quality["task_ids"].as_array().unwrap().len(), 100);
    assert_eq!(
        quality["categories"],
        serde_json::json!({"code": 40, "prose": 30, "continuation": 30})
    );
    assert_eq!(quality["mean_nll_increase_limit_nats_per_token"], 0.05);
    assert_eq!(quality["task_pass_rate_loss_limit_percentage_points"], 2.0);
    let source_bytes = fs::read(root().join("sources.json")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&source_bytes)),
        manifest["sources_sha256"].as_str().unwrap()
    );
    let sources: Value = serde_json::from_slice(&source_bytes).unwrap();
    assert_eq!(
        sources["model"]["revision"],
        "7ae557604adf67be50417f59c2c2f167def9a775"
    );
    assert_eq!(
        sources["model"]["tokenizer_revision"],
        sources["model"]["revision"]
    );
    assert_eq!(
        sources["model"]["engine_working_set_limit_bytes"],
        6 * 1024_u64.pow(3)
    );
    let schema: Value =
        serde_json::from_slice(&fs::read(root().join("result.schema.json")).unwrap()).unwrap();
    assert_eq!(
        schema["$schema"],
        "https://json-schema.org/draft/2020-12/schema"
    );
}
