use super::*;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

fn directory(name: &str) -> PathBuf {
    PathBuf::from(std::env::var_os(name).expect("explicit pinned assistant fixture"))
}

fn f32_values(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len().is_multiple_of(4));
    bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

fn compare(expected: &[u8], actual: &[f32]) {
    let reference = f32_values(expected);
    assert_eq!(reference.len(), actual.len());
    for (i, (&a, &b)) in reference.iter().zip(actual).enumerate() {
        let (a, b) = (f64::from(a), f64::from(b));
        let limit = 1e-5 + 1e-4 * a.abs();
        assert!(
            (a - b).abs() <= limit,
            "assistant word {i} has {b}, expected {a}"
        );
    }
}

fn kv(input: &[f32], keys: usize, width: usize, start: usize) -> [AttentionKvSegment<'_>; 3] {
    assert_eq!(input.len(), 2 * keys * width);
    let (k, v) = input.split_at(keys * width);
    let empty = AttentionKvSegment {
        k: &[],
        v: &[],
        keys: 0,
        head_stride: 0,
        key_start: start + keys,
    };
    [
        AttentionKvSegment {
            k,
            v,
            keys,
            head_stride: k.len(),
            key_start: start,
        },
        empty,
        empty,
    ]
}

fn check_case(assistant: &Assistant, root: &Path, position: usize) {
    let input = std::fs::read(root.join(format!("synthetic-{position}.bin"))).unwrap();
    let expected = root.join(format!("reference-{position}"));
    let output = std::fs::read(expected.join("reference.bin")).unwrap();
    let record: serde_json::Value =
        serde_json::from_slice(&std::fs::read(expected.join("receipt.json")).unwrap()).unwrap();
    assert_eq!(record["position"].as_u64(), Some(position as u64));
    assert_eq!(
        record["snapshot_sha256"].as_str().unwrap(),
        format!("{:x}", Sha256::digest(&input)),
    );
    let arrays = record["arrays"].as_array().unwrap();
    let wanted = |name: &str| {
        let entry = arrays.iter().find(|entry| entry["name"] == name).unwrap();
        let offset = entry["offset"].as_u64().unwrap() as usize;
        let bytes = 4 * entry["elements"].as_u64().unwrap() as usize;
        let array = &output[offset..offset + bytes];
        assert_eq!(
            entry["sha256"].as_str().unwrap(),
            format!("{:x}", Sha256::digest(array))
        );
        array
    };
    let input = f32_values(&input);
    let (x, cache) = input.split_at(2 * TARGET_HIDDEN);
    let local_keys = (position + 1).min(WINDOW);
    let (local, full) = cache.split_at(2 * local_keys * 256);
    let full_keys = position + 1;
    let target = TargetKv {
        position,
        local: kv(local, local_keys, 256, full_keys - local_keys),
        full: kv(full, full_keys, 512, 0),
    };
    let planned = Scratch::planned_bytes(full_keys).unwrap();
    assert!(matches!(
        Scratch::new(full_keys, planned - 1),
        Err(Error::Budget { .. })
    ));
    let mut scratch = Scratch::new(full_keys, planned).unwrap();
    let proposal = assistant
        .propose(x.try_into().unwrap(), &target, &mut scratch)
        .unwrap();
    assert_eq!(Some(proposal.winner as u64), record["winner"].as_u64());
    compare(wanted("post_projection"), proposal.hidden);
    compare(wanted("logits"), proposal.logits);
    assert!(
        wanted("ids")
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .eq(proposal.ids.iter().copied())
    );
    compare(wanted("centroid_scores"), &scratch.centroid_scores);
    compare(wanted("final_norm"), &scratch.normal);
    println!(
        "ASSISTANT_PROPOSAL position={position} winner={} scratch_bytes={planned}",
        record["winner"]
    );
}

#[test]
#[ignore = "explicit official BF16 assistant and independent reference outputs"]
fn official_assistant_draft_matches_independent_reference() {
    let asset = directory("TOKLET_GEMMA_ASSISTANT_DIRECTORY");
    let fixture = directory("TOKLET_GEMMA_ASSISTANT_REFERENCE");
    assert!(matches!(
        Assistant::open(&asset, LOAD_BYTES - 1),
        Err(Error::Budget { .. })
    ));
    let model = Assistant::open(&asset, LOAD_BYTES).unwrap();
    for position in [0, 31, 513] {
        check_case(&model, &fixture, position);
    }
}
