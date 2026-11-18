use super::{Router, Routing, bf16};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const H: usize = 4096;
const EXPECTED_RANGE: &str = "58880a9699bb6affeae7732530063739dc9d787b4e03709ba6795fdc7eee5f8d";
const EXPECTED_FIXTURE: &str = "248c0d81f674a581ff43325182d84fffb5dd674fa8fa07fbd7346eee1a17b1f7";
const EXPECTED_REPORT: &str = "af0e8494c431a34eb2806d96e627f8a0790e670faf1219906d1d05e0f45452d7";

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn pinned(path: &Path, max_bytes: u64, expected: &str) -> Vec<u8> {
    let meta = std::fs::symlink_metadata(path).unwrap();
    assert!(meta.is_file() && meta.len() <= max_bytes);
    let raw = std::fs::read(path).unwrap();
    assert_eq!(raw.len() as u64, meta.len());
    assert_eq!(digest(&raw), expected);
    raw
}

fn bf_words(bytes: &[u8]) -> Vec<bf16> {
    bytes
        .chunks_exact(2)
        .map(|w| bf16::from_bits(u16::from_le_bytes(w.try_into().unwrap())))
        .collect()
}

fn check_coefficients(routing: &Routing, expected: &serde_json::Value, tolerance: f32) -> f32 {
    let actual = routing
        .post
        .into_iter()
        .chain(routing.mix.into_iter().flatten());
    let expected = expected["post"]
        .as_array()
        .unwrap()
        .iter()
        .chain(expected["mix"].as_array().unwrap());
    let mut max = 0.0_f32;
    for (x, y) in actual.zip(expected) {
        let y = y.as_f64().unwrap() as f32;
        assert!(x.is_finite());
        let error = (x - y).abs();
        max = max.max(error);
        assert!(error <= tolerance);
        assert_eq!(bf16::from_f32(x).to_bits(), bf16::from_f32(y).to_bits());
    }
    max
}

#[test]
#[ignore = "explicit sealed layer4 checkpoint router oracle with synthetic branches"]
fn actual_layer4_checkpoint_mhc_matches_independent_oracle() {
    let dir = PathBuf::from(std::env::var_os("TOKLET_SENSENOVA_MHC_ORACLE_DIR").unwrap());
    let checkpoint =
        PathBuf::from(std::env::var_os("TOKLET_SENSENOVA_MHC_CHECKPOINT_RANGE").unwrap());
    let raw = pinned(&checkpoint, 524352, EXPECTED_RANGE);
    let oracle = pinned(&dir.join("oracle.bin"), 73728, EXPECTED_FIXTURE);
    let report = pinned(&dir.join("oracle.json"), 8192, EXPECTED_REPORT);
    assert_eq!((raw.len(), oracle.len()), (524352, 73728));
    let report: serde_json::Value = serde_json::from_slice(&report).unwrap();
    assert_eq!(report["checkpoint_range_sha256"], EXPECTED_RANGE);
    assert_eq!(report["sha256"], EXPECTED_FIXTURE);
    assert_eq!(report["hidden"], H);
    let tolerance = report["coefficient_atol"].as_f64().unwrap() as f32;
    assert_eq!(tolerance, 2e-5);

    let stage = |label: &str, words: usize| {
        let desc = &report["arrays"][label];
        let start = desc["offset"].as_u64().unwrap() as usize;
        assert_eq!(desc["bytes"], words * 2);
        let bytes = &oracle[start..start + words * 2];
        assert_eq!(desc["sha256"], digest(bytes));
        bf_words(bytes)
    };
    let params: Vec<_> = raw
        .chunks_exact(4)
        .map(|w| f32::from_le_bytes(w.try_into().unwrap()))
        .collect();
    let routers = [("attention", 262176 / 4), ("ffn", 0)].map(|(name, bias)| {
        let weight = &params[bias + 8..bias + 8 + 16 * H];
        let bias = (&params[bias..bias + 8]).try_into().unwrap();
        (name, Router::new(H, weight, bias).unwrap())
    });
    let input = stage("initial_hidden", H);
    let mut residual = [input.clone(), input];
    let mut branch = vec![bf16::ZERO; H];
    let mut errors = [0.0; 2];
    for (i, (name, router)) in routers.into_iter().enumerate() {
        let inputs = residual.each_ref().map(Vec::as_slice);
        let routing = router.route(inputs, &mut branch).unwrap();
        assert_eq!(
            branch,
            stage(&format!("{name}_routed"), H),
            "{name} branch input"
        );
        assert!(
            report[name]["minimum_coefficient_midpoint_margin"]
                .as_f64()
                .unwrap()
                > f64::from(2.0 * tolerance)
        );
        errors[i] = check_coefficients(&routing, &report[name], tolerance);

        let update = stage(&format!("{name}_synthetic_branch"), H);
        let mut result = [vec![bf16::ZERO; H], vec![bf16::ZERO; H]];
        routing
            .merge(inputs, &update, result.each_mut().map(Vec::as_mut_slice))
            .unwrap();
        let expected = stage(&format!("{name}_merged"), 2 * H);
        for (actual, expected) in result.iter().zip(expected.chunks_exact(H)) {
            assert_eq!(actual.as_slice(), expected, "{name} merged stream");
        }
        assert_ne!(
            result[0], result[1],
            "{name} must mix distinct residual streams"
        );
        residual = result;
    }
    println!(
        "SENSENOVA_LAYER4_MHC_ORACLE {}",
        serde_json::json!({
            "checkpoint_range_sha256":EXPECTED_RANGE,"oracle_sha256":EXPECTED_FIXTURE,
            "reference_metadata_sha256":EXPECTED_REPORT,"exact_routed_bf16_words":2*H,
            "exact_merged_bf16_words":4*H,"checked_post_and_mix_coefficients":12,
            "coefficient_max_absolute_errors":errors,"coefficient_atol":tolerance,
            "synthetic_branches":true,"whole_model_qualified":false
        })
    );
}
