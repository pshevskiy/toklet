//! Portable integrity checks for independently frozen Q4 v2 real-model probes.
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

#[test]
fn real_model_v2_probes_bind_shapes_gates_and_available_references() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for (model, identity) in [
        (
            "small",
            "6275f1dbdeca79210352463341d0e5416c7ffcd3e708c1ff6303367119fd3a29",
        ),
        (
            "larger",
            "960e24d1dacc37ed7decde6c038f1cbae200a1ca05151861b5142e6d981d22bc",
        ),
    ] {
        let directory = root.join(format!("tests/fixtures/qwen-{model}-q4-v2-v1"));
        let data = fs::read(directory.join("manifest.json")).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&data)), identity);
        let manifest: serde_json::Value = serde_json::from_slice(&data).unwrap();
        assert_eq!(manifest["format"], "toklet-streamed-qwen-storage-oracle-v2");
        assert_eq!(manifest["q4_version"], 2);
        assert_eq!(manifest["cases"].as_object().unwrap().len(), 6);
        assert_eq!(manifest["arrays"].as_object().unwrap().len(), 342);
        assert_eq!(manifest["policy"]["rtol"], 1e-3);
        assert_eq!(manifest["policy"]["atol_floor"], 1e-4);
        assert_eq!(manifest["policy"]["envelope_multiplier"], 2.);
        for (name, item) in manifest["arrays"].as_object().unwrap() {
            let bytes = fs::read(directory.join(item["file"].as_str().unwrap())).unwrap();
            assert_eq!(bytes.len() as u64, item["bytes"].as_u64().unwrap());
            assert_eq!(
                format!("{:x}", Sha256::digest(&bytes)),
                item["sha256"].as_str().unwrap()
            );
            let shape = item["shape"].as_array().unwrap();
            let elements = shape.iter().map(|x| x.as_u64().unwrap()).product::<u64>();
            let indices = item["indices"].as_array().unwrap();
            assert!(indices.windows(2).all(|w| w[0].as_u64() < w[1].as_u64()));
            assert!(indices.iter().all(|x| x.as_u64().unwrap() < elements));
            let size = match item["dtype"].as_str().unwrap() {
                "f16le" => 2,
                "f32le" => 4,
                _ => panic!("frozen dtype"),
            };
            assert_eq!(bytes.len(), indices.len() * size);
            assert_eq!(
                item["full_array_bytes"].as_u64().unwrap(),
                elements * size as u64
            );
            let gate = &item["gate"];
            assert_eq!(gate["rtol"], 1e-3);
            assert_eq!(
                gate["atol"].as_f64().unwrap(),
                1e-4_f64
                    .max(2. * gate["pooled_fp32_fp64_max_abs"].as_f64().unwrap())
                    .max(2. * gate["equivalent_prefix_max_abs"].as_f64().unwrap())
            );
            let new_case =
                name.starts_with("cases.tile17.") || name.starts_with("cases.tile17_offset5.");
            assert_eq!(item["source_fp32_sha256"].is_null(), new_case);
        }
        for case in ["tile17", "tile17_offset5"] {
            assert_eq!(
                manifest["cases"][case]["tokens"].as_array().unwrap().len(),
                17
            );
            assert_eq!(manifest["argmax"][case].as_object().unwrap().len(), 2);
            assert!(manifest["argmax"][case].get("source_fp32").is_none());
        }
        assert_eq!(manifest["cases"]["tile17_offset5"]["offset"], 5);
    }
}
