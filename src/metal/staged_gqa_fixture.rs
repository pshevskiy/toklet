//! Test-only frozen source words and compact raw evidence helpers, not numerics.
use half::f16;
use serde_json::Value;
use sha2::{Digest, Sha256};

pub(super) const RAW: &[u8] = include_bytes!("../../tests/fixtures/staged-gqa-half-v1/arrays.bin");
pub(super) const INPUTS: [&str; 5] = ["q", "old_k", "old_v", "new_k", "new_v"];
pub(super) fn number(v: &Value) -> usize {
    usize::try_from(v.as_u64().unwrap()).unwrap()
}
pub(super) fn fixture() -> Value {
    let bytes = include_bytes!("../../tests/fixtures/staged-gqa-half-v1/manifest.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "f1fc055f5949fa068c86e2519046a398a7e490d8ec362cca18d464d60e7a2d7e"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(RAW)),
        "f7a03826b5e870468aa5596b560d09cacf1ceda3dc4edc63b1d477f12cb11d3d"
    );
    let m: Value = serde_json::from_slice(bytes).unwrap();
    assert_eq!(m["format"], "toklet-staged-gqa-half-v1");
    assert_eq!(
        m["protocol_sha256"],
        "55a1f4227dfed70a08df601044778a51d7f80d5e742ad6976f469050f5784455"
    );
    assert_eq!(m["source_reference_executed"], true);
    assert_eq!(m["target_gpu_executed"], false);
    assert_eq!(m["full_graph_executed"], false);
    assert_eq!(
        m["policy"],
        serde_json::json!({"fatal_reference":"main","atol_floor":1e-4,"envelope_factor":2,"rtol":1e-3})
    );
    assert_eq!(m["stored_arrays"], 143);
    assert_eq!(m["numerical_profile_arrays"], 66);
    assert_eq!(m["physical_state_arrays"], 77);
    assert_eq!(m["cases"].as_array().unwrap().len(), 11);
    assert_eq!(m["files"]["arrays.bin"]["bytes"], RAW.len());
    for (name, raw) in [
        (
            "producer",
            include_bytes!("../../tools/reference/staged_gqa_half.py").as_slice(),
        ),
        (
            "helper",
            include_bytes!("../../tools/reference/gemma3_staged_attention.py").as_slice(),
        ),
    ] {
        assert_eq!(m[name]["sha256"], format!("{:x}", Sha256::digest(raw)));
    }
    m
}
pub(super) fn bytes(d: &Value) -> &'static [u8] {
    assert_eq!(d["file"], "arrays.bin");
    let start = number(&d["offset"]);
    let len = number(&d["bytes"]);
    let raw = &RAW[start..start.checked_add(len).unwrap()];
    assert_eq!(d["sha256"], format!("{:x}", Sha256::digest(raw)));
    let item = match d["dtype"].as_str().unwrap() {
        "u16le" => 2,
        "f32le" => 4,
        _ => panic!("frozen dtype"),
    };
    let count = d["shape"]
        .as_array()
        .unwrap()
        .iter()
        .try_fold(1usize, |n, v| n.checked_mul(number(v)))
        .unwrap();
    assert_eq!(count.checked_mul(item).unwrap(), len);
    raw
}
pub(super) fn words(d: &Value) -> Vec<u16> {
    assert_eq!(d["dtype"], "u16le");
    bytes(d)
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
pub(super) fn values(d: &Value) -> Vec<f64> {
    match d["dtype"].as_str().unwrap() {
        "u16le" => words(d)
            .iter()
            .map(|&w| f64::from(f16::from_bits(w).to_f32()))
            .collect(),
        "f32le" => bytes(d)
            .chunks_exact(4)
            .map(|b| f64::from(f32::from_le_bytes(b.try_into().unwrap())))
            .collect(),
        _ => panic!("frozen dtype"),
    }
}
pub(super) fn hash16(v: &[u16]) -> String {
    let mut h = Sha256::new();
    for w in v {
        h.update(w.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
pub(super) fn hash32(v: &[u32]) -> String {
    let mut h = Sha256::new();
    for w in v {
        h.update(w.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn hex(raw: impl Iterator<Item = u8>, bytes: usize) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes * 2);
    for b in raw {
        s.push(DIGITS[(b >> 4) as usize] as char);
        s.push(DIGITS[(b & 15) as usize] as char);
    }
    s
}
pub(super) fn hex16(v: &[u16]) -> String {
    hex(v.iter().flat_map(|w| w.to_le_bytes()), v.len() * 2)
}
pub(super) fn hex32(v: &[u32]) -> String {
    hex(v.iter().flat_map(|w| w.to_le_bytes()), v.len() * 4)
}

#[test]
fn authenticate_full_fixture_extents_policy_and_envelopes() {
    let m = fixture();
    let mut extents = Vec::new();
    let mut values_count = 0;
    for c in m["cases"].as_array().unwrap() {
        assert_eq!(c["all_active_queries"], true);
        assert_eq!(
            number(&c["selected_query_rows"][number(&c["tokens"]) - 1]),
            number(&c["tokens"]) - 1
        );
        for d in c["inputs"].as_object().unwrap().values() {
            bytes(d);
            extents.push((number(&d["offset"]), number(&d["bytes"])));
        }
        for stage in ["scores", "probabilities", "context"] {
            let d = &c["arrays"][stage];
            let a = values(&d["main"]);
            let b = values(&d["wide"]);
            assert_eq!(a.len(), b.len());
            assert!(a.iter().chain(&b).all(|v| v.is_finite()));
            let envelope = a
                .iter()
                .zip(&b)
                .map(|(a, b)| (a - b).abs())
                .fold(0., f64::max);
            assert_eq!(d["envelope"].as_f64(), Some(envelope));
            assert_eq!(d["gate"]["atol"].as_f64(), Some((2. * envelope).max(1e-4)));
            assert_eq!(d["gate"]["rtol"].as_f64(), Some(1e-3));
            values_count += a.len();
            for profile in ["main", "wide"] {
                let v = &d[profile];
                extents.push((number(&v["offset"]), number(&v["bytes"])));
            }
        }
    }
    extents.sort_unstable();
    assert_eq!(extents.len(), 143);
    let mut end = 0;
    for (start, len) in extents {
        assert_eq!(start, end);
        end += len;
    }
    assert_eq!(end, 22405366);
    assert_eq!(values_count, 753432);
}
