//! Explicit real-artifact capture only; CPU reader makes numerical decisions.
//! No source arithmetic, tokenizer, quality admission or production factory.
use super::{
    real_capture::{self as output, Class, Tag, Writer},
    *,
};
use crate::mlx::lfm2::payload::{self, ContentIdentity, LoadOptions, ModelInput};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Write,
    path::Path,
};

const LIMIT: usize = 6 * 1024 * 1024 * 1024;
const EXTERNAL: usize = 512 * 1024 * 1024;
const PROTOCOL: &[u8] = include_bytes!("../../../tests/fixtures/lfm2-real-source-v1/protocol.json");
const PROTOCOL_SHA: &str = "49005e6e0342adc1e05177063623d149c2319b1966b52d0174ea00d46e6ce59f";
const INVENTORY_SHA: &str = "ecf0ca26c478045d4354dd74f47baddb9d54b4821fa9a3231a7e36c981b9b547";
const GUARD_RESERVE: usize = 1024 * 1024;
const RAW_UPPER: usize = 155_483_328;
// Array kinds are fixed dictionary IDs, never generated from target values.
const STAGE: u32 = 0;
const U: u32 = 1;
const KROT: u32 = 2;
const VPROJ: u32 = 3;
const HISTORY_PRE: u32 = 4;
const HISTORY_POST: u32 = 5;
const KV_INITIAL: u32 = 6;
const KV_ACTIVE: u32 = 7;
const HEAD: u32 = 8;
const GUARDS: u32 = 9;
const PUBLISHED_BEFORE_FAULT: u32 = 10;
const HISTORY_STACK: u32 = 11;
mod observe;
mod resources;
mod workload;

fn io(error: std::io::Error) -> Error {
    Error::Native(format!("capture IO: {error}"))
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn digest(s: &str) -> [u8; 32] {
    assert_eq!(s.len(), 64);
    std::array::from_fn(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
}
fn tag(session: u32, attempt: u32, kind: u32, index: u32, count: usize) -> Tag {
    Tag {
        id: [session, attempt, kind, index],
        shape: [count, 1, 1],
        rank: 1,
        query: [0, 0],
        keys: [0, 0],
    }
}
fn memory(m: Memory) -> Value {
    json!({"packed_weights":m.packed_weights,"direct_norms":m.direct_norms,
        "convolution_coefficients":m.convolution_coefficients,"zero_bias":m.zero_bias,
        "cache":m.cache,"convolution_history":m.convolution_history,"scratch":m.scratch,
        "logits":m.logits,"tokens":m.tokens,"diagnostic_gpu":m.diagnostic_gpu,
        "diagnostic_readback_reserved":m.diagnostic_readback_reserved,"metadata":m.metadata,
        "shared_framework_allowance":m.shared_framework_allowance,"external_resident":m.external_resident,
        "admitted":m.admitted,"cold_peak":m.cold_peak})
}
fn options(diagnostics: bool, all_logits: bool) -> Options {
    Options {
        memory_limit_bytes: LIMIT,
        external_resident_bytes: EXTERNAL,
        all_logits,
        diagnostics,
    }
}
fn token_list(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
        .collect()
}

struct Capture {
    writer: Writer,
    session: u32,
    attempt: u32,
    attempts: usize,
    rows: usize,
    guard_bytes: usize,
    selected_groups: usize,
    primary_rows: usize,
    producer_rows: usize,
    run: u32,
    registry: BTreeMap<usize, (usize, u64)>,
    baseline: BTreeMap<Vec<u32>, String>,
    physical_baseline: BTreeMap<Vec<u32>, (Vec<String>, Vec<String>)>,
    row_hashes: usize,
    history_stacks: usize,
    measured: usize,
    guard_records: usize,
    primary_chunks: usize,
    admissions: [Option<usize>; 3],
    trajectory_failures: usize,
    codec_failures: usize,
    safely_terminal: bool,
}
#[derive(Clone)]
struct State {
    boundary: usize,
    metadata: Value,
    old_kv: Vec<String>,
    old_history: Vec<String>,
    published: String,
    published_rows: usize,
    tokens: Vec<u32>,
    physical: BTreeSet<usize>,
}

fn hash32_iter(words: impl IntoIterator<Item = u32>) -> String {
    let mut hash = Sha256::new();
    let mut block = [0u8; 65536];
    let mut used = 0;
    for word in words {
        block[used..used + 4].copy_from_slice(&word.to_le_bytes());
        used += 4;
        if used == block.len() {
            hash.update(block);
            used = 0;
        }
    }
    hash.update(&block[..used]);
    format!("{:x}", hash.finalize())
}
fn hash16_iter(words: impl IntoIterator<Item = u16>) -> String {
    let mut hash = Sha256::new();
    let mut block = [0u8; 65536];
    let mut used = 0;
    for word in words {
        block[used..used + 2].copy_from_slice(&word.to_le_bytes());
        used += 2;
        if used == block.len() {
            hash.update(block);
            used = 0;
        }
    }
    hash.update(&block[..used]);
    format!("{:x}", hash.finalize())
}
fn greedy(words: &[u32]) -> Result<u32> {
    require(
        words.len() == 65536 && words.iter().all(|&x| f32::from_bits(x).is_finite()),
        "bounded finite full-vocabulary argmax",
    )?;
    let mut best = 0;
    for i in 1..words.len() {
        if f32::from_bits(words[i]) > f32::from_bits(words[best]) {
            best = i;
        }
    }
    Ok(best as u32)
}
fn require(ok: bool, message: &'static str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Error::Options(message))
    }
}

#[test]
#[ignore = "explicit preregistered original LFM2 real capture; external conformance reader required"]
fn selected_real_lfm2_capture_preregistered_workload() -> Result<()> {
    let (control, control_sha) = resources::control()?;
    require(
        sha(PROTOCOL) == PROTOCOL_SHA,
        "embedded source protocol identity",
    )?;
    let protocol: Value =
        serde_json::from_slice(PROTOCOL).map_err(|e| Error::Native(e.to_string()))?;
    let writer = Writer::create(Path::new(&control.output_directory)).map_err(io)?;
    let mut capture = Capture {
        writer,
        session: 0,
        attempt: 0,
        attempts: 0,
        rows: 0,
        guard_bytes: 0,
        selected_groups: 0,
        primary_rows: 0,
        producer_rows: 0,
        run: 0,
        registry: BTreeMap::new(),
        baseline: BTreeMap::new(),
        physical_baseline: BTreeMap::new(),
        row_hashes: 0,
        history_stacks: 0,
        measured: 0,
        guard_records: 0,
        primary_chunks: 0,
        admissions: [None; 3],
        trajectory_failures: 0,
        codec_failures: 0,
        safely_terminal: true,
    };
    capture.event("start",json!({"format":"toklet-lfm2-real-capture-v1","protocol_sha256":PROTOCOL_SHA,
        "inventory_sha256":INVENTORY_SHA,"control_sha256":control_sha,"source_manifest_sha256":control.source_manifest_sha256,
        "source_arrays_sha256":control.source_arrays_sha256,"gates_sha256":control.gates_sha256,
        "owner_manifest":{"path":control.owner_manifest.path,"bytes":control.owner_manifest.bytes,"sha256":control.owner_manifest.sha256},
        "binary":{"path":control.binary.path,"bytes":control.binary.bytes,"sha256":control.binary.sha256},
        "authorization":control.authorization,"stages":protocol["cases"][0]["stages"],
        "limits":{"raw":output::RAW_CAP,"raw_inventory":RAW_UPPER,"journal":output::JOURNAL_CAP,"numeric_capture_reserve":64*1024*1024,
            "external_resident":EXTERNAL,"process":LIMIT,"attempts":104,"attempted_rows":367,"arrays":14672},
        "profile":"native-half-direct-gain-exact-original-F32-affine-center",
        "scope":"64-position trajectory at512 capacity; no source/quality/app admission or numerical decision in capture"}),Class::Header)?;
    let model = resources::load(&control, &mut capture)?;
    let before = resources::model(&mut capture, &model, "model_before")?;
    let result = workload::execute(&mut capture, &model, &protocol);
    let after = if capture.safely_terminal {
        Some(resources::model(&mut capture, &model, "model_after")?)
    } else {
        None
    };
    capture.event(
        "assertions",
        json!({"name":"immutable_model_before_after","passed":after.as_ref()==Some(&before),
        "sampled_at_each_boundary":false}),
        Class::Other,
    )?;
    if let Err(error) = &result {
        capture.event(
            "capture_error",
            json!({"error":format!("{error:?}"),"numerical_qualification":false}),
            Class::Other,
        )?;
    }
    let prefix = capture.writer.prefix().map_err(io)?;
    capture.event("end",json!({"prefix":prefix,"attempts":capture.attempts,"attempted_rows":capture.rows,
        "primary_chunks":capture.primary_chunks,"primary_rows":capture.primary_rows,"selected_groups":capture.selected_groups,
        "producer_rows":capture.producer_rows,"guard_bytes":capture.guard_bytes}),Class::Other)?;
    let summary = capture.writer.finish().map_err(io)?;
    println!(
        "LFM2_REAL_CAPTURE_SUMMARY {}",
        serde_json::to_string(&summary).map_err(|e| Error::Native(e.to_string()))?
    );
    std::io::stdout().flush().map_err(io)?;
    require(
        summary.raw_bytes <= RAW_UPPER && summary.arrays <= 14672,
        "capture frozen output inventory",
    )?;
    require(
        after.as_ref() == Some(&before),
        "immutable model changed or unsafe endpoint",
    )?;
    result
}

#[test]
fn real_capture_protocol_counts_and_complete_observation_bounds() {
    assert_eq!(sha(PROTOCOL), PROTOCOL_SHA);
    let p: Value = serde_json::from_slice(PROTOCOL).unwrap();
    let (mut schedules, mut chunks, mut rows, mut queries, mut ends) = (0, 0, 0, 0, 0);
    for case in p["cases"].as_array().unwrap() {
        assert_eq!(case["stages"].as_array().unwrap().len(), 293);
        let tokens = token_list(&case["tokens"]);
        assert_eq!(tokens[0], 1);
        assert!(tokens.iter().all(|&v| v < 64400));
        for part in case["native_partitions"].as_array().unwrap() {
            schedules += 1;
            queries += case["selected_rows"].as_array().unwrap().len();
            let mut at = 0;
            for n in part.as_array().unwrap() {
                let n = n.as_u64().unwrap() as usize;
                assert!((1..=32).contains(&n));
                chunks += 1;
                rows += n;
                at += n;
                ends += at;
            }
            assert_eq!(at, tokens.len());
        }
    }
    assert_eq!(
        (schedules, chunks, rows, queries, ends),
        (8, 78, 231, 23, 2389)
    );
    assert_eq!(queries * 293, 6739);
    assert_eq!(78 + 4 + 8 + 12 + 2, 104);
    assert_eq!(231 + 11 + 22 + 39 + 64, 367);
    assert_eq!(2389 + 38 + 76 + 51 + 96, 2650);
    assert_eq!(
        6739 + 102 * 22 + 104 * 40 + 104 * 12 + 5 * 24 + 14 + 3 + 126 + 18,
        14672
    );
    assert_eq!(
        30_580_480
            + 303 * 32768
            + 104 * 2 * 82200
            + 2650 * 12288
            + 5 * 6685008
            + 16 * 65536 * 4
            + 3 * 32 * 65536 * 4
            + 1048576
            + 18 * 82200,
        RAW_UPPER
    );
}
#[test]
fn real_capture_guard_owners_and_budget_delta_are_explicit() {
    assert_eq!(24 + 20 + 14 + 5 + 1 + 293, 357);
    assert_eq!(280 * 16 + 13 * 28, 4844);
    assert_eq!(4844 + 14 * 14 + 5 * 28 + 24 * 14 + 20 * 14 + 28, 5824);
    assert_eq!(5824 - 4844, 980);
    let maximum_guards = 126 * 5824 + 3 * 28;
    assert!(maximum_guards <= GUARD_RESERVE);
    assert_eq!(5 + 104 + 13 + 4, 126);
    assert_eq!(126 + 5 + 8 + 1, 140);
    let boundary_events = 126 + 5 + 8 + 1;
    assert!(boundary_events < 160);
    assert_eq!(5 + 7 + 1 + 2 + 3, 18);
    assert_eq!(2 * (32 - 1) * 65536 * 4, 16_252_928);
}
#[test]
fn real_capture_control_contract_and_unknown_fields_fail_closed() {
    let valid = json!({"format":"toklet-lfm2-real-capture-control-v1","artifact_directory":"/synthetic/original",
        "output_directory":"/synthetic/output","binary":{"path":"/synthetic/binary","bytes":1,"sha256":"00"},
        "owner_manifest":{"path":"/synthetic/owner","bytes":1,"sha256":"00"},"protocol_sha256":PROTOCOL_SHA,"inventory_sha256":INVENTORY_SHA,
        "gates_sha256":"7b1879ee54fd5a421d0b5f17f2d9192d291742bdc078ef2f5e90885eb293fbdf",
        "source_manifest_sha256":"2654f475aedb37024b66f6c9a9988369d8ea8138a34e078505adf6fe16635133",
        "source_arrays_sha256":"9db9e7cd82a1a15ef60f75b2cd91b04f20d30553bb64cb55be72d2c201d0837a","authorization":{"synthetic_only":true}});
    let c: resources::Control = serde_json::from_value(valid.clone()).unwrap();
    resources::contract(&c).unwrap();
    // This tests only semantic metadata, not a fake launch: actual control()
    // additionally consumes/authenticates its bytes, executable and owner files.
    for field in [
        "format",
        "protocol_sha256",
        "inventory_sha256",
        "gates_sha256",
        "source_manifest_sha256",
        "source_arrays_sha256",
    ] {
        let mut v = valid.clone();
        v[field] = json!("wrong");
        let c = serde_json::from_value(v).unwrap();
        assert!(resources::contract(&c).is_err());
    }
    let mut v = valid;
    v["relax_gate"] = json!(true);
    assert!(serde_json::from_value::<resources::Control>(v).is_err());
}
#[test]
fn real_capture_full_argmax_stops_mapping_and_zero_hash_scope() {
    let mut words = vec![0f32.to_bits(); 65536];
    words[7] = 1f32.to_bits();
    assert_eq!(greedy(&words).unwrap(), 7);
    words[2] = 1f32.to_bits();
    assert_eq!(greedy(&words).unwrap(), 2);
    words[64400] = 2f32.to_bits();
    assert_eq!(greedy(&words).unwrap(), 64400);
    words[65535] = 3f32.to_bits();
    assert_eq!(greedy(&words).unwrap(), 65535); // never masked to mapped vocabulary
    words[0] = f32::NAN.to_bits();
    assert!(greedy(&words).is_err());
    assert!(greedy(&words[..65535]).is_err());
    assert_ne!(hash32_iter([0]), hash32_iter([0x80000000]));
    let canon = |x: u32| if x & 0x7fff_ffff == 0 { 0 } else { x };
    assert_eq!(hash32_iter([canon(0)]), hash32_iter([canon(0x80000000)]));
}
