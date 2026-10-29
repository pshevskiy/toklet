//! Explicit acquisition/storage/upload controls only. No Session, forward,
//! tokenizer, source numerical fixture, logits or command submission.
use super::*;
use crate::mlx::gemma3::{
    Storage,
    payload::{self, ContentIdentity, LoadOptions, PayloadStore},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::File,
    io::{Read, Write},
    path::PathBuf,
};

const LIMIT: usize = 6 * 1024 * 1024 * 1024;
const CONTAINER: u64 = 150_939_130;
const CONFIG: &[u8] = include_bytes!("../../../tests/fixtures/gemma3-270m-metadata-v1/config.json");
const INDEX: &[u8] = include_bytes!("../../../tests/fixtures/gemma3-270m-metadata-v1/index.json");
const HEADER: &[u8] = include_bytes!("../../../tests/fixtures/gemma3-270m-metadata-v1/header.json");
const CONTAINER_SHA: &str = "bf8b442cff5dd7c9ac6b5695310d5b66a4d0a4c3780f5c774fb90cd0c352bd9d";
fn emit(value: Value) {
    let mut out = std::io::stdout().lock();
    writeln!(out, "GEMMA3_LOADER {}", value).unwrap();
    out.flush().unwrap();
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn digest(text: &str) -> [u8; 32] {
    assert_eq!(text.len(), 64);
    std::array::from_fn(|i| u8::from_str_radix(&text[2 * i..2 * i + 2], 16).unwrap())
}
fn identity() -> ContentIdentity {
    ContentIdentity {
        config_sha256: digest("8fa17cb1689800679eeb5b19ab2d2ef5d425e16cf9f36f59925f130ed976ea38"),
        index_sha256: digest("3ef9f6408eebccbe14976dd8db24d75d848f01c4430aa8b5924d97782a0930fb"),
        header_sha256: digest("994efdf1aa3b4f6bc39d7935b1286380c6f9f49e926e02405fed6ed72f6e578f"),
        container_sha256: digest(CONTAINER_SHA),
        container_bytes: CONTAINER,
    }
}
fn load() -> PayloadStore {
    let directory =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/MLX-Gemma-3-270m-it-4bit-ff1143e");
    let read_metadata = |name: &str, expected: &[u8]| {
        let mut bytes = Vec::new();
        File::open(directory.join(name))
            .unwrap()
            .take((expected.len() + 1) as u64)
            .read_to_end(&mut bytes)
            .unwrap();
        emit(
            json!({"kind":"metadata","name":name,"bytes":bytes.len(),"sha256":hash(&bytes),"expected_sha256":hash(expected)}),
        );
        assert_eq!(bytes, expected);
        bytes
    };
    let config = read_metadata("config.json", CONFIG);
    let index = read_metadata("model.safetensors.index.json", INDEX);
    assert_eq!(
        <[u8; 32]>::from(Sha256::digest(HEADER)),
        identity().header_sha256
    );
    let file = File::open(directory.join("model.safetensors")).unwrap();
    let length = file.metadata().unwrap().len();
    emit(
        json!({"kind":"container_start","bytes":length,"expected_sha256":CONTAINER_SHA,"numerical_execution":false}),
    );
    assert_eq!(length, CONTAINER);
    let result = payload::load(
        &config,
        &index,
        file,
        LoadOptions {
            identity: identity(),
            memory_limit_bytes: LIMIT,
        },
    );
    emit(
        json!({"kind":"load_result","ok":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string)}),
    );
    result.unwrap()
}
fn bytes_record(raw: &[u8], padded: usize) -> Value {
    let mut sha = Sha256::new();
    sha.update(raw);
    for _ in raw.len()..padded {
        sha.update([0]);
    }
    json!({"bytes":padded,"sha256":format!("{:x}", sha.finalize()),"original_bytes":raw.len(),"original_sha256":hash(raw)})
}
fn source_records(store: &PayloadStore) -> Vec<Value> {
    let mut records = Vec::new();
    for t in store.plan().logical_tensors() {
        let value = match t.storage {
            Storage::Affine(_) => {
                let matrix = store.matrix(&t.name).unwrap();
                let l = matrix.layout();
                let (w, s, b) = matrix.original_tensors();
                json!({"kind":"matrix","name":t.name,"rows":l.rows(),"columns":l.columns(),"components":[bytes_record(w,w.len()),bytes_record(s,s.len().div_ceil(4)*4),bytes_record(b,b.len().div_ceil(4)*4)]})
            }
            Storage::Bf16Vector => {
                let gamma = store.raw_gamma(&t.name).unwrap();
                let mut raw = Sha256::new();
                let mut active = Sha256::new();
                let mut full = Sha256::new();
                for g in &gamma {
                    raw.update(g.to_le_bytes());
                    let word = (1.0f32 + g.to_f32()).to_bits().to_le_bytes();
                    active.update(word);
                    full.update(word);
                }
                for _ in 0..GUARD {
                    full.update(0x5a170031u32.to_le_bytes());
                }
                json!({"kind":"norm","name":t.name,"words":gamma.len(),"raw_bytes":gamma.len()*2,"raw_sha256":format!("{:x}", raw.finalize()),"active_sha256":format!("{:x}", active.finalize()),"bytes":(gamma.len()+GUARD)*4,"sha256":format!("{:x}", full.finalize())})
            }
        };
        emit(json!({"kind":"source_tensor","record":value}));
        records.push(value);
    }
    records
}
fn report(store: &PayloadStore) -> Value {
    let r = store.report();
    let id = store.content_identity();
    json!({"payload_bytes":r.payload_bytes,"payload_capacity_bytes":r.payload_capacity_bytes,"packed_matrices":r.packed_matrices,"raw_bf16_norms":r.raw_bf16_norms,"metadata_reservation_bytes":r.metadata_reservation_bytes,"admitted_load_bytes":r.admitted_load_bytes,"container_bytes":id.container_bytes,"container_sha256":id.container_sha256.iter().map(|b|format!("{b:02x}")).collect::<String>(),"identity_debug":format!("{id:?}"),"declared_positions":store.plan().declared_total_positions(),"declared_window":store.plan().declared_sliding_window()})
}
fn check_store(store: &PayloadStore, records: &[Value]) {
    assert_eq!(store.content_identity(), &identity());
    let r = store.report();
    assert_eq!(r.payload_bytes, 150_885_632);
    assert_eq!(r.packed_matrices, 127);
    assert_eq!(r.raw_bf16_norms, 109);
    assert_eq!(records.len(), 236);
    assert_eq!(store.plan().physical_tensors().len(), 490);
    assert_eq!(
        r.admitted_load_bytes,
        r.payload_capacity_bytes + 64 * 1024 * 1024 + 64 * 1024
    );
    assert!(r.admitted_load_bytes <= LIMIT);
    let matrices = records.iter().filter(|x| x["kind"] == "matrix").count();
    let words: usize = records
        .iter()
        .filter(|x| x["kind"] == "norm")
        .map(|x| x["words"].as_u64().unwrap() as usize)
        .sum();
    assert_eq!((matrices, words), (127, 55_936));
    let original: usize = records
        .iter()
        .map(|r| {
            if r["kind"] == "matrix" {
                r["components"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|c| c["original_bytes"].as_u64().unwrap() as usize)
                    .sum()
            } else {
                r["raw_bytes"].as_u64().unwrap() as usize
            }
        })
        .sum();
    assert_eq!(original, r.payload_bytes);
    assert_eq!(store.plan().declared_total_positions(), 32768);
    assert_eq!(store.plan().declared_sliding_window(), 512);
}
#[test]
#[ignore = "explicit acquired original-file storage check; no numerical inference"]
fn original_payload_file_identity_and_capacity_mapping() {
    let store = load();
    let records = source_records(&store);
    emit(json!({"kind":"store_report","report":report(&store)}));
    let mut mappings = Vec::new();
    for capacity in [1, 12, 511, 512, 513, 544] {
        let result = store.prepare_raw_gamma(capacity, 0, LIMIT);
        emit(
            json!({"kind":"prepare_result","capacity":capacity,"ok":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string)}),
        );
        let input = result.unwrap();
        let item = json!({"capacity":capacity,"max_positions":input.config.max_positions,"local_window":input.local_window,"layers":input.weights.layers.len(),"specs":input.specs.len(),"source_resident_bytes":input.source_resident_bytes});
        emit(json!({"kind":"mapping","record":item}));
        mappings.push(item);
    }
    for capacity in [0, 545, usize::MAX] {
        let result = store.prepare_raw_gamma(capacity, 0, LIMIT);
        emit(
            json!({"kind":"rejected_capacity","capacity":capacity,"error":result.as_ref().err().map(ToString::to_string)}),
        );
        assert!(result.is_err());
    }
    check_store(&store, &records);
    for r in mappings {
        let c = r["capacity"].as_u64().unwrap();
        assert_eq!(r["max_positions"], c);
        assert_eq!(r["local_window"], 512.min(c));
        assert_eq!(r["layers"], 18);
        assert_eq!(r["specs"], 18);
        assert!(
            r["source_resident_bytes"].as_u64().unwrap()
                >= store.report().payload_capacity_bytes as u64 + 64 * 1024 * 1024 + 55_936 * 2
        );
    }
    emit(
        json!({"kind":"done","scope":"original_storage_only","matrices":127,"norms":109,"physical_tensors":490,"model_created":false,"session_created":false,"forward_calls":0}),
    );
}
fn half_record(buffer: &Buffer<u16>) -> Value {
    let mut sha = Sha256::new();
    let mut bytes = [0u8; 4096];
    buffer.with_slice(|words| {
        for chunk in words.chunks(2048) {
            for (dst, word) in bytes.chunks_exact_mut(2).zip(chunk) {
                dst.copy_from_slice(&word.to_le_bytes());
            }
            sha.update(&bytes[..chunk.len() * 2]);
        }
    });
    json!({"bytes":buffer.raw.length(),"address":buffer.raw.contents().as_ptr() as usize,"sha256":format!("{:x}", sha.finalize())})
}
fn word_record(words: &[u32], address: usize) -> Value {
    let mut sha = Sha256::new();
    let mut bytes = [0u8; 4096];
    for chunk in words.chunks(1024) {
        for (dst, word) in bytes.chunks_exact_mut(4).zip(chunk) {
            dst.copy_from_slice(&word.to_le_bytes());
        }
        sha.update(&bytes[..chunk.len() * 4]);
    }
    json!({"bytes":words.len()*4,"address":address,"sha256":format!("{:x}", sha.finalize())})
}
fn model_records(model: &Model, label: &str) -> Vec<Value> {
    // No command has referenced these resources: upload is synchronous mapped
    // CPU copying. Owned hashes/addresses escape, never GPU handles or slices.
    let mut matrices = vec![("model.embed_tokens.weight".to_string(), &model.embedding)];
    let mut norms = vec![("model.norm.weight".to_string(), &model.final_norm)];
    for (i, l) in model.layers.iter().enumerate() {
        for (name, tensor) in [
            ("self_attn.q_proj", &l.q),
            ("self_attn.k_proj", &l.k),
            ("self_attn.v_proj", &l.v),
            ("self_attn.o_proj", &l.o),
            ("mlp.gate_proj", &l.gate),
            ("mlp.up_proj", &l.up),
            ("mlp.down_proj", &l.down),
        ] {
            matrices.push((format!("model.layers.{i}.{name}.weight"), tensor));
        }
        for (name, buffer) in [
            ("input_layernorm", &l.norms.input),
            ("post_attention_layernorm", &l.norms.post_attention),
            ("pre_feedforward_layernorm", &l.norms.pre_ff),
            ("post_feedforward_layernorm", &l.norms.post_ff),
            ("self_attn.q_norm", &l.norms.q),
            ("self_attn.k_norm", &l.norms.k),
        ] {
            norms.push((format!("model.layers.{i}.{name}.weight"), buffer));
        }
    }
    let mut out = Vec::new();
    for (name, t) in matrices {
        let metadata = match &t.storage {
            TensorStorage::Mlx(p) => p.test_metadata(),
            _ => panic!("packed original storage required"),
        };
        let value = json!({"kind":"matrix","name":name,"rows":t.layout.rows,"columns":t.layout.columns,"components":[half_record(&t.buffer),word_record(&metadata[0].1,metadata[0].0),word_record(&metadata[1].1,metadata[1].0)]});
        emit(json!({"kind":"resident_tensor","label":label,"record":value}));
        out.push(value);
    }
    for (name, b) in norms {
        let mut value = b.with_slice(|w| word_record(w, b.raw.contents().as_ptr() as usize));
        value["kind"] = json!("norm");
        value["name"] = json!(name);
        emit(json!({"kind":"resident_tensor","label":label,"record":value}));
        out.push(value);
    }
    let mut zero = half_record(&model.zero_bias);
    zero["kind"] = json!("zero_bias");
    zero["name"] = json!("zero_bias");
    emit(json!({"kind":"resident_tensor","label":label,"record":zero}));
    out.push(zero);
    out
}
fn check_resident(actual: &[Value], source: &[Value]) -> (usize, usize, usize) {
    let mut ids = BTreeSet::new();
    let mut packed = 0;
    let mut norms = 0;
    let mut zero = 0;
    for a in actual {
        if a["kind"] == "matrix" {
            let e = source.iter().find(|s| s["name"] == a["name"]).unwrap();
            assert_eq!(a["rows"], e["rows"]);
            assert_eq!(a["columns"], e["columns"]);
            for (x, y) in a["components"]
                .as_array()
                .unwrap()
                .iter()
                .zip(e["components"].as_array().unwrap())
            {
                assert!(ids.insert(x["address"].as_u64().unwrap()));
                assert_eq!(x["bytes"], y["bytes"]);
                assert_eq!(x["sha256"], y["sha256"]);
                packed += x["bytes"].as_u64().unwrap() as usize;
            }
        } else {
            assert!(ids.insert(a["address"].as_u64().unwrap()));
            if a["kind"] == "norm" {
                let e = source.iter().find(|s| s["name"] == a["name"]).unwrap();
                assert_eq!(a["bytes"], e["bytes"]);
                assert_eq!(a["sha256"], e["sha256"]);
                norms += a["bytes"].as_u64().unwrap() as usize;
            } else {
                zero = a["bytes"].as_u64().unwrap() as usize;
                let mut sha = Sha256::new();
                for _ in 0..zero {
                    sha.update([0]);
                }
                assert_eq!(a["sha256"], format!("{:x}", sha.finalize()));
            }
        }
    }
    assert_eq!(actual.len(), 237);
    assert_eq!(ids.len(), 491);
    (packed, norms, zero)
}
#[test]
#[ignore = "explicit acquired original-file resident upload; Metal required, no inference"]
fn original_payload_upload_resident_identity_after_source_drop() {
    let store = load();
    let source = source_records(&store);
    let sr = report(&store);
    emit(json!({"kind":"store_report","report":sr}));
    let input = store.prepare_raw_gamma(544, 0, LIMIT).unwrap();
    let overlap = input.source_resident_bytes;
    drop(input);
    let load_peak = store.report().admitted_load_bytes;
    let result = Model::from_payload_raw_gamma(&store, 544, 0, LIMIT);
    emit(
        json!({"kind":"upload_result","ok":result.is_ok(),"error":result.as_ref().err().map(ToString::to_string),"source_resident_bytes":overlap,"external_resident_bytes":0,"memory_limit":LIMIT,"model_numerical_execution":false}),
    );
    let model = result.unwrap();
    let before = model_records(&model, "before_source_drop");
    let m = model.memory();
    emit(
        json!({"kind":"model_report","device":model.device_name(),"memory_debug":format!("{m:?}"),"packed_weights":m.packed_weights,"effective_norms":m.effective_norms,"zero_bias":m.zero_bias,"metadata":m.metadata,"framework_allowance":m.shared_framework_allowance,"admitted":m.admitted,"cold_peak":m.cold_peak,"requested_capacity":model.config.max_positions,"local_window":model.local_window,"embedding_scale":model.embedding_scale,"cache_plan_bytes_not_allocated":model.cache_bytes,"source_dropped":false}),
    );
    check_store(&store, &source);
    drop(store);
    emit(
        json!({"kind":"source_dropped","cpu_store_retained":false,"observation":"owned Model has no store lifetime; subsequent readback accesses model resources only"}),
    );
    let after = model_records(&model, "after_source_drop");
    let sizes = check_resident(&before, &source);
    assert_eq!(before, after);
    assert_eq!(sizes, (150_773_760, 226_796, 524_302));
    assert_eq!((m.packed_weights, m.effective_norms, m.zero_bias), sizes);
    assert_eq!(
        m.admitted,
        sizes.0 + sizes.1 + sizes.2 + m.metadata + m.shared_framework_allowance
    );
    assert_eq!(m.shared_framework_allowance, 64 * 1024 * 1024);
    assert_eq!(m.cold_peak, (m.admitted + overlap).max(load_peak));
    assert!(m.cold_peak <= LIMIT);
    assert_eq!(model.config.max_positions, 544);
    assert_eq!(model.local_window, 512);
    assert_eq!(model.layers.len(), 18);
    assert_eq!(model.embedding_scale, 25.25);
    for (i, l) in model.layers.iter().enumerate() {
        assert_eq!(
            l.spec.kind,
            if i % 6 == 5 {
                KvKind::Full
            } else {
                KvKind::Sliding
            }
        );
        assert_eq!(
            l.spec.theta,
            if i % 6 == 5 { 1_000_000.0 } else { 10_000.0 }
        );
    }
    emit(
        json!({"kind":"done","scope":"original_resident_upload_only","matrices":127,"norms":109,"effective_norm_words":55936,"unique_resident_buffers":491,"model_created":true,"session_created":false,"forward_calls":0,"gpu_commands_submitted":0,"model_or_text_qualified":false}),
    );
}
