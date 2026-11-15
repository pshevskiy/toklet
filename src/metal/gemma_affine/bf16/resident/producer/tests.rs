use super::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
};
const LIMIT: usize = 6 * 1024 * 1024 * 1024;
const HELPER_RESERVE: usize = 512 * 1024 * 1024;
const DIAGNOSTIC_RESERVE: usize = 64 * 1024 * 1024;
fn read(path: &Path, n: usize) -> Vec<u8> {
    assert_eq!(std::fs::metadata(path).unwrap().len(), n as u64);
    let mut v = Vec::with_capacity(n);
    std::fs::File::open(path)
        .unwrap()
        .take(n as u64 + 1)
        .read_to_end(&mut v)
        .unwrap();
    assert_eq!(v.len(), n);
    v
}
fn pinned(path: &Path, n: usize, sha: &str) -> Vec<u8> {
    let raw = read(path, n);
    assert_eq!(format!("{:x}", Sha256::digest(&raw)), sha);
    raw
}
struct Fixture {
    protocol: Value,
    states: Vec<u8>,
}
impl Fixture {
    fn load() -> Self {
        let root = PathBuf::from(std::env::var_os("TOKLET_GEMMA_PRODUCER_FIXTURE").unwrap());
        let protocol = pinned(
            &root.join("protocol.json"),
            301270,
            "53de539d90152909feeb08e0ddf54700cbf6b992bb9dabaa9467d2eb7bc06fc2",
        );
        Self {
            protocol: serde_json::from_slice(&protocol).unwrap(),
            states: pinned(
                &root.join("states.bin"),
                2711552,
                "29cdaff7bbcb8b1b71367594c01308932b4f1b78fcf57b9003a063d3bd1875ea",
            ),
        }
    }
    fn view(&self, d: &Value) -> Vec<f32> {
        assert_eq!(d["dtype"], "<f4");
        let shape: Vec<_> = d["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_u64().unwrap() as usize)
            .collect();
        let strides: Vec<_> = d["byte_strides"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_u64().unwrap() as usize)
            .collect();
        let n = shape.iter().product();
        let start = d["offset"].as_u64().unwrap() as usize;
        (0..n)
            .map(|mut i| {
                let mut off = start;
                for axis in (0..shape.len()).rev() {
                    off += (i % shape[axis]) * strides[axis];
                    i /= shape[axis];
                }
                f32::from_le_bytes(self.states[off..off + 4].try_into().unwrap())
            })
            .collect()
    }
    fn prefix(&self, case: &Value) -> Vec<u32> {
        let prefill = self.protocol["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| {
                v["sequence"] == case["sequence"]
                    && v["layer"] == case["layer"]
                    && v["mode"] == "prefill"
            })
            .unwrap();
        prefill["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect()
    }
    fn cache(&self, block: &ProducerBlock<'_>, case: &Value) -> ProducerKv {
        let mut cache =
            ProducerKv::new(block, ProducerKv::planned_bytes(block.layer).unwrap()).unwrap();
        let offset = case["query_start"].as_u64().unwrap() as usize;
        if offset != 0 {
            assert_eq!(offset, 3);
            let k = self.view(&case["cache"]["committed_key"]);
            let v = self.view(&case["cache"]["committed_value"]);
            assert_eq!(k.len(), offset * cache.d);
            assert_eq!(v.len(), k.len());
            assert!(k.iter().chain(&v).all(|x| x.is_finite()));
            cache.k.with_mut(|s| {
                for (out, value) in s.iter_mut().zip(&k) {
                    *out = value.to_bits();
                }
            });
            cache.v.with_mut(|s| {
                for (out, value) in s.iter_mut().zip(&v) {
                    *out = value.to_bits();
                }
            });
            cache.offset = offset;
            cache.generation = 1;
            cache.history[..offset].copy_from_slice(&self.prefix(case)[..offset]);
        }
        cache
    }
}
const CAPTURED: [(&str, State); 23] = [
    ("input", State::Hidden),
    ("ple_input", State::PleInput),
    ("input_layernorm", State::InputNorm),
    ("self_attn.q_proj", State::QRaw),
    ("self_attn.q_norm", State::QNorm),
    ("q_rotary", State::QRotary),
    ("self_attn.k_proj", State::KRaw),
    ("self_attn.v_proj", State::VRaw),
    ("self_attn.k_norm", State::KNorm),
    ("self_attn.v_norm", State::VProposal),
    ("k_rotary", State::KProposal),
    ("context", State::Context),
    ("self_attn.o_proj", State::AttentionRaw),
    ("post_attention_layernorm", State::AttentionNorm),
    ("pre_feedforward_layernorm", State::PreFf),
    ("mlp.gate_proj", State::FfGate),
    ("mlp.up_proj", State::FfUp),
    ("mlp.down_proj", State::FfDown),
    ("post_feedforward_layernorm", State::FfNorm),
    ("per_layer_input_gate", State::PleGate),
    ("per_layer_projection", State::PleProjection),
    ("post_per_layer_input_norm", State::PleNorm),
    ("output", State::Output),
];
fn check(actual: &[u32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&w, &r)) in actual.iter().zip(expected).enumerate() {
        let a = f32::from_bits(w);
        assert!(a.is_finite());
        assert!(
            (f64::from(a) - f64::from(r)).abs() <= 1e-5 + 1e-4 * f64::from(r).abs(),
            "original gate index{i}: {a} != {r}"
        );
    }
}
fn report(label: &str, value: &impl serde::Serialize) {
    let stdout = std::io::stdout();
    let mut out = BufWriter::with_capacity(16384, stdout.lock());
    write!(out, "{label} ").unwrap();
    serde_json::to_writer(&mut out, value).unwrap();
    writeln!(out).unwrap();
    out.flush().unwrap();
}
#[derive(serde::Serialize)]
struct FilePin {
    path: String,
    words: usize,
    bytes: usize,
    sha256: String,
}
fn capture(dir: &Path, name: &str, raw: &[u32]) -> FilePin {
    let path = dir.join(format!("{name}.u32le"));
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    let mut out = BufWriter::with_capacity(16384, file);
    let mut hash = Sha256::new();
    let mut block = [0u8; 4096];
    for words in raw.chunks(block.len() / 4) {
        for (w, b) in words.iter().zip(block.chunks_exact_mut(4)) {
            b.copy_from_slice(&w.to_le_bytes());
        }
        let bytes = &block[..words.len() * 4];
        out.write_all(bytes).unwrap();
        hash.update(bytes);
    }
    out.flush().unwrap();
    assert_eq!(
        out.get_ref().metadata().unwrap().len(),
        (raw.len() * 4) as u64
    );
    FilePin {
        path: path.to_str().unwrap().into(),
        words: raw.len(),
        bytes: raw.len() * 4,
        sha256: format!("{:x}", hash.finalize()),
    }
}
fn table(context: &Bf16Context) -> gelu::CertifiedTable {
    let load = |key, n| read(&PathBuf::from(std::env::var_os(key).unwrap()), n);
    let table = load("TOKLET_GELU_TABLE", 3317784);
    let cert = load("TOKLET_GELU_CERTIFICATE", 2309);
    let native = load("TOKLET_GELU_NATIVE_VERIFICATION", 635);
    let review = load("TOKLET_GELU_ARTIFACT_REVIEW", 13881);
    Pipelines::admit_table(
        context,
        &gelu::ApprovedBytes {
            table: &table,
            certificate: &cert,
            native_verification: &native,
            artifact_review: &review,
        },
        gelu::CertifiedTable::planned_owned_bytes(),
    )
    .unwrap()
}
fn source(context: &Bf16Context) -> Bf16Weights {
    let root = PathBuf::from(std::env::var_os("TOKLET_GEMMA_BF16_MODEL").unwrap());
    let config = std::fs::read(root.join("config.json")).unwrap();
    let reserved = HELPER_RESERVE
        + DIAGNOSTIC_RESERVE
        + PIPELINE_POLICY_BYTES
        + gelu::CertifiedTable::planned_owned_bytes()
        + [0, 4]
            .iter()
            .map(|&l| {
                ProducerBlock::planned_bytes(l).unwrap() + ProducerKv::planned_bytes(l).unwrap()
            })
            .sum::<usize>();
    Bf16Weights::load_text_only(
        context,
        std::fs::File::open(root.join("model.safetensors")).unwrap(),
        &config,
        LIMIT - reserved,
    )
    .unwrap()
}
fn weight_digest(source: &Bf16Weights, slots: ProducerSlots) -> String {
    let mut hash = Sha256::new();
    let mut raw = [0u8; 4096];
    for i in slots.matrices {
        let m = source.matrices[i].as_ref().unwrap();
        for tile in &m.tiles {
            tile.with_slice(|s| {
                for words in s.chunks(raw.len() / 4) {
                    for (w, b) in words.iter().zip(raw.chunks_exact_mut(4)) {
                        b.copy_from_slice(&w.to_le_bytes());
                    }
                    hash.update(&raw[..words.len() * 4]);
                }
            });
        }
    }
    format!("{:x}", hash.finalize())
}
#[test]
fn first_producer_geometry_memory_and_roles_without_gpu() {
    let plan = crate::mlx::gemma4::bf16::inspect(
        include_bytes!("../../../../../../tests/fixtures/gemma-unquantized-header-v1/config.json"),
        include_bytes!("../../../../../../tests/fixtures/gemma-unquantized-header-v1/header.json"),
        crate::mlx::gemma4::bf16::CONTAINER_BYTES,
        crate::mlx::gemma4::ModalityPolicy::TextOnly,
    )
    .unwrap();
    let b = Bindings::bf16(&plan).unwrap();
    for layer in [0, 4] {
        let roles = b.producer_slots(layer).unwrap();
        assert_eq!(roles.geometry.producer, layer);
        let l = Layout::new(roles.geometry.head_dim).unwrap();
        assert_eq!(l.regions.len(), 32);
        assert_eq!(
            l.words,
            4 * (13 * H + 37 * roles.geometry.head_dim + 4 * F + 4 * P) + 16 * 4 * 4 + 32 * GUARD
        );
        assert!(l.words < 300000);
        assert!(ProducerBlock::planned_bytes(layer).unwrap() > l.words * 4);
        assert_eq!(
            ProducerKv::planned_bytes(layer).unwrap(),
            2 * ((if layer == 0 { 512 * 256 } else { 4 * 512 }) + GUARD) * 4
                + size_of::<ProducerKv>()
        );
    }
    for layer in [1, 15, 35, usize::MAX] {
        assert!(b.producer_slots(layer).is_err());
        assert!(ProducerBlock::planned_bytes(layer).is_err());
    }
    assert!(Layout::new(257).is_err());
}
#[test]
fn status_contract_failures_dominate_ordered_numeric_replay() {
    let mut words = [0; STATUS_WORDS];
    words[1] = 2;
    words[34] = 1;
    assert!(matches!(
        status_words_decision(&words).unwrap(),
        Some(ReplayReason::Admission { slot: 1 })
    ));
    words[29] = 16;
    assert!(status_words_decision(&words).is_err());
    words[29] = 0;
    for slot in 0..STATUS_WORDS {
        let original = words[slot];
        words[slot] |= 0x80000000;
        assert!(
            status_words_decision(&words).is_err(),
            "unknown slot {slot}"
        );
        words[slot] = original;
    }
    for slot in [11, 12, 13] {
        words[slot] = 1;
        assert!(status_words_decision(&words).is_err());
        words[slot] = 0;
    }
    words[1] = 1;
    assert!(matches!(
        status_words_decision(&words),
        Err(Error::Nonfinite)
    ));
    words.fill(0);
    words[22] = 8;
    words[29] = 1;
    assert!(matches!(
        status_words_decision(&words).unwrap(),
        Some(ReplayReason::Gelu { slot: 22 })
    ));
    words[22] |= 16;
    assert!(status_words_decision(&words).is_err());
    assert!(status_words_decision(&words[..STATUS_WORDS - 1]).is_err());
}
#[test]
#[ignore = "explicit certified release/Apple GPU first BF16 producer block,8cases/23states/46prefix pairs"]
fn first_complete_producer_block_frozen_states_prefixes_and_publication() {
    autoreleasepool(|_| {
        let fixture = Fixture::load();
        let context = Bf16Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        let weights = source(&context);
        let pipelines = Pipelines::new(&context, PIPELINE_POLICY_BYTES).unwrap();
        let table = table(&context);
        let dir = PathBuf::from(std::env::var_os("TOKLET_PRODUCER_BLOCK_RAW_DIR").unwrap());
        std::fs::create_dir_all(&dir).unwrap();
        let dir = dir.canonicalize().unwrap();
        let cancel = AtomicBool::new(false);
        let mut prefixes: BTreeMap<String, Vec<u32>> = BTreeMap::new();
        let mut gpu_prefixes: BTreeMap<String, Vec<u32>> = BTreeMap::new();
        for layer in [0, 4] {
            let mut block = ProducerBlock::new(
                &weights,
                &context,
                &pipelines,
                &table,
                layer,
                ProducerBlock::planned_bytes(layer).unwrap(),
            )
            .unwrap();
            report(
                "PRODUCER_ADMISSION_GROUPS",
                &json!({"layer":layer,
                "matrix_proofs":block.tiles.iter().map(|t|vec![t.maximum_bits(),t.max_input_bits(),t.l1_input_bits()]).collect::<Vec<_>>(),
                "group_proofs":block.admission_tiles.iter().map(|t|vec![t.maximum_bits(),t.max_input_bits(),t.l1_input_bits()]).collect::<Vec<_>>() }),
            );
            let original_weight_digest = weight_digest(&weights, block.slots);
            for case in fixture.protocol["cases"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|v| v["layer"].as_u64() == Some(layer as u64))
            {
                let name = case["name"].as_str().unwrap();
                let seq = case["sequence"].as_str().unwrap();
                let hidden = fixture.view(&case["inputs"]["hidden"]);
                let ple = fixture.view(&case["inputs"]["per_layer_input"]);
                let prefix = fixture.prefix(case);
                let mut cache = fixture.cache(&block, case);
                let old_offset = cache.offset;
                let old_k = cache.k.with_slice(|s| s.to_vec());
                let old_v = cache.v.with_slice(|s| s.to_vec());
                let old_generation = cache.generation;
                let command = context.0.gpu.command().unwrap();
                let mut encoded = block
                    .encode(&mut cache, &command, &prefix, &hidden, &ple, &cancel)
                    .unwrap();
                assert_eq!(encoded.block.metrics().encoded_dispatches, 35);
                assert!(encoded.block.metrics().encoding_complete);
                let regions: Vec<_> = STATES
                    .iter()
                    .map(|&s| {
                        let r = encoded.block.layout.region(s);
                        (
                            format!("{s:?}"),
                            r.start,
                            r.len,
                            encoded.block.layout.active(s, &encoded.frame),
                        )
                    })
                    .collect();
                encoded.submit_and_wait().unwrap();
                let candidate = encoded
                    .with_completed_arena(|s| capture(&dir, &format!("{name}-gpu-candidate"), s))
                    .unwrap();
                let status = encoded.block.status.with_slice(|s| s.to_vec());
                if case["mode"] == "prefill" {
                    for (key, state) in CAPTURED {
                        let r = encoded.block.layout.region(state);
                        let n = encoded.block.layout.active(state, &encoded.frame);
                        let words = encoded
                            .block
                            .arena
                            .with_slice(|s| s[r.start..r.start + n * 3 / 4].to_vec());
                        gpu_prefixes.insert(format!("{seq}:{layer}:{key}"), words);
                    }
                }
                report(
                    "PRODUCER_GPU_CANDIDATE_RAW",
                    &json!({"case":name,"regions":regions,"arena":candidate,"status_words":status,
                    "old_k":capture(&dir,&format!("{name}-old-k"),&old_k),"old_v":capture(&dir,&format!("{name}-old-v"),&old_v),"metrics":encoded.block.metrics(),
                    "model_payload_authenticated_by_existing_loader":true,"uncaptured_internals_are_independent_witnesses":false}),
                );
                let completed = encoded.complete(&cancel).unwrap();
                let validated = completed
                    .0
                    .with_completed_arena(|words| {
                        capture(&dir, &format!("{name}-validated-arena"), words)
                    })
                    .unwrap();
                report(
                    "PRODUCER_VALIDATED_ARENA_RAW",
                    &json!({"case":name,"arena":validated,"reference_replayed":completed.0.replayed,"metrics":completed.metrics()}),
                );
                for (key, state) in CAPTURED {
                    let expected = fixture.view(&case["expected_states"][key]);
                    completed.with_state(state,|words|{
                        report("PRODUCER_COMPLETED_STATE_RAW",&json!({"case":name,"state":key,"words":capture(&dir,&format!("{name}-{key}"),words),
                            "reference_replayed":completed.0.replayed,"independent_reference":true}));
                        check(words,&expected);
                        if case["mode"]=="prefill"{prefixes.insert(format!("{seq}:{layer}:{key}"),words[..words.len()*3/4].to_vec());}
                    });
                }
                completed.with_state(State::Probabilities, |words| {
                    let rows = hidden.len() / H;
                    for head in 0..8 {
                        for row in 0..rows {
                            for key in 0..4 {
                                if key > old_offset + row {
                                    assert_eq!(words[(head * rows + row) * 4 + key], 0);
                                }
                            }
                        }
                    }
                });
                let proposal_k = completed.with_state(State::KProposal, |s| s.to_vec());
                let proposal_v = completed.with_state(State::VProposal, |s| s.to_vec());
                let mut caller = vec![99.; hidden.len()];
                let metrics = completed.publish(Some(&mut caller), &cancel).unwrap();
                check(
                    &caller.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    &fixture.view(&case["expected_output"]),
                );
                assert_eq!(cache.offset, 4);
                assert_eq!(cache.generation, old_generation + 1);
                assert_eq!(cache.history.as_slice(), prefix.as_slice());
                assert_eq!(
                    metrics.new_kv_copy_bytes,
                    2 * (4 - old_offset) * cache.d * 4
                );
                assert!(cache.guards());
                for (before, after, proposal) in [
                    (&old_k, &cache.k, &proposal_k),
                    (&old_v, &cache.v, &proposal_v),
                ] {
                    after.with_slice(|s| {
                        assert_eq!(&s[..old_offset * cache.d], &before[..old_offset * cache.d]);
                        assert_eq!(&s[old_offset * cache.d..4 * cache.d], proposal.as_slice());
                        assert_eq!(&s[4 * cache.d..], &before[4 * cache.d..]);
                    });
                }
                assert_eq!(
                    metrics.completed_commands,
                    if completed_mode(&metrics) { 10 } else { 1 }
                );
                report(
                    "PRODUCER_PUBLICATION",
                    &json!({"case":name,"metrics":metrics,"committed":cache.offset,"capacity":cache.capacity,
                    "window":cache.window,"shared_table_must_be_counted_once_by_identity":true,"quality_or_performance_qualified":false}),
                );
                assert_eq!(metrics.replay_count, 0, "nominal block must stay on GPU");
                assert_eq!(metrics.reference_commands, Some(0));
                assert_eq!(metrics.reference_waits, Some(0));
                assert_eq!(metrics.submitted_commands, Some(1));
                assert_eq!(metrics.completed_commands, 1);
                assert_eq!(metrics.waits, Some(1));
            }
            let fault_case = fixture.protocol["cases"]
                .as_array()
                .unwrap()
                .iter()
                .find(|v| {
                    v["layer"].as_u64() == Some(layer as u64)
                        && v["sequence"] == "original"
                        && v["mode"] == "prefill"
                })
                .unwrap();
            transaction_faults(
                &weights, &context, &pipelines, &table, &fixture, fault_case, &dir,
            );
            assert_eq!(weight_digest(&weights, block.slots), original_weight_digest);
        }
        for pair in fixture.protocol["causal_prefix"].as_array().unwrap() {
            let layer = pair["layer"].as_u64().unwrap();
            let state = pair["state"].as_str().unwrap();
            assert_eq!(
                prefixes[&format!("original:{layer}:{state}")],
                prefixes[&format!("future:{layer}:{state}")]
            );
            assert_eq!(
                gpu_prefixes[&format!("original:{layer}:{state}")],
                gpu_prefixes[&format!("future:{layer}:{state}")]
            );
        }
        report(
            "PRODUCER_COMPLETE",
            &json!({"cases":8,"captured_states_per_case":23,"causal_prefix_pairs":46,"passed":true,"ordinary_graph_changed":false}),
        );
    });
}
fn completed_mode(m: &Metrics) -> bool {
    m.replay_count != 0
}

fn transaction_faults(
    weights: &Bf16Weights,
    context: &Bf16Context,
    pipelines: &Pipelines,
    table: &gelu::CertifiedTable,
    fixture: &Fixture,
    case: &Value,
    dir: &Path,
) {
    let layer = case["layer"].as_u64().unwrap() as usize;
    let hidden = fixture.view(&case["inputs"]["hidden"]);
    let ple = fixture.view(&case["inputs"]["per_layer_input"]);
    let prefix = fixture.prefix(case);
    // Actual source-bound sparse max miss/L1 success; no fabricated matrix or
    // learned scale. Runtime GPU max-miss numerics are independently covered by
    // the qualified BF16 prerequisite; this certificate pins the block's source.
    let slots = weights.bindings.producer_slots(layer).unwrap();
    let down = weights.matrices[slots.matrices[6]].as_ref().unwrap();
    let tile = down.single_tile(context).unwrap();
    assert!(tile.max_input_bits() < f32::MAX.to_bits());
    let next = f32::from_bits(tile.max_input_bits() + 1);
    let mut sparse = vec![0.; down.columns];
    sparse[down.columns - 1] = next;
    let (s, u) = bounds(down.columns, down.maximum, next);
    assert!(s > f64::from(f32::MAX) / 4. || u > ORDERED_ERROR_LIMIT);
    admit(down.columns, down.maximum, &sparse).unwrap();
    let dense_bounds = bounds(down.columns, down.maximum, f32::MAX);
    let expected_error = if dense_bounds.0 > f64::from(f32::MAX) / 4. {
        "Gemma affine input absolute-product bound"
    } else {
        "Gemma affine input ordered-sum accuracy bound"
    };
    assert!(
        matches!(admit(down.columns,down.maximum,&vec![f32::MAX;down.columns]),Err(Error::Options(actual)) if actual==expected_error)
    );
    report(
        "PRODUCER_SOURCE_L1_CONTROL",
        &json!({"layer":layer,"columns":down.columns,"maximum":down.maximum,
        "max_threshold_bits":tile.max_input_bits(),"l1_threshold_bits":tile.l1_input_bits(),"sparse_next_bits":next.to_bits(),"unchanged_host_L1_passes":true,
        "whole_block_control_is_forced_status_not_an_actual_sparse_model_forward":true}),
    );
    for fault in [
        "gelu-reference",
        "admission-reference",
        "late-cancel",
        "stale-frame",
        "late-status",
        "late-nonfinite",
        "guard-failure",
        "reference-checkpoint-failure",
        "early-replay-late-unknown",
        "early-replay-late-proof",
        "early-replay-late-numeric",
    ] {
        let mut block = ProducerBlock::new(
            weights,
            context,
            pipelines,
            table,
            layer,
            ProducerBlock::planned_bytes(layer).unwrap(),
        )
        .unwrap();
        if fault == "gelu-reference" {
            block.fault = Some(Fault::GeluReplay);
        }
        let mut cache = fixture.cache(&block, case);
        let before_k = cache.k.with_slice(|s| s.to_vec());
        let before_v = cache.v.with_slice(|s| s.to_vec());
        let old_offset = cache.offset;
        let old_history = cache.history;
        let old_generation = cache.generation;
        let cancel = AtomicBool::new(false);
        let command = context.0.gpu.command().unwrap();
        let mut encoded = block
            .encode(&mut cache, &command, &prefix, &hidden, &ple, &cancel)
            .unwrap();
        let _ = encoded.hidden().unwrap();
        encoded.submit_and_wait().unwrap();
        encoded.with_completed_arena(|s|{let pin=capture(dir,&format!("fault-layer{layer}-{fault}-gpu-before-injection"),s);
            report("PRODUCER_FAULT_CANDIDATE",&json!({"layer":layer,"fault":fault,"arena":pin,"metrics":encoded.block.metrics()}));}).unwrap();
        if fault == "guard-failure" {
            encoded.block.arena.with_mut(|s| *s.last_mut().unwrap() = 0);
        }
        if fault == "admission-reference"
            || fault == "reference-checkpoint-failure"
            || fault.starts_with("early-replay-")
        {
            encoded.block.status.with_mut(|s| s[1] = 2);
        }
        if fault == "reference-checkpoint-failure" {
            let r = encoded.block.layout.region(State::Hidden);
            encoded
                .block
                .arena
                .with_mut(|s| s[r.start] = f32::NAN.to_bits());
        }
        if fault == "early-replay-late-unknown" {
            encoded.block.status.with_mut(|s| s[34] = 0x80000000);
        }
        if fault == "early-replay-late-proof" {
            encoded.block.status.with_mut(|s| s[29] = 16);
        }
        if fault == "early-replay-late-numeric" {
            encoded.block.status.with_mut(|s| s[34] = 1);
        }
        if fault.starts_with("early-replay-") {
            encoded.block.status.with_slice(|s| {
                report(
                    "PRODUCER_INJECTED_STATUS_RAW",
                    &json!({"layer":layer,"fault":fault,"status_words":&s[..STATUS_WORDS]}),
                )
            });
        }
        let mut caller = vec![99.; hidden.len()];
        let mut success = false;
        if fault == "guard-failure"
            || fault == "reference-checkpoint-failure"
            || fault == "early-replay-late-unknown"
            || fault == "early-replay-late-proof"
        {
            assert!(encoded.complete(&cancel).is_err());
            assert!(block.poisoned);
        } else {
            let completed = encoded.complete(&cancel).unwrap();
            if fault == "late-cancel" {
                cancel.store(true, Ordering::Release);
            }
            if fault == "stale-frame" {
                completed.0.cache.generation += 1;
            }
            if fault == "late-status" {
                completed.0.block.status.with_mut(|s| s[34] |= 0x80000000);
            }
            if fault == "late-nonfinite" {
                let r = completed.0.block.layout.region(State::Output);
                completed
                    .0
                    .block
                    .arena
                    .with_mut(|s| s[r.start] = f32::NAN.to_bits());
            }
            let result = completed.publish(Some(&mut caller), &cancel);
            success = fault == "gelu-reference"
                || fault == "admission-reference"
                || fault == "early-replay-late-numeric";
            if success {
                let metrics = result.unwrap();
                assert_eq!(metrics.replay_count, 1);
                assert_eq!(metrics.reference_commands, Some(9));
                check(
                    &caller.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    &fixture.view(&case["expected_output"]),
                );
            } else {
                assert!(result.is_err());
            }
        }
        let case_name = case["name"].as_str().unwrap();
        let caller_words: Vec<_> = caller.iter().map(|v| v.to_bits()).collect();
        let before_k_pin = capture(
            dir,
            &format!("fault-layer{layer}-{fault}-before-k"),
            &before_k,
        );
        let before_v_pin = capture(
            dir,
            &format!("fault-layer{layer}-{fault}-before-v"),
            &before_v,
        );
        let after_k_pin = cache
            .k
            .with_slice(|s| capture(dir, &format!("fault-layer{layer}-{fault}-after-k"), s));
        let after_v_pin = cache
            .v
            .with_slice(|s| capture(dir, &format!("fault-layer{layer}-{fault}-after-v"), s));
        report(
            "PRODUCER_FAULT_STATE_RAW",
            &json!({"case":case_name,"layer":layer,"fault":fault,"published":success,
            "before_k":before_k_pin,"before_v":before_v_pin,"after_k":after_k_pin,"after_v":after_v_pin,
            "caller":capture(dir,&format!("fault-layer{layer}-{fault}-caller"),&caller_words),
            "old_offset":old_offset,"old_generation":old_generation,"committed":cache.offset,"generation":cache.generation,"metrics":block.metrics()}),
        );
        if !success {
            assert!(caller.iter().all(|&v| v == 99.));
            assert_eq!(cache.offset, old_offset);
            assert_eq!(cache.history, old_history);
            assert_eq!(cache.k.with_slice(|s| s.to_vec()), before_k);
            assert_eq!(cache.v.with_slice(|s| s.to_vec()), before_v);
            assert_eq!(block.metrics().new_kv_copy_bytes, 0);
            assert_eq!(
                cache.generation,
                old_generation + u64::from(fault == "stale-frame")
            );
        }
        report(
            "PRODUCER_FAULT_RESULT",
            &json!({"layer":layer,"fault":fault,"published":success,"metrics":block.metrics(),
            "committed":cache.offset,"poisoned":block.poisoned,"caller_unchanged_on_rejection":!success}),
        );
    }
    resident_checkpoint_control(weights, context, pipelines, table, fixture, case, dir);
}

fn resident_checkpoint_control(
    weights: &Bf16Weights,
    context: &Bf16Context,
    pipelines: &Pipelines,
    table: &gelu::CertifiedTable,
    fixture: &Fixture,
    case: &Value,
    dir: &Path,
) {
    let layer = case["layer"].as_u64().unwrap() as usize;
    let hidden = fixture.view(&case["inputs"]["hidden"]);
    let ple = fixture.view(&case["inputs"]["per_layer_input"]);
    let prefix = fixture.prefix(case);
    let mut h = Buffer::<u32>::new(&context.0.gpu, hidden.len() + GUARD).unwrap();
    let mut p = Buffer::<u32>::new(&context.0.gpu, ple.len() + GUARD).unwrap();
    for (b, values) in [(&mut h, &hidden), (&mut p, &ple)] {
        b.with_mut(|s| {
            s.fill(SENTINEL);
            for (out, value) in s.iter_mut().zip(values) {
                *out = value.to_bits();
            }
        });
    }
    let before_h = h.with_slice(|s| s.to_vec());
    let before_p = p.with_slice(|s| s.to_vec());
    let mut block = ProducerBlock::new(
        weights,
        context,
        pipelines,
        table,
        layer,
        ProducerBlock::planned_bytes(layer).unwrap(),
    )
    .unwrap();
    let mut cache = fixture.cache(&block, case);
    let cancel = AtomicBool::new(false);
    let command = context.0.gpu.command().unwrap();
    assert!(
        block
            .encode_resident(
                &mut cache,
                &command,
                &prefix,
                BufferRange::new(&h, 0, hidden.len() - 1).unwrap(),
                BufferRange::new(&p, 0, ple.len()).unwrap(),
                &cancel
            )
            .is_err()
    );
    assert!(!block.busy && !block.poisoned && !cache.busy);
    assert_eq!(command.status(), MTLCommandBufferStatus::NotEnqueued);
    let mut encoded = block
        .encode_resident(
            &mut cache,
            &command,
            &prefix,
            BufferRange::new(&h, 0, hidden.len()).unwrap(),
            BufferRange::new(&p, 0, ple.len()).unwrap(),
            &cancel,
        )
        .unwrap();
    encoded.submit_and_wait().unwrap();
    encoded.with_completed_arena(|s|{report("PRODUCER_RESIDENT_CANDIDATE",&json!({"layer":layer,
        "arena":capture(dir,&format!("resident-layer{layer}-candidate"),s),"metrics":encoded.block.metrics(),"external_fixture_upload_bytes":(hidden.len()+ple.len())*4}));}).unwrap();
    let completed = encoded.complete(&cancel).unwrap();
    for (key, state) in CAPTURED {
        completed.with_state(state, |s| {
            check(s, &fixture.view(&case["expected_states"][key]))
        });
    }
    let metrics = completed.publish(None, &cancel).unwrap();
    assert_eq!(metrics.speculative_dispatches, 35);
    assert_eq!(metrics.encoded_blit_copies, 2);
    assert_eq!(metrics.input_upload_bytes, 0);
    assert_eq!(
        metrics.device_input_copy_bytes,
        (hidden.len() + ple.len()) * 4
    );
    assert_eq!(metrics.new_kv_copy_bytes, 2 * 4 * cache.d * 4);
    assert_eq!(h.with_slice(|s| s.to_vec()), before_h);
    assert_eq!(p.with_slice(|s| s.to_vec()), before_p);
    report(
        "PRODUCER_RESIDENT_RESULT",
        &json!({"layer":layer,"metrics":metrics,"hidden_retained_on_GPU":true,"ordinary_graph_changed":false}),
    );
    // Reset invalidates only the logical prefix. Reuse real retained resources
    // and run the explicit apply seam against the same independent prefill.
    let old_generation = cache.generation;
    let before_reset_k = cache.k.with_slice(|s| s.to_vec());
    let before_reset_v = cache.v.with_slice(|s| s.to_vec());
    cache.reset().unwrap();
    assert_eq!(cache.offset, 0);
    assert_eq!(cache.history, [0; 4]);
    assert_eq!(cache.generation, old_generation + 1);
    assert_eq!(cache.k.with_slice(|s| s.to_vec()), before_reset_k);
    assert_eq!(cache.v.with_slice(|s| s.to_vec()), before_reset_v);
    let mut output = vec![99.; hidden.len()];
    let reused = block
        .apply(&mut cache, &prefix, &hidden, &ple, &mut output, &cancel)
        .unwrap();
    let output_words: Vec<_> = output.iter().map(|v| v.to_bits()).collect();
    report(
        "PRODUCER_REUSE_RESULT",
        &json!({"layer":layer,"metrics":reused,
        "output":capture(dir,&format!("reuse-layer{layer}-output"),&output_words),
        "committed":cache.offset,"generation":cache.generation,"old_generation":old_generation,
        "reset_preserved_physical_KV":true}),
    );
    check(&output_words, &fixture.view(&case["expected_output"]));
    assert_eq!(cache.offset, 4);
    assert_eq!(cache.generation, old_generation + 2);

    // enqueue() reserves a queue position but does not submit. Dropping the
    // guard must commit/drain while resident borrows still live, preserve KV,
    // and release the queue so a later command can complete.
    cache.reset().unwrap();
    let before_drop_k = cache.k.with_slice(|s| s.to_vec());
    let before_drop_v = cache.v.with_slice(|s| s.to_vec());
    let generation = cache.generation;
    let command = context.0.gpu.command().unwrap();
    let encoded = block
        .encode_resident(
            &mut cache,
            &command,
            &prefix,
            BufferRange::new(&h, 0, hidden.len()).unwrap(),
            BufferRange::new(&p, 0, ple.len()).unwrap(),
            &cancel,
        )
        .unwrap();
    command.enqueue();
    assert_eq!(command.status(), MTLCommandBufferStatus::Enqueued);
    drop(encoded);
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    assert!(command.error().is_none());
    assert!(block.poisoned && !block.busy && !cache.busy);
    assert_eq!(cache.offset, 0);
    assert_eq!(cache.generation, generation);
    assert_eq!(cache.k.with_slice(|s| s.to_vec()), before_drop_k);
    assert_eq!(cache.v.with_slice(|s| s.to_vec()), before_drop_v);
    assert_eq!(h.with_slice(|s| s.to_vec()), before_h);
    assert_eq!(p.with_slice(|s| s.to_vec()), before_p);
    let probe = context.0.gpu.command().unwrap();
    probe.commit();
    probe.waitUntilCompleted();
    assert_eq!(probe.status(), MTLCommandBufferStatus::Completed);
    assert!(probe.error().is_none());
    report(
        "PRODUCER_ENQUEUED_DROP_RESULT",
        &json!({"layer":layer,"metrics":block.metrics(),
        "command_completed":true,"later_queue_probe_completed":true,
        "resident_sources_unchanged":true,"committed":cache.offset,"generation":cache.generation,
        "old_generation":generation,"poisoned":block.poisoned,
        "before_k":capture(dir,&format!("drop-layer{layer}-before-k"),&before_drop_k),
        "before_v":capture(dir,&format!("drop-layer{layer}-before-v"),&before_drop_v),
        "after_k":cache.k.with_slice(|s|capture(dir,&format!("drop-layer{layer}-after-k"),s)),
        "after_v":cache.v.with_slice(|s|capture(dir,&format!("drop-layer{layer}-after-v"),s))}),
    );
}
