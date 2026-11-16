use super::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
};
const LIMIT: usize = 6 * 1024 * 1024 * 1024;
const HELPER: usize = 512 * 1024 * 1024;
const DIAGNOSTIC: usize = 64 * 1024 * 1024;
const PROTOCOL_SHA: &str = "e44783106e13205bafa45915f70e135f5f00b456a87fa74b4e28d121eabea49e";
const DATA_SHA: &str = "bb10b52b36173f09b9ebb024d13b5a4ad46026b048513472682c73de4a7c8763";
fn read(path: &Path, bytes: usize) -> Vec<u8> {
    assert_eq!(path.metadata().unwrap().len(), bytes as u64);
    let mut raw = Vec::with_capacity(bytes);
    std::fs::File::open(path)
        .unwrap()
        .take(bytes as u64 + 1)
        .read_to_end(&mut raw)
        .unwrap();
    assert_eq!(raw.len(), bytes);
    raw
}
fn pinned(path: &Path, bytes: usize, sha: &str) -> Vec<u8> {
    let raw = read(path, bytes);
    assert_eq!(format!("{:x}", Sha256::digest(&raw)), sha);
    raw
}

#[test]
fn recipe_dispatch_metrics_preserve_success_and_fail_closed_on_partial_errors() {
    let mut success = Metrics {
        encoded_dispatches: Some(0),
        ..Metrics::default()
    };
    for count in [35, 35, 32] {
        success.record_recipe_encoding(&Ok(()), count, false);
    }
    assert_eq!(success.encoded_dispatches, Some(102));
    assert!(!success.encoding_complete);

    // Attention may emit one, two or three dispatches, then return Err before
    // its caller records any of them. The count is intentionally unavailable.
    fn fail_after_partial_dispatches(emitted: &Cell<usize>, partial: usize) -> Result<()> {
        for _ in 0..partial {
            emitted.set(emitted.get() + 1);
        }
        Err(Error::Options("synthetic partial attention encoding"))
    }
    for partial in 1..=3 {
        let emitted = Cell::new(0);
        let result = fail_after_partial_dispatches(&emitted, partial);
        assert_eq!(emitted.get(), partial);
        for recorded in [0, 7] {
            for partial_count_unknown in [false, true] {
                let mut metrics = Metrics {
                    encoded_dispatches: Some(35),
                    ..Metrics::default()
                };
                metrics.record_recipe_encoding(&result, recorded, partial_count_unknown);
                assert!(matches!(
                    &result,
                    Err(Error::Options("synthetic partial attention encoding"))
                ));
                assert_eq!(metrics.encoded_dispatches, None);
                assert!(!metrics.encoding_complete);
                metrics.record_recipe_encoding(&Ok(()), 32, false);
                assert_eq!(metrics.encoded_dispatches, None);
                assert!(!metrics.encoding_complete);
                assert_eq!(
                    (metrics.candidate_commands, metrics.candidate_waits),
                    (0, 0)
                );
            }
        }
    }
    success.record_recipe_encoding(&Ok(()), 0, true);
    assert_eq!(success.encoded_dispatches, None);
    success.record_recipe_encoding(&Ok(()), 35, false);
    assert_eq!(success.encoded_dispatches, None);
}

#[test]
fn chain_geometry_and_optional_source_roles_are_admitted_without_gpu() {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gemma-unquantized-header-v1");
    let config = pinned(
        &fixture.join("config.json"),
        4954,
        "1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330",
    );
    let header = pinned(
        &fixture.join("header.json"),
        263952,
        "12740d6fe7a66b316040fa4d77471a8e1809498a71992b3364a6d5417d10662e",
    );
    let plan = crate::mlx::gemma4::bf16::inspect(
        &config,
        &header,
        crate::mlx::gemma4::bf16::CONTAINER_BYTES,
        crate::mlx::gemma4::ModalityPolicy::TextOnly,
    )
    .unwrap();
    let bindings = Bindings::bf16(&plan).unwrap();
    for index in LAYERS {
        let roles = bindings.chain_layer_slots(index).unwrap();
        assert_eq!(
            roles.geometry.producer,
            if index == 15 { 13 } else { index }
        );
        assert_eq!(roles.kv.is_some(), index != 15);
        assert_eq!(
            roles.geometry.intermediate,
            if index == 15 { 12288 } else { 6144 }
        );
    }
    assert!(bindings.chain_layer_slots(35).is_err());
    assert!(bindings.chain_layer_slots(usize::MAX).is_err());
    for rows in [1, 4, 32] {
        for (d, ff, producer) in [(256, 6144, true), (512, 6144, true), (256, 12288, false)] {
            let layout = LayerLayout::new(rows, 512, d, ff, producer).unwrap();
            let mut end = 0;
            for state in STATES {
                let r = layout.region(state);
                assert_eq!(r.start, end);
                end = r.end() + GUARD;
                assert_eq!(r.len, recipe::values(state, rows, 512, d, ff, producer));
            }
            assert_eq!(end, layout.words);
            for state in [
                State::KRaw,
                State::KNorm,
                State::KProposal,
                State::VRaw,
                State::VProposal,
            ] {
                assert_eq!(layout.region(state).len == 0, !producer);
            }
            assert!(LayerStorage::planned_bytes(layout).unwrap() > layout.words * 4);
        }
    }
    for (rows, keys, d, ff, producer) in [
        (0, 512, 256, 6144, true),
        (33, 512, 256, 6144, true),
        (usize::MAX, 512, 256, 6144, true),
        (4, 0, 256, 6144, true),
        (4, 513, 256, 6144, true),
        (4, 512, 257, 6144, true),
        (4, 512, 256, 6144, false),
        (4, 512, 256, 12288, true),
    ] {
        assert!(LayerLayout::new(rows, keys, d, ff, producer).is_err());
    }
    assert_eq!(
        CacheStorage::planned_bytes(256, 512).unwrap()
            + CacheStorage::planned_bytes(512, 512).unwrap()
            - 2 * size_of::<CacheStorage>(),
        3145728 + 4 * GUARD * 4
    );
    for options in [
        Options {
            capacity: 0,
            max_rows: 4,
        },
        Options {
            capacity: 513,
            max_rows: 4,
        },
        Options {
            capacity: 512,
            max_rows: 0,
        },
        Options {
            capacity: 512,
            max_rows: 33,
        },
        Options {
            capacity: usize::MAX,
            max_rows: 4,
        },
    ] {
        assert!(ResidentGraph::planned_bytes(options).is_err());
    }
}

#[test]
fn global_contract_audit_precedes_whole_chain_ordered_replay() {
    let mut words = [[0; STATUS_WORDS]; 3];
    assert!(chain_status_decision(&words).unwrap().is_none());
    words[0][23] = 2;
    words[2][34] = 1;
    assert!(matches!(
        chain_status_decision(&words).unwrap(),
        Some((
            0,
            ChainReplayReason::Original(ReplayReason::Admission { slot: 23 })
        ))
    ));
    for (node, slot, flag) in [(2, 22, 16), (2, 29, 32), (1, 11, 1), (2, 5, 1), (2, 9, 1)] {
        let mut corrupt = words;
        corrupt[node][slot] = flag;
        assert!(chain_status_decision(&corrupt).is_err());
    }
    words[0][0] = 1;
    assert!(matches!(
        chain_status_decision(&words),
        Err(Error::Nonfinite)
    ));
    let mut gelu = [[0; STATUS_WORDS]; 3];
    gelu[0][22] = 8;
    gelu[1][0] = 1;
    assert!(matches!(
        chain_status_decision(&gelu).unwrap(),
        Some((
            0,
            ChainReplayReason::Original(ReplayReason::Gelu { slot: 22 })
        ))
    ));
    let mut later = gelu;
    later[2][29] = 16;
    assert!(chain_status_decision(&later).is_err());
    let mut rms_words = [[0; STATUS_WORDS]; 3];
    rms_words[0][0] = 2;
    rms_words[2][25] = 1;
    assert!(matches!(
        chain_status_decision(&rms_words).unwrap(),
        Some((0, ChainReplayReason::Rms { slot: 0 }))
    ));
    for (node, slot, word) in [(2, 0, 4), (2, 6, 2), (2, 9, 2), (2, 29, 16), (1, 11, 1)] {
        let mut corrupt = rms_words;
        corrupt[node][slot] = word;
        assert!(chain_status_decision(&corrupt).is_err());
    }
    rms_words[0][0] = 3;
    assert!(matches!(
        chain_status_decision(&rms_words),
        Err(Error::Nonfinite)
    ));
    let mut original_rms = [0; STATUS_WORDS];
    original_rms[0] = 2;
    assert!(status_words_decision(&original_rms).is_err());
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(include_bytes!("../../../../../gemma_rms/precision.metal"))
        ),
        "dc16e73c6ef9a43c57e319e62e33090a969ed2b664e47cc0496a02bf9bfca6a8"
    );
    let old = window_rows::old_spans(4, 6, Some(4));
    assert_eq!((old[0].position, old[0].slot, old[0].rows), (3, 3, 1));
    assert_eq!((old[1].position, old[1].slot, old[1].rows), (4, 0, 2));
    let copy = window_rows::copy_runs(4, 6, 3, true);
    assert_eq!(copy[0].rows + copy[1].rows, 3);
}

struct Fixture {
    protocol: Value,
    states: Vec<u8>,
}
impl Fixture {
    fn load() -> Self {
        let root = PathBuf::from(std::env::var_os("TOKLET_GEMMA_DEEP_CHAIN_FIXTURE").unwrap());
        let protocol = pinned(&root.join("protocol.json"), 641748, PROTOCOL_SHA);
        let states = pinned(&root.join("states.bin"), 4268032, DATA_SHA);
        Self {
            protocol: serde_json::from_slice(&protocol).unwrap(),
            states,
        }
    }
    fn view(&self, d: &Value) -> Vec<f32> {
        assert_eq!(d["dtype"], "<f4");
        let shape: Vec<_> = d["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        let strides: Vec<_> = d["byte_strides"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(shape.len(), strides.len());
        let mut raw = Vec::with_capacity(shape.iter().product::<usize>() * 4);
        let count: usize = shape.iter().product();
        for mut index in 0..count {
            let mut offset = d["offset"].as_u64().unwrap() as usize;
            for axis in (0..shape.len()).rev() {
                offset += (index % shape[axis]) * strides[axis];
                index /= shape[axis];
            }
            raw.extend_from_slice(&self.states[offset..offset + 4]);
        }
        assert_eq!(format!("{:x}", Sha256::digest(&raw)), d["logical_sha256"]);
        raw.chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect()
    }
    fn checkpoint(&self, case: &Value) -> (Vec<f32>, [Vec<f32>; 3]) {
        let all = self.protocol["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| {
                c["sequence"] == case["sequence"]
                    && c["mode"] == case["mode"]
                    && c["output_mode"] == "All"
            })
            .unwrap();
        (
            self.view(&all["nodes"][0]["input_hidden"]),
            std::array::from_fn(|node| self.view(&all["nodes"][node]["input_ple"])),
        )
    }
    fn prefix(&self, case: &Value) -> Vec<u32> {
        case["full_prefix_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect()
    }
    fn seed(&self, session: &mut ResidentSession<'_, '_>, case: &Value) {
        let n = case["offset"].as_u64().unwrap() as usize;
        if n != 0 {
            let prefix = self.prefix(case);
            let k = [
                self.view(&case["caches"]["13"]["old_key"]),
                self.view(&case["caches"]["14"]["old_key"]),
            ];
            let v = [
                self.view(&case["caches"]["13"]["old_value"]),
                self.view(&case["caches"]["14"]["old_value"]),
            ];
            session
                .seed_prefix(&prefix[..n], [(&k[0], &v[0]), (&k[1], &v[1])])
                .unwrap();
        }
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
    for (index, (&word, &reference)) in actual.iter().zip(expected).enumerate() {
        let actual = f32::from_bits(word);
        assert!(actual.is_finite() && reference.is_finite());
        assert!(
            (f64::from(actual) - f64::from(reference)).abs()
                <= 1e-5 + 1e-4 * f64::from(reference).abs(),
            "original gate index{index}: {actual} != {reference}"
        );
    }
}
fn core_pipelines(
    context: &Bf16Context,
    table: &gelu::CertifiedTable,
    available: usize,
) -> Pipelines {
    check_budget(PIPELINE_POLICY_BYTES, available).unwrap();
    // Test-only cold constructor: exactly one selected GELU pipeline, never a
    // v1 pipeline compiled and then replaced. Existing six-family policy stays.
    let gpu = &context.0.gpu;
    let gelu = gelu::Pipelines::new_core(gpu).unwrap();
    gelu.validate_table(table).unwrap();
    Pipelines {
        device: gpu.device.clone(),
        rms: rms::Pipelines::new(gpu).unwrap(),
        rotary: rotary::Pipelines::new(gpu).unwrap(),
        attention: attention::Pipelines::new(gpu).unwrap(),
        gelu,
        residual: residual::Pipelines::new(gpu).unwrap(),
        admission: AdmissionPipeline::new(gpu).unwrap(),
    }
}
fn table(context: &Bf16Context) -> gelu::CertifiedTable {
    let load = |key, n| read(&PathBuf::from(std::env::var_os(key).unwrap()), n);
    let table = load("TOKLET_GELU_CORE_TABLE", 5480520);
    let certificate = load("TOKLET_GELU_CORE_CERTIFICATE", 5608);
    let native_verification = load("TOKLET_GELU_CORE_NATIVE_VERIFICATION", 988);
    let artifact_review = load(
        "TOKLET_GELU_CORE_ARTIFACT_REVIEW",
        gelu::core::CORE_REVIEW_BYTES,
    );
    let parent_certificate = load("TOKLET_GELU_CERTIFICATE", 2309);
    let parent_native_verification = load("TOKLET_GELU_NATIVE_VERIFICATION", 635);
    let parent_artifact_review = load("TOKLET_GELU_ARTIFACT_REVIEW", 13881);
    let owner = gelu::CertifiedTable::from_approved_core(
        &context.0.gpu,
        &gelu::core::ApprovedCoreBytes {
            table: &table,
            certificate: &certificate,
            native_verification: &native_verification,
            artifact_review: &artifact_review,
            parent_certificate: &parent_certificate,
            parent_native_verification: &parent_native_verification,
            parent_artifact_review: &parent_artifact_review,
        },
        gelu::CertifiedTable::planned_core_owned_bytes(),
    )
    .unwrap();
    assert_eq!(
        owner.owned_bytes(),
        gelu::CertifiedTable::planned_core_owned_bytes()
    );
    assert_eq!(
        owner.owned_bytes() - gelu::CertifiedTable::planned_owned_bytes(),
        2162736
    );
    owner
}
fn source(context: &Bf16Context) -> Bf16Weights {
    let root = PathBuf::from(std::env::var_os("TOKLET_GEMMA_BF16_MODEL").unwrap());
    let reserved = HELPER
        + DIAGNOSTIC
        + PIPELINE_POLICY_BYTES
        + StridedBf16Pipelines::policy_bytes()
        + rms::precision::POLICY_BYTES
        + gelu::CertifiedTable::planned_core_owned_bytes()
        + 128 * 1024 * 1024;
    Bf16Weights::load_text_only(
        context,
        std::fs::File::open(root.join("model.safetensors")).unwrap(),
        &read(&root.join("config.json"), 4954),
        LIMIT - reserved,
    )
    .unwrap()
}
fn update_words(hash: &mut Sha256, words: &[u32]) {
    let mut raw = [0; 4096];
    for chunk in words.chunks(1024) {
        for (word, out) in chunk.iter().zip(raw.chunks_exact_mut(4)) {
            out.copy_from_slice(&word.to_le_bytes());
        }
        hash.update(&raw[..chunk.len() * 4]);
    }
}
fn digest_words(words: &[u32]) -> String {
    let mut hash = Sha256::new();
    update_words(&mut hash, words);
    format!("{:x}", hash.finalize())
}
fn cache_hash(cache: &CacheStorage) -> String {
    let mut hash = Sha256::new();
    cache.k.with_slice(|s| update_words(&mut hash, s));
    cache.v.with_slice(|s| update_words(&mut hash, s));
    format!("{:x}", hash.finalize())
}
fn weights_hash(graph: &ResidentGraph<'_>) -> String {
    let mut hash = Sha256::new();
    for p in &graph.plans {
        for slot in p.matrix_slots.iter().flatten() {
            let m = graph.source.matrices[*slot].as_ref().unwrap();
            for tile in &m.tiles {
                tile.with_slice(|s| update_words(&mut hash, s));
            }
        }
        for slot in p
            .norm_slots
            .iter()
            .flatten()
            .chain(std::iter::once(&p.slots.scalar))
        {
            for x in &graph.source.vectors[*slot] {
                hash.update(x.to_le_bytes());
            }
        }
    }
    format!("{:x}", hash.finalize())
}
struct RawWriter {
    root: PathBuf,
    file: BufWriter<std::fs::File>,
    bytes: usize,
    cases: Vec<Value>,
    format: &'static str,
}
impl RawWriter {
    fn new(format: &'static str) -> Self {
        let root = PathBuf::from(std::env::var_os("TOKLET_DEEP_CHAIN_RAW_DIR").unwrap());
        std::fs::create_dir(&root).unwrap();
        let file = BufWriter::new(std::fs::File::create_new(root.join("arrays.bin")).unwrap());
        Self {
            root,
            file,
            bytes: 0,
            cases: Vec::new(),
            format,
        }
    }
    fn words(&mut self, words: &[u32]) -> Value {
        assert!(self.bytes + words.len() * 4 <= 16 * 1024 * 1024);
        let offset = self.bytes;
        let mut block = [0; 4096];
        for chunk in words.chunks(1024) {
            for (word, out) in chunk.iter().zip(block.chunks_exact_mut(4)) {
                out.copy_from_slice(&word.to_le_bytes());
            }
            self.file.write_all(&block[..chunk.len() * 4]).unwrap();
        }
        self.bytes += words.len() * 4;
        json!({"offset":offset,"bytes":words.len()*4,"words":words.len(),"sha256":digest_words(words)})
    }
    fn finish(self) {
        self.finish_with_status(true);
    }
    fn finish_with_status(mut self, passed: bool) {
        self.file.flush().unwrap();
        drop(self.file);
        let raw = read(&self.root.join("arrays.bin"), self.bytes);
        let report = json!({"format":self.format,"passed":passed,"fixture_protocol_sha256":PROTOCOL_SHA,"fixture_data_sha256":DATA_SHA,"cases":self.cases,"raw":{"path":self.root.join("arrays.bin"),"bytes":raw.len(),"sha256":format!("{:x}",Sha256::digest(&raw))},"decode_scope":"prefill-derived causal row3 views, not separately capturedT1","uncaptured_internals_are_independent_witnesses":false,"normal_route_selected":false,"deep_chain_rms_precision":"scaled-double-float-guard2",
            "deep_chain_gelu_table":{"variant":"finer-affine-core-v2","table_sha256":gelu::core::TABLE_SHA,"certificate_sha256":gelu::core::CERT_SHA,"native_verification_sha256":gelu::core::NATIVE_SHA,"artifact_review_sha256":gelu::core::CORE_REVIEW_SHA,"parent_table_sha256":"78dfa84189353b4f9e31cc1a6e6e3a6476ef4818c7548c3c3c06e9227764dcc6","payload_bytes":5480520,"added_payload_bytes":2162736,"guard_bytes":56,"owned_bytes":gelu::CertifiedTable::planned_core_owned_bytes(),"parent_owned_bytes":gelu::CertifiedTable::planned_owned_bytes(),"added_owned_bytes":2162736,"combined_GPU_tables":1,"selected_GELU_pipelines":1},"whole_graph_or_quality_or_performance_qualified":false});
        let mut file = std::fs::File::create_new(self.root.join("result.json")).unwrap();
        serde_json::to_writer_pretty(&mut file, &report).unwrap();
        file.write_all(b"\n").unwrap();
    }
}
fn capture_states(tx: &GraphTransaction<'_, '_, '_>, raw: &mut RawWriter) -> Vec<Value> {
    (0..3).map(|node| {
        let layout = tx.session.graph.plans[node].layout;
        let frame = tx.frame.nodes[node];
        let mut states = Vec::new();
        for state in STATES {
            if layout.active(state, frame) == 0 { continue; }
            let words = tx.with_completed_state(node, state, |s|s.to_vec()).unwrap();
            let mut desc = raw.words(&words);
            desc["state"] = json!(state as usize);
            desc["name"] = json!(format!("{state:?}"));
            states.push(desc);
        }
        let status = tx.session.storage[node].status.with_slice(|s|raw.words(&s[..STATUS_WORDS]));
        let mut reasons = Vec::new();
        tx.session.storage[node].rms_reasons.with_slice(|words| {
            for (slot, region) in layout.rms_plan().0.into_iter().enumerate() {
                if let Some(region) = region {
                    let rows = frame.rows * if slot == 1 { 8 } else { 1 };
                    let mut desc = raw.words(&words[region.start..region.start + rows]);
                    desc["norm_slot"] = json!(slot);
                    desc["status_slot"] = json!(rms::precision::STATUS_SLOTS[slot]);
                    reasons.push(desc);
                    tx.session.diagnostic_bytes.set(tx.session.diagnostic_bytes.get()+rows*4);
                }
            }
        });
        json!({"layer":LAYERS[node],"frame":frame,"D":layout.d,"FF":layout.ff,"states":states,"status":status,"rms_reasons":reasons})
    }).collect()
}

fn expected_states(completed: &CompletedGraph<'_, '_, '_>, fixture: &Fixture, case: &Value) {
    for (node, &layer) in LAYERS.iter().enumerate() {
        for (suffix, state) in CAPTURED {
            if let Some(desc) = case["nodes"][node]["expected_states"].get(suffix) {
                let expected = fixture.view(desc);
                completed
                    .with_state(node, state, |s| {
                        let summary = historical_summary(s, &expected);
                        assert_eq!(
                            summary["failing_values"], 0,
                            "historical case{} layer{} state{suffix}: {summary}",
                            case["name"], layer
                        );
                    })
                    .unwrap();
            }
        }
    }
}
fn mask_count(tx: &GraphTransaction<'_, '_, '_>) -> usize {
    let mut count = 0;
    for node in 0..3 {
        let f = tx.frame.nodes[node];
        let window = if node == 1 { None } else { Some(512) };
        tx.with_completed_state(node, State::Probabilities, |s| {
            assert_eq!(s.len(), 8 * f.rows * f.keys);
            for h in 0..8 {
                for q in 0..f.rows {
                    for k in 0..f.keys {
                        let position = f.key_start + k;
                        let query = f.query_start + q;
                        let masked = position > query
                            || window.is_some_and(|w| {
                                position <= (query.saturating_sub(w)) && query >= w
                            });
                        if masked {
                            assert_eq!(s[(h * f.rows + q) * f.keys + k], 0);
                            count += 1;
                        }
                    }
                }
            }
        })
        .unwrap();
    }
    count
}
fn prefix_hashes(tx: &GraphTransaction<'_, '_, '_>) -> Vec<String> {
    let mut hashes = Vec::new();
    for node in 0..3 {
        for (_, state) in CAPTURED {
            let l = tx.session.graph.plans[node].layout;
            if l.active(state, tx.frame.nodes[node]) == 0 {
                continue;
            }
            hashes.push(
                tx.with_completed_state(node, state, |s| digest_words(&s[..s.len() / 4 * 3]))
                    .unwrap(),
            );
        }
    }
    hashes
}
fn mode(case: &Value) -> OutputRows {
    if case["output_mode"] == "Last" {
        OutputRows::Last
    } else {
        OutputRows::All
    }
}
fn ledger(graph: &ResidentGraph<'_>, session: &ResidentSession<'_, '_>) -> usize {
    graph.source.report.retained_bytes
        + CONTEXT_ALLOWANCE_BYTES
        + PIPELINE_POLICY_BYTES
        + graph.admitted_bytes()
        + graph.table.owned_bytes()
        + session.owned_bytes()
        + HELPER
        + DIAGNOSTIC
}

#[test]
#[ignore = "explicit authenticated contiguous13->14->15 eight-cell resident/state/commit qualification"]
fn genuine_deep_chain_eight_cells_share_one_transaction_and_commit_both_caches() {
    autoreleasepool(|_| {
        let fixture = Fixture::load();
        let context = Bf16Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        let source = source(&context);
        let table = table(&context);
        let pipelines = core_pipelines(&context, &table, PIPELINE_POLICY_BYTES);
        let options = Options {
            capacity: 512,
            max_rows: 4,
        };
        let graph = ResidentGraph::new(
            &source,
            &context,
            &pipelines,
            &table,
            options,
            ResidentGraph::planned_bytes(options).unwrap(),
        )
        .unwrap();
        let original_weights = weights_hash(&graph);
        let cancel = AtomicBool::new(false);
        let mut raw = RawWriter::new("toklet-g2-deep-chain-cells-v1");
        let mut original_prefix = None;
        let mut future_prefix = None;
        for case in fixture.protocol["cases"].as_array().unwrap() {
            autoreleasepool(|_| {
                let mut session =
                    ResidentSession::new(&graph, ResidentSession::planned_bytes(&graph).unwrap())
                        .unwrap();
                assert!(ledger(&graph, &session) <= LIMIT);
                fixture.seed(&mut session, case);
                let cache_before = session.caches.each_ref().map(cache_hash);
                let generations = session.caches.each_ref().map(|c| c.generation);
                let (hidden, ple) = fixture.checkpoint(case);
                let prefix = fixture.prefix(case);
                let output_rows = case["final_queries"].as_u64().unwrap() as usize;
                let mut output = vec![99.; output_rows * H];
                let mut tx = session
                    .begin(
                        &prefix,
                        &hidden,
                        [&ple[0], &ple[1], &ple[2]],
                        mode(case),
                        &cancel,
                    )
                    .unwrap();
                assert_eq!(
                    tx.frame.nodes[2].proposal_rows,
                    case["input_rows"].as_u64().unwrap() as usize
                );
                tx.encode_all(&cancel).unwrap();
                assert!(output.iter().all(|&x| x == 99.));
                tx.submit_and_wait().unwrap();
                let candidate_cache = tx.session.caches.each_ref().map(cache_hash);
                assert_eq!(candidate_cache, cache_before);
                let states = capture_states(&tx, &mut raw);
                let masks = mask_count(&tx);
                if case["mode"] == "prefill" && mode(case) == OutputRows::All {
                    let hashes = prefix_hashes(&tx);
                    if case["sequence"] == "original" {
                        original_prefix = Some(hashes);
                    } else {
                        future_prefix = Some(hashes);
                    }
                }
                let completed = tx.complete(&cancel).unwrap();
                expected_states(&completed, &fixture, case);
                assert_eq!(completed.metrics().replay_count, 0);
                assert_eq!(completed.metrics().encoded_dispatches, Some(102));
                let proposals: [(Vec<u32>, Vec<u32>); 2] = std::array::from_fn(|node| {
                    (
                        completed
                            .with_state(node, State::KProposal, |s| s.to_vec())
                            .unwrap(),
                        completed
                            .with_state(node, State::VProposal, |s| s.to_vec())
                            .unwrap(),
                    )
                });
                let metrics = completed.publish(Some(&mut output), &cancel).unwrap();
                assert_eq!(session.committed(), 4);
                assert_eq!(session.prefix(), prefix);
                assert_eq!(session.output().unwrap(), output);
                check(
                    &output.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    &fixture.view(&case["expected_output"]),
                );
                assert_eq!(metrics.candidate_commands, 1);
                assert_eq!(metrics.candidate_waits, 1);
                assert_eq!(metrics.reference_commands, Some(0));
                assert_eq!(
                    metrics.new_kv_copy_bytes,
                    case["input_rows"].as_u64().unwrap() as usize * 6144
                );
                let offset = case["offset"].as_u64().unwrap() as usize;
                let mut kv = Vec::new();
                for node in 0..2 {
                    let c = &session.caches[node];
                    assert_eq!(c.offset, 4);
                    assert_eq!(c.generation, generations[node] + 1);
                    let n = proposals[node].0.len();
                    c.k.with_slice(|s| {
                        assert_eq!(&s[offset * c.d..offset * c.d + n], proposals[node].0)
                    });
                    c.v.with_slice(|s| {
                        assert_eq!(&s[offset * c.d..offset * c.d + n], proposals[node].1)
                    });
                    kv.push(json!({"layer":LAYERS[node],"generation":c.generation,"offset":c.offset,"K":c.k.with_slice(|s|raw.words(&s[..4*c.d])),"V":c.v.with_slice(|s|raw.words(&s[..4*c.d])),"guarded_sha256":cache_hash(c)}));
                }
                let output_raw = raw.words(&output.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
                raw.cases.push(json!({"name":case["name"],"sequence":case["sequence"],"mode":case["mode"],"output_mode":case["output_mode"],"nodes":states,"exact_mask_zeros":masks,"cache_before_sha256":cache_before,"cache_after_candidate_sha256":candidate_cache,"cache_generations_before":generations,"committed_cache":kv,"output":output_raw,"metrics":metrics,"model_payload_authenticated_by_existing_loader":true,"deep_entry_input_bytes":hidden.len()*4,"checkpoint_PLE15_uses_full_All_view_before_Last":true}));
            });
        }
        assert_eq!(original_prefix.as_ref().unwrap().len(), 64);
        assert_eq!(original_prefix, future_prefix);
        assert_eq!(weights_hash(&graph), original_weights);
        raw.finish();
    });
}

fn seeded<'g, 'w>(
    graph: &'g ResidentGraph<'w>,
    fixture: &Fixture,
    case: &Value,
) -> ResidentSession<'g, 'w> {
    let mut session =
        ResidentSession::new(graph, ResidentSession::planned_bytes(graph).unwrap()).unwrap();
    fixture.seed(&mut session, case);
    session
}
fn run<'s, 'g, 'w>(
    session: &'s mut ResidentSession<'g, 'w>,
    fixture: &Fixture,
    case: &Value,
    cancel: &AtomicBool,
) -> GraphTransaction<'s, 'g, 'w> {
    let (hidden, ple) = fixture.checkpoint(case);
    let prefix = fixture.prefix(case);
    let mut tx = session
        .begin(
            &prefix,
            &hidden,
            [&ple[0], &ple[1], &ple[2]],
            mode(case),
            cancel,
        )
        .unwrap();
    tx.encode_all(cancel).unwrap();
    tx.submit_and_wait().unwrap();
    tx
}
fn status_capture(session: &ResidentSession<'_, '_>, raw: &mut RawWriter) -> Value {
    let mut words = [0; 3 * STATUS_WORDS];
    for node in 0..3 {
        session.storage[node].status.with_slice(|s| {
            words[node * STATUS_WORDS..(node + 1) * STATUS_WORDS]
                .copy_from_slice(&s[..STATUS_WORDS])
        });
    }
    raw.words(&words)
}
fn failure(
    raw: &mut RawWriter,
    name: &str,
    session: &ResidentSession<'_, '_>,
    before: &[String; 2],
    prefix: &[u32],
    output: &[f32],
    error: &str,
) {
    let after = session.caches.each_ref().map(cache_hash);
    assert_eq!(&after, before);
    assert_eq!(session.prefix(), prefix);
    assert!(output.iter().all(|&x| x == 99.));
    let statuses = status_capture(session, raw);
    let out = raw.words(&output.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    raw.cases.push(json!({"name":name,"error":error,"statuses":statuses,"cache_before_sha256":before,"cache_after_sha256":after,"prefix_before":prefix,"prefix_after":session.prefix(),"cache_offsets":session.caches.each_ref().map(|c|c.offset),"cache_generations":session.caches.each_ref().map(|c|c.generation),"caller_output":out,"poisoned":session.poisoned,"busy":session.busy,"public_metrics_available":session.metrics().is_ok(),"metrics_private_failure_diagnostic":session.observed_metrics(),"published":false}));
}

#[test]
#[ignore = "explicit whole-chain replay/resident checkpoint/atomic failure/cancel/drop/reuse qualification"]
fn whole_chain_replay_and_lifecycle_keep_exact_committed_prefix() {
    autoreleasepool(|_| {
        let fixture = Fixture::load();
        let context = Bf16Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        let source = source(&context);
        let table = table(&context);
        let pipelines = core_pipelines(&context, &table, PIPELINE_POLICY_BYTES);
        let options = Options {
            capacity: 512,
            max_rows: 4,
        };
        let graph = ResidentGraph::new(
            &source,
            &context,
            &pipelines,
            &table,
            options,
            ResidentGraph::planned_bytes(options).unwrap(),
        )
        .unwrap();
        let weights_before = weights_hash(&graph);
        let cases = fixture.protocol["cases"].as_array().unwrap();
        let prefill = cases
            .iter()
            .find(|c| c["name"] == "original-prefill-all")
            .unwrap();
        let last = cases
            .iter()
            .find(|c| c["name"] == "original-prefill-last")
            .unwrap();
        let decode = cases
            .iter()
            .find(|c| c["name"] == "original-decode-all")
            .unwrap();
        let future = cases
            .iter()
            .find(|c| c["name"] == "future-prefill-last")
            .unwrap();
        let cancel = AtomicBool::new(false);
        let mut raw = RawWriter::new("toklet-g2-deep-chain-controls-v1");
        // The existing forced-reference GPU entry materializes uncertain
        // placeholders at producer13. Every descendant must be recomputed.
        autoreleasepool(|_| {
            let mut session = seeded(&graph, &fixture, prefill);
            session.force_reference_node = Some(0);
            let before = session.caches.each_ref().map(cache_hash);
            let mut output = vec![99.; 4 * H];
            let tx = run(&mut session, &fixture, prefill, &cancel);
            let candidate = capture_states(&tx, &mut raw);
            assert_eq!(tx.session.caches.each_ref().map(cache_hash), before);
            let completed = tx.complete(&cancel).unwrap();
            expected_states(&completed, &fixture, prefill);
            assert_eq!(completed.metrics().replay_count, 1);
            assert_eq!(completed.metrics().replay_node, Some(0));
            assert_eq!(completed.metrics().reference_matrix_operations, 25);
            assert_eq!(completed.metrics().reference_commands, Some(27));
            assert_eq!(completed.metrics().reference_waits, Some(27));
            let published_states = capture_states(&completed.0, &mut raw);
            let metrics = completed.publish(Some(&mut output), &cancel).unwrap();
            check(
                &output.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                &fixture.view(&prefill["expected_output"]),
            );
            let out = raw.words(&output.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            raw.cases.push(json!({"name":"upstream-GPU-reference-request-replays-all-descendants","source_case":prefill["name"],"forced_reference_layer":13,"candidate_nodes":candidate,"final_nodes":published_states,"output":out,"metrics":metrics,"committed":session.committed(),"cache_offsets":session.caches.each_ref().map(|c|c.offset),"cache_generations":session.caches.each_ref().map(|c|c.generation)}));
        });
        // Genuine same-command GPU checkpoint producer, then forced whole-chain
        // replay. External borrowed spans have nonzero offsets and survive wait.
        autoreleasepool(|_| {
            let mut session = seeded(&graph, &fixture, last);
            session.force_reference_node = Some(0);
            let (hidden, ple) = fixture.checkpoint(last);
            let inputs = [&hidden[..], &ple[0][..], &ple[1][..], &ple[2][..]];
            let mut cursor = GUARD;
            let regions = inputs.map(|v| {
                let r = Region {
                    start: cursor,
                    len: v.len(),
                };
                cursor = r.end() + GUARD;
                r
            });
            let gpu = &context.0.gpu;
            let mut uploaded = Buffer::<u32>::new(gpu, cursor).unwrap();
            let mut produced = Buffer::<u32>::new(gpu, cursor).unwrap();
            uploaded.with_mut(|s| {
                s.fill(SENTINEL);
                for (region, input) in regions.iter().zip(inputs) {
                    for (out, x) in s[region.start..region.end()].iter_mut().zip(input) {
                        *out = x.to_bits();
                    }
                }
            });
            produced.with_mut(|s| s.fill(SENTINEL));
            let uploaded_hash = uploaded.with_slice(digest_words);
            let command = gpu.command().unwrap();
            let encoder = command.blitCommandEncoder().unwrap();
            // SAFETY: exact owned same-device disjoint initialized tracked
            // buffers; command retains both while the transaction keeps borrows.
            unsafe {
                for r in regions {
                    encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        &uploaded.raw,
                        r.start * 4,
                        &produced.raw,
                        r.start * 4,
                        r.len * 4,
                    );
                }
            }
            encoder.endEncoding();
            let prefix = fixture.prefix(last);
            let mut output = vec![99.; H];
            let mut tx = session
                .begin_resident(
                    &command,
                    &prefix,
                    regions.map(|r| r.range(&produced, r.len).unwrap()),
                    OutputRows::Last,
                    &cancel,
                )
                .unwrap();
            tx.encode_all(&cancel).unwrap();
            tx.submit_and_wait().unwrap();
            let candidate = capture_states(&tx, &mut raw);
            let completed = tx.complete(&cancel).unwrap();
            expected_states(&completed, &fixture, last);
            assert_eq!(completed.metrics().input_upload_bytes, 0);
            assert_eq!(completed.metrics().reference_commands, Some(27));
            let final_states = capture_states(&completed.0, &mut raw);
            let metrics = completed.publish(Some(&mut output), &cancel).unwrap();
            assert_eq!(metrics.blit_copies, 10);
            assert_eq!(uploaded.with_slice(digest_words), uploaded_hash);
            check(
                &output.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                &fixture.view(&last["expected_output"]),
            );
            let out = raw.words(&output.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            raw.cases.push(json!({"name":"resident-entry-checkpoint-replay-Last-keeps-all-proposals","source_case":last["name"],"candidate_nodes":candidate,"final_nodes":final_states,"output":out,"metrics":metrics,"external_GPU_blit_copies":4,"external_source_upload_bytes":inputs.iter().map(|x|x.len()*4).sum::<usize>(),"external_source_immutable":true,"consumer_queries":1,"donor13_proposal_rows":4,"committed":session.committed()}));
        });
        for (name, earlier_replay, node, slot, word, expected_poison) in [
            (
                "later-contract-after-upstream-uncertainty",
                true,
                2,
                11,
                1,
                true,
            ),
            (
                "later-GELU-proof-after-upstream-uncertainty",
                true,
                2,
                29,
                16,
                true,
            ),
            ("consumer-reserved-KV-status", false, 2, 5, 1, true),
            ("late-consumer-numeric-fatal", false, 2, 34, 1, true),
        ] {
            autoreleasepool(|_| {
                let mut session = seeded(&graph, &fixture, decode);
                if earlier_replay {
                    session.force_reference_node = Some(0);
                }
                let before = session.caches.each_ref().map(cache_hash);
                let prefix_before = session.prefix().to_vec();
                let output = vec![99.; H];
                let tx = run(&mut session, &fixture, decode, &cancel);
                tx.session.storage[node].status.with_mut(|s| s[slot] = word);
                let error = tx.complete(&cancel).err().unwrap().to_string();
                assert_eq!(session.poisoned, expected_poison);
                assert_eq!(session.metrics.reference_commands, Some(0));
                assert!(session.reset().is_err());
                failure(
                    &mut raw,
                    name,
                    &session,
                    &before,
                    &prefix_before,
                    &output,
                    &error,
                );
            });
        }
        // A later numerical fatal is speculative after the first genuine GPU
        // uncertainty and must not replace whole-chain replay with rejection.
        autoreleasepool(|_| {
            let mut session = seeded(&graph, &fixture, decode);
            session.force_reference_node = Some(0);
            let tx = run(&mut session, &fixture, decode, &cancel);
            tx.session.storage[2].status.with_mut(|s| s[34] = 1);
            let completed = tx.complete(&cancel).unwrap();
            expected_states(&completed, &fixture, decode);
            let mut output = vec![99.; H];
            let metrics = completed.publish(Some(&mut output), &cancel).unwrap();
            assert_eq!(metrics.reference_commands, Some(27));
            let out = raw.words(&output.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            raw.cases.push(json!({"name":"speculative-late-numeric-fatal-is-discarded-by-whole-replay","source_case":decode["name"],"output":out,"metrics":metrics,"committed":session.committed()}));
        });
        autoreleasepool(|_| {
            let mut session = seeded(&graph, &fixture, decode);
            let before = session.caches.each_ref().map(cache_hash);
            let prefix_before = session.prefix().to_vec();
            let output = vec![99.; H];
            let tx = run(&mut session, &fixture, decode, &cancel);
            let inactive = 8 * P;
            tx.session.storage[2].intervals[1].with_mut(|s| {
                assert_eq!(s[inactive], SENTINEL);
                s[inactive] = 0;
            });
            let error = tx.complete(&cancel).err().unwrap().to_string();
            assert!(session.poisoned);
            failure(
                &mut raw,
                "inactive-interval-guard-fails-before-publication",
                &session,
                &before,
                &prefix_before,
                &output,
                &error,
            );
        });
        autoreleasepool(|_| {
            let mut session = seeded(&graph, &fixture, decode);
            let before = session.caches.each_ref().map(cache_hash);
            let prefix_before = session.prefix().to_vec();
            let mut output = vec![99.; H];
            let tx = run(&mut session, &fixture, decode, &cancel);
            let mut completed = tx.complete(&cancel).unwrap();
            completed.0.frame.cache_generations[1] ^= 1;
            let error = completed
                .publish(Some(&mut output), &cancel)
                .err()
                .unwrap()
                .to_string();
            assert!(session.poisoned);
            failure(
                &mut raw,
                "second-cache-stale-frame-blocks-first-cache-write",
                &session,
                &before,
                &prefix_before,
                &output,
                &error,
            );
        });
        for at_complete in [true, false] {
            autoreleasepool(|_| {
                let mut session = seeded(&graph, &fixture, decode);
                let before = session.caches.each_ref().map(cache_hash);
                let prefix_before = session.prefix().to_vec();
                let mut output = vec![99.; H];
                let tx = run(&mut session, &fixture, decode, &cancel);
                let error = if at_complete {
                    cancel.store(true, Ordering::Release);
                    tx.complete(&cancel).err().unwrap().to_string()
                } else {
                    let completed = tx.complete(&cancel).unwrap();
                    cancel.store(true, Ordering::Release);
                    completed
                        .publish(Some(&mut output), &cancel)
                        .err()
                        .unwrap()
                        .to_string()
                };
                cancel.store(false, Ordering::Release);
                assert!(!session.poisoned);
                failure(
                    &mut raw,
                    if at_complete {
                        "canceled-after-GPU-completion"
                    } else {
                        "canceled-after-validation-before-atomic-copy"
                    },
                    &session,
                    &before,
                    &prefix_before,
                    &output,
                    &error,
                );
                let (hidden, ple) = fixture.checkpoint(decode);
                let prefix = fixture.prefix(decode);
                session
                    .evaluate(
                        &prefix,
                        &hidden,
                        [&ple[0], &ple[1], &ple[2]],
                        OutputRows::Last,
                        &mut output,
                        &cancel,
                    )
                    .unwrap();
                assert_eq!(session.committed(), 4);
            });
        }
        autoreleasepool(|_| {
            let mut session = seeded(&graph, &fixture, decode);
            let before = session.caches.each_ref().map(cache_hash);
            let prefix_before = session.prefix().to_vec();
            let output = vec![99.; H];
            let (hidden, ple) = fixture.checkpoint(decode);
            let prefix = fixture.prefix(decode);
            let mut tx = session
                .begin(
                    &prefix,
                    &hidden,
                    [&ple[0], &ple[1], &ple[2]],
                    OutputRows::All,
                    &cancel,
                )
                .unwrap();
            tx.encode_all(&cancel).unwrap();
            let command = tx.command.clone();
            command.enqueue();
            drop(tx);
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            assert!(command.error().is_none());
            assert!(session.poisoned);
            failure(
                &mut raw,
                "queued-drop-drains-shared-command-without-publication",
                &session,
                &before,
                &prefix_before,
                &output,
                "explicit unvalidated drop",
            );
        });
        autoreleasepool(|_| {
            let mut session = seeded(&graph, &fixture, decode);
            let before = session.caches.each_ref().map(cache_hash);
            let prefix_before = session.prefix().to_vec();
            let output = vec![99.; H];
            let (hidden, ple) = fixture.checkpoint(decode);
            let prefix = fixture.prefix(decode);
            cancel.store(true, Ordering::Release);
            assert!(
                session
                    .begin(
                        &prefix,
                        &hidden,
                        [&ple[0], &ple[1], &ple[2]],
                        OutputRows::All,
                        &cancel
                    )
                    .is_err()
            );
            cancel.store(false, Ordering::Release);
            let mut changed = prefix.clone();
            changed[0] ^= 1;
            assert!(
                session
                    .begin(
                        &changed,
                        &hidden,
                        [&ple[0], &ple[1], &ple[2]],
                        OutputRows::All,
                        &cancel
                    )
                    .is_err()
            );
            assert!(
                session
                    .begin(
                        &prefix[..3],
                        &hidden,
                        [&ple[0], &ple[1], &ple[2]],
                        OutputRows::All,
                        &cancel
                    )
                    .is_err()
            );
            let mut bad = hidden.clone();
            bad[0] = f32::NAN;
            assert!(
                session
                    .begin(
                        &prefix,
                        &bad,
                        [&ple[0], &ple[1], &ple[2]],
                        OutputRows::All,
                        &cancel
                    )
                    .is_err()
            );
            let mut tx = session
                .begin(
                    &prefix,
                    &hidden,
                    [&ple[0], &ple[1], &ple[2]],
                    OutputRows::All,
                    &cancel,
                )
                .unwrap();
            assert!(tx.session.reset().is_err());
            tx.frame.owner ^= 1;
            assert!(tx.encode_all(&cancel).is_err());
            drop(tx);
            assert!(!session.poisoned);
            failure(
                &mut raw,
                "host-preflight-exact-prefix-cancel-finite-stale-owner",
                &session,
                &before,
                &prefix_before,
                &output,
                "five preflight rejects plus busy reset/stale owner",
            );
        });
        autoreleasepool(|_| {
            let mut session = seeded(&graph, &fixture, decode);
            let before = session.caches.each_ref().map(cache_hash);
            let prefix_before = session.prefix().to_vec();
            let output = vec![99.; H];
            let (hidden, ple) = fixture.checkpoint(decode);
            let prefix = fixture.prefix(decode);
            let gpu = &context.0.gpu;
            let mut input = Buffer::<u32>::new(gpu, H + 3 * P + 4 * GUARD).unwrap();
            let mut cursor = 0;
            let spans = [&hidden[..], &ple[0][..], &ple[1][..], &ple[2][..]].map(|v| {
                let r = Region {
                    start: cursor,
                    len: v.len(),
                };
                cursor = r.end() + GUARD;
                r
            });
            input.with_mut(|s| {
                s.fill(SENTINEL);
                for (r, v) in spans
                    .iter()
                    .zip([&hidden[..], &ple[0][..], &ple[1][..], &ple[2][..]])
                {
                    for (dst, x) in s[r.start..r.end()].iter_mut().zip(v) {
                        *dst = x.to_bits();
                    }
                }
            });
            let unretained = gpu.queue.commandBufferWithUnretainedReferences().unwrap();
            assert!(
                session
                    .begin_resident(
                        &unretained,
                        &prefix,
                        spans.map(|r| r.range(&input, r.len).unwrap()),
                        OutputRows::All,
                        &cancel
                    )
                    .is_err()
            );
            let queued = gpu.command().unwrap();
            queued.enqueue();
            assert!(
                session
                    .begin_resident(
                        &queued,
                        &prefix,
                        spans.map(|r| r.range(&input, r.len).unwrap()),
                        OutputRows::All,
                        &cancel
                    )
                    .is_err()
            );
            queued.commit();
            queued.waitUntilCompleted();
            let command = gpu.command().unwrap();
            let aliased = session.storage[0].arena.raw.clone();
            let mut ranges = spans.map(|r| r.range(&input, r.len).unwrap());
            ranges[0] = BufferRange {
                raw: &aliased,
                offset: 0,
                len: H,
            };
            assert!(
                session
                    .begin_resident(&command, &prefix, ranges, OutputRows::All, &cancel)
                    .is_err()
            );
            command.commit();
            command.waitUntilCompleted();
            assert!(!session.poisoned);
            failure(
                &mut raw,
                "resident-preflight-unretained-queued-and-owned-arena-alias",
                &session,
                &before,
                &prefix_before,
                &output,
                "three real resource rejections",
            );
        });
        autoreleasepool(|_| {
            let mut session = seeded(&graph, &fixture, decode);
            let before = session.caches.each_ref().map(cache_hash);
            let prefix_before = session.prefix().to_vec();
            let output = vec![99.; H];
            let (mut hidden, ple) = fixture.checkpoint(decode);
            hidden[0] = f32::NAN;
            let gpu = &context.0.gpu;
            let mut input = Buffer::<u32>::new(gpu, H + 3 * P + 4 * GUARD).unwrap();
            let mut cursor = 0;
            let spans = [&hidden[..], &ple[0][..], &ple[1][..], &ple[2][..]].map(|v| {
                let r = Region {
                    start: cursor,
                    len: v.len(),
                };
                cursor = r.end() + GUARD;
                r
            });
            input.with_mut(|s| {
                s.fill(SENTINEL);
                for (r, v) in spans
                    .iter()
                    .zip([&hidden[..], &ple[0][..], &ple[1][..], &ple[2][..]])
                {
                    for (dst, x) in s[r.start..r.end()].iter_mut().zip(v) {
                        *dst = x.to_bits();
                    }
                }
            });
            let command = gpu.command().unwrap();
            let prefix = fixture.prefix(decode);
            let mut tx = session
                .begin_resident(
                    &command,
                    &prefix,
                    spans.map(|r| r.range(&input, r.len).unwrap()),
                    OutputRows::All,
                    &cancel,
                )
                .unwrap();
            tx.encode_all(&cancel).unwrap();
            tx.submit_and_wait().unwrap();
            let error = tx.complete(&cancel).err().unwrap().to_string();
            assert!(session.poisoned);
            failure(
                &mut raw,
                "resident-nonfinite-entry-is-fatal-without-publication",
                &session,
                &before,
                &prefix_before,
                &output,
                &error,
            );
        });
        // Same admitted storage changes3rows ->1row ->reset ->4rows/Last.
        // All expectations are causal slices of the frozen deep states.
        autoreleasepool(|_| {
            let mut session = seeded(&graph, &fixture, prefill);
            let (mut hidden, mut ple) = fixture.checkpoint(prefill);
            hidden.truncate(3 * H);
            for p in &mut ple {
                p.truncate(3 * P);
            }
            let prefix = fixture.prefix(prefill);
            let mut first = vec![99.; 3 * H];
            session
                .evaluate(
                    &prefix[..3],
                    &hidden,
                    [&ple[0], &ple[1], &ple[2]],
                    OutputRows::All,
                    &mut first,
                    &cancel,
                )
                .unwrap();
            let (hidden, ple) = fixture.checkpoint(decode);
            let mut output = vec![99.; H];
            session
                .evaluate(
                    &prefix,
                    &hidden,
                    [&ple[0], &ple[1], &ple[2]],
                    OutputRows::Last,
                    &mut output,
                    &cancel,
                )
                .unwrap();
            check(
                &output.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                &fixture.view(&decode["expected_output"]),
            );
            let original = raw.words(&output.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            session.reset().unwrap();
            assert_eq!(session.committed(), 0);
            assert!(session.output().is_none());
            let (hidden, ple) = fixture.checkpoint(future);
            let prefix = fixture.prefix(future);
            let metrics = session
                .evaluate(
                    &prefix,
                    &hidden,
                    [&ple[0], &ple[1], &ple[2]],
                    OutputRows::Last,
                    &mut output,
                    &cancel,
                )
                .unwrap();
            check(
                &output.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                &fixture.view(&future["expected_output"]),
            );
            let future_out = raw.words(&output.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            raw.cases.push(json!({"name":"changing-shape-reuse-and-changed-prefix-reset","original_decode_output":original,"future_Last_output":future_out,"source_cases":[decode["name"],future["name"]],"metrics":metrics,"committed":session.committed(),"cache_generations":session.caches.each_ref().map(|c|c.generation),"prefix":session.prefix()}));
        });
        assert_eq!(weights_hash(&graph), weights_before);
        raw.finish();
    });
}

#[test]
#[ignore = "explicit32 warmed deep-chain evaluations; Rust allocations only, no native/performance claim"]
fn warmed_deep_chain_reuses_owned_storage_without_rust_allocations() {
    autoreleasepool(|_| {
        use crate::model::gemma_reference::tests::Measurement;
        let fixture = Fixture::load();
        let case = fixture.protocol["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == "original-prefill-last")
            .unwrap();
        let context = Bf16Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        let source = source(&context);
        let table = table(&context);
        let pipelines = core_pipelines(&context, &table, PIPELINE_POLICY_BYTES);
        let options = Options {
            capacity: 512,
            max_rows: 4,
        };
        let graph = ResidentGraph::new(
            &source,
            &context,
            &pipelines,
            &table,
            options,
            ResidentGraph::planned_bytes(options).unwrap(),
        )
        .unwrap();
        let mut session = seeded(&graph, &fixture, case);
        let (hidden, ple) = fixture.checkpoint(case);
        let prefix = fixture.prefix(case);
        let cancel = AtomicBool::new(false);
        let mut output = vec![99.; H];
        autoreleasepool(|_| {
            session
                .evaluate(
                    &prefix,
                    &hidden,
                    [&ple[0], &ple[1], &ple[2]],
                    OutputRows::Last,
                    &mut output,
                    &cancel,
                )
                .unwrap();
        });
        let measurement = Measurement::start();
        for _ in 0..32 {
            autoreleasepool(|_| {
                session.reset().unwrap();
                let m = session
                    .evaluate(
                        &prefix,
                        &hidden,
                        [&ple[0], &ple[1], &ple[2]],
                        OutputRows::Last,
                        &mut output,
                        &cancel,
                    )
                    .unwrap();
                assert_eq!(m.replay_count, 0);
                assert_eq!(m.candidate_commands, 1);
                assert_eq!(m.candidate_waits, 1);
            });
        }
        let allocations = measurement.finish();
        assert_eq!(allocations, 0);
        check(
            &output.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            &fixture.view(&case["expected_output"]),
        );
        let mut raw = RawWriter::new("toklet-g2-deep-chain-warm-v1");
        let out = raw.words(&output.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        raw.cases.push(json!({"name":"warm32-deep-chain","source_case":case["name"],"iterations":32,"Rust_allocations":allocations,"native_allocations_measured":false,"measured_commands":32,"measured_waits":32,"source_weights_shared":true,"output":out,"metrics_last":session.metrics().unwrap(),"committed":session.committed(),"cache_generations":session.caches.each_ref().map(|c|c.generation)}));
        raw.finish();
    });
}

// Diagnostic summaries retain the original gate. They do not turn a failed
// historical comparison into admission or change the candidate/replay policy.
fn historical_summary(actual: &[u32], expected: &[f32]) -> Value {
    assert_eq!(actual.len(), expected.len());
    let mut failures = 0;
    let mut max_fraction = 0_f64;
    let mut max_abs_error = 0_f64;
    let mut first = Vec::new();
    for (index, (&word, &reference)) in actual.iter().zip(expected).enumerate() {
        let value = f32::from_bits(word);
        let delta = (f64::from(value) - f64::from(reference)).abs();
        let fraction = delta / (1e-5 + 1e-4 * f64::from(reference).abs());
        if !value.is_finite() || !reference.is_finite() || fraction > 1. {
            failures += 1;
            if first.len() < 8 {
                first.push(json!({"index":index,"actual_bits":word,"expected_bits":reference.to_bits(),"gate_fraction":fraction}));
            }
        }
        max_fraction = max_fraction.max(fraction);
        max_abs_error = max_abs_error.max(delta);
    }
    json!({"values":actual.len(),"failing_values":failures,"max_gate_fraction":max_fraction,"max_abs_error":max_abs_error,"first_failures":first})
}

#[test]
#[ignore = "explicit diagnostic of unchanged whole reference chain against every frozen historical state; retains failure captures"]
fn unchanged_reference_chain_from_same_checkpoint_checks_all_frozen_states() {
    autoreleasepool(|_| {
        let fixture = Fixture::load();
        let context = Bf16Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        let source = source(&context);
        let table = table(&context);
        let pipelines = core_pipelines(&context, &table, PIPELINE_POLICY_BYTES);
        let options = Options {
            capacity: 512,
            max_rows: 4,
        };
        let graph = ResidentGraph::new(
            &source,
            &context,
            &pipelines,
            &table,
            options,
            ResidentGraph::planned_bytes(options).unwrap(),
        )
        .unwrap();
        let original_weights = weights_hash(&graph);
        let cancel = AtomicBool::new(false);
        let mut raw = RawWriter::new("toklet-g2-deep-chain-reference-diagnostic-v1");
        let mut passed = true;
        for case in fixture.protocol["cases"].as_array().unwrap() {
            autoreleasepool(|_| {
                let mut session = seeded(&graph, &fixture, case);
                assert!(ledger(&graph, &session) <= LIMIT);
                let cache_before = session.caches.each_ref().map(cache_hash);
                let generations_before = session.caches.each_ref().map(|c| c.generation);
                let committed_before = session.committed();
                let (hidden, ple) = fixture.checkpoint(case);
                let prefix = fixture.prefix(case);
                // Exercise the same checked host staging; dropping this empty
                // transaction drains exactly one no-op command, counted apart
                // from the reference's real 27 synchronous tile calls.
                let staged = session
                    .begin(
                        &prefix,
                        &hidden,
                        [&ple[0], &ple[1], &ple[2]],
                        mode(case),
                        &cancel,
                    )
                    .unwrap();
                let frame = staged.frame;
                drop(staged);
                assert_eq!(session.metrics.candidate_commands, 1);
                assert_eq!(session.metrics.candidate_waits, 1);
                assert!(!session.poisoned && !session.busy);
                let started = Instant::now();
                session
                    .reference
                    .run(
                        &graph,
                        &session.entry,
                        &session.entry_regions,
                        &session.caches,
                        &frame,
                    )
                    .unwrap();
                let seconds = started.elapsed().as_secs_f64();
                assert_eq!(session.reference.matrix_operations, 25);
                assert_eq!(session.reference.commands, 27);
                assert_eq!(session.reference.waits, 27);
                assert!(session.reference.counts_known);
                let mut nodes = Vec::new();
                for (node, plan) in graph.plans.iter().enumerate() {
                    let mut states = Vec::new();
                    let mut summaries = serde_json::Map::new();
                    for state in STATES {
                        let n = plan.layout.active(state, frame.nodes[node]);
                        if n == 0 {
                            continue;
                        }
                        let words = session.reference.states[node][state as usize][..n]
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>();
                        let mut desc = raw.words(&words);
                        desc["state"] = json!(state as usize);
                        desc["name"] = json!(format!("{state:?}"));
                        states.push(desc);
                    }
                    for (suffix, state) in CAPTURED {
                        if let Some(desc) = case["nodes"][node]["expected_states"].get(suffix) {
                            let expected = fixture.view(desc);
                            let words = session.reference.states[node][state as usize]
                                [..expected.len()]
                                .iter()
                                .map(|v| v.to_bits())
                                .collect::<Vec<_>>();
                            let summary = historical_summary(&words, &expected);
                            if summary["failing_values"] != 0 {
                                passed = false;
                                eprintln!(
                                    "reference historical case{} layer{} state{suffix}: {summary}",
                                    case["name"], LAYERS[node]
                                );
                            }
                            summaries.insert(suffix.into(), summary);
                        }
                    }
                    nodes.push(json!({"layer":LAYERS[node],"frame":frame.nodes[node],"D":plan.layout.d,"FF":plan.layout.ff,"states":states,"historical":summaries}));
                }
                assert_eq!(session.caches.each_ref().map(cache_hash), cache_before);
                assert_eq!(
                    session.caches.each_ref().map(|c| c.generation),
                    generations_before
                );
                assert_eq!(session.committed(), committed_before);
                assert!(session.published_rows == 0);
                raw.cases.push(json!({"name":case["name"],"nodes":nodes,"cache_before_sha256":cache_before,"cache_after_sha256":session.caches.each_ref().map(cache_hash),"cache_generations_before":generations_before,"committed_before":committed_before,"reference_matrix_operations":session.reference.matrix_operations,"reference_commands":session.reference.commands,"reference_waits":session.reference.waits,"reference_input_bytes":session.reference.input_bytes,"reference_readback_bytes":session.reference.readback_bytes,"checkpoint_readback_bytes":session.reference.checkpoint_bytes,"empty_staging_commands":1,"empty_staging_waits":1,"candidate_numerical_dispatches":0,"reference_seconds":seconds,"published":false}));
            });
        }
        assert_eq!(weights_hash(&graph), original_weights);
        // Preserve complete labeled captures and the failed gate summaries
        // before failing a diagnostic test. Existing qualification logs remain
        // success-only, and no historical expectation is replaced.
        raw.finish_with_status(passed);
        assert!(
            passed,
            "unchanged whole reference chain failed historical gates; complete diagnostic capture retained"
        );
    });
}

fn inject_rms_reference(tx: &mut GraphTransaction<'_, '_, '_>) {
    tx.session.storage[0].status.with_mut(|s| s[0] = 2);
    let region = tx.session.graph.plans[0].layout.rms_plan().0[0].unwrap();
    tx.session.storage[0]
        .rms_reasons
        .with_mut(|s| s[region.start] |= 16);
}
fn rms_failure_storage(session: &ResidentSession<'_, '_>, raw: &mut RawWriter) {
    let reasons = session
        .storage
        .iter()
        .map(|s| s.rms_reasons.with_slice(|words| raw.words(words)))
        .collect::<Vec<_>>();
    raw.cases.last_mut().unwrap()["rms_reason_storage"] = json!(reasons);
}

#[test]
#[ignore = "explicit high-accuracy RMS uncertainty/fatal/global-contract/atomic-publication qualification"]
fn high_accuracy_rms_reference_status_and_contracts_preserve_prefix() {
    autoreleasepool(|_| {
        let fixture = Fixture::load();
        let context = Bf16Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        let source = source(&context);
        let table = table(&context);
        let pipelines = core_pipelines(&context, &table, PIPELINE_POLICY_BYTES);
        let options = Options {
            capacity: 512,
            max_rows: 4,
        };
        let graph = ResidentGraph::new(
            &source,
            &context,
            &pipelines,
            &table,
            options,
            ResidentGraph::planned_bytes(options).unwrap(),
        )
        .unwrap();
        let decode = &fixture.protocol["cases"][2];
        let cancel = AtomicBool::new(false);
        let mut raw = RawWriter::new("toklet-g2-deep-chain-rms-controls-v1");
        for (name, natural, late_fatal) in [
            (
                "real-subnormal-input-requests-whole-chain-reference",
                true,
                false,
            ),
            (
                "RMS2-status-replays-all-canonical-descendants",
                false,
                false,
            ),
            ("later-RMS-fatal-is-speculative-after-RMS2", false, true),
        ] {
            autoreleasepool(|_| {
                let mut session = seeded(&graph, &fixture, decode);
                assert!(ledger(&graph, &session) <= LIMIT);
                let before = session.caches.each_ref().map(cache_hash);
                let (mut hidden, ple) = fixture.checkpoint(decode);
                if natural {
                    hidden[0] = f32::from_bits(1);
                }
                let prefix = fixture.prefix(decode);
                let mut tx = session
                    .begin(
                        &prefix,
                        &hidden,
                        [&ple[0], &ple[1], &ple[2]],
                        mode(decode),
                        &cancel,
                    )
                    .unwrap();
                tx.encode_all(&cancel).unwrap();
                tx.submit_and_wait().unwrap();
                if natural {
                    assert_eq!(tx.session.storage[0].status.with_slice(|s| s[0]), 2);
                    let region = graph.plans[0].layout.rms_plan().0[0].unwrap();
                    assert_ne!(
                        tx.session.storage[0]
                            .rms_reasons
                            .with_slice(|s| s[region.start])
                            & 1,
                        0
                    );
                } else {
                    inject_rms_reference(&mut tx);
                }
                if late_fatal {
                    tx.session.storage[2].status.with_mut(|s| s[25] = 1);
                }
                let candidate = capture_states(&tx, &mut raw);
                assert_eq!(tx.session.caches.each_ref().map(cache_hash), before);
                let completed = tx.complete(&cancel).unwrap();
                assert_eq!(completed.metrics().replay_slot, Some(0));
                assert_eq!(completed.metrics().replay_node, Some(0));
                assert_eq!(completed.metrics().reference_commands, Some(27));
                assert_eq!(completed.metrics().reference_waits, Some(27));
                if !natural {
                    expected_states(&completed, &fixture, decode);
                }
                // The natural outside-domain input has no historical graph
                // expectation. Check complete scalar replay upload separately;
                // the two canonical cases still check all untouched history.
                for (node, plan) in graph.plans.iter().enumerate() {
                    for state in STATES {
                        let n = plan.layout.active(state, completed.0.frame.nodes[node]);
                        if n == 0 {
                            continue;
                        }
                        let expected =
                            &completed.0.session.reference.states[node][state as usize][..n];
                        completed
                            .with_state(node, state, |actual| {
                                assert!(
                                    actual.iter().zip(expected).all(|(&a, &e)| a == e.to_bits())
                                );
                            })
                            .unwrap();
                    }
                }
                let final_states = capture_states(&completed.0, &mut raw);
                let mut output = vec![99.; H];
                let metrics = completed.publish(Some(&mut output), &cancel).unwrap();
                assert_eq!(session.committed(), 4);
                assert_eq!(session.caches.each_ref().map(|c| c.generation), [2, 2]);
                let out = raw.words(&output.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
                raw.cases.push(json!({"name":name,"source_case":decode["name"],"natural_subnormal_entry":natural,"candidate_nodes":candidate,"final_nodes":final_states,"output":out,"metrics":metrics,"cache_before_sha256":before,"committed":session.committed(),"cache_generations":session.caches.each_ref().map(|c|c.generation),"historical_states_checked":!natural,"whole_reference_upload_checked":true}));
            });
        }
        for (name, kind) in [
            ("unknown-later-RMS-reason-precedes-RMS2-replay", 0),
            ("consumer-reserved-RMS-status2-is-contract-fatal", 1),
            ("first-RMS-fatal-precedes-later-RMS2", 2),
            ("inactive-RMS-reason-guard-precedes-RMS2-replay", 3),
        ] {
            autoreleasepool(|_| {
                let mut session = seeded(&graph, &fixture, decode);
                let before = session.caches.each_ref().map(cache_hash);
                let prefix = session.prefix().to_vec();
                let output = vec![99.; H];
                let mut tx = run(&mut session, &fixture, decode, &cancel);
                inject_rms_reference(&mut tx);
                match kind {
                    0 => {
                        let region = graph.plans[2].layout.rms_plan().0[0].unwrap();
                        tx.session.storage[2]
                            .rms_reasons
                            .with_mut(|s| s[region.start] |= 1 << 20);
                    }
                    1 => tx.session.storage[2].status.with_mut(|s| s[6] = 2),
                    2 => {
                        tx.session.storage[0].status.with_mut(|s| s[0] = 1);
                        tx.session.storage[2].status.with_mut(|s| s[0] = 2);
                    }
                    3 => {
                        let region = graph.plans[2].layout.rms_plan().0[0].unwrap();
                        let inactive = region.start + tx.frame.nodes[2].rows;
                        tx.session.storage[2].rms_reasons.with_mut(|s| {
                            assert_eq!(s[inactive], SENTINEL);
                            s[inactive] = 0;
                        });
                    }
                    _ => unreachable!(),
                }
                let error = tx.complete(&cancel).err().unwrap().to_string();
                assert!(session.poisoned);
                assert_eq!(session.metrics.reference_commands, Some(0));
                failure(&mut raw, name, &session, &before, &prefix, &output, &error);
                rms_failure_storage(&session, &mut raw);
            });
        }
        autoreleasepool(|_| {
            let mut session = seeded(&graph, &fixture, decode);
            let before = session.caches.each_ref().map(cache_hash);
            let prefix = session.prefix().to_vec();
            let mut tx = run(&mut session, &fixture, decode, &cancel);
            inject_rms_reference(&mut tx);
            let completed = tx.complete(&cancel).unwrap();
            expected_states(&completed, &fixture, decode);
            let region = graph.plans[1].layout.rms_plan().0[0].unwrap();
            completed.0.session.storage[1]
                .rms_reasons
                .with_mut(|s| s[region.start] ^= 16);
            let mut output = vec![99.; H];
            let error = completed
                .publish(Some(&mut output), &cancel)
                .err()
                .unwrap()
                .to_string();
            assert!(session.poisoned);
            failure(
                &mut raw,
                "RMS-reasons-changed-after-validation-block-both-cache-copies",
                &session,
                &before,
                &prefix,
                &output,
                &error,
            );
            rms_failure_storage(&session, &mut raw);
        });
        assert_eq!(raw.cases.len(), 8);
        raw.finish();
    });
}
