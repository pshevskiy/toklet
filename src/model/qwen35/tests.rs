use super::*;
mod context;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal;
mod nine_b_real;
mod reset;
mod two_b;
mod untied;
use crate::mlx::affine::{Layout, MetadataDtype};
use crate::model::gemma_reference::tests::Measurement;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    sync::{
        OnceLock,
        atomic::{AtomicBool, Ordering},
    },
};

const MANIFEST: &str = include_str!("../../../tests/fixtures/qwen35-cpu-graph-v1/manifest.json");
const WEIGHTS: &[u8] = include_bytes!("../../../tests/fixtures/qwen35-cpu-graph-v1/weights.bin");
const REFERENCE: &[u8] =
    include_bytes!("../../../tests/fixtures/qwen35-cpu-graph-v1/reference.bin");

fn manifest() -> &'static Value {
    static M: OnceLock<Value> = OnceLock::new();
    M.get_or_init(|| {
        assert_eq!(
            format!("{:x}", Sha256::digest(MANIFEST.as_bytes())),
            "4defb0e37b0ddec53c53cd94bb5ecadbd7f92b0f1072f4c185e92122330599b8"
        );
        let m: Value = serde_json::from_str(MANIFEST).unwrap();
        assert_eq!(m["format"], "toklet-qwen35-cpu-graph-v1");
        assert_eq!(m["gate"]["atol"].as_f64(), Some(1e-5));
        assert_eq!(m["gate"]["rtol"].as_f64(), Some(1e-4));
        assert_eq!(m["fixture_only"], true);
        assert_eq!(
            format!("{:x}", Sha256::digest(WEIGHTS)),
            m["weights_sha256"]
        );
        assert_eq!(
            format!("{:x}", Sha256::digest(REFERENCE)),
            m["reference_sha256"]
        );
        assert_eq!(m["weights_bytes"].as_u64().unwrap() as usize, WEIGHTS.len());
        assert_eq!(
            m["reference_bytes"].as_u64().unwrap() as usize,
            REFERENCE.len()
        );
        m
    })
}
fn bytes(v: &Value) -> &'static [u8] {
    let start = v["offset"].as_u64().unwrap() as usize;
    let count = v["bytes"].as_u64().unwrap() as usize;
    let b = &WEIGHTS[start..start + count];
    assert_eq!(format!("{:x}", Sha256::digest(b)), v["sha256"]);
    b
}
pub(super) struct Fixture {
    vectors: BTreeMap<String, Vec<f32>>,
}
impl Fixture {
    pub(super) fn new() -> Self {
        let vectors = manifest()["weights"]
            .as_object()
            .unwrap()
            .iter()
            .filter(|(_, v)| v["kind"] == "bf16")
            .map(|(name, v)| {
                (
                    name.clone(),
                    bytes(&v["data"])
                        .chunks_exact(2)
                        .map(|p| {
                            f32::from_bits(
                                u32::from(u16::from_le_bytes(p.try_into().unwrap())) << 16,
                            )
                        })
                        .collect(),
                )
            })
            .collect();
        Self { vectors }
    }
    fn matrix(&self, name: &str) -> Matrix<'_> {
        let v = &manifest()["weights"][name];
        let shape = v["shape"].as_array().unwrap();
        Matrix::from_le_bytes(
            Layout::new(
                shape[0].as_u64().unwrap() as usize,
                shape[1].as_u64().unwrap() as usize,
                4,
                64,
                MetadataDtype::BF16,
            )
            .unwrap(),
            bytes(&v["codes"]),
            bytes(&v["scales"]),
            bytes(&v["biases"]),
        )
        .unwrap()
    }
    fn inputs(&self) -> (Config, Weights<'_>) {
        let c: Config = serde_json::from_value(manifest()["geometry"].clone()).unwrap();
        let mut layers = Vec::new();
        let p = "language_model.model.";
        for i in 0..c.layers {
            let base = format!("{p}layers.{i}.");
            let mat = |s: &str| self.matrix(&format!("{base}{s}.weight"));
            let vec = |s: &str| self.vectors[&format!("{base}{s}")].as_slice();
            let mixing = if i % 4 == 3 {
                MixingWeights::Attention(AttentionWeights {
                    q_gate: mat("self_attn.q_proj"),
                    k: mat("self_attn.k_proj"),
                    v: mat("self_attn.v_proj"),
                    out: mat("self_attn.o_proj"),
                    q_norm: vec("self_attn.q_norm.weight"),
                    k_norm: vec("self_attn.k_norm.weight"),
                })
            } else {
                MixingWeights::Delta(DeltaWeights {
                    qkv: mat("linear_attn.in_proj_qkv"),
                    z: mat("linear_attn.in_proj_z"),
                    a: mat("linear_attn.in_proj_a"),
                    b: mat("linear_attn.in_proj_b"),
                    out: mat("linear_attn.out_proj"),
                    convolution: vec("linear_attn.conv1d.weight"),
                    a_log: vec("linear_attn.A_log"),
                    dt_bias: vec("linear_attn.dt_bias"),
                    norm: vec("linear_attn.norm.weight"),
                })
            };
            layers.push(LayerWeights {
                input_norm: vec("input_layernorm.weight"),
                post_norm: vec("post_attention_layernorm.weight"),
                gate: mat("mlp.gate_proj"),
                up: mat("mlp.up_proj"),
                down: mat("mlp.down_proj"),
                mixing,
            });
        }
        (
            c,
            Weights {
                embedding: self.matrix(&format!("{p}embed_tokens.weight")),
                final_norm: &self.vectors[&format!("{p}norm.weight")],
                layers,
            },
        )
    }
    pub(super) fn model(&self) -> Model<'_> {
        let (c, w) = self.inputs();
        let m = Model::new(c, w).unwrap();
        assert_eq!(
            m.learned_bytes,
            manifest()["backing_bytes"].as_u64().unwrap() as usize
        );
        m
    }
}
fn options() -> Options {
    Options {
        capacity: 17,
        chunk: 17,
        position_offset: 0,
        memory_limit_bytes: MEMORY_LIMIT,
    }
}
pub(super) fn tokens() -> Vec<u32> {
    manifest()["cases"][1]["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect()
}
fn values(row: &Value, profile: &str) -> Vec<f64> {
    let start = row["offset"].as_u64().unwrap() as usize;
    let count = row["count"].as_u64().unwrap() as usize;
    let width = if profile == "ideal_fp64" { 8 } else { 4 };
    let raw = &REFERENCE[start..start + count * width];
    assert_eq!(raw.len(), row["bytes"].as_u64().unwrap() as usize);
    assert_eq!(format!("{:x}", Sha256::digest(raw)), row["sha256"]);
    if width == 8 {
        raw.chunks_exact(8)
            .map(|p| f64::from_le_bytes(p.try_into().unwrap()))
            .collect()
    } else {
        raw.chunks_exact(4)
            .map(|p| f64::from(f32::from_le_bytes(p.try_into().unwrap())))
            .collect()
    }
}
fn key(token: usize, layer: Option<usize>, stage: &str) -> (usize, Option<usize>, String) {
    (token, layer, stage.into())
}
fn observed_key(o: &Observation<'_>) -> (usize, Option<usize>, String) {
    key(
        o.token,
        o.layer,
        &o.head
            .map_or_else(|| o.stage.to_owned(), |h| format!("{}.{h}", o.stage)),
    )
}
fn greedy(v: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}
pub(super) fn assert_float_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "float bits differ at{index}");
    }
}
fn close(actual: &[f32], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    let mut max = 0.0f64;
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let ratio = (f64::from(a) - e).abs() / (1e-5 + 1e-4 * e.abs());
        assert!(
            a.is_finite() && ratio <= 1.,
            "index{i}: {a} vs {e}, ratio{ratio}"
        );
        max = max.max(ratio);
    }
    max
}
pub(super) fn snapshot<M: MatrixStorage, E: Execution<M>>(s: &Session<'_, '_, M, E>) -> String {
    let mut h = Sha256::new();
    for &v in s.history() {
        h.update(v.to_le_bytes());
    }
    if let Some(row) = s.logits() {
        for &v in row {
            h.update(v.to_le_bytes());
        }
    }
    for d in &s.delta {
        for v in d.rnn.committed().iter().chain(d.history.committed()) {
            h.update(v.to_le_bytes());
        }
    }
    let c = s.model.config;
    for k in &s.kv {
        for head in 0..c.kv_heads {
            let start = head * s.options.capacity * c.head_dim;
            for v in k.k[start..start + s.history().len() * c.head_dim]
                .iter()
                .chain(&k.v[start..start + s.history().len() * c.head_dim])
            {
                h.update(v.to_le_bytes());
            }
        }
    }
    format!("{:x}", h.finalize())
}

#[test]
fn independent_profiles_cover_every_stage_state_and_logit_across_chunks() {
    let fixture = Fixture::new();
    let model = fixture.model();
    let mut compared = 0;
    let mut max = 0.0f64;
    for case in manifest()["cases"].as_array().unwrap() {
        let ids: Vec<_> = case["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let offset = case["position_offset"].as_u64().unwrap() as usize;
        let mut expected: BTreeMap<_, Vec<_>> = BTreeMap::new();
        for profile in ["ideal_fp64", "staged_fp32"] {
            for row in case["profiles"][profile].as_array().unwrap() {
                let k = key(
                    row["token"].as_u64().unwrap() as usize,
                    row["layer"].as_u64().map(|v| v as usize),
                    row["stage"].as_str().unwrap(),
                );
                expected.entry(k).or_default().push(values(row, profile));
            }
        }
        for chunk in [1, 3, 17] {
            let mut s = model
                .session(Options {
                    position_offset: offset,
                    ..options()
                })
                .unwrap();
            let mut seen = BTreeMap::new();
            for input in ids.chunks(chunk) {
                s.forward_observed(input, &AtomicBool::new(false), &mut |o| {
                    if o.stage == "before_commit" {
                        assert!(o.values.is_empty());
                        return Ok(());
                    }
                    assert_eq!(o.position, offset + o.token);
                    let k = observed_key(&o);
                    let references = &expected[&k];
                    assert_eq!(references.len(), 2);
                    for reference in references {
                        max = max.max(close(o.values, reference));
                        compared += o.values.len();
                    }
                    assert!(seen.insert(k, ()).is_none());
                    Ok(())
                })
                .unwrap();
                assert_eq!(s.work().tokens_committed, input.len());
                assert_eq!(s.work().projections_completed, input.len() * 32);
                assert_eq!(s.work().gathers_completed, input.len());
                assert_eq!(s.work().recurrence_steps_completed, input.len() * 3);
                assert_eq!(s.work().attention_heads_completed, input.len() * 4);
            }
            assert_eq!(seen.len(), expected.len());
            assert_eq!(s.history(), ids);
            let last = expected.get(&key(ids.len() - 1, None, "logits")).unwrap();
            let want = last[1].iter().map(|&v| v as f32).collect::<Vec<_>>();
            assert_eq!(greedy(s.logits().unwrap()), greedy(&want));
        }
    }
    println!(
        "Qwen3.5 independent hybrid stages: compared={compared},max_gate_ratio={max}; tiny fixtures only"
    );
}

#[test]
fn whole_chunk_partitions_and_reset_are_bit_identical() {
    let f = Fixture::new();
    let m = f.model();
    let ids = tokens();
    let mut full = m.session(options()).unwrap();
    let all = full.forward(&ids).unwrap().to_vec();
    let end = snapshot(&full);
    for chunk in [1, 2, 3, 4, 7, 16] {
        let mut s = m.session(options()).unwrap();
        let mut got = Vec::new();
        for part in ids.chunks(chunk) {
            got.extend_from_slice(s.forward(part).unwrap());
        }
        assert_float_bits(&all, &got);
        assert_eq!(end, snapshot(&s));
        s.reset();
        assert!(s.history().is_empty() && s.logits().is_none());
        assert_float_bits(s.forward(&ids).unwrap(), &all);
        assert_eq!(end, snapshot(&s));
    }
}

#[test]
fn every_observer_failure_preserves_prefix_distribution_and_all_state() {
    let f = Fixture::new();
    let m = f.model();
    let mut clean = m.session(options()).unwrap();
    clean.forward(&[1]).unwrap();
    let prefix = snapshot(&clean);
    clean.forward(&[7]).unwrap();
    let want = snapshot(&clean);
    for fail in 0..=106 {
        let mut s = m.session(options()).unwrap();
        s.forward(&[1]).unwrap();
        let memory = s.memory();
        let mut n = 0;
        let error = s
            .forward_observed(&[7], &AtomicBool::new(false), &mut |_| {
                let i = n;
                n += 1;
                if i == fail {
                    Err(Error::Observation("injected"))
                } else {
                    Ok(())
                }
            })
            .err()
            .unwrap();
        assert!(matches!(error, Error::Observation(_)));
        assert_eq!(n, fail + 1);
        assert_eq!(s.work().tokens_committed, 0);
        assert_eq!(s.work().outcome, Outcome::Error);
        assert_eq!(snapshot(&s), prefix);
        assert_eq!(s.memory(), memory);
        s.forward(&[7]).unwrap();
        assert_eq!(snapshot(&s), want);
    }
}

#[test]
fn cancellation_and_stale_nonfinite_kv_tails_do_not_publish_partial_chunks() {
    let f = Fixture::new();
    let m = f.model();
    let mut expected = m.session(options()).unwrap();
    expected.forward(&[1]).unwrap();
    let prefix = snapshot(&expected);
    expected.forward(&[7, 13]).unwrap();
    let final_state = snapshot(&expected);
    for layer in [Some(0), Some(1), Some(2), Some(3), None] {
        let mut s = m.session(options()).unwrap();
        s.forward(&[1]).unwrap();
        let cancel = AtomicBool::new(false);
        assert!(matches!(
            s.forward_observed(&[7, 13], &cancel, &mut |o| {
                if o.layer == layer
                    && ((layer.is_some() && o.stage == "input_norm")
                        || (layer.is_none() && o.stage == "before_commit"))
                {
                    cancel.store(true, Ordering::Relaxed);
                }
                Ok(())
            }),
            Err(Error::Cancelled)
        ));
        assert_eq!(snapshot(&s), prefix);
        assert_eq!(s.work().tokens_committed, 0);
        assert_eq!(s.work().outcome, Outcome::Cancelled);
        cancel.store(false, Ordering::Relaxed);
        s.forward_with_cancel(&[7, 13], &cancel).unwrap();
        assert_eq!(snapshot(&s), final_state);
    }
    let mut s = m.session(options()).unwrap();
    s.forward(&[1]).unwrap();
    let c = m.config;
    for cache in &mut s.kv {
        for head in 0..c.kv_heads {
            let start = (head * 17 + 1) * c.head_dim;
            let end = (head + 1) * 17 * c.head_dim;
            cache.k[start..end].fill(f32::NAN);
            cache.v[start..end].fill(f32::NAN);
        }
    }
    s.forward(&[7, 13]).unwrap();
    assert_eq!(snapshot(&s), final_state);
}

#[test]
fn admission_shapes_positions_ids_and_arithmetic_failures_are_bounded() {
    let f = Fixture::new();
    let m = f.model();
    let planned = Session::planned_memory(&m, options()).unwrap();
    let mut s = m
        .session(Options {
            memory_limit_bytes: planned.admitted_bytes,
            ..options()
        })
        .unwrap();
    assert_eq!(s.memory(), planned);
    assert!(
        matches!(m.session(Options{memory_limit_bytes:planned.admitted_bytes-1,..options()}),Err(Error::MemoryBudget{required,limit})if required==planned.admitted_bytes&&limit==required-1)
    );
    for o in [
        Options {
            capacity: 0,
            ..options()
        },
        Options {
            capacity: 513,
            ..options()
        },
        Options {
            chunk: 0,
            ..options()
        },
        Options {
            chunk: 33,
            ..options()
        },
        Options {
            position_offset: 4080,
            ..options()
        },
        Options {
            position_offset: usize::MAX,
            ..options()
        },
    ] {
        assert!(m.session(o).is_err());
    }
    s.forward(&[1]).unwrap();
    let before = snapshot(&s);
    for ids in [&[][..], &[32][..], &[0; 18][..]] {
        assert!(s.forward(ids).is_err());
        assert_eq!(snapshot(&s), before);
    }
    s.forward(&tokens()[1..]).unwrap();
    let before = snapshot(&s);
    assert!(s.forward(&[1]).is_err());
    assert_eq!(snapshot(&s), before);
    let (c, mut weights) = f.inputs();
    weights.layers.swap(0, 3);
    assert!(Model::new(c, weights).is_err());
    let (mut c, w) = f.inputs();
    c.value_heads = 3;
    assert!(Model::new(c, w).is_err());
    let (mut c, w) = f.inputs();
    c.rotated_dims = 33;
    assert!(Model::new(c, w).is_err());
    let (mut c, w) = f.inputs();
    c.hidden = usize::MAX;
    assert!(Model::new(c, w).is_err());
    let mut bad = Fixture::new();
    bad.vectors
        .get_mut("language_model.model.norm.weight")
        .unwrap()
        .fill(f32::MAX);
    let (c, w) = bad.inputs();
    let model = Model::new(c, w).unwrap();
    let mut s = model.session(options()).unwrap();
    let before = snapshot(&s);
    assert!(s.forward(&[1, 7]).is_err());
    assert_eq!(snapshot(&s), before);
}

#[test]
fn warmed_forward_failure_retries_and_reset_allocate_no_rust_heap() {
    let f = Fixture::new();
    let m = f.model();
    let mut s = m.session(options()).unwrap();
    s.forward(&[1]).unwrap();
    s.reset();
    let memory = s.memory();
    for _ in 0..3 {
        for input in [&[1, 7, 13][..], &[3][..], &[9, 5][..]] {
            let measure = Measurement::start();
            let result = s.forward(input);
            let n = measure.finish();
            result.unwrap();
            assert_eq!(n, 0);
            assert_eq!(s.memory(), memory);
        }
        let measure = Measurement::start();
        s.reset();
        assert_eq!(measure.finish(), 0);
    }
    let cancel = AtomicBool::new(true);
    let measure = Measurement::start();
    let result = s.forward_with_cancel(&[1], &cancel);
    let n = measure.finish();
    assert!(matches!(result, Err(Error::Cancelled)));
    assert_eq!(n, 0);
    let measure = Measurement::start();
    let result = s.forward(&[1]);
    let n = measure.finish();
    result.unwrap();
    assert_eq!(n, 0);
}

#[test]
#[ignore = "explicit authenticated Qwen3.5-4B CPU first-token capture; no codec or GPU"]
fn actual_source_model_session_first_token() {
    use crate::mlx::qwen35::{self as source, payload};
    use std::{
        fs::File,
        io::{Read, Write},
        path::PathBuf,
        time::Instant,
    };
    let directory = PathBuf::from(std::env::var_os("TOKLET_QWEN35_MODEL_DIRECTORY").unwrap());
    let output = PathBuf::from(std::env::var_os("TOKLET_QWEN35_GRAPH_CAPTURE").unwrap());
    std::fs::create_dir(&output).unwrap();
    let config_file = File::open(directory.join("config.json")).unwrap();
    assert_eq!(
        config_file.metadata().unwrap().len(),
        source::CONFIG_BYTES as u64
    );
    let mut config = Vec::with_capacity(source::CONFIG_BYTES + 1);
    config_file
        .take((source::CONFIG_BYTES + 1) as u64)
        .read_to_end(&mut config)
        .unwrap();
    assert_eq!(config.len(), source::CONFIG_BYTES);
    let input = File::open(directory.join("model.safetensors")).unwrap();
    assert!(input.metadata().unwrap().is_file());
    assert_eq!(input.metadata().unwrap().len(), source::CONTAINER_BYTES);
    let start = Instant::now();
    let store = payload::load(
        &config,
        input,
        payload::LoadOptions {
            memory_limit_bytes: MEMORY_LIMIT - 128 * 1024 * 1024,
            modality: source::ModalityPolicy::TextOnly,
        },
    )
    .unwrap();
    let load_ns = start.elapsed().as_nanos();
    let bind = Instant::now();
    let model = Model::from_cpu_store(&store).unwrap();
    let bind_ns = bind.elapsed().as_nanos();
    assert_eq!(
        (
            model.config.layers,
            model.config.hidden,
            model.config.intermediate,
            model.config.vocab
        ),
        (32, 2560, 9216, 248320)
    );
    let options = Options {
        capacity: 512,
        chunk: 32,
        position_offset: 0,
        memory_limit_bytes: MEMORY_LIMIT - 128 * 1024 * 1024,
    };
    let planned = Session::planned_memory(&model, options).unwrap();
    assert!(
        matches!(model.session(Options{memory_limit_bytes:planned.admitted_bytes-1,..options}),Err(Error::MemoryBudget{required,..})if required==planned.admitted_bytes)
    );
    let mut session = model
        .session(Options {
            memory_limit_bytes: planned.admitted_bytes,
            ..options
        })
        .unwrap();
    assert_eq!(session.memory(), planned);
    let mut body = std::io::BufWriter::with_capacity(
        64 * 1024,
        File::create(output.join("rows.f32le")).unwrap(),
    );
    let mut offset = 0;
    let mut arrays = 0;
    println!();
    let observed = Instant::now();
    session.forward_observed(&[1],&AtomicBool::new(false),&mut |o|{
        if o.stage=="before_commit"{return Ok(());}
        let bytes=o.values.len()*4;if offset+bytes>128*1024*1024{return Err(Error::Observation("bounded capture"));}
        let mut hash=Sha256::new();for &value in o.values{let b=value.to_le_bytes();body.write_all(&b).map_err(|_|Error::Observation("capture write"))?;hash.update(b);}
        println!("QWEN35_GRAPH_ARRAY {}",serde_json::json!({"token":o.token,"position":o.position,"layer":o.layer,"stage":o.stage,"head":o.head,"count":o.values.len(),"offset":offset,"bytes":bytes,"sha256":format!("{:x}",hash.finalize())}));offset+=bytes;arrays+=1;Ok(())
    }).unwrap();
    let observed_ns = observed.elapsed().as_nanos();
    body.flush().unwrap();
    body.get_ref().sync_all().unwrap();
    let expected = session.logits().unwrap().to_vec();
    let before = snapshot(&session);
    let work = session.work();
    assert_eq!(session.history(), [1]);
    assert_eq!(
        (
            work.projections_completed,
            work.gathers_completed,
            work.convolution_steps_completed,
            work.recurrence_steps_completed,
            work.attention_heads_completed
        ),
        (249, 1, 24, 24, 128)
    );
    session.reset();
    let plain = Instant::now();
    let measure = Measurement::start();
    let result = session.forward(&[1]);
    let allocations = measure.finish();
    result.unwrap();
    let plain_ns = plain.elapsed().as_nanos();
    assert_eq!(allocations, 0);
    assert_eq!(session.logits().unwrap(), expected);
    assert_eq!(snapshot(&session), before);
    println!(
        "QWEN35_GRAPH_REPORT {}",
        serde_json::json!({"source_sha256":source::CONTAINER_SHA256,"container_bytes":source::CONTAINER_BYTES,"config_sha256":source::CONFIG_SHA256,"header_sha256":source::HEADER_SHA256,"tokens":[1],"position_offset":0,"capacity":512,"chunk":32,"memory":session.memory(),"work":work,"capture_bytes":offset,"arrays":arrays,"load_ns":load_ns,"bind_ns":bind_ns,"observed_forward_ns":observed_ns,"plain_forward_ns":plain_ns,"warmed_rust_allocations":allocations,"greedy_lowest_id":greedy(session.logits().unwrap()),"first_token_executed":true,"independent_real_source_fidelity_qualified":false,"quality_qualified":false,"managed_adoption_qualified":false,"timing_scope":"Two CPU first-token observations, one with diagnostic I/O; no controlled engine performance qualification"})
    );
}
