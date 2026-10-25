use super::*;
use crate::cpu::PerLayerGateWeights;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

struct Counting;
thread_local! {
    static TRACK:Cell<bool>=const {Cell::new(false)};
    static ALLOCATIONS:Cell<usize>=const {Cell::new(0)};
}
fn count_allocation() {
    let _ = TRACK.try_with(|track| {
        if track.get() {
            ALLOCATIONS.with(|count| count.set(count.get() + 1));
        }
    });
}
// SAFETY: The test-only shim forwards every pointer/layout/size unchanged to
// System. TLS counters inspect no allocated payload and only measure this test
// thread, excluding concurrent harness allocations. Numerical code stays safe.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count_allocation();
        unsafe { System.realloc(pointer, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;

pub(crate) struct Measurement;
impl Measurement {
    pub(crate) fn start() -> Self {
        ALLOCATIONS.with(|v| v.set(0));
        TRACK.with(|v| v.set(true));
        Self
    }
    pub(crate) fn finish(self) -> usize {
        TRACK.with(|v| v.set(false));
        ALLOCATIONS.with(Cell::get)
    }
}
impl Drop for Measurement {
    fn drop(&mut self) {
        TRACK.with(|v| v.set(false));
    }
}

type Input = BTreeMap<String, Vec<f32>>;

fn fixture() -> (Value, Input, Vec<u32>) {
    let receipt =
        include_bytes!("../../../benchmarks/results/2026-10-25-gemma-text-graph-reference-v1.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(receipt)),
        "83c8a7a8ef86d98952ff92d85b9ee108f25241f8bc318665befd2bdee7008649"
    );
    let accepted: Value = serde_json::from_slice(receipt).unwrap();
    assert_eq!(accepted["cpu_source_graph_qualified"], true);
    assert_eq!(accepted["gpu_source_graph_qualified"], false);
    let raw = include_bytes!("../../../tests/fixtures/gemma-text-graph-v1/math.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        accepted["fixture_sha256"].as_str().unwrap()
    );
    let data: Value = serde_json::from_slice(raw).unwrap();
    let weights = data["weights"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, item)| {
            (
                name.clone(),
                item["values"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect(),
            )
        })
        .collect();
    let ids = data["token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    (data, weights, ids)
}

fn weights(input: &Input) -> Weights<'_> {
    Weights {
        embedding: &input["model.embed_tokens.weight"],
        per_layer_embedding: &input["model.embed_tokens_per_layer.weight"],
        per_layer_projection: &input["model.per_layer_model_projection.weight"],
        per_layer_norm: &input["model.per_layer_projection_norm.weight"],
        final_norm: &input["model.norm.weight"],
        layers: std::array::from_fn(|i| {
            let p = format!("model.layers.{i}.");
            let get = |name: &str| input[&(p.clone() + name)].as_slice();
            LayerWeights {
                input_norm: get("input_layernorm.weight"),
                q: get("self_attn.q_proj.weight"),
                q_norm: get("self_attn.q_norm.weight"),
                o: get("self_attn.o_proj.weight"),
                kv: (i < 2).then(|| KvWeights {
                    k: get("self_attn.k_proj.weight"),
                    v: get("self_attn.v_proj.weight"),
                    k_norm: get("self_attn.k_norm.weight"),
                }),
                finish: DenseDecoderWeights {
                    post_attention_norm: get("post_attention_layernorm.weight"),
                    pre_ff_norm: get("pre_feedforward_layernorm.weight"),
                    post_ff_norm: get("post_feedforward_layernorm.weight"),
                    ff_gate: get("mlp.gate_proj.weight"),
                    ff_up: get("mlp.up_proj.weight"),
                    ff_down: get("mlp.down_proj.weight"),
                    scalar: Some(get("layer_scalar")[0]),
                    per_layer: Some(PerLayerGateWeights {
                        width: PLE,
                        gate: get("per_layer_input_gate.weight"),
                        projection: get("per_layer_projection.weight"),
                        post_norm: get("post_per_layer_input_norm.weight"),
                    }),
                },
            }
        }),
    }
}

fn expected(data: &Value, contract: &str, name: &str, before: usize, count: usize) -> Vec<f64> {
    let item = &data[contract][name];
    let shape: Vec<usize> = item["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let axis = if name.ends_with("rotary") || name.ends_with("context") {
        2
    } else {
        1
    };
    let stride: usize = shape[axis + 1..].iter().product();
    item["values"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            let position = (i / stride) % shape[axis];
            (before..before + count).contains(&position)
        })
        .map(|(_, v)| v.as_f64().unwrap())
        .collect()
}
fn close(actual: &[f32], wanted: &[f64]) {
    assert_eq!(actual.len(), wanted.len());
    for (&a, &w) in actual.iter().zip(wanted) {
        assert!(
            a.is_finite() && (f64::from(a) - w).abs() <= 1e-5 + 1e-4 * w.abs(),
            "{a} vs {w}"
        );
    }
}
fn both(data: &Value, name: &str, actual: &[f32], before: usize, count: usize) {
    for contract in ["math", "staged_fp32"] {
        close(actual, &expected(data, contract, name, before, count));
    }
}
fn unheads(values: &[f32], heads: usize, count: usize, dims: usize) -> Vec<f32> {
    (0..heads)
        .flat_map(|head| {
            (0..count).flat_map(move |token| {
                (0..dims).map(move |feature| values[(token * heads + head) * dims + feature])
            })
        })
        .collect()
}
fn pointers(s: &Session<'_, '_>) -> Vec<usize> {
    let mut result = vec![s.caches.as_ptr() as usize];
    result.extend(s.caches.iter().flat_map(|cache| {
        let (k, v) = cache.committed();
        [k.as_ptr() as usize, v.as_ptr() as usize]
    }));
    result.extend(
        s.staging
            .iter()
            .flat_map(|stage| [stage.k.as_ptr() as usize, stage.v.as_ptr() as usize]),
    );
    let x = &s.scratch;
    result.extend(
        [
            &x.hidden,
            &x.next,
            &x.norm,
            &x.attention,
            &x.q,
            &x.q_norm,
            &x.context,
            &x.kv,
            &x.kv_norm,
            &x.ple_lookup,
            &x.ple_projection,
            &x.ple_scaled,
            &x.ple,
            &x.layer_ple,
            &x.dense_work,
            &x.dense_norm,
            &x.gate,
            &x.up,
            &x.product,
            &x.scores,
            &x.probabilities,
            &x.logits,
            &x.published,
        ]
        .map(|v| v.as_ptr() as usize),
    );
    result.sort_unstable();
    result
}
#[derive(Debug, PartialEq)]
struct Snapshot {
    tokens: Vec<u32>,
    logits: Vec<u32>,
    caches: Vec<(Vec<u32>, Vec<u32>, usize, u64)>,
}
fn snapshot(s: &Session<'_, '_>) -> Snapshot {
    Snapshot {
        tokens: s.tokens().to_vec(),
        logits: s.logits().iter().map(|v| v.to_bits()).collect(),
        caches: s
            .caches
            .iter()
            .map(|c| {
                let (k, v) = c.committed();
                (
                    k.iter().map(|x| x.to_bits()).collect(),
                    v.iter().map(|x| x.to_bits()).collect(),
                    c.offset(),
                    c.generation(),
                )
            })
            .collect(),
    }
}

#[test]
fn complete_frozen_graph_chunks_staging_cache_and_immutable_weights() {
    let (data, input, ids) = fixture();
    let before = input.clone();
    let model = Model::new(weights(&input)).unwrap();
    for schedule in data["schedules"].as_array().unwrap() {
        let mut session = model.new_session(usize::MAX).unwrap();
        let storage = pointers(&session);
        let bytes = session.allocated_bytes().unwrap();
        let mut offset = 0;
        for chunk in schedule["chunks"].as_array().unwrap() {
            let n = chunk.as_u64().unwrap() as usize;
            let prior = snapshot(&session);
            for layer in 0..LAYERS {
                session.fault = Some(layer);
                assert!(session.forward(&ids[offset..offset + n]).is_err());
                assert_eq!(snapshot(&session), prior);
                let d = dimension(layer);
                let s = &session.scratch;
                let name = format!("layer.{layer}.");
                for (stage, values) in [
                    ("output", &s.hidden[..n * HIDDEN]),
                    ("self_attn.o_proj", &s.attention[..n * HIDDEN]),
                    ("post_per_layer_input_norm", &s.dense_norm[..n * HIDDEN]),
                    ("per_layer_projection", &s.dense_work[..n * HIDDEN]),
                    ("per_layer_input_gate", &s.gate[..n * PLE]),
                    ("mlp.up_proj", &s.up[..n * intermediate(layer)]),
                    ("self_attn.q_norm", &s.q_norm[..n * HEADS * d]),
                ] {
                    both(&data, &(name.clone() + stage), values, offset, n);
                }
                both(
                    &data,
                    &(name.clone() + "q_rotary"),
                    &unheads(&s.q[..n * HEADS * d], HEADS, n, d),
                    offset,
                    n,
                );
                both(
                    &data,
                    &(name + "context"),
                    &unheads(&s.context[..n * HEADS * d], HEADS, n, d),
                    offset,
                    n,
                );
            }
            session.fault = None;
            both(
                &data,
                "logits",
                session.forward(&ids[offset..offset + n]).unwrap(),
                offset,
                n,
            );
            let s = &session.scratch;
            for (name, values) in [
                ("final_norm", &s.norm[..n * HIDDEN]),
                ("layer.3.output", &s.hidden[..n * HIDDEN]),
                ("layer.3.self_attn.o_proj", &s.attention[..n * HIDDEN]),
                (
                    "layer.3.post_per_layer_input_norm",
                    &s.dense_norm[..n * HIDDEN],
                ),
                ("layer.3.per_layer_projection", &s.dense_work[..n * HIDDEN]),
                ("layer.3.per_layer_input_gate", &s.gate[..n * PLE]),
                ("layer.3.mlp.up_proj", &s.up[..n * 24]),
                ("ple_lookup", &s.ple_lookup[..n * LAYERS * PLE]),
                ("ple_projection", &s.ple_projection[..n * LAYERS * PLE]),
            ] {
                both(&data, name, values, offset, n);
            }
            both(
                &data,
                "layer.3.q_rotary",
                &unheads(&s.q[..n * HEADS * 16], HEADS, n, 16),
                offset,
                n,
            );
            both(
                &data,
                "layer.3.context",
                &unheads(&s.context[..n * HEADS * 16], HEADS, n, 16),
                offset,
                n,
            );
            for owner in 0..2 {
                let d = dimension(owner);
                let stage = &session.staging[owner];
                both(
                    &data,
                    &format!("layer.{owner}.k_rotary"),
                    &stage.k[..n * d],
                    offset,
                    n,
                );
                both(
                    &data,
                    &format!("layer.{owner}.self_attn.v_norm"),
                    &stage.v[..n * d],
                    offset,
                    n,
                );
                let cache = &session.caches[owner];
                let (k, v) = cache.committed();
                let capacity = if owner == 0 { WINDOW } else { MAX_POSITIONS };
                for slot in 0..capacity {
                    if let Some(position) = cache.position_in_slot(slot) {
                        both(
                            &data,
                            &format!("layer.{owner}.k_rotary"),
                            &k[slot * d..(slot + 1) * d],
                            position,
                            1,
                        );
                        both(
                            &data,
                            &format!("layer.{owner}.self_attn.v_norm"),
                            &v[slot * d..(slot + 1) * d],
                            position,
                            1,
                        );
                    }
                }
            }
            offset += n;
            assert_eq!(session.tokens(), &ids[..offset]);
            assert_eq!(pointers(&session), storage);
            assert_eq!(session.allocated_bytes().unwrap(), bytes);
        }
        assert_eq!(offset, MAX_POSITIONS);
        session.reset().unwrap();
        assert!(session.tokens().is_empty() && session.logits().is_empty());
        assert_eq!(pointers(&session), storage);
        both(&data, "logits", session.forward(&ids).unwrap(), 0, 32);
    }
    assert_eq!(input, before);
}

#[test]
fn each_late_failure_preserves_public_prefix_and_recovers_exactly() {
    let (_, input, ids) = fixture();
    let model = Model::new(weights(&input)).unwrap();
    for point in 0..=LAYERS {
        let mut session = model.new_session(usize::MAX).unwrap();
        session.forward(&ids[..3]).unwrap();
        let prior = snapshot(&session);
        session.fault = Some(point);
        assert!(session.forward(&ids[3..8]).is_err());
        assert_eq!(snapshot(&session), prior);
        session.fault = None;
        let recovered = session.forward(&ids[3..8]).unwrap().to_vec();
        let mut control = model.new_session(usize::MAX).unwrap();
        control.forward(&ids[..3]).unwrap();
        assert_eq!(control.forward(&ids[3..8]).unwrap(), recovered);
        assert_eq!(snapshot(&session), snapshot(&control));
    }
}

#[test]
fn invalid_tokens_capacity_empty_and_real_late_overflow_are_atomic() {
    let (_, input, ids) = fixture();
    let model = Model::new(weights(&input)).unwrap();
    let mut session = model.new_session(usize::MAX).unwrap();
    session.forward(&ids[..3]).unwrap();
    let prior = snapshot(&session);
    for tokens in [&[][..], &[1, 17][..], &[u32::MAX][..], &ids[..30]] {
        assert!(session.forward(tokens).is_err());
        assert_eq!(snapshot(&session), prior);
    }
    session.forward(&ids[3..]).unwrap();
    let full = snapshot(&session);
    assert!(session.forward(&[1]).is_err());
    assert_eq!(snapshot(&session), full);
    let mut bad = weights(&input);
    bad.layers[3].finish.scalar = Some(f32::MAX);
    let bad = Model::new(bad).unwrap();
    let mut session = bad.new_session(usize::MAX).unwrap();
    let prior = snapshot(&session);
    assert!(matches!(
        session.forward(&ids[..3]),
        Err(Error::Kernel(cpu::Error::NonFinite {
            tensor: "dense scalar output",
            ..
        }))
    ));
    assert_eq!(snapshot(&session), prior);
    assert!(session.staging[0].k.iter().any(|&v| v != 0.));
}

#[test]
fn all_model_boundary_terms_reject_and_budget_is_complete() {
    let (_, input, _) = fixture();
    let bad = [f32::NAN; 8];
    let mut w = weights(&input);
    w.final_norm = &bad;
    assert!(Model::new(w).is_err());
    let mut w = weights(&input);
    w.layers[0].kv = None;
    assert!(Model::new(w).is_err());
    let mut w = weights(&input);
    w.layers[2].kv = Some(KvWeights {
        k: &[],
        v: &[],
        k_norm: &[],
    });
    assert!(Model::new(w).is_err());
    let mut w = weights(&input);
    w.layers[3].finish.per_layer = None;
    assert!(Model::new(w).is_err());
    let mut w = weights(&input);
    w.layers[3].finish.scalar = None;
    assert!(Model::new(w).is_err());
    let mut w = weights(&input);
    w.layers[3].finish.per_layer.as_mut().unwrap().width = 3;
    assert!(Model::new(w).is_err());
    let mut w = weights(&input);
    w.layers[3].finish.ff_up = &[];
    assert!(Model::new(w).is_err());
    let model = Model::new(weights(&input)).unwrap();
    let planned = model.model_bytes() + model.planned_session_bytes().unwrap();
    assert!(model.new_session(planned - 1).is_err());
    let session = model.new_session(planned).unwrap();
    assert_eq!(
        model.planned_session_bytes().unwrap(),
        session.allocated_bytes().unwrap()
    );
}

#[test]
fn warmed_decode_reset_accounting_and_rejected_admission_allocate_nothing() {
    let (_, input, ids) = fixture();
    let model = Model::new(weights(&input)).unwrap();
    let budget = model.model_bytes() + model.planned_session_bytes().unwrap();
    let measured = Measurement::start();
    let rejected = model.new_session(budget - 1).is_err();
    let allocations = measured.finish();
    assert!(rejected);
    assert_eq!(allocations, 0);
    let mut session = model.new_session(budget).unwrap();
    session.forward(&ids[..3]).unwrap();
    session.reset().unwrap();
    let bytes = session.allocated_bytes().unwrap();
    let measured = Measurement::start();
    for _ in 0..3 {
        for id in &ids {
            session.forward(std::slice::from_ref(id)).unwrap();
        }
        assert_eq!(session.allocated_bytes().unwrap(), bytes);
        session.reset().unwrap();
    }
    let allocations = measured.finish();
    assert_eq!(allocations, 0);
    println!(
        "Gemma synthetic reference: 96 warmed decode steps, 3 resets, {allocations} Rust allocations; model={} session={bytes} bytes",
        model.model_bytes()
    );
}
