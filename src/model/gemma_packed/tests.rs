use super::*;
use crate as engine;
#[path = "../../../tests/common/gemma_packed_fixture.rs"]
pub(super) mod fixture;
#[cfg(all(
    feature = "metal",
    any(target_os = "macos", target_os = "ios"),
    target_arch = "aarch64"
))]
mod hybrid;

const HIDDEN: usize = 64;
const LAYERS: usize = 4;
const HEADS: usize = 2;
const PLE: usize = 64;
const WINDOW: usize = 4;
fn dimension(layer: usize) -> usize {
    if layer.is_multiple_of(2) { 64 } else { 128 }
}
fn intermediate(layer: usize) -> usize {
    if layer < 2 { 64 } else { 128 }
}
fn expected(
    fixture: &fixture::Fixture,
    contract: &str,
    name: &str,
    before: usize,
    count: usize,
) -> Vec<f64> {
    let item = &fixture.data[contract][name];
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
    fixture
        .values(contract, name)
        .into_iter()
        .enumerate()
        .filter(|(i, _)| {
            let position = (i / stride) % shape[axis];
            (before..before + count).contains(&position)
        })
        .map(|(_, v)| v)
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
pub(super) fn both(
    data: &fixture::Fixture,
    name: &str,
    actual: &[f32],
    before: usize,
    count: usize,
) {
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
pub(super) fn pointers<M: MatrixSource, B: Operations<M>>(s: &Session<'_, '_, M, B>) -> Vec<usize> {
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
pub(super) struct Snapshot {
    tokens: Vec<u32>,
    logits: Vec<u32>,
    caches: Vec<(Vec<u32>, Vec<u32>, usize, u64)>,
}
pub(super) fn snapshot<M: MatrixSource, B: Operations<M>>(s: &Session<'_, '_, M, B>) -> Snapshot {
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
    let data = fixture::Fixture::new();
    let model = Model::new(data.weights()).unwrap();
    frozen_graph_checks(
        &data,
        || model.new_session(usize::MAX).unwrap(),
        |_, _, _, _| {},
    );
}
pub(super) fn frozen_graph_checks<
    'model,
    'weights: 'model,
    M: MatrixSource + 'model,
    B: Operations<M>,
>(
    data: &fixture::Fixture,
    mut create: impl FnMut() -> Session<'model, 'weights, M, B>,
    mut observe: impl FnMut(&str, &[f32], usize, usize),
) where
    B::Error: std::fmt::Debug + std::fmt::Display,
{
    let ids = &data.ids;
    let mut both =
        |data: &fixture::Fixture, name: &str, actual: &[f32], before: usize, count: usize| {
            observe(name, actual, before, count);
            self::both(data, name, actual, before, count);
        };
    for schedule in data.data["schedules"].as_array().unwrap() {
        let mut session = create();
        let storage = pointers(&session);
        let bytes = session.allocated_bytes().unwrap();
        let mut offset = 0;
        for chunk in schedule["chunks"].as_array().unwrap() {
            let n = chunk.as_u64().unwrap() as usize;
            let prior = snapshot(&session);
            for layer in 0..LAYERS {
                session.fault = Some(layer);
                let result = session.forward(&ids[offset..offset + n]);
                assert!(
                    result.as_ref().err().is_some_and(|e| e.to_string()
                        == Error::InvalidConfig("injected Gemma layer failure").to_string()),
                    "expected injected layer failure at layer {layer}, actual: {result:?}"
                );
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
                    both(data, &(name.clone() + stage), values, offset, n);
                }
                both(
                    data,
                    &(name.clone() + "q_rotary"),
                    &unheads(&s.q[..n * HEADS * d], HEADS, n, d),
                    offset,
                    n,
                );
                both(
                    data,
                    &(name + "context"),
                    &unheads(&s.context[..n * HEADS * d], HEADS, n, d),
                    offset,
                    n,
                );
            }
            session.fault = None;
            both(
                data,
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
                ("layer.3.mlp.up_proj", &s.up[..n * 128]),
                ("ple_lookup", &s.ple_lookup[..n * LAYERS * PLE]),
                ("ple_projection", &s.ple_projection[..n * LAYERS * PLE]),
            ] {
                both(data, name, values, offset, n);
            }
            both(
                data,
                "layer.3.q_rotary",
                &unheads(&s.q[..n * HEADS * 128], HEADS, n, 128),
                offset,
                n,
            );
            both(
                data,
                "layer.3.context",
                &unheads(&s.context[..n * HEADS * 128], HEADS, n, 128),
                offset,
                n,
            );
            for owner in 0..2 {
                let d = dimension(owner);
                let stage = &session.staging[owner];
                both(
                    data,
                    &format!("layer.{owner}.k_rotary"),
                    &stage.k[..n * d],
                    offset,
                    n,
                );
                both(
                    data,
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
                            data,
                            &format!("layer.{owner}.k_rotary"),
                            &k[slot * d..(slot + 1) * d],
                            position,
                            1,
                        );
                        both(
                            data,
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
        both(data, "logits", session.forward(ids).unwrap(), 0, 32);
    }
}

#[test]
fn each_late_failure_preserves_public_prefix_and_recovers_exactly() {
    let data = fixture::Fixture::new();
    let ids = &data.ids;
    let model = Model::new(data.weights()).unwrap();
    for point in 0..=LAYERS + 2 {
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
    let data = fixture::Fixture::new();
    let ids = &data.ids;
    let model = Model::new(data.weights()).unwrap();
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
    let mut bad = data.weights();
    bad.layers[3].scalar = f32::MAX;
    let bad = Model::new(bad).unwrap();
    let mut session = bad.new_session(usize::MAX).unwrap();
    let prior = snapshot(&session);
    assert!(matches!(
        session.forward(&ids[..3]),
        Err(Error::Kernel(cpu::Error::NonFinite {
            tensor: "Gemma packed scalar output",
            ..
        }))
    ));
    assert_eq!(snapshot(&session), prior);
    assert!(session.staging[0].k.iter().any(|&v| v != 0.));
}

#[test]
fn packed_profile_and_all_model_terms_are_admitted_before_session() {
    let data = fixture::Fixture::new();
    // Every borrowed matrix position is checked, including producer-only K/V.
    for index in 0..4 {
        for field in 0..7 {
            let mut w = data.weights();
            let wrong = data.matrix("model.embed_tokens.weight");
            let layer = &mut w.layers[index];
            match field {
                0 => layer.q = wrong,
                1 => layer.o = wrong,
                2 => layer.ff_gate = wrong,
                3 => layer.ff_up = wrong,
                4 => layer.ff_down = wrong,
                5 => layer.ple_gate = wrong,
                _ => layer.ple_projection = wrong,
            }
            assert!(Model::new(w).is_err());
        }
        for field in 0..6 {
            let mut w = data.weights();
            let layer = &mut w.layers[index];
            match field {
                0 => layer.input_norm = &[],
                1 => layer.q_norm = &[],
                2 => layer.post_attention_norm = &[],
                3 => layer.pre_ff_norm = &[],
                4 => layer.post_ff_norm = &[],
                _ => layer.post_ple_norm = &[],
            }
            assert!(Model::new(w).is_err());
        }
        let mut w = data.weights();
        w.layers[index].scalar = f32::NAN;
        assert!(Model::new(w).is_err());
    }
    for index in 0..2 {
        let mut w = data.weights();
        w.layers[index].kv = None;
        assert!(Model::new(w).is_err());
        for field in 0..3 {
            let mut w = data.weights();
            let kv = w.layers[index].kv.as_mut().unwrap();
            match field {
                0 => kv.k = data.matrix("model.embed_tokens.weight"),
                1 => kv.v = data.matrix("model.embed_tokens.weight"),
                _ => kv.k_norm = &[],
            }
            assert!(Model::new(w).is_err());
        }
    }
    let mut w = data.weights();
    w.layers[2].kv = w.layers[0].kv.take();
    assert!(Model::new(w).is_err());
    for field in 0..6 {
        let mut w = data.weights();
        match field {
            0 => w.embedding = data.matrix("model.per_layer_model_projection.weight"),
            1 => w.per_layer_embedding = data.matrix("model.embed_tokens.weight"),
            2 => w.per_layer_projection = data.matrix("model.embed_tokens.weight"),
            3 => w.per_layer_norm = &[],
            4 => w.final_norm = &[],
            _ => {
                w.layers.pop();
            }
        }
        assert!(Model::new(w).is_err());
    }
    let nonfinite = [f32::INFINITY; 64];
    let mut w = data.weights();
    w.final_norm = &nonfinite;
    assert!(Model::new(w).is_err());
    for (bits, group, dtype) in [
        (8, 64, MetadataDtype::BF16),
        (4, 32, MetadataDtype::BF16),
        (4, 64, MetadataDtype::F16),
        (4, 64, MetadataDtype::F32),
    ] {
        let l = crate::mlx::affine::Layout::new(17, 64, bits, group, dtype).unwrap();
        let codes = vec![0; l.weight_bytes()];
        let metadata = vec![0; l.metadata_bytes()];
        let bad = Matrix::from_le_bytes(l, &codes, &metadata, &metadata).unwrap();
        let mut w = data.weights();
        w.embedding = bad;
        assert!(Model::new(w).is_err());
    }
    let model = Model::new(data.weights()).unwrap();
    let budget = model.model_bytes() + model.planned_session_bytes().unwrap();
    assert!(model.new_session(budget - 1).is_err());
    let session = model.new_session(budget).unwrap();
    assert_eq!(
        session.allocated_bytes().unwrap(),
        model.planned_session_bytes().unwrap()
    );
}

mod chunk;
mod producers;

mod e2b;
