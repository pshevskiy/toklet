use super::*;
use crate::mlx::affine::{Layout, Matrix};
use crate::model::gemma_reference::tests::Measurement;
use safetensors::{Dtype, SafeTensors};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const WEIGHTS: &[u8] = include_bytes!("../../../tests/fixtures/gemma3-cpu-v1/model.safetensors");
const ARRAY: &[u8] = include_bytes!("../../../tests/fixtures/gemma3-cpu-v1/arrays.bin");
fn metadata() -> Value {
    let raw = include_bytes!("../../../tests/fixtures/gemma3-cpu-v1/manifest.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        "bc63d9d478d50ee24e9c78e471ccd4d1ce2f268700cef8fddf7c2c096c7e7566"
    );
    let m: Value = serde_json::from_slice(raw).unwrap();
    assert_eq!(m["stage_groups"], 192);
    assert_eq!(m["stored_arrays"], 192);
    assert_eq!(m["cpu_gate"]["atol"].as_f64(), Some(1e-5));
    assert_eq!(m["cpu_gate"]["rtol"].as_f64(), Some(1e-4));
    assert_eq!(m["source_cpu_graph_executed"], true);
    assert_eq!(m["target_graph_executed"], false);
    assert_eq!(
        m["producer"]["sha256"],
        format!(
            "{:x}",
            Sha256::digest(include_bytes!(
                "../../../tools/reference/gemma3_cpu_graph.py"
            ))
        )
    );
    for (name, data) in [
        ("model.safetensors", WEIGHTS),
        ("arrays.bin", ARRAY),
        (
            "config.json",
            include_bytes!("../../../tests/fixtures/gemma3-cpu-v1/config.json").as_slice(),
        ),
        (
            "components.json",
            include_bytes!("../../../tests/fixtures/gemma3-cpu-v1/components.json").as_slice(),
        ),
    ] {
        assert_eq!(m["files"][name]["bytes"], data.len());
        assert_eq!(
            m["files"][name]["sha256"],
            format!("{:x}", Sha256::digest(data))
        );
    }
    m
}
fn packed(t: &SafeTensors<'static>, name: &str, rows: usize, columns: usize) -> Matrix<'static> {
    let w = t.tensor(&format!("{name}.weight")).unwrap();
    let s = t.tensor(&format!("{name}.scales")).unwrap();
    let b = t.tensor(&format!("{name}.biases")).unwrap();
    assert_eq!(w.dtype(), Dtype::U32);
    assert_eq!(s.dtype(), Dtype::BF16);
    assert_eq!(b.dtype(), Dtype::BF16);
    assert_eq!(w.shape(), [rows, columns / 8]);
    assert_eq!(s.shape(), [rows, columns / 64]);
    assert_eq!(b.shape(), s.shape());
    Matrix::from_le_bytes(
        Layout::new(rows, columns, 4, 64, MetadataDtype::BF16).unwrap(),
        w.data(),
        s.data(),
        b.data(),
    )
    .unwrap()
}
fn gamma(t: &SafeTensors<'_>, name: &str) -> Vec<bf16> {
    let v = t.tensor(name).unwrap();
    assert_eq!(v.dtype(), Dtype::BF16);
    v.data()
        .chunks_exact(2)
        .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())))
        .collect()
}
fn inputs(layers: usize) -> (Config, Vec<LayerSpec>, Weights<Matrix<'static>>) {
    let t = SafeTensors::deserialize(WEIGHTS).unwrap();
    let c = Config {
        vocab_size: 64,
        hidden_size: 64,
        intermediate_size: 128,
        num_layers: layers,
        query_heads: 4,
        kv_heads: 1,
        head_dim: 32,
        max_positions: 12,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
    };
    let specs = (0..layers)
        .map(|i| LayerSpec {
            kind: if i % 2 == 0 {
                KvKind::Sliding
            } else {
                KvKind::Full
            },
            theta: if i % 2 == 0 { 1e4 } else { 1e6 },
        })
        .collect();
    let layers = (0..layers)
        .map(|index| {
            let p = format!("model.layers.{}", index % 2);
            LayerWeights {
                q: packed(&t, &format!("{p}.self_attn.q_proj"), 128, 64),
                k: packed(&t, &format!("{p}.self_attn.k_proj"), 32, 64),
                v: packed(&t, &format!("{p}.self_attn.v_proj"), 32, 64),
                o: packed(&t, &format!("{p}.self_attn.o_proj"), 64, 128),
                gate: packed(&t, &format!("{p}.mlp.gate_proj"), 128, 64),
                up: packed(&t, &format!("{p}.mlp.up_proj"), 128, 64),
                down: packed(&t, &format!("{p}.mlp.down_proj"), 64, 128),
                gamma: Gamma {
                    input: gamma(&t, &format!("{p}.input_layernorm.weight")),
                    post_attention: gamma(&t, &format!("{p}.post_attention_layernorm.weight")),
                    pre_ff: gamma(&t, &format!("{p}.pre_feedforward_layernorm.weight")),
                    post_ff: gamma(&t, &format!("{p}.post_feedforward_layernorm.weight")),
                    q: gamma(&t, &format!("{p}.self_attn.q_norm.weight")),
                    k: gamma(&t, &format!("{p}.self_attn.k_norm.weight")),
                },
            }
        })
        .collect();
    (
        c,
        specs,
        Weights {
            embedding: packed(&t, "model.embed_tokens", 64, 64),
            layers,
            final_gamma: gamma(&t, "model.norm.weight"),
        },
    )
}
fn model(layers: usize) -> Model<Matrix<'static>> {
    let (c, s, w) = inputs(layers);
    Model::new(c, s, 4, w).unwrap()
}
fn values(d: &Value) -> Vec<f32> {
    assert_eq!(d["file"], "arrays.bin");
    assert_eq!(d["dtype"], "f32le");
    let start = d["offset"].as_u64().unwrap() as usize;
    let n = d["bytes"].as_u64().unwrap() as usize;
    let raw = &ARRAY[start..start + n];
    assert_eq!(d["sha256"], format!("{:x}", Sha256::digest(raw)));
    let result: Vec<_> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert!(result.iter().all(|v| v.is_finite()));
    result
}
fn compare(a: f32, b: f32, name: &str, index: usize, stats: &mut (usize, f64, f64)) {
    let error = (f64::from(a) - f64::from(b)).abs();
    let gate = 1e-5 + 1e-4 * f64::from(b).abs();
    assert!(
        a.is_finite() && error <= gate,
        "{name}[{index}]: actual={a} bits={} expected={b} bits={} error={error} gate={gate}",
        a.to_bits(),
        b.to_bits()
    );
    stats.0 += 1;
    stats.1 = stats.1.max(error);
    stats.2 = stats.2.max(error / gate);
}

#[test]
fn frozen_all_stages_available_attention_keys_and_exact_schedules() {
    let metadata = metadata();
    let m = model(2);
    let mut stats = (0, 0.0, 0.0);
    let mut observations = 0;
    let mut full_groups = 0;
    let mut full_values = 0;
    let mut outputs = BTreeMap::new();
    for case in metadata["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let ids: Vec<u32> = case["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let refs: BTreeMap<_, _> = case["arrays"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(name, d)| (name.clone(), values(d)))
            .collect();
        assert_eq!(refs.len(), 48);
        let expected_stages: BTreeSet<_> = refs.keys().cloned().collect();
        let mut full = None;
        for chunk in [12, 1, 3, 7] {
            let mut traced = m.new_session(chunk, usize::MAX).unwrap();
            let mut ordinary = m.new_session(chunk, usize::MAX).unwrap();
            let mut rows = Vec::new();
            for input in ids.chunks(chunk) {
                let mut seen = BTreeSet::new();
                let start = traced.committed;
                let before_values = stats.0;
                forward::execute::<true, _, _, _>(&mut traced, input, &mut |o| {
                    observations += 1;
                    let stage = o
                        .layer
                        .map_or_else(|| o.stage.to_owned(), |i| format!("layers.{i}.{}", o.stage));
                    assert!(seen.insert(stage.clone()), "duplicate stage {stage}");
                    assert!(expected_stages.contains(&stage), "unexpected stage {stage}");
                    assert_eq!((o.rows, o.token_offset), (input.len(), start));
                    let descriptor = &case["arrays"][&stage];
                    let shape: Vec<usize> = descriptor["shape"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|n| n.as_u64().unwrap() as usize)
                        .collect();
                    let reference = &refs[&stage];
                    assert_eq!(reference.len(), shape.iter().product::<usize>());
                    assert_eq!(o.values.len(), o.rows * o.columns * o.heads);
                    if let Some(first) = o.key_start {
                        assert_eq!(shape, [o.heads, ids.len(), ids.len()]);
                        assert_eq!(o.heads, 4);
                        assert!(first + o.columns <= ids.len());
                        for head in 0..o.heads {
                            for row in 0..o.rows {
                                for key in 0..o.columns {
                                    let actual = (head * o.rows + row) * o.columns + key;
                                    let expected = (head * ids.len() + o.token_offset + row)
                                        * ids.len()
                                        + first
                                        + key;
                                    compare(
                                        o.values[actual],
                                        reference[expected],
                                        &stage,
                                        actual,
                                        &mut stats,
                                    );
                                }
                            }
                        }
                    } else {
                        assert_eq!(o.heads, 1);
                        assert_eq!(shape[0], ids.len());
                        assert_eq!(shape[1..].iter().product::<usize>(), o.columns);
                        let first = o.token_offset * o.columns;
                        for (index, (&a, &b)) in o
                            .values
                            .iter()
                            .zip(&reference[first..first + o.values.len()])
                            .enumerate()
                        {
                            compare(a, b, &stage, index, &mut stats);
                        }
                    }
                })
                .unwrap();
                assert_eq!(seen, expected_stages, "incomplete stage inventory");
                if chunk == 12 {
                    full_groups += seen.len();
                    full_values += stats.0 - before_values;
                }
                let current = ordinary.forward(input).unwrap();
                assert_eq!(traced.logits(), current, "ordinary and TRACE differ");
                rows.extend_from_slice(current);
            }
            assert_eq!(traced.tokens(), ids);
            assert_eq!(ordinary.tokens(), ids);
            if let Some(expected) = &full {
                assert_eq!(expected, &rows, "schedule {chunk} differs for {name}");
            } else {
                full = Some(rows.clone());
            }
        }
        outputs.insert(name.to_owned(), full.unwrap());
    }
    assert_eq!(
        &outputs["wrap12"][..11 * 64],
        &outputs["future_changed12"][..11 * 64]
    );
    assert_eq!(observations, 2400);
    assert_eq!(full_groups, 192);
    assert_eq!(full_values, 108816);
    assert_eq!(stats.0, 428832);
    println!(
        "GEMMA3_CPU {}",
        json!({"source_cases":4,"source_stage_groups":full_groups,"one_shot_compared_values":full_values,"observations":observations,
        "compared_values":stats.0,"max_absolute_error":stats.1,"max_gate_fraction":stats.2,
        "main_gate":{"atol":1e-5,"rtol":1e-4},"target_schedules":[12,1,3,7],
        "attention_scope":"actual available absolute-key columns only","exact_numeric_logits_schedules":true,
        "exact_causal_logits_prefix":true,"ordinary_trace_identical":true})
    );
}

#[derive(Debug, PartialEq)]
struct Snapshot {
    tokens: Vec<u32>,
    logits: Vec<f32>,
    cache: Vec<CacheState>,
}
type CacheState = (usize, u64, Vec<f32>, Vec<f32>);
fn snapshot<M: MatrixSource, B: Operations<M>>(s: &Session<'_, M, B>) -> Snapshot {
    Snapshot {
        tokens: s.tokens().to_vec(),
        logits: s.logits().to_vec(),
        cache: s
            .caches
            .iter()
            .map(|c| {
                let (k, v) = c.committed();
                (c.offset(), c.generation(), k.to_vec(), v.to_vec())
            })
            .collect(),
    }
}
fn pointers<M: MatrixSource, B: Operations<M>>(s: &Session<'_, M, B>) -> Vec<usize> {
    let mut p = vec![
        s.history.as_ptr() as usize,
        s.caches.as_ptr() as usize,
        s.staging.as_ptr() as usize,
    ];
    for c in &s.caches {
        let (k, v) = c.committed();
        p.extend([k.as_ptr() as usize, v.as_ptr() as usize]);
    }
    for stage in &s.staging {
        p.extend([stage.k.as_ptr() as usize, stage.v.as_ptr() as usize]);
    }
    let x = &s.scratch;
    let mut scratch = [
        &x.hidden,
        &x.next,
        &x.norm,
        &x.q,
        &x.q_norm,
        &x.kv,
        &x.kv_norm,
        &x.context,
        &x.attention,
        &x.residual,
        &x.gate,
        &x.up,
        &x.gated,
        &x.scores,
        &x.probabilities,
        &x.logits,
        &x.published,
    ]
    .map(|v| v.as_ptr() as usize);
    if scratch[0] > scratch[9] {
        scratch.swap(0, 9);
    }
    p.extend(scratch);
    p
}
#[test]
fn all_layers_and_head_failures_preserve_state_and_forward_allocates_nothing() {
    let ids = [1, 7, 13, 3, 9, 5, 11, 2, 17, 4, 6, 10];
    for layers in [2, 18] {
        let m = model(layers);
        let mut control = m.new_session(7, usize::MAX).unwrap();
        control.forward(&ids[..3]).unwrap();
        control.forward(&ids[3..10]).unwrap();
        let after = snapshot(&control);
        for fault in 0..layers + 2 {
            let mut s = m.new_session(7, usize::MAX).unwrap();
            s.forward(&ids[..3]).unwrap();
            let before = snapshot(&s);
            let addresses = pointers(&s);
            s.fault = Some(fault);
            let measured = Measurement::start();
            let failure = s.forward(&ids[3..10]);
            let allocations = measured.finish();
            let expected = if fault < layers {
                "injected Gemma3 layer failure"
            } else if fault == layers {
                "injected Gemma3 final norm failure"
            } else {
                "injected Gemma3 head failure"
            };
            assert!(matches!(failure,Err(Error::InvalidConfig(message))if message==expected));
            assert_eq!(allocations, 0);
            assert_eq!(snapshot(&s), before);
            assert_eq!(pointers(&s), addresses);
            s.fault = None;
            let measured = Measurement::start();
            let passed = s.forward(&ids[3..10]).is_ok();
            let allocations = measured.finish();
            assert!(passed);
            assert_eq!(allocations, 0);
            assert_eq!(snapshot(&s), after);
            assert_eq!(pointers(&s), addresses);
            let measured = Measurement::start();
            s.reset().unwrap();
            let allocations = measured.finish();
            assert_eq!(allocations, 0);
            assert!(s.tokens().is_empty() && s.logits().is_empty());
            assert_eq!(pointers(&s), addresses);
        }
    }
}

#[test]
fn exact_budgets_invalid_requests_and_full_capacity_reject_before_state_change() {
    let m = model(2);
    for chunk in [1, 3, 7, 12] {
        let required = m.model_bytes() + m.planned_session_bytes(chunk).unwrap();
        let measured = Measurement::start();
        let rejected = m.new_session(chunk, required - 1);
        let allocations = measured.finish();
        assert_eq!(allocations, 0);
        assert!(
            matches!(rejected,Err(crate::session::Error::MemoryBudget{required:r,limit})if r==required&&limit==required-1)
        );
        let mut s = m.new_session(chunk, required).unwrap();
        assert_eq!(m.model_bytes() + s.allocated_bytes().unwrap(), required);
        for part in [1u32; 12].chunks(chunk) {
            s.forward(part).unwrap();
        }
        let before = snapshot(&s);
        for input in [&[1][..], &[64]] {
            assert!(matches!(
                s.forward(input),
                Err(Error::Capacity {
                    offset: 12,
                    tokens: 1,
                    limit: 12
                })
            ));
            assert_eq!(snapshot(&s), before);
        }
        assert!(matches!(
            s.forward(&[]),
            Err(Error::Kernel(cpu::Error::ZeroDimension("Gemma3 new chunk")))
        ));
        assert_eq!(snapshot(&s), before);
        s.reset().unwrap();
        let before = snapshot(&s);
        assert!(matches!(
            s.forward(&[64]),
            Err(Error::Kernel(cpu::Error::Token {
                position: 0,
                token: 64,
                vocab_size: 64
            }))
        ));
        assert_eq!(snapshot(&s), before);
        assert!(
            matches!(s.forward(&vec![1;chunk+1]),Err(Error::ScratchCapacity{tokens,limit}) if tokens==chunk+1&&limit==chunk)
        );
        assert_eq!(snapshot(&s), before);
    }
    assert!(m.new_session(0, usize::MAX).is_err());
    assert!(m.new_session(13, usize::MAX).is_err());
}

#[test]
fn initial_profile_gamma_precision_and_invalid_construction() {
    let exact = offset_norm(vec![bf16::from_f32(2.0f32.powi(-12))], 1).unwrap();
    assert_eq!(exact[0].to_bits(), 0x3f800800);
    assert_ne!(exact[0], bf16::from_f32(exact[0]).to_f32());
    assert_ne!(exact[0], half::f16::from_f32(exact[0]).to_f32());
    assert!(offset_norm(vec![bf16::NAN], 1).is_err());
    assert!(offset_norm(vec![bf16::ZERO], 2).is_err());
    assert_eq!(bf16::from_f64(640.0f64.sqrt()).to_f32(), 25.25);
    assert_eq!(model(2).embedding_scale(), 8.0);
    for change in 0..7 {
        let (mut c, mut specs, mut weights) = inputs(2);
        match change {
            0 => c.kv_heads = 2,
            1 => c.max_positions = 545,
            2 => specs[0].theta = f32::NAN,
            3 => {
                weights.layers[1].gamma.k.pop();
            }
            4 => {
                weights.final_gamma[0] = bf16::INFINITY;
            }
            5 => {
                specs.pop();
            }
            _ => {
                let layer = &mut weights.layers[0];
                std::mem::swap(&mut layer.q, &mut layer.k);
            }
        }
        assert!(Model::new(c, specs, 4, weights).is_err());
    }
    let (c, s, w) = inputs(33);
    assert!(Model::new(c, s, 4, w).is_err());
}

#[test]
fn actual_head256_window512_visibility_and_ring_wrap() {
    use crate::model::window_cache::{WindowKvCache, commit_batch};
    let component: Value = serde_json::from_slice(include_bytes!(
        "../../../tests/fixtures/gemma3-cpu-v1/components.json"
    ))
    .unwrap();
    let k = vec![0.0; 513 * 256];
    let mut v = vec![0.0; 513 * 256];
    v[0] = 8.0;
    v[256] = 4.0;
    v[512 * 256] = 2.0;
    let query = vec![0.0; 256];
    for window in [Some(512), None] {
        let mut cache = WindowKvCache::new(
            1,
            256,
            if window.is_some() { 512 } else { 544 },
            544,
            window,
        )
        .unwrap();
        let frame = cache.prepare(&k, &v, 513).unwrap();
        let mut output = vec![0.0; 256];
        let mut scores = vec![0.0; 513];
        let mut probs = vec![0.0; 513];
        {
            let view = cache.view(&frame).unwrap();
            cpu::span_attention(
                &query,
                &view.segments(),
                &mut output,
                &mut scores,
                &mut probs,
                &cpu::SpanAttentionShape {
                    queries: 1,
                    keys: 513,
                    query_heads: 1,
                    kv_heads: 1,
                    head_dim: 256,
                    query_start: 512,
                    key_start: 0,
                    window,
                    scale: 1.0 / 16.0,
                },
            )
            .unwrap();
        }
        let field = if window.is_some() {
            "local_context"
        } else {
            "global_context"
        };
        let expected: Vec<f32> = component["visibility"][field]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(output, expected);
        if window.is_some() {
            assert_eq!(probs[0], 0.0);
        } else {
            assert!(probs[0] > 0.0);
        }
        assert!(probs[1] > 0.0 && probs[512] > 0.0);
        commit_batch(std::slice::from_mut(&mut cache), &[frame]).unwrap();
        assert_eq!(cache.offset(), 513);
        if window.is_some() {
            assert_eq!(cache.position_in_slot(0), Some(512));
            assert_eq!(cache.position_in_slot(1), Some(1));
        }
        let empty = vec![0.0; 256];
        let frame = cache.prepare(&empty, &empty, 1).unwrap();
        {
            let view = cache.view(&frame).unwrap();
            let n = view.end() - view.start();
            let mut scores = vec![0.0; n];
            let mut p = vec![0.0; n];
            cpu::span_attention(
                &query,
                &view.segments(),
                &mut output,
                &mut scores,
                &mut p,
                &cpu::SpanAttentionShape {
                    queries: 1,
                    keys: n,
                    query_heads: 1,
                    kv_heads: 1,
                    head_dim: 256,
                    query_start: 513,
                    key_start: view.start(),
                    window,
                    scale: 1.0 / 16.0,
                },
            )
            .unwrap();
        }
        if window.is_some() {
            assert_eq!(output[0], 2.0 / 512.0);
        }
        commit_batch(std::slice::from_mut(&mut cache), &[frame]).unwrap();
        assert_eq!(cache.offset(), 514);
    }
}
