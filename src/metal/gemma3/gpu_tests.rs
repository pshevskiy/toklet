//! Explicit tiny-graph qualification. No real checkpoint or speed claim.
use super::*;
use crate::mlx::affine::Layout as AffineLayout;
use crate::model::gemma_reference::tests::Measurement;
use crate::model::gemma3::{Gamma, LayerWeights};
use half::f16;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
};
use safetensors::{Dtype, SafeTensors};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Write,
};
const WEIGHTS: &[u8] = include_bytes!("../../../tests/fixtures/gemma3-cpu-v1/model.safetensors");
const RAW: &[u8] = include_bytes!("../../../tests/fixtures/gemma3-half-tiny-v1/arrays.bin");
const MANIFEST: &[u8] = include_bytes!("../../../tests/fixtures/gemma3-half-tiny-v1/manifest.json");
const LIMIT: usize = crate::weights::INITIAL_MEMORY_LIMIT_BYTES;
// Bounded retained test vectors/raw JSON/fixture parser; numerical diagnostics
// and their maximal FP32 readbacks are already in Session::Memory.
const EXTERNAL: usize = 32 * 1024 * 1024;
const HALF_GUARD: u16 = 0x7e55;
const WORD_GUARD: u32 = 0x5a170031;
fn number(v: &Value) -> usize {
    usize::try_from(v.as_u64().unwrap()).unwrap()
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn emit(value: Value) {
    println!("GEMMA3_GRAPH_RAW {value}");
    std::io::stdout().flush().unwrap();
}
fn fixture() -> Value {
    assert_eq!(
        hash(MANIFEST),
        "f9c8de309701b1ba85cff79ce41eb68c51c6dbc8185581c9bd12e51db9dcde0d"
    );
    assert_eq!(
        hash(RAW),
        "0455fd4b4c0110411756f0d296fb3d8b6ac31702ff79462386a2eccf6126bb56"
    );
    assert_eq!(
        hash(WEIGHTS),
        "5ea7ae502d8d1b385155d62f252bdcc82c637f60324c3937fa42caae53a5989d"
    );
    let m: Value = serde_json::from_slice(MANIFEST).unwrap();
    assert_eq!(m["format"], "toklet-gemma3-half-tiny-graph-v1");
    assert_eq!(m["policy"]["fatal_reference"], "main");
    assert_eq!(m["policy"]["atol_floor"].as_f64(), Some(1e-4));
    assert_eq!(m["policy"]["envelope_factor"], 2);
    assert_eq!(m["policy"]["rtol"].as_f64(), Some(1e-3));
    assert_eq!(m["total_stored_arrays"], 424);
    for (name, data) in [
        (
            "gemma3_cpu_graph.py",
            include_bytes!("../../../tools/reference/gemma3_cpu_graph.py").as_slice(),
        ),
        (
            "gemma3_half_pointwise.py",
            include_bytes!("../../../tools/reference/gemma3_half_pointwise.py").as_slice(),
        ),
        (
            "gemma3_half_rms.py",
            include_bytes!("../../../tools/reference/gemma3_half_rms.py").as_slice(),
        ),
        (
            "gemma3_staged_attention.py",
            include_bytes!("../../../tools/reference/gemma3_staged_attention.py").as_slice(),
        ),
    ] {
        assert_eq!(m["dependencies"][name]["sha256"], hash(data));
        assert_eq!(number(&m["dependencies"][name]["bytes"]), data.len());
    }
    assert_eq!(
        m["input_files"]["components.json"]["sha256"],
        hash(include_bytes!(
            "../../../tests/fixtures/gemma3-cpu-v1/components.json"
        ))
    );

    assert_eq!(
        m["input_files"]["model.safetensors"]["sha256"],
        hash(WEIGHTS)
    );
    assert_eq!(
        m["input_files"]["config.json"]["sha256"],
        hash(include_bytes!(
            "../../../tests/fixtures/gemma3-cpu-v1/config.json"
        ))
    );
    assert_eq!(
        m["producer"]["sha256"],
        hash(include_bytes!(
            "../../../tools/reference/gemma3_half_graph.py"
        ))
    );
    m
}
fn bytes(d: &Value) -> &[u8] {
    assert_eq!(d["file"], "arrays.bin");
    let start = number(&d["offset"]);
    let end = start.checked_add(number(&d["bytes"])).unwrap();
    let b = &RAW[start..end];
    assert_eq!(d["sha256"], hash(b));
    b
}
fn values(d: &Value) -> Vec<f64> {
    let b = bytes(d);
    let v: Vec<_> = match d["dtype"].as_str().unwrap() {
        "u16le" => b
            .chunks_exact(2)
            .map(|v| f16::from_bits(u16::from_le_bytes(v.try_into().unwrap())).to_f64())
            .collect(),
        "f32le" => b
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()) as f64)
            .collect(),
        "f64le" => b
            .chunks_exact(8)
            .map(|v| f64::from_le_bytes(v.try_into().unwrap()))
            .collect(),
        _ => panic!("unexpected fixture dtype"),
    };
    assert!(v.iter().all(|v| v.is_finite()));
    assert_eq!(
        v.len(),
        d["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(number)
            .product::<usize>()
    );
    v
}
fn numeric_bits(v: &[f32]) -> Vec<u32> {
    v.iter()
        .map(|&x| {
            assert!(x.is_finite());
            if x == 0.0 { 0 } else { x.to_bits() }
        })
        .collect()
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
        AffineLayout::new(rows, columns, 4, 64, MetadataDtype::BF16).unwrap(),
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

fn model() -> Model {
    let (c, s, w) = inputs(2);
    Model::from_raw_gamma(c, s, 4, w, EXTERNAL, LIMIT).unwrap()
}
fn options(diagnostics: bool) -> Options {
    Options {
        memory_limit_bytes: LIMIT,
        external_resident_bytes: EXTERNAL,
        all_logits: diagnostics,
        diagnostics,
    }
}
#[test]
fn frozen_half_graph_all_descriptors_and_policy() {
    let m = fixture();
    let mut spans = Vec::new();
    for c in m["cases"].as_array().unwrap() {
        assert_eq!(c["stage_order"].as_array().unwrap().len(), 48);
        for a in c["arrays"].as_object().unwrap().values() {
            let main = values(&a["main"]);
            let wide = values(&a["wide"]);
            let env = main
                .iter()
                .zip(&wide)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert_eq!(a["envelope"].as_f64().unwrap(), env);
            assert_eq!(a["gate"]["atol"].as_f64().unwrap(), 1e-4f64.max(2.0 * env));
            assert_eq!(a["gate"]["rtol"].as_f64().unwrap(), 1e-3);
            for k in ["main", "wide"] {
                let d = &a[k];
                spans.push((number(&d["offset"]), number(&d["bytes"])));
            }
        }
    }
    for c in m["rotary_components"].as_array().unwrap() {
        for d in c["arrays"].as_object().unwrap().values() {
            values(d);
            spans.push((number(&d["offset"]), number(&d["bytes"])));
        }
    }
    spans.sort_unstable();
    let mut end = 0;
    for (start, n) in &spans {
        assert_eq!(*start, end);
        end += n;
    }
    assert_eq!(spans.len(), 424);
    assert_eq!(end, RAW.len());
}
#[test]
fn original_packed_graph_shapes_and_raw_gamma_contract() {
    fixture();
    let (c, s, w) = inputs(2);
    geometry(&c, &s, 4).unwrap();
    let mut count = 1;
    let mut packed = w.embedding.layout().total_bytes();
    for layer in &w.layers {
        for mat in [
            &layer.q,
            &layer.k,
            &layer.v,
            &layer.o,
            &layer.gate,
            &layer.up,
            &layer.down,
        ] {
            count += 1;
            packed += mat.layout().total_bytes();
        }
        assert_eq!(layer.q.layout().rows(), 128);
        assert_eq!(layer.o.layout().columns(), 128);
        for g in [
            &layer.gamma.input,
            &layer.gamma.post_attention,
            &layer.gamma.pre_ff,
            &layer.gamma.post_ff,
            &layer.gamma.q,
            &layer.gamma.k,
        ] {
            assert!(g.iter().all(|v| v.to_f32().is_finite()));
        }
    }
    assert_eq!(count, 15);
    assert_eq!(packed, 52992);
    assert_eq!(
        (bf16::from_f32(2.0f32.powi(-12)).to_f32() + 1.0).to_bits(),
        0x3f800800
    );
}
// Test-only queue-ordered fault, no mapped mutable resource escape. It executes
// after the real layer/head and before normal validation/transaction commit.
pub(super) fn inject_nan(command: &Command<'_>, dst: &ProtocolObject<dyn MTLBuffer>) -> Result<()> {
    let encoder = command
        .blitCommandEncoder()
        .ok_or_else(|| Error::Native("test fault blit unavailable".into()))?;
    // Live retained same-command output, aligned four-byte prefix. Byte 0xff
    // is NaN in both IEEE Half words and the FP32 head word.
    encoder.fillBuffer_range_value(dst, NSRange::new(0, 4), 0xff);
    encoder.endEncoding();
    Ok(())
}
#[derive(Clone)]
struct Capture {
    info: SnapshotInfo,
    actual: Vec<f32>,
}
fn raw_session(s: &Session<'_>, label: &str, result: &Result<()>) -> Value {
    if let Err(error) = s.cache.test_safe_read() {
        emit(
            json!({"kind":"unsafe_read_refused","label":label,"result":format!("{result:?}"),"completion":s.cache.test_completion(),"error":format!("{error:?}")}),
        );
        panic!("mapped graph readback refused until terminal completion");
    }
    let cache = s.cache.test_observe().unwrap();
    let half: Vec<_>=[("hidden",&s.scratch.hidden),("norm",&s.scratch.norm),("branch",&s.scratch.branch),("next",&s.scratch.next),("q",&s.scratch.q),("work",&s.scratch.work),("k_norm",&s.scratch.k_norm),("gate",&s.scratch.gate),("up",&s.scratch.up),("activated",&s.scratch.activated)].into_iter().map(|(name,b)|json!({"name":name,"address":b.raw.contents().as_ptr() as usize,"bytes":b.raw.length(),"words":b.with_slice(|v|v.to_vec())})).collect();
    let float: Vec<_>=[("scores",&s.scratch.scores),("probabilities",&s.scratch.probabilities),("tokens",&s.scratch.tokens),("candidate",&s.candidate),("published",&s.published)].into_iter().map(|(name,b)|json!({"name":name,"address":b.raw.contents().as_ptr() as usize,"bytes":b.raw.length(),"words":b.with_slice(|v|v.to_vec())})).collect();
    let snapshots = s.test_snapshot_words().unwrap();
    emit(
        json!({"kind":"session","label":label,"result":format!("{result:?}"),"phase_a":s.test_phase_a,"cache":cache,"half_buffers":half,"word_buffers":float,"snapshot_words":snapshots,"history_physical":s.history,"published_rows":s.published_rows,"diagnostic_rows":s.diagnostic_rows}),
    );
    cache
}
fn capture(s: &Session<'_>, label: &str) -> Vec<Capture> {
    let mut out = Vec::new();
    for index in 0..s.diagnostic_count() {
        let info = s.diagnostic_info(index).unwrap();
        let mut actual = vec![0.0; info.values()];
        s.copy_diagnostic(index, &mut actual).unwrap();
        emit(
            json!({"kind":"stage","label":label,"index":index,"layer":info.layer,"stage":info.stage,"shape":&info.shape[..info.rank],"query_start":info.absolute_query_start,"key_start":info.absolute_key_start,"f32_bits":actual.iter().map(|v|v.to_bits()).collect::<Vec<_>>()}),
        );
        out.push(Capture { info, actual });
    }
    out
}
fn guards(s: &Session<'_>, cache: &Value) {
    for o in cache["owners"].as_array().unwrap() {
        for b in o.as_array().unwrap() {
            let v = b["words"].as_array().unwrap();
            assert!(v[v.len() - 7..].iter().all(|x| x == 0x7e55));
        }
    }
    assert!(
        cache["status"].as_array().unwrap()[1..]
            .iter()
            .all(|x| x == 0x5a170031)
    );
    let mut addresses = BTreeSet::new();
    for o in cache["owners"].as_array().unwrap() {
        for b in o.as_array().unwrap() {
            assert!(addresses.insert(number(&b["address"])));
        }
    }
    // Nonzero cold-seeded tails must survive all encoders, failures and resets.
    for b in [
        &s.scratch.hidden,
        &s.scratch.norm,
        &s.scratch.branch,
        &s.scratch.next,
        &s.scratch.q,
        &s.scratch.work,
        &s.scratch.k_norm,
        &s.scratch.gate,
        &s.scratch.up,
        &s.scratch.activated,
    ] {
        b.with_slice(|v| assert!(v[v.len() - 7..].iter().all(|&x| x == HALF_GUARD)));
    }
    for b in [
        &s.scratch.scores,
        &s.scratch.probabilities,
        &s.scratch.tokens,
        &s.candidate,
        &s.published,
    ] {
        b.with_slice(|v| assert!(v[v.len() - 7..].iter().all(|&x| x == WORD_GUARD)));
    }
    for (slot, words) in s.test_snapshot_words().unwrap().iter().enumerate() {
        let float = slot == 47 || [10, 11, 32, 33].contains(&slot);
        let guard = if float { 7 } else { 4 };
        let expected = if float {
            WORD_GUARD
        } else {
            u32::from(HALF_GUARD) | (u32::from(HALF_GUARD) << 16)
        };
        assert!(words[words.len() - guard..].iter().all(|&x| x == expected));
    }
}
fn seeded_session(model: &Model, diagnostic: bool) -> Session<'_> {
    let mut s = model.session(options(diagnostic)).unwrap();
    s.cache.test_safe_read().unwrap();
    for b in [
        &mut s.scratch.hidden,
        &mut s.scratch.norm,
        &mut s.scratch.branch,
        &mut s.scratch.next,
        &mut s.scratch.q,
        &mut s.scratch.work,
        &mut s.scratch.k_norm,
        &mut s.scratch.gate,
        &mut s.scratch.up,
        &mut s.scratch.activated,
    ] {
        b.with_mut(|v| {
            let start = v.len() - 7;
            v[start..].fill(HALF_GUARD);
        });
    }
    for b in [
        &mut s.scratch.scores,
        &mut s.scratch.probabilities,
        &mut s.scratch.tokens,
        &mut s.candidate,
        &mut s.published,
    ] {
        b.with_mut(|v| {
            let start = v.len() - 7;
            v[start..].fill(WORD_GUARD);
        });
    }
    s.test_seed_snapshot_guards(HALF_GUARD, WORD_GUARD).unwrap();
    // Retain the actual initialized physical tails, not just a declared constant.
    raw_session(&s, "cold-nonzero-guard-seed", &Ok(()));
    s
}
#[derive(Default)]
struct Stats {
    values: usize,
    stages: usize,
    main_max: f64,
    wide_max: f64,
    wide_exceed: usize,
}
fn compare(case: &Value, captured: &[Capture], stats: &mut Stats) {
    assert_eq!(captured.len(), 48);
    let total = case["tokens"].as_array().unwrap().len();
    for (slot, c) in captured.iter().enumerate() {
        let name = c.info.layer.map_or_else(
            || c.info.stage.to_string(),
            |l| format!("layers.{l}.{}", c.info.stage),
        );
        assert_eq!(case["stage_order"][slot], name);
        let a = &case["arrays"][&name];
        let main = values(&a["main"]);
        let wide = values(&a["wide"]);
        let start = c.info.absolute_query_start;
        let n = c.info.shape[0];
        let scores = matches!(c.info.stage, "attention_scores" | "attention_probabilities");
        let shape: Vec<_> = a["main"]["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(number)
            .collect();
        let atol = a["gate"]["atol"].as_f64().unwrap();
        let rtol = a["gate"]["rtol"].as_f64().unwrap();
        let indices: Vec<usize> = if scores {
            assert_eq!(c.info.rank, 3);
            assert_eq!(c.info.shape[0], 4);
            let rows = c.info.shape[1];
            let keys = c.info.shape[2];
            let expected_start = if c.info.layer == Some(0) {
                start.saturating_sub(3)
            } else {
                0
            };
            assert_eq!(c.info.absolute_key_start, expected_start);
            assert_eq!(keys, start + rows - expected_start);
            (0..4)
                .flat_map(|h| {
                    (0..rows).flat_map(move |r| {
                        (0..keys).map(move |k| (h * total + start + r) * total + expected_start + k)
                    })
                })
                .collect()
        } else {
            let mut expected_shape = shape.clone();
            expected_shape[0] = n;
            assert_eq!(&c.info.shape[..c.info.rank], expected_shape);
            let width: usize = shape[1..].iter().product();
            assert_eq!(c.actual.len(), n * width);
            (start * width..(start + n) * width).collect()
        };
        assert_eq!(indices.len(), c.actual.len());
        let mut main_max = 0.0f64;
        let mut wide_max = 0.0f64;
        let mut failures = 0;
        let mut wide_exceed = 0;
        for (&actual, index) in c.actual.iter().zip(indices) {
            let x = actual as f64;
            let error = (x - main[index]).abs();
            let gate = atol + rtol * main[index].abs();
            let ratio = error / gate;
            if !x.is_finite() || error > gate {
                failures += 1;
            }
            main_max = main_max.max(ratio);
            let wr = (x - wide[index]).abs() / (atol + rtol * wide[index].abs());
            wide_max = wide_max.max(wr);
            wide_exceed += usize::from(wr > 1.0);
        }
        emit(
            json!({"kind":"gate","case":case["name"],"stage":name,"start":start,"values":c.actual.len(),"main_failures":failures,"main_max_fraction":main_max,"wide_max_fraction_diagnostic":wide_max,"wide_exceedances_diagnostic":wide_exceed}),
        );
        assert_eq!(failures, 0, "{} {name} start{start}", case["name"]);
        stats.values += c.actual.len();
        stats.stages += 1;
        stats.main_max = stats.main_max.max(main_max);
        stats.wide_max = stats.wide_max.max(wide_max);
        stats.wide_exceed += wide_exceed;
    }
}
// Independently update physical ring/full expectations from native stage words,
// rather than comparing the cache to the approximate source math profile.
fn expected_cache(expected: &mut [Vec<Vec<u16>>], captures: &[Capture], start: usize) {
    assert_eq!(expected.len(), 2);
    for (layer, expected_layer) in expected.iter_mut().enumerate() {
        for (kv, stage) in ["k_rotary", "v_projection"].into_iter().enumerate() {
            let c = captures
                .iter()
                .find(|c| c.info.layer == Some(layer) && c.info.stage == stage)
                .unwrap();
            let retained = if layer == 0 { 4 } else { 12 };
            for (r, row) in c.actual.chunks_exact(32).enumerate() {
                let slot = if layer == 0 {
                    (start + r) % retained
                } else {
                    start + r
                };
                for (d, &x) in row.iter().enumerate() {
                    expected_layer[kv][slot * 32 + d] = f16::from_f32(x).to_bits();
                }
            }
        }
    }
}
fn assert_cache(cache: &Value, expected: &[Vec<Vec<u16>>]) {
    for (l, e) in expected.iter().enumerate() {
        for (kv, words) in e.iter().enumerate() {
            let actual: Vec<_> = cache["owners"][l][kv]["words"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| number(v) as u16)
                .collect();
            assert_eq!(&actual, words);
        }
    }
}
fn initial_cache(cache: &Value) -> Vec<Vec<Vec<u16>>> {
    (0..2)
        .map(|l| {
            (0..2)
                .map(|k| {
                    cache["owners"][l][k]["words"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| number(v) as u16)
                        .collect()
                })
                .collect()
        })
        .collect()
}
fn schedule(model: &Model, case: &Value, chunk: usize, stats: &mut Stats) -> Vec<f32> {
    let tokens: Vec<_> = case["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| number(v) as u32)
        .collect();
    let mut s = seeded_session(model, true);
    let baseline = s.cache.test_observe().unwrap();
    let mut expected = initial_cache(&baseline);
    let mut logits = Vec::new();
    for (part, ids) in tokens.chunks(chunk).enumerate() {
        let start = part * chunk;
        let result = s.forward(ids);
        let label = format!("{}-chunk{chunk}-at{start}", case["name"].as_str().unwrap());
        let cache = raw_session(&s, &label, &result);
        result.unwrap();
        let all = capture(&s, &label);
        guards(&s, &cache);
        assert_eq!(s.test_phase_a, Some((true, true)));
        assert_eq!(cache["status"][0], 0);
        assert_eq!(s.committed_tokens().unwrap(), &tokens[..start + ids.len()]);
        for c in &all {
            assert_eq!(c.info.absolute_query_start, start);
            let rows = if matches!(c.info.stage, "attention_scores" | "attention_probabilities") {
                c.info.shape[1]
            } else {
                c.info.shape[0]
            };
            assert_eq!(rows, ids.len());
        }
        expected_cache(&mut expected, &all, start);
        assert_cache(&cache, &expected);
        compare(case, &all, stats);
        let mut out = vec![0.0; ids.len() * 64];
        s.copy_all_logits(&mut out).unwrap();
        logits.extend(out);
    }
    let saved = logits.clone();
    let reset = s.reset();
    raw_session(&s, "reset", &reset);
    reset.unwrap();
    assert!(s.committed_tokens().unwrap().is_empty());
    assert!(s.copy_logits(&mut [0.0; 64]).is_err());
    assert!(s.diagnostic_info(0).is_err());
    let mut replay = Vec::new();
    for ids in tokens.chunks(chunk) {
        let r = s.forward(ids);
        let cache = raw_session(&s, "replay", &r);
        r.unwrap();
        guards(&s, &cache);
        let mut out = vec![0.0; ids.len() * 64];
        s.copy_all_logits(&mut out).unwrap();
        replay.extend(out);
    }
    let replay_cache = s.cache.test_observe().unwrap();
    assert_cache(&replay_cache, &expected);
    assert_eq!(s.committed_tokens().unwrap(), tokens);
    assert_eq!(numeric_bits(&saved), numeric_bits(&replay));
    saved
}
#[test]
#[ignore = "requires explicit Apple Metal GPU collection"]
fn resident_gemma3_all_stages_schedules_and_resources() {
    autoreleasepool(|_| {
        let m = fixture();
        let model = model();
        let mut stats = Stats::default();
        let mut saved = BTreeMap::new();
        emit(
            json!({"kind":"model","device":model.device_name(),"packed_weights":model.memory.packed_weights,"effective_norms":model.memory.effective_norms,"zero_bias":model.memory.zero_bias,"admitted":model.memory.admitted,"cold_peak":model.memory.cold_peak}),
        );
        assert_eq!(model.memory.packed_weights, 52992);
        for case in m["cases"].as_array().unwrap() {
            let n = case["tokens"].as_array().unwrap().len();
            let full = schedule(&model, case, n, &mut stats);
            for chunk in [1, 3, 7] {
                let actual = schedule(&model, case, chunk, &mut stats);
                assert_eq!(numeric_bits(&actual), numeric_bits(&full));
            }
            let tokens: Vec<_> = case["tokens"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| number(v) as u32)
                .collect();
            let mut ordinary = seeded_session(&model, false);
            let r = ordinary.forward(&tokens);
            let cache = raw_session(&ordinary, "ordinary-last", &r);
            r.unwrap();
            guards(&ordinary, &cache);
            let mut last = [0.0; 64];
            ordinary.copy_logits(&mut last).unwrap();
            assert_eq!(numeric_bits(&last), numeric_bits(&full[full.len() - 64..]));
            saved.insert(case["name"].as_str().unwrap().to_owned(), full);
        }
        assert_eq!(
            numeric_bits(&saved["wrap12"][..11 * 64]),
            numeric_bits(&saved["future_changed12"][..11 * 64])
        );
        assert_eq!(
            numeric_bits(&saved["single"]),
            numeric_bits(&saved["wrap12"][..64])
        );
        assert_eq!(
            numeric_bits(&saved["short"]),
            numeric_bits(&saved["wrap12"][..4 * 64])
        );
        for diagnostic in [false, true] {
            let good = model.session(options(diagnostic)).unwrap();
            let memory = good.memory();
            emit(
                json!({"kind":"memory","diagnostics":diagnostic,"admitted":memory.admitted,"cold_peak":memory.cold_peak,"parts":[memory.packed_weights,memory.effective_norms,memory.zero_bias,memory.cache,memory.scratch,memory.logits,memory.tokens_and_history,memory.diagnostic_gpu,memory.diagnostic_readback_reserved,memory.metadata,memory.shared_framework_allowance,memory.external_resident]}),
            );
            assert_eq!(
                memory.admitted,
                [
                    memory.packed_weights,
                    memory.effective_norms,
                    memory.zero_bias,
                    memory.cache,
                    memory.scratch,
                    memory.logits,
                    memory.tokens_and_history,
                    memory.diagnostic_gpu,
                    memory.diagnostic_readback_reserved,
                    memory.metadata,
                    memory.shared_framework_allowance,
                    memory.external_resident
                ]
                .into_iter()
                .sum::<usize>()
            );
            assert!(memory.cold_peak >= memory.admitted);
            drop(good);
            let mut exact = options(diagnostic);
            exact.memory_limit_bytes = memory.admitted;
            assert!(model.session(exact).is_ok());
            exact.memory_limit_bytes -= 1;
            assert!(
                matches!(model.session(exact),Err(Error::Budget{required,limit}) if required==memory.admitted&&limit==memory.admitted-1)
            );
        }
        emit(
            json!({"kind":"graph_summary","schedules":16,"main_values":stats.values,"stage_snapshots":stats.stages,"main_max_fraction":stats.main_max,"wide_max_fraction_diagnostic":stats.wide_max,"wide_exceedances_diagnostic":stats.wide_exceed}),
        );
        assert_eq!(stats.stages, 2400);
        assert_eq!(stats.values, 428832);
    });
}
fn old_words(cache: &Value) -> Value {
    json!(
        (0..2)
            .map(|l| vec![
                cache["owners"][l][0]["words"].clone(),
                cache["owners"][l][1]["words"].clone()
            ])
            .collect::<Vec<_>>()
    )
}
fn logits(s: &Session<'_>) -> Vec<f32> {
    let mut v = vec![0.0; 64];
    s.copy_logits(&mut v).unwrap();
    v
}
fn publication(s: &Session<'_>) -> (Vec<u32>, Vec<u32>, usize) {
    s.cache.test_safe_read().unwrap();
    (
        s.history.clone(),
        s.published.with_slice(|v| v.to_vec()),
        s.published_rows,
    )
}
fn assert_same_prefix(before: &Value, after: &Value) {
    assert_eq!(before["offset"], after["offset"]);
    assert_eq!(before["generation"], after["generation"]);
    assert_eq!(old_words(before), old_words(after));
}
fn model_words(m: &Model) -> Value {
    let mut tensors = vec![&m.embedding];
    let mut norms = vec![&m.final_norm];
    for l in &m.layers {
        tensors.extend([&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down]);
        norms.extend([
            &l.norms.input,
            &l.norms.post_attention,
            &l.norms.pre_ff,
            &l.norms.post_ff,
            &l.norms.q,
            &l.norms.k,
        ]);
    }
    json!({"packed":tensors.iter().map(|t|json!({"address":t.buffer.raw.contents().as_ptr() as usize,"bytes":t.buffer.raw.length(),"words":t.buffer.with_slice(|v|v.to_vec()),"metadata":match &t.storage { TensorStorage::Mlx(p)=>p.test_metadata(),_=>panic!("original packed storage required") }})).collect::<Vec<_>>(),"effective_norms":norms.iter().map(|b|json!({"address":b.raw.contents().as_ptr() as usize,"bytes":b.raw.length(),"words":b.with_slice(|v|v.to_vec())})).collect::<Vec<_>>(),"zero_bias":m.zero_bias.with_slice(|v|v.to_vec())})
}
fn check_original_model(record: &Value) {
    let (_, _, w) = inputs(2);
    let mut matrices = vec![&w.embedding];
    let mut gammas = vec![&w.final_gamma];
    for l in &w.layers {
        matrices.extend([&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down]);
        gammas.extend([
            &l.gamma.input,
            &l.gamma.post_attention,
            &l.gamma.pre_ff,
            &l.gamma.post_ff,
            &l.gamma.q,
            &l.gamma.k,
        ]);
    }
    let mut identities = BTreeSet::new();
    for (actual, source) in record["packed"].as_array().unwrap().iter().zip(matrices) {
        assert!(identities.insert(number(&actual["address"])));
        let (weight, scale, bias) = source.original_tensors();
        let actual_weight: Vec<_> = actual["words"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|x| (number(x) as u16).to_le_bytes())
            .collect();
        assert_eq!(actual_weight, weight);
        for (i, original) in [scale, bias].iter().enumerate() {
            assert!(identities.insert(number(&actual["metadata"][i][0])));
            let actual: Vec<_> = actual["metadata"][i][1]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|x| (number(x) as u32).to_le_bytes())
                .collect();
            assert_eq!(&actual[..original.len()], *original);
            assert!(actual[original.len()..].iter().all(|&x| x == 0));
        }
    }
    for (actual, gamma) in record["effective_norms"]
        .as_array()
        .unwrap()
        .iter()
        .zip(gammas)
    {
        assert!(identities.insert(number(&actual["address"])));
        let words: Vec<_> = actual["words"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| number(v) as u32)
            .collect();
        assert_eq!(words.len(), gamma.len() + 7);
        assert_eq!(
            &words[..gamma.len()],
            gamma
                .iter()
                .map(|x| (1.0 + x.to_f32()).to_bits())
                .collect::<Vec<_>>()
        );
        assert!(words[gamma.len()..].iter().all(|&x| x == 0x5a170031));
    }
    assert_eq!(record["packed"].as_array().unwrap().len(), 15);
    assert_eq!(record["effective_norms"].as_array().unwrap().len(), 13);
    assert!(
        record["zero_bias"]
            .as_array()
            .unwrap()
            .iter()
            .all(|x| x == 0)
    );
}
#[test]
#[ignore = "requires explicit Apple Metal GPU collection"]
fn resident_gemma3_rollback_poison_reset_and_zero_allocation() {
    autoreleasepool(|_| {
        fixture();
        let model = model();
        let before_model = model_words(&model);
        emit(json!({"kind":"model_words_before","buffers":before_model}));
        check_original_model(&before_model);
        let tokens = [1, 7, 13, 3, 9, 5, 11, 2, 17, 4, 6, 10];
        let mut s = seeded_session(&model, true);
        let r = s.forward(&tokens[..4]);
        let before = raw_session(&s, "seed", &r);
        r.unwrap();
        let saved = logits(&s);
        let saved_publication = publication(&s);
        for bad in [&[][..], &[64][..], &tokens[..9], &[1u32; 33][..]] {
            let r = s.forward(bad);
            let after = raw_session(&s, "preadmission-reject", &r);
            assert!(matches!(r, Err(Error::Options(_))));
            assert_eq!(before, after);
            assert_eq!(saved_publication, publication(&s));
            guards(&s, &after);
            assert_eq!(numeric_bits(&saved), numeric_bits(&logits(&s)));
        }
        for fault in [TestFault::LateLayer, TestFault::Head] {
            let before = s.cache.test_observe().unwrap();
            s.test_fault = fault;
            let r = s.forward(&tokens[4..7]);
            let after = raw_session(&s, "nonfinite-candidate", &r);
            assert!(matches!(r, Err(Error::Nonfinite)));
            assert!(!s.is_poisoned());
            assert_same_prefix(&before, &after);
            assert_eq!(saved_publication, publication(&s));
            assert_eq!(s.committed_tokens().unwrap(), &tokens[..4]);
            assert_eq!(numeric_bits(&saved), numeric_bits(&logits(&s)));
            assert!(s.diagnostic_info(0).is_err());
            guards(&s, &after);
            s.test_fault = TestFault::None;
        }
        // Failed candidates did not consume history; same trusted continuation
        // remains legal. Reset/replay provides its independent exact witness.
        let r = s.forward(&tokens[4..7]);
        let after = raw_session(&s, "recover-nonfinite", &r);
        r.unwrap();
        guards(&s, &after);
        let continued = logits(&s);
        s.reset().unwrap();
        let r = s.forward(&tokens[..7]);
        raw_session(&s, "trusted-replay-seven", &r);
        r.unwrap();
        assert_eq!(numeric_bits(&continued), numeric_bits(&logits(&s)));
        let before = s.cache.test_observe().unwrap();
        let history = s.committed_tokens().unwrap().to_vec();
        let prior_copy_publication = publication(&s);
        s.test_fault = TestFault::FirstCopy;
        let r = s.forward(&tokens[7..10]);
        let failed = raw_session(&s, "partial-first-copy", &r);
        assert!(matches!(r, Err(Error::ReplayRequired)));
        assert!(s.is_poisoned());
        assert_eq!(prior_copy_publication, publication(&s));
        guards(&s, &failed);
        assert_eq!(before["offset"], failed["offset"]);
        assert_eq!(before["generation"], failed["generation"]);
        assert_ne!(
            before["owners"][0][0]["words"],
            failed["owners"][0][0]["words"]
        );
        assert_eq!(
            before["owners"][0][1]["words"],
            failed["owners"][0][1]["words"]
        );
        assert_eq!(
            before["owners"][1][0]["words"],
            failed["owners"][1][0]["words"]
        );
        assert_eq!(
            before["owners"][1][1]["words"],
            failed["owners"][1][1]["words"]
        );
        assert!(matches!(s.committed_tokens(), Err(Error::ReplayRequired)));
        let mut last = [123.0; 64];
        assert!(matches!(
            s.copy_logits(&mut last),
            Err(Error::ReplayRequired)
        ));
        assert_eq!(last, [123.0; 64]);
        let mut all = vec![123.0; 7 * 64];
        assert!(matches!(
            s.copy_all_logits(&mut all),
            Err(Error::ReplayRequired)
        ));
        assert!(all.iter().all(|&v| v == 123.0));
        assert!(matches!(s.diagnostic_info(0), Err(Error::ReplayRequired)));
        assert!(matches!(
            s.copy_diagnostic(0, &mut last),
            Err(Error::ReplayRequired)
        ));
        let r = s.forward(&[1]);
        let rejected = raw_session(&s, "poisoned-forward", &r);
        assert!(matches!(r, Err(Error::ReplayRequired)));
        assert_eq!(failed, rejected);
        assert_eq!(prior_copy_publication, publication(&s));
        let r = s.reset();
        raw_session(&s, "poison-reset", &r);
        r.unwrap();
        s.test_fault = TestFault::None;
        assert!(s.committed_tokens().unwrap().is_empty());
        let r = s.forward(&history);
        raw_session(&s, "poison-trusted-replay", &r);
        r.unwrap();
        assert_eq!(numeric_bits(&continued), numeric_bits(&logits(&s)));
        let r = s.forward(&tokens[7..]);
        let after = raw_session(&s, "full-capacity", &r);
        r.unwrap();
        guards(&s, &after);
        assert_eq!(s.committed_tokens().unwrap(), tokens);
        let capacity_publication = publication(&s);
        let r = s.forward(&[1]);
        let rejected = raw_session(&s, "capacity-plus-one", &r);
        assert!(matches!(r, Err(Error::Options(_))));
        assert_eq!(after, rejected);
        assert_eq!(capacity_publication, publication(&s));
        drop(s);
        let mut ordinary = seeded_session(&model, false);
        let warm = ordinary.forward(&tokens);
        raw_session(&ordinary, "allocation-warm", &warm);
        warm.unwrap();
        ordinary.reset().unwrap();
        let mut counts = Vec::new();
        for token in tokens {
            let measurement = Measurement::start();
            let r = ordinary.forward(&[token]);
            let count = measurement.finish();
            emit(
                json!({"kind":"allocation_call","sample":counts.len(),"token":token,"count":count,"release_zero_required":!cfg!(debug_assertions),"result":format!("{r:?}")}),
            );
            let observed = raw_session(&ordinary, "measured-ordinary-forward", &r);
            r.unwrap();
            guards(&ordinary, &observed);
            counts.push(count);
        }
        emit(
            json!({"kind":"allocation","scope":"Rust forward boundary including command creation/completion; Objective-C driver allocations excluded","release_zero_required":!cfg!(debug_assertions),"counts":counts}),
        );
        if !cfg!(debug_assertions) {
            assert!(counts.iter().all(|&v| v == 0));
        }
        let after_model = model_words(&model);
        emit(json!({"kind":"model_words_after","buffers":after_model}));
        assert_eq!(before_model, after_model);
        emit(
            json!({"kind":"state_summary","nonfinite_faults":2,"submitted_partial_copy_faults":1,"ordinary_forward_allocation_samples":12,"real_checkpoint":false}),
        );
    });
}
#[test]
#[ignore = "requires explicit Apple Metal GPU collection"]
fn ordinary_rotary_matches_frozen_gemma3_components() {
    autoreleasepool(|_| {
        let m = fixture();
        let gpu = Gpu::new().unwrap();
        let mut values_checked = 0;
        let mut main_max = 0.0f64;
        let mut math_max = 0.0f64;
        for case in m["rotary_components"].as_array().unwrap() {
            let a = &case["arrays"];
            let input: Vec<u16> = bytes(&a["input"])
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let main = values(&a["main"]);
            let wide = values(&a["wide"]);
            let math = values(&a["mathematical"]);
            let bounds = values(&a["math_bound"]);
            let d = number(&case["dim"]);
            let heads = number(&case["heads"]);
            let width = d * heads;
            let theta = case["theta"].as_f64().unwrap() as f32;
            for (row, position) in case["positions"].as_array().unwrap().iter().enumerate() {
                let mut source = Buffer::<u16>::new(&gpu, width + 7).unwrap();
                source.with_mut(|v| {
                    v.fill(0x7e55);
                    v[..width].copy_from_slice(&input[row * width..(row + 1) * width]);
                });
                let mut out = Buffer::<u16>::new(&gpu, width + 7).unwrap();
                out.with_mut(|v| v.fill(0x7e55));
                let mut status = Buffer::<u32>::new(&gpu, 8).unwrap();
                status.with_mut(|v| {
                    v.fill(0x5a170031);
                    v[0] = 0;
                });
                let raw = gpu.command().unwrap();
                let command = Command {
                    raw: &raw,
                    trace: None,
                };
                let encode = gpu.encode(
                    &command,
                    Kernel::Rotary,
                    &[&source.raw, &out.raw],
                    &[
                        1,
                        heads as u32,
                        d as u32,
                        number(position) as u32,
                        theta.to_bits(),
                    ],
                    width / 2,
                    Some(&status),
                );
                raw.commit();
                raw.waitUntilCompleted();
                // Always retain terminal status and physical operands before gates.
                if !matches!(
                    raw.status(),
                    MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
                ) {
                    emit(
                        json!({"kind":"rotary_nonterminal","status":format!("{:?}",raw.status()),"error":raw.error().map(|e|e.to_string())}),
                    );
                    panic!("rotary completion unknown");
                }
                let actual = out.with_slice(|v| v.to_vec());
                let source_after = source.with_slice(|v| v.to_vec());
                let state = status.with_slice(|v| v.to_vec());
                emit(
                    json!({"kind":"rotary","case":case["name"],"position":position,"dim":d,"heads":heads,"theta":theta,"encode":format!("{encode:?}"),"command_status":format!("{:?}",raw.status()),"command_error":raw.error().map(|e|e.to_string()),"input":source_after,"output":actual,"status":state,"resource_addresses":[source.raw.contents().as_ptr() as usize,out.raw.contents().as_ptr() as usize,status.raw.contents().as_ptr() as usize]}),
                );
                encode.unwrap();
                assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
                assert!(raw.error().is_none());
                assert_eq!(state[0], 0);
                assert!(state[1..].iter().all(|&x| x == 0x5a170031));
                assert_eq!(
                    &source_after[..width],
                    &input[row * width..(row + 1) * width]
                );
                assert!(
                    source_after[width..]
                        .iter()
                        .chain(&actual[width..])
                        .all(|&x| x == 0x7e55)
                );
                let atol = case["gate"]["atol"].as_f64().unwrap();
                let rtol = case["gate"]["rtol"].as_f64().unwrap();
                let mut failures = 0;
                let mut wide_max = 0.0f64;
                for (j, &word) in actual[..width].iter().enumerate() {
                    let k = row * width + j;
                    let x = f16::from_bits(word).to_f64();
                    let ratio = (x - main[k]).abs() / (atol + rtol * main[k].abs());
                    let mr = (x - math[k]).abs() / bounds[k];
                    main_max = main_max.max(ratio);
                    math_max = math_max.max(mr);
                    wide_max = wide_max.max((x - wide[k]).abs() / (atol + rtol * wide[k].abs()));
                    if !x.is_finite() || ratio > 1.0 || mr > 1.0 {
                        failures += 1;
                    }
                    values_checked += 1;
                }
                emit(
                    json!({"kind":"rotary_gate","case":case["name"],"position":position,"failures":failures,"wide_max_fraction_diagnostic":wide_max}),
                );
                assert_eq!(failures, 0);
            }
        }
        emit(
            json!({"kind":"rotary_summary","components":8,"commands":32,"main_and_mathematical_values":values_checked,"main_max_fraction":main_max,"mathematical_max_fraction":math_max}),
        );
        assert_eq!(values_checked, 11520);
    });
}
