//! Explicit ignored tiny GPU conformance. Source gates are frozen before target.
//! Raw terminal evidence is emitted before result/numerical/state assertions.
use super::*;
use crate::model::gemma_reference::tests::Measurement;
use half::f16;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Write,
};
const RAW: &[u8] = include_bytes!("../../../tests/fixtures/lfm2-half-tiny-v1/arrays.bin");
const MANIFEST: &[u8] = include_bytes!("../../../tests/fixtures/lfm2-half-tiny-v1/manifest.json");
const ORIGINAL: &[u8] = include_bytes!("../../../tests/fixtures/lfm2-scalar-v1/model.safetensors");
const LIMIT: usize = crate::weights::INITIAL_MEMORY_LIMIT_BYTES;
const EXTERNAL: usize = 32 * 1024 * 1024;
fn number(v: &Value) -> usize {
    usize::try_from(v.as_u64().unwrap()).unwrap()
}
fn hash(v: &[u8]) -> String {
    format!("{:x}", Sha256::digest(v))
}
fn emit(v: Value) {
    println!("LFM2_GRAPH_RAW {v}");
    std::io::stdout().flush().unwrap();
}
fn hex16(v: &[u16]) -> String {
    let mut s = String::with_capacity(v.len() * 4);
    for x in v {
        use std::fmt::Write;
        write!(&mut s, "{x:04x}").unwrap();
    }
    s
}
fn hex32(v: &[u32]) -> String {
    let mut s = String::with_capacity(v.len() * 8);
    for x in v {
        use std::fmt::Write;
        write!(&mut s, "{x:08x}").unwrap();
    }
    s
}
fn fbits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}
fn numeric(v: &[f32]) -> Vec<u32> {
    v.iter()
        .map(|&x| {
            assert!(x.is_finite());
            if x == 0. { 0 } else { x.to_bits() }
        })
        .collect()
}
fn fixture() -> Value {
    assert_eq!(
        hash(MANIFEST),
        "271c4f243a99615a8cb90339781bacd6fbff73f793f389bc735f4031630f47e3"
    );
    assert_eq!(
        hash(RAW),
        "c3b431cc734e6fdc33caf10518dc543a73b44e65d32cd01837b7a571b89362d5"
    );
    assert_eq!(
        hash(ORIGINAL),
        "5965adf8a9663fed07fd2b293e717d72f907ddae931649e3189d161d58257487"
    );
    let m: Value = serde_json::from_slice(MANIFEST).unwrap();
    assert_eq!(m["counts"]["descriptors"], 504);
    assert_eq!(m["counts"]["groups_per_profile"], 252);
    assert_eq!(m["policy"]["fatal_reference"], "main");
    assert_eq!(m["policy"]["atol_floor"], 1e-4);
    assert_eq!(m["policy"]["envelope_factor"], 2);
    assert_eq!(m["policy"]["rtol"], 1e-3);
    m
}
fn values(d: &Value) -> Vec<f32> {
    assert_eq!(d["file"], "arrays.bin");
    let start = number(&d["offset"]);
    let len = number(&d["bytes"]);
    let bytes = &RAW[start..start.checked_add(len).unwrap()];
    assert_eq!(hash(bytes), d["sha256"]);
    let v: Vec<f32> = match d["dtype"].as_str().unwrap() {
        "f16le" => bytes
            .chunks_exact(2)
            .map(|b| f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect(),
        "f32le" => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        _ => panic!("unsupported source dtype"),
    };
    assert_eq!(
        v.len(),
        d["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(number)
            .product::<usize>()
    );
    assert!(v.iter().all(|v| v.is_finite()));
    v
}
#[test]
fn frozen_half_graph_descriptors_and_fixed_gates_are_complete() {
    let m = fixture();
    let mut spans = Vec::new();
    let mut total = 0;
    for c in m["cases"].as_array().unwrap() {
        assert_eq!(c["stages"].as_object().unwrap().len(), 63);
        for item in c["stages"].as_object().unwrap().values() {
            let main = values(&item["main"]);
            let wide = values(&item["wide"]);
            let env = main
                .iter()
                .zip(&wide)
                .map(|(&a, &b)| (a as f64 - b as f64).abs())
                .fold(0., f64::max);
            assert_eq!(item["envelope"].as_f64().unwrap(), env);
            assert_eq!(
                item["gate"]["atol"].as_f64().unwrap(),
                1e-4f64.max(2. * env)
            );
            assert_eq!(item["gate"]["rtol"].as_f64().unwrap(), 1e-3);
            for p in ["main", "wide"] {
                let d = &item[p];
                spans.push((number(&d["offset"]), number(&d["bytes"])));
                total += values(d).len();
            }
        }
    }
    spans.sort_unstable();
    let mut end = 0;
    for (start, len) in spans {
        assert_eq!(start, end);
        end += len;
    }
    assert_eq!((end, total), (RAW.len(), 1097888));
}
fn model() -> Model {
    let (c, w) = tests::inputs();
    Model::from_raw_weights(c, w, 2 * 1024 * 1024, LIMIT).unwrap()
}
fn options(diagnostics: bool, all_logits: bool) -> Options {
    Options {
        memory_limit_bytes: LIMIT,
        external_resident_bytes: EXTERNAL,
        all_logits,
        diagnostics,
    }
}
#[derive(Default)]
struct Metrics {
    groups: usize,
    values: usize,
    failures: usize,
    wide_exceedances: usize,
    max_error: f64,
    max_fraction: f64,
    wide_max_fraction: f64,
}
fn compare(m: &mut Metrics, item: &Value, indices: &[usize], actual: &[f32]) {
    let main = values(&item["main"]);
    let wide = values(&item["wide"]);
    assert_eq!(indices.len(), actual.len());
    let atol = item["gate"]["atol"].as_f64().unwrap();
    let rtol = item["gate"]["rtol"].as_f64().unwrap();
    m.groups += 1;
    for (&i, &x) in indices.iter().zip(actual) {
        assert!(x.is_finite());
        let x = x as f64;
        let a = main[i] as f64;
        let w = wide[i] as f64;
        let error = (x - a).abs();
        let fraction = error / (atol + rtol * a.abs());
        let wide_fraction = (x - w).abs() / (atol + rtol * w.abs());
        m.values += 1;
        m.failures += usize::from(fraction > 1.);
        m.wide_exceedances += usize::from(wide_fraction > 1.);
        m.max_error = m.max_error.max(error);
        m.max_fraction = m.max_fraction.max(fraction);
        m.wide_max_fraction = m.wide_max_fraction.max(wide_fraction);
    }
}
fn half_words(b: &Buffer<u16>) -> Vec<u16> {
    b.with_slice(|v| v.to_vec())
}
fn word_words(b: &Buffer<u32>) -> Vec<u32> {
    b.with_slice(|v| v.to_vec())
}
fn bytehash16(v: &[u16]) -> String {
    hash(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
}
fn bytehash32(v: &[u32]) -> String {
    hash(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
}
#[derive(Clone)]
struct State {
    cache: Value,
    histories: Vec<(Vec<u16>, Vec<u16>)>,
    published: Vec<u32>,
    tokens: Vec<u32>,
    rows: usize,
    ids: BTreeSet<usize>,
}
fn state(s: &Session<'_>, label: &str, result: &Result<()>, full_scratch: bool) -> State {
    if let Err(e) = s.cache.test_safe_read() {
        emit(
            json!({"kind":"unsafe_read_refused","label":label,"result":format!("{result:?}"),"completion":s.cache.test_completion(),"error":format!("{e:?}")}),
        );
        panic!("in-flight mapped read refused");
    }
    let cache = s.cache.test_observe().unwrap();
    let mut ids = BTreeSet::new();
    let histories: Vec<_> = s
        .histories
        .iter()
        .map(|h| (half_words(&h.old), half_words(&h.candidate)))
        .collect();
    let history_raw:Vec<_>=s.histories.iter().zip(&histories).map(|(h,(a,b))|{
        ids.insert(h.old.raw.contents().as_ptr() as usize);ids.insert(h.candidate.raw.contents().as_ptr() as usize);
        json!({"old_address":h.old.raw.contents().as_ptr() as usize,"candidate_address":h.candidate.raw.contents().as_ptr() as usize,"old":hex16(a),"candidate":hex16(b)})
    }).collect();
    let half:Vec<_>=[("hidden",&s.scratch.hidden),("norm",&s.scratch.norm),("branch",&s.scratch.branch),("bcx",&s.scratch.bcx),("u",&s.scratch.u),("z",&s.scratch.z),("q",&s.scratch.q),("work",&s.scratch.work),("k",&s.scratch.k),("k_norm",&s.scratch.k_norm),("v",&s.scratch.v),("gate",&s.scratch.gate),("up",&s.scratch.up),("activated",&s.scratch.activated)].into_iter().map(|(name,b)|{
        ids.insert(b.raw.contents().as_ptr() as usize);let v=half_words(b);
        json!({"name":name,"address":b.raw.contents().as_ptr() as usize,"bytes":b.raw.length(),"sha256":bytehash16(&v),"tail":hex16(&v[v.len()-GUARD..]),"full":if full_scratch {Some(hex16(&v))}else{None}})
    }).collect();
    let words:Vec<_>=[("scores",&s.scratch.scores),("probabilities",&s.scratch.probabilities),("token_upload",&s.scratch.tokens),("candidate",&s.candidate),("published",&s.published)].into_iter().map(|(name,b)|{
        ids.insert(b.raw.contents().as_ptr() as usize);let v=word_words(b);
        json!({"name":name,"address":b.raw.contents().as_ptr() as usize,"bytes":b.raw.length(),"sha256":bytehash32(&v),"tail":hex32(&v[v.len()-GUARD..]),"full":if full_scratch || matches!(name,"candidate"|"published") {Some(hex32(&v))}else{None}})
    }).collect();
    for b in cache["owners"][0].as_array().unwrap() {
        ids.insert(number(&b["address"]));
    }
    let snapshot_records:Vec<_>=s.snapshot_buffers().unwrap().into_iter().map(|(address,half,v)|{
        ids.insert(address);let guards=if half {4}else{7};
        json!({"address":address,"half":half,"bytes":v.len()*4,"sha256":bytehash32(&v),"tail":hex32(&v[v.len()-guards..]),"full":if full_scratch {Some(hex32(&v))}else{None}})
    }).collect();
    let published = word_words(&s.published);
    emit(
        json!({"kind":"state","label":label,"result":format!("{result:?}"),"phase_a":s.test_phase_a,"cache":cache,"histories":history_raw,"half_scratch":half,"words":words,"snapshot_records":snapshot_records,"token_physical":s.tokens,"published_rows":s.published_rows,"diagnostic_rows":s.diagnostic_rows}),
    );
    State {
        cache,
        histories,
        published,
        tokens: s.tokens.clone(),
        rows: s.published_rows,
        ids,
    }
}
fn guards(s: &Session<'_>, st: &State) {
    for b in [
        &s.scratch.hidden,
        &s.scratch.norm,
        &s.scratch.branch,
        &s.scratch.bcx,
        &s.scratch.u,
        &s.scratch.z,
        &s.scratch.q,
        &s.scratch.work,
        &s.scratch.k,
        &s.scratch.k_norm,
        &s.scratch.v,
        &s.scratch.gate,
        &s.scratch.up,
        &s.scratch.activated,
    ] {
        b.with_slice(|v| assert!(v[v.len() - GUARD..].iter().all(|&x| x == HALF_PAD)));
    }
    for b in [
        &s.scratch.scores,
        &s.scratch.probabilities,
        &s.scratch.tokens,
        &s.candidate,
        &s.published,
    ] {
        b.with_slice(|v| assert!(v[v.len() - GUARD..].iter().all(|&x| x == WORD_PAD)));
    }
    for (a, b) in &st.histories {
        for v in [a, b] {
            assert!(v[v.len() - GUARD..].iter().all(|&x| x == HALF_PAD));
        }
    }
    for b in st.cache["owners"][0].as_array().unwrap() {
        let v = b["words"].as_array().unwrap();
        assert!(
            v[v.len() - GUARD..]
                .iter()
                .all(|x| number(x) == usize::from(HALF_PAD))
        );
    }
    let status = st.cache["status"].as_array().unwrap();
    assert!(status[1..].iter().all(|v| number(v) == WORD_PAD as usize));
    s.snapshot_guards();
}
fn memory_proof(s: &Session<'_>, st: &State) {
    let half: usize = [
        &s.scratch.hidden,
        &s.scratch.norm,
        &s.scratch.branch,
        &s.scratch.bcx,
        &s.scratch.u,
        &s.scratch.z,
        &s.scratch.q,
        &s.scratch.work,
        &s.scratch.k,
        &s.scratch.k_norm,
        &s.scratch.v,
        &s.scratch.gate,
        &s.scratch.up,
        &s.scratch.activated,
    ]
    .iter()
    .map(|b| b.raw.length())
    .sum();
    let cache_numeric: usize = st.cache["owners"][0]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| number(&b["bytes"]))
        .sum::<usize>()
        + st.cache["status"].as_array().unwrap().len() * 4;
    let histories: usize = s
        .histories
        .iter()
        .map(|h| h.old.raw.length() + h.candidate.raw.length())
        .sum();
    let diag: usize = s
        .snapshot_buffers()
        .unwrap()
        .iter()
        .map(|(_, _, v)| v.len() * 4)
        .sum();
    let m = s.memory();
    emit(
        json!({"kind":"memory_owners","half_scratch":half,"cache_numeric":cache_numeric,"history_banks":histories,"diagnostic_gpu":diag,"memory":{"packed":m.packed_weights,"norms":m.direct_norms,"taps":m.convolution_coefficients,"zero_bias":m.zero_bias,"cache":m.cache,"history":m.convolution_history,"scratch":m.scratch,"logits":m.logits,"tokens":m.tokens,"diagnostic_gpu":m.diagnostic_gpu,"readback":m.diagnostic_readback_reserved,"metadata":m.metadata,"framework":m.shared_framework_allowance,"external":m.external_resident,"admitted":m.admitted,"cold_peak":m.cold_peak}}),
    );
    assert_eq!((half, cache_numeric, histories), (237764, 22616, 4152));
    assert_eq!(
        m.scratch,
        half + s.scratch.scores.raw.length() + s.scratch.probabilities.raw.length()
    );
    assert!(m.cache >= cache_numeric);
    assert_eq!(m.convolution_history, histories);
    assert_eq!(m.diagnostic_gpu, diag);
    assert_eq!(
        m.logits,
        s.candidate.raw.length() + s.published.raw.length()
    );
    assert_eq!(
        m.tokens,
        s.scratch.tokens.raw.length() + s.tokens.capacity() * 4
    );
    assert_eq!(
        m.admitted,
        [
            m.packed_weights,
            m.direct_norms,
            m.convolution_coefficients,
            m.zero_bias,
            m.cache,
            m.convolution_history,
            m.scratch,
            m.logits,
            m.tokens,
            m.diagnostic_gpu,
            m.diagnostic_readback_reserved,
            m.metadata,
            m.shared_framework_allowance,
            m.external_resident
        ]
        .iter()
        .sum::<usize>()
    );
    assert!(m.admitted <= LIMIT);
    assert!(m.cold_peak >= m.admitted);
}
fn old_kv(st: &State, kv: usize) -> Vec<u16> {
    st.cache["owners"][0][kv]["words"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| number(x) as u16)
        .collect()
}
fn published_unchanged(a: &State, b: &State, kv: bool) {
    assert_eq!(a.tokens, b.tokens);
    assert_eq!(a.published, b.published);
    assert_eq!(a.rows, b.rows);
    for owner in 0..2 {
        assert_eq!(a.histories[owner].0, b.histories[owner].0);
    }
    if kv {
        for k in 0..2 {
            assert_eq!(old_kv(a, k), old_kv(b, k));
        }
    }
}
fn resource_words(m: &Model) -> Value {
    let original = safetensors::SafeTensors::deserialize(ORIGINAL).unwrap();
    let mut weights = Vec::new();
    let mut push = |name: String, t: &Tensor| {
        let w = half_words(&t.buffer);
        let TensorStorage::Mlx(p) = &t.storage else {
            panic!("not packed")
        };
        let meta = p.test_metadata();
        weights.push(json!({"name":name,"rows":t.layout.rows,"columns":t.layout.columns,"address":t.buffer.raw.contents().as_ptr() as usize,"bytes":t.buffer.raw.length(),"sha256":bytehash16(&w),"original_sha256":hash(original.tensor(&format!("{name}.weight")).unwrap().data()),"metadata":meta.into_iter().zip(["scales","biases"]).map(|((address,v),suffix)|json!({"address":address,"bytes":v.len()*4,"sha256":bytehash32(&v),"original_sha256":hash(original.tensor(&format!("{name}.{suffix}")).unwrap().data())})).collect::<Vec<_>>()}));
    };
    push("model.embed_tokens".into(), &m.embedding);
    for (i, l) in m.layers.iter().enumerate() {
        for (name, t) in [("w1", &l.gate), ("w3", &l.up), ("w2", &l.down)] {
            push(format!("model.layers.{i}.feed_forward.{name}"), t);
        }
    }
    for (name, t) in [
        ("q_proj", &m.attention[0].q),
        ("k_proj", &m.attention[0].k),
        ("v_proj", &m.attention[0].v),
        ("out_proj", &m.attention[0].o),
    ] {
        push(format!("model.layers.1.self_attn.{name}"), t);
    }
    for (i, c) in m.convolution.iter().enumerate() {
        push(format!("model.layers.{}.conv.in_proj", i * 2), &c.bcx);
        push(format!("model.layers.{}.conv.out_proj", i * 2), &c.output);
    }
    let mut coefficients = Vec::new();
    let mut coefficient = |name: String, b: &Buffer<u32>| {
        let v = word_words(b);
        let tensor = original.tensor(&name).unwrap();
        let mut expected: Vec<u32> = tensor
            .data()
            .chunks_exact(2)
            .map(|x| {
                bf16::from_bits(u16::from_le_bytes(x.try_into().unwrap()))
                    .to_f32()
                    .to_bits()
            })
            .collect();
        expected.extend([WORD_PAD; GUARD]);
        coefficients.push(json!({"name":name,"address":b.raw.contents().as_ptr() as usize,"bytes":b.raw.length(),"words":hex32(&v),"sha256":bytehash32(&v),"original_widened_sha256":bytehash32(&expected)}));
    };
    coefficient("model.embedding_norm.weight".into(), &m.final_norm);
    for (i, l) in m.layers.iter().enumerate() {
        coefficient(
            format!("model.layers.{i}.operator_norm.weight"),
            &l.operator_norm,
        );
        coefficient(format!("model.layers.{i}.ffn_norm.weight"), &l.ffn_norm);
    }
    coefficient(
        "model.layers.1.self_attn.q_layernorm.weight".into(),
        &m.attention[0].q_norm,
    );
    coefficient(
        "model.layers.1.self_attn.k_layernorm.weight".into(),
        &m.attention[0].k_norm,
    );
    for (i, c) in m.convolution.iter().enumerate() {
        coefficient(
            format!("model.layers.{}.conv.conv.weight", i * 2),
            &c.coefficients,
        );
    }
    json!({"weights":weights,"coefficients":coefficients,"zero_bias":{"address":m.zero_bias.raw.contents().as_ptr() as usize,"bytes":m.zero_bias.raw.length(),"words":hex16(&half_words(&m.zero_bias))}})
}
fn original_resources(v: &Value) {
    assert_eq!(v["weights"].as_array().unwrap().len(), 18);
    let mut ids = BTreeSet::new();
    let mut packed = 0;
    for w in v["weights"].as_array().unwrap() {
        assert_eq!(w["sha256"], w["original_sha256"]);
        assert!(ids.insert(number(&w["address"])));
        packed += number(&w["bytes"]);
        for m in w["metadata"].as_array().unwrap() {
            assert_eq!(m["sha256"], m["original_sha256"]);
            assert!(ids.insert(number(&m["address"])));
            packed += number(&m["bytes"]);
        }
    }
    assert_eq!(packed, 746496);
    assert_eq!(v["coefficients"].as_array().unwrap().len(), 11);
    for c in v["coefficients"].as_array().unwrap() {
        assert!(ids.insert(number(&c["address"])));
        assert_eq!(c["sha256"], c["original_widened_sha256"]);
    }
    assert!(ids.insert(number(&v["zero_bias"]["address"])));
    assert_eq!(ids.len(), 66);
}
struct Capture {
    name: String,
    info: SnapshotInfo,
    values: Vec<f32>,
}
fn captures(s: &Session<'_>, label: &str) -> Vec<Capture> {
    let mut out = Vec::new();
    for i in 0..s.diagnostic_count() {
        let info = s.diagnostic_info(i).unwrap();
        let mut values = vec![0.; info.values()];
        s.copy_diagnostic(i, &mut values).unwrap();
        let name = info.layer.map_or_else(
            || info.stage.to_owned(),
            |l| format!("layers.{l}.{}", info.stage),
        );
        emit(
            json!({"kind":"stage","label":label,"slot":i,"name":name,"shape":info.shape,"rank":info.rank,"query_start":info.absolute_query_start,"key_start":info.absolute_key_start,"f32_words":hex32(&fbits(&values))}),
        );
        out.push(Capture { name, info, values });
    }
    out
}
fn math_indices(c: &Value, cap: &Capture, start: usize, n: usize) -> Vec<usize> {
    let source_tokens = c["tokens"].as_array().unwrap().len();
    let item = &c["stages"][&cap.name];
    let shape = item["shape"].as_array().unwrap();
    if cap.name.ends_with("attention_scores") || cap.name.ends_with("attention_probabilities") {
        assert_eq!(cap.info.shape, [4, n, start + n]);
        assert_eq!(cap.info.absolute_key_start, 0);
        (0..4)
            .flat_map(|h| {
                (0..n).flat_map(move |r| {
                    (0..start + n).map(move |k| (h * source_tokens + start + r) * source_tokens + k)
                })
            })
            .collect()
    } else {
        let width: usize = shape[1..].iter().map(number).product();
        let mut expected = vec![n];
        expected.extend(shape[1..].iter().map(number));
        assert_eq!(cap.info.rank, expected.len());
        assert_eq!(&cap.info.shape[..cap.info.rank], expected.as_slice());
        assert_eq!(cap.values.len(), n * width);
        (start * width..(start + n) * width).collect()
    }
}
fn state_compare(
    metrics: &mut Metrics,
    c: &Value,
    before: &State,
    after: &State,
    start: usize,
    n: usize,
) {
    for (owner, layer) in [(0, 0), (1, 2)] {
        for (stage, row, actual) in [
            ("history_before", start, &before.histories[owner].0),
            ("history_after", start + n - 1, &after.histories[owner].0),
        ] {
            let name = format!("layers.{layer}.{stage}");
            let xs: Vec<_> = actual[..2 * H]
                .iter()
                .map(|&b| f16::from_bits(b).to_f32())
                .collect();
            compare(
                metrics,
                &c["stages"][&name],
                &(row * 2 * H..(row + 1) * 2 * H).collect::<Vec<_>>(),
                &xs,
            );
        }
    }
    for (kv, name) in [
        (0, "layers.1.kv_keys_after"),
        (1, "layers.1.kv_values_after"),
    ] {
        let words = old_kv(after, kv);
        let end = start + n;
        let mut actual = Vec::new();
        let mut indices = Vec::new();
        for head in 0..2 {
            for row in 0..end {
                for f in 0..D {
                    actual.push(f16::from_bits(words[(head * CAP + row) * D + f]).to_f32());
                    indices.push(((end - 1) * 2 * CAP + head * CAP + row) * D + f);
                }
            }
        }
        compare(metrics, &c["stages"][name], &indices, &actual);
    }
}
fn exact_state_from_actual_operands(
    before: &State,
    after: &State,
    captures: &[Capture],
    start: usize,
    n: usize,
) {
    let find = |name: &str| &captures.iter().find(|x| x.name == name).unwrap().values;
    for (owner, layer) in [(0, 0), (1, 2)] {
        let u = find(&format!("layers.{layer}.u"));
        let mut expected = before.histories[owner].0.clone();
        if n == 1 {
            expected.copy_within(H..2 * H, 0);
            for f in 0..H {
                expected[H + f] = f16::from_f32(u[f]).to_bits();
            }
        } else {
            for j in 0..2 * H {
                expected[j] = f16::from_f32(u[(n - 2) * H + j]).to_bits();
            }
        }
        assert_eq!(expected, after.histories[owner].0);
    }
    for (kv, name) in [(0, "layers.1.k_rotary"), (1, "layers.1.v_projection")] {
        let input = find(name);
        let mut expected = old_kv(before, kv);
        for t in 0..n {
            for head in 0..2 {
                for f in 0..D {
                    expected[(head * CAP + start + t) * D + f] =
                        f16::from_f32(input[(t * 2 + head) * D + f]).to_bits();
                }
            }
        }
        assert_eq!(expected, old_kv(after, kv));
    }
}
fn completed(st: &State) {
    assert_eq!(st.cache["command"]["completed"], true);
    assert!(st.cache["command"]["error"].is_null());
    assert_eq!(st.cache["status"][0], 0);
}
fn result(metrics: &Metrics, label: &str) {
    emit(
        json!({"kind":"summary","label":label,"groups":metrics.groups,"values":metrics.values,"primary_main_failures":metrics.failures,"wide_diagnostic_exceedances":metrics.wide_exceedances,"max_error":metrics.max_error,"max_fraction":metrics.max_fraction,"wide_max_fraction":metrics.wide_max_fraction}),
    );
}

#[test]
#[ignore = "explicit Apple GPU tiny hybrid graph numerical/state qualification"]
fn resident_lfm2_all_stages_schedules_and_resources() {
    let fixture = fixture();
    let model = model();
    let original = resource_words(&model);
    emit(json!({"kind":"model","resources":original,"memory":format!("{:?}",model.memory())}));
    original_resources(&original);
    let mut metrics = Metrics::default();
    let mut states = Metrics::default();
    let mut schedule_count = 0;
    let mut token_count = 0;
    let mut causal: BTreeMap<String, Vec<u32>> = BTreeMap::new();
    for case in fixture["cases"].as_array().unwrap() {
        let tokens: Vec<u32> = case["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| number(v) as u32)
            .collect();
        let mut baseline: Option<Vec<u32>> = None;
        for (schedule, partition) in case["partitions"].as_array().unwrap().iter().enumerate() {
            schedule_count += 1;
            let mut session = model.session(options(true, true)).unwrap();
            let mut before = state(&session, "initial", &Ok(()), false);
            guards(&session, &before);
            memory_proof(&session, &before);
            let stable_ids = before.ids.clone();
            let mut start = 0;
            let mut all_logits = Vec::new();
            for part in partition.as_array().unwrap() {
                let n = number(part);
                token_count += n;
                let label = format!("{}/{schedule}/{start}/{n}", case["name"].as_str().unwrap());
                let outcome = session.forward(&tokens[start..start + n]);
                let after = state(&session, &label, &outcome, outcome.is_err());
                outcome.unwrap();
                let captured = captures(&session, &label);
                guards(&session, &after);
                completed(&after);
                assert_eq!(after.ids, stable_ids);
                assert_eq!(session.test_phase_a, Some((true, true)));
                assert_eq!(session.committed_tokens().unwrap(), &tokens[..start + n]);
                assert_eq!(captured.len(), 57);
                let names: BTreeSet<_> = captured.iter().map(|c| c.name.as_str()).collect();
                assert_eq!(names.len(), 57);
                let expected_names: BTreeSet<_> = case["stages"]
                    .as_object()
                    .unwrap()
                    .keys()
                    .filter(|name| {
                        !name.contains("history_before")
                            && !name.contains("history_after")
                            && !name.ends_with("kv_keys_after")
                            && !name.ends_with("kv_values_after")
                    })
                    .map(String::as_str)
                    .collect();
                assert_eq!(names, expected_names);
                for cap in &captured {
                    assert_eq!(cap.info.absolute_query_start, start);
                    let indices = math_indices(case, cap, start, n);
                    compare(
                        &mut metrics,
                        &case["stages"][&cap.name],
                        &indices,
                        &cap.values,
                    );
                    if cap.name.ends_with("attention_probabilities") {
                        for h in 0..4 {
                            for r in 0..n {
                                for k in start + r + 1..start + n {
                                    assert_eq!(
                                        cap.values[(h * n + r) * (start + n) + k].to_bits(),
                                        0
                                    );
                                }
                            }
                        }
                    }
                    // Causal rows/allowed score columns also form a prospective
                    // exact same-native schedule and future-tail comparison.
                    let width = if cap.name.ends_with("attention_scores")
                        || cap.name.ends_with("attention_probabilities")
                    {
                        0
                    } else {
                        cap.values.len() / n
                    };
                    for r in 0..n {
                        let values = if width == 0 {
                            (0..4)
                                .flat_map(|h| {
                                    (0..=start + r)
                                        .map(move |k| cap.values[(h * n + r) * (start + n) + k])
                                })
                                .collect::<Vec<_>>()
                        } else {
                            cap.values[r * width..(r + 1) * width].to_vec()
                        };
                        let key = format!(
                            "{}/{}/{}",
                            if case["name"] == "future_changed12" {
                                "context12"
                            } else {
                                case["name"].as_str().unwrap()
                            },
                            cap.name,
                            start + r
                        );
                        let words = numeric(&values);
                        if case["name"] != "future_changed12" || start + r + 1 < tokens.len() {
                            if let Some(previous) = causal.get(&key) {
                                assert_eq!(previous, &words, "same-native causal stage {key}");
                            } else {
                                causal.insert(key, words);
                            }
                        }
                    }
                }
                state_compare(&mut states, case, &before, &after, start, n);
                exact_state_from_actual_operands(&before, &after, &captured, start, n);
                let mut logits = vec![0.; n * V];
                session.copy_all_logits(&mut logits).unwrap();
                let saved = &captured.iter().find(|c| c.name == "logits").unwrap().values;
                assert_eq!(numeric(&logits), numeric(saved));
                let mut last = vec![0.; V];
                session.copy_logits(&mut last).unwrap();
                assert_eq!(numeric(&last), numeric(&logits[(n - 1) * V..]));
                all_logits.extend(numeric(&logits));
                start += n;
                before = after;
            }
            assert_eq!(start, tokens.len());
            if let Some(ref b) = baseline {
                assert_eq!(b, &all_logits);
            } else {
                baseline = Some(all_logits);
            }
        }
    }
    let final_resources = resource_words(&model);
    emit(json!({"kind":"model_final","resources":final_resources}));
    assert_eq!(original, final_resources);
    result(&metrics, "mathematical_stages");
    result(&states, "actual_boundary_states");
    assert_eq!(
        (schedule_count, token_count, metrics.groups, states.groups),
        (15, 133, 3135, 330)
    );
    assert_eq!((metrics.values, states.values), (1923728, 199936));
    assert_eq!(metrics.failures, 0);
    assert_eq!(states.failures, 0);
}

fn validator_preflight(m: &Model) {
    let left = half_buffer(&m.gpu, 2, H).unwrap();
    let right = half_buffer(&m.gpu, 2, H).unwrap();
    let status = word_buffer(&m.gpu, 1).unwrap();
    let short = Buffer::<u32>::new(&m.gpu, GUARD).unwrap();
    let mut alias = Buffer::<u32>::new(&m.gpu, 1 + GUARD).unwrap();
    alias.raw = left.raw.clone();
    let raw = m.gpu.command().unwrap();
    let command = Command {
        raw: &raw,
        trace: None,
    };
    let before = (half_words(&left), half_words(&right), word_words(&status));
    let results = [
        m.cache
            .encode_half_pair(&command, &left, &right, 0, 0, &status),
        m.cache
            .encode_half_pair(&command, &left, &right, 1, 2 * H, &status),
        m.cache
            .encode_half_pair(&command, &left, &right, 0, 2 * H + GUARD + 1, &status),
        m.cache
            .encode_half_pair(&command, &left, &right, usize::MAX, 1, &status),
        m.cache
            .encode_half_pair(&command, &left, &left, 0, 2 * H, &status),
        m.cache
            .encode_half_pair(&command, &left, &right, 0, 2 * H, &short),
        m.cache
            .encode_half_pair(&command, &left, &right, 0, 2 * H, &alias),
    ];
    let after = (half_words(&left), half_words(&right), word_words(&status));
    emit(
        json!({"kind":"history_validator_preflight","results":results.iter().map(|r|format!("{r:?}")).collect::<Vec<_>>(),"command_status":format!("{:?}",raw.status()),"left":hex16(&after.0),"right":hex16(&after.1),"status":hex32(&after.2)}),
    );
    assert!(results.iter().all(Result::is_err));
    assert_eq!(raw.status(), MTLCommandBufferStatus::NotEnqueued);
    assert_eq!(before, after);
}

#[test]
#[ignore = "explicit Apple GPU mixed history/KV rollback, replay and allocation qualification"]
fn resident_lfm2_mixed_rollback_reset_and_zero_allocation() {
    let m = model();
    let immutable = resource_words(&m);
    emit(json!({"kind":"model","resources":immutable,"memory":format!("{:?}",m.memory())}));
    original_resources(&immutable);
    validator_preflight(&m);
    let tokens = [1, 7, 13, 3, 9, 5, 11, 2, 17, 4, 6, 10];
    let mut baseline = m.session(options(true, true)).unwrap();
    let r = baseline.forward(&tokens);
    let base_state = state(&baseline, "baseline12", &r, r.is_err());
    r.unwrap();
    guards(&baseline, &base_state);
    completed(&base_state);
    let mut base = vec![0.; CAP * V];
    baseline.copy_all_logits(&mut base).unwrap();
    let mut session = m.session(options(true, true)).unwrap();
    let r = session.forward(&tokens[..3]);
    let initial = state(&session, "valid_prefix3", &r, r.is_err());
    r.unwrap();
    guards(&session, &initial);
    completed(&initial);
    for (label, bad) in [
        ("empty", &[][..]),
        ("token", &[64][..]),
        ("capacity", &tokens[..][..]),
    ] {
        let r = session.forward(bad);
        let after = state(&session, label, &r, false);
        assert!(matches!(r, Err(Error::Options(_))));
        published_unchanged(&initial, &after, true);
        assert_eq!(initial.cache, after.cache);
        guards(&session, &after);
    }
    // A malformed reset destination must not clear KV/other histories first.
    let len = session.histories[1].candidate.len;
    session.histories[1].candidate.len = 0;
    let reset = session.reset();
    session.histories[1].candidate.len = len;
    let after = state(&session, "reset_prevalidation_reject", &reset, false);
    assert!(reset.is_err());
    published_unchanged(&initial, &after, true);
    assert_eq!(initial.cache, after.cache);
    let faults = [
        TestFault::OmitHistory(0),
        TestFault::OmitHistory(1),
        TestFault::PartialHistory(0),
        TestFault::PartialHistory(1),
        TestFault::AfterLayer(0),
        TestFault::AfterLayer(1),
        TestFault::AfterLayer(2),
        TestFault::Head,
        TestFault::FirstCopy,
    ];
    for fault in faults {
        if fault == TestFault::FirstCopy {
            // Reset intentionally preserves inactive KV. Earlier full replay
            // already wrote these future tokens, so use fresh guarded storage
            // to make the partial-copy mutation independently observable.
            drop(session);
            session = m.session(options(true, true)).unwrap();
        }
        session.reset().unwrap();
        session.test_fault = TestFault::None;
        let r = session.forward(&tokens[..3]);
        let before = state(&session, "fault_prefix3", &r, r.is_err());
        r.unwrap();
        guards(&session, &before);
        session.test_fault = fault;
        let r = session.forward(&tokens[3..5]);
        let after = state(&session, &format!("fault_{fault:?}"), &r, true);
        assert!(r.is_err());
        guards(&session, &after);
        published_unchanged(&before, &after, fault != TestFault::FirstCopy);
        assert_eq!(session.diagnostic_rows, 0);
        assert!(session.diagnostic_info(0).is_err());
        assert_eq!(session.test_phase_a, Some((true, true)));
        if fault == TestFault::FirstCopy {
            assert!(session.is_poisoned());
            assert!(matches!(r, Err(Error::ReplayRequired)));
            assert_ne!(old_kv(&before, 0), old_kv(&after, 0));
            // FirstRunThenFail copies only head0 K's first contiguous run:
            // two rows from staging into committed positions3 and4.
            let mut expected_k = old_kv(&before, 0);
            expected_k[3 * D..5 * D].copy_from_slice(&old_kv(&after, 2)[..2 * D]);
            assert_eq!(expected_k, old_kv(&after, 0));
            assert_eq!(old_kv(&before, 1), old_kv(&after, 1));
            assert!(session.committed_tokens().is_err());
            assert!(session.copy_logits(&mut [0.; V]).is_err());
            assert!(session.copy_all_logits(&mut vec![0.; 3 * V]).is_err());
            let again = session.forward(&tokens[3..4]);
            let poison = state(&session, "poison_forward_reject", &again, false);
            assert!(matches!(again, Err(Error::ReplayRequired)));
            assert_eq!(after.cache, poison.cache);
        } else {
            assert!(!session.is_poisoned());
            assert!(matches!(r, Err(Error::Nonfinite)));
            assert_eq!(session.committed_tokens().unwrap(), &tokens[..3]);
            assert_ne!(after.cache["status"][0], 0);
        }
        let reset = session.reset();
        let empty = state(&session, "reset_for_replay", &reset, false);
        reset.unwrap();
        guards(&session, &empty);
        assert_eq!(number(&empty.cache["offset"]), 0);
        assert!(session.committed_tokens().unwrap().is_empty());
        assert!(session.copy_logits(&mut [0.; V]).is_err());
        for (a, b) in &empty.histories {
            assert!(a[..2 * H].iter().chain(&b[..2 * H]).all(|&x| x == 0));
        }
        // Reset does not pretend stale inactive physical KV slots are zero.
        assert_eq!(old_kv(&empty, 0), old_kv(&after, 0));
        assert_eq!(old_kv(&empty, 1), old_kv(&after, 1));
        session.test_fault = TestFault::None;
        let mut joined = Vec::new();
        for (start, n) in [(0, 5), (5, 7)] {
            let r = session.forward(&tokens[start..start + n]);
            let current = state(&session, "replay5_7", &r, r.is_err());
            r.unwrap();
            guards(&session, &current);
            completed(&current);
            let mut output = vec![0.; n * V];
            session.copy_all_logits(&mut output).unwrap();
            joined.extend(output);
        }
        assert_eq!(numeric(&joined), numeric(&base));
        let replay = state(&session, "replay_final", &Ok(()), false);
        for h in 0..2 {
            assert_eq!(replay.histories[h].0, base_state.histories[h].0);
        }
        for kv in 0..2 {
            assert_eq!(old_kv(&replay, kv), old_kv(&base_state, kv));
        }
    }
    // Last-row mode executes the ordinary public forward path, no test fault
    // instrumentation selected. Rust allocator scope includes both phase waits.
    let mut ordinary = m.session(options(false, false)).unwrap();
    let warm = ordinary.forward(&tokens[..1]);
    let warm_state = state(&ordinary, "ordinary_warmup", &warm, warm.is_err());
    warm.unwrap();
    guards(&ordinary, &warm_state);
    completed(&warm_state);
    ordinary.reset().unwrap();
    let mut allocations = Vec::new();
    for (i, &token) in tokens.iter().enumerate() {
        let measurement = Measurement::start();
        let r = ordinary.forward(&[token]);
        let count = measurement.finish();
        emit(
            json!({"kind":"allocation","call":i,"count":count,"result":format!("{r:?}"),"profile":if cfg!(debug_assertions){"debug_observation"}else{"release_rust_required_zero"}}),
        );
        let observed = state(&ordinary, "ordinary_decode", &r, r.is_err());
        r.unwrap();
        guards(&ordinary, &observed);
        completed(&observed);
        allocations.push(count);
        let mut row = [0.; V];
        ordinary.copy_logits(&mut row).unwrap();
        assert_eq!(numeric(&row), numeric(&base[i * V..(i + 1) * V]));
        assert!(ordinary.copy_all_logits(&mut row).is_err());
        assert_eq!(ordinary.diagnostic_count(), 0);
    }
    if !cfg!(debug_assertions) {
        assert_eq!(allocations, [0; CAP]);
    }
    // Exact session budget admission and its one-byte-under rejection are real
    // constructor checks, independently of sparse process-memory observations.
    for (diag, all) in [(false, false), (false, true), (true, true)] {
        let s = m.session(options(diag, all)).unwrap();
        let required = s.memory().admitted;
        drop(s);
        let mut exact = options(diag, all);
        exact.memory_limit_bytes = required;
        let accepted = m.session(exact).unwrap();
        emit(
            json!({"kind":"memory","diagnostics":diag,"all_logits":all,"memory":format!("{:?}",accepted.memory()),"required":required}),
        );
        drop(accepted);
        exact.memory_limit_bytes = required - 1;
        assert!(
            matches!(m.session(exact),Err(Error::Budget{required:r,limit:l}) if r==required && l+1==required)
        );
    }
    let final_resources = resource_words(&m);
    emit(json!({"kind":"model_final","resources":final_resources}));
    assert_eq!(immutable, final_resources);
    emit(
        json!({"kind":"state_summary","faults":9,"replay_chunks":18,"required_release_zero_allocations":12,"hardware_failure_injection":false,"partial_copy_is_real_copy_then_forced_protocol_error":true}),
    );
}
