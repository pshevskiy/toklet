use super::*;
use crate as engine;
#[path = "../../../tests/common/lfm2_scalar_fixture.rs"]
mod fixture;
use crate::model::gemma_reference::tests::Measurement;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}
fn model() -> Model<'static> {
    let (c, w) = fixture::inputs();
    Model::new(c, w).unwrap()
}
#[derive(Debug, PartialEq)]
struct State {
    kv: Vec<(Vec<u32>, Vec<u32>, usize, u64)>,
    hist: Vec<(Vec<u32>, usize, u64)>,
    tokens: Vec<u32>,
    published: Vec<u32>,
    offset: usize,
    generation: u64,
    rows: usize,
}
fn state(s: &Session<'_, '_>) -> State {
    State {
        kv: s
            .caches
            .iter()
            .map(|c| {
                let (k, v) = c.committed();
                (bits(k), bits(v), c.offset(), c.generation())
            })
            .collect(),
        hist: s
            .histories
            .iter()
            .map(|h| (bits(h.committed()), h.offset(), h.generation()))
            .collect(),
        tokens: s.tokens.clone(),
        published: bits(&s.scratch.published),
        offset: s.committed,
        generation: s.generation,
        rows: s.published_rows,
    }
}
#[derive(Default)]
struct Metrics {
    values: usize,
    observations: usize,
    max_abs: f64,
    max_fraction: f64,
}
fn compare(actual: &[f32], expected: &[f32], name: &str, m: &mut Metrics) {
    assert_eq!(actual.len(), expected.len(), "{name}");
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && b.is_finite());
        let error = (f64::from(a) - f64::from(b)).abs();
        let bound = 1e-5 + 1e-4 * f64::from(b).abs();
        m.values += 1;
        m.max_abs = m.max_abs.max(error);
        m.max_fraction = m.max_fraction.max(error / bound);
        assert!(
            error <= bound,
            "{name}[{i}] actual={a:?}({:08x}) source={b:?}({:08x}) error={error} limit={bound}",
            a.to_bits(),
            b.to_bits()
        );
    }
}
fn state_groups(
    s: &Session<'_, '_>,
    arrays: &BTreeMap<String, Vec<f32>>,
    row: usize,
    before: bool,
    m: &mut Metrics,
) {
    for (owner, layer) in [0, 2].into_iter().enumerate() {
        let name = format!(
            "layers.{layer}.history_{}",
            if before { "before" } else { "after" }
        );
        compare(
            s.histories[owner].committed(),
            &arrays[&name][row * 512..(row + 1) * 512],
            &name,
            m,
        );
        m.observations += 1;
    }
    if !before {
        for (name, actual) in [
            ("layers.1.kv_keys_after", s.caches[0].committed().0),
            ("layers.1.kv_values_after", s.caches[0].committed().1),
        ] {
            let source = &arrays[name][row * 2 * 12 * 64..(row + 1) * 2 * 12 * 64];
            for head in 0..2 {
                compare(
                    &actual[head * 12 * 64..head * 12 * 64 + s.committed * 64],
                    &source[head * 12 * 64..head * 12 * 64 + s.committed * 64],
                    name,
                    m,
                );
            }
            m.observations += 1;
        }
    }
}
fn stages(case: &Value) -> BTreeMap<String, Vec<f32>> {
    case["arrays"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(n, v)| (n.clone(), fixture::values(v)))
        .collect()
}
fn schedules(n: usize) -> Vec<Vec<usize>> {
    let mut all = vec![vec![n], vec![1; n]];
    for step in [3, 7] {
        let mut left = n;
        let mut v = vec![];
        while left > 0 {
            let take = step.min(left);
            v.push(take);
            left -= take;
        }
        all.push(v);
    }
    all.sort();
    all.dedup();
    all
}
#[test]
fn all_source_stages_actual_state_partitions_and_causality() {
    let metadata = fixture::metadata();
    let model = model();
    let mut metrics = Metrics::default();
    let mut one_shot_groups = 0;
    let mut one_shot_values = 0;
    let mut by_case = BTreeMap::new();
    for case in metadata["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let ids: Vec<_> = case["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        let arrays = stages(case);
        let expected_names: BTreeSet<_> = arrays
            .keys()
            .filter(|n| {
                !n.contains("history_")
                    && !n.contains("kv_keys_after")
                    && !n.contains("kv_values_after")
            })
            .cloned()
            .collect();
        assert_eq!(expected_names.len(), 57);
        let mut reference = None;
        let mut final_physical = None;
        for chunks in schedules(ids.len()) {
            let mut s = model.new_session(12, usize::MAX).unwrap();
            let mut output = vec![];
            let mut start = 0;
            for &n in &chunks {
                let end = start + n;
                state_groups(&s, &arrays, start, true, &mut metrics);
                let mut seen = BTreeSet::new();
                forward::execute::<true, _>(&mut s, &ids[start..end], &mut |o| {
                    let key = match o.layer {
                        Some(l) => format!("layers.{l}.{}", o.stage),
                        None => o.stage.to_string(),
                    };
                    assert!(seen.insert(key.clone()), "duplicate {key}");
                    assert_eq!(o.token_offset, start);
                    assert_eq!(o.rows, n);
                    let desc = &case["arrays"][&key];
                    let shape: Vec<_> = desc["shape"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_u64().unwrap() as usize)
                        .collect();
                    let source = &arrays[&key];
                    if let Some(key_start) = o.key_start {
                        assert_eq!(shape, [4, ids.len(), ids.len()]);
                        assert_eq!(o.heads, 4);
                        assert_eq!(key_start, 0);
                        assert_eq!(o.columns, end);
                        assert_eq!(o.values.len(), 4 * n * end);
                        for head in 0..4 {
                            for row in 0..n {
                                let a =
                                    &o.values[(head * n + row) * end..(head * n + row + 1) * end];
                                let base = (head * ids.len() + start + row) * ids.len();
                                compare(a, &source[base..base + end], &key, &mut metrics);
                                if o.stage == "attention_probabilities" {
                                    for value in &a[start + row + 1..] {
                                        assert_eq!(value.to_bits(), 0, "masked +0");
                                    }
                                }
                            }
                        }
                    } else {
                        assert_eq!(shape[0], ids.len());
                        assert_eq!(shape[1..].iter().product::<usize>(), o.columns);
                        assert_eq!(o.heads, 1);
                        assert_eq!(o.values.len(), (n - 1) * o.row_stride + o.columns);
                        for row in 0..n {
                            compare(
                                &o.values[row * o.row_stride..row * o.row_stride + o.columns],
                                &source[(start + row) * o.columns..(start + row + 1) * o.columns],
                                &key,
                                &mut metrics,
                            );
                        }
                    }
                    metrics.observations += 1;
                    if chunks.len() == 1 {
                        one_shot_groups += 1;
                        one_shot_values += o.rows * o.columns * o.heads;
                    }
                    Ok(())
                })
                .unwrap();
                assert_eq!(seen, expected_names);
                state_groups(&s, &arrays, end - 1, false, &mut metrics);
                output.extend_from_slice(s.logits());
                assert_eq!(s.tokens(), &ids[..end]);
                start = end;
            }
            if let Some(ref r) = reference {
                assert_eq!(bits(&output), *r, "partition {name} {chunks:?}");
            } else {
                reference = Some(bits(&output));
            }
            let actual_physical = (
                s.caches
                    .iter()
                    .map(|cache| {
                        let (k, v) = cache.committed();
                        (bits(k), bits(v))
                    })
                    .collect::<Vec<_>>(),
                s.histories
                    .iter()
                    .map(|history| bits(history.committed()))
                    .collect::<Vec<_>>(),
            );
            if let Some(expected) = &final_physical {
                assert_eq!(
                    &actual_physical, expected,
                    "exact partition final KV/history"
                );
            } else {
                final_physical = Some(actual_physical);
            }
            let before = state(&s);
            assert!(s.forward(&[]).is_err());
            assert_eq!(state(&s), before);
            s.reset().unwrap();
            let mut ordinary = vec![];
            for chunk in ids.chunks(3) {
                ordinary.extend_from_slice(s.forward(chunk).unwrap());
            }
            assert_eq!(bits(&ordinary), bits(&output), "ordinary/reset/replay");
        }
        by_case.insert(name.to_owned(), reference.unwrap());
    }
    assert_eq!(one_shot_groups, 228);
    assert_eq!(one_shot_values, 405584);
    let cases = metadata["cases"].as_array().unwrap();
    let a = cases[2]["name"].as_str().unwrap();
    let b = cases[3]["name"].as_str().unwrap();
    assert_eq!(&by_case[a][..11 * 64], &by_case[b][..11 * 64]);
    println!(
        "LFM2_SCALAR_SOURCE values={} observations={} full_math_groups={} full_math_values={} max_abs={} max_fraction={}",
        metrics.values,
        metrics.observations,
        one_shot_groups,
        one_shot_values,
        metrics.max_abs,
        metrics.max_fraction
    );
}
#[test]
fn failures_retry_and_fallible_trace_preserve_every_public_owner() {
    let m = model();
    for fault in 0..5 {
        let mut s = m.new_session(12, usize::MAX).unwrap();
        s.forward(&[1, 7, 13]).unwrap();
        let before = state(&s);
        s.fault = Some(fault);
        assert!(s.forward(&[3, 9]).is_err());
        assert_eq!(state(&s), before);
        s.fault = None;
        let result = bits(s.forward(&[3, 9]).unwrap());
        let mut fresh = m.new_session(12, usize::MAX).unwrap();
        fresh.forward(&[1, 7, 13]).unwrap();
        assert_eq!(result, bits(fresh.forward(&[3, 9]).unwrap()));
        assert_eq!(state(&s), state(&fresh));
    }
    let mut s = m.new_session(12, usize::MAX).unwrap();
    s.forward(&[1]).unwrap();
    let before = state(&s);
    let mut count = 0;
    let error = forward::execute::<true, _>(&mut s, &[7], &mut |_| {
        count += 1;
        if count == 57 {
            Err(Error::InvalidConfig("external observer failure"))
        } else {
            Ok(())
        }
    });
    assert!(error.is_err());
    assert_eq!(count, 57);
    assert_eq!(state(&s), before);
    for ids in [&[64][..], &[1; 12][..], &[][..]] {
        assert!(s.forward(ids).is_err());
        assert_eq!(state(&s), before);
    }
}
fn publish(
    s: &mut Session<'_, '_>,
    kv: &[crate::model::window_cache::PreparedKv<'_>],
    hist: &[crate::model::short_conv3::Prepared<'_>],
    tokens: &[u32],
    logits: &[f32],
) -> Result<(), Error> {
    forward::Publication {
        caches: &mut s.caches,
        histories: &mut s.histories,
        tokens: &mut s.tokens,
        published: &mut s.scratch.published,
        committed: &mut s.committed,
        generation: &mut s.generation,
        rows: &mut s.published_rows,
        capacity: 12,
        chunk: s.chunk,
        vocab: 64,
    }
    .commit(kv, hist, tokens, logits)
}
#[test]
fn mixed_publication_prevalidates_both_families_and_all_destinations() {
    let m = model();
    let mut s = m.new_session(12, usize::MAX).unwrap();
    let mut other = m.new_session(12, usize::MAX).unwrap();
    let k = vec![0.25; 128];
    let v = vec![0.5; 128];
    let u = vec![0.125; 256];
    let coeff = vec![0.5; 768];
    let mut candidates = [vec![0.; 512], vec![0.; 512], vec![0.; 512]];
    let mut work = vec![0.; 512];
    let mut out = vec![0.; 256];
    let [a, b, c] = &mut candidates;
    let f0 = s.histories[0]
        .prepare_chunk(&u, &coeff, a, &mut work, &mut out, 1)
        .unwrap();
    let f1 = s.histories[1]
        .prepare_chunk(&u, &coeff, b, &mut work, &mut out, 1)
        .unwrap();
    let foreign = other.histories[1]
        .prepare_chunk(&u, &coeff, c, &mut work, &mut out, 1)
        .unwrap();
    let fk = s.caches[0].prepare(&k, &v, 1).unwrap();
    let foreign_k = other.caches[0].prepare(&k, &v, 1).unwrap();
    let before = state(&s);
    assert!(publish(&mut s, &[fk], &[f0, foreign], &[1], &[0.; 64]).is_err());
    assert_eq!(state(&s), before);
    // Re-prepare reusable candidates for each ownership ordering; no mutation.
    let f0 = s.histories[0]
        .prepare_chunk(&u, &coeff, a, &mut work, &mut out, 1)
        .unwrap();
    let fk = s.caches[0].prepare(&k, &v, 1).unwrap();
    assert!(publish(&mut s, &[foreign_k], &[f0, f1], &[1], &[0.; 64]).is_err());
    assert_eq!(state(&s), before);
    let f0 = s.histories[0]
        .prepare_chunk(&u, &coeff, a, &mut work, &mut out, 1)
        .unwrap();
    let f1 = s.histories[1]
        .prepare_chunk(&u, &coeff, b, &mut work, &mut out, 1)
        .unwrap();
    assert!(publish(&mut s, &[fk], &[f1, f0], &[1], &[0.; 64]).is_err());
    assert_eq!(state(&s), before);
    for mode in 0..5 {
        let f0 = s.histories[0]
            .prepare_chunk(&u, &coeff, a, &mut work, &mut out, 1)
            .unwrap();
        let f1 = s.histories[1]
            .prepare_chunk(&u, &coeff, b, &mut work, &mut out, 1)
            .unwrap();
        let fk = s.caches[0].prepare(&k, &v, 1).unwrap();
        let mut logits = [0.; 64];
        if mode == 0 {
            logits[63] = f32::NAN;
        }
        let result = match mode {
            0 => publish(&mut s, &[fk], &[f0, f1], &[1], &logits),
            1 => publish(&mut s, &[fk], &[f0], &[1], &logits),
            2 => publish(&mut s, &[], &[f0, f1], &[1], &logits),
            3 => publish(&mut s, &[fk], &[f0, f1], &[1, 2], &[0.; 128]),
            _ => publish(&mut s, &[fk], &[f0, f1], &[64], &logits),
        };
        assert!(result.is_err());
        assert_eq!(state(&s), before);
    }
    let f0 = s.histories[0]
        .prepare_chunk(&u, &coeff, a, &mut work, &mut out, 1)
        .unwrap();
    let f1 = s.histories[1]
        .prepare_chunk(&u, &coeff, b, &mut work, &mut out, 1)
        .unwrap();
    let fk = s.caches[0].prepare(&k, &v, 1).unwrap();
    publish(&mut s, &[fk], &[f0, f1], &[1], &[0.; 64]).unwrap();
    assert_eq!(s.tokens(), [1]);
    assert_eq!(s.generation, 1);
    other.reset().unwrap();
}
#[test]
fn reset_and_generation_exhaustion_leave_full_physical_prefix_valid() {
    let m = model();
    let mut s = m.new_session(12, usize::MAX).unwrap();
    s.forward(&[1, 7, 13]).unwrap();
    let before = state(&s);
    s.histories[1].set_generation_for_test(u64::MAX);
    let broken = state(&s);
    assert!(s.reset().is_err());
    assert_eq!(state(&s), broken);
    assert!(s.forward(&[3]).is_err());
    assert_eq!(state(&s), broken);
    s.histories[1].set_generation_for_test(s.generation);
    s.reset().unwrap();
    assert!(s.tokens().is_empty() && s.logits().is_empty());
    assert!(
        s.histories
            .iter()
            .all(|h| h.committed().iter().all(|x| x.to_bits() == 0))
    );
    for (i, k) in s.caches.iter().enumerate() {
        assert_eq!(bits(k.committed().0), before.kv[i].0);
        assert_eq!(bits(k.committed().1), before.kv[i].1);
        assert_eq!(k.offset(), 0);
    }
    s.generation = u64::MAX;
    for k in &mut s.caches {
        k.set_generation_for_test(u64::MAX);
    }
    for h in &mut s.histories {
        h.set_generation_for_test(u64::MAX);
    }
    let before = state(&s);
    assert!(s.reset().is_err());
    assert!(s.forward(&[1]).is_err());
    assert_eq!(state(&s), before);
}
#[test]
fn checked_geometry_exact_budget_and_allocation_free_warmed_step() {
    let m = model();
    let required = m.model_bytes() + m.planned_session_bytes(12).unwrap();
    assert!(
        matches!(m.new_session(12,required-1),Err(crate::session::Error::MemoryBudget{required:r,limit:l})if r==required&&l==required-1)
    );
    let mut s = m.new_session(12, required).unwrap();
    assert_eq!(s.allocated_bytes().unwrap() + m.model_bytes(), required);
    assert!(m.new_session(0, usize::MAX).is_err());
    assert!(m.new_session(13, usize::MAX).is_err());
    let (c, mut w) = fixture::inputs();
    w.convolution[1].coefficients.pop();
    assert!(Model::new(c, w).is_err());
    let (c, mut w) = fixture::inputs();
    w.layers[2].kind = Kind::Convolution(0);
    assert!(Model::new(c, w).is_err());
    let (mut c, w) = fixture::inputs();
    c.kv_heads = 1;
    assert!(Model::new(c, w).is_err());
    let (c, mut w) = fixture::inputs();
    w.final_gamma[1] = bf16::NAN;
    assert!(Model::new(c, w).is_err());
    s.forward(&[1, 7, 13]).unwrap();
    s.reset().unwrap();
    let pointers = |s: &Session<'_, '_>| {
        let mut x: Vec<_> = [
            &s.scratch.hidden,
            &s.scratch.next,
            &s.scratch.norm,
            &s.scratch.bcx,
            &s.scratch.u,
            &s.scratch.conv,
            &s.scratch.context,
            &s.scratch.projected,
            &s.scratch.q,
            &s.scratch.qn,
            &s.scratch.k,
            &s.scratch.kn,
            &s.scratch.v,
            &s.scratch.gate,
            &s.scratch.up,
            &s.scratch.gated,
            &s.scratch.scores,
            &s.scratch.probabilities,
            &s.scratch.logits,
            &s.scratch.published,
            &s.scratch.work,
        ]
        .map(|v| v.as_ptr() as usize)
        .to_vec();
        for owner in &s.caches {
            let (k, v) = owner.committed();
            x.extend([k.as_ptr() as usize, v.as_ptr() as usize]);
        }
        for owner in &s.histories {
            x.push(owner.committed().as_ptr() as usize);
        }
        for owner in &s.kv_stage {
            x.extend([owner.k.as_ptr() as usize, owner.v.as_ptr() as usize]);
        }
        for owner in &s.conv_stage {
            x.push(owner.as_ptr() as usize);
        }
        x.push(s.tokens.as_ptr() as usize);
        x.sort();
        x
    };
    let before = pointers(&s);
    for ids in [&[1, 7, 13][..], &[3][..], &[9, 5][..]] {
        let measure = Measurement::start();
        let result = s.forward(ids);
        let allocations = measure.finish();
        result.unwrap();
        assert_eq!(allocations, 0);
        assert_eq!(pointers(&s), before);
    }
    let measure = Measurement::start();
    let result = s.reset();
    let allocations = measure.finish();
    result.unwrap();
    assert_eq!(allocations, 0);
    assert_eq!(pointers(&s), before);
}
