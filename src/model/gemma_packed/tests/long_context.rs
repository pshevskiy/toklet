//! Explicit scalar witness context, checked against a pre-target full-sequence oracle.
use super::*;
use crate::model::gemma_reference::tests::Measurement;
use serde_json::Value;
use sha2::{Digest, Sha256};

const MANIFEST: &[u8] =
    include_bytes!("../../../../tests/fixtures/gemma-long-witness-v1/manifest.json");
const RAW: &[u8] = include_bytes!("../../../../tests/fixtures/gemma-long-witness-v1/arrays.bin");

pub(crate) fn oracle() -> Value {
    assert_eq!(
        format!("{:x}", Sha256::digest(MANIFEST)),
        "bee93a03eb8f3ff6fe1e46919621d8702c1f4eb1a7ada850edb6a987fc750465"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(RAW)),
        "2940891fd6a32d22c7fa574aad633d8adc3681a1281c23722a97e1111e938ec3"
    );
    serde_json::from_slice(MANIFEST).unwrap()
}

pub(crate) fn ids(oracle: &Value) -> Vec<u32> {
    oracle["tokens"]["original"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect()
}

pub(crate) fn check(
    oracle: &Value,
    case: &str,
    name: &str,
    values: &[f32],
    before: usize,
    count: usize,
) {
    let mut emitted = false;
    for contract in ["math", "staged_fp32"] {
        let item = oracle["arrays"]
            .as_array()
            .unwrap()
            .iter()
            .find(|x| x["sequence"] == "original" && x["contract"] == contract && x["name"] == name)
            .unwrap();
        let shape: Vec<_> = item["source_shape"]
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
        let heads: usize = shape[..axis].iter().product();
        let stride: usize = shape[axis + 1..].iter().product();
        assert_eq!(values.len(), heads * count * stride);
        let positions = item["positions"].as_array().unwrap();
        let selected: Vec<_> = positions
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                let p = p.as_u64().unwrap() as usize;
                (before..before + count).contains(&p).then_some((i, p))
            })
            .collect();
        let width = if contract == "math" { 8 } else { 4 };
        let offset = item["offset"].as_u64().unwrap() as usize;
        let mut actual = Vec::new();
        let mut wanted = Vec::new();
        for head in 0..heads {
            for &(index, position) in &selected {
                for feature in 0..stride {
                    let start =
                        offset + ((head * positions.len() + index) * stride + feature) * width;
                    wanted.push(if width == 8 {
                        f64::from_le_bytes(RAW[start..start + 8].try_into().unwrap())
                    } else {
                        f64::from(f32::from_le_bytes(
                            RAW[start..start + 4].try_into().unwrap(),
                        ))
                    });
                    actual.push(values[(head * count + position - before) * stride + feature]);
                }
            }
        }
        if !emitted && !selected.is_empty() {
            println!(
                "GEMMA_LONG_NATIVE_STATE {}",
                serde_json::json!({"case":case,"name":name,"positions":selected.iter().map(|x|x.1).collect::<Vec<_>>(),"heads":heads,"stride":stride,"values":actual})
            );
            emitted = true;
        }
        close(&actual, &wanted);
    }
}

#[test]
fn explicit_witness_context_keeps_default_and_call_limits_separate() {
    let data = fixture::Fixture::new();
    for capacity in [0, 1026, usize::MAX] {
        assert!(Model::new_witness_with_context(data.weights(), capacity).is_err());
    }
    assert_eq!(Model::new(data.weights()).unwrap().layout.positions, 32);
    for capacity in [1, 3, 32, 33, 511, 512, 513, 1024, 1025] {
        let model = Model::new_witness_with_context(data.weights(), capacity).unwrap();
        let planned = model.planned_session_bytes().unwrap();
        let session = model.new_session(model.model_bytes() + planned).unwrap();
        assert_eq!(session.capacity(), capacity);
        assert_eq!(session.chunk_capacity(), capacity.min(32));
        assert_eq!(session.allocated_bytes().unwrap(), planned);
        for chunk in [0, 33, usize::MAX] {
            let measured = Measurement::start();
            let rejected = model
                .new_session_with_capacity(capacity, chunk, usize::MAX)
                .is_err();
            assert_eq!(measured.finish(), 0);
            assert!(rejected);
        }
    }
}

#[test]
#[ignore = "explicit long scalar numerical/state qualification"]
fn frozen_long_logits_kv_boundaries_and_failure_retry() {
    let data = fixture::Fixture::new();
    let oracle = oracle();
    let ids = ids(&oracle);
    for (capacity, chunk, uneven) in [
        (511, 32, true),
        (512, 32, true),
        (513, 32, true),
        (1024, 32, true),
        (1025, 1, false),
        (1025, 3, false),
        (1025, 32, false),
        (1025, 32, true),
    ] {
        let case = format!("c{capacity}-chunk{chunk}-uneven{uneven}");
        let model = Model::new_witness_with_context(data.weights(), capacity).unwrap();
        let planned = model
            .planned_session_bytes_with_capacity(capacity, chunk)
            .unwrap();
        let measured = Measurement::start();
        assert!(
            model
                .new_session_with_capacity(capacity, chunk, model.model_bytes() + planned - 1)
                .is_err()
        );
        assert_eq!(measured.finish(), 0);
        let mut session = model
            .new_session_with_capacity(capacity, chunk, model.model_bytes() + planned)
            .unwrap();
        let addresses = pointers(&session);
        let mut before = 0;
        let mut turn = 0;
        println!(
            "GEMMA_LONG_NATIVE_MEMORY {}",
            serde_json::json!({"case":case,"capacity":capacity,"chunk":chunk,"planned":planned,"actual":session.allocated_bytes().unwrap(),"model_bytes":model.model_bytes(),
            "session_header_bytes":std::mem::size_of_val(&session),
            "route_heap_bytes":session.plan.allocated_metadata_bytes()-std::mem::size_of::<SharedKvPlan>(),
            "history_bytes":session.history.capacity()*4,
            "cache_owner_capacity":session.caches.capacity(),"cache_owner_bytes":std::mem::size_of::<WindowKvCache>(),
            "staging_owner_capacity":session.staging.capacity(),"staging_owner_bytes":std::mem::size_of::<Staging>(),
            "cache_payload_bytes":session.caches.iter().map(|c|{let(k,v)=c.committed();(k.len()+v.len())*4}).sum::<usize>(),
            "staging_payload_bytes":session.staging.iter().map(|s|(s.k.capacity()+s.v.capacity())*4).sum::<usize>(),
            "scratch_elements":session.scratch.capacities()})
        );
        while before < capacity {
            let next_cut = [511, 512, 513, 1024, 1025, capacity]
                .into_iter()
                .filter(|&x| x > before)
                .min()
                .unwrap();
            let count = (if uneven {
                [31, 7, 1, 32, 3][turn % 5]
            } else {
                chunk
            })
            .min(chunk)
            .min(next_cut - before)
            .min(capacity - before);
            if capacity == 1025
                && uneven
                && oracle["selected_positions"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|x| (before..before + count).contains(&(x.as_u64().unwrap() as usize)))
            {
                let prior = snapshot(&session);
                for layer in 0..LAYERS {
                    session.fault = Some(layer);
                    let measured = Measurement::start();
                    let failed = session.forward(&ids[before..before + count]);
                    let allocations = measured.finish();
                    assert!(matches!(
                        failed,
                        Err(Error::InvalidConfig("injected Gemma layer failure"))
                    ));
                    assert_eq!(allocations, 0);
                    assert_eq!(snapshot(&session), prior);
                    let d = dimension(layer);
                    let s = &session.scratch;
                    let prefix = format!("layer.{layer}.");
                    for (name, values) in [
                        ("output", &s.hidden[..count * HIDDEN]),
                        ("self_attn.o_proj", &s.attention[..count * HIDDEN]),
                        ("self_attn.q_norm", &s.q_norm[..count * HEADS * d]),
                        ("post_per_layer_input_norm", &s.dense_norm[..count * HIDDEN]),
                        ("per_layer_projection", &s.dense_work[..count * HIDDEN]),
                        ("per_layer_input_gate", &s.gate[..count * PLE]),
                        ("mlp.up_proj", &s.up[..count * intermediate(layer)]),
                    ] {
                        check(
                            &oracle,
                            &case,
                            &(prefix.clone() + name),
                            values,
                            before,
                            count,
                        );
                    }
                    check(
                        &oracle,
                        &case,
                        &(prefix.clone() + "q_rotary"),
                        &unheads(&s.q[..count * HEADS * d], HEADS, count, d),
                        before,
                        count,
                    );
                    check(
                        &oracle,
                        &case,
                        &(prefix + "context"),
                        &unheads(&s.context[..count * HEADS * d], HEADS, count, d),
                        before,
                        count,
                    );
                }
                session.fault = None;
            }
            if [511, 512, 1024].contains(&before) {
                let prior = snapshot(&session);
                session.fault = Some(LAYERS + 2);
                let measured = Measurement::start();
                let failed = session.forward(&ids[before..before + count]);
                let allocations = measured.finish();
                assert!(matches!(
                    failed,
                    Err(Error::InvalidConfig("injected Gemma final failure"))
                ));
                assert_eq!(allocations, 0);
                assert_eq!(snapshot(&session), prior);
                session.fault = None;
            }
            let measured = Measurement::start();
            let rows = session.forward(&ids[before..before + count]).unwrap();
            let allocations = measured.finish();
            assert_eq!(allocations, 0);
            check(&oracle, &case, "logits", rows, before, count);
            for producer in 0..2 {
                let d = dimension(producer);
                let stage = &session.staging[producer];
                check(
                    &oracle,
                    &case,
                    &format!("layer.{producer}.k_rotary"),
                    &stage.k[..count * d],
                    before,
                    count,
                );
                check(
                    &oracle,
                    &case,
                    &format!("layer.{producer}.self_attn.v_norm"),
                    &stage.v[..count * d],
                    before,
                    count,
                );
            }
            before += count;
            turn += 1;
        }
        assert_eq!(session.tokens(), &ids[..capacity]);
        assert_eq!(pointers(&session), addresses);
        for producer in 0..2 {
            let cache = &session.caches[producer];
            let d = dimension(producer);
            let (k, v) = cache.committed();
            for slot in 0..(if producer == 0 {
                4.min(capacity)
            } else {
                capacity
            }) {
                if let Some(position) = cache.position_in_slot(slot) {
                    check(
                        &oracle,
                        &case,
                        &format!("layer.{producer}.k_rotary"),
                        &k[slot * d..(slot + 1) * d],
                        position,
                        1,
                    );
                    check(
                        &oracle,
                        &case,
                        &format!("layer.{producer}.self_attn.v_norm"),
                        &v[slot * d..(slot + 1) * d],
                        position,
                        1,
                    );
                }
            }
        }
        let full = snapshot(&session);
        assert!(
            matches!(session.forward(&[1]),Err(Error::Capacity{offset,tokens:1,limit})if offset==capacity&&limit==capacity)
        );
        assert_eq!(snapshot(&session), full);
        let measured = Measurement::start();
        session.reset().unwrap();
        assert_eq!(measured.finish(), 0);
        assert!(session.tokens().is_empty() && session.logits().is_empty());
        assert_eq!(pointers(&session), addresses);
        check(
            &oracle,
            &case,
            "logits",
            session.forward(&ids[..chunk.min(capacity)]).unwrap(),
            0,
            chunk.min(capacity),
        );
        println!(
            "GEMMA_LONG_NATIVE_COMPLETE {}",
            serde_json::json!({"case":case,"capacity":capacity,"chunks":turn,"zero_forward_allocations":true,"failure_retry":true,"overflow_atomic":true,"reset_replay":true,"pointers_stable":true})
        );
    }
}
