//! Two-head W512/full state witness, including multirow wrap-straddling appends.
use super::*;
use crate::model::gemma_reference::tests::Measurement;

fn digest(values: &[f32]) -> String {
    let mut h = Sha256::new();
    for value in values {
        h.update(value.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn value(producer: usize, head: usize, position: usize, feature: usize, v: bool) -> f32 {
    let x = (producer * 100000 + head * 10000 + position * 4 + feature + 1) as f32;
    if v { -x } else { x }
}

#[test]
#[ignore = "explicit W512 structural context/state qualification"]
fn frozen_two_head_w512_segments_routes_and_wrap_commits() {
    let raw = include_bytes!("../../../../tests/fixtures/gemma-long-capacity-v1/expected.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        "ffeb7e3c375f061de09cc4f47b232dbadf98bcb168e37176a5107f411eaa8f29"
    );
    let data: Value = serde_json::from_slice(raw).unwrap();
    let geometry: Vec<_> = (0..35)
        .map(|i| LayerKvGeometry {
            kind: if i % 5 == 4 {
                KvKind::Full
            } else {
                KvKind::Sliding
            },
            heads: 2,
            head_dim: if i % 5 == 4 { 3 } else { 2 },
        })
        .collect();
    let wide = SharedKvPlan::new(&geometry, 20, 512, 1025, 1025).unwrap();
    let mut views = 0;
    let mut straddles = 0;
    for (case_index, case) in data["cases"].as_array().unwrap().iter().enumerate() {
        let capacity = case["capacity"].as_u64().unwrap() as usize;
        let plan = wide.narrowed(capacity).unwrap();
        let mut caches = plan.new_caches().unwrap();
        let addresses: Vec<_> = caches
            .iter()
            .map(|c| {
                let (k, v) = c.committed();
                (k.as_ptr(), v.as_ptr())
            })
            .collect();
        for step in case["steps"].as_array().unwrap() {
            let before = step["before"].as_u64().unwrap() as usize;
            let count = step["count"].as_u64().unwrap() as usize;
            if before < 512 && before + count > 512 || before < 1024 && before + count > 1024 {
                straddles += 1;
            }
            let input: Vec<_> = (0..15)
                .map(|producer| {
                    let dims = geometry[producer].head_dim;
                    let load = |v| {
                        let mut result = Vec::new();
                        for head in 0..2 {
                            for position in before..before + count {
                                for feature in 0..dims {
                                    result.push(value(producer, head, position, feature, v));
                                }
                            }
                        }
                        result
                    };
                    (load(false), load(true))
                })
                .collect();
            let frames: Vec<_> = input
                .iter()
                .enumerate()
                .map(|(i, (k, v))| plan.prepare_producer(i, &caches, k, v, count).unwrap())
                .collect();
            for (layer, layer_geometry) in geometry.iter().enumerate() {
                let producer = plan.producer_for(layer).unwrap();
                assert_eq!(producer, data["routes"][layer].as_u64().unwrap() as usize);
                let expected = &step["producers"][producer];
                let view = plan.view(layer, &caches, &frames).unwrap();
                let mut k = Vec::new();
                let mut v = Vec::new();
                for head in 0..2 {
                    for span in view.head(head).unwrap() {
                        k.extend_from_slice(span.k);
                        v.extend_from_slice(span.v);
                    }
                }
                let dims = layer_geometry.head_dim;
                let segments:Vec<_>=view.segments().iter().filter(|s|s.keys>0).map(|s|{
                    let compact=|x:&[f32]|{let mut values=Vec::new();for head in 0..2 {values.extend_from_slice(&x[head*s.head_stride..head*s.head_stride+s.keys*dims]);}digest(&values)};
                    serde_json::json!({"start":s.key_start,"keys":s.keys,"head_stride":s.head_stride,"k_sha256":compact(s.k),"v_sha256":compact(s.v)})
                }).collect();
                let visible: Vec<_> = (before..before + count)
                    .map(|q| {
                        let r = view.visible_range(q).unwrap();
                        [r.start, r.end]
                    })
                    .collect();
                let record = serde_json::json!({"case":case_index,"before":before,"count":count,"layer":layer,"producer":producer,"start":view.start(),"end":view.end(),"k_sha256":digest(&k),"v_sha256":digest(&v),"segments":segments,"visible":visible});
                println!("GEMMA_LONG_KV_VIEW {record}");
                for key in [
                    "start", "end", "k_sha256", "v_sha256", "segments", "visible",
                ] {
                    assert_eq!(record[key], expected[key], "{key}");
                }
                views += 1;
            }
            let prior = snapshot(&caches);
            drop(frames);
            assert_eq!(snapshot(&caches), prior);
            let frames: Vec<_> = input
                .iter()
                .enumerate()
                .map(|(i, (k, v))| plan.prepare_producer(i, &caches, k, v, count).unwrap())
                .collect();
            let measured = Measurement::start();
            plan.commit(&mut caches, &frames).unwrap();
            assert_eq!(measured.finish(), 0);
            for (producer, cache) in caches.iter().enumerate() {
                let (k, v) = cache.committed();
                let expected = &step["producers"][producer];
                let record = serde_json::json!({"case":case_index,"before":before,"producer":producer,"k_sha256":digest(k),"v_sha256":digest(v),"offset":cache.offset()});
                println!("GEMMA_LONG_KV_COMMIT {record}");
                assert_eq!(record["k_sha256"], expected["committed_k_sha256"]);
                assert_eq!(record["v_sha256"], expected["committed_v_sha256"]);
                assert_eq!(cache.offset(), before + count);
            }
        }
        assert_eq!(
            addresses,
            caches
                .iter()
                .map(|c| {
                    let (k, v) = c.committed();
                    (k.as_ptr(), v.as_ptr())
                })
                .collect::<Vec<_>>()
        );
        let full = snapshot(&caches);
        let data = [1.; 4];
        assert!(plan.prepare_producer(0, &caches, &data, &data, 1).is_err());
        assert_eq!(snapshot(&caches), full);
        println!(
            "GEMMA_LONG_KV_COMPLETE {}",
            serde_json::json!({"case":case_index,"capacity":capacity,"steps":case["steps"].as_array().unwrap().len(),"pointers_stable":true,"overflow_atomic":true})
        );
    }
    assert_eq!(straddles, 2);
    println!(
        "GEMMA_LONG_KV_SUMMARY {}",
        serde_json::json!({"views":views,"straddling_appends":straddles})
    );
}
