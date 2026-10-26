//! Logical wide/narrow equivalence against source data frozen before this code.
use super::*;
use crate::model::gemma_reference::tests::Measurement;

fn fixture() -> (Value, &'static [u8]) {
    let manifest =
        include_bytes!("../../../../tests/fixtures/gemma-session-capacity-v1/manifest.json");
    let binary = include_bytes!("../../../../tests/fixtures/gemma-session-capacity-v1/kv.bin");
    assert_eq!(
        format!("{:x}", Sha256::digest(manifest)),
        "96f1aba1335e3efd7b72f4b36fa05b7925970ef480d5e4a373b8e90787e25dfc"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(binary)),
        "3450e059297fc605d68e7bd957b2f0abd8676ec68ff85fa198beffc1099710b4"
    );
    (serde_json::from_slice(manifest).unwrap(), binary)
}

fn inputs(data: &Value, binary: &[u8], before: usize, count: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
    data["producers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|producer| {
            let dims = producer["head_dim"].as_u64().unwrap() as usize;
            let load = |kind: &str| {
                let offset = producer[kind]["offset"].as_u64().unwrap() as usize;
                let mut result = Vec::new();
                for head in 0..2 {
                    for position in before..before + count {
                        for feature in 0..dims {
                            let index = offset + ((head * 32 + position) * dims + feature) * 4;
                            result.push(f32::from_le_bytes(
                                binary[index..index + 4].try_into().unwrap(),
                            ));
                        }
                    }
                }
                result
            };
            (load("k"), load("v"))
        })
        .collect()
}

fn proposals<'a>(
    plan: &SharedKvPlan,
    caches: &[WindowKvCache],
    input: &'a [(Vec<f32>, Vec<f32>)],
    count: usize,
) -> Vec<PreparedKv<'a>> {
    input
        .iter()
        .enumerate()
        .map(|(index, (k, v))| plan.prepare_producer(index, caches, k, v, count).unwrap())
        .collect()
}

#[test]
fn narrowed_lifetimes_preserve_wide_logical_heads_routes_and_ownership() {
    let (data, binary) = fixture();
    let geometries: Vec<_> = (0..35)
        .map(|layer| {
            let full = layer % 5 == 4;
            LayerKvGeometry {
                kind: if full { KvKind::Full } else { KvKind::Sliding },
                heads: 2,
                head_dim: if full { 3 } else { 2 },
            }
        })
        .collect();
    let wide = SharedKvPlan::new(&geometries, 20, 512, 32, 512).unwrap();
    for invalid in [0, 33, usize::MAX] {
        let measured = Measurement::start();
        let sizing = wide.planned_narrowed_cache_bytes(invalid);
        let construction = wide.narrowed(invalid);
        let allocations = measured.finish();
        assert!(sizing.is_err() && construction.is_err());
        assert_eq!(allocations, 0);
    }
    let mut records = 0;
    for (case_index, case) in data["cases"].as_array().unwrap().iter().enumerate() {
        let capacity = case["capacity"].as_u64().unwrap() as usize;
        let chunk = case["chunk_rows"].as_u64().unwrap() as usize;
        let plan = wide.narrowed(capacity).unwrap();
        assert_eq!(plan.window, capacity);
        assert_eq!(plan.global_capacity, capacity);
        assert_eq!(plan.max_positions, capacity);
        let mut wide_caches = wide.new_caches().unwrap();
        let mut caches = plan.new_caches().unwrap();
        assert_eq!(
            plan.allocated_cache_bytes(&caches).unwrap(),
            wide.planned_narrowed_cache_bytes(capacity).unwrap()
        );
        for layer in 0..35 {
            assert_eq!(
                plan.producer_for(layer).unwrap(),
                data["routes"][layer].as_u64().unwrap() as usize
            );
        }
        let first = inputs(&data, binary, 0, 1);
        let foreign = plan.new_caches().unwrap();
        let foreign_frames = proposals(&plan, &foreign, &first, 1);
        let empty = snapshot(&caches);
        assert!(matches!(
            plan.commit(&mut caches, &foreign_frames),
            Err(Error::InvalidConfig("stale or foreign KV proposal"))
        ));
        assert_eq!(snapshot(&caches), empty);
        let stale = proposals(&plan, &caches, &first, 1);
        crate::model::window_cache::reset_batch(&mut caches).unwrap();
        let reset = snapshot(&caches);
        assert!(matches!(
            plan.commit(&mut caches, &stale),
            Err(Error::InvalidConfig("stale or foreign KV proposal"))
        ));
        assert_eq!(snapshot(&caches), reset);
        drop(stale);
        for before in (0..capacity).step_by(chunk) {
            let count = chunk.min(capacity - before);
            let input = inputs(&data, binary, before, count);
            let expected = inputs(&data, binary, 0, before + count);
            let wide_frames = proposals(&wide, &wide_caches, &input, count);
            let frames = proposals(&plan, &caches, &input, count);
            for (variant, selected, owners, prepared) in [
                ("wide", &wide, &wide_caches, &wide_frames),
                ("narrow", &plan, &caches, &frames),
            ] {
                for (layer, geometry) in geometries.iter().enumerate() {
                    let view = selected.view(layer, owners, prepared).unwrap();
                    let producer = selected.producer_for(layer).unwrap();
                    let dims = geometry.head_dim;
                    let segments: Vec<_> = view.segments().iter().map(|span| {
                        let digest = |values: &[f32]| {
                            let mut hasher = Sha256::new();
                            for head in 0..2 {
                                for value in &values[head * span.head_stride..head * span.head_stride + span.keys * dims] {
                                    hasher.update(value.to_bits().to_le_bytes());
                                }
                            }
                            format!("{:x}", hasher.finalize())
                        };
                        serde_json::json!({"key_start":span.key_start,"keys":span.keys,
                            "head_stride":span.head_stride,"k_sha256":digest(span.k),"v_sha256":digest(span.v)})
                    }).collect();
                    let visible: Vec<_> = (before..before + count)
                        .map(|query| {
                            let range = view.visible_range(query).unwrap();
                            assert_eq!(range, 0..query + 1);
                            (range.start, range.end)
                        })
                        .collect();
                    let mut k = Vec::new();
                    let mut v = Vec::new();
                    for head in 0..2 {
                        for span in view.head(head).unwrap() {
                            k.extend(span.k.iter().map(|value| value.to_bits()));
                            v.extend(span.v.iter().map(|value| value.to_bits()));
                        }
                    }
                    println!(
                        "GEMMA_SESSION_CAPACITY_VIEW {}",
                        serde_json::json!({
                        "case":case_index,"capacity":capacity,"chunk_rows":chunk,"variant":variant,
                        "before":before,"rows":count,"layer":layer,"producer":producer,
                        "start":view.start(),"end":view.end(),"heads":2,"head_dim":dims,"k_bits":&k,"v_bits":&v,
                        "segments":segments,"visible_ranges":visible})
                    );
                    assert_eq!((view.start(), view.end()), (0, before + count));
                    assert!(
                        k.iter()
                            .copied()
                            .eq(expected[producer].0.iter().map(|v| v.to_bits()))
                    );
                    assert!(
                        v.iter()
                            .copied()
                            .eq(expected[producer].1.iter().map(|v| v.to_bits()))
                    );
                    records += 1;
                }
            }
            wide.commit(&mut wide_caches, &wide_frames).unwrap();
            plan.commit(&mut caches, &frames).unwrap();
        }
        let committed = snapshot(&caches);
        let overflow = plan.prepare_producer(0, &caches, &first[0].0, &first[0].1, 1);
        assert!(
            matches!(overflow, Err(Error::Capacity{offset,tokens:1,limit})if offset==capacity&&limit==capacity)
        );
        assert_eq!(snapshot(&caches), committed);
        println!(
            "GEMMA_SESSION_CAPACITY_KV_COMPLETE {}",
            serde_json::json!({
            "case":case_index,"capacity":capacity,"chunk_rows":chunk,
            "cache_payload_bytes":caches.iter().map(|cache| {let(k,v)=cache.committed();(k.len()+v.len())*4}).sum::<usize>(),
            "logical_prefix_preserved":true,"overflow_rejected":true,"foreign_rejected":true,"stale_rejected":true})
        );
    }
    assert_eq!(records, 6860);
}
