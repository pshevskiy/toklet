//! Small two-table GPU projection graph; no real9B GPU source admission.
use super::*;
use crate::metal::gemma_affine::{
    CONTEXT_ALLOWANCE_BYTES, Context, Weights as GpuWeights, with_projection_failure,
};

fn head_weights(context: &Context) -> GpuWeights {
    let matrix = head();
    let layout = matrix.layout();
    let (codes, scales, biases) = matrix.original_tensors();
    context
        .weights_from_fill(layout, 16384, MEMORY_LIMIT, |first, child, c, s, b| {
            let span = layout.row_span(first..first + child.rows()).unwrap();
            c.copy_from_slice(&codes[span.weight_range()]);
            s.copy_from_slice(&scales[span.metadata_range()]);
            b.copy_from_slice(&biases[span.metadata_range()]);
            Ok(())
        })
        .unwrap()
}
fn bind<'a>(fixture: &'a Fixture, gpu: &'a [GpuWeights], head: &'a GpuWeights) -> MetalModel<'a> {
    let (c, w) = fixture.inputs();
    let mut slot = 0;
    let weights = super::super::metal::map_weights(w, |matrix| {
        assert_eq!(matrix.layout(), gpu[slot].layout());
        let m = MetalMatrix::diagnostic_binding(&gpu[slot], slot);
        slot += 1;
        m
    });
    assert_eq!(slot, gpu.len());
    let mut model =
        MetalModel::new_untied(c, weights, MetalMatrix::diagnostic_binding(head, slot)).unwrap();
    model.learned_bytes = gpu.iter().map(GpuWeights::owned_bytes).sum::<usize>()
        + head.owned_bytes()
        + fixture
            .vectors
            .values()
            .map(|v| v.capacity() * 4)
            .sum::<usize>()
        + CONTEXT_ALLOWANCE_BYTES
        + METADATA_ALLOWANCE;
    model
}

#[test]
#[ignore = "explicit independent two-table Metal projections,CPU state and repair; no real9B GPU"]
fn independent_two_table_graph_inventory_state_repair_and_allocations() {
    println!();
    let fixture = Fixture::new();
    let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
    let gpu = super::super::metal::upload(&context, &fixture);
    let head = head_weights(&context);
    let model = bind(&fixture, &gpu, &head);
    let mut compared = 0;
    let mut maximum = 0.0f64;
    for case in data()["cases"].as_array().unwrap() {
        let ids: Vec<u32> = case["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let offset = case["position_offset"].as_u64().unwrap() as usize;
        let expected = expected(case);
        for chunk in [1, 3, 17] {
            let mut s = model
                .session(
                    &context,
                    Options {
                        position_offset: offset,
                        ..options()
                    },
                )
                .unwrap();
            let mut seen = BTreeMap::new();
            for part in ids.chunks(chunk) {
                s.forward_observed(part, &AtomicBool::new(false), &mut |o| {
                    if o.stage == "before_commit" {
                        return Ok(());
                    }
                    let key = observed_key(&o);
                    for row in &expected[&key] {
                        maximum = maximum.max(close(o.values, row));
                        compared += o.values.len();
                    }
                    assert!(seen.insert(key, ()).is_none());
                    Ok(())
                })
                .unwrap();
                assert_eq!(s.work().projections_completed, part.len() * 32);
            }
            assert_eq!(seen.len(), expected.len());
            assert_eq!(s.history(), ids);
        }
    }
    let ids = tokens();
    let mut clean = model.session(&context, options()).unwrap();
    clean.forward(&ids).unwrap();
    let target = snapshot(&clean);
    let expected_logits = clean.logits().unwrap().to_vec();
    let mut s = model.session(&context, options()).unwrap();
    s.forward(&ids[..5]).unwrap();
    let prefix = snapshot(&s);
    assert!(
        s.forward_observed(
            &ids[5..],
            &AtomicBool::new(false),
            &mut |o| if o.stage == "logits" {
                Err(Error::Observation("untied logits"))
            } else {
                Ok(())
            }
        )
        .is_err()
    );
    assert_eq!(snapshot(&s), prefix);
    assert!(!s.requires_repair());
    with_projection_failure(1, || assert!(s.forward(&ids[5..]).is_err()));
    assert_eq!(snapshot(&s), prefix);
    assert!(s.requires_repair());
    let memory = s.memory();
    let limit = s.options.memory_limit_bytes;
    s.options.memory_limit_bytes = memory.admitted_bytes;
    assert!(matches!(
        s.repair_execution(&context),
        Err(Error::MemoryBudget { .. })
    ));
    assert_eq!(snapshot(&s), prefix);
    s.options.memory_limit_bytes = limit;
    s.repair_execution(&context).unwrap();
    assert!(!s.requires_repair());
    assert_eq!(s.memory(), memory);
    let measure = Measurement::start();
    let result = s.forward(&ids[5..]);
    let allocations = measure.finish();
    result.unwrap();
    assert_eq!(allocations, 0);
    assert_eq!(snapshot(&s), target);
    assert_float_bits(s.logits().unwrap(), &expected_logits);
    let planned = model.planned_session_memory(options()).unwrap();
    assert!(matches!(
        model.session(
            &context,
            Options {
                memory_limit_bytes: planned.admitted_bytes - 1,
                ..options()
            }
        ),
        Err(Error::MemoryBudget { .. })
    ));
    let (c, w) = fixture.inputs();
    let mut slot = 0;
    let weights = super::super::metal::map_weights(w, |_| {
        let m = MetalMatrix::diagnostic_binding(&gpu[slot], slot);
        slot += 1;
        m
    });
    let alias =
        MetalModel::new_untied(c, weights, MetalMatrix::diagnostic_binding(&gpu[0], 0)).unwrap();
    assert!(alias.session(&context, options()).is_err());
    println!(
        "UNTIED_METAL_REPORT {}",
        serde_json::json!({"passed":true,"compared_values":compared,"maximum_gate_ratio":maximum,"workloads":4,"partitions":[1,3,17],"extra_head_slot":true,"duplicate_head_slot_rejected":true,"repair_overlap_rejected":true,"warmed_allocations":0,"real_model_qualified":false})
    );
}
