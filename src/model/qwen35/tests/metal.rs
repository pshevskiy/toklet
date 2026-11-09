//! Explicit independent coupled GPU projection/CPU-state controls.
use super::*;
use crate::metal::gemma_affine::{
    CONTEXT_ALLOWANCE_BYTES, Context, Weights as GpuWeights, with_projection_failure,
};

mod generation;
mod nine_b_real;
mod real;

pub(super) fn map_weights<'a, M, N>(
    weights: Weights<'a, M>,
    mut map: impl FnMut(M) -> N,
) -> Weights<'a, N> {
    Weights {
        embedding: map(weights.embedding),
        final_norm: weights.final_norm,
        layers: weights
            .layers
            .into_iter()
            .map(|w| {
                let mixing = match w.mixing {
                    MixingWeights::Delta(d) => MixingWeights::Delta(DeltaWeights {
                        qkv: map(d.qkv),
                        z: map(d.z),
                        a: map(d.a),
                        b: map(d.b),
                        out: map(d.out),
                        convolution: d.convolution,
                        a_log: d.a_log,
                        dt_bias: d.dt_bias,
                        norm: d.norm,
                    }),
                    MixingWeights::Attention(a) => MixingWeights::Attention(AttentionWeights {
                        q_gate: map(a.q_gate),
                        k: map(a.k),
                        v: map(a.v),
                        out: map(a.out),
                        q_norm: a.q_norm,
                        k_norm: a.k_norm,
                    }),
                };
                LayerWeights {
                    input_norm: w.input_norm,
                    post_norm: w.post_norm,
                    gate: map(w.gate),
                    up: map(w.up),
                    down: map(w.down),
                    mixing,
                }
            })
            .collect(),
    }
}
pub(super) fn upload(context: &Context, fixture: &Fixture) -> Vec<GpuWeights> {
    let (_, weights) = fixture.inputs();
    let mut gpu = Vec::new();
    let _ = map_weights(weights, |matrix| {
        let layout = matrix.layout();
        let (codes, scales, biases) = matrix.original_tensors();
        let weights = context
            .weights_from_fill(layout, 16384, MEMORY_LIMIT, |first, child, c, s, b| {
                let span = layout.row_span(first..first + child.rows()).unwrap();
                c.copy_from_slice(&codes[span.weight_range()]);
                s.copy_from_slice(&scales[span.metadata_range()]);
                b.copy_from_slice(&biases[span.metadata_range()]);
                Ok(())
            })
            .unwrap();
        gpu.push(weights);
    });
    gpu
}
fn bind<'a>(fixture: &'a Fixture, gpu: &'a [GpuWeights]) -> MetalModel<'a> {
    let (config, weights) = fixture.inputs();
    let mut slot = 0;
    let weights = map_weights(weights, |matrix| {
        assert_eq!(matrix.layout(), gpu[slot].layout());
        let view = MetalMatrix::diagnostic_binding(&gpu[slot], slot);
        slot += 1;
        view
    });
    assert_eq!(slot, gpu.len());
    let mut model = MetalModel::new(config, weights).unwrap();
    model.learned_bytes = gpu.iter().map(GpuWeights::owned_bytes).sum::<usize>()
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
#[ignore = "explicit tiny independent FP32 Metal projections with CPU recurrent/attention state"]
fn independent_all_stages_partitions_recovery_and_repair() {
    println!();
    let fixture = Fixture::new();
    let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
    let gpu = upload(&context, &fixture);
    let model = bind(&fixture, &gpu);
    let mut compared = 0;
    let mut maximum = 0.0f64;
    let mut case_count = 0;
    for case in manifest()["cases"].as_array().unwrap() {
        let ids: Vec<_> = case["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let offset = case["position_offset"].as_u64().unwrap() as usize;
        let mut expected: BTreeMap<_, Vec<_>> = BTreeMap::new();
        for profile in ["ideal_fp64", "staged_fp32"] {
            for row in case["profiles"][profile].as_array().unwrap() {
                expected
                    .entry(key(
                        row["token"].as_u64().unwrap() as usize,
                        row["layer"].as_u64().map(|v| v as usize),
                        row["stage"].as_str().unwrap(),
                    ))
                    .or_default()
                    .push(values(row, profile));
            }
        }
        let mut reference: Option<(Vec<f32>, String)> = None;
        for chunk in [17, 1, 3] {
            let mut session = model
                .session(
                    &context,
                    Options {
                        position_offset: offset,
                        ..options()
                    },
                )
                .unwrap();
            let mut seen = BTreeMap::new();
            let mut logits = Vec::new();
            for input in ids.chunks(chunk) {
                let out = session
                    .forward_observed(input, &AtomicBool::new(false), &mut |o| {
                        if o.stage == "before_commit" {
                            return Ok(());
                        }
                        assert_eq!(o.position, offset + o.token);
                        let k = observed_key(&o);
                        for want in &expected[&k] {
                            maximum = maximum.max(close(o.values, want));
                            compared += o.values.len();
                        }
                        assert!(seen.insert(k, ()).is_none());
                        Ok(())
                    })
                    .unwrap();
                logits.extend_from_slice(out);
                assert_eq!(session.work().projections_completed, input.len() * 32);
                assert_eq!(session.work().tokens_committed, input.len());
            }
            assert_eq!(seen.len(), expected.len());
            let state = snapshot(&session);
            if let Some((ref bits, ref end)) = reference {
                assert_float_bits(&logits, bits);
                assert_eq!(&state, end);
            } else {
                reference = Some((logits, state));
            }
            case_count += 1;
        }
    }
    let ids = tokens();
    let mut session = model.session(&context, options()).unwrap();
    let all = session.forward(&ids).unwrap().to_vec();
    let final_state = snapshot(&session);
    for _ in 0..3 {
        let measure = Measurement::start();
        session.reset();
        let allocations = measure.finish();
        assert_eq!(allocations, 0);
        let measure = Measurement::start();
        let result = session.forward(&ids);
        let allocations = measure.finish();
        assert_float_bits(result.unwrap(), &all);
        assert_eq!(allocations, 0);
        assert_eq!(snapshot(&session), final_state);
    }
    session.reset();
    session.forward(&ids[..5]).unwrap();
    let prefix = snapshot(&session);
    for cache in &mut session.kv {
        for head in 0..model.config.kv_heads {
            let start = (head * session.options.capacity + 5) * model.config.head_dim;
            let end = (head + 1) * session.options.capacity * model.config.head_dim;
            cache.k[start..end].fill(f32::NAN);
            cache.v[start..end].fill(f32::NAN);
        }
    }
    assert_eq!(snapshot(&session), prefix);
    let mut observed = 0;
    assert!(
        session
            .forward_observed(&ids[5..], &AtomicBool::new(false), &mut |_| {
                observed += 1;
                if observed == 107 {
                    return Err(Error::Observation("injected observer"));
                }
                Ok(())
            })
            .is_err()
    );
    assert_eq!(snapshot(&session), prefix);
    assert!(!session.requires_repair());
    let cancel = AtomicBool::new(false);
    assert!(matches!(
        session.forward_observed(&ids[5..], &cancel, &mut |_| {
            cancel.store(true, Ordering::Relaxed);
            Ok(())
        }),
        Err(Error::Cancelled)
    ));
    assert_eq!(snapshot(&session), prefix);
    assert!(!session.requires_repair());
    with_projection_failure(1, || assert!(session.forward(&ids[5..]).is_err()));
    assert!(session.requires_repair());
    assert_eq!(snapshot(&session), prefix);
    assert!(matches!(
        session.forward(&ids[5..]),
        Err(Error::ExecutionPoisoned)
    ));
    let memory = session.memory();
    let limit = session.options.memory_limit_bytes;
    session.options.memory_limit_bytes = memory.admitted_bytes;
    assert!(matches!(
        session.repair_execution(&context),
        Err(Error::MemoryBudget { .. })
    ));
    assert!(session.requires_repair());
    assert_eq!(snapshot(&session), prefix);
    session.options.memory_limit_bytes = limit;
    session.repair_execution(&context).unwrap();
    assert_eq!(session.memory(), memory);
    assert!(!session.requires_repair());
    assert_eq!(snapshot(&session), prefix);
    let measure = Measurement::start();
    let result = session.forward(&ids[5..]);
    let allocations = measure.finish();
    assert_float_bits(result.unwrap(), &all[5 * model.config.vocab..]);
    assert_eq!(allocations, 0);
    assert_eq!(snapshot(&session), final_state);
    let mut reset_poison = model.session(&context, options()).unwrap();
    with_projection_failure(1, || assert!(reset_poison.forward(&ids).is_err()));
    assert!(reset_poison.requires_repair());
    reset_poison.reset();
    assert!(reset_poison.requires_repair());
    assert!(reset_poison.history().is_empty() && reset_poison.logits().is_none());
    assert!(matches!(
        reset_poison.forward(&ids),
        Err(Error::ExecutionPoisoned)
    ));
    reset_poison.repair_execution(&context).unwrap();
    assert_float_bits(reset_poison.forward(&ids).unwrap(), &all);
    assert_eq!(snapshot(&reset_poison), final_state);
    let planned = model.planned_session_memory(options()).unwrap();
    assert!(
        matches!(model.session(&context,Options {memory_limit_bytes:planned.admitted_bytes-1,..options()}),Err(Error::MemoryBudget {required,..}) if required==planned.admitted_bytes)
    );
    println!(
        "QWEN35_METAL_TINY_REPORT {}",
        serde_json::json!({"passed":true,"workloads":4,"schedules":case_count,"compared_values":compared,"maximum_gate_ratio":maximum,"partitions":[17,1,3],"warmed_allocations":0,"repair_overlap_rejected":true,"poison_requires_repair":true,"reset_preserves_poison":true,"committed_prefix_preserved":true,"real_model_qualified":false})
    );
}
