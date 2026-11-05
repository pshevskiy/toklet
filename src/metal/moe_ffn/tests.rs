use super::*;
use crate::model::gemma_reference::tests::Measurement;
use half::f16;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus};

fn plan() -> Plan {
    Plan::new([3, 2, 1, 2, 1], 4, [3, 3, 2, 4, 3]).unwrap()
}

#[test]
fn shared_scratch_counts_route_buffers_once() {
    let plan = plan();
    let required = size_of::<Scratch>() + 282;
    assert_eq!(plan.admit(64, required).unwrap(), required);
    assert!(matches!(
        plan.admit(64, required - 1),
        Err(Error::Budget { .. })
    ));
    assert!(plan.admit(63, required).is_err());
}

#[test]
#[ignore = "Explicit Apple GPU combined MoE router and expert scratch"]
fn moe_ffn_shared_buffers_run_in_one_command() {
    objc2::rc::autoreleasepool(|_| {
        let gpu = Gpu::new().unwrap();
        let router = router::Operation::new(&gpu, 65 * 1024 * 1024).unwrap();
        let experts = experts::Operation::new(&gpu, 65 * 1024 * 1024).unwrap();
        let plan = plan();
        let required = size_of::<Scratch>() + 282;
        let mut scratch = Scratch::new(&gpu, plan, required).unwrap();
        let allocated = [
            &scratch.scores,
            &scratch.gate,
            &scratch.up,
            &scratch.activated,
            &scratch.values,
        ]
        .map(|b| b.raw.length())
        .into_iter()
        .sum::<usize>()
            + scratch.ids.raw.length()
            + scratch.gates.raw.length();
        assert_eq!(allocated, 282);
        let [input_len, fused_len, down_len, ..] = plan.experts.half;
        let mut input = Buffer::<u16>::new(&gpu, input_len).unwrap();
        let mut fused = Buffer::<u16>::new(&gpu, fused_len).unwrap();
        let mut down = Buffer::<u16>::new(&gpu, down_len).unwrap();
        let mut delta = Buffer::<u16>::new(&gpu, plan.experts.half[7]).unwrap();
        let mut status = Buffer::<u32>::new(&gpu, plan.experts.word[2]).unwrap();
        let half = |x| f16::from_f32(x).to_bits();
        input.with_mut(|b| {
            b.fill(0x7e55);
            b[..2].copy_from_slice(&[half(1.), 0]);
            b[3..5].copy_from_slice(&[half(1.), 0]);
        });
        fused.with_mut(|b| {
            b[..8].copy_from_slice(&[half(2.), 0, half(1.), 0, half(2.), 0, half(1.), 0])
        });
        down.with_mut(|b| b[..4].copy_from_slice(&[half(1.), half(-1.), half(2.), half(1.)]));
        for invalid in [false, true] {
            scratch.scores.with_mut(|b| {
                b.fill(0x7e55);
                b[..2].copy_from_slice(&[0, half(1.)]);
                b[4..6].copy_from_slice(&[half(1.), if invalid { 0x7e00 } else { 0 }]);
            });
            delta.with_mut(|b| b.fill(0x7e55));
            status.with_mut(|b| b[0] = 0);
            let raw = gpu.command().unwrap();
            let command = Command {
                raw: &raw,
                trace: None,
            };
            let measurement = Measurement::start();
            let result = scratch.encode(
                &router,
                &experts,
                &command,
                2,
                LayerIo {
                    input: &input,
                    fused: &fused,
                    down: &down,
                    delta: &delta,
                    status: &status,
                },
            );
            let allocations = measurement.finish();
            result.unwrap();
            if !cfg!(debug_assertions) {
                assert_eq!(allocations, 0);
            }
            raw.commit();
            raw.waitUntilCompleted();
            assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
            status.with_slice(|b| assert_eq!(b[0], u32::from(invalid)));
            delta.with_slice(|b| {
                // SiLU(2) rounded to Half is 0x3f0c. The first expert selected
                // doubles its first output; the second negates its second.
                assert_eq!(&b[..3], &[0x430c, 0x3f0c, 0x7e55]);
                if !invalid {
                    assert_eq!(&b[3..5], &[0x3f0c, 0xbf0c]);
                }
                assert!(b[5..].iter().all(|&v| v == 0x7e55));
            });
        }
    });
}

#[test]
#[ignore = "Explicit Apple GPU projected scores through shared MoE scratch"]
fn moe_ffn_projection_feeds_router_in_same_command() {
    use crate::metal::memory::{Layout, matrix};
    objc2::rc::autoreleasepool(|_| {
        let gpu = Gpu::new().unwrap();
        let router = router::Operation::new(&gpu, 65 * 1024 * 1024).unwrap();
        let experts = experts::Operation::new(&gpu, 65 * 1024 * 1024).unwrap();
        let layout = Layout::new(2, 8, true).unwrap();
        let plan = Plan::new([2, 2, 1, 8, 1], 4, [layout.stride, 3, 2, 9, 8]).unwrap();
        let scratch = Scratch::new(&gpu, plan, 4096).unwrap();
        let mut input = Buffer::<u16>::new(&gpu, layout.elements).unwrap();
        let view = matrix(&input, layout, true).unwrap();
        let mut weights = Tensor::allocate(&gpu, Layout::new(2, 8, true).unwrap()).unwrap();
        let zero = Buffer::<u16>::new(&gpu, 2).unwrap();
        let mut fused = Buffer::<u16>::new(&gpu, plan.experts.half[1]).unwrap();
        let mut down = Buffer::<u16>::new(&gpu, plan.experts.half[2]).unwrap();
        let delta = Buffer::<u16>::new(&gpu, plan.experts.half[7]).unwrap();
        let status = Buffer::<u32>::new(&gpu, plan.experts.word[2]).unwrap();
        let half = |v| f16::from_f32(v).to_bits();
        input.with_mut(|b| {
            b[0] = half(1.);
            b[layout.stride + 1] = half(1.);
        });
        weights.buffer.with_mut(|b| {
            b[1] = half(1.);
            b[weights.layout.stride] = half(1.);
        });
        fused.with_mut(|b| {
            for e in 0..2 {
                b[e * 16..e * 16 + 2].fill(half(2.));
                b[e * 16 + 8..e * 16 + 10].fill(half(1.));
            }
        });
        down.with_mut(|b| {
            b[0] = half(1.);
            b[8] = half(2.);
        });
        let io = LayerIo {
            input: &input,
            fused: &fused,
            down: &down,
            delta: &delta,
            status: &status,
        };
        let raw = gpu.command().unwrap();
        let command = Command {
            raw: &raw,
            trace: None,
        };
        let other = Buffer::<u16>::new(&gpu, layout.elements).unwrap();
        let wrong_view = matrix(&other, layout, true).unwrap();
        let short_zero = Buffer::<u16>::new(&gpu, 1).unwrap();
        let short_status = Buffer::<u32>::new(&gpu, 7).unwrap();
        let short_io = LayerIo {
            status: &short_status,
            ..io
        };
        for (layer, zero, input_view) in [
            (&io, &zero, &*wrong_view),
            (&io, &short_zero, &*view),
            (&short_io, &zero, &*view),
        ] {
            assert!(
                scratch
                    .project_scores(
                        &gpu,
                        &command,
                        2,
                        layer,
                        RouterProjection {
                            weights: &weights,
                            zero,
                            input_view
                        },
                    )
                    .is_err()
            );
        }
        scratch
            .project_scores(
                &gpu,
                &command,
                2,
                &io,
                RouterProjection {
                    weights: &weights,
                    zero: &zero,
                    input_view: &view,
                },
            )
            .unwrap();
        scratch.encode(&router, &experts, &command, 2, io).unwrap();
        raw.commit();
        raw.waitUntilCompleted();
        assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
        status.with_slice(|b| assert_eq!(b[0], 0));
        scratch.scores.with_slice(|b| {
            assert_eq!(&b[..2], &[0, half(1.)]);
            assert_eq!(&b[4..6], &[half(1.), 0]);
        });
        scratch.ids.with_slice(|b| assert_eq!([b[0], b[3]], [1, 0]));
        delta.with_slice(|b| assert_eq!([b[0], b[8]], [0x430c, 0x3f0c]));
    });
}

#[test]
#[ignore = "Explicit Apple GPU authenticated Granite MoE session fixture"]
fn moe_session_authenticated_chunk_replay_and_budget() {
    use crate::metal::{Error, FeedForwardScratch, Session};
    use sha2::{Digest, Sha256};
    use std::fs;
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/granite-moe-session-v1");
    let config = fs::read(dir.join("config.json")).unwrap();
    let artifact = fs::read(dir.join("model.safetensors")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&config)),
        "9008bbb3bb8c932911d89e648886eebb797700e9e96d9dc2b4f23f412eb23c8b"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(&artifact)),
        "6f0c0dba727b6b2455ea0836fdde1cc403f82a23769976ba01e1545be055139e"
    );
    let oracle_bytes = fs::read(dir.join("oracle.json")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&oracle_bytes)),
        "191e07f74f2b4f961320481ccb788bf31afaa3064b5543a3cb0a20dc87fd94a2"
    );
    let oracle: serde_json::Value = serde_json::from_slice(&oracle_bytes).unwrap();
    assert_eq!(
        oracle["producer_sha256"],
        format!(
            "{:x}",
            Sha256::digest(include_bytes!(
                "../../../tools/reference/granite_moe_session_fixture.py"
            ))
        )
    );
    let expected = || source::ExpectedArtifact {
        repository: "toklet/granite-moe-session-fixture",
        revision: "0000000000000000000000000000000000000000",
        config_bytes: config.len() as u64,
        config_sha256: Sha256::digest(&config).into(),
        model_bytes: artifact.len() as u64,
        model_sha256: Sha256::digest(&artifact).into(),
    };
    let limit = 512 * 1024 * 1024;
    assert!(matches!(
        Model::load_granite_moe(
            &dir,
            expected(),
            weights::LoadOptions {
                context_limit: 64,
                memory_limit_bytes: 70 * 1024 * 1024
            }
        ),
        Err(Error::Budget { .. })
    ));
    let model = Model::load_granite_moe(
        &dir,
        expected(),
        weights::LoadOptions {
            context_limit: 64,
            memory_limit_bytes: limit,
        },
    )
    .unwrap();
    assert_eq!(
        model.native_attention_mode(),
        crate::metal::AttentionMode::Reference
    );
    assert_eq!(
        model.native_session_modes(),
        (
            crate::metal::ProjectionMode::HalfCompensated,
            crate::metal::NormalizationMode::Compensated
        )
    );
    let receipt = model.granite_moe_import_receipt().unwrap();
    assert_eq!(receipt.source_model_sha256(), expected().model_sha256);
    assert_eq!(
        receipt.converted_values() * 2 + 20 * 14,
        model.weights_bytes()
    );
    let tokens: Vec<_> = oracle["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let check = |logits: &[f32], index: usize| {
        for (column, &value) in logits.iter().enumerate() {
            let expected = oracle["logits"][index][column].as_f64().unwrap() as f32;
            assert_eq!(
                value.to_bits(),
                expected.to_bits(),
                "row {index} column {column}: {value} vs {expected}"
            );
        }
    };
    for chunk in [1, 3, 32] {
        let mut options = session::Options {
            capacity: 32,
            position_offset: 5,
            prefill_chunk_size: chunk,
            memory_limit_bytes: limit,
        };
        let mut s = Session::new(&model, options).unwrap();
        let memory = s.memory();
        let FeedForwardScratch::Moe(ffn) = &s.scratch.feed_forward else {
            panic!("expected MoE scratch")
        };
        let common = &s.scratch;
        let half = [
            &common.x,
            &common.norm,
            &common.q,
            &common.qr,
            &common.k,
            &common.kr,
            &common.v,
            &common.context,
            &common.delta,
            &common.zero,
            &ffn.scores,
            &ffn.gate,
            &ffn.up,
            &ffn.activated,
            &ffn.values,
        ];
        let words = [
            &common.floats,
            &common.scores,
            &common.probs,
            &common.tokens,
            &common.status,
            &ffn.ids,
            &ffn.gates,
        ];
        let actual = half
            .into_iter()
            .map(|b| b.raw.length())
            .chain(words.into_iter().map(|b| b.raw.length()))
            .sum::<usize>()
            + size_of::<Scratch>();
        assert_eq!(memory.scratch_bytes, actual);
        assert_eq!(
            memory.metadata_bytes,
            model.allocated_layer_metadata_bytes().unwrap()
                + s.cache.capacity() * size_of::<crate::metal::Cache>()
                + s.views.capacity() * size_of::<crate::metal::Views>()
        );
        options.memory_limit_bytes = memory.admitted_bytes - 1;
        assert!(matches!(
            Session::new(&model, options),
            Err(Error::Budget { .. })
        ));
        s.append(&tokens[..3]).unwrap();
        s.evaluate().unwrap();
        check(s.logits().unwrap(), 2);
        let committed = s.logits().unwrap().to_vec();
        s.append(&[2]).unwrap();
        assert!(matches!(s.evaluate(), Err(Error::Nonfinite)));
        assert_eq!(s.evaluated_len(), 3);
        assert!(s.logits().is_none());
        s.discard_pending();
        assert_eq!(s.logits().unwrap(), committed);
        s.append(&tokens[3..]).unwrap();
        s.evaluate().unwrap();
        check(s.logits().unwrap(), 4);
        assert!(s.append(&[0; 28]).is_err());
        assert_eq!(s.evaluated_len(), 5);
        s.reset();
        s.append(&tokens).unwrap();
        s.evaluate().unwrap();
        check(s.logits().unwrap(), 4);
        let FeedForwardScratch::Moe(ffn) = &s.scratch.feed_forward else {
            unreachable!()
        };
        ffn.ids.with_slice(|ids| {
            let active = if chunk == 32 {
                tokens.len()
            } else if chunk == 3 {
                2
            } else {
                1
            };
            let last = oracle["routes"][1].as_array().unwrap();
            assert_eq!(
                &ids[..active],
                &last[last.len() - active..]
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u32)
                    .collect::<Vec<_>>()
            );
        });
    }
}
