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
