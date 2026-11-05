use super::*;
use crate::{
    cpu::{RouterShape, route_topk},
    model::gemma_reference::tests::Measurement,
};
use half::f16;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
const INPUT_GUARD: u16 = 0x7e55;
const OUT_GUARD: u32 = 0x7fa12345;
const BUDGET: usize = 1024 * 1024;
fn finish(raw: &ProtocolObject<dyn MTLCommandBuffer>) {
    raw.commit();
    raw.waitUntilCompleted();
    assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
}
fn init(r: &mut Resident, status: u32) {
    r.input.with_mut(|x| x.fill(INPUT_GUARD));
    r.ids.with_mut(|x| x.fill(OUT_GUARD));
    r.gates.with_mut(|x| x.fill(OUT_GUARD));
    r.status.with_mut(|x| {
        x.fill(OUT_GUARD);
        x[0] = status;
    });
}
fn fixture() -> Value {
    let raw = include_bytes!("../../../tests/fixtures/moe-router-half-v1/fixture.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        "071fbe82797e4b7eb1fb7348597aa618273bcc5a5f5cb61e12c88c2bd88c55ad"
    );
    let f: Value = serde_json::from_slice(raw).unwrap();
    for (key, source) in [
        (
            "producer_sha256",
            include_bytes!("../../../tools/reference/moe_router_half_fixture.py").as_slice(),
        ),
        (
            "scalar_sha256",
            include_bytes!("../../../tools/reference/moe_router_fixture.py").as_slice(),
        ),
    ] {
        assert_eq!(f[key], format!("{:x}", Sha256::digest(source)));
    }
    f
}
#[test]
fn router_plan_rejects_invalid_dimensions_limits_and_short_budget() {
    let p = Plan::new(32, 32, 8, 37, 13).unwrap();
    assert_eq!(
        (p.input_len, p.output_len, p.operand_bytes),
        (1191, 423, 5798)
    );
    let required = 5798 + size_of::<Resident>();
    assert_eq!(p.admit(2382, required).unwrap(), required);
    assert!(matches!(
        p.admit(2382, required - 1),
        Err(Error::Budget { .. })
    ));
    assert!(p.admit(2381, required).is_err());
    for (t, e, k, i, o) in [
        (0, 32, 8, 37, 13),
        (33, 32, 8, 37, 13),
        (32, 0, 8, 37, 13),
        (32, 33, 8, 37, 13),
        (32, 32, 0, 37, 13),
        (32, 32, 33, 37, 33),
        (32, 32, 8, 31, 13),
        (32, 32, 8, 37, 7),
        (32, 32, 8, usize::MAX, 13),
        (32, 32, 8, 37, u32::MAX as usize),
    ] {
        assert!(Plan::new(t, e, k, i, o).is_err());
    }
}
#[test]
#[ignore = "Explicit Apple GPU Half-score router fixtures and resident allocation"]
fn moe_router_half_frozen_cases_and_resident_reuse() {
    objc2::rc::autoreleasepool(|_| {
        let f = fixture();
        let gpu = Gpu::new().unwrap();
        assert!(matches!(
            Operation::new(&gpu, PIPELINE_RESERVATION + size_of::<Operation>() - 1),
            Err(Error::Budget { .. })
        ));
        let op = Operation::new(&gpu, PIPELINE_RESERVATION + size_of::<Operation>()).unwrap();
        let mut slots = 0;
        let mut launches = 0;
        let mut maximum_error = 0_f64;
        let mut max_allocations = 0;
        for case in f["cases"].as_array().unwrap() {
            let n = |k: &str| case[k].as_u64().unwrap() as usize;
            let t = n("tokens");
            let e = n("experts");
            let k = n("selected");
            let p = Plan::new(32, e, k, e + 5, k + 3).unwrap();
            let words = |name: &str| {
                case[name]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as u32)
                    .collect::<Vec<_>>()
            };
            let input = words("half_words");
            let ids = words("ids");
            let gates = words("gate_bits");
            let scores: Vec<_> = input
                .iter()
                .map(|&v| f16::from_bits(v as u16).to_f32())
                .collect();
            let mut cpu_ids = vec![0; t * k];
            let mut cpu_gates = vec![0.; t * k];
            route_topk(
                &scores,
                &mut cpu_ids,
                &mut cpu_gates,
                RouterShape {
                    tokens: t,
                    experts: e,
                    selected: k,
                },
            )
            .unwrap();
            assert_eq!(cpu_ids, ids);
            let required = p.admit(gpu.device.maxBufferLength(), BUDGET).unwrap();
            assert!(matches!(
                Resident::new(&gpu, p, required - 1),
                Err(Error::Budget { .. })
            ));
            let mut r = Resident::new(&gpu, p, required).unwrap();
            assert_eq!(
                r.input.raw.length()
                    + r.ids.raw.length()
                    + r.gates.raw.length()
                    + r.status.raw.length(),
                p.operand_bytes
            );
            init(&mut r, 0);
            r.input.with_mut(|b| {
                for row in 0..t {
                    for expert in 0..e {
                        b[row * p.input_stride + expert] = input[row * e + expert] as u16;
                    }
                }
            });
            let before = r.input.with_slice(|b| b.to_vec());
            for _ in 0..3 {
                let raw = gpu.command().unwrap();
                let cmd = Command {
                    raw: &raw,
                    trace: None,
                };
                let m = Measurement::start();
                let result = op.encode(&cmd, r.plan, t, r.invocation());
                let allocations = m.finish();
                result.unwrap();
                max_allocations = max_allocations.max(allocations);
                if !cfg!(debug_assertions) {
                    assert_eq!(allocations, 0);
                }
                finish(&raw);
                launches += 1;
                r.input.with_slice(|b| assert_eq!(b, before));
                r.status.with_slice(|b| {
                    assert_eq!(b[0], 0);
                    assert!(b[1..].iter().all(|&v| v == OUT_GUARD));
                });
                r.ids.with_slice(|found| {
                    r.gates.with_slice(|g| {
                        for i in 0..p.output_len {
                            let row = i / p.output_stride;
                            let slot = i % p.output_stride;
                            if row < t && slot < k {
                                let logical = row * k + slot;
                                assert_eq!(found[i], ids[logical], "{}", case["name"]);
                                let actual = f32::from_bits(g[i]);
                                assert!(actual.is_finite() && actual >= 0.);
                                for reference in
                                    [f32::from_bits(gates[logical]), cpu_gates[logical]]
                                {
                                    let error = (f64::from(actual) - f64::from(reference)).abs();
                                    maximum_error = maximum_error.max(error);
                                    assert!(
                                        error
                                            <= f["gate_atol"].as_f64().unwrap()
                                                + f["gate_rtol"].as_f64().unwrap()
                                                    * f64::from(reference).abs(),
                                        "{} slot{logical}: {actual} vs {reference}",
                                        case["name"]
                                    );
                                }
                                if case["name"] == "mass-order" {
                                    if logical == 0 {
                                        assert_eq!(g[i], 1_f32.to_bits());
                                    } else {
                                        assert!(actual > 0. && actual < 2_f32.powi(-24));
                                    }
                                }
                            } else {
                                assert_eq!(found[i], OUT_GUARD);
                                assert_eq!(g[i], OUT_GUARD);
                            }
                        }
                    })
                });
                slots += t * k;
            }
        }
        println!(
            "MOE_ROUTER_HALF {}",
            json!({"device":gpu.device.name().to_string(),"cases":f["cases"].as_array().unwrap().len(),"launches":launches,"checked_slots":slots,"max_gate_abs_error":maximum_error,"max_encode_rust_allocations":max_allocations,"release_zero_required":!cfg!(debug_assertions),"session_or_checkpoint_qualified":false})
        );
    });
}
#[test]
#[ignore = "Explicit Apple GPU invalid-row status, borrowed buffers and padding"]
fn moe_router_half_invalid_rows_and_shared_status() {
    objc2::rc::autoreleasepool(|_| {
        let gpu = Gpu::new().unwrap();
        let op = Operation::new(&gpu, PIPELINE_RESERVATION + size_of::<Operation>()).unwrap();
        let p = Plan::new(32, 32, 8, 37, 13).unwrap();
        let mut r = Resident::new(&gpu, p, BUDGET).unwrap();
        let short = Buffer::<u32>::new(&gpu, STATUS - 1).unwrap();
        for bad in [0x7e00, 0x7c00, 0xfc00] {
            init(&mut r, 0x80);
            r.input.with_mut(|b| {
                for row in 0..9 {
                    b[row * p.input_stride..row * p.input_stride + 32].fill(0);
                }
                b[8 * p.input_stride + 31] = bad;
            });
            let raw = gpu.command().unwrap();
            let cmd = Command {
                raw: &raw,
                trace: None,
            };
            assert!(op.encode(&cmd, p, 0, r.invocation()).is_err());
            assert!(op.encode(&cmd, p, 33, r.invocation()).is_err());
            assert!(
                op.encode(
                    &cmd,
                    p,
                    9,
                    Invocation {
                        status: &short,
                        ..r.invocation()
                    }
                )
                .is_err()
            );
            assert!(
                op.encode(
                    &cmd,
                    p,
                    9,
                    Invocation {
                        gates: &r.ids,
                        ..r.invocation()
                    }
                )
                .is_err()
            );
            op.encode(&cmd, p, 9, r.invocation()).unwrap();
            op.encode(&cmd, p, 1, r.invocation()).unwrap();
            finish(&raw);
            r.status.with_slice(|b| {
                assert_eq!(b[0], 0x81);
                assert!(b[1..].iter().all(|&x| x == OUT_GUARD));
            });
            r.ids.with_slice(|ids| {
                r.gates.with_slice(|g| {
                    for i in 0..p.output_len {
                        let row = i / p.output_stride;
                        let slot = i % p.output_stride;
                        if row < 8 && slot < 8 {
                            assert_eq!(ids[i], slot as u32);
                            assert_eq!(g[i], 0.125_f32.to_bits());
                        } else {
                            assert_eq!(ids[i], OUT_GUARD);
                            assert_eq!(g[i], OUT_GUARD);
                        }
                    }
                })
            });
        }
    });
}
