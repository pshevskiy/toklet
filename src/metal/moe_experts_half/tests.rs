use super::*;
use crate::model::gemma_reference::tests::Measurement;
use half::f16;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const HALF_GUARD: u16 = 0x7e55;
const WORD_GUARD: u32 = 0x7fa12345;
const BUDGET: usize = 128 * 1024 * 1024;

fn fixture() -> Value {
    let raw = include_bytes!("../../../tests/fixtures/moe-experts-half-v1/fixture.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        "a54750600516cc6b94d18d9e13e4912857482eac5b982e6f1bf5f0491ecc5fdd"
    );
    let fixture: Value = serde_json::from_slice(raw).unwrap();
    for (i, source) in [
        include_bytes!("../../../tools/reference/moe_experts_half_fixture.py").as_slice(),
        include_bytes!("../../../tools/reference/moe_experts_fixture.py").as_slice(),
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(
            fixture["producers"][i],
            format!("{:x}", Sha256::digest(source))
        );
    }
    fixture
}

fn words(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

fn fill_rows<T: Copy>(buffer: &mut [T], source: &[T], width: usize, stride: usize) {
    for (row, values) in source.chunks_exact(width).enumerate() {
        buffer[row * stride..row * stride + width].copy_from_slice(values);
    }
}

fn finish(command: &ProtocolObject<dyn MTLCommandBuffer>) {
    command.commit();
    command.waitUntilCompleted();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
}

fn reset(resident: &mut Resident) {
    for buffer in &mut resident.half {
        buffer.with_mut(|b| b.fill(HALF_GUARD));
    }
    for buffer in &mut resident.word {
        buffer.with_mut(|b| b.fill(WORD_GUARD));
    }
    resident.word[2].with_mut(|b| b[0] = 0);
}

fn check_rows(
    buffer: &Buffer<u16>,
    expected: &Value,
    width: usize,
    stride: usize,
    stage: &str,
) -> usize {
    let words = words(expected);
    let rows = words.len() / width;
    buffer.with_slice(|buffer| {
        for (i, &value) in buffer.iter().enumerate() {
            let row = i / stride;
            let column = i % stride;
            let expected = if row < rows && column < width {
                words[row * width + column] as u16
            } else {
                HALF_GUARD
            };
            assert_eq!(value, expected, "{stage} row {row} col {column}");
        }
    });
    words.len()
}

#[test]
fn expert_plan_rejects_bad_geometry_limits_and_budget() {
    let geometry = [2, 4, 2, 3, 5];
    let strides = [5, 3, 7, 4, 6];
    let plan = Plan::new(geometry, strides).unwrap();
    assert_eq!(plan.half, [17, 127, 67, 35, 35, 35, 23, 19]);
    assert_eq!(plan.word, [13, 13, 8]);
    assert_eq!(plan.operand_bytes, 852);
    let required = 852 + size_of::<Resident>();
    assert_eq!(plan.admit(254, required).unwrap(), required);
    assert!(plan.admit(253, required).is_err());
    assert!(matches!(
        plan.admit(254, required - 1),
        Err(Error::Budget { .. })
    ));
    for invalid in [
        [0, 4, 2, 3, 5],
        [33, 4, 2, 3, 5],
        [2, 33, 2, 3, 5],
        [2, 4, 5, 3, 5],
        [2, 4, 2, 0, 5],
        [2, 4, 2, 3, 0],
        [2, 4, 2, usize::MAX, 5],
    ] {
        assert!(Plan::new(invalid, strides).is_err());
    }
    for (index, width) in [3, 2, 5, 3, 3].into_iter().enumerate() {
        let mut invalid = strides;
        invalid[index] = width - 1;
        assert!(Plan::new(geometry, invalid).is_err());
        invalid[index] = u32::MAX as usize;
        assert!(Plan::new(geometry, invalid).is_err());
    }
}

#[test]
#[ignore = "Explicit Apple GPU Half expert stages and resident reuse"]
fn moe_experts_half_frozen_stages() {
    objc2::rc::autoreleasepool(|_| {
        let fixture = fixture();
        let gpu = Gpu::new().unwrap();
        let operation =
            Operation::new(&gpu, PIPELINE_RESERVATION + size_of::<Operation>()).unwrap();
        let mut checked = 0;
        let mut max_allocations = 0;
        let mut launches = 0;
        for case in fixture["cases"].as_array().unwrap() {
            let n = |key: &str| case[key].as_u64().unwrap() as usize;
            let (tokens, experts, selected, hidden, intermediate) = (
                n("tokens"),
                n("experts"),
                n("selected"),
                n("hidden"),
                n("intermediate"),
            );
            let strides = [
                hidden + 3,
                selected + 2,
                intermediate + 5,
                hidden + 1,
                hidden + 2,
            ];
            let plan = Plan::new(
                [tokens + 1, experts, selected, hidden, intermediate],
                strides,
            )
            .unwrap();
            let required = plan.admit(gpu.device.maxBufferLength(), BUDGET).unwrap();
            assert!(matches!(
                Resident::new(&gpu, plan, required - 1),
                Err(Error::Budget { .. })
            ));
            let mut resident = Resident::new(&gpu, plan, required).unwrap();
            let bytes = resident
                .half
                .iter()
                .map(|b| b.raw.length())
                .chain(resident.word.iter().map(|b| b.raw.length()))
                .sum::<usize>();
            assert_eq!(bytes, plan.operand_bytes);
            reset(&mut resident);
            let input: Vec<_> = words(&case["input"])
                .into_iter()
                .map(|v| v as u16)
                .collect();
            resident.half[0].with_mut(|b| fill_rows(b, &input, hidden, strides[0]));
            for (binding, name, matrix) in [
                (1, "fused", [2 * intermediate, hidden]),
                (2, "down", [hidden, intermediate]),
            ] {
                resident.half[binding].with_mut(|buffer| {
                    let logical = &mut buffer[..experts * matrix[0] * matrix[1]];
                    if case[name].is_array() {
                        for (destination, value) in logical.iter_mut().zip(words(&case[name])) {
                            *destination = value as u16;
                        }
                    } else {
                        logical.fill(0);
                        for entry in case[format!("{name}_coordinates")].as_array().unwrap() {
                            let position = words(entry);
                            let index = (position[0] as usize * matrix[0] + position[1] as usize)
                                * matrix[1]
                                + position[2] as usize;
                            logical[index] = position[3] as u16;
                        }
                    }
                });
            }
            resident.word[0].with_mut(|b| fill_rows(b, &words(&case["ids"]), selected, strides[1]));
            resident.word[1]
                .with_mut(|b| fill_rows(b, &words(&case["gate_bits"]), selected, strides[1]));
            // Hash immutable operands outside the encode measurement.
            let digest = |b: &Buffer<u16>| {
                b.with_slice(|v| {
                    let mut hash = Sha256::new();
                    for value in v {
                        hash.update(value.to_le_bytes());
                    }
                    hash.finalize()
                })
            };
            let before = resident.half[..3].iter().map(digest).collect::<Vec<_>>();
            let route = resident.word[..2]
                .iter()
                .map(|b| b.with_slice(|s| s.to_vec()))
                .collect::<Vec<_>>();
            for _ in 0..2 {
                let raw = gpu.command().unwrap();
                let command = Command {
                    raw: &raw,
                    trace: None,
                };
                let measurement = Measurement::start();
                let result =
                    operation.encode(&command, resident.plan, tokens, resident.invocation());
                let allocations = measurement.finish();
                result.unwrap();
                max_allocations = max_allocations.max(allocations);
                if !cfg!(debug_assertions) {
                    assert_eq!(allocations, 0);
                }
                finish(&raw);
                launches += 1;
                resident.word[2].with_slice(|b| {
                    assert_eq!(b[0], 0, "{}", case["name"]);
                    assert!(b[1..].iter().all(|&x| x == WORD_GUARD));
                });
                for (binding, name, width, stride) in [
                    (3, "gate", intermediate, strides[2]),
                    (4, "up", intermediate, strides[2]),
                    (5, "activated", intermediate, strides[2]),
                    (6, "values", hidden, strides[3]),
                    (7, "output", hidden, strides[4]),
                ] {
                    checked +=
                        check_rows(&resident.half[binding], &case[name], width, stride, name);
                }
            }
            assert_eq!(
                resident.half[..3].iter().map(digest).collect::<Vec<_>>(),
                before
            );
            for (buffer, expected) in resident.word[..2].iter().zip(route) {
                buffer.with_slice(|b| assert_eq!(b, expected));
            }
        }
        println!(
            "MOE_EXPERT_HALF {}",
            json!({"device":gpu.device.name().to_string(),"cases":fixture["cases"].as_array().unwrap().len(),"launches":launches,"exact_stage_values":checked,"max_encode_rust_allocations":max_allocations})
        );
    });
}

#[test]
#[ignore = "Explicit Apple GPU weighted reduction and candidate failure"]
fn moe_experts_half_order_and_status() {
    objc2::rc::autoreleasepool(|_| {
        let fixture = fixture();
        let gpu = Gpu::new().unwrap();
        let operation =
            Operation::new(&gpu, PIPELINE_RESERVATION + size_of::<Operation>()).unwrap();
        for control in fixture["reduction"].as_array().unwrap() {
            let weights = words(&control["gates"]);
            let selected = weights.len();
            let plan = Plan::new([1, selected, selected, 1, 1], [1, selected, 1, 1, 1]).unwrap();
            let mut resident = Resident::new(&gpu, plan, BUDGET).unwrap();
            reset(&mut resident);
            resident.half[6].with_mut(|b| {
                for (slot, value) in words(&control["values"]).into_iter().enumerate() {
                    b[slot] = value as u16;
                }
            });
            resident.word[0].with_mut(|b| {
                for (slot, id) in b[..selected].iter_mut().enumerate() {
                    *id = slot as u32;
                }
            });
            resident.word[1].with_mut(|b| b[..selected].copy_from_slice(&weights));
            let raw = gpu.command().unwrap();
            operation
                .encode_stages(
                    &Command {
                        raw: &raw,
                        trace: None,
                    },
                    plan,
                    1,
                    resident.invocation(),
                    3,
                )
                .unwrap();
            finish(&raw);
            resident.half[7].with_slice(|b| {
                assert_eq!(
                    b[0],
                    control["expected"].as_u64().unwrap() as u16,
                    "{}",
                    control["name"]
                )
            });
            resident.word[2].with_slice(|b| assert_eq!(b[0], 0));
        }
        let plan = Plan::new([2, 2, 2, 1, 1], [2, 3, 2, 2, 2]).unwrap();
        let mut resident = Resident::new(&gpu, plan, BUDGET).unwrap();
        for (ids, gate, input, expected) in [
            ([0, u32::MAX], 1., 1., 1),
            ([0, 0], 1., 1., 1),
            ([0, 1], f32::NAN, 1., 1),
            ([0, 1], -1., 1., 1),
            ([0, 1], 1., 65504., 2),
        ] {
            reset(&mut resident);
            resident.half[0].with_mut(|b| {
                b[0] = f16::ONE.to_bits();
                b[2] = f16::from_f32(input).to_bits();
            });
            resident.half[1].with_mut(|b| b[..4].fill(f16::from_f32(2.).to_bits()));
            resident.half[2].with_mut(|b| b[..2].fill(f16::ONE.to_bits()));
            resident.word[0].with_mut(|b| {
                b[..2].copy_from_slice(&[0, 1]);
                b[3..5].copy_from_slice(&ids);
            });
            resident.word[1].with_mut(|b| {
                b[..2].fill(0.5_f32.to_bits());
                b[3] = 1_f32.to_bits();
                b[4] = gate.to_bits();
            });
            resident.word[2].with_mut(|b| b[0] = 0x80);
            let raw = gpu.command().unwrap();
            let command = Command {
                raw: &raw,
                trace: None,
            };
            let mut aliased = resident.invocation();
            aliased.half[4] = aliased.half[3];
            assert!(operation.encode(&command, plan, 2, aliased).is_err());
            assert!(
                operation
                    .encode(&command, plan, 3, resident.invocation())
                    .is_err()
            );
            operation
                .encode(&command, plan, 2, resident.invocation())
                .unwrap();
            operation
                .encode(&command, plan, 1, resident.invocation())
                .unwrap();
            finish(&raw);
            resident.word[2].with_slice(|b| {
                assert_eq!(b[0], 0x80 | expected);
                assert!(b[1..].iter().all(|&x| x == WORD_GUARD));
            });
            resident.half[7].with_slice(|b| {
                assert!(f16::from_bits(b[0]).is_finite());
                assert_eq!(b[1], HALF_GUARD);
                assert!(b[3..].iter().all(|&x| x == HALF_GUARD));
            });
        }
    });
}
