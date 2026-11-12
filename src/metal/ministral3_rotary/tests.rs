use super::*;
use crate::model::{gemma_reference::tests::Measurement, ministral3_rotary as cpu_rotary};
use half::f16;
use objc2::rc::autoreleasepool;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::time::Instant;

const PROFILE_SHA: &str = "0f8d25bec77f2f6901e86830b7da22b75808fccb7800ce939f17f386206f4c9f";
const FIXTURE_SHA: &str = "0e1eb4f6a9056bf1725d8265e905ec7fe8ddc3c690e8a802f58121d96f36c427";
const HEAD_DIM: usize = 128;

fn binding(value: &Value, raw: &[u8]) {
    assert_eq!(value["bytes"].as_u64().unwrap(), raw.len() as u64);
    assert_eq!(value["sha256"], format!("{:x}", Sha256::digest(raw)));
}

fn fixture() -> (Value, Value) {
    let profile_raw = include_bytes!("../../../tests/fixtures/ministral3-rotary-v1/profile.json");
    let fixture_raw = include_bytes!("../../../tests/fixtures/ministral3-rotary-v1/fixture.json");
    assert_eq!(format!("{:x}", Sha256::digest(profile_raw)), PROFILE_SHA);
    assert_eq!(format!("{:x}", Sha256::digest(fixture_raw)), FIXTURE_SHA);
    assert!(fixture_raw.len() <= 2_097_152);
    let profile: Value = serde_json::from_slice(profile_raw).unwrap();
    let fixture: Value = serde_json::from_slice(fixture_raw).unwrap();
    assert_eq!(fixture["format"], "toklet-ministral3-yarn-fixture-v1");
    assert_eq!(
        fixture["profile"]["file"],
        "tests/fixtures/ministral3-rotary-v1/profile.json"
    );
    binding(&fixture["profile"], profile_raw);
    let producers = fixture["producers"].as_object().unwrap();
    assert_eq!(producers.len(), 2);
    for (path, raw) in [
        (
            "tools/reference/ministral3_rotary_fixture.py",
            include_bytes!("../../../tools/reference/ministral3_rotary_fixture.py").as_slice(),
        ),
        (
            "tools/reference/test_ministral3_rotary_fixture.py",
            include_bytes!("../../../tools/reference/test_ministral3_rotary_fixture.py").as_slice(),
        ),
    ] {
        binding(&producers[path], raw);
    }
    assert_eq!(fixture["gates"], profile["gates"]);
    assert_eq!(
        fixture["external_source_pins"],
        profile["external_source_pins"]
    );
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 11);
    assert_eq!(profile["cases"].as_array().unwrap().len(), 11);
    (fixture, profile)
}

fn n(value: &Value, key: &str) -> usize {
    usize::try_from(value[key].as_u64().unwrap()).unwrap()
}

fn words16(value: &Value) -> Vec<u16> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u16::try_from(v.as_u64().unwrap()).unwrap())
        .collect()
}

fn words32(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
        .collect()
}

fn halves(words: &[u16]) -> Vec<f64> {
    words
        .iter()
        .map(|&v| f64::from(f16::from_bits(v).to_f32()))
        .collect()
}

fn floats(words: &[u32]) -> Vec<f64> {
    words
        .iter()
        .map(|&v| f64::from(f32::from_bits(v)))
        .collect()
}

fn hash16(words: &[u16]) -> String {
    let mut hash = Sha256::new();
    for word in words {
        hash.update(word.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn table_hash(tables: &Tables) -> String {
    let mut hash = Sha256::new();
    hash.update(tables.base().to_le_bytes());
    hash.update(tables.capacity().to_le_bytes());
    tables.coefficients.with_slice(|v| {
        for word in v {
            hash.update(word.to_le_bytes());
        }
    });
    tables.gains.with_slice(|v| {
        for word in v {
            hash.update(word.to_le_bytes());
        }
    });
    format!("{:x}", hash.finalize())
}

// Identical frozen formula, evaluated separately for each exact case/array.
// Native results never participate in deriving its envelope or gate.
fn compare(name: &str, actual: &[f64], main: &[f64], wide: &[f64]) -> usize {
    assert!(!main.is_empty());
    assert_eq!(actual.len(), main.len());
    assert_eq!(main.len(), wide.len());
    assert!(main.iter().chain(wide).all(|v| v.is_finite()));
    let spread = main
        .iter()
        .zip(wide)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let atol = (2.0 * spread).max(1e-4);
    for (profile, reference) in [("main", main), ("wide", wide)] {
        for (index, (&got, &want)) in actual.iter().zip(reference).enumerate() {
            let limit = atol + 1e-3 * want.abs();
            assert!(
                got.is_finite() && (got - want).abs() <= limit,
                "{name}/{profile}[{index}]: got={got} want={want} limit={limit}"
            );
        }
    }
    2 * actual.len()
}

fn input_recipe(input: &[u16], pattern: &str, seed: usize, profile: &Value) {
    let edges = words16(&profile["input_recipe"]["edge_words"]);
    assert_eq!(edges.len(), 16);
    for (i, &word) in input.iter().enumerate() {
        let want = match pattern {
            "normal" => {
                f16::from_f32((((i * 37 + seed) % 257) as i32 - 128) as f32 / 128.).to_bits()
            }
            "edges" => edges[(i + seed + 7 * (i / 64)) % 16],
            _ => panic!("unrecognized frozen input recipe"),
        };
        assert_eq!(word, want, "input[{i}]");
        assert!(f16::from_bits(word).is_finite());
    }
}

fn cpu_table(base: usize, capacity: usize) -> cpu_rotary::Rotary {
    cpu_rotary::Rotary::new(
        base,
        capacity,
        cpu_rotary::Rotary::planned_bytes(capacity).unwrap(),
    )
    .unwrap()
}

#[test]
fn frozen_fixture_bindings_shapes_inputs_and_independent_gates() {
    let (fixture, profile) = fixture();
    let mut inputs = 0;
    let mut coordinates = 0;
    for (case, selected) in fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .zip(profile["cases"].as_array().unwrap())
    {
        for (key, value) in selected.as_object().unwrap() {
            assert_eq!(&case[key], value);
        }
        let tokens = n(case, "tokens");
        let table = cpu_table(n(case, "offset"), tokens);
        let periods: Vec<_> = table.periods().iter().map(|&v| f64::from(v)).collect();
        assert_eq!(
            table
                .periods()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            words32(&fixture["periods"]["main"])
        );
        coordinates += compare(
            "periods",
            &periods,
            &floats(&words32(&fixture["periods"]["main"])),
            &floats(&words32(&fixture["periods"]["wide"])),
        );
        for (kind, input_key, output_key, heads, seed) in [
            (
                Kind::Keys,
                "key_input",
                "keys",
                n(case, "kv_heads"),
                n(case, "k_seed"),
            ),
            (
                Kind::Queries,
                "query_input",
                "queries",
                n(case, "query_heads"),
                n(case, "q_seed"),
            ),
        ] {
            let p = Plan::new(
                kind,
                tokens,
                heads,
                n(case, "offset"),
                n(case, "offset"),
                tokens,
            )
            .unwrap();
            let input = words16(&case[input_key]);
            assert_eq!(input.len(), tokens * heads * HEAD_DIM);
            assert_eq!(p.input_len(), input.len() + GUARD);
            input_recipe(&input, case["pattern"].as_str().unwrap(), seed, &profile);
            inputs += input.len();
            let main = halves(&words16(&case["references"]["main"][output_key]));
            let wide = halves(&words16(&case["references"]["wide"][output_key]));
            assert_eq!(main.len(), input.len());
            // Both witnesses must themselves pass the prospectively frozen gate.
            compare(output_key, &main, &main, &wide);
        }
        for key in ["coefficients", "query_scale"] {
            let main = &case["references"]["main"][key];
            let wide = &case["references"]["wide"][key];
            let expected = tokens * if key == "coefficients" { 128 } else { 1 };
            assert_eq!(main.as_array().unwrap().len(), expected);
            assert_eq!(wide.as_array().unwrap().len(), expected);
        }
    }
    assert_eq!(inputs, 62_464);
    assert_eq!(coordinates, 1_408);
}

#[test]
fn checked_plan_span_sizes_and_device_table_budget() {
    for kind in [Kind::Keys, Kind::Queries] {
        let p = Plan::new(kind, 8, 32, 16_383, 16_380, 12).unwrap();
        let len = 8 * 32 * 128 + GUARD;
        assert_eq!(p.input_len(), len);
        assert_eq!(p.output_len(), len);
        assert_eq!(p.status_len(), 1 + GUARD);
        assert_eq!(p.operand_bytes().unwrap(), len * 4 + (1 + GUARD) * 4);
        for (tokens, heads, offset, base, capacity) in [
            (0, 1, 0, 0, 1),
            (513, 1, 0, 0, 512),
            (1, 0, 0, 0, 1),
            (1, 33, 0, 0, 1),
            (1, usize::MAX, 0, 0, 1),
            (1, 1, 0, 0, 0),
            (1, 1, 0, 0, 513),
            (1, 1, 1, 2, 1),
            (2, 1, 3, 2, 2),
            (1, 1, usize::MAX, 0, 1),
            (1, 1, 262_144, 262_144, 1),
            (1, 1, 0, usize::MAX, 1),
            (1, 1, 0, 262_143, 2),
            (usize::MAX, 1, 0, 0, 512),
        ] {
            assert!(Plan::new(kind, tokens, heads, offset, base, capacity).is_err());
        }
        assert!(Plan::new(kind, 1, 32, 262_143, 262_143, 1).is_ok());
        assert!(Plan::new(kind, 512, 32, 0, 0, 512).is_ok());
    }
    for capacity in [1, 8, 512] {
        assert_eq!(
            Tables::planned_bytes(capacity).unwrap(),
            size_of::<Tables>() + (capacity * 128 + GUARD) * 4 + (capacity + GUARD) * 2
        );
    }
    for capacity in [0, 513, usize::MAX] {
        assert!(Tables::planned_bytes(capacity).is_err());
    }
}

struct Io {
    plan: Plan,
    input: Buffer<u16>,
    output: Buffer<u16>,
    status: Buffer<u32>,
    input_hash: String,
}

impl Io {
    fn new(gpu: &Gpu, plan: Plan, input: &[u16]) -> Self {
        assert_eq!(input.len() + GUARD, plan.input_len());
        let mut source = Buffer::new(gpu, plan.input_len()).unwrap();
        source.with_mut(|v| {
            v.fill(HALF_SENTINEL);
            v[..input.len()].copy_from_slice(input);
        });
        let mut output = Buffer::new(gpu, plan.output_len()).unwrap();
        output.with_mut(|v| v.fill(HALF_SENTINEL));
        let mut status = Buffer::new(gpu, plan.status_len()).unwrap();
        status.with_mut(|v| {
            v.fill(SENTINEL);
            v[0] = 0;
        });
        let input_hash = source.with_slice(hash16);
        assert_eq!(
            source.raw.length() + output.raw.length() + status.raw.length(),
            plan.operand_bytes().unwrap()
        );
        Self {
            plan,
            input: source,
            output,
            status,
            input_hash,
        }
    }
    fn invocation<'a>(&'a self, tables: &'a Tables) -> Invocation<'a> {
        Invocation {
            input: &self.input,
            output: &self.output,
            status: &self.status,
            tables,
        }
    }
    fn validate(&self) {
        assert_eq!(self.input.with_slice(hash16), self.input_hash);
        self.output.with_slice(|v| {
            assert!(
                v[self.plan.output_len() - GUARD..]
                    .iter()
                    .all(|&v| v == HALF_SENTINEL)
            )
        });
        self.status
            .with_slice(|v| assert!(v[1..].iter().all(|&v| v == SENTINEL)));
    }
    fn reset(&mut self, status: u32) {
        self.output.with_mut(|v| v.fill(HALF_SENTINEL));
        self.status.with_mut(|v| {
            v.fill(SENTINEL);
            v[0] = status;
        });
    }
    fn output(&self) -> Vec<u16> {
        self.output
            .with_slice(|v| v[..self.plan.output_len() - GUARD].to_vec())
    }
}

fn complete(raw: &ProtocolObject<dyn MTLCommandBuffer>) {
    raw.commit();
    raw.waitUntilCompleted();
    assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
    assert!(raw.error().is_none());
}

fn execute(gpu: &Gpu, op: &Operation, tables: &Tables, io: &Io) {
    let raw = gpu.command().unwrap();
    op.encode(
        &Command {
            raw: &raw,
            trace: None,
        },
        io.plan,
        io.invocation(tables),
    )
    .unwrap();
    complete(&raw);
    io.validate();
}

fn uploaded_tables(cpu: &cpu_rotary::Rotary, tables: &Tables) {
    assert_eq!(tables.base(), cpu.base());
    assert_eq!(tables.capacity(), cpu.capacity());
    let expected: Vec<_> = (cpu.base()..cpu.base() + cpu.capacity())
        .flat_map(|p| cpu.coefficient_row(p).unwrap())
        .flat_map(|v| v.iter().map(|v| v.to_bits()))
        .collect();
    tables.coefficients.with_slice(|v| {
        assert_eq!(&v[..expected.len()], expected);
        assert!(v[expected.len()..].iter().all(|&v| v == SENTINEL));
    });
    let gains: Vec<_> = (cpu.base()..cpu.base() + cpu.capacity())
        .map(|p| cpu.query_scale(p).unwrap().to_bits())
        .collect();
    tables.gains.with_slice(|v| {
        assert_eq!(&v[..gains.len()], gains);
        assert!(v[gains.len()..].iter().all(|&v| v == HALF_SENTINEL));
    });
}

#[test]
#[ignore = "Explicit completed Metal Ministral fixtures/replay/guard/preflight qualification"]
fn gpu_frozen_fixtures_replay_chunks_guards_and_preflight() {
    autoreleasepool(|_| {
        let (fixture, _) = fixture();
        let gpu = Gpu::new().unwrap();
        let required = PIPELINE_ALLOWANCE + size_of::<Operation>();
        assert!(matches!(
            Operation::new(&gpu, required - 1),
            Err(Error::Budget { .. })
        ));
        let op = Operation::new(&gpu, required).unwrap();
        let (mut arrays, mut coordinates, mut exact_replays, mut chunk_calls, mut rebase_calls) =
            (0, 0, 0, 0, 0);
        for case in fixture["cases"].as_array().unwrap() {
            let (base, tokens) = (n(case, "offset"), n(case, "tokens"));
            let cpu = cpu_table(base, tokens);
            let required = Tables::planned_bytes(tokens).unwrap();
            assert!(matches!(
                Tables::new(&gpu, &cpu, required - 1),
                Err(Error::Budget { .. })
            ));
            let tables = Tables::new(&gpu, &cpu, required).unwrap();
            uploaded_tables(&cpu, &tables);
            assert_eq!(tables.owned_bytes(), required);
            let before = table_hash(&tables);
            for (key, float) in [("coefficients", true), ("query_scale", false)] {
                let main = &case["references"]["main"][key];
                let wide = &case["references"]["wide"][key];
                let got = if float {
                    tables
                        .coefficients
                        .with_slice(|v| floats(&v[..tokens * 128]))
                } else {
                    tables.gains.with_slice(|v| halves(&v[..tokens]))
                };
                let (main, wide) = if float {
                    (floats(&words32(main)), floats(&words32(wide)))
                } else {
                    (halves(&words16(main)), halves(&words16(wide)))
                };
                coordinates += compare(key, &got, &main, &wide);
                arrays += 1;
            }
            assert_eq!(
                cpu.periods()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                words32(&fixture["periods"]["main"])
            );
            let periods: Vec<_> = cpu.periods().iter().map(|&v| f64::from(v)).collect();
            coordinates += compare(
                "periods",
                &periods,
                &floats(&words32(&fixture["periods"]["main"])),
                &floats(&words32(&fixture["periods"]["wide"])),
            );
            arrays += 1;
            assert_eq!(
                tables.gains.with_slice(|v| v[..tokens].to_vec()),
                words16(&case["references"]["main"]["query_scale"])
            );
            for (kind, input_key, output_key, heads) in [
                (Kind::Keys, "key_input", "keys", n(case, "kv_heads")),
                (
                    Kind::Queries,
                    "query_input",
                    "queries",
                    n(case, "query_heads"),
                ),
            ] {
                let input = words16(&case[input_key]);
                let plan = Plan::new(kind, tokens, heads, base, base, tokens).unwrap();
                let mut io = Io::new(&gpu, plan, &input);
                execute(&gpu, &op, &tables, &io);
                assert_eq!(io.status.with_slice(|v| v[0]), 0);
                let out = io.output();
                coordinates += compare(
                    output_key,
                    &halves(&out),
                    &halves(&words16(&case["references"]["main"][output_key])),
                    &halves(&words16(&case["references"]["wide"][output_key])),
                );
                arrays += 1;
                println!(
                    "MINISTRAL3_METAL_RAW {}",
                    json!({"case":case["name"],"kind":output_key,"tokens":tokens,"heads":heads,"absolute_offset":base,"input_sha256":hash16(&input),"guarded_input_before_sha256":io.input_hash,"guarded_input_after_sha256":io.input.with_slice(hash16),"output_words":out,"status_words":io.status.with_slice(|v|v.to_vec()),"tables_before_sha256":before,"tables_after_sha256":table_hash(&tables),"operand_bytes":plan.operand_bytes().unwrap(),"completed":true})
                );
                io.reset(0);
                execute(&gpu, &op, &tables, &io);
                assert_eq!(io.output(), out);
                assert_eq!(io.status.with_slice(|v| v[0]), 0);
                exact_replays += 1;
                let mut chunks = Vec::new();
                let mut rebased = Vec::new();
                let stride = heads * HEAD_DIM;
                for token in 0..tokens {
                    let span = &input[token * stride..(token + 1) * stride];
                    let p = Plan::new(kind, 1, heads, base + token, base, tokens).unwrap();
                    let chunk = Io::new(&gpu, p, span);
                    execute(&gpu, &op, &tables, &chunk);
                    assert_eq!(chunk.status.with_slice(|v| v[0]), 0);
                    chunks.extend(chunk.output());
                    chunk_calls += 1;
                    let cpu = cpu_table(base + token, 1);
                    let one = Tables::new(&gpu, &cpu, Tables::planned_bytes(1).unwrap()).unwrap();
                    uploaded_tables(&cpu, &one);
                    let one_hash = table_hash(&one);
                    let p = Plan::new(kind, 1, heads, base + token, base + token, 1).unwrap();
                    let chunk = Io::new(&gpu, p, span);
                    execute(&gpu, &op, &one, &chunk);
                    assert_eq!(chunk.status.with_slice(|v| v[0]), 0);
                    rebased.extend(chunk.output());
                    assert_eq!(table_hash(&one), one_hash);
                    rebase_calls += 1;
                }
                assert_eq!(chunks, out);
                assert_eq!(rebased, out);
                assert_eq!(table_hash(&tables), before);
                uploaded_tables(&cpu, &tables);
            }
        }
        let preflight = preflight(&gpu, &op);
        assert_eq!(coordinates, 134_592);
        assert_eq!(arrays, 55);
        assert_eq!(exact_replays, 22);
        assert_eq!(chunk_calls, 64);
        assert_eq!(rebase_calls, 64);
        println!(
            "MINISTRAL3_METAL_SUMMARY {}",
            json!({"device":gpu.device.name().to_string(),"fixture_cases":11,"case_arrays":arrays,"profile_coordinate_comparisons":coordinates,"GPU_output_profile_coordinates":124928,"uploaded_coefficient_gain_profile_coordinates":8256,"CPU_period_profile_coordinates":1408,"exact_replays":exact_replays,"single_row_chunk_calls":chunk_calls,"single_row_rebased_calls":rebase_calls,"preflight":preflight,"all_completed":true,"model_graph_qualified":false})
        );
    });
}

fn preflight(gpu: &Gpu, op: &Operation) -> Value {
    let cpu = cpu_table(0, 1);
    let mut tables = Tables::new(gpu, &cpu, Tables::planned_bytes(1).unwrap()).unwrap();
    let before = table_hash(&tables);
    let plan = Plan::new(Kind::Keys, 1, 1, 0, 0, 1).unwrap();
    let mut io = Io::new(gpu, plan, &[0; HEAD_DIM]);
    let raw = gpu.command().unwrap();
    let command = Command {
        raw: &raw,
        trace: None,
    };
    assert!(
        op.encode(
            &command,
            plan,
            Invocation {
                output: &io.input,
                ..io.invocation(&tables)
            }
        )
        .is_err()
    );
    let short_input = Buffer::new(gpu, plan.input_len() - 1).unwrap();
    let short_output = Buffer::new(gpu, plan.output_len() - 1).unwrap();
    let short_status = Buffer::new(gpu, plan.status_len() - 1).unwrap();
    for invocation in [
        Invocation {
            input: &short_input,
            ..io.invocation(&tables)
        },
        Invocation {
            output: &short_output,
            ..io.invocation(&tables)
        },
        Invocation {
            status: &short_status,
            ..io.invocation(&tables)
        },
    ] {
        assert!(op.encode(&command, plan, invocation).is_err());
    }
    let wrong = Plan::new(Kind::Keys, 1, 1, 1, 1, 1).unwrap();
    assert!(op.encode(&command, wrong, io.invocation(&tables)).is_err());
    let unretained = gpu.queue.commandBufferWithUnretainedReferences().unwrap();
    assert!(!unretained.retainedReferences());
    assert!(
        op.encode(
            &Command {
                raw: &unretained,
                trace: None
            },
            plan,
            io.invocation(&tables)
        )
        .is_err()
    );
    let completed = gpu.command().unwrap();
    complete(&completed);
    assert!(
        op.encode(
            &Command {
                raw: &completed,
                trace: None
            },
            plan,
            io.invocation(&tables)
        )
        .is_err()
    );
    let old = io.input.raw.clone();
    io.input.raw = tables.coefficients.raw.clone();
    assert!(op.encode(&command, plan, io.invocation(&tables)).is_err());
    io.input.raw = old;
    let old = tables.gains.raw.clone();
    tables.gains.raw = io.status.raw.clone();
    assert!(op.encode(&command, plan, io.invocation(&tables)).is_err());
    tables.gains.raw = old;
    let old = tables.coefficients.len;
    tables.coefficients.len = 0;
    assert!(op.encode(&command, plan, io.invocation(&tables)).is_err());
    tables.coefficients.len = old;
    let old = tables.gains.len;
    tables.gains.len = 0;
    assert!(op.encode(&command, plan, io.invocation(&tables)).is_err());
    tables.gains.len = old;
    let devices = MTLCopyAllDevices();
    let other = devices
        .iter()
        .find(|d| d.registryID() != gpu.device.registryID());
    let mut foreign_checks = 0;
    if let Some(device) = other {
        let buffer = |bytes| {
            device
                .newBufferWithLength_options(
                    bytes,
                    MTLResourceOptions::StorageModeShared
                        | MTLResourceOptions::HazardTrackingModeTracked,
                )
                .unwrap()
        };
        let original = io.input.raw.clone();
        io.input.raw = buffer(original.length());
        assert!(op.encode(&command, plan, io.invocation(&tables)).is_err());
        io.input.raw = original;
        foreign_checks += 1;
        let original = io.output.raw.clone();
        io.output.raw = buffer(original.length());
        assert!(op.encode(&command, plan, io.invocation(&tables)).is_err());
        io.output.raw = original;
        foreign_checks += 1;
        let original = io.status.raw.clone();
        io.status.raw = buffer(original.length());
        assert!(op.encode(&command, plan, io.invocation(&tables)).is_err());
        io.status.raw = original;
        foreign_checks += 1;
        let original = tables.coefficients.raw.clone();
        tables.coefficients.raw = buffer(original.length());
        assert!(op.encode(&command, plan, io.invocation(&tables)).is_err());
        tables.coefficients.raw = original;
        foreign_checks += 1;
        let original = tables.gains.raw.clone();
        tables.gains.raw = buffer(original.length());
        assert!(op.encode(&command, plan, io.invocation(&tables)).is_err());
        tables.gains.raw = original;
        foreign_checks += 1;
        let queue = device.newCommandQueue().unwrap();
        let foreign = queue.commandBuffer().unwrap();
        assert!(
            op.encode(
                &Command {
                    raw: &foreign,
                    trace: None
                },
                plan,
                io.invocation(&tables)
            )
            .is_err()
        );
        foreign_checks += 1;
    }
    assert_eq!(raw.status(), MTLCommandBufferStatus::NotEnqueued);
    assert_eq!(table_hash(&tables), before);
    io.validate();
    assert!(
        io.output
            .with_slice(|v| v.iter().all(|&v| v == HALF_SENTINEL))
    );
    assert_eq!(io.status.with_slice(|v| v[0]), 0);
    json!({"same_device_preflight_rejections":11,"foreign_device_rejections":foreign_checks,"foreign_device_test_available":foreign_checks>0,"all_rejections_before_encoder_submission":true,"immutable_outputs_status_tables":true})
}

#[test]
#[ignore = "Explicit completed Metal nonfinite/first-store/second-store overflow and owner reuse"]
fn gpu_numeric_status_both_half_stores_and_owner_reuse() {
    autoreleasepool(|_| {
        let gpu = Gpu::new().unwrap();
        let op = Operation::new(&gpu, PIPELINE_ALLOWANCE + size_of::<Operation>()).unwrap();
        let mut checked = 0;
        for (base, tokens, words, label) in [
            (
                0,
                1,
                {
                    let mut v = vec![0; 128];
                    v[0] = 0x7e01;
                    v
                },
                "NaN",
            ),
            (
                0,
                1,
                {
                    let mut v = vec![0; 128];
                    v[127] = 0xfc00;
                    v
                },
                "negative_infinity",
            ),
            (
                0,
                2,
                {
                    let mut v = vec![0; 256];
                    v[128] = f16::MAX.to_bits();
                    v[192] = f16::MAX.to_bits();
                    v
                },
                "rotated_Half_overflow",
            ),
            (
                16_384,
                1,
                {
                    let mut v = vec![0; 128];
                    v[63] = f16::MAX.to_bits();
                    v
                },
                "scaled_query_Half_overflow",
            ),
        ] {
            let cpu = cpu_table(base, tokens);
            let tables = Tables::new(&gpu, &cpu, Tables::planned_bytes(tokens).unwrap()).unwrap();
            let before = table_hash(&tables);
            for (kind, name) in [(Kind::Keys, "keys"), (Kind::Queries, "queries")] {
                let plan = Plan::new(kind, tokens, 1, base, base, tokens).unwrap();
                let mut io = Io::new(&gpu, plan, &words);
                io.status.with_mut(|v| v[0] = 4);
                execute(&gpu, &op, &tables, &io);
                let status = io.status.with_slice(|v| v[0]);
                let should_fail = label != "scaled_query_Half_overflow" || name == "queries";
                assert_eq!(status, if should_fail { 5 } else { 4 });
                if should_fail {
                    assert!(io.output().iter().any(|&v| !f16::from_bits(v).is_finite()));
                } else {
                    assert!(io.output().iter().all(|&v| f16::from_bits(v).is_finite()));
                }
                println!(
                    "MINISTRAL3_METAL_STATUS {}",
                    json!({"case":label,"kind":name,"status":status,"prior_status":4,"numeric_failure_expected":should_fail,"completed":true,"partial_output_discard_required":should_fail})
                );
                let good = vec![f16::ONE.to_bits(); tokens * HEAD_DIM];
                let good = Io::new(&gpu, plan, &good);
                execute(&gpu, &op, &tables, &good);
                assert_eq!(good.status.with_slice(|v| v[0]), 0);
                assert!(good.output().iter().all(|&v| f16::from_bits(v).is_finite()));
                assert_eq!(table_hash(&tables), before);
                uploaded_tables(&cpu, &tables);
                checked += 1;
            }
        }
        assert_eq!(checked, 8);
        println!(
            "MINISTRAL3_METAL_STATUS_SUMMARY {}",
            json!({"checks":checked,"numeric_failures":7,"good_reuse_checks":8,"monotone_status_preserved":true,"device_tables_unchanged":true,"session_KV_state_owned":false})
        );
    });
}

#[test]
#[ignore = "Explicit release Metal Rust encode allocation and completed operator-clock diagnostics"]
fn gpu_release_encode_allocation_and_completed_operator_clocks() {
    assert!(
        !cfg!(debug_assertions),
        "Run this measurement in release mode"
    );
    autoreleasepool(|_| {
        let (fixture, _) = fixture();
        let setup = Instant::now();
        let gpu = Gpu::new().unwrap();
        let op = Operation::new(&gpu, PIPELINE_ALLOWANCE + size_of::<Operation>()).unwrap();
        println!(
            "MINISTRAL3_METAL_SETUP {}",
            json!({"device":gpu.device.name().to_string(),"device_pipeline_setup_seconds":setup.elapsed().as_secs_f64(),"pipeline_allowance_bytes":PIPELINE_ALLOWANCE,"CPU_profile_sha256":PROFILE_SHA,"fixture_sha256":FIXTURE_SHA,"release":true,"trace_enabled":false,"model_performance_qualified":false})
        );
        let (mut calls, mut rejected, mut allocations) = (0, 0, 0);
        for name in ["actual_heads0", "actual_heads_boundary"] {
            let setup = Instant::now();
            let case = fixture["cases"]
                .as_array()
                .unwrap()
                .iter()
                .find(|c| c["name"] == name)
                .unwrap();
            let (base, tokens) = (n(case, "offset"), n(case, "tokens"));
            let cpu = cpu_table(base, tokens);
            let tables = Tables::new(&gpu, &cpu, Tables::planned_bytes(tokens).unwrap()).unwrap();
            let before = table_hash(&tables);
            let key_plan =
                Plan::new(Kind::Keys, tokens, n(case, "kv_heads"), base, base, tokens).unwrap();
            let query_plan = Plan::new(
                Kind::Queries,
                tokens,
                n(case, "query_heads"),
                base,
                base,
                tokens,
            )
            .unwrap();
            let keys = Io::new(&gpu, key_plan, &words16(&case["key_input"]));
            let queries = Io::new(&gpu, query_plan, &words16(&case["query_input"]));
            println!(
                "MINISTRAL3_METAL_CASE_SETUP {}",
                json!({"case":name,"CPU_table_upload_operand_setup_seconds":setup.elapsed().as_secs_f64(),"CPU_table_owner_bytes":cpu.allocated_bytes().unwrap(),"GPU_table_owner_bytes":tables.owned_bytes(),"key_operand_bytes":key_plan.operand_bytes().unwrap(),"query_operand_bytes":query_plan.operand_bytes().unwrap()})
            );
            for io in [&keys, &queries] {
                for _ in 0..3 {
                    execute(&gpu, &op, &tables, io);
                }
                for _ in 0..12 {
                    let raw = gpu.command().unwrap();
                    let command = Command {
                        raw: &raw,
                        trace: None,
                    };
                    let measured = Measurement::start();
                    let result = op.encode(&command, io.plan, io.invocation(&tables));
                    let count = measured.finish();
                    result.unwrap();
                    assert_eq!(count, 0);
                    allocations += count;
                    calls += 1;
                    complete(&raw);
                    io.validate();
                    assert_eq!(io.status.with_slice(|v| v[0]), 0);
                }
                for _ in 0..4 {
                    let raw = gpu.command().unwrap();
                    let command = Command {
                        raw: &raw,
                        trace: None,
                    };
                    let measured = Measurement::start();
                    let result = op.encode(
                        &command,
                        io.plan,
                        Invocation {
                            output: &io.input,
                            ..io.invocation(&tables)
                        },
                    );
                    let count = measured.finish();
                    assert!(result.is_err());
                    assert_eq!(count, 0);
                    allocations += count;
                    rejected += 1;
                    assert_eq!(raw.status(), MTLCommandBufferStatus::NotEnqueued);
                }
            }
            for run in 1..=2 {
                for sample in 0..13 {
                    let raw = gpu.command().unwrap();
                    let command = Command {
                        raw: &raw,
                        trace: None,
                    };
                    let completed = Instant::now();
                    let encode = Instant::now();
                    op.encode(&command, key_plan, keys.invocation(&tables))
                        .unwrap();
                    op.encode(&command, query_plan, queries.invocation(&tables))
                        .unwrap();
                    let encode_seconds = encode.elapsed().as_secs_f64();
                    let submit = Instant::now();
                    complete(&raw);
                    let submit_wait_seconds = submit.elapsed().as_secs_f64();
                    let completed_seconds = completed.elapsed().as_secs_f64();
                    let gpu_seconds = raw.GPUEndTime() - raw.GPUStartTime();
                    assert!(gpu_seconds.is_finite() && gpu_seconds > 0.);
                    assert!(encode_seconds.is_finite() && encode_seconds >= 0.);
                    assert!(completed_seconds.is_finite() && completed_seconds > 0.);
                    keys.validate();
                    queries.validate();
                    assert_eq!(keys.status.with_slice(|v| v[0]), 0);
                    assert_eq!(queries.status.with_slice(|v| v[0]), 0);
                    compare(
                        "keys",
                        &halves(&keys.output()),
                        &halves(&words16(&case["references"]["main"]["keys"])),
                        &halves(&words16(&case["references"]["wide"]["keys"])),
                    );
                    compare(
                        "queries",
                        &halves(&queries.output()),
                        &halves(&words16(&case["references"]["main"]["queries"])),
                        &halves(&words16(&case["references"]["wide"]["queries"])),
                    );
                    println!(
                        "MINISTRAL3_METAL_OPERATOR_SAMPLE {}",
                        json!({"case":name,"tokens":tokens,"query_heads":n(case,"query_heads"),"kv_heads":n(case,"kv_heads"),"absolute_offset":base,"run":run,"sample":sample,"warmup":sample<3,"host_encode_seconds":encode_seconds,"host_submit_wait_seconds":submit_wait_seconds,"host_encode_to_completion_seconds":completed_seconds,"completed_command_GPU_seconds":gpu_seconds,"encoded_operations":2,"submissions":1,"completion_waits":1,"trace_enabled":false,"key_output_sha256":hash16(&keys.output()),"query_output_sha256":hash16(&queries.output()),"operator_diagnostic_only":true,"engine_speedup_qualified":false,"summary_observer_overhead_qualified":false})
                    );
                }
            }
            assert_eq!(table_hash(&tables), before);
            uploaded_tables(&cpu, &tables);
        }
        assert_eq!(calls, 48);
        assert_eq!(rejected, 16);
        assert_eq!(allocations, 0);
        println!(
            "MINISTRAL3_METAL_ALLOCATION_SUMMARY {}",
            json!({"release_admitted_encode_calls":calls,"rejected_encode_calls":rejected,"measured_thread_Rust_allocations":allocations,"driver_allocations_measured":false,"diagnostic_cases":2,"runs_per_case":2,"warmups_per_run":3,"retained_samples_per_run":10,"total_operator_samples":52,"end_to_end_latency_qualified":false,"O1_observer_overhead_qualified":false})
        );
    });
}
