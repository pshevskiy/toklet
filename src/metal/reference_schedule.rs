//! Explicit unqualified scheduling experiment for unchanged Reference arithmetic.
//! A model owns the optional pipelines; a session borrows them through completion.
use super::{
    Error, Result,
    memory::{Buffer, Gpu},
    profile::Command,
};
use crate::{mlx::dense::Architecture, model, session};
use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_foundation::NSString;
use objc2_metal::*;
use std::ptr::NonNull;

/// Independent-output scheduling only. Serial remains every existing default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub enum ReferenceSchedule {
    #[default]
    Serial,
    /// Unqualified, explicitly admitted Qwen3 diagnostic experiment.
    ParallelOutputsExperimental,
}

pub(super) fn validate_options(
    admitted: bool,
    architecture: Option<Architecture>,
    config: &model::Config,
    options: session::Options,
) -> Result<()> {
    if !admitted || architecture != Some(Architecture::Qwen3) {
        return Err(Error::Options(
            "parallel Reference requires the explicit Qwen3 context diagnostic model",
        ));
    }
    crate::mlx::execution::qwen3_context_geometry(config, Architecture::Qwen3)
        .map_err(|e| Error::Options(e.message()))?;
    if options.position_offset != 0
        || !(1..=4096).contains(&options.capacity)
        || !matches!(options.prefill_chunk_size, 32 | 128)
    {
        return Err(Error::Options(
            "Reference schedule diagnostic requires offset0, capacity1..4096 and chunk32/128",
        ));
    }
    Ok(())
}

fn product(parts: &[usize]) -> Result<usize> {
    let count = parts
        .iter()
        .try_fold(1usize, |n, v| n.checked_mul(*v))
        .ok_or(Error::Options("parallel Reference index overflow"))?;
    if count == 0 || count > u32::MAX as usize {
        return Err(Error::Options("parallel Reference index outside UInt32"));
    }
    Ok(count)
}
struct Shape {
    grid: [usize; 3],
    bytes: [usize; 6],
}
impl Shape {
    fn new(shape: [u32; 8]) -> Result<Self> {
        let [rows, heads, kvheads, dim, cap, chunk, offset, valid] = shape.map(|v| v as usize);
        if rows == 0
            || rows > chunk
            || !(matches!(chunk, 32 | 128)
                || (cfg!(test) && (heads, kvheads, dim) == (4, 2, 16) && (1..=32).contains(&chunk)))
            || !((heads, kvheads, dim) == (16, 8, 128)
                || (cfg!(test) && (heads, kvheads, dim) == (4, 2, 16)))
            || valid == 0
            || valid > 4096
            || valid > cap
            || offset.checked_add(rows).is_none_or(|end| end > valid)
        {
            return Err(Error::Options(
                "parallel Reference shape outside diagnostic scope",
            ));
        }
        // Padded capacity is permitted for primitive guards; public session
        // admission separately caps its logical capacity at4096, offset0.
        let query = product(&[rows, heads, dim])?;
        let cache = product(&[kvheads, cap, dim])?;
        let scratch = product(&[heads, chunk, cap])?;
        let mut bytes = [query, cache, cache, scratch, scratch, query];
        for (n, width) in bytes.iter_mut().zip([2, 2, 2, 4, 4, 2]) {
            *n = n
                .checked_mul(width)
                .filter(|v| *v <= isize::MAX as usize)
                .ok_or(Error::Options("parallel Reference byte count overflow"))?;
        }
        Ok(Self {
            grid: [
                product(&[rows, heads, valid])?,
                product(&[rows, heads])?,
                query,
            ],
            bytes,
        })
    }
}
fn grid(count: usize, width: usize, maximum: usize) -> Result<MTLSize> {
    if count == 0 || width == 0 || width > maximum {
        return Err(Error::Unsupported(
            "parallel Reference launch width unsupported",
        ));
    }
    let groups = count.div_ceil(width);
    if groups
        .checked_mul(width)
        .is_none_or(|n| n > u32::MAX as usize)
    {
        return Err(Error::Options("parallel Reference rounded grid overflow"));
    }
    Ok(MTLSize {
        width: groups,
        height: 1,
        depth: 1,
    })
}

pub(super) struct Pipelines {
    labels: [Retained<NSString>; 3],
    maximum_group_width: usize,
    scores: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    softmax: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    context: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}
impl Pipelines {
    pub(super) fn new(gpu: &Gpu) -> Result<Self> {
        autoreleasepool(|_| {
            // Same policy as the Serial library, initialized only at explicit cold
            // experimental admission. These widths cannot alter Gpu::width.
            let options = MTLCompileOptions::new();
            options.setLanguageVersion(MTLLanguageVersion::Version2_3);
            options.setMathMode(MTLMathMode::Safe);
            options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
            if options.languageVersion() != MTLLanguageVersion::Version2_3
                || options.mathMode() != MTLMathMode::Safe
                || options.mathFloatingPointFunctions() != MTLMathFloatingPointFunctions::Precise
            {
                return Err(Error::Unsupported(
                    "precise parallel Reference compile policy unavailable",
                ));
            }
            let library = gpu
                .device
                .newLibraryWithSource_options_error(
                    &NSString::from_str(&format!(
                        "{}\n{}",
                        include_str!("kernels.metal"),
                        include_str!("reference_schedule.metal")
                    )),
                    Some(&options),
                )
                .map_err(|e| Error::Native(e.to_string()))?;
            let compile = |name| {
                let function = library
                    .newFunctionWithName(&NSString::from_str(name))
                    .ok_or_else(|| Error::Native("parallel Reference function absent".into()))?;
                gpu.device
                    .newComputePipelineStateWithFunction_error(&function)
                    .map_err(|e| Error::Native(e.to_string()))
            };
            Ok(Self {
                labels: [
                    "reference_scores_parallel",
                    "reference_softmax_serial",
                    "reference_context_parallel",
                ]
                .map(NSString::from_str),
                maximum_group_width: gpu.device.maxThreadsPerThreadgroup().width,
                scores: compile("attention_reference_scores_parallel")?,
                softmax: compile("attention_reference_softmax_serial")?,
                context: compile("attention_reference_context_parallel")?,
            })
        })
    }
    pub(super) fn encode(
        &self,
        command: &Command<'_>,
        buffers: [&ProtocolObject<dyn MTLBuffer>; 6],
        shape: [u32; 8],
        status: &Buffer<u32>,
    ) -> Result<()> {
        let checked = Shape::new(shape)?;
        if status.len < 1
            || buffers
                .iter()
                .zip(checked.bytes)
                .any(|(b, n)| b.length() < n)
        {
            return Err(Error::Options("parallel Reference buffer extent differs"));
        }
        let passes = [
            (&*self.scores, "reference_scores_parallel"),
            (&*self.softmax, "reference_softmax_serial"),
            (&*self.context, "reference_context_parallel"),
        ];
        // Validate every dispatch before creating an encoder; no partial encode
        // from a later invalid launch. Validation uses stack storage; no new
        // numerical buffer or Vec is allocated here.
        let mut grids = [MTLSize {
            width: 0,
            height: 1,
            depth: 1,
        }; 3];
        for (i, (pipeline, _)) in passes.iter().enumerate() {
            grids[i] = grid(
                checked.grid[i],
                pipeline.threadExecutionWidth(),
                pipeline
                    .maxTotalThreadsPerThreadgroup()
                    .min(self.maximum_group_width),
            )?;
        }
        for (i, (pipeline, name)) in passes.iter().enumerate() {
            let (index, encoder) = if let Some(trace) = command.trace {
                let (index, encoder) = trace.encoder(command.raw, name)?;
                (Some(index), encoder)
            } else {
                (
                    None,
                    command
                        .computeCommandEncoder()
                        .ok_or_else(|| Error::Native("parallel Reference encoder absent".into()))?,
                )
            };
            encoder.setLabel(Some(&self.labels[i]));
            encoder.setComputePipelineState(pipeline);
            // SAFETY: private session/tests bind same-device owned disjoint
            // buffers with checked extents and u16/u32 alignment. Each output
            // index has one writer; tracked hazards and ended encoders order
            // cache -> scores -> probabilities -> context. Inputs/resources
            // remain retained through command completion; no CPU access in flight.
            unsafe {
                for (index, buffer) in buffers.iter().enumerate() {
                    encoder.setBuffer_offset_atIndex(Some(buffer), 0, index);
                }
                encoder.setBytes_length_atIndex(
                    NonNull::new(shape.as_ptr().cast_mut()).unwrap().cast(),
                    std::mem::size_of_val(&shape),
                    6,
                );
                encoder.setBuffer_offset_atIndex(Some(&status.raw), 0, 7);
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                grids[i],
                MTLSize {
                    width: pipeline.threadExecutionWidth(),
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            if let Some(index) = index {
                command.trace.unwrap().finish_encoder(index);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights;

    fn config() -> model::Config {
        model::Config {
            hidden_size: 1024,
            intermediate_size: 3072,
            query_heads: 16,
            kv_heads: 8,
            head_dim: 128,
            num_layers: 28,
            vocab_size: 151936,
            max_positions: 40960,
            rope_theta: 1e6,
            rms_norm_eps: 1e-6,
        }
    }
    fn options() -> session::Options {
        session::Options {
            capacity: 4096,
            prefill_chunk_size: 32,
            position_offset: 0,
            memory_limit_bytes: weights::INITIAL_MEMORY_LIMIT_BYTES,
        }
    }
    #[test]
    fn cold_schedule_admission_cannot_grant_model_or_position_capability() {
        assert_eq!(ReferenceSchedule::default(), ReferenceSchedule::Serial);
        let c = config();
        let o = options();
        validate_options(true, Some(Architecture::Qwen3), &c, o).unwrap();
        assert!(validate_options(false, Some(Architecture::Qwen3), &c, o).is_err());
        for architecture in [None, Some(Architecture::Qwen2), Some(Architecture::Llama)] {
            assert!(validate_options(true, architecture, &c, o).is_err());
        }
        for capacity in [0, 4097, 40960, usize::MAX] {
            assert!(
                validate_options(
                    true,
                    Some(Architecture::Qwen3),
                    &c,
                    session::Options { capacity, ..o }
                )
                .is_err()
            );
        }
        for position_offset in [1, 4095, 4096, 40960, usize::MAX] {
            assert!(
                validate_options(
                    true,
                    Some(Architecture::Qwen3),
                    &c,
                    session::Options {
                        position_offset,
                        ..o
                    }
                )
                .is_err()
            );
        }
        for prefill_chunk_size in [0, 1, 31, 33, 127, 129, usize::MAX] {
            assert!(
                validate_options(
                    true,
                    Some(Architecture::Qwen3),
                    &c,
                    session::Options {
                        prefill_chunk_size,
                        ..o
                    }
                )
                .is_err()
            );
        }
        for capacity in [512, 1025, 4096] {
            for prefill_chunk_size in [32, 128] {
                validate_options(
                    true,
                    Some(Architecture::Qwen3),
                    &c,
                    session::Options {
                        capacity,
                        prefill_chunk_size,
                        ..o
                    },
                )
                .unwrap();
            }
        }
        let mut wrong = config();
        wrong.head_dim = 64;
        assert!(validate_options(true, Some(Architecture::Qwen3), &wrong, o).is_err());
        wrong = config();
        wrong.num_layers = 27;
        assert!(validate_options(true, Some(Architecture::Qwen3), &wrong, o).is_err());
    }
    #[test]
    fn shape_and_launch_bounds_precede_encoding() {
        let shape = [32, 16, 8, 128, 4096, 32, 4064, 4096];
        let checked = Shape::new(shape).unwrap();
        assert_eq!(checked.grid, [32 * 16 * 4096, 32 * 16, 32 * 16 * 128]);
        assert_eq!(
            checked.bytes,
            [131072, 8388608, 8388608, 8388608, 8388608, 131072]
        );
        // The tiny fixture's offset5 leaves capacity27. Its test-only
        // chunk27 must work without admitting that chunk for production Qwen3.
        let tiny = Shape::new([3, 4, 2, 16, 27, 27, 0, 3]).unwrap();
        assert_eq!(tiny.grid, [36, 12, 192]);
        assert!(Shape::new([3, 16, 8, 128, 27, 27, 0, 3]).is_err());
        assert!(Shape::new([3, 4, 2, 16, 27, 0, 0, 3]).is_err());
        assert!(Shape::new([3, 4, 2, 16, 33, 33, 0, 3]).is_err());
        // Primitive guards and one valid future key are not session admission.
        Shape::new([1, 16, 8, 128, 4099, 32, 4094, 4096]).unwrap();
        for (index, value) in [
            (0, 0),
            (0, 33),
            (1, 15),
            (2, 16),
            (3, 64),
            (4, 4095),
            (5, 64),
            (6, 4096),
            (7, 4097),
        ] {
            let mut bad = shape;
            bad[index] = value;
            assert!(Shape::new(bad).is_err());
        }
        let mut bad = shape;
        bad[4] = u32::MAX;
        assert!(Shape::new(bad).is_err());
        assert!(product(&[usize::MAX, 2]).is_err());
        for (count, width, maximum) in [
            (0, 32, 32),
            (1, 0, 32),
            (1, 64, 32),
            (u32::MAX as usize, 32, 32),
        ] {
            assert!(grid(count, width, maximum).is_err());
        }
        assert_eq!(grid(33, 32, 32).unwrap().width, 2);
    }

    const GUARD: usize = 7;
    const HALF_GUARD: u16 = 0x3555;
    const FLOAT_GUARD: u32 = 0x7fa1_2345;
    #[derive(Clone, Copy, Debug)]
    enum Values {
        General,
        Uniform,
        Dominant(usize),
    }
    fn upload(gpu: &Gpu, values: &[u16]) -> Buffer<u16> {
        let mut buffer = Buffer::new(gpu, values.len() + GUARD).unwrap();
        buffer.with_mut(|b| {
            b.fill(HALF_GUARD);
            b[..values.len()].copy_from_slice(values)
        });
        buffer
    }
    fn hash(words: &[u32]) -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        for word in words {
            h.update(word.to_le_bytes());
        }
        format!("{:x}", h.finalize())
    }
    fn compare(name: &str, expected: &[u32], actual: &[u32], row_width: usize) {
        let mismatch = expected.iter().zip(actual).position(|(a, b)| a != b);
        println!(
            "{}",
            serde_json::json!({"kind":"parallel_reference_bits","buffer":name,
            "expected_words":expected.len(),"actual_words":actual.len(),"expected_sha256":hash(expected),
            "actual_sha256":hash(actual),"first_mismatch":mismatch.map(|i|serde_json::json!({"index":i,"expected":expected[i],"actual":actual[i]}))})
        );
        if let Some(index) = mismatch {
            let start = index / row_width * row_width;
            let end = (start + row_width).min(expected.len()).min(actual.len());
            println!(
                "{}",
                serde_json::json!({"kind":"parallel_reference_failing_row",
                "buffer":name,"start":start,"expected_bits":expected[start..end],"actual_bits":actual[start..end]})
            );
        }
        assert_eq!(actual.len(), expected.len());
        assert!(mismatch.is_none(), "{name} exact bits differ");
    }
    #[test]
    #[ignore = "Explicit proposed Serial/parallel exact attention controls; no model"]
    fn parallel_reference_complete_arrays_and_deep_analytical_controls() {
        use super::super::memory::Kernel;
        use half::f16;
        autoreleasepool(|_| {
            let gpu = Gpu::new().unwrap();
            let old_width = gpu.width;
            let pipelines = gpu.parallel_reference().unwrap();
            assert_eq!(gpu.width, old_width);
            assert!(std::ptr::eq(pipelines, gpu.parallel_reference().unwrap()));
            // Fixed cases: analytical means/selections plus irregular multirow
            // nontrivial data. No fixture or allowance depends on GPU outcomes.
            let cases = [
                (1, 32, 1023, 1024, Values::Uniform),
                (1, 32, 2047, 2048, Values::Uniform),
                (1, 32, 4095, 4096, Values::Uniform),
                (1, 32, 4094, 4096, Values::Dominant(0)),
                (1, 32, 4094, 4096, Values::Dominant(2047)),
                (1, 32, 4094, 4096, Values::Dominant(4094)),
                (1, 32, 0, 1, Values::General),
                (1, 32, 4095, 4096, Values::General),
                (31, 32, 1018, 1049, Values::General),
                (32, 32, 4064, 4096, Values::General),
                (127, 128, 2048, 2175, Values::General),
                (128, 128, 3968, 4096, Values::General),
            ];
            for (rows, chunk, offset, valid, values) in cases {
                let (heads, kvheads, dim) = (16, 8, 128);
                let cap = valid + 3;
                let shape =
                    [rows, heads, kvheads, dim, cap, chunk, offset, valid].map(|n| n as u32);
                let mut q_values = vec![HALF_GUARD; chunk * heads * dim];
                for (i, value) in q_values[..rows * heads * dim].iter_mut().enumerate() {
                    *value = f16::from_f32(match values {
                        Values::Uniform => 0.,
                        Values::Dominant(_) => {
                            if i % dim == 0 {
                                8.
                            } else {
                                0.
                            }
                        }
                        Values::General => ((i * 17 % 61) as i32 - 30) as f32 / 64.,
                    })
                    .to_bits();
                }
                let mut k_values = vec![0x7e00; kvheads * cap * dim];
                let mut v_values = k_values.clone();
                for head in 0..kvheads {
                    for position in 0..valid {
                        for feature in 0..dim {
                            let index = (head * cap + position) * dim + feature;
                            let (key, value) = match values {
                                Values::Uniform => (
                                    0.,
                                    (head + 1) as f32 / 16.
                                        + if (position + feature) % 2 == 0 {
                                            1. / 16.
                                        } else {
                                            -1. / 16.
                                        },
                                ),
                                Values::Dominant(winner) => (
                                    if feature != 0 {
                                        0.
                                    } else if position > offset {
                                        4096.
                                    } else if position == winner {
                                        128.
                                    } else {
                                        -128.
                                    },
                                    (head + 1) as f32 / 16.
                                        + ((position % 17) as i32 - 8) as f32 / 64.
                                        + ((feature % 5) as i32 - 2) as f32 / 128.,
                                ),
                                Values::General => (
                                    ((head * 11 + position * 7 + feature * 3) % 53) as f32 / 32.
                                        - 26. / 32.,
                                    ((head * 13 + position * 5 + feature * 19) % 97) as f32 / 64.
                                        - 48. / 64.,
                                ),
                            };
                            k_values[index] = f16::from_f32(key).to_bits();
                            if !matches!(values, Values::Dominant(_)) || position <= offset {
                                v_values[index] = f16::from_f32(value).to_bits();
                            }
                        }
                    }
                }
                let q = upload(&gpu, &q_values);
                let k = upload(&gpu, &k_values);
                let v = upload(&gpu, &v_values);
                let mut output = upload(&gpu, &vec![HALF_GUARD; chunk * heads * dim]);
                let scratch = heads * chunk * cap;
                let mut scores = Buffer::<u32>::new(&gpu, scratch + GUARD).unwrap();
                let mut probs = Buffer::<u32>::new(&gpu, scratch + GUARD).unwrap();
                let mut status = Buffer::<u32>::new(&gpu, 8).unwrap();
                let mut baseline: Option<[Vec<u32>; 4]> = None;
                for schedule in [
                    ReferenceSchedule::Serial,
                    ReferenceSchedule::ParallelOutputsExperimental,
                ] {
                    output.with_mut(|b| b.fill(HALF_GUARD));
                    scores.with_mut(|b| b.fill(FLOAT_GUARD));
                    probs.with_mut(|b| b.fill(FLOAT_GUARD));
                    status.with_mut(|b| b.fill(0));
                    let raw = gpu.command().unwrap();
                    let command = Command {
                        raw: &raw,
                        trace: None,
                    };
                    let buffers: [&ProtocolObject<dyn MTLBuffer>; 6] =
                        [&q.raw, &k.raw, &v.raw, &scores.raw, &probs.raw, &output.raw];
                    match schedule {
                        ReferenceSchedule::Serial => gpu
                            .encode(
                                &command,
                                Kernel::Attention,
                                &buffers,
                                &shape,
                                rows * heads,
                                Some(&status),
                            )
                            .unwrap(),
                        ReferenceSchedule::ParallelOutputsExperimental => {
                            pipelines.encode(&command, buffers, shape, &status).unwrap()
                        }
                    }
                    raw.commit();
                    raw.waitUntilCompleted();
                    let actual = [
                        scores.with_slice(|b| b.to_vec()),
                        probs.with_slice(|b| b.to_vec()),
                        output.with_slice(|b| b.iter().map(|v| u32::from(*v)).collect()),
                        status.with_slice(|b| b.to_vec()),
                    ];
                    let input_preserved = [(&q, &q_values), (&k, &k_values), (&v, &v_values)]
                        .iter()
                        .all(|(b, original)| {
                            b.with_slice(|words| {
                                words[..original.len()] == **original
                                    && words[original.len()..].iter().all(|v| *v == HALF_GUARD)
                            })
                        });
                    let scratch_padding = [&actual[0], &actual[1]].iter().all(|words| {
                        (0..heads).all(|h| {
                            (0..chunk).all(|t| {
                                let base = (h * chunk + t) * cap;
                                let used = if t < rows { valid } else { 0 };
                                words[base + used..base + cap]
                                    .iter()
                                    .all(|v| *v == FLOAT_GUARD)
                            })
                        }) && words[scratch..].iter().all(|v| *v == FLOAT_GUARD)
                    });
                    let output_padding = actual[2][rows * heads * dim..]
                        .iter()
                        .all(|v| *v == u32::from(HALF_GUARD));
                    println!(
                        "{}",
                        serde_json::json!({"kind":"parallel_reference_case","shape":shape,"case":format!("{values:?}"),
                        "schedule":schedule,"completed":raw.status()==MTLCommandBufferStatus::Completed,
                        "status":actual[3],"inputs_preserved":input_preserved,"scratch_padding":scratch_padding,"output_padding":output_padding})
                    );
                    assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
                    assert_eq!(actual[3], [0; 8]);
                    assert!(input_preserved && scratch_padding && output_padding);
                    if !matches!(values, Values::General) {
                        let expected: Vec<u32> = (0..heads * dim)
                            .map(|i| {
                                let head = i / dim / 2;
                                let feature = i % dim;
                                let value = (head + 1) as f32 / 16.
                                    + match values {
                                        Values::Dominant(p) => {
                                            ((p % 17) as i32 - 8) as f32 / 64.
                                                + ((feature % 5) as i32 - 2) as f32 / 128.
                                        }
                                        _ => 0.,
                                    };
                                u32::from(f16::from_f32(value).to_bits())
                            })
                            .collect();
                        compare(
                            "analytical_context",
                            &expected,
                            &actual[2][..heads * dim],
                            dim,
                        );
                        for head in 0..heads {
                            for p in 0..valid {
                                let expected = match values {
                                    Values::Uniform => 1. / valid as f32,
                                    Values::Dominant(winner) => {
                                        if p == winner {
                                            1.
                                        } else {
                                            0.
                                        }
                                    }
                                    _ => unreachable!(),
                                };
                                let observed = actual[1][head * chunk * cap + p];
                                if observed != expected.to_bits() {
                                    println!(
                                        "{}",
                                        serde_json::json!({"kind":"analytical_probability_failure","head":head,"position":p,"expected_bits":expected.to_bits(),"actual_bits":observed})
                                    );
                                }
                                assert_eq!(observed, expected.to_bits());
                            }
                        }
                    }
                    if let Some(expected) = &baseline {
                        for (name, (expected, actual)) in
                            ["scores", "probabilities", "context", "status"]
                                .into_iter()
                                .zip(expected.iter().zip(&actual))
                        {
                            compare(
                                name,
                                expected,
                                actual,
                                match name {
                                    "scores" | "probabilities" => cap,
                                    "context" => dim,
                                    _ => 8,
                                },
                            );
                        }
                    } else {
                        baseline = Some(actual);
                    }
                }
            }
        });
    }
    #[test]
    #[ignore = "Explicit proposed tied/untied tiny graph bit bridge; opens existing fixture and GPU"]
    fn parallel_reference_tiny_graph_stage_bits() {
        use super::super::{
            AttentionMode, Model, NormalizationMode, ProjectionMode, ReadbackMode, Session,
        };
        use sha2::{Digest, Sha256};
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/mlx-qwen3-v1");
        let bytes = std::fs::read(root.join("manifest.json")).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            "5dc5b7ec0fccd8ac6876814b88a917107993aab5f5e07dcb222f57ec377a7a4a"
        );
        let manifest: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        for (path, pin) in manifest["files"].as_object().unwrap() {
            let bytes = std::fs::read(root.join(path)).unwrap();
            assert_eq!(bytes.len() as u64, pin["bytes"]);
            assert_eq!(format!("{:x}", Sha256::digest(&bytes)), pin["sha256"]);
        }
        for variant in ["tied", "untied"] {
            let model = Model::load_mlx(
                &root.join(variant),
                weights::LoadOptions {
                    context_limit: 32,
                    memory_limit_bytes: 128 * 1024 * 1024,
                },
            )
            .unwrap();
            for (case, spec) in manifest["variants"][variant]["cases"].as_object().unwrap() {
                let tokens: Vec<u32> = spec["tokens"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u32)
                    .collect();
                let offset = spec["position_offset"].as_u64().unwrap() as usize;
                let capacity = 32 - offset;
                let options = session::Options {
                    capacity,
                    prefill_chunk_size: 32.min(capacity),
                    position_offset: offset,
                    memory_limit_bytes: 256 * 1024 * 1024,
                };
                assert!(
                    Session::with_qwen3_reference_schedule_diagnostic(
                        &model,
                        options,
                        ReadbackMode::FullLogits,
                        ReferenceSchedule::ParallelOutputsExperimental
                    )
                    .is_err()
                );
                let mut serial = Session::with_attention_mode(
                    &model,
                    options,
                    ProjectionMode::MlxAffine,
                    NormalizationMode::Compensated,
                    ReadbackMode::FullLogits,
                    AttentionMode::Reference,
                )
                .unwrap();
                let mut parallel = Session::with_attention_mode(
                    &model,
                    options,
                    ProjectionMode::MlxAffine,
                    NormalizationMode::Compensated,
                    ReadbackMode::FullLogits,
                    AttentionMode::Reference,
                )
                .unwrap();
                assert_eq!(serial.reference_schedule(), ReferenceSchedule::Serial);
                assert_eq!(parallel.reference_schedule(), ReferenceSchedule::Serial);
                // Unit-test-only injection, inaccessible to integrations/public
                // constructors. This cannot grant a tiny-model app capability.
                parallel.parallel_reference = Some(model.gpu.parallel_reference().unwrap());
                for session in [&mut serial, &mut parallel] {
                    session.enable_diagnostics(&[0, 1]).unwrap();
                    session.append(&tokens).unwrap();
                    session.evaluate().unwrap();
                }
                let expected = serial.diagnostics().unwrap();
                let actual = parallel.diagnostics().unwrap();
                println!(
                    "{}",
                    serde_json::json!({"kind":"parallel_reference_tiny_graph","variant":variant,"case":case,"expected_stages":expected.len(),"actual_stages":actual.len(),"expected_history":serial.history(),"actual_history":parallel.history(),"expected_evaluated":serial.evaluated_len(),"actual_evaluated":parallel.evaluated_len()})
                );
                assert_eq!(expected.len(), actual.len());
                assert_eq!(serial.history(), parallel.history());
                assert_eq!(serial.evaluated_len(), parallel.evaluated_len());
                assert_eq!(serial.pending_len(), parallel.pending_len());
                for (expected, actual) in expected.iter().zip(&actual) {
                    assert_eq!(
                        (
                            expected.stage,
                            expected.layer,
                            expected.rows,
                            expected.columns
                        ),
                        (actual.stage, actual.layer, actual.rows, actual.columns)
                    );
                    let e: Vec<_> = expected.values.iter().map(|v| v.to_bits()).collect();
                    let a: Vec<_> = actual.values.iter().map(|v| v.to_bits()).collect();
                    compare(expected.stage, &e, &a, expected.columns);
                }
            }
        }
    }
}
