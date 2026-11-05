//! Bounded decode-only split-KV witness. No session or model policy selects it.
use super::*;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandEncoder, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLLibrary,
    MTLSize,
};

type SplitPipeline =
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLComputePipelineState>>;
type MetalBuffer = objc2::runtime::ProtocolObject<dyn MTLBuffer>;

fn split_pipelines(gpu: &Gpu) -> [SplitPipeline; 2] {
    let library = mlx::precise_library(gpu, include_str!("split_kv.metal")).unwrap();
    ["attention_split_partial", "attention_split_merge"].map(|name| {
        let function = library
            .newFunctionWithName(&NSString::from_str(name))
            .unwrap();
        gpu.device
            .newComputePipelineStateWithFunction_error(&function)
            .unwrap()
    })
}

fn encode_split(
    raw: &objc2::runtime::ProtocolObject<dyn MTLCommandBuffer>,
    pipelines: &[SplitPipeline; 2],
    bindings: [&[&MetalBuffer]; 2],
    shape: &[u32; 6],
    status: &Buffer<u32>,
) {
    let heads = shape[0] as usize;
    let live = (shape[4] as usize).div_ceil(256);
    for ((pipeline, buffers), groups) in pipelines.iter().zip(bindings).zip([heads * live, heads]) {
        let width = pipeline.threadExecutionWidth();
        assert!(width >= 32 && width <= pipeline.maxTotalThreadsPerThreadgroup());
        let encoder = raw.computeCommandEncoder().unwrap();
        encoder.setComputePipelineState(pipeline);
        // SAFETY: All logical dimensions are the checked fixture
        // bounds above. Each stage fits its buffers and dispatches
        // one SIMD group per live packet/head. The two encoders run
        // in order with tracked resources, all live through completion.
        unsafe {
            for (index, buffer) in buffers.iter().enumerate() {
                encoder.setBuffer_offset_atIndex(Some(buffer), 0, index);
            }
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::from(shape).cast(),
                size_of_val(shape),
                buffers.len(),
            );
            encoder.setBuffer_offset_atIndex(Some(&status.raw), 0, buffers.len() + 1);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
    }
}

#[test]
#[ignore = "explicit Apple GPU split-KV, independent oracle and poisoned tails"]
fn split_kv_matches_fp64_at_tile_and_partition_boundaries() {
    autoreleasepool(|_| {
        let gpu = Gpu::new().unwrap();
        let pipelines = split_pipelines(&gpu);
        let (heads, kv_heads, dimension, capacity) = (14usize, 2usize, 64usize, 1152usize);
        let partitions = capacity.div_ceil(256);
        let active = heads * dimension;
        let cache_elements = kv_heads * capacity * dimension;
        let workspace = heads * partitions * (dimension + 2);
        assert_eq!(workspace * size_of::<u32>(), 18_480);
        let mut query = Buffer::<u16>::new(&gpu, active + 8).unwrap();
        let mut key = Buffer::<u16>::new(&gpu, cache_elements + 8).unwrap();
        let mut value = Buffer::<u16>::new(&gpu, cache_elements + 8).unwrap();
        let mut partial = Buffer::<u32>::new(&gpu, workspace + 8).unwrap();
        let mut output = Buffer::<u16>::new(&gpu, active + 8).unwrap();
        let mut status = Buffer::<u32>::new(&gpu, 8).unwrap();
        let nan_half = f16::NAN.to_bits();
        let nan_float = f32::NAN.to_bits();
        let half_value = |bits| f64::from(f16::from_bits(bits).to_f32());
        let mut compared = 0;
        let mut maximum_error = 0.0f64;
        for depth in [1usize, 15, 16, 17, 255, 256, 257, 511, 512, 513, 1024, 1152] {
            for pattern in 0..3 {
                let fill = |buffer: &mut Buffer<u16>, is_query: bool, is_key: bool| {
                    buffer.with_mut(|words| {
                        words.fill(nan_half);
                        let rows = if is_query { heads } else { kv_heads * depth };
                        for row in 0..rows {
                            let position = row % depth;
                            let start = if is_query {
                                row * dimension
                            } else {
                                ((row / depth) * capacity + position) * dimension
                            };
                            for (feature, word) in
                                words[start..start + dimension].iter_mut().enumerate()
                            {
                                let x = match (pattern, is_query, is_key) {
                                    (1, true, _) => 1.0,
                                    (1, _, true) => (position / 256) as f32 * 8.0 - 16.0,
                                    (2, _, true) => 0.0,
                                    _ => {
                                        ((row * 31
                                            + feature * 19
                                            + usize::from(is_key) * 11
                                            + usize::from(is_query) * 7)
                                            % 97) as f32
                                            / 32.0
                                            - 1.5
                                    }
                                };
                                *word = f16::from_f32(x).to_bits();
                            }
                        }
                    });
                };
                fill(&mut query, true, false);
                fill(&mut key, false, true);
                fill(&mut value, false, false);
                // The final case injects an active nonfinite value, independently
                // of the NaNs in all unwritten cache slots and packet tails.
                let invalid = depth == 257 && pattern == 2;
                if invalid {
                    value.with_mut(|words| words[256 * dimension] = nan_half);
                }
                let inputs = [&query, &key, &value].map(|buffer| buffer.with_slice(|x| x.to_vec()));
                partial.with_mut(|words| words.fill(nan_float));
                output.with_mut(|words| words.fill(nan_half));
                status.with_mut(|words| words.fill(0));
                let live = depth.div_ceil(256);
                let shape = [heads, kv_heads, dimension, capacity, depth, partitions]
                    .map(|value| u32::try_from(value).unwrap());
                let raw = gpu.command().unwrap();
                let bindings: [&[&objc2::runtime::ProtocolObject<dyn MTLBuffer>]; 2] = [
                    &[&query.raw, &key.raw, &value.raw, &partial.raw],
                    &[&partial.raw, &output.raw],
                ];
                encode_split(&raw, &pipelines, bindings, &shape, &status);
                raw.commit();
                raw.waitUntilCompleted();
                assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
                status.with_slice(|words| {
                    assert_eq!(words[0], u32::from(invalid));
                    assert!(words[1..].iter().all(|&word| word == 0));
                });
                partial.with_slice(|words| {
                    let packet = dimension + 2;
                    for head in 0..heads {
                        let unused =
                            (head * partitions + live) * packet..(head + 1) * partitions * packet;
                        assert!(words[unused].iter().all(|&word| word == nan_float));
                    }
                    assert!(words[workspace..].iter().all(|&word| word == nan_float));
                });
                output.with_slice(|words| {
                    assert!(words[active..].iter().all(|&word| word == nan_half));
                    for head in 0..heads {
                        let cache_head = head / (heads / kv_heads);
                        let row = |position| (cache_head * capacity + position) * dimension;
                        let q = &inputs[0][head * dimension..][..dimension];
                        let mut weights: Vec<f64> = (0..depth).map(|position| {
                            q.iter().zip(&inputs[1][row(position)..][..dimension])
                                .map(|(&q, &k)| half_value(q) * half_value(k)).sum::<f64>()
                                / (dimension as f64).sqrt()
                        }).collect();
                        let maximum = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        for weight in &mut weights {
                            *weight = (*weight - maximum).exp();
                        }
                        let mass: f64 = weights.iter().sum();
                        for feature in 0..dimension {
                            let actual = half_value(words[head * dimension + feature]);
                            if invalid && cache_head == 0 && feature == 0 {
                                assert!(actual.is_nan());
                                continue;
                            }
                            let expected = weights.iter().enumerate().map(|(position, weight)| {
                                weight * half_value(inputs[2][row(position) + feature])
                            }).sum::<f64>() / mass;
                            let error = (actual - expected).abs();
                            assert!(actual.is_finite() && error <= 1e-4 + 1e-3 * expected.abs(),
                                "depth {depth} pattern {pattern} head {head} feature {feature}: {actual} vs {expected}");
                            maximum_error = maximum_error.max(error);
                            compared += 1;
                        }
                    }
                });
                for (buffer, expected) in [&query, &key, &value].into_iter().zip(inputs) {
                    buffer.with_slice(|words| assert_eq!(words, expected));
                }
            }
        }
        assert_eq!(compared, 36 * active - heads / kv_heads);
        println!(
            "SPLIT_KV {{\"compared\":{compared},\"max_abs_error\":{maximum_error},\"workspace_bytes\":{}}}",
            workspace * size_of::<u32>()
        );
    });
}

#[test]
#[ignore = "explicit paired split-KV command timing on resident Mac fixtures"]
fn split_kv_decode_timing() {
    autoreleasepool(|_| {
        let gpu = Gpu::new().unwrap();
        let pipelines = split_pipelines(&gpu);
        let (heads, kv_heads, dimension, capacity) = (14usize, 2usize, 64usize, 1152usize);
        let partitions = capacity.div_ceil(256);
        let mut query = Buffer::<u16>::new(&gpu, heads * dimension).unwrap();
        let mut key = Buffer::<u16>::new(&gpu, kv_heads * capacity * dimension).unwrap();
        let mut value = Buffer::<u16>::new(&gpu, kv_heads * capacity * dimension).unwrap();
        for (buffer, seed) in [(&mut query, 3), (&mut key, 11), (&mut value, 37)] {
            buffer.with_mut(|words| {
                for (i, word) in words.iter_mut().enumerate() {
                    *word = f16::from_f32(((i * 17 + seed) % 101) as f32 / 32.0 - 1.5).to_bits();
                }
            });
        }
        let partial = Buffer::<u32>::new(&gpu, heads * partitions * (dimension + 2)).unwrap();
        let baseline = Buffer::<u16>::new(&gpu, heads * dimension).unwrap();
        let split = Buffer::<u16>::new(&gpu, heads * dimension).unwrap();
        let status = Buffer::<u32>::new(&gpu, 1).unwrap();
        let reverse = std::env::var_os("TOKLET_SPLIT_TIMING_REVERSE").is_some();
        for depth in [128usize, 1025, 1056, 1088, 1151] {
            let split_shape = [heads, kv_heads, dimension, capacity, depth, partitions]
                .map(|value| u32::try_from(value).unwrap());
            let online_shape = [1, heads, kv_heads, dimension, capacity, 1, depth - 1, depth]
                .map(|value| u32::try_from(value).unwrap());
            let run = |candidate| {
                let wall_start = std::time::Instant::now();
                let raw = gpu.command().unwrap();
                if candidate {
                    encode_split(
                        &raw,
                        &pipelines,
                        [
                            &[&query.raw, &key.raw, &value.raw, &partial.raw],
                            &[&partial.raw, &split.raw],
                        ],
                        &split_shape,
                        &status,
                    );
                } else {
                    gpu.encode(
                        &Command {
                            raw: &raw,
                            trace: None,
                        },
                        Kernel::AttentionDecode,
                        &[&query.raw, &key.raw, &value.raw, &baseline.raw],
                        &online_shape,
                        heads,
                        Some(&status),
                    )
                    .unwrap();
                }
                raw.commit();
                raw.waitUntilCompleted();
                let wall = wall_start.elapsed().as_secs_f64();
                assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
                assert_eq!(status.with_slice(|words| words[0]), 0);
                let gpu = raw.GPUEndTime() - raw.GPUStartTime();
                assert!(gpu.is_finite() && gpu > 0.0 && wall.is_finite() && wall > 0.0);
                [gpu, wall]
            };
            let mut samples = Vec::with_capacity(20);
            for trial in 0..13 {
                let split_first = (trial % 2 != 0) ^ reverse;
                for candidate in [split_first, !split_first] {
                    let [gpu_seconds, wall_seconds] = run(candidate);
                    if trial >= 3 {
                        samples.push(
                            serde_json::json!({ "trial":trial - 3, "candidate":candidate,
                            "gpu_seconds":gpu_seconds, "wall_seconds":wall_seconds }),
                        );
                    }
                }
                baseline.with_slice(|a| {
                    split.with_slice(|b| {
                        for (&base, &candidate) in a.iter().zip(b) {
                            let base = f64::from(f16::from_bits(base).to_f32());
                            let candidate = f64::from(f16::from_bits(candidate).to_f32());
                            assert!(
                                base.is_finite()
                                    && candidate.is_finite()
                                    && (base - candidate).abs() <= 1e-4 + 1e-3 * base.abs()
                            );
                        }
                    })
                });
            }
            println!(
                "SPLIT_TIMING {}",
                serde_json::json!({ "depth":depth, "reverse":reverse, "samples":samples })
            );
        }
    });
}

#[test]
#[ignore = "explicit Qwen2.5 layer-zero split-KV same-input Metal witness"]
fn split_kv_real_same_input_rounding_witness() {
    use std::path::Path;

    autoreleasepool(|_| {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("models/MLX-Qwen2.5-0.5B-Instruct-8bit-6474d4c");
        let (capacity, depth, heads, kv_heads, dimension) =
            (1152usize, 1025usize, 14usize, 2usize, 64usize);
        let budget = 2560 * 1024 * 1024;
        let model = Model::load_mlx_qwen2_audited(
            &path,
            weights::LoadOptions {
                context_limit: capacity,
                memory_limit_bytes: budget,
            },
        )
        .unwrap();
        let mut session = Session::with_attention_mode(
            &model,
            session::Options {
                capacity,
                position_offset: 0,
                prefill_chunk_size: 128,
                memory_limit_bytes: budget,
            },
            ProjectionMode::MlxAffine,
            NormalizationMode::Compensated,
            ReadbackMode::FullLogits,
            AttentionMode::Online,
        )
        .unwrap();
        let layers: Vec<_> = (0..model.config.num_layers).collect();
        session.enable_diagnostics(&layers).unwrap();
        let prompt: Vec<_> = include_bytes!("../../../benchmarks/contract-v1/prompt_1024.u32le")
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect();
        let continuation = include_bytes!("../../../benchmarks/contract-v1/continuation_128.u32le");
        session.set_tokens(&prompt).unwrap();
        session.evaluate().unwrap();
        session
            .append(&[u32::from_le_bytes(continuation[..4].try_into().unwrap())])
            .unwrap();
        assert_eq!(session.evaluate_next_chunk().unwrap(), 1);
        assert_eq!(session.evaluated_len(), depth);

        let snapshots = session.diagnostics().unwrap();
        let gpu = &model.gpu;
        let pipelines = split_pipelines(gpu);
        let partitions = capacity.div_ceil(256);
        let shape =
            [heads, kv_heads, dimension, capacity, depth, partitions].map(|word| word as u32);
        let online_shape = [
            1,
            heads,
            kv_heads,
            dimension,
            capacity,
            128,
            depth - 1,
            depth,
        ]
        .map(|word| word as u32);
        let mut query = Buffer::<u16>::new(gpu, heads * dimension).unwrap();
        let baseline = Buffer::<u16>::new(gpu, heads * dimension).unwrap();
        let split = Buffer::<u16>::new(gpu, heads * dimension).unwrap();
        let partial = Buffer::<u32>::new(gpu, heads * partitions * (dimension + 2)).unwrap();
        let status = Buffer::<u32>::new(gpu, 8).unwrap();
        let half = |bits| f64::from(f16::from_bits(bits).to_f32());

        for layer in layers {
            let snapshot = |name| {
                snapshots
                    .iter()
                    .find(|item| item.layer == Some(layer) && item.stage == name)
                    .unwrap()
            };
            let q = &snapshot("q_rotary").values;
            let expected_context = &snapshot("attention_context").values;
            query.with_mut(|bits| {
                for (slot, &value) in bits.iter_mut().zip(q) {
                    *slot = f16::from_f32(value).to_bits();
                }
            });
            let cache = &session.cache[layer];
            let raw = gpu.command().unwrap();
            gpu.encode(
                &Command {
                    raw: &raw,
                    trace: None,
                },
                Kernel::AttentionDecode,
                &[&query.raw, &cache.k.raw, &cache.v.raw, &baseline.raw],
                &online_shape,
                heads,
                Some(&status),
            )
            .unwrap();
            encode_split(
                &raw,
                &pipelines,
                [
                    &[&query.raw, &cache.k.raw, &cache.v.raw, &partial.raw],
                    &[&partial.raw, &split.raw],
                ],
                &shape,
                &status,
            );
            raw.commit();
            raw.waitUntilCompleted();
            assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
            assert!(status.with_slice(|bits| bits.iter().all(|&word| word == 0)));
            let base = baseline.with_slice(|bits| bits.to_vec());
            let result = split.with_slice(|bits| bits.to_vec());
            assert!(
                base.iter()
                    .zip(expected_context)
                    .all(|(&bits, &value)| half(bits) == f64::from(value))
            );
            let Some(index) = base.iter().zip(&result).position(|(a, b)| a != b) else {
                continue;
            };
            let head = index / dimension;
            let feature = index % dimension;
            let cache_head = head / (heads / kv_heads);
            let key = cache.k.with_slice(|bits| bits.to_vec());
            let value = cache.v.with_slice(|bits| bits.to_vec());
            let q = &q[head * dimension..][..dimension];
            let row = |position| (cache_head * capacity + position) * dimension;
            let scores: Vec<_> = (0..depth)
                .map(|position| {
                    q.iter()
                        .zip(&key[row(position)..][..dimension])
                        .map(|(&q, &k)| f64::from(q) * half(k))
                        .sum::<f64>()
                        / (dimension as f64).sqrt()
                })
                .collect();
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let (mass, numerator) =
                scores
                    .iter()
                    .enumerate()
                    .fold((0.0, 0.0), |(mass, sum), (position, score)| {
                        let weight = (score - maximum).exp();
                        (
                            mass + weight,
                            sum + weight * half(value[row(position) + feature]),
                        )
                    });
            let ideal = numerator / mass;
            let packets = partial.with_slice(|words| {
                (0..depth.div_ceil(256))
                    .map(|partition| {
                        let start = (head * partitions + partition) * (dimension + 2);
                        [words[start], words[start + 1], words[start + 2 + feature]]
                            .map(|x| f64::from(f32::from_bits(x)))
                    })
                    .collect::<Vec<_>>()
            });
            let packet_max = packets
                .iter()
                .map(|p| p[0])
                .fold(f64::NEG_INFINITY, f64::max);
            let (packet_mass, packet_sum) = packets.iter().fold((0.0, 0.0), |(mass, sum), p| {
                let scale = (p[0] - packet_max).exp();
                (mass + scale * p[1], sum + scale * p[2])
            });
            let baseline_word = half(base[index]);
            let split_word = half(result[index]);
            let midpoint = (baseline_word + split_word) / 2.0;
            let ideal_round = half(f16::from_f64(ideal).to_bits());
            let packet_round = half(f16::from_f64(packet_sum / packet_mass).to_bits());
            println!(
                "SPLIT_WITNESS {}",
                serde_json::json!({
                    "depth":depth,"first_different_layer":layer,"head":head,"feature":feature,
                    "baseline_half":baseline_word,"split_half":split_word,"midpoint":midpoint,
                    "fp64_oracle":ideal,"fp64_oracle_half":ideal_round,
                    "packet_fp64_merge":packet_sum/packet_mass,"packet_fp64_merge_half":packet_round,
                    "differing_words":base.iter().zip(&result).filter(|(a,b)|a!=b).count(),
                    "packets":packets
                })
            );
            return;
        }
        panic!("expected a split rounding difference on identical captured operands");
    });
}
