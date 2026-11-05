//! Bounded decode-only split-KV witness. No session or model policy selects it.
use super::*;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandEncoder, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLLibrary,
    MTLSize,
};

#[test]
#[ignore = "explicit Apple GPU split-KV, independent oracle and poisoned tails"]
fn split_kv_matches_fp64_at_tile_and_partition_boundaries() {
    autoreleasepool(|_| {
        let gpu = Gpu::new().unwrap();
        let library = mlx::precise_library(&gpu, include_str!("split_kv.metal")).unwrap();
        let pipelines = ["attention_split_partial", "attention_split_merge"].map(|name| {
            let function = library
                .newFunctionWithName(&NSString::from_str(name))
                .unwrap();
            gpu.device
                .newComputePipelineStateWithFunction_error(&function)
                .unwrap()
        });
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
                for ((pipeline, buffers), groups) in
                    pipelines.iter().zip(bindings).zip([heads * live, heads])
                {
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
                            std::ptr::NonNull::from(&shape).cast(),
                            size_of_val(&shape),
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
