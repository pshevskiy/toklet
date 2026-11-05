//! Explicit fixture for the isolated staged SIMD arithmetic.
use super::*;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandEncoder, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLLibrary,
    MTLSize,
};

const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const CHUNK: usize = 32;
const GUARD: usize = 7;
const UNWRITTEN: u32 = 0x7fc0_0123;
const POISON: u16 = 0x7e01;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

struct Fixture {
    q: Buffer<u16>,
    k: Buffer<u16>,
    v: Buffer<u16>,
    scores: Buffer<u32>,
    weights: Buffer<u32>,
    output: Buffer<u16>,
    status: Buffer<u32>,
    dimension: usize,
    capacity: usize,
}

impl Fixture {
    fn new(gpu: &Gpu, dimension: usize, capacity: usize, valid: usize) -> Self {
        assert!((1..=128).contains(&dimension) && valid <= capacity && capacity <= 512);
        let mut q = Buffer::new(gpu, CHUNK * HEADS * dimension + GUARD).unwrap();
        let mut k = Buffer::new(gpu, KV_HEADS * capacity * dimension + GUARD).unwrap();
        let mut v = Buffer::new(gpu, k.len).unwrap();
        let fill = |words: &mut [u16], seed| {
            for (index, word) in words.iter_mut().enumerate() {
                *word = f16::from_f32(((index * 19 + seed) % 103) as f32 / 32. - 1.5).to_bits();
            }
        };
        q.with_mut(|words| fill(&mut words[..CHUNK * HEADS * dimension], 3));
        for (buffer, seed) in [(&mut k, 17), (&mut v, 43)] {
            buffer.with_mut(|words| {
                words.fill(POISON);
                for head in 0..KV_HEADS {
                    let start = head * capacity * dimension;
                    fill(
                        &mut words[start..start + valid * dimension],
                        seed + head * 13,
                    );
                }
            });
        }
        let scratch_words = HEADS * CHUNK * capacity + GUARD;
        Self {
            q,
            k,
            v,
            scores: Buffer::new(gpu, scratch_words).unwrap(),
            weights: Buffer::new(gpu, scratch_words).unwrap(),
            output: Buffer::new(gpu, CHUNK * HEADS * dimension + GUARD).unwrap(),
            status: Buffer::new(gpu, GUARD + 1).unwrap(),
            dimension,
            capacity,
        }
    }

    fn run(&mut self, gpu: &Gpu, pipeline: &Pipeline, offset: usize, rows: usize) -> Vec<u16> {
        assert!((1..=CHUNK).contains(&rows) && offset + rows <= self.capacity);
        self.scores.with_mut(|words| words.fill(UNWRITTEN));
        self.weights.with_mut(|words| words.fill(UNWRITTEN));
        self.output.with_mut(|words| words.fill(POISON));
        self.status.with_mut(|words| {
            words.fill(UNWRITTEN);
            words[0] = 0;
        });
        let width = pipeline.threadExecutionWidth();
        assert!((32..=pipeline.maxTotalThreadsPerThreadgroup()).contains(&width));
        let shape = [
            rows,
            HEADS,
            KV_HEADS,
            self.dimension,
            self.capacity,
            CHUNK,
            offset,
        ]
        .map(|word| u32::try_from(word).unwrap());
        let raw = gpu.command().unwrap();
        let encoder = raw.computeCommandEncoder().unwrap();
        encoder.setComputePipelineState(pipeline);
        // SAFETY: The fixture bounds allocate every region in the shader ABI.
        // One SIMD group owns each active query/head, and buffers live through
        // completion. CPU reads follow the completed command.
        unsafe {
            for (slot, buffer) in [
                &self.q.raw,
                &self.k.raw,
                &self.v.raw,
                &self.scores.raw,
                &self.weights.raw,
                &self.output.raw,
            ]
            .into_iter()
            .enumerate()
            {
                encoder.setBuffer_offset_atIndex(Some(buffer), 0, slot);
            }
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::from(&shape).cast(),
                size_of_val(&shape),
                6,
            );
            encoder.setBuffer_offset_atIndex(Some(&self.status.raw), 0, 7);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: rows * HEADS,
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
        raw.commit();
        raw.waitUntilCompleted();
        assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
        self.status
            .with_slice(|words| assert!(words[1..].iter().all(|&x| x == UNWRITTEN)));
        for buffer in [&self.scores, &self.weights] {
            buffer.with_slice(|words| {
                for head in 0..HEADS {
                    for row in 0..CHUNK {
                        let start = (head * CHUNK + row) * self.capacity;
                        let active = if row < rows { offset + row + 1 } else { 0 };
                        assert!(
                            words[start + active..start + self.capacity]
                                .iter()
                                .all(|&x| x == UNWRITTEN)
                        );
                    }
                }
                assert!(words[words.len() - GUARD..].iter().all(|&x| x == UNWRITTEN));
            });
        }
        self.output.with_slice(|words| {
            let active = rows * HEADS * self.dimension;
            assert!(words[active..].iter().all(|&x| x == POISON));
            words[..active].to_vec()
        })
    }

    fn oracle(&self, offset: usize, rows: usize) -> Vec<f64> {
        let [q, k, v] = [&self.q, &self.k, &self.v].map(|buf| {
            buf.with_slice(|words| {
                words
                    .iter()
                    .map(|&x| f64::from(f16::from_bits(x).to_f32()))
                    .collect::<Vec<_>>()
            })
        });
        let d = self.dimension;
        let mut output = Vec::new();
        for row in 0..rows {
            for head in 0..HEADS {
                let kv = head / (HEADS / KV_HEADS);
                let start = (kv * self.capacity) * d;
                let query = &q[(row * HEADS + head) * d..][..d];
                let mut weights: Vec<_> = k[start..]
                    .chunks_exact(d)
                    .take(offset + row + 1)
                    .map(|key| {
                        key.iter().zip(query).map(|(a, b)| a * b).sum::<f64>() / (d as f64).sqrt()
                    })
                    .collect();
                let maximum = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                for weight in &mut weights {
                    *weight = (*weight - maximum).exp();
                }
                let mass = weights.iter().sum::<f64>();
                output.extend((0..d).map(|feature| {
                    weights
                        .iter()
                        .enumerate()
                        .map(|(pos, weight)| weight * v[start + pos * d + feature])
                        .sum::<f64>()
                        / mass
                }));
            }
        }
        output
    }
}

#[test]
#[ignore = "explicit Apple GPU staged SIMD workspace and causal schedule controls"]
fn staged_simd_guards_and_schedules() {
    autoreleasepool(|_| {
        let gpu = Gpu::new().unwrap();
        let library = mlx::precise_library(
            &gpu,
            concat!(
                include_str!("../kernels.metal"),
                "\n",
                include_str!("staged.metal")
            ),
        )
        .unwrap();
        let function = library
            .newFunctionWithName(&NSString::from_str("attention_staged_simd"))
            .unwrap();
        let pipeline = gpu
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .unwrap();
        let mut comparisons = 0;
        let mut max_error = 0.0f64;
        for dimension in [9, 64, 127] {
            for depth in [1usize, 17, 32, 33, 129, 512] {
                let rows = depth.min(CHUNK);
                let offset = depth - rows;
                let mut fixture = Fixture::new(&gpu, dimension, (depth + 3).min(512), depth);
                let expected = fixture.oracle(offset, rows);
                let batch = fixture.run(&gpu, &pipeline, offset, rows);
                assert_eq!(fixture.status.with_slice(|words| words[0]), 0);
                let q = fixture.q.with_slice(|words| words.to_vec());
                let row_width = HEADS * dimension;
                for row in 0..rows {
                    fixture.q.with_mut(|words| {
                        words[..row_width].copy_from_slice(&q[row * row_width..][..row_width])
                    });
                    let single = fixture.run(&gpu, &pipeline, offset + row, 1);
                    assert_eq!(fixture.status.with_slice(|words| words[0]), 0);
                    assert_eq!(single, batch[row * row_width..][..row_width]);
                }
                fixture.q.with_mut(|words| words.copy_from_slice(&q));
                assert_eq!(fixture.run(&gpu, &pipeline, offset, rows), batch);
                for (&word, expected) in batch.iter().zip(expected) {
                    let actual = f64::from(f16::from_bits(word).to_f32());
                    let delta = (actual - expected).abs();
                    assert!(actual.is_finite() && delta <= 1e-4 + 1e-3 * expected.abs());
                    max_error = max_error.max(delta);
                    comparisons += 1;
                }
            }
        }
        let mut fault = Fixture::new(&gpu, 9, 4, 1);
        fault.k.with_mut(|words| words[0] = POISON);
        fault.run(&gpu, &pipeline, 0, 1);
        assert_ne!(fault.status.with_slice(|words| words[0]) & 2, 0);
        fault.k.with_mut(|words| words[0] = 0);
        fault.v.with_mut(|words| words[0] = POISON);
        fault.run(&gpu, &pipeline, 0, 1);
        assert_eq!(fault.status.with_slice(|words| words[0]), 1);
        println!(
            "STAGED_LEAF {{\"comparisons\":{comparisons},\"max_abs_error\":{max_error},\"cases\":18}}"
        );
    });
}
