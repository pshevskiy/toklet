//! Explicit scalar-reduction/SIMD-store Half RMS experiment.
use super::{Buffer, Command, Error, Gpu, Result, mlx};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::*;
use std::ptr::NonNull;

pub(super) struct Pipeline {
    state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    width: usize,
}
impl Pipeline {
    pub(super) fn new(gpu: &Gpu) -> Result<Self> {
        let library = mlx::precise_library(
            gpu,
            concat!(
                include_str!("kernels.metal"),
                "\n",
                include_str!("rms_store.metal")
            ),
        )?;
        let function = library
            .newFunctionWithName(&NSString::from_str("rms_parallel_store"))
            .ok_or_else(|| Error::Native("RMS store kernel absent".into()))?;
        let state = gpu
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| Error::Native(error.to_string()))?;
        let width = state.threadExecutionWidth();
        if width == 0 || width > state.maxTotalThreadsPerThreadgroup() {
            return Err(Error::Unsupported("RMS store SIMD width"));
        }
        Ok(Self { state, width })
    }
    pub(super) fn encode(
        &self,
        command: &Command<'_>,
        x: &Buffer<u16>,
        w: &Buffer<u16>,
        output: &Buffer<u16>,
        shape: [u32; 4],
        status: &Buffer<u32>,
    ) -> Result<()> {
        let [rows, columns, epsilon, compensated] = shape;
        let elements = (rows as usize)
            .checked_mul(columns as usize)
            .filter(|n| *n > 0 && *n <= u32::MAX as usize)
            .ok_or(Error::Options("RMS store extent overflow"))?;
        let eps = f32::from_bits(epsilon);
        if rows > 32
            || columns > 4096
            || !eps.is_finite()
            || eps <= 0.
            || compensated > 1
            || x.len < elements
            || output.len < elements
            || w.len < columns as usize
            || status.len == 0
        {
            return Err(Error::Options(
                "RMS store shape/buffer/epsilon outside pilot bounds",
            ));
        }
        let (index, encoder) = if let Some(trace) = command.trace {
            let (index, encoder) = trace.encoder(command.raw, "rms_parallel_store")?;
            (Some(index), encoder)
        } else {
            (
                None,
                command
                    .computeCommandEncoder()
                    .ok_or_else(|| Error::Native("RMS store encoder absent".into()))?,
            )
        };
        encoder.setComputePipelineState(&self.state);
        // SAFETY: Private checked ABI: contiguous Half arrays and UInt32 shape,
        // same-device tracked buffers retained through completion. Weights are a
        // distinct immutable owner. One SIMD group
        // writes each disjoint row. In-place x/output is safe: lane0 finishes all
        // reduction reads before broadcasting inv; each lane then reads/writes
        // only its own columns. CPU access occurs only after GPU completion.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&x.raw), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&w.raw), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&output.raw), 0, 2);
            encoder.setBytes_length_atIndex(NonNull::from(&shape).cast(), size_of_val(&shape), 3);
            encoder.setBuffer_offset_atIndex(Some(&status.raw), 0, 4);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: rows as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.width,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        if let (Some(trace), Some(index)) = (command.trace, index) {
            trace.finish_encoder(index);
        }
        Ok(())
    }
}

impl super::Session<'_> {
    /// Explicit Apple Silicon Mac experiment for the bounded MiniCPM5 geometry.
    /// Arithmetic remains Compensated; no ordinary factory selects this kernel.
    pub fn enable_rms_parallel_store(&mut self) -> Result<()> {
        let c = &self.model.config;
        if !cfg!(all(target_os = "macos", target_arch = "aarch64"))
            || !self.model.is_mlx()
            || self.model.moe.is_some()
            || self.model.qwen3_context_diagnostic
            || (
                c.hidden_size,
                c.intermediate_size,
                c.num_layers,
                c.query_heads,
                c.kv_heads,
                c.head_dim,
                c.vocab_size,
            ) != (2048, 6144, 42, 16, 2, 128, 130560)
            || self.model.output.is_none()
            || self.model.layers.iter().any(|l| l.qk_norms.is_some())
            || self.normalization_mode != super::NormalizationMode::Compensated
            || self.projection_mode != super::ProjectionMode::MlxAffine
            || self.attention_mode != super::AttentionMode::Reference
            || self.options.capacity > 512
            || self.options.prefill_chunk_size > 32
            || self.parallel_reference.is_some()
            || self.affine_tile4.is_some()
            || self.rms_parallel_store.is_some()
            || !self.history.is_empty()
            || self.evaluated != 0
        {
            return Err(Error::Options(
                "RMS parallel stores require a fresh bounded MiniCPM5 Mac session",
            ));
        }
        let admitted = super::add(self.memory.admitted_bytes, super::ALLOWANCE)?;
        if admitted > self.options.memory_limit_bytes {
            return Err(Error::Budget {
                required: admitted,
                limit: self.options.memory_limit_bytes,
            });
        }
        let allowance = super::add(self.memory.metadata_allowance_bytes, super::ALLOWANCE)?;
        let pipeline = objc2::rc::autoreleasepool(|_| self.model.gpu.rms_parallel_store())?;
        self.rms_parallel_store = Some(pipeline);
        self.memory.metadata_allowance_bytes = allowance;
        self.memory.admitted_bytes = admitted;
        Ok(())
    }
    /// Separately report the implementation choice; normalization math is unchanged.
    pub fn rms_parallel_store_enabled(&self) -> bool {
        self.rms_parallel_store.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::{NormalizationMode, rms};
    use half::f16;
    use objc2::rc::autoreleasepool;
    use serde_json::json;

    fn finish(raw: &ProtocolObject<dyn MTLCommandBuffer>) {
        raw.commit();
        raw.waitUntilCompleted();
        assert_eq!(raw.status(), MTLCommandBufferStatus::Completed);
        assert!(raw.error().is_none());
    }
    #[test]
    #[ignore = "Explicit Apple GPU RMS serial-reduction/SIMD-store oracle and guards"]
    fn parallel_store_exact_shapes_status_and_analytical_rows() {
        autoreleasepool(|_| {
            let gpu = Gpu::new().unwrap();
            let pipeline = Pipeline::new(&gpu).unwrap();
            let mut cases = 0;
            let mut values = 0;
            for width in [1, 3, 31, 32, 33, 127, 128, 129, 1536, 2047, 2048, 2049] {
                for rows in [1, 3, 4, 17, 32] {
                    for uniform in [false, true] {
                        let mut x = Buffer::<u16>::new(&gpu, rows * width + 7).unwrap();
                        let mut weights = Buffer::<u16>::new(&gpu, width + 7).unwrap();
                        let mut out = Buffer::<u16>::new(&gpu, rows * width + 7).unwrap();
                        let mut status = Buffer::<u32>::new(&gpu, 8).unwrap();
                        x.with_mut(|b| {
                            b.fill(0x3555);
                            for (i, v) in b[..rows * width].iter_mut().enumerate() {
                                *v = f16::from_f32(if uniform {
                                    1.
                                } else {
                                    ((i * 17 + i / width) % 127) as f32 / 32. - 2.
                                })
                                .to_bits();
                            }
                        });
                        weights.with_mut(|b| {
                            b.fill(0x3555);
                            for (i, v) in b[..width].iter_mut().enumerate() {
                                *v = f16::from_f32([1., 0.5, -0.25][i % 3]).to_bits();
                            }
                        });
                        for mode in [NormalizationMode::Original, NormalizationMode::Compensated] {
                            let mut expected = None;
                            for candidate in [false, true] {
                                out.with_mut(|b| b.fill(0x3555));
                                status.with_mut(|b| b.fill(4));
                                let raw = gpu.command().unwrap();
                                let command = Command {
                                    raw: &raw,
                                    trace: None,
                                };
                                if candidate {
                                    pipeline
                                        .encode(
                                            &command,
                                            &x,
                                            &weights,
                                            &out,
                                            [
                                                rows as u32,
                                                width as u32,
                                                1e-6f32.to_bits(),
                                                u32::from(mode == NormalizationMode::Compensated),
                                            ],
                                            &status,
                                        )
                                        .unwrap();
                                } else {
                                    rms(
                                        &gpu, &command, &x, &weights, &out, rows, width, 1e-6,
                                        mode, &status,
                                    )
                                    .unwrap();
                                }
                                finish(&raw);
                                assert_eq!(status.with_slice(|b| b[0]), 4);
                                let actual = out.with_slice(|b| b.to_vec());
                                assert!(actual[rows * width..].iter().all(|v| *v == 0x3555));
                                if uniform {
                                    for (i, v) in actual[..rows * width].iter().enumerate() {
                                        assert_eq!(*v, weights.with_slice(|b| b[i % width]));
                                    }
                                }
                                if let Some(ref expected) = expected {
                                    assert_eq!(&actual, expected);
                                    cases += 1;
                                    values += rows * width;
                                } else {
                                    expected = Some(actual);
                                }
                            }
                        }
                    }
                }
            }
            let mut inplace = Buffer::<u16>::new(&gpu, 2055).unwrap();
            let mut weights = Buffer::<u16>::new(&gpu, 2055).unwrap();
            let mut expected = Buffer::<u16>::new(&gpu, 2055).unwrap();
            let mut flags = Buffer::<u32>::new(&gpu, 8).unwrap();
            inplace.with_mut(|b| {
                b.fill(0x3555);
                for (i, v) in b[..2048].iter_mut().enumerate() {
                    *v = f16::from_f32((i % 19) as f32 / 8. - 1.).to_bits();
                }
            });
            weights.with_mut(|b| b.fill(f16::ONE.to_bits()));
            expected.with_mut(|b| b.fill(0x3555));
            flags.with_mut(|b| b.fill(0));
            let raw = gpu.command().unwrap();
            let command = Command {
                raw: &raw,
                trace: None,
            };
            rms(
                &gpu,
                &command,
                &inplace,
                &weights,
                &expected,
                1,
                2048,
                1e-6,
                NormalizationMode::Compensated,
                &flags,
            )
            .unwrap();
            finish(&raw);
            let raw = gpu.command().unwrap();
            let command = Command {
                raw: &raw,
                trace: None,
            };
            pipeline
                .encode(
                    &command,
                    &inplace,
                    &weights,
                    &inplace,
                    [1, 2048, 1e-6f32.to_bits(), 1],
                    &flags,
                )
                .unwrap();
            finish(&raw);
            assert_eq!(
                inplace.with_slice(|b| b.to_vec()),
                expected.with_slice(|b| b.to_vec())
            );
            assert_eq!(flags.with_slice(|b| b[0]), 0);
            // An output overflow must retain pre-existing flags and guard words.
            let mut x = Buffer::<u16>::new(&gpu, 11).unwrap();
            let mut w = Buffer::<u16>::new(&gpu, 11).unwrap();
            let mut out = Buffer::<u16>::new(&gpu, 11).unwrap();
            let mut status = Buffer::<u32>::new(&gpu, 8).unwrap();
            x.with_mut(|b| b[..4].copy_from_slice(&[f16::MAX.to_bits(), 0, 0, 0]));
            w.with_mut(|b| b.fill(f16::MAX.to_bits()));
            let mut expected = None;
            for candidate in [false, true] {
                out.with_mut(|b| b.fill(0x3555));
                status.with_mut(|b| b.fill(4));
                let raw = gpu.command().unwrap();
                let command = Command {
                    raw: &raw,
                    trace: None,
                };
                if candidate {
                    pipeline
                        .encode(
                            &command,
                            &x,
                            &w,
                            &out,
                            [1, 4, 1e-6f32.to_bits(), 1],
                            &status,
                        )
                        .unwrap();
                } else {
                    rms(
                        &gpu,
                        &command,
                        &x,
                        &w,
                        &out,
                        1,
                        4,
                        1e-6,
                        NormalizationMode::Compensated,
                        &status,
                    )
                    .unwrap();
                }
                finish(&raw);
                assert_eq!(status.with_slice(|b| b[0]), 5);
                let actual = out.with_slice(|b| b.to_vec());
                if let Some(ref expected) = expected {
                    assert_eq!(&actual, expected);
                } else {
                    expected = Some(actual);
                }
            }
            let raw = gpu.command().unwrap();
            let command = Command {
                raw: &raw,
                trace: None,
            };
            for shape in [
                [0, 4, 1e-6f32.to_bits(), 1],
                [1, 0, 1e-6f32.to_bits(), 1],
                [33, 4, 1e-6f32.to_bits(), 1],
                [1, 4097, 1e-6f32.to_bits(), 1],
                [1, 4, f32::NAN.to_bits(), 1],
                [1, 4, 0, 1],
                [1, 4, 1e-6f32.to_bits(), 2],
                [1, 12, 1e-6f32.to_bits(), 1],
            ] {
                assert!(
                    pipeline
                        .encode(&command, &x, &w, &out, shape, &status)
                        .is_err()
                );
            }
            println!(
                "{}",
                json!({"event":"rms_store_gpu_parity","cases":cases,"finite_output_values":values,"uniform_analytical":true,"overflow_flags":true,"tail_guards":true,"invalid_controls":8,"inplace_exact":true})
            );
        });
    }
    #[test]
    #[ignore = "Explicit tiny MLX RMS graph parity, dispatch, state/replay and selection"]
    fn parallel_store_tiny_graph_replay_and_dispatch() {
        use crate::{
            metal::{
                AttentionMode, Model, NormalizationMode, ProjectionMode, ReadbackMode, Session,
            },
            session, weights,
        };
        autoreleasepool(|_| {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/mlx-qwen-v1/model");
            let model = Model::load_mlx(
                &path,
                weights::LoadOptions {
                    context_limit: 32,
                    ..Default::default()
                },
            )
            .unwrap();
            let options = session::Options {
                capacity: 32,
                prefill_chunk_size: 4,
                position_offset: 0,
                memory_limit_bytes: weights::INITIAL_MEMORY_LIMIT_BYTES,
            };
            let create = || {
                Session::with_attention_mode(
                    &model,
                    options,
                    ProjectionMode::MlxAffine,
                    NormalizationMode::Compensated,
                    ReadbackMode::FullLogits,
                    AttentionMode::Reference,
                )
                .unwrap()
            };
            let mut serial = create();
            let mut parallel = create();
            let before = serde_json::to_value(parallel.memory()).unwrap();
            assert!(!parallel.rms_parallel_store_enabled());
            assert!(parallel.enable_rms_parallel_store().is_err());
            assert_eq!(serde_json::to_value(parallel.memory()).unwrap(), before);
            // Test-only injection qualifies the mechanism on the independent tiny
            // graph. The public selector correctly rejects this non-MiniCPM model.
            parallel.memory.metadata_allowance_bytes += super::super::ALLOWANCE;
            parallel.memory.admitted_bytes += super::super::ALLOWANCE;
            assert!(parallel.memory.admitted_bytes <= options.memory_limit_bytes);
            parallel.rms_parallel_store = Some(model.gpu.rms_parallel_store().unwrap());
            assert!(parallel.native_owned_memory().is_err());
            serial.enable_diagnostics(&[0, 1]).unwrap();
            parallel.enable_diagnostics(&[0, 1]).unwrap();
            parallel.enable_profile().unwrap();
            let compare = |a: &Session<'_>, b: &Session<'_>| {
                assert_eq!(a.history(), b.history());
                assert_eq!(a.evaluated_len(), b.evaluated_len());
                assert_eq!(a.pending_len(), b.pending_len());
                let left = a.diagnostics().unwrap();
                let right = b.diagnostics().unwrap();
                assert_eq!(left.len(), right.len());
                for (a, b) in left.iter().zip(&right) {
                    assert_eq!(
                        (a.layer, a.stage, a.rows, a.columns),
                        (b.layer, b.stage, b.rows, b.columns)
                    );
                    assert_eq!(
                        a.values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        b.values.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                    );
                }
                assert_eq!(
                    a.greedy_token().unwrap().unwrap(),
                    b.greedy_token().unwrap().unwrap()
                );
                let profile = b.profile().unwrap();
                assert_eq!(
                    profile
                        .events
                        .iter()
                        .filter(|e| e.name == "rms_parallel_store")
                        .count(),
                    5
                );
                assert!(profile.events.iter().all(|e| e.name != "rms"));
                assert_eq!(b.timing().unwrap().command_count, 1);
            };
            serial.set_tokens(&[0, 7, 13, 3]).unwrap();
            parallel.set_tokens(&[0, 7, 13, 3]).unwrap();
            serial.evaluate().unwrap();
            parallel.evaluate().unwrap();
            compare(&serial, &parallel);
            serial.append(&[9]).unwrap();
            parallel.append(&[9]).unwrap();
            serial.evaluate().unwrap();
            parallel.evaluate().unwrap();
            compare(&serial, &parallel);
            assert!(
                parallel
                    .finish_chunk(1, Err(super::super::Error::Completion))
                    .is_err()
            );
            assert_eq!(parallel.evaluated_len(), 0);
            assert!(parallel.profile().is_err());
            serial.reset();
            parallel.reset();
            assert!(parallel.rms_parallel_store_enabled());
            assert_eq!(parallel.pending_len(), 0);
            serial.set_tokens(&[0, 7, 13]).unwrap();
            parallel.set_tokens(&[0, 7, 13]).unwrap();
            serial.evaluate().unwrap();
            parallel.evaluate().unwrap();
            compare(&serial, &parallel);
            println!(
                "{}",
                json!({"event":"rms_store_tiny_graph","all_stages_exact":true,"dispatches_per_chunk":5,"default_unselected":true,"public_shape_rejection":true,"reset_replay":true})
            );
        });
    }
    #[test]
    #[ignore = "Explicit M1 release RMS completed-GPU screening; no engine speed claim"]
    fn parallel_store_screen_completed_gpu() {
        autoreleasepool(|_| {
            let gpu = Gpu::new().unwrap();
            let pipeline = Pipeline::new(&gpu).unwrap();
            for run in [1, 2] {
                for rows in [1, 4, 17, 32] {
                    let mut x = Buffer::<u16>::new(&gpu, rows * 2048 + 7).unwrap();
                    let mut w = Buffer::<u16>::new(&gpu, 2055).unwrap();
                    let out = Buffer::<u16>::new(&gpu, rows * 2048 + 7).unwrap();
                    let mut status = Buffer::<u32>::new(&gpu, 8).unwrap();
                    x.with_mut(|b| {
                        for (i, v) in b.iter_mut().enumerate() {
                            *v = f16::from_f32((i % 31) as f32 / 16. - 1.).to_bits();
                        }
                    });
                    w.with_mut(|b| b.fill(f16::ONE.to_bits()));
                    for sample in 0..13 {
                        let first = (run == 2) ^ (sample % 2 == 1);
                        for candidate in [first, !first] {
                            status.with_mut(|b| b.fill(0));
                            let raw = gpu.command().unwrap();
                            let command = Command {
                                raw: &raw,
                                trace: None,
                            };
                            for _ in 0..8 {
                                if candidate {
                                    pipeline
                                        .encode(
                                            &command,
                                            &x,
                                            &w,
                                            &out,
                                            [rows as u32, 2048, 1e-6f32.to_bits(), 1],
                                            &status,
                                        )
                                        .unwrap();
                                } else {
                                    rms(
                                        &gpu,
                                        &command,
                                        &x,
                                        &w,
                                        &out,
                                        rows,
                                        2048,
                                        1e-6,
                                        NormalizationMode::Compensated,
                                        &status,
                                    )
                                    .unwrap();
                                }
                            }
                            finish(&raw);
                            assert_eq!(status.with_slice(|b| b[0]), 0);
                            let elapsed = raw.GPUEndTime() - raw.GPUStartTime();
                            assert!(elapsed.is_finite() && elapsed > 0.);
                            println!(
                                "{}",
                                json!({"event":"rms_store_micro","run":run,"rows":rows,"width":2048,"sample":sample,"warmup":sample<3,"candidate":candidate,"encoders":8,"completed_gpu_seconds":elapsed,"end_to_end_qualified":false})
                            );
                        }
                    }
                }
            }
        });
    }
}
