//! Explicit private scheduling variant; exact qualified guard2 arithmetic.
//! Caller initializes disjoint contexts/header and PENDING, retains all borrows,
//! checks terminal receipts and owns whole-transaction replay/publication.
use super::super::*;

pub(in crate::metal) const PENDING: u32 = 4;
pub(in crate::metal) const STRUCTURAL: u32 = 8;
pub(in crate::metal) const MAX_EPOCH: u32 = (1 << 30) - 1;
pub(in crate::metal) const VERSION: u32 = 0x5250_0000;

pub(in crate::metal) fn epoch(value: u64) -> Result<u32> {
    let value = u32::try_from(value)
        .map_err(|_| Error::Options("paired RMS epoch exhausted; recreate session"))?;
    if !(1..=MAX_EPOCH).contains(&value) {
        return Err(Error::Options(
            "paired RMS epoch exhausted; recreate session",
        ));
    }
    Ok(value)
}
#[derive(Clone, Copy)]
pub(in crate::metal) struct Shape {
    pub rows: usize,
    pub width: usize,
    pub learned: bool,
    pub epoch: u32,
    pub op_tag: u32,
}
#[derive(Clone, Copy)]
pub(in crate::metal) struct PairRanges<'a> {
    pub input: BufferRange<'a>,
    pub weights: BufferRange<'a>,
    pub output: BufferRange<'a>,
    pub status: BufferRange<'a>,
    pub reasons: BufferRange<'a>,
    pub context: BufferRange<'a>,
    pub header: BufferRange<'a>,
}
#[derive(Clone, Copy)]
pub(in crate::metal) struct Slot<'a> {
    pub context: BufferRange<'a>,
    pub header: BufferRange<'a>,
    pub op_tag: u32,
}
pub(in crate::metal) struct Pipelines {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    a: [Retained<ProtocolObject<dyn MTLComputePipelineState>>; 2],
    b: [Retained<ProtocolObject<dyn MTLComputePipelineState>>; 2],
    widths: [usize; 2],
}
impl Pipelines {
    pub(in crate::metal) fn new(
        gpu: &Gpu,
        original: &super::super::Pipelines,
        available: usize,
    ) -> Result<Self> {
        if available < super::POLICY_BYTES {
            return Err(Error::Budget {
                required: super::POLICY_BYTES,
                limit: available,
            });
        }
        if !std::ptr::eq(&*original.device, &*gpu.device) {
            return Err(Error::Options("paired RMS original device differs"));
        }
        autoreleasepool(|_| {
            let options = MTLCompileOptions::new();
            options.setLanguageVersion(MTLLanguageVersion::Version2_3);
            options.setMathMode(MTLMathMode::Safe);
            options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
            if options.languageVersion() != MTLLanguageVersion::Version2_3
                || options.mathMode() != MTLMathMode::Safe
                || options.mathFloatingPointFunctions() != MTLMathFloatingPointFunctions::Precise
            {
                return Err(Error::Unsupported("paired RMS precise math unavailable"));
            }
            let library = gpu
                .device
                .newLibraryWithSource_options_error(
                    &NSString::from_str(concat!(
                        include_str!("../precision.metal"),
                        include_str!("two_phase.metal")
                    )),
                    Some(&options),
                )
                .map_err(|e| Error::Native(e.to_string()))?;
            let pipeline = |name| {
                let function = library
                    .newFunctionWithName(&NSString::from_str(name))
                    .ok_or_else(|| Error::Native("paired RMS entry absent".into()))?;
                gpu.device
                    .newComputePipelineStateWithFunction_error(&function)
                    .map_err(|e| Error::Native(e.to_string()))
            };
            let a = [
                pipeline("gemma_rms_phase_a_unlearned")?,
                pipeline("gemma_rms_phase_a_learned")?,
            ];
            let b = [
                pipeline("gemma_rms_phase_b_unlearned")?,
                pipeline("gemma_rms_phase_b_learned")?,
            ];
            let widths = a.each_ref().map(|p| p.threadExecutionWidth());
            if widths != original.widths || widths != b.each_ref().map(|p| p.threadExecutionWidth())
            {
                return Err(Error::Unsupported("paired RMS original SIMD widths differ"));
            }
            for (index, &width) in widths.iter().enumerate() {
                if width == 0
                    || width > u16::MAX as usize
                    || width > a[index].maxTotalThreadsPerThreadgroup()
                    || width > b[index].maxTotalThreadsPerThreadgroup()
                {
                    return Err(Error::Unsupported("paired RMS launch width unavailable"));
                }
            }
            Ok(Self {
                device: gpu.device.clone(),
                a,
                b,
                widths,
            })
        })
    }
    pub(in crate::metal) fn groups(&self, width: usize, learned: bool) -> Result<usize> {
        if !(1..=1536).contains(&width) {
            return Err(Error::Options("paired RMS width differs"));
        }
        Ok(width.div_ceil(self.widths[usize::from(learned)]))
    }
    fn geometry(&self, shape: Shape) -> Result<(Geometry, [u32; 5], usize, usize)> {
        epoch(u64::from(shape.epoch))?;
        let g = Geometry::new(shape.rows, shape.width, EPSILON, shape.learned)?;
        let variant = usize::from(shape.learned);
        let words = [
            g.words[0],
            g.words[1],
            g.words[2],
            shape.epoch,
            shape.op_tag,
        ];
        Ok((g, words, variant, self.groups(shape.width, shape.learned)?))
    }
    pub(in crate::metal) fn preflight(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        r: &PairRanges<'_>,
        shape: Shape,
    ) -> Result<()> {
        let (g, _, _, _) = self.geometry(shape)?;
        if r.input.len != g.values
            || r.weights.len != g.weights
            || r.output.len != g.values
            || r.status.len != 1
            || r.reasons.len != shape.rows
            || r.context.len != shape.rows * 8
            || r.header.len != 4
        {
            return Err(Error::Options("paired RMS borrowed extent differs"));
        }
        validate_encode_ranges(
            command,
            &self.device,
            &[
                r.input, r.weights, r.output, r.status, r.reasons, r.context, r.header,
            ],
            2,
        )
    }
    pub(in crate::metal) fn encode(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        r: &PairRanges<'_>,
        shape: Shape,
    ) -> Result<()> {
        self.encode_inner(command, r, shape, false)
    }
    /// Bounded fault control only. It encodes real A, then returns the original
    /// error without B. The transaction must drain, mark count unknown and poison.
    pub(in crate::metal) fn encode_partial(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        r: &PairRanges<'_>,
        shape: Shape,
    ) -> Result<()> {
        self.encode_inner(command, r, shape, true)
    }
    fn encode_inner(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        r: &PairRanges<'_>,
        shape: Shape,
        fail_after_a: bool,
    ) -> Result<()> {
        self.preflight(command, r, shape)?;
        let (_, words, variant, groups) = self.geometry(shape)?;
        let a = command
            .computeCommandEncoder()
            .ok_or_else(|| Error::Native("paired RMS A encoder absent".into()))?;
        a.setComputePipelineState(&self.a[variant]);
        // SAFETY: whole-pair preflight proved exact checked same-device tracked
        // shared extents, disjoint writes, retained NotEnqueued command. Shape
        // bytes copy immediately; caller excludes CPU views to completion.
        unsafe {
            for (slot, b) in [r.input, r.weights, r.context].iter().enumerate() {
                a.setBuffer_offset_atIndex(Some(b.raw), b.offset * 4, slot);
            }
            a.setBytes_length_atIndex(NonNull::from(&words).cast(), size_of::<[u32; 5]>(), 3);
            for (slot, b) in [r.status, r.reasons, r.header].iter().enumerate() {
                a.setBuffer_offset_atIndex(Some(b.raw), b.offset * 4, slot + 4);
            }
        }
        a.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: shape.rows,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.widths[variant],
                height: 1,
                depth: 1,
            },
        );
        a.endEncoding();
        if fail_after_a {
            return Err(Error::Options(
                "paired RMS synthetic B encoder failure after A",
            ));
        }
        let b = command
            .computeCommandEncoder()
            .ok_or_else(|| Error::Native("paired RMS B encoder absent".into()))?;
        b.setComputePipelineState(&self.b[variant]);
        // SAFETY: same preflight/lifetimes as A; the tracked encoder boundary
        // orders A writes before B reads without an intermediate CPU wait.
        unsafe {
            for (slot, range) in [r.input, r.weights, r.output].iter().enumerate() {
                b.setBuffer_offset_atIndex(Some(range.raw), range.offset * 4, slot);
            }
            b.setBytes_length_atIndex(NonNull::from(&words).cast(), size_of::<[u32; 5]>(), 3);
            for (slot, range) in [r.status, r.reasons, r.context, r.header]
                .iter()
                .enumerate()
            {
                b.setBuffer_offset_atIndex(Some(range.raw), range.offset * 4, slot + 4);
            }
        }
        b.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups,
                height: shape.rows,
                depth: 1,
            },
            MTLSize {
                width: self.widths[variant],
                height: 1,
                depth: 1,
            },
        );
        b.endEncoding();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn paired_epoch_never_truncates_or_resets_the_session_counter() {
        assert_eq!(epoch(1).unwrap(), 1);
        assert_eq!(epoch(u64::from(MAX_EPOCH)).unwrap(), MAX_EPOCH);
        for bad in [
            0,
            u64::from(MAX_EPOCH) + 1,
            1u64 << 32,
            (1u64 << 32) + 1,
            u64::MAX,
        ] {
            assert!(epoch(bad).is_err());
        }
        assert_eq!(VERSION & 0x000f_ffff, 0);
        assert_eq!(PENDING | STRUCTURAL, 12);
    }

    #[test]
    #[ignore = "explicit Rust borrowed paired RMS offset/alias/guard/one-wait ABI qualification"]
    fn paired_rust_encoder_matches_guard2_with_offsets_and_readonly_alias() {
        autoreleasepool(|_| {
            let gpu = Gpu::new().unwrap();
            let original = super::super::super::Pipelines::new(&gpu).unwrap();
            let reference =
                super::super::Pipelines::new(&gpu, &original, super::super::POLICY_BYTES).unwrap();
            let pipeline = Pipelines::new(&gpu, &original, super::super::POLICY_BYTES).unwrap();
            fn owned(gpu: &Gpu, words: usize) -> Buffer<u32> {
                let mut b = Buffer::new(gpu, words + 14).unwrap();
                b.with_mut(|s| s.fill(SENTINEL));
                b
            }
            let mut input = owned(&gpu, 14);
            input.with_mut(|s| s[7..21].fill(1f32.to_bits()));
            let output = owned(&gpu, 14);
            let mut status = owned(&gpu, 1);
            status.with_mut(|s| s[7] = PENDING);
            let reasons = owned(&gpu, 2);
            let mut context = owned(&gpu, 24);
            context.with_mut(|s| {
                s[11] = 0;
                s[19] = 0;
            });
            let mut header = owned(&gpu, 4);
            header.with_mut(|s| {
                s[7..11].copy_from_slice(&[1, 77, 2 * pipeline.groups(7, true).unwrap() as u32, 0])
            });
            let reference_output = owned(&gpu, 14);
            let mut reference_status = owned(&gpu, 1);
            reference_status.with_mut(|s| s[7] = 0);
            let reference_reasons = owned(&gpu, 2);
            let ranges = PairRanges {
                input: BufferRange::new(&input, 7, 14).unwrap(),
                weights: BufferRange::new(&input, 7, 7).unwrap(),
                output: BufferRange::new(&output, 7, 14).unwrap(),
                status: BufferRange::new(&status, 7, 1).unwrap(),
                reasons: BufferRange::new(&reasons, 7, 2).unwrap(),
                context: BufferRange::new(&context, 7, 16).unwrap(),
                header: BufferRange::new(&header, 7, 4).unwrap(),
            };
            let shape = Shape {
                rows: 2,
                width: 7,
                learned: true,
                epoch: 1,
                op_tag: 77,
            };
            let command = gpu.command().unwrap();
            for bad in [
                Shape { rows: 0, ..shape },
                Shape { rows: 257, ..shape },
                Shape {
                    width: 1537,
                    ..shape
                },
                Shape { epoch: 0, ..shape },
                Shape {
                    epoch: MAX_EPOCH + 1,
                    ..shape
                },
            ] {
                assert!(pipeline.preflight(&command, &ranges, bad).is_err());
            }
            for bad in [
                PairRanges {
                    output: ranges.input,
                    ..ranges
                },
                PairRanges {
                    context: BufferRange {
                        len: 15,
                        ..ranges.context
                    },
                    ..ranges
                },
                PairRanges {
                    header: BufferRange {
                        len: 3,
                        ..ranges.header
                    },
                    ..ranges
                },
            ] {
                assert!(pipeline.preflight(&command, &bad, shape).is_err());
            }
            let unretained = gpu.queue.commandBufferWithUnretainedReferences().unwrap();
            assert!(pipeline.preflight(&unretained, &ranges, shape).is_err());
            reference
                .encode(
                    &command,
                    &DeviceBuffers {
                        input: ranges.input,
                        weights: ranges.weights,
                        output: BufferRange::new(&reference_output, 7, 14).unwrap(),
                        status: BufferRange::new(&reference_status, 7, 1).unwrap(),
                    },
                    BufferRange::new(&reference_reasons, 7, 2).unwrap(),
                    2,
                    7,
                    true,
                )
                .unwrap();
            pipeline.encode(&command, &ranges, shape).unwrap();
            command.commit();
            command.waitUntilCompleted();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            assert!(command.error().is_none());
            let words = |b: &Buffer<u32>| b.with_slice(|s| s.to_vec());
            let raw = serde_json::json!({"format":"toklet-paired-RMS-Rust-offset-v1","rows":2,"width":7,"epoch":1,"op_tag":77,"commands":1,"waits":1,"dispatches":3,"readonly_input_weights_alias":true,"input":words(&input),"output":words(&output),"reference_output":words(&reference_output),"status":words(&status),"reference_status":words(&reference_status),"reasons":words(&reasons),"reference_reasons":words(&reference_reasons),"context":words(&context),"header":words(&header)});
            println!("PAIRED_RMS_RUST_RAW {raw}");
            for b in [
                &input,
                &output,
                &status,
                &reasons,
                &context,
                &header,
                &reference_output,
                &reference_status,
                &reference_reasons,
            ] {
                b.with_slice(|s| {
                    assert_eq!(&s[..7], &[SENTINEL; 7]);
                    assert_eq!(&s[s.len() - 7..], &[SENTINEL; 7]);
                });
            }
            context.with_slice(|s| assert_eq!(&s[23..31], &[SENTINEL; 8]));
            input.with_slice(|s| assert_eq!(&s[7..21], &[1f32.to_bits(); 14]));
            assert_eq!(words(&output), words(&reference_output));
            assert_eq!(words(&status), words(&reference_status));
            assert_eq!(words(&reasons), words(&reference_reasons));
            status.with_slice(|s| assert_eq!(s[7], 0));
            let mut expected = [0f32; 14];
            crate::cpu::rmsnorm(&[1f32; 14], &[1f32; 7], &mut expected, 2, 7, EPSILON).unwrap();
            output.with_slice(|s| {
                for (a, e) in s[7..21].iter().zip(expected) {
                    assert!(
                        (f64::from(f32::from_bits(*a)) - f64::from(e)).abs()
                            <= 1e-5 + 1e-4 * f64::from(e).abs()
                    );
                }
            });
            assert!(pipeline.preflight(&command, &ranges, shape).is_err());
        });
    }
}
