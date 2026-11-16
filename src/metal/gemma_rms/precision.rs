//! Private qualification-only higher-accuracy RMS. The ordinary RMS owner and
//! pilot keep their original pipelines. Status1=fatal input,2=whole-transaction
//! reference request; no CPU view/reset/submit/wait/publication in encode.
use super::*;

pub(in crate::metal) const POLICY_BYTES: usize = 64 * 1024 * 1024;
pub(in crate::metal) const KNOWN_REASON_BITS: u32 = 0x8000_0fff;
pub(in crate::metal) const STATUS_SLOTS: [usize; 8] = [0, 3, 6, 16, 18, 25, 32, 9];

pub(in crate::metal) struct Pipelines {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    pipelines: [Retained<ProtocolObject<dyn MTLComputePipelineState>>; 2],
    widths: [usize; 2],
}
impl Pipelines {
    pub(in crate::metal) fn new(
        gpu: &Gpu,
        original: &super::Pipelines,
        available: usize,
    ) -> Result<Self> {
        if available < POLICY_BYTES {
            return Err(Error::Budget {
                required: POLICY_BYTES,
                limit: available,
            });
        }
        if !std::ptr::eq(&*original.device, &*gpu.device) {
            return Err(Error::Options("precision RMS original device differs"));
        }
        autoreleasepool(|_| {
            let options = MTLCompileOptions::new();
            options.setLanguageVersion(MTLLanguageVersion::Version2_3);
            options.setMathMode(MTLMathMode::Safe);
            options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
            if options.mathMode() != MTLMathMode::Safe
                || options.mathFloatingPointFunctions() != MTLMathFloatingPointFunctions::Precise
                || options.languageVersion() != MTLLanguageVersion::Version2_3
            {
                return Err(Error::Unsupported("precision RMS precise math unavailable"));
            }
            let library = gpu
                .device
                .newLibraryWithSource_options_error(
                    &NSString::from_str(include_str!("precision.metal")),
                    Some(&options),
                )
                .map_err(|e| Error::Native(e.to_string()))?;
            let pipeline = |name| {
                let function = library
                    .newFunctionWithName(&NSString::from_str(name))
                    .ok_or_else(|| Error::Native("precision RMS entry absent".into()))?;
                gpu.device
                    .newComputePipelineStateWithFunction_error(&function)
                    .map_err(|e| Error::Native(e.to_string()))
            };
            let pipelines = [
                pipeline("gemma_rms_precision_unlearned")?,
                pipeline("gemma_rms_precision_learned")?,
            ];
            let widths = pipelines.each_ref().map(|p| p.threadExecutionWidth());
            if widths != original.widths {
                return Err(Error::Unsupported(
                    "precision RMS original SIMD widths differ",
                ));
            }
            for (pipeline, &width) in pipelines.iter().zip(&widths) {
                if width == 0
                    || width > pipeline.maxTotalThreadsPerThreadgroup()
                    || width > u16::MAX as usize
                {
                    return Err(Error::Unsupported("precision RMS launch width unavailable"));
                }
            }
            Ok(Self {
                device: gpu.device.clone(),
                pipelines,
                widths,
            })
        })
    }
    /// Same checked range/command ABI plus a persistent disjoint per-row reason
    /// range. Caller retains all ranges and excludes CPU views to completion.
    pub(in crate::metal) fn encode(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        buffers: &DeviceBuffers<'_>,
        reasons: BufferRange<'_>,
        rows: usize,
        width: usize,
        learned: bool,
    ) -> Result<()> {
        let g = Geometry::new(rows, width, EPSILON, learned)?;
        if buffers.input.len != g.values
            || buffers.output.len != g.values
            || buffers.weights.len != g.weights
            || buffers.status.len != 1
            || reasons.len != rows
        {
            return Err(Error::Options("precision RMS borrowed extent differs"));
        }
        validate_encode_ranges(
            command,
            &self.device,
            &[
                buffers.input,
                buffers.weights,
                buffers.output,
                buffers.status,
                reasons,
            ],
            2,
        )?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| Error::Native("precision RMS encoder absent".into()))?;
        let variant = usize::from(learned);
        encoder.setComputePipelineState(&self.pipelines[variant]);
        // SAFETY: exact checked lengths, same-device shared/tracked buffers,
        // disjoint writes, retained NotEnqueued command; shape bytes copy now.
        unsafe {
            for (slot, b) in [buffers.input, buffers.weights, buffers.output]
                .iter()
                .enumerate()
            {
                encoder.setBuffer_offset_atIndex(Some(b.raw), b.offset * 4, slot);
            }
            encoder.setBytes_length_atIndex(
                NonNull::from(&g.words).cast(),
                size_of::<[u32; 3]>(),
                3,
            );
            encoder.setBuffer_offset_atIndex(
                Some(buffers.status.raw),
                buffers.status.offset * 4,
                4,
            );
            encoder.setBuffer_offset_atIndex(Some(reasons.raw), reasons.offset * 4, 5);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: rows,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: self.widths[variant],
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        Ok(())
    }
}

pub(in crate::metal) mod two_phase;
