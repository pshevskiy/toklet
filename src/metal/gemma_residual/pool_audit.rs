//! Integer audit/retirement entry in the existing residual pipeline family.
use super::*;

pub(in crate::metal) const SENTINEL: u32 = 0x5a17_0031;
pub(in crate::metal) const GUARD: u32 = 1;
pub(in crate::metal) const NONFINITE: u32 = 2;
pub(in crate::metal) const DESCRIPTOR: u32 = 4;
pub(in crate::metal) const MAX_POOL_WORDS: usize = 6_594_784;
pub(in crate::metal) const MAX_CHUNKS: usize = 6472;
#[derive(Clone, Copy)]
pub(in crate::metal) struct Shape {
    pub pool_words: usize,
    pub chunks: usize,
    pub fill_after: bool,
}
pub(in crate::metal) struct Pipeline {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    width: usize,
}
fn geometry(shape: Shape, lengths: [usize; 5], width: usize) -> Result<()> {
    if !(1..=MAX_POOL_WORDS).contains(&shape.pool_words)
        || !(1..=MAX_CHUNKS).contains(&shape.chunks)
        || !(1..=1024).contains(&width)
        || lengths != [128, 4 * shape.chunks, 32, shape.pool_words, shape.chunks]
    {
        return Err(Error::Options("pool audit shape differs"));
    }
    Ok(())
}
/// Missing/sentinel, unknown, descriptor or guard receipts are fatal before
/// replay. NONFINITE remains separate until the old post-replay finite gate.
pub(in crate::metal) fn receipt(word: u32) -> Result<bool> {
    if word & !(GUARD | NONFINITE | DESCRIPTOR) != 0 || word & (GUARD | DESCRIPTOR) != 0 {
        return Err(Error::Options(
            "pool audit guard/descriptor receipt differs",
        ));
    }
    Ok(word & NONFINITE == 0)
}
impl Pipeline {
    pub(in crate::metal) fn new(
        gpu: &Gpu,
        library: &ProtocolObject<dyn MTLLibrary>,
    ) -> Result<Self> {
        let function = library
            .newFunctionWithName(&NSString::from_str("gemma_pool_audit_fill_v1"))
            .ok_or_else(|| Error::Native("pool audit entry absent".into()))?;
        let pipeline = gpu
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| Error::Native(e.to_string()))?;
        let width = pipeline.threadExecutionWidth();
        if !(1..=1024).contains(&width)
            || width > pipeline.maxTotalThreadsPerThreadgroup()
            || pipeline
                .staticThreadgroupMemoryLength()
                .checked_add((width * 4).next_multiple_of(16))
                .is_none_or(|n| n > gpu.device.maxThreadgroupMemoryLength())
        {
            return Err(Error::Unsupported("pool audit launch unavailable"));
        }
        Ok(Self {
            device: gpu.device.clone(),
            pipeline,
            width,
        })
    }
    pub(in crate::metal) fn metadata_bytes(&self) -> usize {
        size_of::<Self>()
    }
    /// Exact immutable coverage, per-node active counts, canaries and receipt
    /// initialization are cold-owner contracts; no CPU view or wait here.
    #[allow(
        clippy::too_many_arguments,
        reason = "Concrete sealed five-buffer audit ABI at one operation boundary"
    )]
    pub(in crate::metal) fn preflight(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        spans: BufferRange<'_>,
        work: BufferRange<'_>,
        active: BufferRange<'_>,
        pool: BufferRange<'_>,
        receipts: BufferRange<'_>,
        shape: Shape,
    ) -> Result<()> {
        let ranges = [spans, work, active, pool, receipts];
        geometry(shape, ranges.map(|r| r.len), self.width)?;
        for range in ranges {
            let bytes = range
                .offset
                .checked_add(range.len)
                .and_then(|n| n.checked_mul(4));
            if range.raw.storageMode() != MTLStorageMode::Shared
                || range.raw.hazardTrackingMode() != MTLHazardTrackingMode::Tracked
                || bytes.is_none_or(|n| n > range.raw.length())
            {
                return Err(Error::Options("pool audit buffer extent/mode differs"));
            }
        }
        validate_encode_ranges(command, &self.device, &ranges, 3)
    }
    #[allow(
        clippy::too_many_arguments,
        reason = "Concrete sealed five-buffer audit ABI at one operation boundary"
    )]
    pub(in crate::metal) fn encode(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        spans: BufferRange<'_>,
        work: BufferRange<'_>,
        active: BufferRange<'_>,
        pool: BufferRange<'_>,
        receipts: BufferRange<'_>,
        shape: Shape,
    ) -> Result<()> {
        self.preflight(command, spans, work, active, pool, receipts, shape)?;
        let constants = [
            shape.chunks as u32,
            shape.pool_words as u32,
            u32::from(shape.fill_after),
            SENTINEL,
        ];
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| Error::Native("pool audit encoder absent".into()))?;
        encoder.setComputePipelineState(&self.pipeline);
        // SAFETY: checked same-device retained/tracked command and exact shared
        // ranges, disjoint writes, u32 ABI and supported queried local storage.
        // Cold owner seals full work coverage; caller owns terminal validation.
        unsafe {
            for (i, range) in [spans, work, active, pool, receipts].iter().enumerate() {
                encoder.setBuffer_offset_atIndex(Some(range.raw), range.offset * 4, i);
            }
            encoder.setBytes_length_atIndex(
                NonNull::from(&constants).cast(),
                size_of::<[u32; 4]>(),
                5,
            );
            encoder.setThreadgroupMemoryLength_atIndex((self.width * 4).next_multiple_of(16), 0);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: shape.chunks,
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
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pool_audit_geometry_and_receipt_classes_are_bounded_and_separate() {
        let shape = Shape {
            pool_words: 4096,
            chunks: 32,
            fill_after: true,
        };
        assert!(geometry(shape, [128, 128, 32, 4096, 32], 3).is_ok());
        assert!(geometry(shape, [128, 128, 32, 4096, 32], 1024).is_ok());
        for lengths in [
            [127, 128, 32, 4096, 32],
            [128, 127, 32, 4096, 32],
            [128, 128, 31, 4096, 32],
            [128, 128, 32, 4095, 32],
            [128, 128, 32, 4096, 31],
        ] {
            assert!(geometry(shape, lengths, 32).is_err());
        }
        for width in [0, 1025, usize::MAX] {
            assert!(geometry(shape, [128, 128, 32, 4096, 32], width).is_err());
        }
        for (pool_words, chunks) in [
            (0, 32),
            (MAX_POOL_WORDS + 1, 32),
            (4096, 0),
            (4096, MAX_CHUNKS + 1),
        ] {
            assert!(
                geometry(
                    Shape {
                        pool_words,
                        chunks,
                        fill_after: false
                    },
                    [128, 128, 32, 4096, 32],
                    32
                )
                .is_err()
            );
        }
        assert!(receipt(0).unwrap());
        assert!(!receipt(NONFINITE).unwrap());
        for bad in [
            GUARD,
            DESCRIPTOR,
            GUARD | NONFINITE,
            DESCRIPTOR | NONFINITE,
            7,
            8,
            SENTINEL,
        ] {
            assert!(receipt(bad).is_err());
        }
    }
}
