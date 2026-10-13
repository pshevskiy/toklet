use super::{Error, Result, profile::Command};
use objc2::{AnyThread, rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSProcessInfo, NSString};
use objc2_metal::*;
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication, MPSSupportsMTLDevice,
};
use std::{
    marker::PhantomData,
    mem::{align_of, size_of},
    ptr::NonNull,
};

pub(super) trait Scalar: Copy + Default {}
impl Scalar for u16 {}
impl Scalar for u32 {}
pub(super) struct Buffer<T: Scalar> {
    pub raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub len: usize,
    marker: PhantomData<T>,
}
impl<T: Scalar> Buffer<T> {
    pub fn new(gpu: &Gpu, len: usize) -> Result<Self> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or(Error::Options("buffer bytes overflow"))?;
        if bytes == 0 || bytes > gpu.device.maxBufferLength() || bytes > isize::MAX as usize {
            return Err(Error::Options("buffer exceeds device/index limits"));
        }
        let raw = gpu
            .device
            .newBufferWithLength_options(
                bytes,
                MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::HazardTrackingModeTracked,
            )
            .ok_or_else(|| Error::Native("shared allocation failed".into()))?;
        if raw.length() != bytes
            || raw.storageMode() != MTLStorageMode::Shared
            || raw.hazardTrackingMode() != MTLHazardTrackingMode::Tracked
            || !(raw.contents().as_ptr() as usize).is_multiple_of(align_of::<T>())
        {
            return Err(Error::Native("shared buffer contract differs".into()));
        }
        // SAFETY: closed u16/u32 inventory admits every bit pattern; exact size,
        // alignment and isize bounds checked. Newly owned buffer is not in flight.
        unsafe {
            raw.contents().as_ptr().cast::<T>().write_bytes(0, len);
        }
        Ok(Self {
            raw,
            len,
            marker: PhantomData,
        })
    }
    pub fn with_mut<R>(&mut self, f: impl FnOnce(&mut [T]) -> R) -> R {
        // SAFETY: owned initialized aligned allocation. Private callers only
        // upload before submission/after completion; no slice escapes the HRTB.
        unsafe {
            f(std::slice::from_raw_parts_mut(
                self.raw.contents().as_ptr().cast::<T>(),
                self.len,
            ))
        }
    }
    pub fn with_slice<R>(&self, f: impl FnOnce(&[T]) -> R) -> R {
        // SAFETY: private callers have checked command completion (or no command
        // has referenced this buffer). Initialized closed scalar bits, exact len.
        unsafe {
            f(std::slice::from_raw_parts(
                self.raw.contents().as_ptr().cast::<T>(),
                self.len,
            ))
        }
    }
}
#[derive(Clone, Copy)]
pub(super) struct Layout {
    pub rows: usize,
    pub columns: usize,
    pub stride: usize,
    pub elements: usize,
}
impl Layout {
    pub fn new(rows: usize, columns: usize, half: bool) -> Result<Self> {
        if rows == 0 || columns == 0 {
            return Err(Error::Options("empty matrix"));
        }
        let scalar = if half { 2 } else { 4 };
        // SAFETY: nonzero checked dimensions and concrete Float16/Float32 dtype.
        let bytes = unsafe {
            MPSMatrixDescriptor::rowBytesForColumns_dataType(
                columns,
                if half {
                    MPSDataType::Float16
                } else {
                    MPSDataType::Float32
                },
            )
        };
        if !bytes.is_multiple_of(scalar) || bytes / scalar < columns {
            return Err(Error::Options("invalid queried stride"));
        }
        let stride = bytes / scalar;
        let elements = rows
            .checked_mul(stride)
            .and_then(|n| n.checked_add(7))
            .ok_or(Error::Options("layout overflow"))?;
        if elements > u32::MAX as usize {
            return Err(Error::Options("shader indexing overflow"));
        }
        Ok(Self {
            rows,
            columns,
            stride,
            elements,
        })
    }
    pub fn view(self, rows: usize, columns: usize) -> Result<Self> {
        if rows == 0 || rows > self.rows || columns == 0 || columns > self.stride {
            return Err(Error::Options("matrix view exceeds storage"));
        }
        Ok(Self {
            rows,
            columns,
            ..self
        })
    }
    pub fn bytes(self, scalar: usize) -> Result<usize> {
        self.elements
            .checked_mul(scalar)
            .ok_or(Error::Options("layout bytes overflow"))
    }
}
pub(super) fn matrix<T: Scalar>(
    b: &Buffer<T>,
    l: Layout,
    half: bool,
) -> Result<Retained<MPSMatrix>> {
    matrix_offset(b, l, half, 0)
}
/// Offset is in scalar elements; descriptor strides remain those of the owner.
pub(super) fn matrix_offset<T: Scalar>(
    b: &Buffer<T>,
    l: Layout,
    half: bool,
    offset: usize,
) -> Result<Retained<MPSMatrix>> {
    let size = if half { 2 } else { 4 };
    if size_of::<T>() != size
        || b.len != l.elements
        || l.rows
            .checked_mul(l.stride)
            .and_then(|n| n.checked_add(offset))
            .and_then(|n| n.checked_add(7))
            .is_none_or(|n| n > b.len)
        || l.columns > l.stride
    {
        return Err(Error::Options("matrix buffer/type differs"));
    }
    // SAFETY: concrete dtype matches closed scalar size/alignment; checked rows,
    // columns, stride and initialized storage. Descriptor/matrix retain buffer.
    unsafe {
        let d = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            l.rows,
            l.columns,
            l.stride * size,
            if half {
                MPSDataType::Float16
            } else {
                MPSDataType::Float32
            },
        );
        Ok(MPSMatrix::initWithBuffer_offset_descriptor(
            MPSMatrix::alloc(),
            &b.raw,
            offset * size,
            &d,
        ))
    }
}
pub(super) fn multiply(
    gpu: &Gpu,
    n: usize,
    inputs: usize,
    outputs: usize,
) -> Retained<MPSMatrixMultiplication> {
    // SAFETY: callers bind admitted same-device exact n×inputs, outputs×inputs,
    // n×outputs matrices. Row-major right transpose, alpha/beta1 resident bias.
    unsafe {
        MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(MPSMatrixMultiplication::alloc(),&gpu.device,false,true,n,outputs,inputs,1.,1.)
    }
}
#[derive(Clone, Copy)]
pub(super) enum Kernel {
    Embedding,
    Rms,
    Bias,
    Store,
    Rotary,
    Cache,
    Attention,
    Residual,
    Silu,
    Gating,
    Validate,
    Marker,
    Q4Matvec,
    Q4Prefill,
    Q4Compensated,
    Q4CompensatedPrefill,
}
impl Kernel {
    pub fn name(self) -> &'static str {
        [
            "embedding",
            "rms",
            "bias",
            "store",
            "rotary",
            "cache",
            "attention",
            "residual",
            "silu",
            "gating",
            "validate_float",
            "marker",
            "q4_matvec",
            "q4_prefill",
            "q4_compensated",
            "q4_compensated_prefill",
        ][self as usize]
    }
}
pub(super) struct Gpu {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipelines: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub width: usize,
}
impl Gpu {
    pub(super) fn execution_width(&self, kernel: Kernel) -> usize {
        self.pipelines[kernel as usize].threadExecutionWidth()
    }
    pub fn new() -> Result<Self> {
        if NSProcessInfo::processInfo()
            .operatingSystemVersion()
            .majorVersion
            < 15
        {
            return Err(Error::Unsupported("Metal requires macOS15+"));
        }
        let device = MTLCreateSystemDefaultDevice().ok_or(Error::Unsupported("no Metal device"))?;
        if !device.hasUnifiedMemory() || !unsafe { MPSSupportsMTLDevice(Some(&device)) } {
            return Err(Error::Unsupported("requires unified-memory MPS device"));
        }
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version2_3);
        options.setMathMode(MTLMathMode::Safe);
        options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
        if options.languageVersion() != MTLLanguageVersion::Version2_3
            || options.mathMode() != MTLMathMode::Safe
            || options.mathFloatingPointFunctions() != MTLMathFloatingPointFunctions::Precise
        {
            return Err(Error::Unsupported("precise math policy unavailable"));
        }
        let library = device
            .newLibraryWithSource_options_error(
                &NSString::from_str(include_str!("kernels.metal")),
                Some(&options),
            )
            .map_err(|e| Error::Native(e.to_string()))?;
        let mut pipelines = Vec::new();
        let mut width = device.maxThreadsPerThreadgroup().width;
        for k in [
            Kernel::Embedding,
            Kernel::Rms,
            Kernel::Bias,
            Kernel::Store,
            Kernel::Rotary,
            Kernel::Cache,
            Kernel::Attention,
            Kernel::Residual,
            Kernel::Silu,
            Kernel::Gating,
            Kernel::Validate,
            Kernel::Marker,
            Kernel::Q4Matvec,
            Kernel::Q4Prefill,
            Kernel::Q4Compensated,
            Kernel::Q4CompensatedPrefill,
        ] {
            let f = library
                .newFunctionWithName(&NSString::from_str(k.name()))
                .ok_or_else(|| Error::Native("kernel absent".into()))?;
            let p = device
                .newComputePipelineStateWithFunction_error(&f)
                .map_err(|e| Error::Native(e.to_string()))?;
            width = width
                .min(p.threadExecutionWidth())
                .min(p.maxTotalThreadsPerThreadgroup());
            pipelines.push(p);
        }
        if width == 0 {
            return Err(Error::Unsupported("empty launch width"));
        }
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| Error::Native("queue absent".into()))?;
        Ok(Self {
            device,
            queue,
            pipelines,
            width,
        })
    }
    pub fn command(&self) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        let c = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Error::Native("command absent".into()))?;
        if !c.retainedReferences() {
            return Err(Error::Unsupported("unretained resources"));
        }
        Ok(c)
    }
    pub fn encode(
        &self,
        c: &Command<'_>,
        k: Kernel,
        buffers: &[&ProtocolObject<dyn MTLBuffer>],
        shape: &[u32],
        count: usize,
        status: Option<&Buffer<u32>>,
    ) -> Result<()> {
        if count == 0 || count > u32::MAX as usize || shape.is_empty() {
            return Err(Error::Options("invalid dispatch"));
        }
        // Q4 assigns one queried SIMD group to each output row. Other kernels
        // assign one scalar lane per item; never assume an Apple SIMD width.
        let width = if matches!(
            k,
            Kernel::Q4Matvec
                | Kernel::Q4Prefill
                | Kernel::Q4Compensated
                | Kernel::Q4CompensatedPrefill
        ) {
            self.execution_width(k)
        } else {
            self.width
        };
        if width == 0 || width > self.pipelines[k as usize].maxTotalThreadsPerThreadgroup() {
            return Err(Error::Unsupported("kernel launch width unsupported"));
        }
        let groups = if matches!(
            k,
            Kernel::Q4Matvec
                | Kernel::Q4Prefill
                | Kernel::Q4Compensated
                | Kernel::Q4CompensatedPrefill
        ) {
            count
        } else {
            count.div_ceil(width)
        };
        if groups
            .checked_mul(width)
            .is_none_or(|n| n > u32::MAX as usize)
        {
            return Err(Error::Options("rounded dispatch overflow"));
        }
        let (index, e) = if let Some(trace) = c.trace {
            let (i, e) = trace.encoder(c.raw, k.name())?;
            (Some(i), e)
        } else {
            (
                None,
                c.computeCommandEncoder()
                    .ok_or_else(|| Error::Native("encoder absent".into()))?,
            )
        };
        e.setLabel(Some(&NSString::from_str(k.name())));
        e.setComputePipelineState(&self.pipelines[k as usize]);
        // SAFETY: closed private call sites bind the documented exact shader ABIs,
        // initialized admitted layouts, copied UInt32 params and optional atomic
        // status after params. Residual is the sole in-place numerical operand.
        // Owners/cmd retain resources through checked wait; CPU access excluded.
        unsafe {
            for (i, b) in buffers.iter().enumerate() {
                e.setBuffer_offset_atIndex(Some(b), 0, i);
            }
            e.setBytes_length_atIndex(
                NonNull::new(shape.as_ptr().cast_mut()).unwrap().cast(),
                std::mem::size_of_val(shape),
                buffers.len(),
            );
            if let Some(status) = status {
                e.setBuffer_offset_atIndex(Some(&status.raw), 0, buffers.len() + 1);
            }
        }
        e.dispatchThreadgroups_threadsPerThreadgroup(
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
        e.endEncoding();
        if let Some(index) = index {
            c.trace.unwrap().finish_encoder(index);
        }
        Ok(())
    }
    pub fn marker(&self, c: &Command<'_>, source: &Buffer<u32>, name: &'static str) -> Result<()> {
        if let Some(trace) = c.trace {
            let (index, e) = trace.encoder(c.raw, name)?;
            e.setComputePipelineState(&self.pipelines[Kernel::Marker as usize]);
            e.setLabel(Some(&NSString::from_str(name)));
            // SAFETY: same-device live disjoint u32 buffers with at least one
            // initialized element. Marker atomically ORs zero into the result bits to establish
            // tracked read/write ordering around MPS, preserving every result bit;
            // its dedicated sink retains the observed bits. Atomic u32 alignment
            // matches the admitted four-byte FP32 storage. Access is serialized.
            unsafe {
                e.setBuffer_offset_atIndex(Some(&source.raw), 0, 0);
                e.setBuffer_offset_atIndex(Some(&trace.sink.raw), 0, 1);
            }
            e.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            );
            e.endEncoding();
            trace.finish_encoder(index);
        }
        Ok(())
    }
}
