//! Standalone Rust binding qualification; not an inference backend.
//! Native buffers are private to one synchronous probe on one host thread.
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};

const GUARDS: usize = 7;
const INPUT_GUARD: u32 = 0x4642_fedc;
const OUTPUT_GUARD: u32 = 0x46a3_5ce1;
const LIMIT: usize = 6 * 1024 * 1024 * 1024;
const ALLOWANCE: usize = 64 * 1024 * 1024;
const SHAPES: [usize; 3] = [1, 17, 257];

#[derive(Debug)]
struct Error {
    stage: &'static str,
    message: String,
}
fn error(stage: &'static str, message: impl Into<String>) -> Error {
    Error {
        stage,
        message: message.into(),
    }
}

fn bytes(count: usize) -> Result<usize, Error> {
    if count == 0 || u32::try_from(count).is_err() {
        return Err(error("admission", "count must fit a nonzero shader uint"));
    }
    let bytes = count
        .checked_add(GUARDS)
        .and_then(|n| n.checked_mul(size_of::<f32>()))
        .ok_or_else(|| error("admission", "buffer byte count overflow"))?;
    if bytes > isize::MAX as usize {
        return Err(error(
            "admission",
            "buffer exceeds Rust pointer offset limit",
        ));
    }
    Ok(bytes)
}
fn admission(count: usize, limit: usize) -> Result<usize, Error> {
    // Two shared allocations, one host input and two diagnostic readbacks.
    // Shared CPU/GPU storage is counted once, not twice.
    let total = bytes(count)?
        .checked_mul(5)
        .and_then(|n| n.checked_add(ALLOWANCE))
        .ok_or_else(|| error("admission", "working-set count overflow"))?;
    if limit > LIMIT || total > limit {
        return Err(error(
            "admission",
            "probe exceeds the supplied 6-GiB-or-smaller budget",
        ));
    }
    Ok(total)
}
fn groups(count: usize, width: usize) -> Result<usize, Error> {
    if count == 0 || width == 0 {
        return Err(error("dispatch", "zero count or execution width"));
    }
    let groups = count.div_ceil(width);
    let rounded = groups
        .checked_mul(width)
        .ok_or_else(|| error("dispatch", "rounded launch overflow"))?;
    if u32::try_from(rounded).is_err() {
        return Err(error(
            "dispatch",
            "rounded launch exceeds shader uint indexing",
        ));
    }
    Ok(groups)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod metal {
    use super::*;
    use objc2::{
        rc::{Retained, autoreleasepool},
        runtime::ProtocolObject,
    };
    use objc2_foundation::{NSProcessInfo, NSString};
    use objc2_metal::*;
    use std::{marker::PhantomData, ptr::NonNull, time::Instant};

    // Closed private scalar inventory: zero bytes and every loaded bit pattern
    // are valid Rust values for all three types. u16 carries stored Metal Half
    // bits; u32 carries token IDs without floating conversion.
    trait Scalar: Copy + Default {}
    impl Scalar for f32 {}
    impl Scalar for u16 {}
    impl Scalar for u32 {}

    /// One owner; no raw pointer, native handle, borrowed slice or clone escapes.
    /// All accesses happen before submission or after waitUntilCompleted.
    struct Shared<T: Scalar> {
        raw: Retained<ProtocolObject<dyn MTLBuffer>>,
        elements: usize,
        scalar: PhantomData<T>,
    }
    impl<T: Scalar> Shared<T> {
        fn new(device: &ProtocolObject<dyn MTLDevice>, bytes: usize) -> Result<Self, Error> {
            if bytes == 0
                || bytes > device.maxBufferLength()
                || bytes > isize::MAX as usize
                || !bytes.is_multiple_of(size_of::<T>())
            {
                return Err(error(
                    "allocation",
                    "buffer length exceeds qualified limits",
                ));
            }
            let raw = device
                .newBufferWithLength_options(
                    bytes,
                    MTLResourceOptions::StorageModeShared
                        | MTLResourceOptions::HazardTrackingModeTracked,
                )
                .ok_or_else(|| error("allocation", "Metal shared allocation failed"))?;
            if raw.length() != bytes
                || raw.storageMode() != MTLStorageMode::Shared
                || raw.hazardTrackingMode() != MTLHazardTrackingMode::Tracked
                || !(raw.contents().as_ptr() as usize).is_multiple_of(align_of::<T>())
            {
                return Err(error(
                    "allocation",
                    "shared length/storage/alignment differs",
                ));
            }
            let elements = bytes / size_of::<T>();
            // SAFETY: new owned shared allocation; actual length/alignment and
            // isize bounds checked above. No GPU command references it yet.
            // Zero initializes every f32/u16/u32 slot, including guard storage.
            unsafe { raw.contents().as_ptr().cast::<T>().write_bytes(0, elements) };
            Ok(Self {
                raw,
                elements,
                scalar: PhantomData,
            })
        }
        fn write(&mut self, values: &[T]) -> Result<(), Error> {
            if values.len() != self.elements {
                return Err(error("upload", "complete buffer length differs"));
            }
            // SAFETY: owned initialized aligned allocation, exact checked
            // length. Host Vec cannot alias the private Metal allocation. No
            // GPU use is in flight: before first submission or after completion.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    self.raw.contents().as_ptr().cast::<T>(),
                    self.elements,
                )
            };
            Ok(())
        }
        fn fill(&mut self, value: T) {
            // SAFETY: initialized aligned owned allocation of elements T's.
            // Called before submission, or after the prior command finished.
            unsafe {
                std::slice::from_raw_parts_mut(
                    self.raw.contents().as_ptr().cast::<T>(),
                    self.elements,
                )
                .fill(value)
            };
        }
        fn read(&self) -> Vec<T> {
            // SAFETY: initialized aligned owned allocation; every read caller
            // has waited for command completion, including command-error paths.
            // The slice is immediately copied; no borrow escapes this method.
            unsafe {
                std::slice::from_raw_parts(self.raw.contents().as_ptr().cast::<T>(), self.elements)
                    .to_vec()
            }
        }
    }

    struct Probe {
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        width: usize,
    }
    impl Probe {
        fn case(&mut self, count: usize, limit: usize) -> Result<Value, Error> {
            let admitted = admission(count, limit)?;
            let bytes = bytes(count)?;
            let launch = groups(count, self.width)?;
            let mut input = Shared::<f32>::new(&self.device, bytes)?;
            let mut output = Shared::<f32>::new(&self.device, bytes)?;
            let mut host_input = vec![f32::from_bits(INPUT_GUARD); count + GUARDS];
            for (index, value) in host_input[..count].iter_mut().enumerate() {
                *value = (index % 29) as f32 * 0.25 - 3.5;
            }
            input.write(&host_input)?;
            let mut rows = Vec::with_capacity(13);
            for repetition in 0..13 {
                output.fill(f32::from_bits(OUTPUT_GUARD));
                let row = autoreleasepool(|_| {
                    let started = Instant::now();
                    let command = self
                        .queue
                        .commandBuffer()
                        .ok_or_else(|| error("command", "command buffer unavailable"))?;
                    if !command.retainedReferences() {
                        return Err(error("command", "unretained references are unsupported"));
                    }
                    let encoder = command
                        .computeCommandEncoder()
                        .ok_or_else(|| error("command", "compute encoder unavailable"))?;
                    encoder.setComputePipelineState(&self.pipeline);
                    let uniform =
                        u32::try_from(count).map_err(|_| error("dispatch", "count overflow"))?;
                    // SAFETY: exact shader signature: readonly FP32 input at0,
                    // writable FP32 output at1, copied UInt32 count at2. Offset0
                    // is aligned; both allocations cover count+7 elements.
                    // Count/rounded grid are checked, and the shader guards all
                    // tail lanes. Owners remain live through the synchronous
                    // wait; native command also retains resources. Host access
                    // is excluded between commit and completion, with tracked
                    // hazards and shared storage on the qualified device.
                    unsafe {
                        encoder.setBuffer_offset_atIndex(Some(&input.raw), 0, 0);
                        encoder.setBuffer_offset_atIndex(Some(&output.raw), 0, 1);
                        encoder.setBytes_length_atIndex(
                            NonNull::from(&uniform).cast(),
                            size_of::<u32>(),
                            2,
                        );
                    }
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: launch,
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
                    command.commit();
                    // No fallible host work, allocation or readback between
                    // commit and this wait. Errors are inspected only afterward.
                    command.waitUntilCompleted();
                    let seconds = started.elapsed().as_secs_f64();
                    if command.status() != MTLCommandBufferStatus::Completed
                        || command.error().is_some()
                    {
                        return Err(error(
                            "completion",
                            format!("GPU command failed: {:?}", command.error()),
                        ));
                    }
                    let actual = output.read();
                    let retained_input = input.read();
                    if retained_input
                        .iter()
                        .zip(&host_input)
                        .any(|(a, b)| a.to_bits() != b.to_bits())
                    {
                        return Err(error(
                            "validation",
                            "readonly input or input guards changed",
                        ));
                    }
                    for (index, &value) in actual.iter().enumerate() {
                        let expected = if index < count {
                            (host_input[index] + 1.0).to_bits()
                        } else {
                            OUTPUT_GUARD
                        };
                        if value.to_bits() != expected {
                            return Err(error(
                                "validation",
                                format!(
                                    "output/guard {index}: {} versus {expected}",
                                    value.to_bits()
                                ),
                            ));
                        }
                    }
                    Ok(json!({"phase":if repetition<3 {"warmup"} else {"sample"},
                        "repetition":if repetition<3 {repetition+1} else {repetition-2},
                        "encode_commit_wait_seconds":seconds,"command_completed":true,
                        "input_bits":retained_input.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),
                        "output_bits":actual.iter().map(|x|x.to_bits()).collect::<Vec<_>>()}))
                })?;
                rows.push(row);
            }
            Ok(
                json!({"count":count,"guards":GUARDS,"threadgroup_width":self.width,
                "threadgroups":launch,"shared_buffer_bytes":bytes*2,
                "host_input_and_readback_bytes":bytes*3,"admitted_bytes":admitted,
                "allocations_reused_across_repetitions":true,"rows":rows}),
            )
        }
    }

    fn prepare(
        source: &str,
        function_name: &str,
        admitted: usize,
    ) -> Result<(Probe, Value), Error> {
        let version = NSProcessInfo::processInfo().operatingSystemVersion();
        if version.majorVersion < 15 {
            return Err(error(
                "capability",
                "math policy requires macOS 15 or newer",
            ));
        }
        let started = Instant::now();
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| error("device", "Metal device unavailable"))?;
        if !device.hasUnifiedMemory() {
            return Err(error(
                "capability",
                "probe qualifies unified Apple Silicon memory only",
            ));
        }
        let discovery = started.elapsed().as_secs_f64();
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version2_3);
        options.setMathMode(MTLMathMode::Safe);
        options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
        if options.languageVersion() != MTLLanguageVersion::Version2_3
            || options.mathMode() != MTLMathMode::Safe
            || options.mathFloatingPointFunctions() != MTLMathFloatingPointFunctions::Precise
        {
            return Err(error(
                "capability",
                "compile options differ from the qualified policy",
            ));
        }
        let started = Instant::now();
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
            .map_err(|e| error("shader_compilation", e.to_string()))?;
        let compilation = started.elapsed().as_secs_f64();
        let function = library
            .newFunctionWithName(&NSString::from_str(function_name))
            .ok_or_else(|| error("pipeline", "compiled function unavailable"))?;
        let started = Instant::now();
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| error("pipeline", e.to_string()))?;
        let pipeline_seconds = started.elapsed().as_secs_f64();
        let device_threads = device.maxThreadsPerThreadgroup();
        let width = pipeline
            .threadExecutionWidth()
            .min(pipeline.maxTotalThreadsPerThreadgroup())
            .min(device_threads.width);
        if width == 0 || device_threads.height == 0 || device_threads.depth == 0 {
            return Err(error(
                "capability",
                "invalid pipeline/device dispatch limits",
            ));
        }
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| error("command", "queue unavailable"))?;
        let metadata = json!({"device":device.name().to_string(),"registry_id":device.registryID(),
                "unified_memory":device.hasUnifiedMemory(),"max_buffer_length":device.maxBufferLength(),
                "recommended_working_set_bytes":device.recommendedMaxWorkingSetSize(),
                "execution_width":pipeline.threadExecutionWidth(),
                "pipeline_max_threads":pipeline.maxTotalThreadsPerThreadgroup(),
                "device_max_threads":[device_threads.width,device_threads.height,device_threads.depth],
                "language_version":"2.3","math_mode":"safe","math_functions":"precise",
                "minimum_macos_major":15,"device_discovery_seconds":discovery,
                "shader_compile_seconds":compilation,"pipeline_compile_seconds":pipeline_seconds,
                "maximum_admitted_bytes":admitted});
        Ok((
            Probe {
                device,
                queue,
                pipeline,
                width,
            },
            metadata,
        ))
    }

    pub(super) fn run(invalid_shader: bool, limit: usize) -> Result<Value, Error> {
        let admitted = SHAPES
            .into_iter()
            .map(|count| admission(count, limit))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .unwrap_or(0);
        autoreleasepool(|_| {
            let source = if invalid_shader {
                "intentional invalid Metal source"
            } else {
                include_str!("rust_metal_probe.metal")
            };
            let (mut probe, metadata) = prepare(source, "toklet_add_one", admitted)?;
            let mut cases = Vec::new();
            for count in SHAPES {
                cases.push(autoreleasepool(|_| probe.case(count, limit))?);
            }
            Ok(json!({"metadata":metadata,"cases":cases}))
        })
    }

    mod rms {
        include!("rust_metal_rms.rs");
    }
    pub(super) fn run_rms(path: &std::path::Path, limit: usize) -> Result<Value, Error> {
        rms::run(path, limit)
    }
    mod embedding {
        include!("rust_metal_embedding.rs");
    }
    pub(super) fn run_embedding(path: &std::path::Path, limit: usize) -> Result<Value, Error> {
        embedding::run(path, limit)
    }
    mod residual {
        include!("rust_metal_residual.rs");
    }
    pub(super) fn run_residual(path: &std::path::Path, limit: usize) -> Result<Value, Error> {
        residual::run(path, limit)
    }
    mod silu {
        include!("rust_metal_silu.rs");
    }
    pub(super) fn run_silu(path: &std::path::Path, limit: usize) -> Result<Value, Error> {
        silu::run(path, limit)
    }
    mod gating {
        include!("rust_metal_gating.rs");
    }
    pub(super) fn run_gating(path: &std::path::Path, limit: usize) -> Result<Value, Error> {
        gating::run(path, limit)
    }
}

fn run() -> Result<Value, Error> {
    let mut args = std::env::args_os().skip(1);
    let mut invalid = false;
    let mut limit = LIMIT;
    let mut has_budget = false;
    let mut rms_input = None;
    let mut embedding_input = None;
    let mut residual_input = None;
    let mut silu_input = None;
    let mut gating_input = None;
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("--invalid-shader") if !invalid => invalid = true,
            Some("--rms-input") if rms_input.is_none() => {
                rms_input = Some(std::path::PathBuf::from(
                    args.next()
                        .ok_or_else(|| error("arguments", "missing RMS input"))?,
                ));
            }
            Some("--embedding-input") if embedding_input.is_none() => {
                embedding_input =
                    Some(std::path::PathBuf::from(args.next().ok_or_else(|| {
                        error("arguments", "missing embedding input")
                    })?));
            }
            Some("--residual-input") if residual_input.is_none() => {
                residual_input =
                    Some(std::path::PathBuf::from(args.next().ok_or_else(|| {
                        error("arguments", "missing residual input")
                    })?));
            }
            Some("--silu-input") if silu_input.is_none() => {
                silu_input = Some(std::path::PathBuf::from(
                    args.next()
                        .ok_or_else(|| error("arguments", "missing SiLU input"))?,
                ));
            }
            Some("--gating-input") if gating_input.is_none() => {
                gating_input = Some(std::path::PathBuf::from(
                    args.next()
                        .ok_or_else(|| error("arguments", "missing gating input"))?,
                ));
            }
            Some("--budget-bytes") if !has_budget => {
                let value = args
                    .next()
                    .ok_or_else(|| error("arguments", "missing budget"))?;
                limit = value
                    .to_str()
                    .ok_or_else(|| error("arguments", "non-Unicode budget"))?
                    .parse()
                    .map_err(|_| error("arguments", "invalid budget"))?;
                has_budget = true;
            }
            _ => return Err(error("arguments", "unsupported probe argument")),
        }
    }
    if usize::from(invalid)
        + usize::from(rms_input.is_some())
        + usize::from(embedding_input.is_some())
        + usize::from(residual_input.is_some())
        + usize::from(silu_input.is_some())
        + usize::from(gating_input.is_some())
        > 1
    {
        return Err(error("arguments", "probe operation modes are exclusive"));
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        if let Some(path) = rms_input {
            metal::run_rms(&path, limit)
        } else if let Some(path) = embedding_input {
            metal::run_embedding(&path, limit)
        } else if let Some(path) = residual_input {
            metal::run_residual(&path, limit)
        } else if let Some(path) = silu_input {
            metal::run_silu(&path, limit)
        } else if let Some(path) = gating_input {
            metal::run_gating(&path, limit)
        } else {
            metal::run(invalid, limit)
        }
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = (
            invalid,
            limit,
            rms_input,
            embedding_input,
            residual_input,
            silu_input,
            gating_input,
        );
        Err(error("capability", "requires Apple Silicon macOS"))
    }
}

fn main() -> std::process::ExitCode {
    let result = run();
    let passed = result.is_ok();
    let message = match result {
        Ok(value) => json!({"event":"idle_result","passed":true,"result":value}),
        Err(e) => {
            json!({"event":"idle_result","passed":false,"error":{"stage":e.stage,"message":e.message}})
        }
    };
    let mut out = io::stdout().lock();
    if writeln!(out, "{message}")
        .and_then(|_| out.flush())
        .is_err()
    {
        return std::process::ExitCode::FAILURE;
    }
    // All submitted work is complete and resources are released. Remain alive
    // so the driver can read this child's kernel-reported physical peak.
    let mut ack = String::new();
    if io::stdin().lock().read_line(&mut ack).is_err() || ack != "done\n" || !passed {
        return std::process::ExitCode::FAILURE;
    }
    std::process::ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn allocation_admission_rejects_empty_overflow_and_insufficient_budget() {
        assert!(bytes(0).is_err());
        assert!(bytes(usize::MAX).is_err());
        assert_eq!(bytes(1).unwrap(), 32);
        assert!(admission(1, ALLOWANCE + 159).is_err());
        assert_eq!(admission(1, ALLOWANCE + 160).unwrap(), ALLOWANCE + 160);
        assert!(admission(1, LIMIT + 1).is_err());
    }
    #[test]
    fn rounded_tails_cannot_wrap_shader_indices() {
        assert!(groups(0, 32).is_err());
        assert!(groups(1, 0).is_err());
        assert_eq!(groups(1, 32).unwrap(), 1);
        assert_eq!(groups(17, 32).unwrap(), 1);
        assert_eq!(groups(257, 32).unwrap(), 9);
        assert!(groups(u32::MAX as usize, 32).is_err());
        assert!(groups(usize::MAX, 2).is_err());
    }
}
