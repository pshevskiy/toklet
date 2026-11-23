//! Admission and dispatch for the pinned, single-export Nanbeige Metal image.
use super::{
    Buffer, Context, Error, HIDDEN, INTERMEDIATE, Options, QUERY_WIDTH, Result, Sizes, add,
    kernel::{Encoder, Op, RawBuffer, encode_pipeline},
    plan,
};
use crate::metal::mlx::precise_library_on;
use dispatch2::DispatchData;
use half::bf16;
use objc2::{
    rc::Retained,
    runtime::{NSObjectProtocol, ProtocolObject},
    sel,
};
use objc2_foundation::{NSProcessInfo, NSString};
use objc2_metal::*;
use sha2::{Digest, Sha256};
use std::{
    fs::OpenOptions, io::Read, os::unix::fs::OpenOptionsExt, path::Path, ptr::NonNull,
    time::Instant,
};

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
const FILE_BYTES: usize = 13_472;
const FILE_SHA: &str = "78496343fbbcb95ac8949cb4c384baeb8c0edf4f8f87a37e506b844b02d88353";
const SYMBOL: &str = "affine_qmm_t_bfloat16_t_gs_64_b_4_alN_true_batch_0";
const STARTUP_RESERVE: usize = 1 << 20;

#[cfg(test)]
thread_local! { static TAKE_STARTUP_POISON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }

#[cfg(test)]
pub(super) fn poison_next_startup() {
    TAKE_STARTUP_POISON.with(|pending| pending.set(true));
}

#[derive(Clone, Copy, Debug)]
pub struct QmmReceipt {
    pub reserved_bytes: usize,
    pub physical_delta_bytes: usize,
    pub device_before_bytes: usize,
    pub device_after_bytes: usize,
    pub hash_seconds: f64,
    pub load_seconds: f64,
    pub pipeline_seconds: f64,
    pub basis_seconds: f64,
    pub image_bytes: usize,
    pub file_sha256: &'static str,
}

pub(super) struct Qmm {
    qmm: Pipeline,
    scan: Pipeline,
    pub(super) receipt: QmmReceipt,
}

fn io_error(e: std::io::Error) -> Error {
    Error::Native(format!("Nanbeige QMM metallib: {e}"))
}

fn authenticate(path: &Path) -> Result<Box<[u8]>> {
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .map_err(io_error)?;
    let info = file.metadata().map_err(io_error)?;
    if !info.is_file() || info.len() != FILE_BYTES as u64 {
        return Err(Error::Unsupported("Nanbeige QMM exact image extent"));
    }
    let mut bytes = vec![0; FILE_BYTES].into_boxed_slice();
    file.read_exact(&mut bytes).map_err(io_error)?;
    if file.read(&mut [0]).map_err(io_error)? != 0
        || format!("{:x}", Sha256::digest(&bytes)) != FILE_SHA
    {
        return Err(Error::Unsupported("Nanbeige QMM exact image digest"));
    }
    Ok(bytes)
}

pub(super) fn aligned(buffer: &RawBuffer) -> bool {
    buffer.respondsToSelector(sel!(gpuAddress))
        && buffer.gpuAddress() != 0
        && buffer.gpuAddress().is_multiple_of(16)
}

pub(super) fn disjoint_exact_ranges<const N: usize>(buffers: [(&RawBuffer, usize); N]) -> bool {
    let mut ranges = [(0, 0); N];
    for (index, (buffer, bytes)) in buffers.into_iter().enumerate() {
        if !aligned(buffer) || buffer.length() != bytes {
            return false;
        }
        let start = buffer.gpuAddress();
        let Some(end) = start.checked_add(bytes as u64) else {
            return false;
        };
        if ranges[..index]
            .iter()
            .any(|&(lo, hi)| start < hi && lo < end)
        {
            return false;
        }
        ranges[index] = (start, end);
    }
    true
}

fn budget(base: usize, extra: usize, limit: usize) -> Result<()> {
    let required = add(base, extra)?;
    if required > limit {
        return Err(Error::Budget { required, limit });
    }
    Ok(())
}

impl Qmm {
    pub(super) fn prepare(
        context: &Context,
        options: Options,
        sizes: &Sizes,
        dense_bytes: usize,
        packed: (&plan::Plan, &[Buffer<u32>]),
        path: &Path,
    ) -> Result<Self> {
        let device = &context.device;
        if !cfg!(target_os = "macos")
            || NSProcessInfo::processInfo()
                .operatingSystemVersion()
                .majorVersion
                != 26
            || options.prefill_chunk_size != 128
            || device.name().to_string() != "Apple M1 Pro"
            || !device.supportsFamily(MTLGPUFamily::Apple7)
            || !context.has_gate_up_compact()
            || !context.has_down_k64()
        {
            return Err(Error::Unsupported("Nanbeige QMM exact capability"));
        }
        let (plan, weights) = packed;
        for block in &plan.blocks {
            for (matrix, k, n) in [
                (block.query, HIDDEN, QUERY_WIDTH),
                (block.attention_out, QUERY_WIDTH, HIDDEN),
                (block.down, INTERMEDIATE, HIDDEN),
            ] {
                if (matrix.layout.columns(), matrix.layout.rows()) != (k, n)
                    || !disjoint_exact_ranges([
                        (&weights[matrix.weights].raw, k * n / 2),
                        (&weights[matrix.scales].raw, k * n / 32),
                        (&weights[matrix.biases].raw, k * n / 32),
                    ])
                {
                    return Err(Error::Unsupported("Nanbeige QMM packed role layout"));
                }
            }
        }
        let maximum = device.maxThreadsPerThreadgroup();
        if maximum.width < 32 || maximum.height < 2 || maximum.depth < 2 {
            return Err(Error::Unsupported("Nanbeige QMM thread geometry"));
        }
        let before = device.currentAllocatedSize();
        let limit = usize::try_from(device.recommendedMaxWorkingSetSize())
            .unwrap_or(usize::MAX)
            .min(options.memory_limit_bytes);
        let logical = add(sizes.memory.admitted_bytes, dense_bytes)?;
        let session = sizes.memory.admitted_bytes - sizes.memory.weights_bytes;
        budget(
            logical,
            FILE_BYTES + STARTUP_RESERVE,
            options.memory_limit_bytes,
        )?;
        budget(add(before, session)?, FILE_BYTES + STARTUP_RESERVE, limit)?;
        let clock = Instant::now();
        let image = authenticate(path)?;
        let hash_seconds = clock.elapsed().as_secs_f64();
        let clock = Instant::now();
        // Metal consumes the validated allocation itself. No second open of the path.
        let data = DispatchData::from_boxed(image);
        let ml = device
            .newLibraryWithData_error(&data)
            .map_err(|e| Error::Native(e.to_string()))?;
        if ml.functionNames().count() != 1 {
            return Err(Error::Unsupported("Nanbeige QMM single-export image"));
        }
        let load_seconds = clock.elapsed().as_secs_f64();
        let clock = Instant::now();
        let scan_source = precise_library_on(
            device,
            concat!(
                include_str!("../nanbeige.metal"),
                "\n",
                include_str!("qmm_scan.metal")
            ),
        )?;
        let pipeline = |library: &ProtocolObject<dyn MTLLibrary>, name: &str| -> Result<Pipeline> {
            let function = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or(Error::Unsupported("Nanbeige QMM exact export"))?;
            let p = device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| Error::Native(e.to_string()))?;
            if p.threadExecutionWidth() != 32
                || p.maxTotalThreadsPerThreadgroup() < 128
                || p.staticThreadgroupMemoryLength() > device.maxThreadgroupMemoryLength()
            {
                return Err(Error::Unsupported(
                    "Nanbeige QMM compiled pipeline geometry",
                ));
            }
            Ok(p)
        };
        let qmm = pipeline(&ml, SYMBOL)?;
        let scan = pipeline(&scan_source, "toklet_qmm_bf16_scan")?;
        let pipeline_seconds = clock.elapsed().as_secs_f64();
        let after = device.currentAllocatedSize();
        let physical_delta = after.saturating_sub(before);
        let reserved = FILE_BYTES.max(physical_delta);
        budget(
            logical,
            add(reserved, STARTUP_RESERVE)?,
            options.memory_limit_bytes,
        )?;
        budget(
            add(before, session)?,
            add(reserved, STARTUP_RESERVE)?,
            limit,
        )?;
        let mut result = Self {
            qmm,
            scan,
            receipt: QmmReceipt {
                reserved_bytes: reserved,
                physical_delta_bytes: physical_delta,
                device_before_bytes: before,
                device_after_bytes: after,
                hash_seconds,
                load_seconds,
                pipeline_seconds,
                basis_seconds: 0.0,
                image_bytes: FILE_BYTES,
                file_sha256: FILE_SHA,
            },
        };
        let clock = Instant::now();
        objc2::rc::autoreleasepool(|_| result.signed_basis(context))?;
        result.receipt.basis_seconds = clock.elapsed().as_secs_f64();
        let after_basis = device.currentAllocatedSize();
        let final_delta = after_basis.saturating_sub(before);
        result.receipt.device_after_bytes = after_basis;
        result.receipt.physical_delta_bytes = final_delta;
        result.receipt.reserved_bytes = FILE_BYTES.max(final_delta);
        budget(
            logical,
            result.receipt.reserved_bytes,
            options.memory_limit_bytes,
        )?;
        budget(add(before, session)?, result.receipt.reserved_bytes, limit)?;
        Ok(result)
    }

    pub(super) fn encode(&self, encoder: &Encoder, buffers: [&RawBuffer; 6], k: usize, n: usize) {
        self.encode_selected(encoder, buffers, k, n, None);
    }

    pub(super) fn encode_selected(
        &self,
        encoder: &Encoder,
        buffers: [&RawBuffer; 6],
        k: usize,
        n: usize,
        bm64: Option<&Pipeline>,
    ) {
        let constants = [k as i32, n as i32, 128i32];
        encoder.setComputePipelineState(bm64.unwrap_or(&self.qmm));
        // SAFETY: pinned packed Q4/contiguous BF16 geometry; the tracked
        // same-device buffers are retained until completion by Context.
        // The PSO copies the three signed 32-bit constants at indices 5–7.
        unsafe {
            for (i, buffer) in buffers[..5].iter().copied().enumerate() {
                encoder.setBuffer_offset_atIndex(Some(buffer), 0, i);
            }
            for (i, value) in constants.iter().enumerate() {
                encoder.setBytes_length_atIndex(NonNull::from(value).cast(), 4, 5 + i);
            }
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n / 32,
                height: if bm64.is_some() { 2 } else { 4 },
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 2,
                depth: 2,
            },
        );
        self.scan_output(encoder, &buffers[4..], 128 * n);
    }

    pub(super) fn scan_output(&self, encoder: &Encoder, buffers: &[&RawBuffer], words: usize) {
        encode_pipeline(
            encoder,
            &self.scan,
            128,
            buffers,
            &[words as u32],
            [words.div_ceil(128), 1],
        );
    }

    fn signed_basis(&self, context: &Context) -> Result<()> {
        const K: usize = 128;
        const N: usize = 64;
        const OUTPUT: usize = 128 * N;
        let coefficient = |row: usize, col: usize| {
            let seed = 11 + 3 * row + 7 * (col / 64);
            let sign = if seed.is_multiple_of(2) { 0 } else { 0x8000 };
            let off = col % 64;
            (
                ((seed + off + off / 16) % 16) as u32,
                sign | (0x3b00 + (seed % 32) as u16),
                (sign ^ 0x8000) | (0x3a80 + (seed % 64) as u16),
            )
        };
        let device = &context.device;
        let mut x = Buffer::<u16>::on_device(device, 128 * K)?;
        let mut packed = Buffer::<u32>::on_device(device, N * K / 8)?;
        let mut scales = Buffer::<u16>::on_device(device, N * K / 64)?;
        let mut biases = Buffer::<u16>::on_device(device, N * K / 64)?;
        let mut outputs = [
            Buffer::<u16>::on_device(device, OUTPUT + 16)?,
            Buffer::<u16>::on_device(device, OUTPUT + 16)?,
            Buffer::<u16>::on_device(device, OUTPUT + 16)?,
        ];
        let mut status = Buffer::<u32>::on_device(device, 2)?;
        if ![
            &x.raw,
            &scales.raw,
            &biases.raw,
            &packed.raw,
            &outputs[0].raw,
            &outputs[1].raw,
            &outputs[2].raw,
        ]
        .iter()
        .all(|raw| aligned(raw))
        {
            return Err(Error::Unsupported("Nanbeige QMM signed basis alignment"));
        }
        status.with_mut(|w| w.copy_from_slice(&[0, u32::MAX]));
        for out in &mut outputs {
            out.with_mut(|w| w.fill(u16::MAX));
        }
        x.with_mut(|w| {
            w.fill(0);
            for i in 0..128 {
                w[i * K + i] = 0x3f80;
            }
        });
        packed.with_mut(|w| {
            w.fill(0);
            for row in 0..N {
                for col in 0..K {
                    let flat = row * K + col;
                    w[flat / 8] |= coefficient(row, col).0 << (4 * (col % 8));
                }
            }
        });
        scales.with_mut(|w| {
            for (i, v) in w.iter_mut().enumerate() {
                *v = coefficient(i / 2, (i % 2) * 64).1;
            }
        });
        biases.with_mut(|w| {
            for (i, v) in w.iter_mut().enumerate() {
                *v = coefficient(i / 2, (i % 2) * 64).2;
            }
        });
        let cmd = context.command()?;
        let encoder = cmd.computeCommandEncoder().ok_or(Error::Completion)?;
        for (op, output) in [Op::GateUpCompact, Op::DownK64].into_iter().zip(&outputs) {
            context.encode(
                &encoder,
                op,
                &[
                    &packed.raw,
                    &scales.raw,
                    &biases.raw,
                    &x.raw,
                    &output.raw,
                    &status.raw,
                ],
                &[128, K as u32, N as u32, 0],
                [N / 32, 4],
            );
        }
        self.encode(
            &encoder,
            [
                &packed.raw,
                &scales.raw,
                &biases.raw,
                &x.raw,
                &outputs[2].raw,
                &status.raw,
            ],
            K,
            N,
        );
        encoder.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        if cmd.status() != MTLCommandBufferStatus::Completed
            || !status.with_slice(|x| x == [0, u32::MAX])
        {
            return Err(Error::Completion);
        }
        #[cfg(test)]
        if TAKE_STARTUP_POISON.with(|pending| pending.replace(false)) {
            outputs[2].with_mut(|w| w[0] ^= 1);
        }
        let exact = outputs.iter().all(|out| {
            out.with_slice(|w| {
                w[OUTPUT..].iter().all(|&x| x == u16::MAX)
                    && w[..OUTPUT].iter().enumerate().all(|(i, &word)| {
                        let (q, scale, bias) = coefficient(i % N, i / N);
                        let value = bf16::from_bits(scale).to_f32() * q as f32
                            + bf16::from_bits(bias).to_f32();
                        let expected = bf16::from_f32(value).to_bits();
                        word == expected
                    })
            })
        });
        if !exact {
            return Err(Error::Unsupported("Nanbeige QMM signed startup parity"));
        }
        Ok(())
    }
}
