//! Private producer-block GeGLU seam; no model/session route selects it.
//!
//! Gate/up/activation/product are contiguous FP32 [rows,width]. Activation
//! remains materialized before the separate FP32 product. Diagnostics contain
//! eight FP32 words/lane (reference intervals and pre-placeholder speculation).
//! This encoder never waits, reads back, initializes status or publishes state.
use super::{
    Error, Result,
    memory::{Buffer, BufferRange, Gpu, validate_buffer_bytes, validate_encode_ranges},
};
use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_foundation::NSString;
use objc2_metal::*;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    io::Read,
    mem::size_of,
    process::{Command, Stdio},
    ptr::NonNull,
};

const RECORDS: usize = 138241;
const TABLE_BYTES: usize = 3_317_784;
const TABLE_WORDS: usize = TABLE_BYTES / 4;
const GUARD: usize = 7;
const SENTINEL: u32 = 0x7fc12345;
const TABLE_SHA: &str = "78dfa84189353b4f9e31cc1a6e6e3a6476ef4818c7548c3c3c06e9227764dcc6";
const CERT_SHA: &str = "6e9a3ae04b87c35d60bf80b064e7b5f448cfb13f678621a79e0b213e38d3a683";
const NATIVE_SHA: &str = "94244856b3b2f29ac0073b73022d126395125d19e6793189d3cf797aab0f6550";
const SOURCE_SHA: &str = "f3d5a9b1f374217f554afe946af7f35e1ce5685bcc723d45c41d8b5552a74e20";
const SHADER_SHA: &str = "02fe4f2c38447f7f482c55c13543ea62ca07e30e1072aa283aa002f81a951671";
const PROFILE: &str = "unchanged-rust-f64-ordered-cubic-host-tanh-f32-activation-f32-product-v1";
const BUILD_PROFILE: &str = "release-opt3-default-fp";
const COMPILER: &str = "rustc 1.90.0 (1159e78c4 2025-09-14) (Homebrew)\nbinary: rustc\ncommit-hash: 1159e78c4747b02ef996e55082b704c09b970588\ncommit-date: 2025-09-14\nhost: aarch64-apple-darwin\nrelease: 1.90.0\nLLVM version: 21.1.1";
const HOST_UNAME: &str = "Darwin Dmitrijs-MacBook-Pro-2.local 25.6.0 Darwin Kernel Version 25.6.0: Fri Jul 31 19:18:49 PDT 2026; root:xnu-12377.161.14~5/RELEASE_ARM64_T6000 arm64";
const HOST_VERSION: &str =
    "ProductName:\t\tmacOS\nProductVersion:\t\t26.6.2\nBuildVersion:\t\t25G83";
/// Provenance, not approval of this integration or any full block.
const ARTIFACT_REVIEW_SHA: &str =
    "f523adfffa42035e8badd4f192d2df50e9931807c6a1a69baa29a80c3b093cf3";

/// Trusted build markers are explicit qualification inputs. Ordinary builds
/// without them fail closed at cold table admission; no default route changes.
/// The caller/build receipt must bind these markers to the actual compiler and
/// flags. This module cannot discover its own compiler from a running binary.
fn validate_compiled_reference() -> Result<()> {
    if option_env!("TOKLET_GELU_CERTIFIED_RUSTC_VV") != Some(COMPILER)
        || option_env!("TOKLET_GELU_CERTIFIED_BUILD_PROFILE") != Some(BUILD_PROFILE)
        || cfg!(debug_assertions)
    {
        return Err(Error::Unsupported(
            "Gemma GELU requires explicit certified release build binding",
        ));
    }
    if format!("{:x}", Sha256::digest(include_bytes!("../cpu/gelu.rs"))) != SOURCE_SHA {
        return Err(Error::Unsupported(
            "Gemma GELU CPU reference source changed",
        ));
    }
    Ok(())
}
fn host_output(program: &str, args: &[&str]) -> Result<String> {
    let mut child = Command::new(program)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|_| Error::Unsupported("Gemma GELU native host binding unavailable"))?;
    let mut bytes = Vec::with_capacity(1025);
    let read = child
        .stdout
        .take()
        .unwrap()
        .take(1025)
        .read_to_end(&mut bytes);
    if read.is_err() || bytes.len() > 1024 {
        let _ = child.kill();
        let _ = child.wait();
        return Err(Error::Unsupported(
            "Gemma GELU native host binding exceeds bound",
        ));
    }
    if !child
        .wait()
        .map_err(|_| Error::Unsupported("Gemma GELU native host wait failed"))?
        .success()
    {
        return Err(Error::Unsupported("Gemma GELU native host binding failed"));
    }
    String::from_utf8(bytes)
        .map(|s| s.trim().to_owned())
        .map_err(|_| Error::Unsupported("Gemma GELU native host text differs"))
}
/// Derive the mathematical platform key from immutable recorded provenance.
/// Remove only the known nodename field. Preserve the remaining bytes exactly;
/// this is not a whitespace-normalizing parser for a live `uname -a` response.
fn math_host_identity(recorded_uname: &str) -> Option<String> {
    let (system, node_and_rest) = recorded_uname.split_once(' ')?;
    let (node, rest) = node_and_rest.split_once(' ')?;
    if system.is_empty()
        || node.is_empty()
        || rest.is_empty()
        || system.bytes().any(|b| b.is_ascii_whitespace())
        || node.bytes().any(|b| b.is_ascii_whitespace())
        || rest.starts_with(' ')
        || rest.ends_with(' ')
        || recorded_uname.bytes().any(|b| b.is_ascii_control())
    {
        return None;
    }
    Some(format!("{system} {rest}"))
}
/// Runtime identity excludes the mutable nodename at its source (`-srvm`).
/// Certificate material retains its original complete hostname and exact hash.
fn validate_native_host_identity(math_uname: &str, version: &str) -> Result<()> {
    let qualified = math_host_identity(HOST_UNAME).ok_or(Error::Unsupported(
        "Gemma GELU qualified host identity malformed",
    ))?;
    if math_uname != qualified || version != HOST_VERSION {
        return Err(Error::Unsupported(
            "Gemma GELU native host/OS requires requalification",
        ));
    }
    Ok(())
}
fn matches(raw: &[u8], len: usize, sha: &str) -> bool {
    raw.len() == len && format!("{:x}", Sha256::digest(raw)) == sha
}
/// Bounded external payload and immutable small evidence. No artifact paths or
/// multi-MiB include_bytes are part of ordinary generation. Caller may release
/// these CPU slices after admission; the table retains one shared GPU payload.
pub(super) struct ApprovedBytes<'a> {
    pub table: &'a [u8],
    pub certificate: &'a [u8],
    pub native_verification: &'a [u8],
    pub artifact_review: &'a [u8],
}
fn validate_material(b: &ApprovedBytes<'_>) -> Result<()> {
    if !matches(b.table, TABLE_BYTES, TABLE_SHA)
        || !matches(b.certificate, 2309, CERT_SHA)
        || !matches(b.native_verification, 635, NATIVE_SHA)
        || !matches(b.artifact_review, 13881, ARTIFACT_REVIEW_SHA)
    {
        return Err(Error::Options(
            "Gemma GELU approved payload/evidence identity differs",
        ));
    }
    let certificate: Value = serde_json::from_slice(b.certificate)
        .map_err(|_| Error::Options("Gemma GELU certificate JSON"))?;
    let verified: Value = serde_json::from_slice(b.native_verification)
        .map_err(|_| Error::Options("Gemma GELU native verification JSON"))?;
    let review: Value = serde_json::from_slice(b.artifact_review)
        .map_err(|_| Error::Options("Gemma GELU artifact review JSON"))?;
    if certificate["format"] != "toklet-gelu-full-finite-certificate-v1"
        || certificate["profile"] != PROFILE
        || certificate["source_sha256"] != SOURCE_SHA
        || certificate["compiler_binding"] != COMPILER
        || certificate["host_uname"] != HOST_UNAME
        || certificate["host_sw_vers"] != HOST_VERSION
        || certificate["records"].as_u64() != Some(RECORDS as u64)
        || certificate["unique_points"].as_u64() != Some(4_278_190_080)
        || certificate["table"]["sha256"] != TABLE_SHA
        || verified["passed"].as_bool() != Some(true)
        || verified["certificate_sha256"] != CERT_SHA
        || verified["table_sha256"] != TABLE_SHA
        || verified["unique_points"].as_u64() != Some(4_278_190_080)
        || review["approved"].as_bool() != Some(true)
    {
        return Err(Error::Options("Gemma GELU qualification profile differs"));
    }
    // Hashes bind every field/version/mode, not just the header; no arbitrary
    // raw table can construct this owner. Small explicit structural checks keep
    // its fixed six-u32 ABI visible even if trusted hashes are later revised.
    for record in b.table.chunks_exact(24) {
        let mode = u32::from_le_bytes(record[16..20].try_into().unwrap());
        let version = u32::from_le_bytes(record[20..24].try_into().unwrap());
        if mode > 3 || version != 1 {
            return Err(Error::Options("Gemma GELU table record ABI"));
        }
    }
    Ok(())
}

pub(super) struct CertifiedTable {
    buffer: Buffer<u32>,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
}
impl CertifiedTable {
    pub(super) fn planned_owned_bytes() -> usize {
        (TABLE_WORDS + 2 * GUARD) * 4 + size_of::<Self>()
    }
    pub(super) fn from_approved(
        gpu: &Gpu,
        b: &ApprovedBytes<'_>,
        available_owned_bytes: usize,
    ) -> Result<Self> {
        let required = Self::planned_owned_bytes();
        if available_owned_bytes < required {
            return Err(Error::Budget {
                required,
                limit: available_owned_bytes,
            });
        }
        validate_material(b)?;
        validate_compiled_reference()?;
        validate_native_host_identity(
            &host_output("/usr/bin/uname", &["-srvm"])?,
            &host_output("/usr/bin/sw_vers", &[])?,
        )?;
        validate_buffer_bytes((TABLE_WORDS + 2 * GUARD) * 4, gpu.device.maxBufferLength())?;
        let mut buffer = Buffer::new(gpu, TABLE_WORDS + 2 * GUARD)?;
        buffer.with_mut(|out| {
            out.fill(SENTINEL);
            for (out, bytes) in out[GUARD..GUARD + TABLE_WORDS]
                .iter_mut()
                .zip(b.table.chunks_exact(4))
            {
                *out = u32::from_le_bytes(bytes.try_into().unwrap());
            }
        });
        Ok(Self {
            buffer,
            device: gpu.device.clone(),
        })
    }
    pub(super) fn owned_bytes(&self) -> usize {
        // Both admitted variants own this same header and seven-word guards.
        // No parent GPU payload or second table owner is retained for v2.
        self.buffer.len * 4 + size_of::<Self>()
    }
    fn table_words(&self) -> Result<usize> {
        let words = match self.buffer.len {
            n if n == TABLE_WORDS + 2 * GUARD => TABLE_WORDS,
            #[cfg(test)]
            n if n == core::TABLE_BYTES / 4 + 2 * GUARD => core::TABLE_BYTES / 4,
            _ => {
                return Err(Error::Options(
                    "Gemma GELU table allocation is not a sealed variant",
                ));
            }
        };
        // The accepted fixed sizes above make multiplication bounded. Reject
        // a relabelled owner length before cold graph allocation or encoding.
        if self.buffer.raw.length() != self.buffer.len * 4 {
            return Err(Error::Options("Gemma GELU table physical length differs"));
        }
        Ok(words)
    }
    /// Cold block preflight, before any earlier operator is encoded. No CPU
    /// buffer view, submission or completion wait is needed for device identity.
    pub(super) fn validate_device(&self, gpu: &Gpu) -> Result<()> {
        self.table_words()?;
        if !std::ptr::eq(&*self.device, &*gpu.device)
            || !std::ptr::eq(&*self.buffer.raw.device(), &*gpu.device)
        {
            return Err(Error::Options("Gemma GELU table/device differs"));
        }
        Ok(())
    }
    fn range(&self) -> Result<BufferRange<'_>> {
        BufferRange::new(&self.buffer, GUARD, self.table_words()?)
    }
    /// Cold table remains immutable. Caller invokes this only at a checked
    /// whole-command completion boundary, before any speculative publication.
    /// A failed completion/guard poisons the block owner and requires recreation;
    /// no CPU view is taken for an uncompleted/error command.
    pub(super) fn check_completed_guards(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<()> {
        if command.status() != MTLCommandBufferStatus::Completed
            || !std::ptr::eq(&*command.device(), &*self.device)
        {
            return Err(Error::Completion);
        }
        let table_words = self.table_words()?;
        if !self.buffer.with_slice(|s| {
            s[..GUARD]
                .iter()
                .chain(&s[GUARD + table_words..])
                .all(|&x| x == SENTINEL)
        }) {
            return Err(Error::Options("Gemma GELU table guard changed"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub(super) struct Shape {
    pub rows: usize,
    pub width: usize,
}
impl Shape {
    fn lengths(self) -> Result<(usize, usize)> {
        if !(1..=32).contains(&self.rows) || !(1..=12288).contains(&self.width) {
            return Err(Error::Options("Gemma GELU shape outside1..32x1..12288"));
        }
        let values = self
            .rows
            .checked_mul(self.width)
            .ok_or(Error::Options("Gemma GELU shape overflow"))?;
        let intervals = values
            .checked_mul(8)
            .ok_or(Error::Options("Gemma GELU diagnostic overflow"))?;
        Ok((values, intervals))
    }
}
pub(super) struct DeviceBuffers<'a> {
    pub gate: BufferRange<'a>,
    pub up: BufferRange<'a>,
    pub activated: BufferRange<'a>,
    pub product: BufferRange<'a>,
    pub status: BufferRange<'a>,
    pub lane_flags: BufferRange<'a>,
    pub intervals: BufferRange<'a>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CompletionStatus {
    CandidateReady,
    NeedsReference,
    Fatal(u32),
}
impl CompletionStatus {
    pub(super) fn parse(word: u32) -> Result<Self> {
        if word & !31 != 0 {
            return Err(Error::Options("Gemma GELU unknown completion status"));
        }
        let fatal = word & 23;
        Ok(if fatal != 0 {
            Self::Fatal(fatal)
        } else if word & 8 != 0 {
            Self::NeedsReference
        } else {
            Self::CandidateReady
        })
    }
}
pub(super) struct Pipelines {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    width: usize,
    table_words: usize,
}
impl Pipelines {
    pub(super) fn new(gpu: &Gpu) -> Result<Self> {
        if !matches(include_bytes!("gemma_gelu.metal"), 9176, SHADER_SHA) {
            return Err(Error::Options("Gemma GELU qualified shader changed"));
        }
        Self::compile(gpu, include_str!("gemma_gelu.metal"), TABLE_WORDS)
    }
    // One selection at construction, shared ABI and concrete encode thereafter.
    // The existing compiler-policy reservation covers one selected pipeline.
    fn compile(gpu: &Gpu, source: &str, table_words: usize) -> Result<Self> {
        autoreleasepool(|_| {
            let options = MTLCompileOptions::new();
            options.setLanguageVersion(MTLLanguageVersion::Version2_3);
            options.setMathMode(MTLMathMode::Safe);
            options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
            if options.languageVersion() != MTLLanguageVersion::Version2_3
                || options.mathMode() != MTLMathMode::Safe
                || options.mathFloatingPointFunctions() != MTLMathFloatingPointFunctions::Precise
            {
                return Err(Error::Unsupported("Gemma GELU precise math unavailable"));
            }
            let library = gpu
                .device
                .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
                .map_err(|e| Error::Native(e.to_string()))?;
            let function = library
                .newFunctionWithName(&NSString::from_str("gated_gelu"))
                .ok_or_else(|| Error::Native("Gemma GELU kernel absent".into()))?;
            let pipeline = gpu
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| Error::Native(e.to_string()))?;
            let width = pipeline.threadExecutionWidth();
            if width == 0
                || width > pipeline.maxTotalThreadsPerThreadgroup()
                || width > u32::MAX as usize
            {
                return Err(Error::Unsupported("Gemma GELU launch width unavailable"));
            }
            Ok(Self {
                device: gpu.device.clone(),
                pipeline,
                width,
                table_words,
            })
        })
    }
    /// Cold graph preflight also uses this before earlier dependent operators.
    /// Same-device fixed payload length identifies the sealed table variant.
    pub(super) fn validate_table(&self, table: &CertifiedTable) -> Result<()> {
        if self.table_words != table.table_words()? || !std::ptr::eq(&*self.device, &*table.device)
        {
            return Err(Error::Options(
                "Gemma GELU table/pipeline variant or device differs",
            ));
        }
        Ok(())
    }
    /// Encode only. Caller zeroes shared transaction status, retains all inputs,
    /// buffers/table and command until checked terminal completion, then checks
    /// guards and parses fatal status before uncertainty. NEEDS_REFERENCE means
    /// discard/replay the WHOLE block/transaction from its retained checkpoint;
    /// lane patching after dependent operators is not a valid recovery.
    /// Failure/cancellation cannot advance committed KV/history or caller output.
    pub(super) fn encode(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        table: &CertifiedTable,
        b: &DeviceBuffers<'_>,
        shape: Shape,
    ) -> Result<()> {
        self.encode_inner(command, table, b, shape, false)
    }
    #[cfg(test)]
    pub(super) fn encode_forced_reference(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        table: &CertifiedTable,
        b: &DeviceBuffers<'_>,
        shape: Shape,
    ) -> Result<()> {
        self.encode_inner(command, table, b, shape, true)
    }
    fn encode_inner(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        table: &CertifiedTable,
        b: &DeviceBuffers<'_>,
        shape: Shape,
        force_reference: bool,
    ) -> Result<()> {
        let (values, intervals) = shape.lengths()?;
        let table_range = table.range()?;
        self.validate_table(table)?;
        let ranges = [
            b.gate,
            b.up,
            table_range,
            b.activated,
            b.product,
            b.status,
            b.lane_flags,
            b.intervals,
        ];
        let lengths = [
            values,
            values,
            self.table_words,
            values,
            values,
            1,
            values,
            intervals,
        ];
        for (r, n) in ranges.iter().zip(lengths) {
            if r.len != n
                || r.offset
                    .checked_add(r.len)
                    .and_then(|n| n.checked_mul(4))
                    .is_none_or(|n| n > r.raw.length())
                || r.offset.checked_mul(4).is_none_or(|n| n >= r.raw.length())
                || r.raw.storageMode() != MTLStorageMode::Shared
                || r.raw.hazardTrackingMode() != MTLHazardTrackingMode::Tracked
            {
                return Err(Error::Options(
                    "Gemma GELU borrowed range/length contract differs",
                ));
            }
        }
        validate_encode_ranges(command, &self.device, &ranges, 3)?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| Error::Native("Gemma GELU encoder absent".into()))?;
        encoder.setComputePipelineState(&self.pipeline);
        let count = values as u32;
        let force = u32::from(force_reference);
        // SAFETY: fixed byte-identical qualified shader ABI, checked initialized
        // shared tracked u32 ranges, exact lengths/offsets and no writable overlap
        // (including table/status). Command/device/retention preflight passed.
        // setBytes copies scalars; caller owns synchronization and publication.
        unsafe {
            for (slot, r) in [b.gate, b.up, b.activated, b.product].iter().enumerate() {
                encoder.setBuffer_offset_atIndex(Some(r.raw), r.offset * 4, slot);
            }
            encoder.setBytes_length_atIndex(NonNull::from(&count).cast(), 4, 4);
            encoder.setBuffer_offset_atIndex(Some(b.status.raw), b.status.offset * 4, 5);
            encoder.setBuffer_offset_atIndex(Some(table_range.raw), table_range.offset * 4, 6);
            encoder.setBuffer_offset_atIndex(Some(b.lane_flags.raw), b.lane_flags.offset * 4, 7);
            encoder.setBuffer_offset_atIndex(Some(b.intervals.raw), b.intervals.offset * 4, 8);
            encoder.setBytes_length_atIndex(NonNull::from(&force).cast(), 4, 9);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: values.div_ceil(self.width),
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
// Only the isolated deep-chain qualification consumer selects the core overlay.
// Remove this gate only with a qualified ordinary graph consumer.
#[cfg(test)]
pub(super) mod core;
#[cfg(test)]
mod tests;
