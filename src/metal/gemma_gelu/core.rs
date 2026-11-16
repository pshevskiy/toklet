//! Explicit sealed core-overlay material for the isolated deep-chain consumer.
//! Parent material/default constructors retain v1. This module is test-gated
//! until a real graph consumer and its combined qualification remove the gate.
use super::*;

pub(in crate::metal) const RECORDS: usize = 228355;
pub(in crate::metal) const TABLE_BYTES: usize = 5_480_520;
const TABLE_WORDS: usize = TABLE_BYTES / 4;
pub(in crate::metal) const TABLE_SHA: &str =
    "f034904775faedd628f92f2b324354310a512f13284d9f316dfc2265ab1907f8";
pub(in crate::metal) const CERT_SHA: &str =
    "6c9defb81f62a3974154f1425219c1ebb3b101f67176e060fdfcad463faabaf8";
pub(in crate::metal) const NATIVE_SHA: &str =
    "c9849a45efa6362f62698c737610f8ebf2081affbd2a450b7951a0fc8b48b58e";
const SHADER_SHA: &str = "7ad303748504528c1e6f8243c4ba6c710af21f3907b2df3442eb97f7d9ad2d21";
const ADMISSION_SHA: &str = "c1b6d94920cadc123ab66caef7719c815b8bb247dff7664e3ff7d15b18dd91cc";
// Formal artifact approval supplies provenance for bounded coupled tests.
// Production integration/full-model qualification remain separate. Never accept
// a caller-selected review hash or edit the original parent certificate.
pub(in crate::metal) const CORE_REVIEW_SHA: &str =
    "0b36c8c2e12948aebc65eb888ef760aea0586a6495787ce1f36cff2c6993ac2d";
pub(in crate::metal) const CORE_REVIEW_BYTES: usize = 4047;

/// One combined CPU payload, plus small sealed evidence. The parent table is
/// borrowed from the combined prefix; a second parent GPU buffer is not made.
pub(in crate::metal) struct ApprovedCoreBytes<'a> {
    pub table: &'a [u8],
    pub certificate: &'a [u8],
    pub native_verification: &'a [u8],
    pub artifact_review: &'a [u8],
    pub parent_certificate: &'a [u8],
    pub parent_native_verification: &'a [u8],
    pub parent_artifact_review: &'a [u8],
}

pub(super) fn validate_material(b: &ApprovedCoreBytes<'_>) -> Result<()> {
    if !matches(b.table, TABLE_BYTES, TABLE_SHA)
        || !matches(b.certificate, 5608, CERT_SHA)
        || !matches(b.native_verification, 988, NATIVE_SHA)
    {
        return Err(Error::Options("Gemma GELU core approved identity differs"));
    }
    // The original validator, old sealed full hostname and all v1 hashes stay
    // unchanged. Its parent table is the existing borrowed combined prefix.
    super::validate_material(&ApprovedBytes {
        table: &b.table[..super::TABLE_BYTES],
        certificate: b.parent_certificate,
        native_verification: b.parent_native_verification,
        artifact_review: b.parent_artifact_review,
    })?;
    if !matches(b.artifact_review, CORE_REVIEW_BYTES, CORE_REVIEW_SHA) {
        return Err(Error::Options("Gemma GELU core artifact approval differs"));
    }
    let c: Value = serde_json::from_slice(b.certificate)
        .map_err(|_| Error::Options("Gemma GELU core certificate JSON"))?;
    let native: Value = serde_json::from_slice(b.native_verification)
        .map_err(|_| Error::Options("Gemma GELU core native verification JSON"))?;
    let review: Value = serde_json::from_slice(b.artifact_review)
        .map_err(|_| Error::Options("Gemma GELU core artifact review JSON"))?;
    let key = math_host_identity(HOST_UNAME)
        .ok_or(Error::Options("Gemma GELU qualified math host malformed"))?;
    if c["format"] != "toklet-gelu-finer-affine-core-certificate-v2"
        || c["profile"] != PROFILE
        || c["source_sha256"] != SOURCE_SHA
        || c["compiler_binding"] != COMPILER
        || c["host_sw_vers"] != HOST_VERSION
        || c["math_host_policy"] != "uname-srvm-exact-excluding-nodename-v1"
        || c["math_host_identity"] != key
        || c["host_uname"]
            .as_str()
            .and_then(math_host_identity)
            .as_deref()
            != Some(key.as_str())
        || c["admission_protocol_sha256"] != ADMISSION_SHA
        || c["fresh_full_finite_sweep"] != false
        || c["parent_prefix_byteexact"] != true
        || c["table"]["sha256"] != TABLE_SHA
        || c["table"]["bytes"].as_u64() != Some(TABLE_BYTES as u64)
        || c["parent"]["certificate"]["sha256"] != super::CERT_SHA
        || c["parent"]["independent_verification"]["sha256"] != super::NATIVE_SHA
        || c["parent"]["artifact_review"]["sha256"] != ARTIFACT_REVIEW_SHA
        || c["parent"]["table"]["sha256"] != super::TABLE_SHA
        || native["format"] != "toklet-gelu-core-composed-independent-verification-v2"
        || native["passed"] != true
        || native["diagnostic_only"] != false
        || native["native_admission_receipt"] != true
        || native["certificate_sha256"] != CERT_SHA
        || native["table_sha256"] != TABLE_SHA
        || native["parent_prefix_byteexact"] != true
        || native["outside_mapping_and_disjoint_coverage_verified"] != true
        || review["approved"] != true
        || review["sealed_material"]["table_sha256"] != TABLE_SHA
        || review["sealed_material"]["table_bytes"].as_u64() != Some(TABLE_BYTES as u64)
        || review["sealed_material"]["certificate_sha256"] != CERT_SHA
        || review["sealed_material"]["ordinary_verification_sha256"] != NATIVE_SHA
    {
        return Err(Error::Options("Gemma GELU core qualification differs"));
    }
    for (name, count) in [
        ("records", RECORDS as u64),
        ("record_bytes", 24),
        ("record_version", 1),
        ("parent_records", 138241),
        ("core_records", 90114),
        ("core_records_per_sign", 45057),
        ("core_lo_bits", 0x3b800000),
        ("core_hi_bits", 0x41000000),
        ("core_shift", 11),
        ("fresh_core_unique_points", 184549378),
        ("inherited_outside_unique_points", 4093640702),
        ("unique_runtime_finite_points", 4278190080),
        ("negative_fine_overlap_records_byteexact", 8193),
    ] {
        if c[name].as_u64() != Some(count) {
            return Err(Error::Options("Gemma GELU core coverage/layout differs"));
        }
    }
    for name in [
        "fresh_core_unique_points",
        "inherited_outside_unique_points",
        "unique_runtime_finite_points",
    ] {
        if native[name] != c[name] {
            return Err(Error::Options("Gemma GELU core native coverage differs"));
        }
    }
    for record in b.table.chunks_exact(24) {
        let mode = u32::from_le_bytes(record[16..20].try_into().unwrap());
        let version = u32::from_le_bytes(record[20..24].try_into().unwrap());
        if mode > 3 || version != 1 {
            return Err(Error::Options("Gemma GELU core table record ABI"));
        }
    }
    for fine in 0..8193usize {
        let magnitude = 0x40000000 + fine * 2048;
        let child = 138241 + 45057 + (magnitude - 0x3b800000) / 2048;
        let parent = 130048 + fine;
        if b.table[child * 24..(child + 1) * 24] != b.table[parent * 24..(parent + 1) * 24] {
            return Err(Error::Options("Gemma GELU core fine overlap differs"));
        }
    }
    Ok(())
}
impl CertifiedTable {
    pub(in crate::metal) fn planned_core_owned_bytes() -> usize {
        (TABLE_WORDS + 2 * GUARD) * 4 + size_of::<Self>()
    }
    pub(in crate::metal) fn from_approved_core(
        gpu: &Gpu,
        b: &ApprovedCoreBytes<'_>,
        available_owned_bytes: usize,
    ) -> Result<Self> {
        let required = Self::planned_core_owned_bytes();
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
}
impl Pipelines {
    pub(in crate::metal) fn new_core(gpu: &Gpu) -> Result<Self> {
        if !matches(include_bytes!("core.metal"), 10345, SHADER_SHA) {
            return Err(Error::Options("Gemma GELU qualified core shader changed"));
        }
        Self::compile(gpu, include_str!("core.metal"), TABLE_WORDS)
    }
}
