//! Cold historical evidence for one exact assembled model/codec owner pair.
//! No session, filesystem, GPU or preference-storage work occurs here.
use super::complete::SourceReceipt;
use super::*;
use crate::{
    app_chat::qwen3_content::QwenCodecContentIdentity,
    mlx::qwen::QwenSourceContentIdentity,
    weights::content::{Manifest, qwen3_manifest, smol_manifest},
};
use serde::Serialize;

const REPOSITORY: &str = "mlx-community/Qwen3-0.6B-8bit";
const REVISION: &str = "11de96878523501bcaa86104e3c186de07ff9068";

// These are engine-owned semantic identifiers, never caller-provided strings.
// Bump planning when incompatible factory caps, modes, layout/admission or
// transition accounting changes; wire schema version is independent. A version
// only partitions untrusted preferences, never waives fresh admission.
#[derive(Clone, Copy, Serialize)]
enum BackendPolicy {
    #[serde(rename = "metal_native_dense_v1")]
    MetalNativeDense,
}
#[derive(Clone, Copy, Serialize)]
enum LoadPolicy {
    #[serde(rename = "qwen3_audited_v1")]
    Qwen3Audited,
    #[serde(rename = "smol_witness_audited_v1")]
    SmolWitnessAudited,
}
#[derive(Clone, Copy, Serialize)]
enum TextPolicy {
    #[serde(rename = "qwen3-plain-nonthinking-v1")]
    Qwen3PlainNonthinking,
    #[serde(rename = "smol-chatml-v1")]
    SmolChatml,
}
#[derive(Clone, Copy, Serialize)]
enum PlanningPolicy {
    #[serde(rename = "native_mlx_planning_v1")]
    NativeMlx,
}
#[derive(Serialize)]
struct PreferenceNamespace {
    version: u32,
    manifest_algorithm: &'static str,
    manifest_sha256: String,
    backend: BackendPolicy,
    load: LoadPolicy,
    text: TextPolicy,
    planning: PlanningPolicy,
}
#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum Role {
    SourceMetadataConsumed,
    CodecConfigVerified,
    SourceBytesDeliveredToAssembledModel,
    ReferenceContainerDigestAnchor,
    TokenizerParserConsumed,
    FixedTextPolicyVerified,
    AncillaryVerified,
}
#[derive(Serialize)]
struct Asset {
    name: &'static str,
    bytes: u64,
    sha256: String,
    roles: &'static [Role],
}
#[derive(Serialize)]
struct IdentityReport {
    schema_version: u32,
    report_kind: &'static str,
    repository: &'static str,
    revision: &'static str,
    preference_namespace: PreferenceNamespace,
    assets: [Asset; 6],
    // Capture-time correlation only, deliberately outside the stable namespace.
    model_instance: u64,
    loaded_position_limit: usize,
    source_container_physical_ranges: usize,
    source_container_verified_reads: usize,
    model_assembled: bool,
    codec_constructed: bool,
    exact_model_pair: bool,
    source_receipts_equal: bool,
    historical_evidence: bool,
    current_disk_identity_evaluated: bool,
    resident_buffer_identity_evaluated: bool,
    qualification_evaluated: bool,
    whole_app_device_fit_evaluated: bool,
    preferences_require_fresh_admission: bool,
}

#[derive(Clone, Copy)]
enum IdentityProfile {
    Qwen3,
    Smol,
}
impl IdentityProfile {
    fn manifest(self) -> Result<Manifest<'static>> {
        match self {
            Self::Qwen3 => qwen3_manifest(),
            Self::Smol => smol_manifest(),
        }
        .map_err(|e| fail(ENGINE_ERROR, e.to_string()))
    }
    fn repository(self) -> &'static str {
        match self {
            Self::Qwen3 => REPOSITORY,
            Self::Smol => "mlx-community/SmolLM2-135M-Instruct-8bit",
        }
    }
    fn revision(self) -> &'static str {
        match self {
            Self::Qwen3 => REVISION,
            Self::Smol => "0f0d9b8218915bc34d401e1a340b8c049d300d5e",
        }
    }
    fn text(self) -> &'static str {
        match self {
            Self::Qwen3 => "qwen3-plain-nonthinking-v1",
            Self::Smol => "smol-chatml-v1",
        }
    }
}
#[derive(Clone)]
struct Evidence {
    profile: IdentityProfile,
    manifest: [u8; 32],
    digests: [[u8; 32]; 6],
    container_bytes: u64,
    ranges: usize,
    reads: usize,
    text_policy: &'static str,
}
fn same_source(a: &QwenSourceContentIdentity, b: &QwenSourceContentIdentity) -> Result<()> {
    if a != b {
        return Err(fail(INVALID, "model and codec source receipts differ"));
    }
    Ok(())
}
fn evidence(
    source: &QwenSourceContentIdentity,
    codec: &QwenCodecContentIdentity,
) -> Result<Evidence> {
    same_source(source, codec.source())?;
    Ok(Evidence {
        profile: IdentityProfile::Qwen3,
        manifest: codec.complete_manifest_sha256(),
        digests: [
            source.config_sha256(),
            source.container_sha256(),
            source.index_sha256(),
            codec.tokenizer_sha256(),
            codec.tokenizer_config_policy_sha256(),
            codec.ancillary_special_map_sha256(),
        ],
        container_bytes: source.container_bytes(),
        ranges: source.container_physical_ranges(),
        reads: source.verified_container_reads(),
        text_policy: codec.text_policy(),
    })
}
fn smol_evidence(
    source: &crate::mlx::qwen::SourceContentIdentity,
    codec: &crate::app_chat::content::CodecContentIdentity,
) -> Result<Evidence> {
    if source != codec.source() {
        return Err(fail(INVALID, "model and codec source receipts differ"));
    }
    let manifest = codec.complete_manifest_sha256().ok_or_else(|| {
        fail(
            UNSUPPORTED,
            "legacy Smol codec has no complete ancillary evidence",
        )
    })?;
    let ancillary = codec
        .ancillary_special_map_sha256()
        .ok_or_else(|| fail(UNSUPPORTED, "Smol codec lacks ancillary evidence"))?;
    Ok(Evidence {
        profile: IdentityProfile::Smol,
        manifest,
        digests: [
            source.config_sha256(),
            source.reference_container_sha256(),
            source.index_sha256(),
            codec.tokenizer_sha256(),
            codec.tokenizer_config_sha256(),
            ancillary,
        ],
        container_bytes: source.container_bytes(),
        ranges: source.physical_ranges(),
        reads: source.verified_reads(),
        text_policy: "smol-chatml-v1",
    })
}

fn hex(value: [u8; 32]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(64);
    for byte in value {
        text.push(DIGITS[(byte >> 4) as usize] as char);
        text.push(DIGITS[(byte & 15) as usize] as char);
    }
    text
}
fn build(evidence: Evidence, instance: u64, loaded: usize) -> Result<IdentityReport> {
    let profile = evidence.profile;
    let manifest = profile.manifest()?;
    let files = manifest.files();
    // This fixed report identifies the same repository/revision bytes as the
    // source expectation; copied presentation constants cannot silently drift.
    let presented = Manifest::new(profile.repository(), profile.revision(), files)
        .map_err(|e| fail(ENGINE_ERROR, e.to_string()))?;
    if files.len() != 6
        || presented.identity() != manifest.identity()
        || evidence.manifest != manifest.identity()
        || files
            .iter()
            .zip(evidence.digests)
            .any(|(f, d)| f.sha256 != d)
        || evidence.container_bytes != files[1].bytes
        || !(3..=crate::weights::content::MAX_RANGES).contains(&evidence.ranges)
        || evidence.reads < evidence.ranges
        || evidence.text_policy != profile.text()
        || instance == 0
        || loaded == 0
        || (matches!(profile, IdentityProfile::Qwen3) && loaded > 512)
        || (matches!(profile, IdentityProfile::Smol) && evidence.ranges != 694)
    {
        return Err(fail(
            UNSUPPORTED,
            "unsupported complete model/codec identity evidence",
        ));
    }
    let source_roles: &[Role] = match profile {
        IdentityProfile::Qwen3 => &[Role::SourceBytesDeliveredToAssembledModel],
        IdentityProfile::Smol => &[
            Role::SourceBytesDeliveredToAssembledModel,
            Role::ReferenceContainerDigestAnchor,
        ],
    };
    let roles: [&'static [Role]; 6] = [
        &[Role::SourceMetadataConsumed, Role::CodecConfigVerified],
        source_roles,
        &[Role::SourceMetadataConsumed],
        &[Role::TokenizerParserConsumed],
        &[Role::FixedTextPolicyVerified],
        &[Role::AncillaryVerified],
    ];
    let assets = std::array::from_fn(|i| Asset {
        name: files[i].name,
        bytes: files[i].bytes,
        sha256: hex(evidence.digests[i]),
        roles: roles[i],
    });
    Ok(IdentityReport {
        schema_version: 1,
        report_kind: "accepted_model_chat_identity",
        repository: profile.repository(),
        revision: profile.revision(),
        preference_namespace: PreferenceNamespace {
            version: 1,
            manifest_algorithm: "toklet-full-manifest-v1",
            manifest_sha256: hex(manifest.identity()),
            backend: BackendPolicy::MetalNativeDense,
            load: match profile {
                IdentityProfile::Qwen3 => LoadPolicy::Qwen3Audited,
                IdentityProfile::Smol => LoadPolicy::SmolWitnessAudited,
            },
            text: match profile {
                IdentityProfile::Qwen3 => TextPolicy::Qwen3PlainNonthinking,
                IdentityProfile::Smol => TextPolicy::SmolChatml,
            },
            planning: PlanningPolicy::NativeMlx,
        },
        assets,
        model_instance: instance,
        loaded_position_limit: loaded,
        source_container_physical_ranges: evidence.ranges,
        source_container_verified_reads: evidence.reads,
        model_assembled: true,
        codec_constructed: true,
        exact_model_pair: true,
        source_receipts_equal: true,
        historical_evidence: true,
        current_disk_identity_evaluated: false,
        resident_buffer_identity_evaluated: false,
        qualification_evaluated: false,
        whole_app_device_fit_evaluated: false,
        preferences_require_fresh_admission: true,
    })
}
fn same_model(model: &ModelHandle, chat: &ChatHandle) -> Result<()> {
    let retained = chat
        ._model
        .as_ref()
        .ok_or_else(|| fail(UNSUPPORTED, "chat has no retained audited model"))?;
    if !Rc::ptr_eq(&model.model, retained) {
        return Err(fail(INVALID, "model and chat retain different allocations"));
    }
    Ok(())
}

/// Capture historical identity for the exact assembled model/codec pair.
/// Snapshot copy/release use the existing compatibility APIs and JSON allowance.
/// No session is borrowed: an unrelated busy session is neither queried nor
/// mutated. All calls still obey the creating-thread and no-concurrent-use rules.
/// # Safety
/// Follow parent foreign pointer, live owner and disjoint output contracts.
/// A valid out is cleared before validation; original owners remain unchanged.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn toklet_model_chat_identity_create(
    model: *const ModelHandle,
    chat: *const ChatHandle,
    out: *mut *mut CompatibilityHandle,
    error: *mut NativeError,
) -> u32 {
    unsafe {
        boundary(error, || {
            pointer(out)?;
            out.write(ptr::null_mut());
            pointer(model)?;
            pointer(chat)?;
            let model = &*model;
            let chat = &*chat;
            owner(model.owner)?;
            owner(chat.owner)?;
            same_model(model, chat)?;
            let (source, loaded) = SourceReceipt::model(model)?;
            let evidence = match source {
                SourceReceipt::Qwen3(source) => evidence(
                    &source,
                    chat.codec
                        .qwen3_content_identity()
                        .ok_or_else(|| fail(UNSUPPORTED, "chat has no audited Qwen3 codec"))?,
                )?,
                SourceReceipt::Smol(source) => smol_evidence(
                    &source,
                    chat.codec
                        .content_identity()
                        .ok_or_else(|| fail(UNSUPPORTED, "chat has no audited Smol codec"))?,
                )?,
            };
            let report = build(evidence, model.instance, loaded)?;
            out.write(compatibility::capture(&report)?);
            Ok(())
        })
    }
}
#[cfg(test)]
mod tests;

#[cfg(test)]
mod ffi_tests;
