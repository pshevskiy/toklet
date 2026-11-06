//! The official owner authenticates its shard as a whole and retains file-backed
//! PLE. Its evidence differs from the physical-range delivery of dense MLX.
use super::*;
use crate::{
    app_chat::gemma4::{CHAT_TEMPLATE, GENERATION_CONFIG, OFFICIAL_TOKENIZER_CONFIG, TOKENIZER},
    metal::gemma_affine::OfficialBf16,
    mlx::gemma4::bf16::{CONFIG_BYTES, CONFIG_SHA256, CONTAINER_BYTES, rows::CONTAINER_SHA256},
    weights::content::{ExpectedFile, canonical_manifest_identity},
};

#[derive(Serialize)]
struct OfficialNamespace {
    version: u32,
    manifest_algorithm: &'static str,
    manifest_sha256: String,
    backend: &'static str,
    load: &'static str,
    text: &'static str,
    planning: &'static str,
}

#[derive(Serialize)]
struct OfficialReport {
    schema_version: u32,
    report_kind: &'static str,
    repository: &'static str,
    revision: &'static str,
    preference_namespace: OfficialNamespace,
    assets: [Asset; 6],
    model_instance: u64,
    loaded_position_limit: usize,
    whole_source_sha256_verified: bool,
    file_backed_ple: bool,
    model_assembled: bool,
    codec_constructed: bool,
    exact_model_pair: bool,
    historical_evidence: bool,
    current_disk_identity_evaluated: bool,
    resident_buffer_identity_evaluated: bool,
    qualification_evaluated: bool,
    whole_app_device_fit_evaluated: bool,
    preferences_require_fresh_admission: bool,
}

fn asset(
    (name, bytes, sha256): (&'static str, usize, &'static str),
    roles: &'static [Role],
) -> Asset {
    Asset {
        name,
        bytes: bytes as u64,
        sha256: sha256.into(),
        roles,
    }
}

fn expected(asset: &Asset) -> Result<ExpectedFile<'static>> {
    let bad_pin = || fail(ENGINE_ERROR, "invalid built-in official Gemma digest");
    let mut sha256 = [0; 32];
    if asset.sha256.len() != 64 || !asset.sha256.is_ascii() {
        return Err(bad_pin());
    }
    for (i, byte) in sha256.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&asset.sha256[2 * i..2 * i + 2], 16).map_err(|_| bad_pin())?;
    }
    Ok(ExpectedFile {
        name: asset.name,
        bytes: asset.bytes,
        sha256,
    })
}

pub(super) fn capture(instance: u64, loaded: usize) -> Result<*mut CompatibilityHandle> {
    let assets = [
        asset(CHAT_TEMPLATE, &[Role::FixedTextPolicyVerified]),
        asset(
            ("config.json", CONFIG_BYTES, CONFIG_SHA256),
            &[Role::SourceMetadataConsumed, Role::CodecConfigVerified],
        ),
        asset(GENERATION_CONFIG, &[Role::FixedTextPolicyVerified]),
        Asset {
            name: "model.safetensors",
            bytes: CONTAINER_BYTES,
            sha256: CONTAINER_SHA256.into(),
            roles: &[
                Role::WholeSourceAuthenticated,
                Role::FileBackedPleAuthenticated,
            ],
        },
        asset(TOKENIZER, &[Role::TokenizerParserConsumed]),
        asset(OFFICIAL_TOKENIZER_CONFIG, &[Role::AncillaryVerified]),
    ];
    let [a, b, c, d, e, f] = assets.each_ref().map(expected);
    let repository = OfficialBf16::REPOSITORY;
    let revision = OfficialBf16::REVISION;
    let digest =
        canonical_manifest_identity(repository, revision, [a?, b?, c?, d?, e?, f?].into_iter());
    compatibility::capture(&OfficialReport {
        schema_version: 1,
        report_kind: "accepted_official_gemma_bf16_identity",
        repository,
        revision,
        preference_namespace: OfficialNamespace {
            version: 1,
            manifest_algorithm: "toklet-full-manifest-v1",
            manifest_sha256: hex(digest),
            backend: "metal_official_gemma_bf16_v1",
            load: "gemma_bf16_audited_v1",
            text: "gemma4-basic-nonthinking-v1",
            planning: "official_gemma_bf16_planning_v1",
        },
        assets,
        model_instance: instance,
        loaded_position_limit: loaded,
        whole_source_sha256_verified: true,
        file_backed_ple: true,
        model_assembled: true,
        codec_constructed: true,
        exact_model_pair: true,
        historical_evidence: true,
        current_disk_identity_evaluated: false,
        resident_buffer_identity_evaluated: false,
        qualification_evaluated: false,
        whole_app_device_fit_evaluated: false,
        preferences_require_fresh_admission: true,
    })
}
