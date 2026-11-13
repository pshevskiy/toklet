//! Bounded diagnostic manifest identity, separate from tensor or lineage proof.
use super::{ArtifactPins, BODY_BYTES, REPOSITORY, REVISION, invalid};
use crate::weights::Error;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

pub const MAX_MANIFEST_BYTES: usize = 64 * 1024;
pub const FORMAT: &str = "toklet-qwen35-2b-affine-child-v1";
const SOURCE_CONFIG: &str = "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4";
const SOURCE_HEADER: &str = "ed13c70a5f6268600b730dbc480262f640decf50083cddb23a107f201b6c0060";
const SOURCE_CONTAINER: &str = "f04765ef799190d41131132d43d69624e9ed6eac7b7cdd2a65a2eb3e266878ab";

fn keys(value: &Value, wanted: &[&str]) -> Result<(), Error> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid("diagnostic manifest object required"))?;
    if object.len() != wanted.len() || !wanted.iter().all(|key| object.contains_key(*key)) {
        return Err(invalid("diagnostic manifest fields differ"));
    }
    Ok(())
}

fn digest(value: &Value) -> Result<[u8; 32], Error> {
    let text = value
        .as_str()
        .filter(|v| v.len() == 64)
        .ok_or_else(|| invalid("diagnostic manifest SHA256 required"))?;
    if !text
        .bytes()
        .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        return Err(invalid("diagnostic manifest SHA256 spelling differs"));
    }
    let mut output = [0; 32];
    for (out, pair) in output.iter_mut().zip(text.as_bytes().chunks_exact(2)) {
        let digit = |c: u8| if c <= b'9' { c - b'0' } else { c - b'a' + 10 };
        *out = digit(pair[0]) * 16 + digit(pair[1]);
    }
    if output == [0; 32] {
        return Err(invalid("placeholder diagnostic manifest SHA256 rejected"));
    }
    Ok(output)
}

fn integer(value: &Value) -> Result<u64, Error> {
    value
        .as_u64()
        .ok_or_else(|| invalid("diagnostic manifest unsigned integer required"))
}

fn size(value: &Value) -> Result<usize, Error> {
    usize::try_from(integer(value)?)
        .map_err(|_| invalid("diagnostic manifest size exceeds address space"))
}

/// Bind a caller-selected manifest digest and extract the exact child identities.
/// This does not read a tensor, replay either referenced certificate, establish
/// publisher ancestry, or grant model, context, codec, quality or fit admission.
/// The tensor loader separately authenticates the complete child before escape.
pub fn pins(manifest: &[u8], expected_sha256: [u8; 32]) -> Result<ArtifactPins, Error> {
    if manifest.is_empty()
        || manifest.len() > MAX_MANIFEST_BYTES
        || expected_sha256 == [0; 32]
        || <[u8; 32]>::from(Sha256::digest(manifest)) != expected_sha256
    {
        return Err(invalid(
            "bounded externally pinned diagnostic manifest required",
        ));
    }
    let value = crate::mlx::qwen::parse_json(manifest)?;
    keys(
        &value,
        &[
            "format",
            "passed",
            "artifact_pins",
            "source",
            "lineage_certificate_sha256",
            "conversion_manifest_sha256",
            "storage",
            "qualification",
        ],
    )?;
    if value["format"].as_str() != Some(FORMAT) || value["passed"].as_bool() != Some(true) {
        return Err(invalid("successful diagnostic child manifest required"));
    }
    let artifact = &value["artifact_pins"];
    keys(
        artifact,
        &[
            "config_bytes",
            "config_sha256",
            "header_bytes",
            "header_sha256",
            "container_bytes",
            "container_sha256",
        ],
    )?;
    if value["source"]
        != json!({
            "repository": REPOSITORY,
            "revision": REVISION,
            "container_sha256": SOURCE_CONTAINER,
            "config_sha256": SOURCE_CONFIG,
            "header_sha256": SOURCE_HEADER,
        })
    {
        return Err(invalid("selected diagnostic source declaration differs"));
    }
    digest(&value["lineage_certificate_sha256"])?;
    digest(&value["conversion_manifest_sha256"])?;
    let storage = &value["storage"];
    keys(
        storage,
        &[
            "bits",
            "group_size",
            "mode",
            "logical_tensors",
            "physical_tensors",
            "body_bytes",
        ],
    )?;
    for (key, wanted) in [
        ("bits", 4),
        ("group_size", 64),
        ("logical_tensors", 320),
        ("physical_tensors", 694),
        ("body_bytes", BODY_BYTES as u64),
    ] {
        if integer(&storage[key])? != wanted {
            return Err(invalid("diagnostic child storage declaration differs"));
        }
    }
    if storage["mode"].as_str() != Some("affine")
        || value["qualification"]
            != json!({
                "diagnostic_only": true,
                "model_numerics_quality_fit_performance_adoption_qualified": false,
            })
    {
        return Err(invalid(
            "diagnostic-only qualification declaration required",
        ));
    }
    let result = ArtifactPins {
        config_bytes: size(&artifact["config_bytes"])?,
        config_sha256: digest(&artifact["config_sha256"])?,
        header_bytes: size(&artifact["header_bytes"])?,
        header_sha256: digest(&artifact["header_sha256"])?,
        container_bytes: integer(&artifact["container_bytes"])?,
        container_sha256: digest(&artifact["container_sha256"])?,
    };
    result.validate()?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        // Synthetic identities exercise cold metadata only, never a production
        // child or a claim of authenticated tensor/certificate contents.
        json!({
            "format": FORMAT, "passed": true,
            "artifact_pins": {
                "config_bytes": 1, "config_sha256": "11".repeat(32),
                "header_bytes": 8, "header_sha256": "22".repeat(32),
                "container_bytes": BODY_BYTES + 16, "container_sha256": "33".repeat(32),
            },
            "source": {"repository":REPOSITORY,"revision":REVISION,
                "container_sha256":SOURCE_CONTAINER,"config_sha256":SOURCE_CONFIG,"header_sha256":SOURCE_HEADER},
            "lineage_certificate_sha256": "44".repeat(32),
            "conversion_manifest_sha256": "55".repeat(32),
            "storage": {"bits":4,"group_size":64,"mode":"affine", "logical_tensors":320,
                "physical_tensors":694,"body_bytes":BODY_BYTES},
            "qualification": {"diagnostic_only":true,
                "model_numerics_quality_fit_performance_adoption_qualified":false},
        })
    }
    fn parse(value: &Value) -> Result<ArtifactPins, Error> {
        let bytes = serde_json::to_vec(value).unwrap();
        pins(&bytes, Sha256::digest(&bytes).into())
    }

    #[test]
    fn bounded_manifest_identity_extracts_only_diagnostic_child_pins() {
        let value = fixture();
        let parsed = parse(&value).unwrap();
        assert_eq!(parsed.config_bytes, 1);
        assert_eq!(parsed.header_bytes, 8);
        assert_eq!(parsed.container_bytes, BODY_BYTES as u64 + 16);
        assert_eq!(parsed.config_sha256, [0x11; 32]);
        assert_eq!(parsed.header_sha256, [0x22; 32]);
        assert_eq!(parsed.container_sha256, [0x33; 32]);
        let bytes = serde_json::to_vec(&value).unwrap();
        assert!(pins(&bytes, [0x10; 32]).is_err());
        assert!(pins(&bytes, [0; 32]).is_err());
        for bytes in [Vec::new(), vec![b' '; MAX_MANIFEST_BYTES + 1]] {
            assert!(pins(&bytes, Sha256::digest(&bytes).into()).is_err());
        }
    }

    #[test]
    fn self_rehashed_schema_source_precision_and_qualification_mutations_reject() {
        let mutations = [
            ("/format", json!("other")),
            ("/passed", json!(false)),
            ("/source/revision", json!("current")),
            ("/source/container_sha256", json!("ff".repeat(32))),
            ("/source/config_sha256", json!("ff".repeat(32))),
            ("/source/header_sha256", json!("ff".repeat(32))),
            ("/storage/group_size", json!(32)),
            ("/storage/bits", json!(4.0)),
            ("/storage/body_bytes", json!(BODY_BYTES - 1)),
            ("/storage/physical_tensors", json!(693)),
            ("/artifact_pins/config_bytes", json!(u64::MAX)),
            ("/artifact_pins/header_bytes", json!(512 * 1024 + 1)),
            ("/artifact_pins/container_bytes", json!(BODY_BYTES + 17)),
            ("/artifact_pins/header_sha256", json!("00".repeat(32))),
            ("/artifact_pins/container_sha256", json!("AA".repeat(32))),
            ("/lineage_certificate_sha256", json!("44")),
            ("/conversion_manifest_sha256", json!("00".repeat(32))),
            ("/qualification/diagnostic_only", json!(false)),
            (
                "/qualification/model_numerics_quality_fit_performance_adoption_qualified",
                json!(true),
            ),
        ];
        for (pointer, replacement) in mutations {
            let mut changed = fixture();
            *changed.pointer_mut(pointer).unwrap() = replacement;
            assert!(parse(&changed).is_err(), "{pointer}");
        }
        for pointer in [
            "",
            "/artifact_pins",
            "/storage",
            "/source",
            "/qualification",
        ] {
            let mut changed = fixture();
            changed
                .pointer_mut(pointer)
                .unwrap()
                .as_object_mut()
                .unwrap()
                .insert("unbound".into(), json!(true));
            assert!(parse(&changed).is_err(), "extra {pointer}");
        }
        let mut missing = fixture();
        missing.as_object_mut().unwrap().remove("source");
        assert!(parse(&missing).is_err());
    }

    #[test]
    fn duplicate_root_or_nested_fields_reject_even_with_matching_external_hash() {
        let original = serde_json::to_string(&fixture()).unwrap();
        for (needle, replacement) in [
            ("\"passed\":true", "\"passed\":true,\"passed\":true"),
            ("\"bits\":4", "\"bits\":4,\"bits\":4"),
        ] {
            assert_eq!(original.matches(needle).count(), 1);
            let changed = original.replace(needle, replacement);
            assert!(
                pins(
                    changed.as_bytes(),
                    Sha256::digest(changed.as_bytes()).into()
                )
                .is_err()
            );
        }
    }
}
