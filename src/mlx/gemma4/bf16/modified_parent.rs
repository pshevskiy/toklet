//! Sealed test-only parent identity; never substitutes the official payload pin.
use super::*;

pub(crate) const REPOSITORY: &str = "p-e-w/gemma-4-E2B-it-heretic-ara";
pub(crate) const REVISION: &str = "c9a1d4c031981f14d86eeb0c7d87de7fafd34513";
pub(crate) const CONTAINER_SHA256: &str =
    "15605c16e479a8dc53928dd4b521a4ead9fe66222dc14d1ce392384e999a544b";
pub(crate) const CONFIG: &[u8] =
    include_bytes!("../../../../tests/fixtures/gemma-heretic-bf16-metadata-v1/config.json");

pub(super) fn inspect(
    config: &[u8],
    header: &[u8],
    bytes: u64,
    policy: ModalityPolicy,
) -> Result<TextPlan, Error> {
    if policy != ModalityPolicy::TextOnly
        || bytes != CONTAINER_BYTES
        || config != CONFIG
        || header.len() != HEADER_BYTES
        || format!("{:x}", Sha256::digest(header)) != HEADER_SHA256
    {
        return Err(invalid("modified BF16 parent metadata differs"));
    }
    let mut document = crate::mlx::qwen::parse_json(config)?;
    if document["text_config"]
        .as_object_mut()
        .and_then(|text| text.remove("moe_intermediate_size"))
        != Some(serde_json::Value::Null)
    {
        return Err(invalid("modified BF16 inactive alias differs"));
    }
    // Only the known disabled alias is removed for semantic validation.
    // Payload authentication still uses this parent's distinct complete digest.
    layout(document, header, bytes)
}

#[test]
fn parent_and_official_pins_are_disjoint_with_identical_selected_geometry() {
    let header =
        include_bytes!("../../../../tests/fixtures/gemma-unquantized-header-v1/header.json");
    let original =
        include_bytes!("../../../../tests/fixtures/gemma-unquantized-header-v1/config.json");
    assert_eq!(CONFIG.len(), 4989);
    assert_eq!(
        format!("{:x}", Sha256::digest(CONFIG)),
        "09f31bccdb5185282ee0ac6a41586958b39523c94cee0205dac6d39407169edb"
    );
    let parent = inspect(CONFIG, header, CONTAINER_BYTES, ModalityPolicy::TextOnly).unwrap();
    let official =
        super::inspect(original, header, CONTAINER_BYTES, ModalityPolicy::TextOnly).unwrap();
    assert!(super::inspect(CONFIG, header, CONTAINER_BYTES, ModalityPolicy::TextOnly).is_err());
    assert!(inspect(original, header, CONTAINER_BYTES, ModalityPolicy::TextOnly).is_err());
    assert!(inspect(CONFIG, header, CONTAINER_BYTES, ModalityPolicy::Reject).is_err());
    assert!(
        inspect(
            CONFIG,
            header,
            CONTAINER_BYTES - 1,
            ModalityPolicy::TextOnly
        )
        .is_err()
    );
    let mut corrupt = header.to_vec();
    corrupt[100] ^= 1;
    assert!(inspect(CONFIG, &corrupt, CONTAINER_BYTES, ModalityPolicy::TextOnly).is_err());
    assert_eq!(REPOSITORY, "p-e-w/gemma-4-E2B-it-heretic-ara");
    assert_eq!(REVISION.len(), 40);
    assert_eq!(parent.text_tensors().len(), official.text_tensors().len());
    for (name, tensor) in parent.text_tensors() {
        let original = &official.text_tensors()[name];
        assert_eq!(tensor.dtype, original.dtype);
        assert_eq!(tensor.shape, original.shape);
        assert_eq!(tensor.range, original.range);
    }
    assert_eq!(
        parent.resident_tensor_bytes(),
        official.resident_tensor_bytes()
    );
    assert_ne!(CONTAINER_SHA256, rows::CONTAINER_SHA256);
}
