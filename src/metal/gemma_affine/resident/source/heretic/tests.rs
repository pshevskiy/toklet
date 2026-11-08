use super::*;
use std::io::Cursor;
const CONFIG: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/gemma4-heretic-metadata-v1/config.json"
));
const HEADER: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/gemma4-heretic-metadata-v1/header.json"
));
fn options() -> LoadOptions {
    LoadOptions {
        container_bytes: uncensored::CONTAINER_BYTES,
        container_sha256: std::array::from_fn(|i| {
            u8::from_str_radix(&uncensored::CONTAINER_SHA256[2 * i..2 * i + 2], 16).unwrap()
        }),
        available_bytes: crate::weights::INITIAL_MEMORY_LIMIT_BYTES,
        modality: ModalityPolicy::TextOnly,
    }
}
fn header_only() -> Cursor<Vec<u8>> {
    Cursor::new([(HEADER.len() as u64).to_le_bytes().as_slice(), HEADER].concat())
}

#[test]
fn pinned_mixed_source_resolves_typed_components_without_authenticating_missing_payload() {
    let mut reader = header_only();
    let mut p = Prepared::read(CONFIG, &mut reader, &options()).unwrap();
    assert_eq!(
        (
            p.inputs.regions.len(),
            p.inputs.matrices.len(),
            p.inputs.vectors.len()
        ),
        (2649, 277, 262)
    );
    assert_eq!(p.inputs.regions.iter().filter(|r| r.text).count(), 1094);
    assert_eq!(p.inputs.regions[p.bf16_region].bytes, 8960 * 1536 * 2);
    assert_eq!(reader.position(), 335752);
    assert!(p.inputs.finish(&options()).is_err());
    assert!(p.inputs.authenticate(&mut reader, &options()).is_err());
    assert!(p.inputs.finish(&options()).is_err());
}

#[test]
fn exact_mixed_roles_and_twelve_affine_shapes_match_the_pinned_recipe() {
    let mut reader = header_only();
    let p = Prepared::read(CONFIG, &mut reader, &options()).unwrap();
    let b = Bindings::heretic(&p.cold).unwrap();
    use crate::mlx::gemma4::bf16::execution::{OperationId, Role};
    let root = b
        .operation_source(OperationId {
            role: Role::PerLayerInputProjection,
            layer: None,
        })
        .unwrap();
    assert!(matches!(
        p.cold.matrices()[root].storage,
        MatrixStorage::Bf16 {
            rows: 8960,
            columns: 1536
        }
    ));
    assert_eq!(
        b.operation_source(OperationId {
            role: Role::OutputHead,
            layer: None
        })
        .unwrap(),
        b.embedding_index()
    );
    let mut shapes = Vec::new();
    for (i, m) in p.cold.matrices().iter().enumerate() {
        assert_eq!(
            b.operation_source(b.operation_identity(i).unwrap())
                .unwrap(),
            i
        );
        if i == b.ple_embedding_index() {
            continue;
        }
        if let MatrixStorage::Affine(l) = m.storage
            && !shapes.contains(&(l.rows(), l.columns()))
        {
            shapes.push((l.rows(), l.columns()));
        }
    }
    shapes.sort_unstable();
    assert_eq!(
        shapes,
        [
            (256, 1536),
            (512, 1536),
            (1536, 256),
            (1536, 2048),
            (1536, 4096),
            (1536, 6144),
            (1536, 12288),
            (2048, 1536),
            (4096, 1536),
            (6144, 1536),
            (12288, 1536),
            (262144, 1536)
        ]
    );
}

#[test]
fn new_source_rejects_header_bounds_pins_and_implicit_modalities_before_payload() {
    for n in [0_u64, u64::MAX, 1024 * 1024 + 1] {
        assert!(Prepared::read(CONFIG, &mut Cursor::new(n.to_le_bytes()), &options()).is_err());
    }
    let mut bad = CONFIG.to_vec();
    bad[0] ^= 1;
    assert!(Prepared::read(&bad, &mut header_only(), &options()).is_err());
    let mut opts = options();
    opts.modality = ModalityPolicy::Reject;
    assert!(Prepared::read(CONFIG, &mut header_only(), &opts).is_err());
    let mut opts = options();
    opts.container_bytes -= 1;
    assert!(Prepared::read(CONFIG, &mut header_only(), &opts).is_err());
}

#[test]
fn raw_bf16_component_needs_authentication_exact_coverage_and_single_consumption() {
    let bytes = [1_u8, 2, 3, 4, 5, 6, 7, 8];
    let (mut inputs, mut reader, opts) = super::super::test_support::stream(&[], &[&bytes]);
    let region = inputs.vectors[0].region;
    let mut raw = [0x55; 8];
    assert!(
        inputs
            .component_read(&mut reader, region, &mut raw)
            .is_err()
    );
    assert_eq!(raw, [0x55; 8]);
    inputs.authenticate(&mut reader, &opts).unwrap();
    assert!(
        inputs
            .component_read(&mut reader, region, &mut raw[..7])
            .is_err()
    );
    assert!(inputs.finish(&opts).is_err());
    inputs
        .component_read(&mut reader, region, &mut raw)
        .unwrap();
    assert_eq!(raw, bytes);
    assert!(
        inputs
            .component_read(&mut reader, region, &mut raw)
            .is_err()
    );
    assert_eq!(inputs.finish(&opts).unwrap().text_components(), 1);
}

#[test]
fn bf16_second_pass_mutation_never_completes_the_source_proof() {
    let bytes = [1_u8, 2, 3, 4, 5, 6, 7, 8];
    let (mut inputs, mut reader, opts) = super::super::test_support::stream(&[], &[&bytes]);
    let region = inputs.vectors[0].region;
    inputs.authenticate(&mut reader, &opts).unwrap();
    reader.get_mut()[inputs.regions[region].offset as usize] ^= 1;
    assert!(
        inputs
            .component_read(&mut reader, region, &mut [0; 8])
            .is_err()
    );
    assert!(!inputs.regions[region].consumed);
    assert!(inputs.finish(&opts).is_err());
}
