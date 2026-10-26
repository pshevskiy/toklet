//! Private synthetic storage streams; cannot construct public E2bWeights.
use super::*;
use std::io::Cursor;

pub(in crate::metal::gemma_affine::resident) fn stream(
    matrices: &[(Layout, [&[u8]; 3])],
    vectors: &[&[u8]],
) -> (Inputs, Cursor<Vec<u8>>, LoadOptions) {
    let prefix = 2_u64.to_le_bytes();
    let header = b"{}".to_vec();
    let mut data = [prefix.as_slice(), &header].concat();
    let mut regions = Vec::new();
    let mut append = |raw: &[u8], text: bool| {
        let index = regions.len();
        regions.push(Region {
            offset: data.len() as u64,
            bytes: raw.len(),
            text,
            digest: [0; 32],
            consumed: false,
        });
        data.extend_from_slice(raw);
        index
    };
    append(b"excluded-synthetic-modality", false);
    let mut matrix_inputs = Vec::new();
    for &(layout, arrays) in matrices {
        matrix_inputs.push(MatrixInput {
            layout,
            regions: arrays.map(|raw| append(raw, true)),
        });
    }
    let vector_inputs = vectors
        .iter()
        .map(|raw| {
            assert!(raw.len().is_multiple_of(2));
            VectorInput {
                values: raw.len() / 2,
                region: append(raw, true),
            }
        })
        .collect();
    let options = LoadOptions {
        container_bytes: data.len() as u64,
        container_sha256: Sha256::digest(&data).into(),
        available_bytes: 512 * 1024 * 1024,
        modality: ModalityPolicy::TextOnly,
    };
    let inputs = Inputs {
        prefix,
        header,
        regions,
        matrices: matrix_inputs,
        vectors: vector_inputs,
        config_sha256: Sha256::digest(b"synthetic-config").into(),
        index_sha256: Sha256::digest(b"synthetic-index").into(),
        authenticated: false,
    };
    let mut reader = Cursor::new(data);
    reader.set_position(10);
    (inputs, reader, options)
}
