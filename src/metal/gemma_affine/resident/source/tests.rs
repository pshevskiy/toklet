use super::*;
use std::io::{Cursor, ErrorKind};

fn fixture() -> (Inputs, Cursor<Vec<u8>>, LoadOptions) {
    fixture_width(4)
}
fn fixture_width(bits: u8) -> (Inputs, Cursor<Vec<u8>>, LoadOptions) {
    let layout = Layout::new(3, 64, bits, 64, MetadataDtype::BF16).unwrap();
    let codes = vec![0_u8; layout.weight_bytes()];
    let scales = [0_u8; 6];
    let biases = [0_u8; 6];
    test_support::stream(
        &[(layout, [&codes, &scales, &biases])],
        &[&[0, 128, 128, 63]],
    )
}
fn read_complete_matrix(inputs: &mut Inputs, reader: &mut (impl Read + Seek)) -> Result<()> {
    let bits = inputs.matrices[0].layout.bits();
    let mut source = inputs.matrix_read(0)?;
    for (first, rows) in [(0, 2), (2, 1)] {
        let layout = Layout::new(rows, 64, bits, 64, MetadataDtype::BF16).unwrap();
        source.fill(
            reader,
            first,
            layout,
            [
                &mut vec![0; layout.weight_bytes()],
                &mut vec![0; rows * 2],
                &mut vec![0; rows * 2],
            ],
        )?;
    }
    source.finish(inputs)
}
#[test]
fn changed_last_affine8_chunk_cannot_complete_authentication() {
    let (mut inputs, mut reader, options) = fixture_width(8);
    inputs.authenticate(&mut reader, &options).unwrap();
    let code = inputs.matrices[0].regions[0];
    let region = &inputs.regions[code];
    reader.get_mut()[region.offset as usize + region.bytes - 1] ^= 1;
    assert!(read_complete_matrix(&mut inputs, &mut reader).is_err());
    assert!(!inputs.regions[code].consumed);
    assert!(inputs.finish(&options).is_err());
}
#[test]
fn exact_full_anchor_then_complete_selected_consumption_is_required_for_identity() {
    let (mut inputs, mut reader, options) = fixture();
    assert!(inputs.finish(&options).is_err());
    assert!(inputs.matrix_read(0).is_err());
    inputs.authenticate(&mut reader, &options).unwrap();
    assert!(inputs.authenticate(&mut reader, &options).is_err());
    assert!(inputs.finish(&options).is_err());
    read_complete_matrix(&mut inputs, &mut reader).unwrap();
    assert!(inputs.matrix_read(0).is_err());
    assert!(inputs.finish(&options).is_err());
    let mut vector = [0; 4];
    inputs.vector_read(&mut reader, 0, &mut vector).unwrap();
    assert_eq!(vector, [0, 128, 128, 63]);
    assert!(inputs.vector_read(&mut reader, 0, &mut vector).is_err());
    let identity = inputs.finish(&options).unwrap();
    assert_eq!(identity.container_sha256, options.container_sha256);
    assert_eq!(identity.text_components, 4);
}
#[test]
fn changed_excluded_bytes_fail_the_historical_anchor_and_late_eof_keeps_typed_io() {
    let (mut inputs, mut reader, options) = fixture();
    reader.get_mut()[10] ^= 1;
    assert!(matches!(
        inputs.authenticate(&mut reader, &options),
        Err(Error::Options("Gemma source container identity differs"))
    ));
    let (mut inputs, mut reader, options) = fixture();
    reader.get_mut().pop();
    assert!(
        matches!(inputs.authenticate(&mut reader, &options), Err(Error::Weights(crate::weights::Error::Io { source, .. })) if source.kind()==ErrorKind::UnexpectedEof)
    );
    assert!(inputs.finish(&options).is_err());
}
#[test]
fn changed_selected_bytes_in_second_pass_never_complete_consumption() {
    let (mut inputs, mut reader, options) = fixture();
    inputs.authenticate(&mut reader, &options).unwrap();
    let offset = inputs.regions[inputs.matrices[0].regions[0]].offset as usize;
    reader.get_mut()[offset] ^= 1;
    assert!(read_complete_matrix(&mut inputs, &mut reader).is_err());
    assert!(!inputs.regions[inputs.matrices[0].regions[0]].consumed);
    assert!(inputs.finish(&options).is_err());
    let (mut inputs, mut reader, options) = fixture();
    inputs.authenticate(&mut reader, &options).unwrap();
    let offset = inputs.regions[inputs.vectors[0].region].offset as usize;
    reader.get_mut()[offset] ^= 1;
    assert!(inputs.vector_read(&mut reader, 0, &mut [0; 4]).is_err());
    assert!(!inputs.regions[inputs.vectors[0].region].consumed);
}
#[test]
fn wrong_seek_and_noncontiguous_or_duplicate_chunks_fail_before_identity() {
    struct WrongSeek(Cursor<Vec<u8>>);
    impl Read for WrongSeek {
        fn read(&mut self, raw: &mut [u8]) -> std::io::Result<usize> {
            self.0.read(raw)
        }
    }
    impl Seek for WrongSeek {
        fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
            self.0.seek(position).map(|n| n + 1)
        }
    }
    let (mut inputs, mut reader, options) = fixture();
    inputs.authenticate(&mut reader, &options).unwrap();
    assert!(
        inputs
            .vector_read(&mut WrongSeek(reader.clone()), 0, &mut [0; 4])
            .is_err()
    );
    let mut source = inputs.matrix_read(0).unwrap();
    let layout = Layout::new(1, 64, 4, 64, MetadataDtype::BF16).unwrap();
    assert!(
        source
            .fill(
                &mut reader,
                1,
                layout,
                [&mut [0; 32], &mut [0; 2], &mut [0; 2]]
            )
            .is_err()
    );
    source
        .fill(
            &mut reader,
            0,
            layout,
            [&mut [0; 32], &mut [0; 2], &mut [0; 2]],
        )
        .unwrap();
    assert!(
        source
            .fill(
                &mut reader,
                0,
                layout,
                [&mut [0; 32], &mut [0; 2], &mut [0; 2]]
            )
            .is_err()
    );
    assert!(source.finish(&mut inputs).is_err());
    assert!(inputs.finish(&options).is_err());
}
#[test]
fn original_io_object_survives_typed_error_wrapping() {
    #[derive(Debug)]
    struct Original;
    impl std::fmt::Display for Original {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("original source failure")
        }
    }
    impl std::error::Error for Original {}
    struct Failed;
    impl Read for Failed {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other(Original))
        }
    }
    let (mut inputs, _, options) = fixture();
    match inputs.authenticate(&mut Failed, &options) {
        Err(Error::Weights(crate::weights::Error::Io { source, .. })) => {
            assert!(source.get_ref().unwrap().is::<Original>())
        }
        _ => panic!("original typed IO error replaced"),
    }
}
