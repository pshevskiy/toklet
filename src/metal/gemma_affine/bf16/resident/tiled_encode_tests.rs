use super::*;
use project::bf16_encode::Bf16Tile;
use std::io::Cursor;

#[test]
#[ignore = "explicit original resident Matrix tiled factory and whole-source maximum binding"]
fn original_tile_factory_keeps_last_tile_maximum_in_shared_proof() {
    autoreleasepool(|_| {
        let context = Bf16Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        let mut source = Vec::with_capacity(8193 * 64 * 2);
        for row in 0..8193 {
            let value = if row < 8192 { 0.5_f32 } else { 2_f32 };
            for _ in 0..64 {
                source.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
            }
        }
        let mut charged = 0;
        let matrix = upload_matrix(
            &context.0.gpu,
            &mut Cursor::new(source),
            8193,
            64,
            &mut charged,
            16 * 1024 * 1024,
        )
        .unwrap();
        assert_eq!(matrix.maximum, 2.);
        assert!(matrix.single_tile(&context).is_err());
        let borrowed = matrix.borrowed_tiled(&context).unwrap();
        assert_eq!(borrowed.tile_count(), 2);
        assert_eq!(borrowed.admission_tile().maximum_bits(), 2_f32.to_bits());
        let whole = Bf16Tile::new(&context.0.gpu, 8192, 64, &matrix.tiles[0], 2.).unwrap();
        let first_only = Bf16Tile::new(&context.0.gpu, 8192, 64, &matrix.tiles[0], 0.5).unwrap();
        assert_eq!(
            borrowed.admission_tile().max_input_bits(),
            whole.max_input_bits()
        );
        assert_eq!(
            borrowed.admission_tile().l1_input_bits(),
            whole.l1_input_bits()
        );
        assert!(whole.max_input_bits() < first_only.max_input_bits());
        assert!(whole.l1_input_bits() < first_only.l1_input_bits());
        println!(
            "BF16_TILED_FACTORY {{\"tile_count\":2,\"whole_maximum\":2,\"first_tile_maximum\":0.5,\"weight_copy_in_view\":false}}"
        );
    });
}
