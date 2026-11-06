use super::*;
use std::{
    fs::OpenOptions,
    os::unix::fs::{FileExt, OpenOptionsExt},
};

fn official_plan() -> TextPlan {
    use crate::mlx::gemma4::{ModalityPolicy, bf16};
    bf16::inspect(
        include_bytes!("../../../../../tests/fixtures/gemma-unquantized-header-v1/config.json"),
        include_bytes!("../../../../../tests/fixtures/gemma-unquantized-header-v1/header.json"),
        bf16::CONTAINER_BYTES,
        ModalityPolicy::TextOnly,
    )
    .unwrap()
}

#[test]
fn storage_plan_charges_exact_tiles_and_widened_vectors() {
    let plan = official_plan();
    let mut matrix_tiles = 0;
    let mut vector_values = 0;
    for tensor in plan.text_tensors().values() {
        match *tensor.shape.as_slice() {
            [rows, _] if !std::ptr::eq(tensor, plan.per_layer_embedding()) => {
                matrix_tiles += rows.div_ceil(TILE_ROWS);
            }
            [values] => vector_values += values,
            _ => {}
        }
    }
    let expected = plan.resident_tensor_bytes()
        + vector_values * 2
        + matrix_tiles * (GUARD * 4 + size_of::<Buffer<u32>>())
        + MATRIX_SLOTS * size_of::<Option<Matrix>>()
        + VECTOR_SLOTS * size_of::<Vec<f32>>()
        + size_of::<Bf16Weights>()
        - size_of::<Source>();
    assert_eq!(plan_storage(&plan, usize::MAX).unwrap(), expected);
    assert!(plan_storage(&plan, 1024).is_err());
}

#[test]
#[ignore = "explicit official BF16 resident Metal storage and tied-table witness"]
fn official_bf16_upload_authenticates_and_retains_original_projection_rows() {
    let path = std::path::PathBuf::from(std::env::var_os("TOKLET_GEMMA_BF16_MODEL").unwrap());
    let config = std::fs::read(path.join("config.json")).unwrap();
    let open = || {
        OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(path.join("model.safetensors"))
            .unwrap()
    };
    let plan = official_plan();
    let source_minimum = Prepared::new(open(), &config, usize::MAX)
        .unwrap()
        .minimum_source_memory()
        .unwrap();
    let context = Bf16Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
    let budget = plan_storage(&plan, context.0.gpu.device.maxBufferLength()).unwrap()
        + CONTEXT_ALLOWANCE_BYTES
        + VECTOR_STAGE_BYTES
        + source_minimum.load_bytes;
    assert!(matches!(
        Bf16Weights::load_text_only(&context, open(), &config, budget - 1),
        Err(Error::Budget { required, limit }) if required == budget && limit == budget - 1
    ));
    let owner = Bf16Weights::load_text_only(&context, open(), &config, budget).unwrap();
    let device = context.device_name();
    drop(context);
    assert_eq!(owner.device_name(), device);
    assert_eq!((owner.matrix_count(), owner.vector_count()), (277, 262));
    assert_eq!(owner.report().admitted_load_bytes, budget);
    assert_eq!(owner.report().maximum_abs_coefficient, 1.2109375);
    let file = open();
    let ids = [0, 8191, 8192, 262143, 0];
    let mut output = vec![0.; ids.len() * 1536];
    owner.gather_embeddings(&ids, &mut output).unwrap();
    let mut raw = vec![0u8; 1536 * 2];
    let embedding = &plan.text_tensors()["model.language_model.embed_tokens.weight"];
    for (&id, row) in ids.iter().zip(output.chunks_exact(1536)) {
        let offset = plan.payload_offset() + embedding.range.start + id as usize * raw.len();
        file.read_exact_at(&mut raw, offset as u64).unwrap();
        check_widened(&raw, row);
    }
    let mut unchanged = [7.];
    assert!(owner.gather_embeddings(&[0], &mut unchanged).is_err());
    assert_eq!(unchanged, [7.]);

    let matrix_tensors = plan
        .text_tensors()
        .iter()
        .filter(|(_, t)| t.shape.len() == 2);
    for ((name, tensor), matrix) in matrix_tensors.zip(&owner.matrices) {
        let Some(matrix) = matrix else {
            assert!(std::ptr::eq(tensor, plan.per_layer_embedding()));
            continue;
        };
        assert_eq!(tensor.shape, [matrix.rows, matrix.columns]);
        let data = plan.payload_offset() + tensor.range.start;
        for (tile_index, tile) in matrix.tiles.iter().enumerate() {
            let start_row = tile_index * TILE_ROWS;
            let rows = (matrix.rows - start_row).min(TILE_ROWS);
            let row_words = matrix.columns / 2;
            tile.with_slice(|words| {
                assert!(
                    words[rows * row_words..]
                        .iter()
                        .all(|&word| word == SENTINEL)
                );
                if name.ends_with("embed_tokens.weight")
                    || name.ends_with("layers.16.mlp.down_proj.weight")
                {
                    for row in [0, rows - 1] {
                        let mut original = vec![0u8; matrix.columns * 2];
                        let offset = data + (start_row + row) * original.len();
                        file.read_exact_at(&mut original, offset as u64).unwrap();
                        let actual = &words[row * row_words..(row + 1) * row_words];
                        assert!(
                            actual
                                .iter()
                                .flat_map(|word| word.to_le_bytes())
                                .eq(original)
                        );
                    }
                }
            });
        }
    }

    let vectors = plan.text_tensors().values().filter(|t| t.shape.len() == 1);
    for (tensor, values) in vectors.zip(&owner.vectors) {
        let mut raw = vec![0u8; tensor.range.len()];
        let offset = plan.payload_offset() + tensor.range.start;
        file.read_exact_at(&mut raw, offset as u64).unwrap();
        check_widened(&raw, values);
    }
    let ple = owner.per_layer_embeddings();
    let mut scratch = ple.workspace(2, usize::MAX).unwrap();
    let mut output = vec![0.; 2 * 8960];
    ple.gather(&mut scratch, &[2, 262143], &mut output).unwrap();
    let mut raw = vec![0u8; 8960 * 2];
    let base = plan.payload_offset() + plan.per_layer_embedding().range.start;
    for (id, row) in [2, 262143].into_iter().zip(output.chunks_exact(8960)) {
        let offset = base + id * raw.len();
        file.read_exact_at(&mut raw, offset as u64).unwrap();
        check_widened(&raw, row);
    }
    println!("BF16_RESIDENT device={device} report={:?}", owner.report());
}

fn check_widened(raw: &[u8], values: &[f32]) {
    assert_eq!(raw.len(), values.len() * 2);
    for (word, value) in raw.chunks_exact(2).zip(values) {
        assert_eq!(value.to_bits().to_be_bytes(), [word[1], word[0], 0, 0]);
    }
}
