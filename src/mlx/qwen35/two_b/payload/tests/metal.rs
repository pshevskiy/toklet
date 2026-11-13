use super::*;
use crate::metal::{
    Error as MetalError,
    gemma_affine::{CONTEXT_ALLOWANCE_BYTES, Context},
};
#[test]
#[ignore = "explicit tiny2B typed consuming GPU storage; no model/graph qualification"]
fn original_f32_parameters_and_consumed_gpu_storage_remain_exact() {
    let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
    let limit = 6 * 1024 * 1024 * 1024;
    let required =
        tiny().report().retained_tensor_capacity_bytes + CONTEXT_ALLOWANCE_BYTES + 64 * 1024 * 1024;
    assert!(
        matches!(tiny().into_metal(&context,required-1),Err(MetalError::Budget { required:r,limit:l })
        if r==required &&l==required-1)
    );
    let gpu = tiny().into_metal(&context, limit).unwrap();
    assert_eq!(
        gpu.vector(ALOG)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        F32_BITS
    );
    assert_eq!(gpu.convolution(CONV).unwrap(), [0.5, -1.0, 2.0, 0.0]);
    assert_eq!(gpu.report().retained_cpu_matrix_bytes, 0);
    assert_eq!(gpu.report().small_cpu_capacity_bytes, 36);
    let (_, weights) = gpu.matrix(MATRIX).unwrap();
    let mut gather = context.gather_session(64, 1, limit).unwrap();
    let mut row = [0.0; 64];
    gather.apply_weights(weights, &[0], &mut row).unwrap();
    assert_eq!(row, [5.0; 64]);
    drop(gpu);
    let mut bad = tiny();
    bad.components.matrices.get_mut(MATRIX).unwrap().scales[..2]
        .copy_from_slice(&0x7f80u16.to_le_bytes());
    assert!(bad.into_metal(&context, limit).is_err());
    assert_eq!(
        tiny()
            .into_metal(&context, limit)
            .unwrap()
            .report()
            .retained_cpu_matrix_bytes,
        0
    );
}
