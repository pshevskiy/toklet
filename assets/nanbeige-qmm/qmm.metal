// The original implementation is attributed in vendor/LICENSE.
// This translation unit selects a single exported specialization.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

instantiate_kernel(
    "affine_qmm_t_bfloat16_t_gs_64_b_4_alN_true_batch_0",
    affine_qmm_t,
    bfloat16_t,
    64,
    4,
    true,
    false)
