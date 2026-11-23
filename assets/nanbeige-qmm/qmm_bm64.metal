// The original implementation is attributed in vendor/LICENSE.
// Change only the selected token-row tile from the shipped 32 to 64.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

instantiate_kernel(
    "toklet_qmm_bm64",
    affine_qmm_t,
    bfloat16_t,
    64,
    4,
    true,
    false,
    false,
    64,
    32,
    32)
