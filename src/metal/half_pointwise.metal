#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

void pointwise_store(device half* output, uint index, float value,
                     device atomic_uint* status) {
    half rounded = half(value);
    output[index] = rounded;
    if (!isfinite(value) || !isfinite(float(rounded)))
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
}

// shape=[rows,width,input_stride,output_stride,FP32_factor_bits].
// Source and destination own independent guarded strided Half storage.
kernel void half_tanh_gelu(device const half* input [[buffer(0)]],
                           device half* output [[buffer(1)]],
                           constant uint* shape [[buffer(2)]],
                           device atomic_uint* status [[buffer(3)]],
                           uint index [[thread_position_in_grid]]) {
    if (index >= shape[0]*shape[1]) return;
    uint row = index/shape[1], col = index%shape[1];
    float x = float(input[row*shape[2]+col]);
    float cube = (x*x)*x;
    float z = 0.7978845608028654f*(x+0.044715f*cube);
    float value = (0.5f*x)*(1.0f+tanh(z));
    pointwise_store(output, row*shape[3]+col, value, status);
}

kernel void half_scalar_scale(device const half* input [[buffer(0)]],
                              device half* output [[buffer(1)]],
                              constant uint* shape [[buffer(2)]],
                              device atomic_uint* status [[buffer(3)]],
                              uint index [[thread_position_in_grid]]) {
    if (index >= shape[0]*shape[1]) return;
    uint row = index/shape[1], col = index%shape[1];
    float value = float(input[row*shape[2]+col])*as_type<float>(shape[4]);
    pointwise_store(output, row*shape[3]+col, value, status);
}
