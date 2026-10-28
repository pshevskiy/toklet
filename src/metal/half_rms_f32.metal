#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// The existing opt-in RMS Neumaier order, with FP32 effective scales. Half
// squares are exact FP32 products; scaling is never narrowed before multiply.
float2 rms_f32_add(float2 sum, float value) {
    float next = sum.x + value;
    float error = abs(sum.x) >= abs(value) ? (sum.x-next)+value : (value-next)+sum.x;
    return float2(next, sum.y+error);
}

// shape=[rows,width,input_stride,output_stride,exact_fp32_epsilon_bits].
// Each thread owns one complete row. Padding/guards are never read or written.
kernel void half_rms_f32_scale(device const half* input [[buffer(0)]],
                              device const float* scales [[buffer(1)]],
                              device half* output [[buffer(2)]],
                              constant uint* shape [[buffer(3)]],
                              device atomic_uint* status [[buffer(4)]],
                              uint row [[thread_position_in_grid]]) {
    if (row >= shape[0]) return;
    uint source = row*shape[2], destination = row*shape[3];
    float2 sum = float2(0.0f);
    for (uint col=0; col<shape[1]; ++col) {
        float value = float(input[source+col]);
        sum = rms_f32_add(sum, value*value);
    }
    float total = sum.x+sum.y;
    float inverse = 1.0f/sqrt(total/float(shape[1])+as_type<float>(shape[4]));
    for (uint col=0; col<shape[1]; ++col) {
        float value = (float(input[source+col])*inverse)*scales[col];
        half rounded = half(value);
        output[destination+col] = rounded;
        if (!isfinite(value) || !isfinite(float(rounded)))
            atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    }
}
