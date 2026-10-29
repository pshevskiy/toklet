#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// shape=[rows,width,a_offset,a_stride,b_offset,b_stride,out_offset,out_stride].
// One global lane owns one output. Input aliases are readonly; output/status
// cannot alias either input. No inactive cell is read or written.
kernel void strided_half_product(device const half* a [[buffer(0)]],
                                 device const half* b [[buffer(1)]],
                                 device half* output [[buffer(2)]],
                                 constant uint* shape [[buffer(3)]],
                                 device atomic_uint* status [[buffer(4)]],
                                 uint item [[thread_position_in_grid]]) {
    uint width=shape[1];
    if (item>=shape[0]*width) return;
    uint row=item/width, col=item%width;
    float left=float(a[shape[2]+row*shape[3]+col]);
    float right=float(b[shape[4]+row*shape[5]+col]);
    float product=left*right;
    half stored=half(product);
    output[shape[6]+row*shape[7]+col]=stored;
    if (!isfinite(left)||!isfinite(right)||!isfinite(product)||!isfinite(float(stored)))
        atomic_fetch_or_explicit(status,1u,memory_order_relaxed);
}
