#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// Experimental tiny split-half layout [tokens, heads,8], theta1e6.
// One lane owns a pair, with no in-place input/output aliasing. FP32 angles,
// precise trigonometry and uncontracted FP32 products/sums precede Half stores.
kernel void toklet_rotary(device const half* input [[buffer(0)]],
                          device half* output [[buffer(1)]],
                          constant uint* shape [[buffer(2)]],
                          uint pair [[thread_position_in_grid]]) {
    uint rows=shape[0], heads=shape[1], dimension=shape[2], offset=shape[3];
    uint half_dim=dimension/2, pairs=rows*heads*half_dim;
    if (pair>=pairs) return;
    uint feature=pair%half_dim, head=(pair/half_dim)%heads;
    uint token=pair/(heads*half_dim);
    uint first=(token*heads+head)*dimension+feature, second=first+half_dim;
    float inverse=pow(1000000.0f, -2.0f*float(feature)/float(dimension));
    float angle=float(offset+token)*inverse;
    float cosine=cos(angle), sine=sin(angle);
    float a=float(input[first]), b=float(input[second]);
    output[first]=half(a*cosine-b*sine);
    output[second]=half(b*cosine+a*sine);
}
