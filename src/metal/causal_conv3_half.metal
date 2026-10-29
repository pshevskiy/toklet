#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// shape=[rows,width,input_stride,output_stride]. Each global lane owns (t,h).
// Old/candidate are [2,H], taps [H,3] oldest/middle/current. Shared sticky
// status is never reset here. Dirty output/candidate on failure must be discarded.
kernel void causal_conv3_half(device const ushort* input [[buffer(0)]],
                              device const float* scales [[buffer(1)]],
                              device half* output [[buffer(2)]],
                              device const ushort* old [[buffer(3)]],
                              device ushort* candidate [[buffer(4)]],
                              constant uint* shape [[buffer(5)]],
                              device atomic_uint* status [[buffer(6)]],
                              uint item [[thread_position_in_grid]]) {
    uint rows=shape[0], width=shape[1];
    if (item>=rows*width) return;
    uint row=item/width, col=item%width;
    float x0=row<2 ? float(as_type<half>(old[row*width+col]))
                  : float(as_type<half>(input[(row-2)*shape[2]+col]));
    float x1=row==0 ? float(as_type<half>(old[width+col]))
                   : float(as_type<half>(input[(row-1)*shape[2]+col]));
    float x2=float(as_type<half>(input[row*shape[2]+col]));
    float c0=scales[3*col], c1=scales[3*col+1], c2=scales[3*col+2];
    float p0=x0*c0, p1=x1*c1, p2=x2*c2;
    float s01=p0+p1;
    float s012=s01+p2;
    half rounded=half(s012);
    output[row*shape[3]+col]=rounded;
    if (!isfinite(x0)||!isfinite(x1)||!isfinite(x2)||
        !isfinite(c0)||!isfinite(c1)||!isfinite(c2)||
        !isfinite(p0)||!isfinite(p1)||!isfinite(p2)||
        !isfinite(s01)||!isfinite(s012)||!isfinite(float(rounded)))
        atomic_fetch_or_explicit(status,1u,memory_order_relaxed);
    // Exactly H GLOBAL lanes write BOTH rows, including T=1. Integer bit copies
    // preserve negative zero/subnormals; no conversion or concurrent writers.
    if (item<width) {
        candidate[col]=rows==1 ? old[width+col]
            : input[(rows-2)*shape[2]+col];
        candidate[width+col]=input[(rows-1)*shape[2]+col];
    }
}
