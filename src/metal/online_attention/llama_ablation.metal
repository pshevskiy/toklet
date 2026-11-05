#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

inline void add_term(thread float2& sum, float value) {
    float next=sum.x+value;
    sum.y+=abs(sum.x)>=abs(value) ? (sum.x-next)+value : (value-next)+sum.x;
    sum.x=next;
}

// The frozen row 2/head 1/channel 5 uses KV head zero and positions 0..3.
// Preserve the actual Online dot and exponential calculation for every variant.
kernel void llama_offset_ablation(
    device const half* q [[buffer(0)]],
    device const half* k [[buffer(1)]],
    device const half* v [[buffer(2)]],
    device float* result [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]]) {
    float scores[3];
    for (uint p=0;p<3;++p) {
        float dot=0;
        for (uint j=0;j<4;++j) {
            uint f=lane+j*width;
            if (f<8) dot+=float(q[72+f])*float(k[p*8+f]);
        }
        scores[p]=simd_sum(dot)/sqrt(8.0f);
    }
    if (lane!=0) return;
    float maximum=max(scores[0],max(scores[1],scores[2]));
    float weights[3], mass=0, numerator=0, staged=0;
    float2 precise_mass(0.0f), precise_numerator(0.0f);
    for (uint p=0;p<3;++p) {
        float value=float(v[p*8+5]);
        float w=exp(scores[p]-maximum);
        weights[p]=w;
        mass+=w;
        float product=w*value;
        numerator+=product;
        add_term(precise_mass,w);
        add_term(precise_numerator,product);
        add_term(precise_numerator,fma(w,value,-product));
        result[p]=scores[p];
        result[p+3]=w;
    }
    for (uint p=0;p<3;++p) {
        float probability=weights[p]/mass;
        staged+=probability*float(v[p*8+5]);
    }
    result[6]=mass;
    result[7]=numerator;
    result[8]=numerator/mass;
    result[9]=staged;
    result[10]=precise_mass.x;
    result[11]=precise_mass.y;
    result[12]=precise_numerator.x;
    result[13]=precise_numerator.y;
}
