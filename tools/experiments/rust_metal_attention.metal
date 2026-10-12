#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
// Diagnostic tiny full-prefix GQA: Q[tokens,4,8], K/V[2,32,8]. One lane
// owns one query token/head and its score/probability row. No head expansion.
// Stable FP32 ordered reductions; future probabilities are exactly zero.
kernel void toklet_attention(device const half* q [[buffer(0)]],
                             device const half* k [[buffer(1)]],
                             device const half* v [[buffer(2)]],
                             device float* scores [[buffer(3)]],
                             device float* probabilities [[buffer(4)]],
                             device half* context [[buffer(5)]],
                             constant uint& rows [[buffer(6)]],
                             uint query [[thread_position_in_grid]]) {
    if(query>=rows*4) return;
    uint token=query/4,head=query%4,kv=head/2;
    uint row=(head*rows+token)*rows;
    float maximum=-INFINITY;
    for(uint position=0;position<rows;++position){
        float sum=0.0f;
        for(uint feature=0;feature<8;++feature)
            sum=sum+float(q[query*8+feature])*float(k[(kv*32+position)*8+feature]);
        float score=sum/sqrt(8.0f);scores[row+position]=score;
        if(position<=token) maximum=max(maximum,score);
    }
    float denominator=0.0f;
    for(uint position=0;position<rows;++position){
        float value=position<=token?exp(scores[row+position]-maximum):0.0f;
        probabilities[row+position]=value;denominator=denominator+value;
    }
    for(uint position=0;position<=token;++position)
        probabilities[row+position]=probabilities[row+position]/denominator;
    for(uint feature=0;feature<8;++feature){
        float sum=0.0f;
        for(uint position=0;position<=token;++position)
            sum=sum+probabilities[row+position]*float(v[(kv*32+position)*8+feature]);
        context[query*8+feature]=half(sum);
    }
}
