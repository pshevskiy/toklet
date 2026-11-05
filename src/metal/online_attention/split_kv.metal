#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

constant uint partition_size=256;
constant uint tile_size=16;

// Decode-only split-KV experiment. Both kernels use one SIMD group per item.
// s: [query heads, KV heads, head width, capacity, depth, allocated partitions].
// Partial packets [query head, allocated partition, (max, mass, numerator[d])].
kernel void attention_split_partial(
    device const half* q [[buffer(0)]],
    device const half* k [[buffer(1)]],
    device const half* v [[buffer(2)]],
    device float* partial [[buffer(3)]],
    constant uint* s [[buffer(4)]],
    device atomic_uint* status [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]]) {
    uint heads=s[0], d=s[2], cap=s[3], head=group%heads, split=group/heads;
    uint kv=head/(heads/s[1]), first=split*partition_size, end=first+min(partition_size,s[4]-first);
    float query[4], numerator[4], maximum=-INFINITY, mass=0;
    for (uint j=0;j<4;++j) {
        uint feature=lane+j*width;
        query[j]=feature<d ? float(q[head*d+feature]) : 0.0f;
        numerator[j]=0;
    }
    for (uint tile=first;tile<end;tile+=tile_size) {
        uint count=min(tile_size,end-tile);
        float scores[tile_size];
        float next=maximum;
        for (uint p=0;p<count;++p) {
            float dot=0;
            for (uint j=0;j<4;++j) {
                uint feature=lane+j*width;
                float key=feature<d ? float(k[(kv*cap+tile+p)*d+feature]) : 0.0f;
                dot+=query[j]*key;
            }
            scores[p]=simd_sum(dot)/sqrt(float(d));
            next=max(next,scores[p]);
        }
        float rescale=exp(maximum-next);
        mass*=rescale;
        for (uint j=0;j<4;++j) numerator[j]*=rescale;
        for (uint p=0;p<count;++p) {
            float weight=exp(scores[p]-next);
            mass+=weight;
            for (uint j=0;j<4;++j) {
                uint feature=lane+j*width;
                if (feature<d) numerator[j]+=weight*float(v[(kv*cap+tile+p)*d+feature]);
            }
        }
        maximum=next;
    }
    uint packet=(head*s[5]+split)*(d+2);
    if (lane==0) {
        if (!isfinite(maximum) || !isfinite(mass) || mass<=0)
            atomic_fetch_or_explicit(status,2u,memory_order_relaxed);
        partial[packet]=maximum;
        partial[packet+1]=mass;
    }
    for (uint j=0;j<4;++j) {
        uint feature=lane+j*width;
        if (feature<d) partial[packet+2+feature]=numerator[j];
    }
}

kernel void attention_split_merge(
    device const float* partial [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint* s [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]]) {
    uint d=s[2], count=(s[4]-1)/partition_size+1, base=head*s[5]*(d+2);
    float maximum=-INFINITY;
    for (uint p=0;p<count;++p) maximum=max(maximum,partial[base+p*(d+2)]);
    float mass=0, numerator[4]={0,0,0,0};
    for (uint p=0;p<count;++p) {
        uint packet=base+p*(d+2);
        float weight=exp(partial[packet]-maximum);
        mass+=weight*partial[packet+1];
        for (uint j=0;j<4;++j) {
            uint feature=lane+j*width;
            if (feature<d) numerator[j]+=weight*partial[packet+2+feature];
        }
    }
    if (lane==0 && (!isfinite(mass) || mass<=0))
        atomic_fetch_or_explicit(status,2u,memory_order_relaxed);
    for (uint j=0;j<4;++j) {
        uint feature=lane+j*width;
        if (feature<d) {
            float value=numerator[j]/mass;
            half rounded=half(value);
            output[head*d+feature]=rounded;
            if (!isfinite(value) || !isfinite(float(rounded)))
                atomic_fetch_or_explicit(status,1u,memory_order_relaxed);
        }
    }
}
