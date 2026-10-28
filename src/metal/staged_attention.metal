#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// ABI: tokens,D,score_capacity,committed,view_start,keys,
//      old_first_slot,old_first_rows,old_second_rows,window_or_zero,chunk_capacity.
// The two old spans are computed by the shared checked host row geometry.
device const half* staged_kv_row(device const half* old,
                                 device const half* staged,
                                 constant uint* s, uint key) {
    if (key < s[7]) return old+(s[6]+key)*s[1];
    if (key < s[7]+s[8]) return old+(key-s[7])*s[1];
    return staged+(key-s[7]-s[8])*s[1];
}
bool staged_visible(uint absolute, uint query, uint window) {
    return absolute<=query && (window==0 || query-absolute<window);
}

// One serial Reference thread owns one query/head; fixed KV1 and Q4 do not
// expand KV. Scores/probabilities are [4,chunk_capacity,score_capacity];
// Q/context are [chunk_capacity,4,D]. Old/new Half KV remain immutable.
kernel void staged_reference_attention(
    device const half* q [[buffer(0)]],
    device const half* old_k [[buffer(1)]],
    device const half* old_v [[buffer(2)]],
    device const half* new_k [[buffer(3)]],
    device const half* new_v [[buffer(4)]],
    device float* scores [[buffer(5)]],
    device float* probabilities [[buffer(6)]],
    device half* context [[buffer(7)]],
    constant uint* s [[buffer(8)]],
    device atomic_uint* status [[buffer(9)]],
    uint item [[thread_position_in_grid]]) {
    if (item>=s[0]*4u) return;
    uint token=item/4u, head=item%4u, dim=s[1];
    uint qbase=(token*4u+head)*dim;
    uint outbase=(head*s[10]+token)*s[2];
    uint query=s[3]+token;
    // Explicitly validate every active staged row, including new values not
    // visible to an early query. Do not depend on an absent projection check.
    for (uint f=0;f<dim;++f) {
        if (!isfinite(float(q[qbase+f])) ||
            !isfinite(float(new_k[token*dim+f])) ||
            !isfinite(float(new_v[token*dim+f])))
            atomic_fetch_or_explicit(status,1u,memory_order_relaxed);
    }
    float root=sqrt(float(dim)), maximum=-INFINITY;
    for (uint key=0;key<s[5];++key) {
        device const half* k=staged_kv_row(old_k,new_k,s,key);
        float total=0.0f;
        for (uint f=0;f<dim;++f) total=total+float(q[qbase+f])*float(k[f]);
        float score=total/root;
        scores[outbase+key]=score;
        if (!isfinite(score)) atomic_fetch_or_explicit(status,1u,memory_order_relaxed);
        if (staged_visible(s[4]+key,query,s[9])) maximum=max(maximum,score);
    }
    float mass=0.0f;
    for (uint key=0;key<s[5];++key) {
        float weight=0.0f;
        if (staged_visible(s[4]+key,query,s[9])) weight=exp(scores[outbase+key]-maximum);
        probabilities[outbase+key]=weight;
        mass=mass+weight;
    }
    if (!isfinite(mass) || mass<=0.0f) atomic_fetch_or_explicit(status,1u,memory_order_relaxed);
    for (uint key=0;key<s[5];++key) {
        float probability=probabilities[outbase+key]/mass;
        probabilities[outbase+key]=probability;
        if (!isfinite(probability)) atomic_fetch_or_explicit(status,1u,memory_order_relaxed);
    }
    for (uint f=0;f<dim;++f) {
        float total=0.0f;
        for (uint key=0;key<s[5];++key) {
            if (!staged_visible(s[4]+key,query,s[9])) continue;
            device const half* v=staged_kv_row(old_v,new_v,s,key);
            float value=float(v[f]);
            if (!isfinite(value)) atomic_fetch_or_explicit(status,1u,memory_order_relaxed);
            total=total+probabilities[outbase+key]*value;
        }
        half rounded=half(total);
        context[qbase+f]=rounded;
        if (!isfinite(total) || !isfinite(float(rounded)))
            atomic_fetch_or_explicit(status,1u,memory_order_relaxed);
    }
}
