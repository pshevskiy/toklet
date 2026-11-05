// Experimental SIMD score reduction with global staged probabilities.
// Uses the exact Reference score/probability buffer ABI and GQA layout.
kernel void attention_staged_simd(
    device const half* q [[buffer(0)]], device const half* k [[buffer(1)]],
    device const half* v [[buffer(2)]], device float* scores [[buffer(3)]],
    device float* weights [[buffer(4)]], device half* out [[buffer(5)]],
    constant uint* s [[buffer(6)]], device atomic_uint* status [[buffer(7)]],
    uint query [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]]) {
    uint rows=s[0], heads=s[1], kv_heads=s[2], d=s[3], cap=s[4];
    if (query>=rows*heads) return;
    if (width<32 || d>128) {
        if (lane==0) atomic_fetch_or_explicit(status,2u,memory_order_relaxed);
        return;
    }
    uint token=query/heads, head=query%heads, kv=head/(heads/kv_heads);
    uint base=(head*s[5]+token)*cap, causal=s[6]+token+1;
    float scale=sqrt(float(d)), maximum=-INFINITY, query_values[4];
    for (uint j=0;j<4;++j) {
        uint f=lane+j*width;
        query_values[j]=f<d ? float(q[query*d+f]) : 0.0f;
    }
    for (uint p=0;p<causal;++p) {
        float dot=0.0f;
        for (uint j=0;j<4;++j) {
            uint f=lane+j*width;
            float key=f<d ? float(k[(kv*cap+p)*d+f]) : 0.0f;
            dot+=query_values[j]*key;
        }
        float score=simd_sum(dot)/scale;
        if (lane==0) scores[base+p]=score;
        maximum=max(maximum,score);
    }
    simdgroup_barrier(mem_flags::mem_device);
    float mass=0.0f;
    for (uint p=0;p<causal;++p) {
        float weight=exp(scores[base+p]-maximum);
        if (lane==0) weights[base+p]=weight;
        mass+=weight;
    }
    if (lane==0 && (!isfinite(mass) || mass<=0.0f))
        atomic_fetch_or_explicit(status,2u,memory_order_relaxed);
    simdgroup_barrier(mem_flags::mem_device);
    float numerator[4]={};
    for (uint p=0;p<causal;++p) {
        float probability=weights[base+p]/mass;
        for (uint j=0;j<4;++j) {
            uint f=lane+j*width;
            if (f<d) numerator[j]+=probability*float(v[(kv*cap+p)*d+f]);
        }
    }
    for (uint j=0;j<4;++j) {
        uint f=lane+j*width;
        if (f<d) checked_half(out,query*d+f,numerator[j],status);
    }
}
