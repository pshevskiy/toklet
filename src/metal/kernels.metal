#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
inline void checked_half(device half* out,uint i,float v,device atomic_uint* status) {
    half h=half(v);
    out[i]=h;
    if(!isfinite(v)||!isfinite(float(h)))atomic_fetch_or_explicit(status,1u,memory_order_relaxed);
}

kernel void embedding(device const ushort* w[[buffer(0)]],device const uint* tokens[[buffer(1)]],device ushort* out[[buffer(2)]],constant uint* s[[buffer(3)]],uint i[[thread_position_in_grid]]) {
    if(i>=s[0]*s[1])return;
    out[i]=w[tokens[i/s[1]]*s[1]+i%s[1]];
}

kernel void rms(device const half* x[[buffer(0)]],device const half* w[[buffer(1)]],device half* out[[buffer(2)]],constant uint* s[[buffer(3)]],device atomic_uint* status[[buffer(4)]],uint row[[thread_position_in_grid]]) {
    if(row>=s[0])return;
    uint base=row*s[1];
    float total=0;
    for(uint i=0;i<s[1];++i) {
        float v=float(x[base+i]);
        total=total+v*v;
    }
    float inv=1.0f/sqrt(total/float(s[1])+as_type<float>(s[2]));
    for(uint i=0;i<s[1];++i)checked_half(out,base+i,(float(x[base+i])*inv)*float(w[i]),status);
}

kernel void bias(device const half* b[[buffer(0)]],device float* out[[buffer(1)]],constant uint4& s[[buffer(2)]],uint i[[thread_position_in_grid]]) {
    if(i>=s.x*s.y)return;
    out[(i/s.y)*s.z+i%s.y]=float(b[i%s.y]);
}

kernel void store(device const float* x[[buffer(0)]],device half* out[[buffer(1)]],constant uint4& s[[buffer(2)]],device atomic_uint* status[[buffer(3)]],uint i[[thread_position_in_grid]]) {
    if(i>=s.x*s.y)return;
    checked_half(out,(i/s.y)*s.w+i%s.y,x[(i/s.y)*s.z+i%s.y],status);
}

kernel void rotary(device const half* x[[buffer(0)]],device half* out[[buffer(1)]],constant uint* s[[buffer(2)]],device atomic_uint* status[[buffer(3)]],uint pair[[thread_position_in_grid]]) {
    uint half_dim=s[2]/2;
    if(pair>=s[0]*s[1]*half_dim)return;
    uint f=pair%half_dim,head=(pair/half_dim)%s[1],token=pair/(s[1]*half_dim);
    uint a=(token*s[1]+head)*s[2]+f,b=a+half_dim;
    float inv=pow(1000000.0f,-2.0f*float(f)/float(s[2]));
    float angle=float(s[3]+token)*inv,c=cos(angle),v=sin(angle);
    float av=float(x[a]),bv=float(x[b]);
    checked_half(out,a,av*c-bv*v,status);
    checked_half(out,b,bv*c+av*v,status);
}

// [rows, kv_heads, dimension, capacity, placement]. Cache heads are never expanded.
kernel void cache(device const ushort* k[[buffer(0)]],device const ushort* v[[buffer(1)]],device ushort* kc[[buffer(2)]],device ushort* vc[[buffer(3)]],constant uint* s[[buffer(4)]],uint i[[thread_position_in_grid]]) {
    uint width=s[1]*s[2];
    if(i>=s[0]*width)return;
    uint token=i/width,head=(i%width)/s[2],f=i%s[2];
    uint dst=(head*s[3]+s[4]+token)*s[2]+f;
    kc[dst]=k[i];
    vc[dst]=v[i];
}

// One query/head lane, stable max softmax, serial FP32 reductions. Scores and
// probabilities occupy [query_heads, chunk_capacity, cache_capacity]; logical
// rows/valid length are separate from allocated strides. Future slots are +0.
kernel void attention(device const half* q[[buffer(0)]],device const half* k[[buffer(1)]],device const half* v[[buffer(2)]],device float* scores[[buffer(3)]],device float* probs[[buffer(4)]],device half* out[[buffer(5)]],constant uint* s[[buffer(6)]],device atomic_uint* status[[buffer(7)]],uint query[[thread_position_in_grid]]) {
    uint rows=s[0],heads=s[1],kvheads=s[2],d=s[3],cap=s[4],chunk=s[5],offset=s[6],valid=s[7];
    if(query>=rows*heads)return;
    uint token=query/heads,head=query%heads,kv=head/(heads/kvheads),base=(head*chunk+token)*cap,causal=offset+token+1;
    float scale=sqrt(float(d)),maximum=-INFINITY;
    for(uint p=0;p<valid;++p) {
        float acc=0;
        for(uint f=0;f<d;++f)acc=acc+float(q[(token*heads+head)*d+f])*float(k[(kv*cap+p)*d+f]);
        float score=acc/scale;
        scores[base+p]=score;
        if(p<causal)maximum=max(maximum,score);
    }
    float sum=0;
    for(uint p=0;p<valid;++p) {
        float e=p<causal?exp(scores[base+p]-maximum):0.0f;
        probs[base+p]=e;
        sum=sum+e;
    }
    if(!isfinite(sum)||sum<=0)atomic_fetch_or_explicit(status,2u,memory_order_relaxed);
    for(uint p=0;p<valid;++p)probs[base+p]=probs[base+p]/sum;
    for(uint f=0;f<d;++f) {
        float acc=0;
        for(uint p=0;p<causal;++p)acc=acc+probs[base+p]*float(v[(kv*cap+p)*d+f]);
        checked_half(out,(token*heads+head)*d+f,acc,status);
    }
}

kernel void residual(device half* x[[buffer(0)]],device const half* delta[[buffer(1)]],constant uint& n[[buffer(2)]],device atomic_uint* status[[buffer(3)]],uint i[[thread_position_in_grid]]) {
    if(i>=n)return;
    checked_half(x,i,float(x[i])+float(delta[i]),status);
}

kernel void silu(device const half* x[[buffer(0)]],device half* out[[buffer(1)]],constant uint& n[[buffer(2)]],device atomic_uint* status[[buffer(3)]],uint i[[thread_position_in_grid]]) {
    if(i>=n)return;
    float v=float(x[i]),e=exp(-abs(v));
    float sigmoid=v>=0?1.0f/(1.0f+e):e/(1.0f+e);
    checked_half(out,i,v*sigmoid,status);
}

kernel void gating(device const half* a[[buffer(0)]],device const half* b[[buffer(1)]],device half* out[[buffer(2)]],constant uint& n[[buffer(3)]],device atomic_uint* status[[buffer(4)]],uint i[[thread_position_in_grid]]) {
    if(i>=n)return;
    checked_half(out,i,float(a[i])*float(b[i]),status);
}

kernel void validate_float(device const float* x[[buffer(0)]],constant uint4& s[[buffer(1)]],device atomic_uint* status[[buffer(2)]],uint i[[thread_position_in_grid]]) {
    if(i>=s.x*s.y)return;
    if(!isfinite(x[(i/s.y)*s.z+i%s.y]))atomic_fetch_or_explicit(status,4u,memory_order_relaxed);
}


// Diagnostic dependency marker; production execution never dispatches it.
kernel void marker(device atomic_uint* source[[buffer(0)]],device uint* sink[[buffer(1)]],uint i[[thread_position_in_grid]]) {
    if(i==0)sink[0]=atomic_fetch_or_explicit(source,0u,memory_order_relaxed);
}

// One SIMD group per [token, output row]. Adjacent lanes read adjacent columns,
// reconstruct affine weights in Float32, and reduce without expanded storage.
// [tokens, inputs, outputs, packed row bytes, input Half stride, output Half stride].
kernel void q4_matvec(device const uchar* w[[buffer(0)]],device const half* bias[[buffer(1)]],device const half* x[[buffer(2)]],device half* out[[buffer(3)]],constant uint* s[[buffer(4)]],device atomic_uint* status[[buffer(5)]],uint group[[threadgroup_position_in_grid]],uint lane[[thread_index_in_simdgroup]],uint width[[threads_per_simdgroup]]) {
    uint token=group/s[2],row=group%s[2];
    if(token>=s[0])return;
    float sum=0.0f;
    for(uint column=lane;column<s[1];column+=width) {
        uint start=64+row*s[3]+(column/64)*36;
        device const ushort* metadata=reinterpret_cast<device const ushort*>(w+start);
        float scale=float(as_type<half>(metadata[0])),b=float(as_type<half>(metadata[1]));
        uint code=(w[start+4+(column%64)/2]>>(4*(column%2)))&15;
        float weight=scale*float(code)+b;
        sum=sum+weight*float(x[token*s[4]+column]);
    }
    float value=simd_sum(sum)+float(bias[row]);
    if(lane==0)checked_half(out,token*s[5]+row,value,status);
}
