#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
// One queried SIMD group selects the lowest ID with the largest finite logit.
// Invalid values set the chunk status; their candidate is never published.
kernel void greedy_token(device const uint* logits[[buffer(0)]],device uint* out[[buffer(1)]],constant uint* s[[buffer(2)]],device atomic_uint* status[[buffer(3)]],uint group[[threadgroup_position_in_grid]],uint lane[[thread_index_in_simdgroup]],uint width[[threads_per_simdgroup]]) {
    if(group!=0)return;
    uint best=0,id=0xffffffffu;
    for(uint i=lane;i<s[0];i+=width) {
        uint bits=logits[s[1]+i];
        if((bits&0x7f800000u)==0x7f800000u)atomic_fetch_or_explicit(status,1u,memory_order_relaxed);
        else {
            // IEEE finite order, with both zero signs sharing a key. Integer
            // comparison also preserves subnormal distinctions without relying
            // on device floating-point arithmetic for the selection itself.
            uint key=(bits&0x7fffffffu)==0?0x80000000u:(bits&0x80000000u)!=0?~bits:bits^0x80000000u;
            if(key>best || (key==best && i<id)) {best=key;id=i;}
        }
    }
    for(uint hop=1;hop<width;) {
        uint other=simd_shuffle_down(best,hop),other_id=simd_shuffle_down(id,hop);
        if(lane%hop==0 && (lane/hop)%2==0 && lane<width-hop && (other>best || (other==best && other_id<id))) {best=other;id=other_id;}
        if(hop>=width-hop)break;
        hop*=2;
    }
    if(lane==0) {
        out[0]=id;
        if(id==0xffffffffu)atomic_fetch_or_explicit(status,1u,memory_order_relaxed);
    }
}
inline void checked_half(device half* out,uint i,float v,device atomic_uint* status) {
    half h=half(v);
    out[i]=h;
    if(!isfinite(v)||!isfinite(float(h)))atomic_fetch_or_explicit(status,1u,memory_order_relaxed);
}
// Neumaier's two-component FP32 sum, shared by the opt-in accurate reductions.
inline float2 compensated_add(float2 sum,float value) {
    float next=sum.x+value;
    float error=abs(sum.x)>=abs(value)?(sum.x-next)+value:(value-next)+sum.x;
    return float2(next,sum.y+error);
}

kernel void embedding(device const ushort* w[[buffer(0)]],device const uint* tokens[[buffer(1)]],device ushort* out[[buffer(2)]],constant uint* s[[buffer(3)]],uint i[[thread_position_in_grid]]) {
    if(i>=s[0]*s[1])return;
    out[i]=w[tokens[i/s[1]]*s[1]+i%s[1]];
}

kernel void rms(device const half* x[[buffer(0)]],device const half* w[[buffer(1)]],device half* out[[buffer(2)]],constant uint* s[[buffer(3)]],device atomic_uint* status[[buffer(4)]],uint row[[thread_position_in_grid]]) {
    if(row>=s[0])return;
    uint base=row*s[1];
    float total=0;
    if(s[3]!=0) {
        // Half squares are exact FP32 products. Compensate their positive
        // reduction only; division, sqrt, inverse and scaling stay unchanged.
        float2 sum=float2(0.0f);
        for(uint i=0;i<s[1];++i) {
            float v=float(x[base+i]);
            sum=compensated_add(sum,v*v);
        }
        total=sum.x+sum.y;
    } else {
        for(uint i=0;i<s[1];++i) {
            float v=float(x[base+i]);
            total=total+v*v;
        }
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

// [rows, heads, head_dim, absolute_offset, positive finite theta as FP32 bits].
kernel void rotary(device const half* x[[buffer(0)]],device half* out[[buffer(1)]],constant uint* s[[buffer(2)]],device atomic_uint* status[[buffer(3)]],uint pair[[thread_position_in_grid]]) {
    uint half_dim=s[2]/2;
    if(pair>=s[0]*s[1]*half_dim)return;
    uint f=pair%half_dim,head=(pair/half_dim)%s[1],token=pair/(s[1]*half_dim);
    uint a=(token*s[1]+head)*s[2]+f,b=a+half_dim;
    float inv=pow(as_type<float>(s[4]),-2.0f*float(f)/float(s[2]));
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

// One queried SIMD group owns one head and R query rows. Features are
// striped across lanes; contiguous K/V features are read together. Prefill
// shares each K load between two rows (V traversal remains per query). Only 16 scores per row are live,
// never [rows, depth]. FP32 tile maximum/mass/numerator obey online softmax:
// m'=max(m,tilemax), l'=exp(m-m')*l+sum(exp(score-m')).
// q/out: [rows, query_heads, d]; K/V: [kv_heads, capacity, d]. Cache
// placement is relative to this session; RoPE already used absolute positions.
// Host admits d<=128 and queried SIMD width>=32, hence four feature slots.
template <uint R>
void online_attention(device const half* q, device const half* k,
                      device const half* v, device half* out,
                      constant uint* s, device atomic_uint* status,
                      uint group, uint lane, uint width) {
    uint heads=s[1],d=s[3],cap=s[4],head=group%heads;
    uint first=(group/heads)*R,kv=head/(heads/s[2]);
    float queries[R][4],numerator[R][4],maximum[R],mass[R];
    for(uint r=0;r<R;++r) {
        maximum[r]=-INFINITY;mass[r]=0;
        for(uint j=0;j<4;++j) {
            uint f=lane+j*width;
            queries[r][j]=(first+r<s[0] && f<d)?float(q[((first+r)*heads+head)*d+f]):0.0f;
            numerator[r][j]=0;
        }
    }
    uint end=s[6]+min(first+R,s[0]);
    for(uint tile=0;tile<end;tile+=16) {
        float scores[R][16];
        for(uint p=0;p<min(16u,end-tile);++p) {
            float dots[R];
            for(uint r=0;r<R;++r)dots[r]=0;
            for(uint j=0;j<4;++j) {
                uint f=lane+j*width;
                float key=f<d?float(k[(kv*cap+tile+p)*d+f]):0.0f;
                for(uint r=0;r<R;++r)dots[r]+=queries[r][j]*key;
            }
            for(uint r=0;r<R;++r)scores[r][p]=simd_sum(dots[r])/sqrt(float(d));
        }
        for(uint r=0;r<R;++r) {
            if(first+r>=s[0] || tile>=s[6]+first+r+1)continue;
            uint count=min(16u,s[6]+first+r+1-tile);
            float next=maximum[r];
            for(uint p=0;p<count;++p)next=max(next,scores[r][p]);
            float rescale=exp(maximum[r]-next);
            mass[r]*=rescale;
            for(uint j=0;j<4;++j)numerator[r][j]*=rescale;
            for(uint p=0;p<count;++p) {
                float weight=exp(scores[r][p]-next);
                mass[r]+=weight;
                for(uint j=0;j<4;++j) {
                    uint f=lane+j*width;
                    if(f<d)numerator[r][j]+=weight*float(v[(kv*cap+tile+p)*d+f]);
                }
            }
            maximum[r]=next;
        }
    }
    for(uint r=0;r<R;++r) {
        if(first+r>=s[0])continue;
        if(!isfinite(mass[r]) || mass[r]<=0)atomic_fetch_or_explicit(status,2u,memory_order_relaxed);
        for(uint j=0;j<4;++j) {
            uint f=lane+j*width;
            if(f<d)checked_half(out,((first+r)*heads+head)*d+f,numerator[r][j]/mass[r],status);
        }
    }
}
kernel void attention_decode_online(device const half* q[[buffer(0)]],device const half* k[[buffer(1)]],device const half* v[[buffer(2)]],device half* out[[buffer(3)]],constant uint* s[[buffer(4)]],device atomic_uint* status[[buffer(5)]],uint group[[threadgroup_position_in_grid]],uint lane[[thread_index_in_simdgroup]],uint width[[threads_per_simdgroup]]) {
    online_attention<1>(q,k,v,out,s,status,group,lane,width);
}
kernel void attention_prefill_online(device const half* q[[buffer(0)]],device const half* k[[buffer(1)]],device const half* v[[buffer(2)]],device half* out[[buffer(3)]],constant uint* s[[buffer(4)]],device atomic_uint* status[[buffer(5)]],uint group[[threadgroup_position_in_grid]],uint lane[[thread_index_in_simdgroup]],uint width[[threads_per_simdgroup]]) {
    online_attention<2>(q,k,v,out,s,status,group,lane,width);
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

// One SIMD group per [eight-token tile, output row]. The ordered column
// traversal/reduction follows q4_matvec for the same queried SIMD width.
// Each reconstructed Float32
// weight is reused across eight independent lane accumulators. Partial tiles
// neither read nor write inactive rows. ABI/strides are identical to matvec.
kernel void q4_prefill(device const uchar* w[[buffer(0)]],device const half* bias[[buffer(1)]],device const half* x[[buffer(2)]],device half* out[[buffer(3)]],constant uint* s[[buffer(4)]],device atomic_uint* status[[buffer(5)]],uint group[[threadgroup_position_in_grid]],uint lane[[thread_index_in_simdgroup]],uint width[[threads_per_simdgroup]]) {
    uint first=(group/s[2])*8,row=group%s[2];
    if(first>=s[0])return;
    uint active=min(8u,s[0]-first);
    float sums[8]={0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
    for(uint column=lane;column<s[1];column+=width) {
        uint start=64+row*s[3]+(column/64)*36;
        device const ushort* metadata=reinterpret_cast<device const ushort*>(w+start);
        float scale=float(as_type<half>(metadata[0])),b=float(as_type<half>(metadata[1]));
        uint code=(w[start+4+(column%64)/2]>>(4*(column%2)))&15;
        float weight=scale*float(code)+b;
        // Fixed indices let the compiler retain eight independent accumulators
        // in registers. The dynamically indexed prototype lost end-to-end time.
        #pragma unroll
        for(uint token=0;token<8;token++)
            if(token<active)sums[token]=sums[token]+weight*float(x[(first+token)*s[4]+column]);
    }
    #pragma unroll
    for(uint token=0;token<8;token++) {
        if(token<active) {
            float value=simd_sum(sums[token])+float(bias[row]);
            if(lane==0)checked_half(out,(first+token)*s[5]+row,value,status);
        }
    }
}

// Neumaier's two-component FP32 sum. Explicit fma below recovers the product
// residual; reconstruction itself remains the same rounded FP32 format value.
// Round an inexact two-component result to an odd Float32 significand before
// Half RNE. This retains which side of a Half midpoint the low component lies
// on, avoiding double rounding through a nearest-even Float32 midpoint.
inline float compensated_store_value(float2 sum) {
    float value=sum.x+sum.y;
    if(!isfinite(value))return value;
    float tail=abs(sum.x)>=abs(sum.y)?(sum.x-value)+sum.y:(sum.y-value)+sum.x;
    uint bits=as_type<uint>(value);
    if(tail!=0.0f && (bits&1u)==0) {
        if(value==0.0f)return copysign(as_type<float>(1u),tail);
        bits=((tail>0.0f)==(value>0.0f))?bits+1u:bits-1u;
        value=as_type<float>(bits);
    }
    return value;
}
inline float2 compensated_reduce(float2 sum,uint lane,uint width) {
    // Merge disjoint adjacent subtrees for any queried positive width. All
    // lanes participate in shuffles; out-of-range results are never consumed.
    for(uint hop=1;hop<width;) {
        float hi=simd_shuffle_down(sum.x,hop),lo=simd_shuffle_down(sum.y,hop);
        if(lane%hop==0 && (lane/hop)%2==0 && lane<width-hop) {
            sum=compensated_add(sum,hi);
            sum=compensated_add(sum,lo);
        }
        if(hop>=width-hop)break;
        hop*=2;
    }
    return sum;
}
// Resident Half [out,in] weights and [tokens,in] activations. Products of two
// finite Half values are exactly representable in FP32, including subnormals.
// A queried SIMD group owns one complete output, independently of batch shape.
// Shape: rows,in,out,weight_stride,input_stride,output_stride,input_offset.
inline float2 half_compensated_dot(device const half* w,device const half* bias,device const half* x,constant uint* s,uint group,uint lane,uint width) {
    uint token=group/s[2],row=group%s[2];
    float2 sum=float2(0.0f);
    for(uint column=lane;column<s[1];column+=width)
        sum=compensated_add(sum,float(w[row*s[3]+column])*float(x[s[6]+token*s[4]+column]));
    sum=compensated_reduce(sum,lane,width);
    if(lane==0)sum=compensated_add(sum,float(bias[row]));
    return sum;
}
kernel void half_compensated(device const half* w[[buffer(0)]],device const half* bias[[buffer(1)]],device const half* x[[buffer(2)]],device half* out[[buffer(3)]],constant uint* s[[buffer(4)]],device atomic_uint* status[[buffer(5)]],uint group[[threadgroup_position_in_grid]],uint lane[[thread_index_in_simdgroup]],uint width[[threads_per_simdgroup]]) {
    if(group/s[2]>=s[0])return;
    float2 sum=half_compensated_dot(w,bias,x,s,group,lane,width);
    if(lane==0)checked_half(out,(group/s[2])*s[5]+group%s[2],compensated_store_value(sum),status);
}
kernel void half_compensated_float(device const half* w[[buffer(0)]],device const half* bias[[buffer(1)]],device const half* x[[buffer(2)]],device float* out[[buffer(3)]],constant uint* s[[buffer(4)]],device atomic_uint* status[[buffer(5)]],uint group[[threadgroup_position_in_grid]],uint lane[[thread_index_in_simdgroup]],uint width[[threads_per_simdgroup]]) {
    if(group/s[2]>=s[0])return;
    float2 sum=half_compensated_dot(w,bias,x,s,group,lane,width);
    if(lane==0) {
        // Final logits retain ordinary nearest FP32 rounding; odd rounding is
        // only for the subsequent Half conversion in half_compensated.
        float value=sum.x+sum.y;
        out[(group/s[2])*s[5]+group%s[2]]=value;
        if(!isfinite(value))atomic_fetch_or_explicit(status,1u,memory_order_relaxed);
    }
}
kernel void q4_compensated(device const uchar* w[[buffer(0)]],device const half* bias[[buffer(1)]],device const half* x[[buffer(2)]],device half* out[[buffer(3)]],constant uint* s[[buffer(4)]],device atomic_uint* status[[buffer(5)]],uint group[[threadgroup_position_in_grid]],uint lane[[thread_index_in_simdgroup]],uint width[[threads_per_simdgroup]]) {
    uint token=group/s[2],row=group%s[2];
    if(token>=s[0])return;
    float2 sum=float2(0.0f);
    for(uint column=lane;column<s[1];column+=width) {
        uint start=64+row*s[3]+(column/64)*36;
        device const ushort* metadata=reinterpret_cast<device const ushort*>(w+start);
        float scale=float(as_type<half>(metadata[0])),b=float(as_type<half>(metadata[1]));
        uint code=(w[start+4+(column%64)/2]>>(4*(column%2)))&15;
        float weight=scale*float(code)+b,input=float(x[token*s[4]+column]);
        float product=weight*input;
        sum=compensated_add(sum,product);
        sum=compensated_add(sum,fma(weight,input,-product));
    }
    sum=compensated_reduce(sum,lane,width);
    if(lane==0) {
        sum=compensated_add(sum,float(bias[row]));
        checked_half(out,token*s[5]+row,compensated_store_value(sum),status);
    }
}

// Same compensated operation sequence per lane/token as q4_compensated, but
// reconstruct each weight once for an eight-token tile. Fixed indices keep
// independent high/low sums in registers. Partial tiles never access padding.
kernel void q4_compensated_prefill(device const uchar* w[[buffer(0)]],device const half* bias[[buffer(1)]],device const half* x[[buffer(2)]],device half* out[[buffer(3)]],constant uint* s[[buffer(4)]],device atomic_uint* status[[buffer(5)]],uint group[[threadgroup_position_in_grid]],uint lane[[thread_index_in_simdgroup]],uint width[[threads_per_simdgroup]]) {
    uint first=(group/s[2])*8,row=group%s[2];
    if(first>=s[0])return;
    uint active=min(8u,s[0]-first);
    float2 sums[8]={float2(0.0f),float2(0.0f),float2(0.0f),float2(0.0f),float2(0.0f),float2(0.0f),float2(0.0f),float2(0.0f)};
    for(uint column=lane;column<s[1];column+=width) {
        uint start=64+row*s[3]+(column/64)*36;
        device const ushort* metadata=reinterpret_cast<device const ushort*>(w+start);
        float scale=float(as_type<half>(metadata[0])),b=float(as_type<half>(metadata[1]));
        uint code=(w[start+4+(column%64)/2]>>(4*(column%2)))&15;
        float weight=scale*float(code)+b;
        #pragma unroll
        for(uint token=0;token<8;token++) {
            if(token<active) {
                float input=float(x[(first+token)*s[4]+column]),product=weight*input;
                sums[token]=compensated_add(sums[token],product);
                sums[token]=compensated_add(sums[token],fma(weight,input,-product));
            }
        }
    }
    #pragma unroll
    for(uint token=0;token<8;token++) {
        if(token<active) {
            float2 sum=compensated_reduce(sums[token],lane,width);
            if(lane==0) {
                sum=compensated_add(sum,float(bias[row]));
                checked_half(out,(first+token)*s[5]+row,compensated_store_value(sum),status);
            }
        }
    }
}

// V2 group8/record8 specialization of the unchanged compensated arithmetic.
// Format selection occurs on the host; no per-column version branch.
kernel void q4_v2_compensated(device const uchar* w[[buffer(0)]],device const half* bias[[buffer(1)]],device const half* x[[buffer(2)]],device half* out[[buffer(3)]],constant uint* s[[buffer(4)]],device atomic_uint* status[[buffer(5)]],uint group[[threadgroup_position_in_grid]],uint lane[[thread_index_in_simdgroup]],uint width[[threads_per_simdgroup]]) {
    uint token=group/s[2],row=group%s[2];
    if(token>=s[0])return;
    float2 sum=float2(0.0f);
    for(uint column=lane;column<s[1];column+=width) {
        uint start=64+row*s[3]+(column/8)*8;
        device const ushort* metadata=reinterpret_cast<device const ushort*>(w+start);
        float scale=float(as_type<half>(metadata[0])),b=float(as_type<half>(metadata[1]));
        uint code=(w[start+4+(column%8)/2]>>(4*(column%2)))&15;
        float weight=scale*float(code)+b,input=float(x[token*s[4]+column]);
        float product=weight*input;
        sum=compensated_add(sum,product);
        sum=compensated_add(sum,fma(weight,input,-product));
    }
    sum=compensated_reduce(sum,lane,width);
    if(lane==0) {
        sum=compensated_add(sum,float(bias[row]));
        checked_half(out,token*s[5]+row,compensated_store_value(sum),status);
    }
}

// Same compensated operation sequence per lane/token as q4_compensated, but
// reconstruct each weight once for an eight-token tile. Fixed indices keep
// independent high/low sums in registers. Partial tiles never access padding.
kernel void q4_v2_compensated_prefill(device const uchar* w[[buffer(0)]],device const half* bias[[buffer(1)]],device const half* x[[buffer(2)]],device half* out[[buffer(3)]],constant uint* s[[buffer(4)]],device atomic_uint* status[[buffer(5)]],uint group[[threadgroup_position_in_grid]],uint lane[[thread_index_in_simdgroup]],uint width[[threads_per_simdgroup]]) {
    uint first=(group/s[2])*8,row=group%s[2];
    if(first>=s[0])return;
    uint active=min(8u,s[0]-first);
    float2 sums[8]={float2(0.0f),float2(0.0f),float2(0.0f),float2(0.0f),float2(0.0f),float2(0.0f),float2(0.0f),float2(0.0f)};
    for(uint column=lane;column<s[1];column+=width) {
        uint start=64+row*s[3]+(column/8)*8;
        device const ushort* metadata=reinterpret_cast<device const ushort*>(w+start);
        float scale=float(as_type<half>(metadata[0])),b=float(as_type<half>(metadata[1]));
        uint code=(w[start+4+(column%8)/2]>>(4*(column%2)))&15;
        float weight=scale*float(code)+b;
        #pragma unroll
        for(uint token=0;token<8;token++) {
            if(token<active) {
                float input=float(x[(first+token)*s[4]+column]),product=weight*input;
                sums[token]=compensated_add(sums[token],product);
                sums[token]=compensated_add(sums[token],fma(weight,input,-product));
            }
        }
    }
    #pragma unroll
    for(uint token=0;token<8;token++) {
        if(token<active) {
            float2 sum=compensated_reduce(sums[token],lane,width);
            if(lane==0) {
                sum=compensated_add(sum,float(bias[row]));
                checked_half(out,(first+token)*s[5]+row,compensated_store_value(sum),status);
            }
        }
    }
}
