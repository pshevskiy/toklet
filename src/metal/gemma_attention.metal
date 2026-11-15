#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
// Fixed ABI: Q,K,H,HK,D,q_start,k_start,window,n0,n1,n2,stride0,stride1,stride2,scale_bits.
// Q/context [Q,H,D]; score/probability [H,Q,K]; KV segments [HK,stride].
// Dispatches share one tracked command; no host reads between dependent kernels.
float error_add(float a,float b,float rounded) {
    return abs(a)>=abs(b) ? (a-rounded)+b : (b-rounded)+a;
}
void add3(thread float3& sum,float value) {
    float high=sum.x+value;
    float error=error_add(sum.x,value,high);
    float middle=sum.y+error;
    sum.z+=error_add(sum.y,error,middle);
    sum.x=high;sum.y=middle;
}
void product3(thread float3& sum,float a,float b) {
    float product=a*b;
    add3(sum,product);add3(sum,fma(a,b,-product));
}
float round3(float3 sum) {
    if(sum.z==0.0f)return sum.x+sum.y;
    float parts[3];uint count=0;
    for(uint term=0;term<3;++term) {
        float value=sum[term];uint kept=0;
        for(uint i=0;i<count;++i) {
            float next=value+parts[i];float error=error_add(value,parts[i],next);
            if(error!=0.0f)parts[kept++]=error;value=next;
        }
        parts[kept++]=value;count=kept;
    }
    float value=parts[--count];
    while(count!=0) {
        float part=parts[--count];float next=value+part;
        float error=error_add(value,part,next);value=next;
        if(error==0.0f)continue;
        if(count!=0 && ((error>0.0f && parts[count-1]>0.0f)||(error<0.0f && parts[count-1]<0.0f))) {
            float twice=error+error;float adjusted=value+twice;
            if(adjusted-value==twice)value=adjusted;
        }
        break;
    }
    return value;
}
device const float* row3(device const float* a,device const float* b,device const float* c,
                         constant uint* s,uint head,uint key) {
    if(key<s[8])return a+head*s[11]+key*s[4];
    key-=s[8];if(key<s[9])return b+head*s[12]+key*s[4];
    key-=s[9];return c+head*s[13]+key*s[4];
}
bool visible(constant uint* s,uint q,uint key) {
    uint position=s[6]+key,query=s[5]+q;
    return position<=query && (s[7]==0 || query-position<s[7]);
}
#define ARGS device const float* q [[buffer(0)]], device const float* k0 [[buffer(1)]], device const float* v0 [[buffer(2)]], device const float* k1 [[buffer(3)]], device const float* v1 [[buffer(4)]], device const float* k2 [[buffer(5)]], device const float* v2 [[buffer(6)]], device float* scores [[buffer(7)]], device float* probs [[buffer(8)]], device float* context [[buffer(9)]], constant uint* s [[buffer(10)]], device atomic_uint* status [[buffer(11)]]
void reject(device atomic_uint* status) {atomic_fetch_or_explicit(status,1u,memory_order_relaxed);}
kernel void validate_inputs(ARGS,uint item [[thread_position_in_grid]]) {
    if(item<s[0]*s[2]*s[4] && !isfinite(q[item]))reject(status);
    if(item<s[1]*s[3]*s[4]) {
        uint feature=item%s[4],key=(item/s[4])%s[1],head=item/(s[4]*s[1]);
        if(!isfinite(row3(k0,k1,k2,s,head,key)[feature]) || !isfinite(row3(v0,v1,v2,s,head,key)[feature]))reject(status);
    }
}
kernel void scores3(ARGS,uint group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint lanes [[threads_per_simdgroup]]) {
    uint key=group%s[1],item=group/s[1],query=item%s[0],head=item/s[0];
    if(head>=s[2])return;
    device const float* k=row3(k0,k1,k2,s,head/(s[2]/s[3]),key);
    uint base=(query*s[2]+head)*s[4];float3 total=float3(0.0f);
    for(uint f=lane;f<s[4];f+=lanes)product3(total,q[base+f],k[f]);
    for(uint hop=1;hop<lanes;hop*=2) {
        float a=simd_shuffle_down(total.x,hop),b=simd_shuffle_down(total.y,hop),c=simd_shuffle_down(total.z,hop);
        if(lane%(2*hop)==0 && lane+hop<lanes){add3(total,a);add3(total,b);add3(total,c);}
    }
    if(lane==0) {
        float value=round3(total)*as_type<float>(s[14]);
        scores[group]=value;if(!isfinite(value))reject(status);
    }
}
kernel void softmax3(ARGS,uint item [[thread_position_in_grid]]) {
    if(item>=s[0]*s[2])return;
    uint query=item%s[0],base=item*s[1];float maximum=-INFINITY;
    for(uint key=0;key<s[1];++key)if(visible(s,query,key))maximum=max(maximum,scores[base+key]);
    if(!isfinite(maximum)){reject(status);return;}
    float3 mass=float3(0.0f);
    for(uint key=0;key<s[1];++key) {
        float value=visible(s,query,key) ? exp(scores[base+key]-maximum) : 0.0f;
        probs[base+key]=value;add3(mass,value);
    }
    float denominator=round3(mass);
    if(!isfinite(denominator) || denominator<=0.0f){reject(status);return;}
    for(uint key=0;key<s[1];++key) {
        float value=probs[base+key]/denominator;probs[base+key]=value;
        if(!isfinite(value))reject(status);
    }
}
kernel void context3(ARGS,uint item [[thread_position_in_grid]]) {
    if(item>=s[0]*s[2]*s[4])return;
    uint feature=item%s[4],head=(item/s[4])%s[2],query=item/(s[4]*s[2]);
    uint base=(head*s[0]+query)*s[1],kvhead=head/(s[2]/s[3]);float3 total=float3(0.0f);
    for(uint key=0;key<s[1];++key)if(visible(s,query,key))
        product3(total,probs[base+key],row3(v0,v1,v2,s,kvhead,key)[feature]);
    float value=round3(total);context[item]=value;if(!isfinite(value))reject(status);
}
