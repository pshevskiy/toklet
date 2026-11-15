#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
// Status: invalid1, activation2, product4, NEEDS_REFERENCE8, table-contract16.
// Lane flags: fine-tail1,admitted2,reference4,identity8,constant16,reference-only32,placeholder64.
// Table:138241records; intercept,slope,E64,E32,mode,version (6u32).
constant float MIN_NORMAL = 0x1.0p-126f;
float widen_up(float x) {
    if (!isfinite(x)) return x;
    if (x == 0.0f) return MIN_NORMAL;
    if (abs(x) < MIN_NORMAL) return x < 0.0f ? 0.0f : MIN_NORMAL;
    float result = as_type<float>(as_type<uint>(x) + (x > 0.0f ? 1u : uint(-1)));
    return result<0.0f && abs(result)<MIN_NORMAL ? 0.0f : result;
}
float widen_down(float x) {
    if (!isfinite(x)) return x;
    if (x == 0.0f) return -MIN_NORMAL;
    if (abs(x) < MIN_NORMAL) return x < 0.0f ? -MIN_NORMAL : 0.0f;
    float result=as_type<float>(as_type<uint>(x) + (x > 0.0f ? uint(-1) : 1u));
    return result>0.0f && result<MIN_NORMAL ? 0.0f : result;
}
bool interval_ok(float lo, float hi) { return isfinite(lo) && isfinite(hi) && lo <= hi; }
bool admitted(float value, float lo, float hi) {
    if (!isfinite(value) || !interval_ok(lo,hi)) return false;
    float error = widen_up(max(abs(value-lo),abs(value-hi)));
    float magnitude = lo<=0.0f && hi>=0.0f ? 0.0f : min(abs(lo),abs(hi));
    // Both constants are strictly no larger than original binary64 constants.
    // Downward multiplication/addition prevents a rounded-up tolerance rescue.
    float relative = widen_down(as_type<float>(0x38d1b717u) * magnitude);
    float allowance = widen_down(as_type<float>(0x3727c5acu) + relative);
    return isfinite(error) && allowance >= 0.0f && error <= allowance;
}
uint record_for(uint bits) {
    uint m=bits&0x7fffffffu;
    if((bits>>31)!=0u && m>=0x40000000u && m<=0x41000000u)
        return 130048u+((bits-0xc0000000u)>>11);
    uint coarse=m>>15;
    return (bits>>31)==0u ? coarse : 65280u+coarse-(coarse>=33280u ? 512u : 0u);
}
uint delta_for(uint bits) {
    uint m=bits&0x7fffffffu;
    return (bits>>31)!=0u && m>=0x40000000u && m<=0x41000000u
        ? (bits-0xc0000000u)&2047u : m&32767u;
}
bool subnormal_bits(uint bits) {uint m=bits&0x7fffffffu;return m!=0u&&m<0x00800000u;}
void placeholder(device float* activated,device float* output,uint item,uint upbits) {
    // Preserve the separate product boundary even for speculation: +0*up has
    // up's sign. The pre-placeholder approximation stays in diagnostic slots.
    reinterpret_cast<device uint*>(activated)[item]=0u;
    reinterpret_cast<device uint*>(output)[item]=upbits&0x80000000u;
}
kernel void gated_gelu(
    device const float* gate [[buffer(0)]],device const float* up [[buffer(1)]],
    device float* activated [[buffer(2)]],device float* output [[buffer(3)]],
    constant uint& elements [[buffer(4)]],device atomic_uint* status [[buffer(5)]],
    device const uint* table [[buffer(6)]],device uint* lane_flags [[buffer(7)]],
    device float* intervals [[buffer(8)]],constant uint& force_reference [[buffer(9)]],
    uint item [[thread_position_in_grid]]) {
    if(item>=elements)return;
    uint xb=reinterpret_cast<device const uint*>(gate)[item],ub=reinterpret_cast<device const uint*>(up)[item];
    placeholder(activated,output,item,ub);lane_flags[item]=0;
    for(uint n=0;n<8;++n)intervals[item*8+n]=0.0f;
    if((xb&0x7f800000u)==0x7f800000u || (ub&0x7f800000u)==0x7f800000u) {
        atomic_fetch_or_explicit(status,1u,memory_order_relaxed);return;
    }
    uint record=record_for(xb),delta=delta_for(xb),base=record*6u;
    uint mode=table[base+4],version=table[base+5];
    if(record>=138241u||mode>3u||version!=1u) {
        atomic_fetch_or_explicit(status,16u,memory_order_relaxed);return;
    }
    uint flags=record>=130048u ? 1u : 0u;
    if(mode==3u) {
        lane_flags[item]=flags|32u|64u|4u;
        atomic_fetch_or_explicit(status,8u,memory_order_relaxed);return;
    }
    float c=as_type<float>(table[base]),m=as_type<float>(table[base+1]);
    float e64=as_type<float>(table[base+2]),e32=as_type<float>(table[base+3]);
    if(!isfinite(c)||!isfinite(m)||!isfinite(e64)||!isfinite(e32)||e64<0.0f||e32<0.0f) {
        atomic_fetch_or_explicit(status,16u,memory_order_relaxed);return;
    }
    float x=as_type<float>(xb),u=as_type<float>(ub),a;
    bool exact=mode==1u||mode==2u;
    bool uncertain=subnormal_bits(xb)||subnormal_bits(ub)||force_reference!=0u;
    float l64,h64,l32,h32;
    if(mode==2u) {a=x;l64=x;h64=x;l32=x;h32=x;flags|=8u;}
    else if(mode==1u) {
        a=c;flags|=16u;l32=c;h32=c;
        if(e64==0.0f){l64=c;h64=c;}
        else{l64=widen_down(c-e64);h64=widen_up(c+e64);}
    } else {
        a=fma(m,float(delta),c);float product=m*float(delta);
        uncertain|=subnormal_bits(as_type<uint>(product))||(m!=0.0f&&delta!=0u&&product==0.0f);
        float lo=widen_down(widen_down(product)+c),hi=widen_up(widen_up(product)+c);
        uncertain|=!interval_ok(lo,hi)||a<lo||a>hi;
        l64=widen_down(lo-e64);h64=widen_up(hi+e64);
        l32=widen_down(lo-e32);h32=widen_up(hi+e32);
    }
    float p=a*u;
    intervals[item*8]=l64;intervals[item*8+1]=h64;
    intervals[item*8+2]=l32;intervals[item*8+3]=h32;
    intervals[item*8+6]=a;intervals[item*8+7]=p;
    if(!isfinite(a)) {
        if(exact){atomic_fetch_or_explicit(status,2u,memory_order_relaxed);return;}
        lane_flags[item]=flags|64u|4u;atomic_fetch_or_explicit(status,8u,memory_order_relaxed);return;
    }
    if(!isfinite(p)) {
        if(exact){atomic_fetch_or_explicit(status,4u,memory_order_relaxed);return;}
        // Approximation overflow is not proof of reference overflow. Keep
        // finite memory-safe placeholders and replay the entire transaction.
        lane_flags[item]=flags|64u|4u;atomic_fetch_or_explicit(status,8u,memory_order_relaxed);return;
    }
    uncertain|=subnormal_bits(as_type<uint>(a))||subnormal_bits(as_type<uint>(p))
        ||(a!=0.0f&&u!=0.0f&&p==0.0f);
    float lp,hp;
    if(exact&&!uncertain) {lp=p;hp=p;}
    else if(exact&&(a==0.0f||u==0.0f)) {lp=p;hp=p;}
    else if(u>=0.0f){lp=widen_down(l32*u);hp=widen_up(h32*u);}
    else{lp=widen_down(h32*u);hp=widen_up(l32*u);}
    float left=l32*u,right=h32*u;
    uncertain|=subnormal_bits(as_type<uint>(left))||subnormal_bits(as_type<uint>(right))
        ||(l32!=0.0f&&u!=0.0f&&left==0.0f)||(h32!=0.0f&&u!=0.0f&&right==0.0f);
    intervals[item*8+4]=lp;intervals[item*8+5]=hp;
    uncertain|=!admitted(a,l64,h64)||!admitted(a,l32,h32)||!admitted(p,lp,hp);
    if(uncertain){
        lane_flags[item]=flags|4u;
        // Sensitive a/up/p may have flushed. Do not publish an unproved value;
        // retain its raw diagnostics, and preserve exact placeholder product.
        if(subnormal_bits(xb)||subnormal_bits(ub)||subnormal_bits(as_type<uint>(a))||subnormal_bits(as_type<uint>(p))||(a!=0.0f&&u!=0.0f&&p==0.0f)){
            placeholder(activated,output,item,ub);lane_flags[item]|=64u;
        }else{activated[item]=a;output[item]=p;}
        atomic_fetch_or_explicit(status,8u,memory_order_relaxed);
    } else {
        if(mode==2u)reinterpret_cast<device uint*>(activated)[item]=xb;
        else if(mode==1u)reinterpret_cast<device uint*>(activated)[item]=table[base];
        else activated[item]=a;
        output[item]=p;lane_flags[item]=flags|2u;
    }
}
kernel void validate_fma(device const uint* points [[buffer(0)]],device const uint* table [[buffer(1)]],
                         device uint* result [[buffer(2)]],constant uint& count [[buffer(3)]],
                         uint item [[thread_position_in_grid]]) {
    if(item>=count)return;
    uint bits=points[item*4],record=points[item*4+1],delta=points[item*4+2],base=record*6u,mode=table[base+4];
    float c=as_type<float>(table[base]),m=as_type<float>(table[base+1]);
    float a,lo,hi;
    if(mode==0u){a=fma(m,float(delta),c);float p=m*float(delta);lo=widen_down(widen_down(p)+c);hi=widen_up(widen_up(p)+c);}
    else if(mode==1u){a=c;lo=c;hi=c;}
    else if(mode==2u){a=as_type<float>(bits);lo=a;hi=a;}
    else{a=0.0f;lo=0.0f;hi=0.0f;}
    result[item*4]=as_type<uint>(a);result[item*4+1]=as_type<uint>(lo);result[item*4+2]=as_type<uint>(hi);
    result[item*4+3]=(record_for(bits)<<4)|mode|uint(a>=lo&&a<=hi)*8u;
}
kernel void validate_rounding(device const uint* input [[buffer(0)]],device uint* output [[buffer(1)]],
                              constant uint& count [[buffer(2)]],uint item [[thread_position_in_grid]]) {
    if(item>=count)return;float x=as_type<float>(input[item]);
    output[item*2]=as_type<uint>(widen_down(x));output[item*2+1]=as_type<uint>(widen_up(x));
}
kernel void validate_multiply(device const uint* input [[buffer(0)]],device uint* output [[buffer(1)]],
                              constant uint& count [[buffer(2)]],uint item [[thread_position_in_grid]]) {
    if(item>=count)return;uint ab=input[item*2],bb=input[item*2+1];
    float p=as_type<float>(ab)*as_type<float>(bb);uint pb=as_type<uint>(p);
    bool sensitive=subnormal_bits(ab)||subnormal_bits(bb)||subnormal_bits(pb)
       ||((ab&0x7fffffffu)!=0u&&(bb&0x7fffffffu)!=0u&&(pb&0x7fffffffu)==0u);
    output[item*2]=pb;output[item*2+1]=uint(sensitive);
}
