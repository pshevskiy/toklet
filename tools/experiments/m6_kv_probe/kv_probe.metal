// Isolated component v1: unchanged online recurrence; only K/V loads differ.
// Include after the exact original kernels.metal (for checked_half).
template <bool Checked>
float dequant_i8(device const ushort* codes, device const ushort* scales,
                 uint index, device atomic_uint* status) {
    uint byte=(uint(codes[index/2])>>((index%2)*8))&255u;
    int code=byte>=128u?int(byte)-256:int(byte);
    ushort scale=scales[index/32];
    if(Checked && (code==-128 || (scale&0x8000u)!=0 || (scale&0x7c00u)==0x7c00u || (scale==0 && code!=0))) {
        atomic_fetch_or_explicit(status,2u,memory_order_relaxed);
        return 0.0f;
    }
    return float(code)*float(as_type<half>(scale));
}
template <uint R, bool Checked>
void online_attention_i8(device const half* q, device const ushort* k,
                      device const ushort* v, device const ushort* ks,
                      device const ushort* vs, device half* out,
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
                float key=f<d?dequant_i8<Checked>(k,ks,(kv*cap+tile+p)*d+f,status):0.0f;
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
                    if(f<d)numerator[r][j]+=weight*dequant_i8<Checked>(v,vs,(kv*cap+tile+p)*d+f,status);
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
kernel void attention_decode_i8_probe(device const half* q[[buffer(0)]],device const ushort* k[[buffer(1)]],device const ushort* v[[buffer(2)]],device const ushort* ks[[buffer(3)]],device const ushort* vs[[buffer(4)]],device half* out[[buffer(5)]],constant uint* s[[buffer(6)]],device atomic_uint* status[[buffer(7)]],uint group[[threadgroup_position_in_grid]],uint lane[[thread_index_in_simdgroup]],uint width[[threads_per_simdgroup]]) {
    online_attention_i8<1,false>(q,k,v,ks,vs,out,s,status,group,lane,width);
}
kernel void attention_decode_i8_probe_checked(device const half* q[[buffer(0)]],device const ushort* k[[buffer(1)]],device const ushort* v[[buffer(2)]],device const ushort* ks[[buffer(3)]],device const ushort* vs[[buffer(4)]],device half* out[[buffer(5)]],constant uint* s[[buffer(6)]],device atomic_uint* status[[buffer(7)]],uint group[[threadgroup_position_in_grid]],uint lane[[thread_index_in_simdgroup]],uint width[[threads_per_simdgroup]]) {
    online_attention_i8<1,true>(q,k,v,ks,vs,out,s,status,group,lane,width);
}
// Exact FP32-bit witnesses expose subnormal/sign reconstruction without a Half output cast.
template <bool Checked>
void decode_words(device const ushort* codes,device const ushort* scales,device uint* out,
                  constant uint* count,device atomic_uint* status,uint i) {
    if(i<count[0])out[i]=as_type<uint>(dequant_i8<Checked>(codes,scales,i,status));
}
kernel void i8_words_probe(device const ushort* c[[buffer(0)]],device const ushort* s[[buffer(1)]],device uint* o[[buffer(2)]],constant uint* n[[buffer(3)]],device atomic_uint* st[[buffer(4)]],uint i[[thread_position_in_grid]]) {decode_words<false>(c,s,o,n,st,i);}
kernel void i8_words_probe_checked(device const ushort* c[[buffer(0)]],device const ushort* s[[buffer(1)]],device uint* o[[buffer(2)]],constant uint* n[[buffer(3)]],device atomic_uint* st[[buffer(4)]],uint i[[thread_position_in_grid]]) {decode_words<true>(c,s,o,n,st,i);}
