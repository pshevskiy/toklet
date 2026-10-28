
// Unqualified Reference scheduling experiment. The three ordered dispatches
// preserve each Serial reduction/expression and all existing buffer layouts.
// Same ABI as attention: q/k/v/scores/probs/context, shape[8], atomic status.
kernel void attention_reference_scores_parallel(device const half* q[[buffer(0)]],device const half* k[[buffer(1)]],device const half* v[[buffer(2)]],device float* scores[[buffer(3)]],device float* probs[[buffer(4)]],device half* out[[buffer(5)]],constant uint* s[[buffer(6)]],device atomic_uint* status[[buffer(7)]],uint item[[thread_position_in_grid]]) {
    uint rows=s[0],heads=s[1],kvheads=s[2],d=s[3],cap=s[4],chunk=s[5],valid=s[7];
    if(item>=rows*heads*valid)return;
    uint query=item/valid,p=item%valid,token=query/heads,head=query%heads,kv=head/(heads/kvheads),base=(head*chunk+token)*cap;
    float scale=sqrt(float(d)),acc=0;
    for(uint f=0;f<d;++f)acc=acc+float(q[(token*heads+head)*d+f])*float(k[(kv*cap+p)*d+f]);
    float score=acc/scale;
    scores[base+p]=score;
}
kernel void attention_reference_softmax_serial(device const half* q[[buffer(0)]],device const half* k[[buffer(1)]],device const half* v[[buffer(2)]],device float* scores[[buffer(3)]],device float* probs[[buffer(4)]],device half* out[[buffer(5)]],constant uint* s[[buffer(6)]],device atomic_uint* status[[buffer(7)]],uint query[[thread_position_in_grid]]) {
    uint rows=s[0],heads=s[1],cap=s[4],chunk=s[5],offset=s[6],valid=s[7];
    if(query>=rows*heads)return;
    uint token=query/heads,head=query%heads,base=(head*chunk+token)*cap,causal=offset+token+1;
    float maximum=-INFINITY;
    for(uint p=0;p<valid;++p)if(p<causal)maximum=max(maximum,scores[base+p]);
    float sum=0;
    for(uint p=0;p<valid;++p) {
        float e=p<causal?exp(scores[base+p]-maximum):0.0f;
        probs[base+p]=e;
        sum=sum+e;
    }
    if(!isfinite(sum)||sum<=0)atomic_fetch_or_explicit(status,2u,memory_order_relaxed);
    for(uint p=0;p<valid;++p)probs[base+p]=probs[base+p]/sum;
}
kernel void attention_reference_context_parallel(device const half* q[[buffer(0)]],device const half* k[[buffer(1)]],device const half* v[[buffer(2)]],device float* scores[[buffer(3)]],device float* probs[[buffer(4)]],device half* out[[buffer(5)]],constant uint* s[[buffer(6)]],device atomic_uint* status[[buffer(7)]],uint item[[thread_position_in_grid]]) {
    uint rows=s[0],heads=s[1],kvheads=s[2],d=s[3],cap=s[4],chunk=s[5],offset=s[6];
    if(item>=rows*heads*d)return;
    uint query=item/d,f=item%d,token=query/heads,head=query%heads,kv=head/(heads/kvheads),base=(head*chunk+token)*cap,causal=offset+token+1;
    float acc=0;
    for(uint p=0;p<causal;++p)acc=acc+probs[base+p]*float(v[(kv*cap+p)*d+f]);
    checked_half(out,(token*heads+head)*d+f,acc,status);
}
