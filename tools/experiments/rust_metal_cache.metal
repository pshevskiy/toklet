#include <metal_stdlib>
using namespace metal;
// Lossless raw UInt16 Half-storage layout write, preserving zero/subnormal bits. K already contains rotary output; V does
// not rotate. Session validity/commit is outside this diagnostic primitive.
kernel void toklet_cache_write(device const ushort* k [[buffer(0)]],
                               device const ushort* v [[buffer(1)]],
                               device ushort* cache_k [[buffer(2)]],
                               device ushort* cache_v [[buffer(3)]],
                               constant uint* shape [[buffer(4)]],
                               uint index [[thread_position_in_grid]]) {
    uint rows=shape[0],offset=shape[1],capacity=shape[2];
    if(index>=rows*16) return;
    uint token=index/16,head=(index/8)%2,feature=index%8;
    uint target=(head*capacity+offset+token)*8+feature;
    cache_k[target]=k[index];cache_v[target]=v[index];
}
