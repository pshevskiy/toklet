// Appended to the unchanged residual library; no floating-point operations.
kernel void gemma_pool_audit_fill_v1(
    device const uint* spans [[buffer(0)]], device const uint* work [[buffer(1)]],
    device const uint* active [[buffer(2)]], device uint* pool [[buffer(3)]],
    device uint* receipts [[buffer(4)]], constant uint* shape [[buffer(5)]],
    threadgroup uint* scratch [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    if (group.x >= shape[0]) return;
    const uint sentinel = 0x5a170031u;
    uint wi = group.x * 4u;
    uint span = work[wi], begin = work[wi + 1u], count = work[wi + 2u];
    bool valid = span < 32u && count > 0u && count <= 1024u && work[wi + 3u] == 1u
        && shape[1] > 0u && shape[1] <= 6594784u && shape[0] <= 6472u
        && shape[2] <= 1u && shape[3] == sentinel;
    uint start = 0u, capacity = 0u, kind = 0u, used = 0u;
    if (valid) {
        uint si = span * 4u;
        start = spans[si]; capacity = spans[si + 1u]; kind = spans[si + 2u];
        used = active[span];
        valid = spans[si + 3u] == 1u && kind <= 1u && start <= shape[1];
        if (valid) valid = capacity <= shape[1] - start;
        if (valid) valid = 7u <= shape[1] - start - capacity;
        if (valid) valid = used <= capacity && begin <= capacity + 7u;
        if (valid) valid = count <= capacity + 7u - begin;
    }
    // All lanes in this group take the same branch, before any pool read/fill.
    if (!valid) {
        if (lane == 0u) receipts[group.x] = 4u;
        return;
    }
    uint flags = 0u;
    for (uint j = lane; j < count; j += threads.x) {
        uint relative = begin + j, address = start + relative;
        uint word = pool[address];
        if (relative >= used) {
            if (word != sentinel) flags |= 1u;
        } else if (kind == 1u && (word & 0x7f800000u) == 0x7f800000u) {
            flags |= 2u;
        }
        // Check the old word first. This same invocation performs retirement.
        if (shape[2] != 0u) pool[address] = sentinel;
    }
    scratch[lane] = flags;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Works for every positive queried width, including non-power-of-two tails.
    for (uint stride = 1u; stride < threads.x; stride *= 2u) {
        if (lane % (2u * stride) == 0u && lane + stride < threads.x)
            scratch[lane] |= scratch[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) receipts[group.x] = scratch[0];
}
