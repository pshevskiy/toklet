# LFM3 BF16 eager CPU graph and authenticated weights

Compare the admitted 24-layer BF16 candidate against layer-streamed Torch
2.9.1 CPU operations following the exact pinned modified-parent eager graph.
Use the authenticated child payload whose 286 tensors equal the parent under
13 singleton convolution-axis transforms. Read weights through bounded streams.
Reference processes verify all 28,982 pinned runtime files at both endpoints
and stay within the separate 512 MiB component/reference audit cap.

Use one unpadded sequence of token IDs [124894, 42, 1234, 567]. Freeze schedules
[4], [2,1,1] and [1,3]. Compare exact BF16 digests of each of 24 layer outputs,
the final embedding norm and last-row logits independently for each chunk.
Record actual conv history and head-major K/V proposals separately. Preserve
the prior component, original full-model numerical, quality and memory gates.
The fixed graph uses pre- and post-operator norms, two BF16 residual adds per
layer, gate/up BF16 stores, seven-tap convolution and full or W256 attention.

Native working-set admission includes one BF16 weight bank, positional
coefficients, committed state, per-owner proposals, reusable numerical scratch,
committed token history and both logit buffers. The 6 GiB Mac ceiling holds.
Native allocation and reserve bounds are accounted before weights are delivered.
Reject invalid token IDs/capacity before numerical writes. Failed arithmetic or
trace leaves the prior published logits, tokens and all state unchanged. Measure
load separately from prepared single-token and short-chunk execution. This
scope is a real model numerical baseline, with codec, CLI policy and broader
quality/adoption qualification still required.
