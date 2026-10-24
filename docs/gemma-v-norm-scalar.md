# Scalar unlearned V-norm mapping

The existing scalar `rmsnorm` now passes the
[frozen unlearned V fixture](gemma-v-norm-reference.md) when supplied an immutable
unit-scale vector. This is a tested numerical recipe, with no new production
kernel, model adapter, capability or ABI. The
[acceptance receipt](../benchmarks/results/2026-10-24-v-norm-scalar-v1.json)
binds the exact test/source/fixture/check evidence.

Flatten `[batch,tokens,kv_heads,D]` into B×T×Hkv independent D-wide rows and call
the existing direct-scale RMS operation with `[D]` exact-one weights. FP64
mean-square reduction, positive finite epsilon and FP64 normalization round once
to FP32 output. No norm across heads/tokens or sum-only L2 reduction is introduced.
The source fixture's epsilon is f32(1e−6). Existing kernel validation and overflow
behavior remain unchanged.

The test shares one immutable512-element heap vector across all rows, heads,
chunks and both D256/D512 cases, borrowing only the required prefix. A future
Gemma adapter must establish and preserve exact-one ownership and count that
2KiB allocation once, separately from checkpoint weights, sessions and scratch.
This task does not add that adapter or its allocation plan. No per-row vector,
projection, KV transpose, normalized-V storage or attention graph is introduced.

Three [methods](../tests/gemma_v_norm.rs) authenticate the prior accepted receipt
before parsing the fixture, compare seven nonempty cases against ideal/staged
contracts, and compare staged FP32 output bits exactly. Repeated independent
head-row chunks reproduce full output with the same allocation pointers and
immutable input/scale bits. Hand mean-vs-sum, signed zero, finite-FP32 extremes
and subnormal normalization are also checked; these robustness cases do not
claim upstream fast-reduction parity. Invalid epsilon, geometry/size overflow,
length and late nonfinite inputs preserve output and borrowed data.

The first copied-cache full run is retained as failed and uncredited: an inherited
CLI test embedded the preceding target's absolute executable path. Both the
failed test and exact probed parent executable are retained. Cleaning only
Toklet package artifacts in the new private target fixes provenance without
source/test/tolerance changes. The next CPU suite passes; its collector then
rejects the legitimate cfg-disabled empty default MLX-CLI harness. That failed
collection is also retained. The accepted chain inherits only its four individually
successful checks and completes the remaining three with unchanged sources.
Post-CPU/pre-Metal snapshots corroborate the current executable paths; seven
CLI-spawning integration binaries point into this target, while the eighth is
independently listed as an empty default harness. Separate default/Metal engine
snapshots and parent/protected hashes remain bound.

Formatting, strict default/Metal Clippy,260 CPU tests,282 Metal-feature tests
and both iOS library compile targets pass. Ignored hardware suites remain
unexecuted. All production numerical/shader sources are unchanged from the
parent revision. Model/session ownership, graph/Metal integration, BF16/quantized
storage, fit, quality, performance, phone and long-reply gates remain open,
including the [upstream rotary precision gate](gemma-rotary-precision.md).
