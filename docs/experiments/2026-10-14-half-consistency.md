# Larger Half batch consistency localization

The unchanged failed short probe first diverges at layer 0 Q projection. Complete
normalized inputs agree, but the Original MPS single-row and batched products
produce 15 different stored Half values. Isolated projection replay reproduces
both graph outputs exactly. This localizes the first discrepancy to the
shape-dependent MPS projection arithmetic, without claiming that it is the only
cause of the final NLL discrepancy or that a replacement passes quality gates.

## Fixed inputs and capture

The checkpoint, config, failed scorer record and tokens `[9707,11,358,498]` are
pinned to the [larger Half rejection](2026-10-14-larger-half-loss.md). Capacity
is1025; configured chunks 1/128 execute one/four active rows. Original projection,
Original RMS, Half KV and FP32 logits remain unchanged. One resident model serves
four sequential sessions: baseline diagnostics, then all-layer diagnostics for
each chunk mode. Each session is released and its autorelease pool drained before
the next. Every capture includes all four causal rows; the first three scored rows
reproduce the saved full-logit SHA, target bits and NLL independently.

The all-layer capture includes 451 stages in execution order and 56 complete
committed K/V prefixes, each `[2,4,128]`. Diagnostics preserve baseline embedding,
final normalization and logits bit-for-bit on both paths. All entries are bounded
exclusive arrays with dimensions, byte counts and SHA256 identities. This is
diagnostic execution; it is not a latency measurement or a full-corpus quality run.

## First divergence and independent arithmetic

Embedding and layer0 input normalization agree exactly. Layer0 Q has15 differing
values, first at row0/output274, maximum absolute difference0.001953125 and maximum
Half ULP distance15. Of451 stage pairs,449 differ at or after this point; all56
committed cache-prefix pairs differ. Downstream differences may combine propagated
error and additional independently shape-dependent operations; they require
separate qualification after any proposed arithmetic change.

Replay uses the complete identical `[4,1536]` input and resident `[1536,1536]`
Half Q weights with Half bias. It preserves transpose, alpha/beta, bias
initialization and the actual FP32 scratch stride151936, executing one row per
call at capacity1 and four rows at capacity128. Both stored replay outputs match
their original graph captures exactly.

The analyzer computes FP64 dot products and exact dyadic integer dot-plus-bias
sums for all15 differing coordinates. Exact sums are rounded directly to Half
with round-to-nearest/ties-even, including sign, subnormal and overflow boundaries.
Of the15 coordinates, single-row matches the exact Half result at6, batch at8,
and neither at1. This is not a reason to select either MPS shape as an oracle.

For example, row1/output142 has exact sum `4342487 / 2^35`, which rounds to Half
bits2084. Single-row produces2097 and batch2082. Maximum raw FP32 error against
FP64 over the15 selected output rows and all four inputs is
0.000005717411113437265 single-row and0.0000012176333257229999 batched. These are
selected-row diagnostic errors, not new acceptance tolerances.

## Bounds, checks and provenance

Maximum admission including the diagnostic extra256MiB is4,798,494,330 bytes.
Sampled conservative worker-plus-driver native lifetime charge is3,885,354,872
bytes; both fit6GiB. The owned worker completes exactly one ignored native test,
exits0, is reaped, and has no cleanup error. Endpoint source/config/code/binary
identities are unchanged. Analysis and endpoint hashing start after GPU completion.
Energy, continuous thermal and system-pressure evidence is unavailable.

The native test is `metal::half_consistency::larger_half_probe_capture`; production
code gains only its `cfg(test)` module hook. A frozen release test binary and
complete source/helper snapshot are retained at
`artifacts/m5-half-consistency-native-binary-v1/` and
`artifacts/m5-half-consistency-source-v1/`. Cargo uses only
`CARGO_TARGET_DIR=artifacts/m5-checks`; protected CPU binaries are unchanged.
Formatting, strict Clippy in both configurations,87 ordinary CPU tests,
97 ordinary Metal-feature tests and178 offline Python tests pass. The additional
Metal tests are shared frozen scorer unit checks; native GPU execution is explicit.

[Collection and independently recomputed analysis](../../benchmarks/results/2026-10-14-larger-half-consistency-v1.json),
[native stdout](../../benchmarks/results/2026-10-14-larger-half-consistency-v1.stdout),
[native stderr](../../benchmarks/results/2026-10-14-larger-half-consistency-v1.stderr),
and [receipt](../../benchmarks/results/2026-10-14-half-consistency-qualification-v1.json)
bind the sources, arrays, binary and checks. Astra reviews code and saved evidence
before commit, including actual diagnostic perturbation and replay-match results.

## Next mechanism

A bounded, explicit compensated Half projection is now a justified candidate.
It must first pass independent primitive rounding checks and complete tiny/selected
larger graph checks, then the unchanged short consistency guard before full loss
collection. Neither this localization nor a future consistency pass overrides the
fixed quality gates. CPU/scalar, Original modes, FullLogits and chunk128 remain
defaults. Both Q4 quality failures, M5.2/M5.6 and overall M5 remain open.
