# M6 explicit longer-context admission component

15 October 2026. This task exposes experimental longer contexts for qualification;
it does not qualify long real-model prefixes, quality, performance or supported
release limits. M6 remains in progress.

Loading accepts an explicit context limit up to32,768, bounded by the source
configuration. Default loading and CPU sessions retain2,048. Metal admission
checks capacity, chunk, checked absolute offset plus capacity, model limit,
checked KV size and complete session memory before allocation. The6GiB ceiling
remains. Reference and Online are both available for matched later comparisons;
Reference continues to admit its larger score matrices rather than hiding them.

The CLI adds `--metal-attention reference|online`. Reference remains default;
Online adds an actual `attention_mode: Online` execution field. Existing default
and explicit Reference execution schemas remain unchanged. CPU use, unknown or
duplicate choices and excessive context fail. Metal permits an explicitly
requested capacity through32K; token-file reads are bounded by the actual
capacity, and output reservation still occurs before inference/state mutation.
No CPU, projection, readback or chunk default changes.

The [component receipt](../../benchmarks/results/2026-10-15-m6-long-admission-component-v1.json)
binds a full source snapshot, check logs and archived CLI collection/replay input.
Formatting, strict CPU/Metal all-target Clippy,97 CPU tests and110 ordinary
Metal-feature tests pass. Explicit GPU checks cover:

- Allocation at4096/8192/16384/32768 using tiny geometry, exact KV formulas,
  depth-independent Online scratch, calculated Reference scratch differences,
  exact and one-byte-low budgets, count/offset overflow, one-token-over-capacity
  rejection with unchanged state, and short actual suffix execution at high
  absolute rotary positions. This does not build a32K real-model prefix.
- Independent FP64 split-half rotary math on exact Half inputs, dimensions64/128,
  six offsets through absolute position32767:6912 values pass the envelope frozen
  before collection. Maximum observed absolute error is0.003663682704437843.
  The envelope is
  `(|a|+|b|)*(8*f32::EPSILON*(abs(angle)+1)+0.0005)+1e-5`.
  This is a predeclared numerical envelope motivated by FP32 angle/frequency and
  Half rounding effects; it is not a proven vendor transcendental error bound,
  an unchanged model quality gate or a positional extension claim.
- Six serial native CLI workers,122 exact raw events: unchanged default and
  explicit Reference behavior, Online at all four capacities, independent
  readiness/statistics checks, greedy IDs, cached/pending continuation, invalid
  IDs, reset and capacity-plus-output rejection preserving the valid prefix.
  All workers exit0 with empty cleanup errors and unchanged endpoint pins.

Maximum complete admission in this tiny CLI collection is205,869,900 bytes;
maximum joint owned-worker/driver native lifetime charge is31,737,056 bytes.
Admission counts resident allocations independently of native charge, which may
omit shared/clean pages. These figures do not characterize real Qwen memory.

Astra accepted the final code and bound evidence with no remaining findings.
Retain this explicit qualification interface. Next are matched real-model
Reference/Online loss and workload collections, long-prefix behavior and memory,
attention profiling, and a conditional KV precision decision. The pinned models'
32K configuration limits alone do not establish supported inference depths.
