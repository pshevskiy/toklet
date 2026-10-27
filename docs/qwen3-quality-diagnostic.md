# Explicit Qwen3 1025-position quality diagnostic

The native Rust engine now exposes
`Model::load_mlx_qwen3_quality_diagnostic` for the admitted Qwen3 0.6B geometry
at up to1025 positions. This is an experimental quality-execution resource
boundary. Ordinary loaders and compatibility inspection retain512; app limits,
known C512 numerical rejection and quality status remain unchanged.

The loader selects its private import variant once, before device creation.
It reuses all ordinary geometry, RoPE/epsilon, packed-weight, memory and session
checks. Other architectures and tiny Qwen3 are excluded. The audited SmolLM2
identity remains attached only by its existing dedicated import variant.
No numerical kernel, reduction, activation precision or cache layout changes.

Immutable packed eight-bit weights remain model-owned. Sessions own separate
per-layer Half K/V buffers `[8, capacity,128]`; at1025 positions all28 layers'
K/V require117555200 bytes. Scratch and optional all-row snapshots follow the
existing checked planner. Diagnostics admit GPU snapshots and bounded CPU copies
before execution; no second model copy is required.

## Actual checks on Apple M1 Pro

The release GPU witness uses the pinned tested checkpoint and unchanged
MlxAffine/Compensated/Reference/Half profile. It verifies:

- ordinary1025 and diagnostic1026 geometry rejection with exact error categories;
- capacity/offset overflow, exact session budget success and one-byte-short
  rejection with exact required/limit evidence;
- partial-prefill discard/resume, complete1025 commitment, cached reuse and
  per-operation state preservation after rejected append/replacement;
- exact final and1024-prefix distributions across128/37-row chunk schedules;
- unchanged preceding distribution at position1023 when the full final chunk
  includes a changed future token; this is one causal-prefix row witness;
- one sampled token remaining pending at the1025 boundary after cancellation,
  rejected continuation preserving history/lengths/unavailable logits, then
  successful commitment of that pending token.

The successful whole process peaks at1126729000 physical/992460800 RSS bytes,
below6GiB. Session logical admission is920447766 bytes for chunk128;
chunk37 with snapshots admits894147736 bytes. These are different resource
plans, not two simultaneous model copies. All owned build/test groups close and
source/model/reference endpoints pass.

CPU tests, formatting and strict default/Metal all-target Clippy pass. The
first controlled build was stopped after selecting an old Rust toolchain from
PATH; the retry explicitly pins installed Rust1.90. The first GPU run then
exposed a test assumption: pending sampled history correctly makes logits
unavailable. The test now checks `None` and its preservation. Both failed
attempts remain in the [public evidence](../benchmarks/results/2026-10-27-qwen3-quality-diagnostic-v1.json).
No engine behavior or gate was changed to pass either retry.

These checks establish diagnostic capacity/state behavior and self-consistency,
not independent1025 numerical qualification, full quality, latency or iPhone
qualification. The [active quality task](qwen3-mac-delivery-plan.md) still needs
the complete10000-target/100-task comparison against the pinned unquantized
baseline, with unchanged0.05-nat/2-percentage-point rules and a candidate frozen
before scoring. App delivery and broader40960-position qualification stay open.
