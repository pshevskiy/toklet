# Shared physical KV sizing and capacity core

M8 B0, 28 October 2026. Retained after implementation checks and independent
Astra review. This is the first task in the [active shared-block plan](shared-block-model-plan.md).
The [compact evidence](../benchmarks/results/2026-10-28-capacity-core-v1.json)
binds the exact five source files, raw check receipts and reviewer receipt.

## Implementation

`src/capacity.rs` contains cold checked arithmetic. `KvOwner` describes a
physical K/V producer: KV heads, head dimension, implemented F16/F32 storage,
full/sliding retention, per-array guards and producer count. Each separate array
uses head-major `[kv_heads, retained_capacity, head_dim]` storage. Shared
consumers are not additional producers. It checks multiplication/addition and
the per-array `isize` bound before allocation. This descriptor deliberately
covers the current equal-width K/V owners; it is not an MLA/recurrent or
compressed-cache representation.

Dense CPU cache sizing, Metal F16 sizing and existing FP32 window/shared-owner
planning use this same arithmetic. Their original allocation totals, guards,
metadata and staging rules are preserved. Scratch and immutable weights remain
separate costs; a sliding retained cache does not remove new-chunk staging.

The pure `capacity::search` function accepts a checked constructor recipe,
already implemented capacity interval, numerical budget and per-buffer ceiling.
For nondecreasing footprints it returns the largest fitting capacity and an
exact adjacent excluded witness, or the minimum witness when none fits. Planner
errors propagate; overflow is not treated as an allocation probe. Exhausting the
supplied interval means its upper bound fits, not that a model maximum was found.

Metal planning resolves the five fixed activation/MPS layouts once, then uses
the same capacity-dependent constructor equations for each search candidate.
The additive library function `metal::planning::inspect_mlx_capacity` returns
the existing resource report plus an optional capacity range. Existing inspect
functions and their schema remain unchanged. Each inspection still resolves
the device and pipelines once; the bisection itself performs arithmetic only.

The range applies to a fixed chunk, offset and modes, within the requested
already admitted upper capacity. Its minimum is the fixed chunk size. A caller
must separately honor both source/resident loading flags, account for process
and app owners, and pass fresh factory admission before allocation. A fitting
range neither reserves memory nor certifies graph, text or numerical support.

## Validation and decision

Formatting, strict default/Metal all-target Clippy, default tests and Metal
library tests pass: 375 default and 254 Metal-library tests, with overlap.
The release tiny-GPU integration binary builds and all three explicit tests
pass: frozen tied/untied graph oracles, pending/token/state behavior, and exact
memory/profile boundaries. These tiny tests peak at 26,394,624 bytes RSS and
21,987,904 bytes physical footprint; these are test-process observations, not
real-model memory claims. No real model or iPhone suite ran for this task.

Planning checks include literal prior F16/F32 totals, shared producers,
window/metadata accounting, invalid/overflow inputs, 4000 exhaustive solver
budget/buffer combinations and exact constructor equality across attention and
readback modes. The device-edge test selects 257 and rejects 258 because the
buffer limit is crossed. A test-only small vocabulary ensures capacity-dependent
KV/score buffers determine this edge instead of constant vocabulary scratch.

Astra independently authenticated all five source files, both raw receipts and
their outputs, and verified all nine recorded process groups closed. No blockers
remain. Retain the small planning core: it supplies the future model/report
foundation while leaving numerical execution, eligibility, caps and defaults
unchanged. New architecture composition is B1; user controls follow in B5.
