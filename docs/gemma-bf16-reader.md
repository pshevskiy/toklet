# Official BF16 reader and bounded projection checks

The [cold target worker](../tools/reference/gemma4_bf16_rows_check.py) now
qualifies positive official-file binding for the [bounded BF16 reader](bf16-source.md)
against the previously frozen [independent row/scalar oracle](gemma-bf16-rows-oracle.md).
This is a Python reference path for the unquantized baseline. Whole-graph
arithmetic, upstream BF16 execution, model fit, quality, performance and
app/full-window support remain separate gates.

The exact source is the 10246621918-byte official checkpoint, SHA
`2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550`.
One retained authenticated descriptor backs all views. The fixed official layout
resolves each of the 540 selected tensor names, row-major shapes and file offsets.
All 60 unused shared-consumer tensors remain excluded. No new numerical kernel
or operator hierarchy is introduced: conversion and projection reuse existing
`affine4_tiles` operations.

## Frozen schedules and independent replay

Before launching the target, the
[stdlib checker](../tools/reference/check_gemma4_bf16_rows.py) derives the complete
schedule from the original oracle. The plan is 973550 bytes, SHA
`bb63f6e72690bfa959273b710a8315376e3958962d0b6e490febf6d81b77dc82`.
It fixes 1366 output captures and all 4004 ordered source reads, including
offsets, byte counts and hashes. The target does not produce expected values.

Each matrix's four sampled rows is accessed through grouped adjacent reads,
split single-row reads and repeated/nonmonotonic gathers. Complete vectors have
direct and repeated-gather checks. The 128-by-12288 maximum tile is read at once,
as two 64-row halves and as a repeated/nonmonotonic 128-row aggregate gather.
Rejected 129-row contiguous/gather requests leave the audit unchanged; a valid
read afterward proves recovery. Altered layout bytes, an unknown tensor, every
excluded tensor and a different authenticated source reject with intended errors.
There are 65 admission/rejection controls in total.

All 14061417 returned FP32 values match exact bits, shapes and dtypes. Two complete
dyadic-input projections run in both FP32 and FP64, with results saved before
comparison to each independent scalar reference. FP32 maximum absolute errors
are `2.9802322387695312e-8` and `7.142079994082451e-8`; both pass the unchanged
`1e-5 + 1e-4*abs(reference)` gate. FP64 outputs match the scalar values exactly
under their separately frozen `1e-13 + 1e-13*abs(reference)` gate. These two
stimuli do not qualify arbitrary projections or the model graph.

The independent checker replays saved target bytes, metadata, full coverage,
projection errors and the final source audit. The 4004 reads return 105193170
bytes; their ordered chain is
`60aea00c94af7f0a28e011702aafa751977cc2b24a87c74bde0cde33b0a7597b`.
The owner closes without poisoning. The chain binds consumed reads; it remains
separate from the original full-file authentication.

Five [saved-output corruption controls](../tools/reference/test_gemma4_bf16_rows_check.py)
coherently update applicable hashes and reject wrong row bits, dtype, audit,
finite out-of-tolerance projection output and NaN projection output at distinct
intended gates. They never open or modify source checkpoint files.

## Runtime, memory and scope

The shared runtime helper accepts explicit additional caller sources; its old
defaults remain unchanged. Thirteen ownership tests and the original 37-value
cold witness pass again. The official worker admits the complete pinned source,
Python/native and NumPy2.3.2/MLX0.32.3 runtime union before numerical imports,
then checks actual origins and native images at exit. MLX and text package files
are admitted but those packages do not execute in this check.

On Apple M1 Pro / MacBookPro18,3 / 16 GiB / macOS 26.6.2 (25G83), the completed
cold official worker peaks at 286606296 physical bytes and 308854784 RSS bytes,
below its 512 MiB diagnostic policy. This includes runtime admission, loaded
oracle data, conversions and projections. It is not whole-model memory fit or
generation performance. The 56288676-byte target output remains in ignored
`artifacts/m8-gemma-bf16-target-checks-v1/official/arrays.bin`; the pre-target
plan and result identities are in the [public record](../benchmarks/results/2026-10-26-gemma-bf16-reader-v1.json).

Core Rust/shaders/examples and protected release binaries remain unchanged;
no unrelated engine/GPU/device suites are claimed. The 128-row IO bound is
independent of numerical input/output context. Next compose original BF16
matrices and vectors through the existing graph weight interface, validate that
reference, and separate numerical context capacity from chunk scratch before
running the complete 1025-position quality workload. Existing precision failures,
resident scheduling and model-driven full-window/app/phone delivery remain open.
