# Gemma borrowed Q4 GPU projection

16 November 2026. Private prerequisite for a resident common-Q4 consumer.
All required checks and the combined Astra code review pass. Astra approves
retention of this private prerequisite with no required fixes. Ordinary model/session/CLI selection is unchanged; the new module
is compiled only for macOS tests.

`Affine4MatrixView` borrows existing immutable `Weights` chunks. The cached
pipeline encodes one whole-matrix max/L1 admission followed by one product per
actual chunk, without submission, wait, staging, readback or publication. The
caller retains buffers/command through checked completion, validates guards and
statuses, and owns cancellation, whole-operation reference replay and unsafe
reuse poisoning. The unchanged synchronous `Projection::apply` on the same
owner is the oracle and prescribed fallback, not a second model.

Weights retain original low-nibble-first UInt32 codes, group64 and BF16
scale/bias metadata. Inputs/outputs remain FP32. The appended Metal entry
preserves the original float2 compensation, FMA reconstruction/residual, loop
and SIMD reduction byte-for-byte; only its name and final store address change
to `token*total_rows+first_row+local_row`. Queried execution widths must match.
Safe/Precise Metal2.3 and the original contract-off source remain.

Cold proof uses the loader's maximum over all16 codebook values, including
unused codes. This is a general exact FP32 maximum, never rounded to BF16.
Threshold searches use unchanged host max/L1 bounds and check endpoints.
The original admission shader is unchanged. Before the first encoder, all
actual chunk layouts, guarded physical ranges, lengths, offsets, device/command state
and writable aliases are checked. Read-only sharing is allowed. The complete
output is `[tokens,total_rows]`; bounds are rows1..262144, columns64..12288
in multiples64, tokens1..32, at most64 chunks of at most16384 rows.

Unknown/guard contracts are fatal. Nonfinite-input status precedes uncertainty;
a proof miss requests whole-operation replay before speculative product failure.
Encoding never resets caller status. No automatic model replay owner is added.
Natural accepted cases needed no successful reference replay; forced request,
sticky status and caller-output controls cover the primitive's decision contract.

Actual qualification on Apple M1 Pro passes2 pure and6 new GPU tests: original
31 families,19L1 families and6 native pipeline observations;16 complete dyadic
shapes including16383/16384/16385 boundaries, T1/T32 and the full262144x64 output;
17 preflight/late-failure cases;32 warmed same-shape evaluations with zero Rust
allocations;2 real coefficient fixtures; and5 sharing/publication controls.
The bounded261184-byte real fixture covers PLE256x1536 and six actual head rows,
not the full real head or model. No model checkpoint is loaded. Twelve new
normal/optimized streaming readers pass original gates, full output/transport
checks and60 semantic mutation invocations. Uncaptured preflight guards remain
Rust assertions; timing and whole-model numerical claims are not inferred.

Seven original GPU regressions also pass: affine8, affine8 storage, native
arithmetic, three strided BF16 checks and the whole-matrix proof factory, plus
six original readers. Across all35 stages the maximum sampled process charge
is505348096 bytes including32MiB helper, below512MiB; new-path/reader maximum is
452132864 bytes. Policy allowances, actual packed owners, guarded scratch,
oracle/result buffers and helper are separate ledger entries; native PSO bytes
are unknown. Shared logical copies are not physical bus traffic. Rust allocation
results do not measure native allocations or establish a latency improvement.

The first source child removes one extra EOF newline. A later child splits a
42-field capture JSON literal into15/15/12 objects without schema/math changes.
Failed attempts remain: a newly visible Cargo identity rejected by the guard
and a JSON macro recursion limit. The departed Cargo identity's exact ancestry
was not captured. Static builds now scan named processes before and
after compilation, retaining continuous power/thermal/idle and owned-memory
checks. All numerical/GPU execution retains continuous named scans. A final
required-suite compilation also stopped on a newly visible compiler identity;
its child prebuilds with no-run,
then requires no new `Compiling` output during full test execution.

Fmt, strict CPU/Metal Clippy and certified opt3/default-FP release unit pass.
The actual1046-test/317ignored listing derives the final1052-pass/409ignored
Metal expectation. The actual unfiltered Metal suite passes1052/409ignored
without compilation during test execution; release CLI passes. The combined
Astra review approves this scoped private prerequisite. CPU847/53ignored reuse reconstructs all4009
unchanged inputs and original logs. Full graph composition, public routing,
whole quality, request latency, external wins and overhead gates remain open.

[Code/check packet](../artifacts/mlx-gpu-gap-2026-11-15/g0-q4-borrowed-encode-code-review-prep-1/review-packet-v1.json),
[applied source](../artifacts/mlx-gpu-gap-2026-11-15/g0-q4-borrowed-encode-checks-run-3/source-freeze.json),
[actual GPU/readers](../artifacts/mlx-gpu-gap-2026-11-15/g0-q4-borrowed-encode-checks-run-3/checks-gpu-1/result.json).

[Final required checks](../artifacts/mlx-gpu-gap-2026-11-15/g0-q4-borrowed-encode-checks-run-3/checks-required-2/result.json).

[Combined Astra code review](../artifacts/mlx-gpu-gap-2026-11-15/astra-q4-borrowed-code-review-1/verdict.json).
