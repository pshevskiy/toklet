# Qwen3.5-4B FP32 Metal projection qualification

Q1.F tests the existing compensated FP32 affine Metal projection on all 249
original packed matrices of `avan-ag/Qwen3.5-4B-Uncensored-MLX-4bit`, revision
`fb98fbc59c71a567f3e23435386c9e93951261db`. It introduces explicit diagnostics,
not a coupled GPU model or managed route. Ordinary engine code is unchanged.

The [frozen contract](../artifacts/qwen35-gpu-projections-2026-11-08/acceptance-contract.json)
selects batches 1/17 before implementation. Operands are the exact FP32 input
vectors at the last token of the qualified 17-input/offset-37 CPU capture. The17
stored rows cycle exact scalars 1, -1, 0.5. These native vectors are isolated kernel
inputs; no native projection output is an independent reference target.

The independent producer authenticates the original source before publication,
reconstructs bounded 128-row FP32 weight tiles and computes FP64 BLAS dots over
exact FP32 operands. Every output is compared at the unchanged
`1e-5 + 1e-4*abs(reference)` gate. Source and numerical-runtime closure, full
output seals, real physical peaks and owned process-group exit are required.
The source payload starts at odd byte 119837; no alignment cast is assumed.

The existing Metal owner keeps original affine4/group64/BF16 chunks resident,
with maximum 16,384 rows per chunk. All249workspaces and caller input/output
buffers are allocated before timing. The diagnostic charges both packed CPU and
GPU copies, scratch, staging, context and conservative diagnostic allowances
against 6 GiB. A future performance loader should avoid retaining the CPU copy.
Completed calls validate status/guards and publish transactional full outputs.
Per-matrix NaN preflight rejection preserves outputs and reusable workspace;
warmed Rust allocation observation is separate from timing. Metal framework
command/encoder allocation is outside that Rust allocation claim.

Performance hypothesis: the resident FP32 Metal sequence reduces complete 249
projection caller time versus the original scalar CPU matrix views. Each of two
cohorts alternates CPU/Metal order, with three warmups and ten measurements per
engine. GPU time includes staging, encode/commit, completed waits, validation and
output copy. Setup/load, diagnostic I/O and allocation observation are separate.
This is an operator sequence, with real but frozen independent inputs; it is not
TTFT, prefill/decode throughput, a coupled GPU graph, a sustained test or adoption.

The first owned run passes every 24,846,336 output comparison with no failure;
maximum gate ratio is 0.000580. Native/reference physical peaks are 5.298 GB and
0.172 GB. Admitted diagnostic ownership is 5.608 GB. Scalar sequence medians
are 8.149/8.137 s and Metal medians 0.202/0.191 s across the two cohorts,
97.52%/97.65% lower caller time. Raw samples retain the first cohort's 1.265 s
Metal outlier; there is no tail-percentile claim. Hardware is Apple M1 Pro,
16 GiB, macOS 26.6.2, AC power with low-power mode off; `pmset` reports no
recorded thermal or performance warnings. This is not an energy measurement.
A fresh owned process reproduces the entire output capture bit for bit; both
processes pass 24,846,336 comparisons each. Confirmation medians are
8.074/8.047 s scalar and 0.196/0.196 s Metal (97.57%/97.57% lower). Both peaks
are below 5.299 GB. Confirmation system-wide free-memory observations are
70%/65%; these are endpoints, not a pressure time series. Retain the existing
projection candidate for concrete executor work; full-model benefit remains open.

Required fmt/both strict all-target Clippy and 631 CPU/765 Metal tests pass
(39/309 ignored). Four operand controls, eight reader controls, eight unchanged
adapter controls and 20 damaged-evidence controls pass. The existing independent
tiny GPU transaction witness passes four cases/88 steps, including sticky poison,
output preservation, immutable reuse and construction-context teardown. It does
not inject driver/in-flight failures. The initial zero-test adapter discovery
was corrected and is retained in evidence. Ordinary native53 package/C smokes
are reused after comparing 389 pins: only two `cfg(test)` parents differ.
The strict reader independently derives CPU load/caller/scratch ownership,
reconciles packed GPU owners and requires receipt/footer agreement. Astra found
that the first reader accepted inconsistent positive admission/caller totals;
that P2 is corrected and resealed underreporting controls are rejected. The old
footer omits raw Vec capacity, so its charged residual is bounded between the
3,598,336 operand bytes and an 8 MiB diagnostic bound (current residual4MiB),
not represented as an independent capacity observation. Both original native
receipts/captures remain untouched; strict recomparison preserves all numerical
and timing results. Rust/check/producer inputs are unchanged.
No ordinary engine instrumentation was added; no new overhead claim is made.

Astra approves the scoped commit after the P2 fix, with no remaining findings. [Verification](../artifacts/qwen35-gpu-projections-2026-11-08/verification.json),
[checks/review/commit](../artifacts/qwen35-gpu-projections-2026-11-08/publication-checks.json).

Next: integrate only a qualified projection candidate into a concrete bounded
Qwen executor, then independently validate coupled recurrent/history/KV/logit
state and rollback at admitted contexts. Codec, task/loss quality, full request
metrics, overhead, baseline/sustained evidence and managed adoption remain open.
