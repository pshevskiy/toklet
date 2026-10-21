# Full implementation-plan audit

The M0–M7 acceptance criteria are implemented and have accepted exit evidence in
their recorded scopes. This audit compared the plan's 32 checked tasks and exit
conditions with current source, independent fixtures/tests and the closure
reports. It found and corrected two kinds of remaining gap: a stale test-only
producer inventory that broke provenance regressions, and current documentation
that still presented completed work as future work. Earlier dated reports and
qualifications stay immutable.

## Requirement-to-evidence index

| Milestone / tasks | Implementation and accepted exit evidence | Scope retained |
|---|---|---|
| M0.1–M0.4 | [Package and CI](../../Cargo.toml), [independent tiny fixture](../../tests/fixtures/tiny-v1/manifest.json), [frozen contract](../../benchmarks/contract-v1/manifest.json), [first MLX baseline](../../benchmarks/results/2026-10-09-mlx-bf16-512-128.md) | CPU checkout/tests need no model download or Apple framework; GPU/model suites are explicit |
| M1.1–M1.4 | [Scalar kernels](../../src/cpu/mod.rs), [decoder](../../src/model/mod.rs), [all-array tests](../../tests/cpu_decoder.rs), [frozen numerical policy](../../tests/fixtures/tiny-v1/manifest.json) | Scalar FP32-storage reference and independent fixtures remain available |
| M2.1–M2.5 | [Validated loader](../../src/weights/mod.rs), [session](../../src/session.rs), [generation](../../src/generation.rs), [CLI](../../src/bin/cli/mod.rs), [M2.5 scalar/external closure](2026-10-11-m2.5-scalar-comparison.md) | All eight workloads, actual 2,048-position boundary, full application and sustained scope; external representation/timing/memory limits disclosed |
| M3.1–M3.3 | [CPU retention](2026-10-12-cpu-tiled-retention.md), [bounded CPU/ANE conclusion](2026-10-11-ane-feasibility-closure.md) | Tiled CPU is explicit; scalar stays default. Inconclusive ANE acceleration closes research without adopting an ANE engine |
| M4.1–M4.4 | [Resident Metal decoder](2026-10-13-metal-engine.md), [actual GPU trace and calibrated stage attribution](2026-10-13-metal-trace.md) | Functional FP16 generation, ownership, allocation and failure/scheduling gates; performance/quality selection is separate |
| M5.1–M5.6 | [Performance release](2026-10-15-m5-release.md), [accepted receipt](../../benchmarks/results/2026-10-15-m5-release-qualification-v1.json) | Small compensated Half and larger frozen group8 Q4v2; full quality, repeated workloads/application, sustained scope and useful named tiled-prefill gain |
| M6.1–M6.3 | [Long-context release](2026-10-19-m6-release.md), [accepted receipt](../../benchmarks/results/2026-10-19-m6-release-qualification-v1.json) | Selected 4K/8K/16K/32K Online/FP16-KV cases, quality/state/memory/application; I8 evaluated without production adoption |
| M7.1–M7.3 | [Native boundary](../native-abi.md), [Pro device report](2026-10-21-m7-iphone14pro.md), [explicitly amended closure](2026-10-21-m7-release.md), [accepted receipt](../../benchmarks/results/2026-10-21-m7-release-qualification-v1.json) | iPhone14Pro/iOS27.0.1, selected models and named limits; original base-device target explicitly amended by the user |

CPU and Metal session implementations retain distinct sampled history and
committed KV, validate admission before mutation, preserve valid prefixes or
require explicit replay after failure, and reuse bounded resident allocations.
Native handles preserve owning-thread, callback, cancellation and lifetime
contracts. Source inspection and fresh unit checks supplement the accepted
numerical/device evidence; they do not replace it or create new performance
qualification.

## Closed gaps

The grading-admission unit fixture had added only the historical Q4v2 module to
its frozen producer inventory. Native-bridge and online-attention modules added
later made two tests reject the fixture before reaching their intended
inventory and binary-replacement assertions. The fixture now synthesizes
identities only for current Rust source files, preserves every existing recorded identity, and leaves
missing non-Rust inputs missing. Its regression explicitly removes Cargo.lock
and requires exact production inventory rejection. Production admission,
immutable native records, executable identity checks and numerical/quality gates
are unchanged. The [test repair report](2026-10-21-grading-fixture-audit.md) and
[check receipt](../../benchmarks/results/2026-10-21-full-plan-checks-v1.json) retain
the original failing suite/source and the final successful checks.

The native ABI guide now documents implemented Metal cancellation and the
qualified Pro result instead of future backend/device work. README's accumulated
implementation narrative is explicitly historical, with current closure links.
The benchmark guide points the old profiling probe's pending statement to its
completed M2.5 and CPU-retention evidence. PLAN labels the historical starting
point and links this requirement index.

## Explicit limits that are not incomplete plan tasks

The plan permits a bounded inconclusive ANE outcome and reporting external
baseline gaps. It conditionally evaluates I8 KV, stochastic sampling and later
serving/research features; it does not require adopting rejected mechanisms.
Consequently this audit does not turn the following limits into new milestones:

- External FP16 rankings and default replacements remain unclaimed; comparisons
  must preserve their original representation, quality and memory scopes.
- Q4v1/small-Q4v2 quality failures and removed non-beneficial fusion prototypes
  remain rejected. FP16 KV stays after the completed conditional I8 experiment;
  full-engine I8 adoption would require its own unmet adoption gates.
- Stochastic sampling, speculation, paging, downloads and multi-session serving
  remain conditional/deferred product or research scope.
- Energy units remain unavailable after bounded actual instrumentation attempts.
  Real OS warning delivery/jetsam, other devices and other OS versions remain
  unqualified. Synthetic warning-handler coverage is labelled explicitly.
- Remote CI execution is not inferred from local checks; the configured ordinary
  CI job is CPU-only. Ignored hardware/model tests remain distinct from the
  explicitly collected accepted GPU/model/device evidence.

## Fresh verification

Cargo 1.90 formatting and CPU/Metal-feature all-target Clippy with warnings denied
pass. CPU tests pass 111 with 8 ignored; Metal-feature tests pass 123 with 59 ignored.
All 127 Python test modules pass 887 tests with no skips (816 benchmark/experiment,
71 reference/quantization); the focused ten-test grading suite also passes.
[Raw evidence](../../benchmarks/results/2026-10-21-full-plan-checks-raw-v1.json)
retains commands, source hashes, original failed attempts and final logs. Rust
builds use an isolated target directory; both protected CPU release binaries and
all 33 numerical-source files retain their prior accepted hashes.

The test-only repair was reviewed and committed separately as `ff8ba28`.
The [audit receipt](../../benchmarks/results/2026-10-21-full-plan-audit-v1.json)
records the separate Astra documentation review, link/content/whitespace checks,
accepted closure hashes and unchanged numerical/protected-binary identities.
No tolerance, frozen corpus, checkpoint, artifact format, default, performance claim or device scope changes
in this audit. No new numerical or physical qualification is asserted.
