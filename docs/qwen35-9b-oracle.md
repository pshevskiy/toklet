# Independent Qwen3.5-9B one-input oracle

Q2.D validates the [pinned9B CPU binding](qwen35-untied-head.md) against a separate
source-authenticated Python graph. Both native captures pass all33,320,448values
under the original `ideal_fp64` and `staged_fp32` profiles, with zero failures,
maximum gate ratio0.1874 and matching lowest-ID greedy results. Astra approves
scoped retention with no blocking findings. This qualifies only raw input `[1]` at position0; contextual,
GPU, codec, request, quality and managed adoption gates remain open.

The original4GiB affine row reader,4B source entry point, graph arithmetic and
fixtures are unchanged. A separate `LargeMatrixRows` constructor caps file extent
at6GiB and validates strict integer dimensions, offsets, complete groups,
nonoverlap and endpoints. It inherits the exact original row-read and component
methods; reconstruction and tiled projection arithmetic are unchanged. Tiny
sparse-file tests cross4GiB without allocating a large tensor and check scalar
values, tails, gather, projections, malformed spans, truncation and nonfinite
coefficients. File-extent support grants no source or memory eligibility.

The independent9B binding authenticates all5,038,160,953source bytes and the exact
config/header, consumes927physical components and binds250packed matrices,
153BF16 vectors and24convolutions. Its separately named output table replaces
only the final projection; direct input lookup keeps the embedding. The frozen
small two-table fixture checks all stages against both profiles and rejects head
substitution. Public source entry points reject cross-source fixtures. Immutable
source reads retain descriptor/deadline checks, and mutation poisons the owner.
No target engine, MLX graph, full FP32 model or memory mapping supplies the oracle.

The reference runtime is admitted before numerical imports using a fresh owned
bytecode cache,1,381runtime files,95Python native files and11producer source files.
A copied virtual environment leaves the primary environment unchanged. Complete
source endpoints, imported NumPy origins and loaded native images close the run.
The95.023s reference process has a530,252,880-byte lifetime physical peak. Its
199,922,688-byte capture contains2,102arrays: every final stage, recurrent/KV state
and complete logit row in both profiles. Durations describe this diagnostic run,
not an engine comparison or performance gain.

The fresh optimized native test binary produces two bitwise-identical
66,640,896-byte captures, each with1,051arrays. Native lifetime physical peaks are
5,310,582,960/5,310,550,192bytes; admitted owner/policy bytes are5,363,086,344 with
an additional128MiB diagnostic reserve, within6GiB. Each test checks exact reset/
replay, a late-head observer failure at a nonzero prefix, cancellation and zero
warmed Rust allocations. All249projections complete on the successful input;
failed work commits no token and preserves the previous distribution and state.
Loading and a final full-stream source hash bind original source endpoints.
These results use explicit child-only `MallocLargeCache=0`; application defaults
and aggregate/default-allocator fit remain unqualified.

The reader checks strict schemas and types, source/binary/compiled-input pins,
prelaunch inputs, runtime closure, complete arrays, hashes, finite values and
scope flags. It encloses reported phase times in the completed owned lifetime
and independently folds live native memory counters back to recorded peaks and
conservative charge sums. Two copied valid captures pass;19damaged/resealed
controls reject. A finite rewritten embedding value with updated array/body
hashes still fails the independent coupled gate. All original files remain
unchanged. The gate stays `1e-5 + 1e-4*abs(reference)` for both profiles.

Formatting, strict CPU/Metal all-target Clippy and667CPU/801Metal tests pass
(46/325ignored). The33relevant Python tests include original4B/row/fixture
regressions and14new controls. Numerical production and ABI/dependency sources
are unchanged: the only Rust changes add a module under the existing `cfg(test)`
guard. An explicit reuse audit checks native61 source/output pins, three46export
slices, Csmoke fixtures/binaries and all190inputs per old standalone probe.
Q2.C's engine baselines are reused with their original scopes; fresh capture
logits match native61 bitwise. No new package, engine speedup or app-link change
is claimed from test-only code.

Two reversed row-helper cohorts compare the old and new readers on the same
`[4096,4096]` packed matrix and one FP64 activation, with3warmups+10samples per
variant per process. All output bits match. Projection medians are54.351→54.923ms
(+1.05%) and56.349→55.632ms (−1.27%);12warmups and40measured samples are retained.
The numerical methods are the same inherited functions. Hardware/sysctl
access was denied and thermal/performance queries failed with0xe00002bc; AC power
was observed after collection. No usable hardware/thermal snapshot or stability
qualification follows. These descriptive warm helper timings establish no engine
speedup,1% overhead bound, tail or sustained qualification.

Audited cache cleanup meets9GB headroom and preserves690pinned source/output/
link-dependency files; the2GiB disk guard holds. One unused setup snippet read
an incorrect relative path before writing a collector; no source or capture was
changed by that failure. The6GiBMac ceiling,512capacity/32chunk,512MiB codec policy,
[4B128/256 rejection](qwen35-context256.md) and deferred physical iPhone scope stay
binding. Further9B context, GPU, default allocator/codec/application fit,
request/overhead and quality/adoption need separate evidence.

[Contract](../artifacts/qwen35-9b-oracle-2026-11-09/acceptance-contract.json),
[required checks](../artifacts/qwen35-9b-oracle-2026-11-09/required-checks-1/checks.json),
[first comparison](../artifacts/qwen35-9b-oracle-2026-11-09/fidelity-1.json),
[second comparison](../artifacts/qwen35-9b-oracle-2026-11-09/fidelity-2.json),
[evidence controls](../artifacts/qwen35-9b-oracle-2026-11-09/evidence-controls-1.json),
[reuse audit](../artifacts/qwen35-9b-oracle-2026-11-09/production-reuse-audit.json),
[helper measurements](../artifacts/qwen35-9b-oracle-2026-11-09/helper-benchmark-summary.json).

[Checks/review/commit](../artifacts/qwen35-9b-oracle-2026-11-09/publication-checks.json),
[Astra](../artifacts/qwen35-9b-oracle-2026-11-09/astra-review.json),
[telemetry availability](../artifacts/qwen35-9b-oracle-2026-11-09/helper-snapshot-scope.json),
[next scope](../artifacts/qwen35-9b-oracle-2026-11-09/next-scope-analysis.json).
