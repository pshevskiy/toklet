# Exact Qwen3.5 recurrence reference

Q2.G0 adds a separately named vector recurrence implementation for the independent
9B reference. The original scalar graph, scalar producer, tiny fixtures and default
reference reader remain unchanged. A generated graph replaces only the recurrence
block; reversing that patch must restore the original scalar source byte for byte.
The vector reader requires an explicit selector, a distinct result format and a
pre-execution 14-file source freeze with graph/helper/builder provenance. It rejects
implicit substitution into scalar evidence. Astra approves scoped retention, commit
and explicit reference-helper use, with no actionable findings.

The helper keeps state layout `[value_heads, value_dim, key_dim]` and grouped key
head mapping. It vectorizes independent state cells. FP32 products and updates are
separate rounded operations; accumulation remains left to right from positive zero.
FP64 products are materialized separately and reduced in the original `math.fsum`
order. Staged inputs must already be rounded FP32 values, including signed zero.
Dimensions and finite operands/results are bounded and checked. Numerical imports
remain after runtime admission; no engine kernels, dependencies or ABI change.

The pinned 9B one-input and context17/offset37 experiments authenticate the full
5.038GB source independently, use fresh owned bytecode caches and closed Python/native
runtime manifests, and collect both unchanged profiles. All 4,204 arrays and
67,173,376 values match the original frozen scalar captures bitwise. Complete body
SHA-256 values also match. The 6GiB Mac ceiling, 512 capacity / 32 chunk, 512MiB
codec policy, original tolerances and binding 4B 128/256 rejection remain unchanged.

| Reference workload | Compared values | Vector owned time | Vector physical peak | Original scalar peak |
| --- | ---: | ---: | ---: | ---: |
| One input, offset0 | 33,320,448 | 84.291s | 612,271,232B | 530,252,880B |
| 17 inputs, offset37 | 33,852,928 | 1,107.984s | 603,014,440B | 613,909,656B |

These elapsed values are diagnostic costs. The original context17 collection ran
alongside Rust checks, so these observations do not establish a controlled
whole-reference speedup. The one-input vector peak increases by82,018,352B;
all observed peaks and owned-group charges remain within the unchanged budget.

Two matched recurrence-only cohorts run on Apple M1 Pro16GiB / macOS26.6.2 with
NumPy2.3.2. Both use real `(16,32,128,128)` geometry, three warmups and ten measured
steps per profile and implementation; scalar/vector order is interleaved and
reversed in the second cohort. All80 measured and24 warmup samples, input
identities, exact output checks, IQRs and sample standard deviations are retained.
The timed boundary includes the full helper, list/array conversion and validation;
input construction, hashing, comparison and logging stay outside it.

| Profile | Cohort | Scalar median | Vector median | Lower step time |
| --- | ---: | ---: | ---: | ---: |
| FP64 | 1 | 145.261ms | 58.074ms | 60.02% |
| FP64 | 2 | 144.988ms | 57.638ms | 60.25% |
| Staged FP32 | 1 | 612.795ms | 27.282ms | 95.55% |
| Staged FP32 | 2 | 596.305ms | 28.424ms | 95.23% |

Owned peaks are108,069,608 /108,298,936B. AC power, nominal thermal state and Low
Power Mode off are observed at each cohort boundary; continuous stability is not
qualified. This measures a reference helper component, not Toklet inference,
request overhead, reliable tails, sustained throughput or controlled whole-reference
performance. A second run confirms the component retain threshold only.

The final55-test Python suite passes, including tiny tied/untied full traces,
grouped heads, repeated steps, signed zero, retained subnormals, cancellation,
overflow, malformed inputs and provenance controls. The full portable Qwen sweep also passes all175 tests. Both old CPU/Metal scalar-default one-input reports recompute unchanged:
33.32M comparisons each, zero failures, maximum ratio0.1874. Required formatting,
both strict Clippy configurations,667CPU/803Metal tests (46/329 ignored) and native62
package/Csmokes are reused only after a fresh audit of3,966 unchanged Cargo/native
inputs, original raw logs, source/package outputs and probe dependencies. This
Python-only change receives no new native/application qualification.

Two valid closed references and18 damaged/resealed controls pass; a finite one-ULP
body rewrite with updated hashes remains structurally valid but fails the exact
bit-parity gate. One valid timing closure and8 resealed timing/scope/order/runtime
controls pass. Original captures and all six frozen scalar anchors remain unchanged.

The first untied test incorrectly compared packed BLAS arithmetic with a dense
`fsum` fixture. The corrected test independently requires dense execution to match
the frozen fixture and packed vector execution to match the original packed scalar
path, without changing a tolerance or fixture. A test tile initially exceeded the
existing128-row reader limit; tiled iteration fixes the test, with the guard unchanged.
Initial setup syntax errors occurred before numerical writes. The first context17
attempt hit the unchanged2GiB disk guard before numerical capture; it is retained.
A fresh attempt after measured4.36GB headroom completes successfully. Audited cleanup
removes reproducible compiler/test caches, substitutes independent APFS clones for
byte-identical dependency files and preserves932 protected pins. The earlier cache
headroom target was missed; the successful attempt's declared minimum and live
guards pass. No model, scalar golden or frozen runtime file is removed.

Next is 9B 128/256 evaluation, with separately frozen
workloads/resource admission. The new producer currently accepts only one/context17.
Whole-GPU execution, default allocator/codec/application fit, request metrics and
overhead, quality, managed adoption, physical iPhone and all other roadmap gates
remain outside this prerequisite. The full improvement goal stays active.

[Contract](../artifacts/qwen35-reference-vector-2026-11-09/acceptance-contract.json),
[one-input parity](../artifacts/qwen35-reference-vector-2026-11-09/parity-one.json),
[context17 parity](../artifacts/qwen35-reference-vector-2026-11-09/parity-context17.json),
[helper contract](../artifacts/qwen35-reference-vector-2026-11-09/micro-contract.json),
[raw helper samples and dispersion](../artifacts/qwen35-reference-vector-2026-11-09/micro-summary.json),
[reference controls](../artifacts/qwen35-reference-vector-2026-11-09/evidence-controls.json),
[timing controls](../artifacts/qwen35-reference-vector-2026-11-09/micro-controls.json),
[Rust/native reuse audit](../artifacts/qwen35-reference-vector-2026-11-09/rust-reuse-audit.json),
[cache preservation](../artifacts/qwen35-reference-vector-2026-11-09/cache-space-reclaimed.json),
[test cache cleanup](../artifacts/qwen35-reference-vector-2026-11-09/ordinary-test-cache-reclaimed.json).
[Checks/review/commit](../artifacts/qwen35-reference-vector-2026-11-09/publication-checks.json).
