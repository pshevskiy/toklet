# Llama3 Metal block composition

B1c2b composes the previously isolated Half rotary block into native MLX-format
Llama execution. This document records its implementation and bounded
validation scope.

## Ownership and admission

The immutable model optionally owns one precise rotary pipeline and one guarded
FP32 inverse-frequency buffer. Sessions borrow that operation and encode into
their existing command and scratch buffers. Q and K use the same operation;
there is no per-layer frequency upload, new session positional allocation or
operator-level host wait. The ordinary rotary path is preserved.

Learned tensor bytes retain their existing meaning. `positional_bytes` in
session memory and `positional_resident_bytes` in runtime reports account for
the frequency allocation plus its Rust owner, separately from learned weights.
Both fields are omitted when zero. `model_resident_bytes()` is the checked sum
used when subtracting a shared model from multi-session totals. The existing
compiler/context allowance remains an estimate; driver pipeline allocations are
not measured physical residency.

Loading admits both source conversion and resident requirements. Cold resource
inspection compiles the optional pipeline to check availability but never uploads
frequencies, weights or session numerical buffers. The shared Swift decoder
accepts legacy missing-zero fields, checks positional equality and overflow, and
includes the owner in session and resident sums.

## Structural Llama contract

The previous tiny/Smol checkpoint geometry tuples are replaced by validation of
the implemented Llama blocks, model configuration, tensor/packed index bounds,
actual queried device layouts and buffer limits. Llama remains Reference-attention
only with an initial implemented limit of 512 positions. The source's declared
context stays separate. Tokenization, chat templates and artifact qualification
remain separate integration requirements.

The native Llama constant domain requires rotary base `theta >= 1`, positive
normal FP32 RMS epsilon and representable positive normal inverse frequencies.
The base restriction keeps initial-range angles finite and excludes subnormal
inputs to the ordinary GPU power operation. This prevents a subnormal
constant silently flushing to zero. It does not promise all intermediate values
are normal. Scalar configuration/reference behavior remains broader. For scaled
RoPE the checked frequency is the uploaded value; the unchanged ordinary shader
still computes its own FP32 power and is not bit-identical by this host check.

Metadata acceptance means `CompatibleUntested`, not a real checkpoint quality or
application allowlist claim. Packed projection coefficients retain their FP32
reconstruction. Half embedding lookup does not round the shared tied vocabulary
projection coefficients.

## Independent reference and checks

The new `llama3-half-graph-v1` fixture is separate from the unchanged scalar graph,
ordinary Half graph and isolated rotary references. Before target outcomes,
Astra reviewed its producer and frozen export. Ordered FP32 and wide FP64
profiles use identical Half storage boundaries, original packed reconstruction
and separate tied/untied heads. The inherited per-stage policy is
`atol=max(1e-4, 2*maxabs(main-wide))`, `rtol=1e-3`; every frozen stage has
`atol=1e-4`. No target-derived tolerance is used.

Eight cases cover tied/untied graphs, causal last-token changes and offsets 0,
5 and 509. The fixture has 39 stages and 38,336 values per profile. Existing GPU
diagnostics expose 35 stages (36,736 values per profile), excluding both layers'
scores/probabilities; this task does not expand that API. Full, three-row and
one-row schedules collect every chunk rather than only the final snapshot.

Offset 509 starts with an empty cache and evaluates three local keys at absolute
positions 509–511. It witnesses the rotary/capacity boundary, not a 512-key
context-quality run. No real Llama checkpoint, plain-text Chat, iPhone result or
performance improvement is established by the tiny graph.

See [TODO](../TODO.md), [prior rotary block](llama3-metal-rotary.md),
[index contracts](metal-index-contracts.md), and [improvement record](improvements.md).

The final cold checks pass: formatting, strict Clippy for default and Metal,
384 default tests, 265 Metal-library tests and 9/9 targeted Llama tests
(overlapping suites; the new explicit GPU test is ignored in ordinary runs).
Three optimized test binaries are frozen for the bounded GPU collection.
Shared Swift resource-plan checks pass against current CPU and Metal static
libraries, including existing worker lifecycle/source revalidation cases; the
same source set passes strict unsigned iOS typechecking. This is compilation
coverage, not physical iPhone qualification.

The first lint attempt found manual saturating arithmetic; the final helper uses
`saturating_mul` with the same overflow rejection. Astra separately caught the
below-1 theta admission gap before GPU collection. A private Swift wrapper first
omitted the existing `NativeContext.swift` dependency; the corrected source list
passes. These attempts and the reference-export wrapper formatting failure remain
in the raw evidence, without changing frozen gates.

Seven explicit GPU tests pass on Apple M1 Pro, macOS 26.6.2 (25G83). The scaled
consumer checks **1,680 snapshots and 110,208 values**. Full/three/one-row
schedules, unchanged causal prefixes and reset/resume distributions satisfy exact
numeric equality; signed-zero bit identity is not separately qualified. Cache
reuse, pending-history discard, partial-prefill resume, changed-prefix replay
and rejected boundary mutations preserve the expected committed state.

The separate resource test checks identical learned tensor bytes, one shared
positional owner, actual/report totals, exact and one-byte-short load/session
budgets and capacity-search boundaries. Existing tied/untied ordinary Llama graph
and session suites, dual-gate isolated scaled rotary, ordinary rotary and strided
projection/gather regressions pass. All seven owned groups close cleanly.
Tiny GPU process maximum RSS ranges from 17,006,592 to 32,178,176 bytes; these
are verification observations, not real-checkpoint memory or performance claims.

Retain the composed block in this scope. The next task is a bounded real
Llama 3.2 1B source/text comparison on Mac, followed by separate quality and
application admission. [Machine-readable evidence](../benchmarks/results/2026-10-28-llama3-metal-composition-v1.json)
binds the frozen sources, binaries, raw logs, independent export and reviews.
