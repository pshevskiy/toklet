# LFM2 actual-input conformance operators

B4e3c1 adds bounded local arithmetic contracts and synthetic controls. This is
not a model forward, capture reader or native real-model qualification. The
independent source and its literal gates remain fixed; B4e3c2/c3 still require
complete native operands, exhaustive dispatch, exact state reconstruction and
the bounded real-model experiment.

The module consumes finite stored FP32 arrays and verifies Half-valued operands.
The caller must authenticate original weights, source-group gates and captured
identities. It never substitutes source intermediates for missing native input.
Shared Gemma wrappers/limits and all native kernels remain unchanged; only the
existing generic bit/interval/sqrt/provider primitives and qualified streamed
exact-affine helper are imported.

Original gather, BCx views, multiply and ordinary residual preserve their declared
stores. Direct-gain RMS uses raw BF16 gains without+1, FP32 epsilon1e-5,
ascending compensated squares and one final Half store. Conv3 consumes actual
current-chunk u and actual old history, three separate FP32 products and one
Half store; candidate history is the raw-u suffix.

Affine checks at most128 original rows,4608 columns and four selected input
rows per stripe. Exact RNE centers certify unchanged nonzero interior values;
neighbors, zero and maxfinite require exact dyadic brackets. A representable
result is a singleton: exact cancellation admits canonical+0 only, and nonzero
underflow retains the correct zero sign. Numeric guard zero equivalence does
not override these word-level projection semantics.

SiLU propagates discrete exp4ULP candidates through the stable formula. D64
RoPE at theta1e6/positions0–63 propagates pow16ULP and sin/cos4ULP candidates;
both output coordinates must share the same frequency/cos/sin candidate.
Word-based membership preserves signed zero. A LFM-only enumerator includes both
zero words when a trigonometric provider interval contains zero; exp retains its
nonnegative/+0 profile. Shared Gemma enumeration is unchanged.

Q16 attention probabilities consume every materialized native score column and
require causal masked positive-zero words. Provider endpoints propagate through
serial FP32 mass and division. The result is a conservative per-value enclosure,
not proof that the whole probability vector shares one realizable hidden exp
vector. Score/context/head/teacher strict source comparisons remain mandatory.

Every local result also needs the same source group's literal atol/rtol guard;
semantic and numeric failure counts stay separate. Later reader obligations are
303 strict groups/1,965,024 values and6440 local groups/11,971,552 values. These
are prospective obligations, not results of this operator task. Coupled drift
is retained separately even where local semantics establish eligibility.

Fourteen controls cover domains, exact stores, zero/underflow/rounding edges,
full128×4608 stripe bounds, compensation, raw history suffixes, provider grids,
reachable joint rotary pairs and independent semantic/numeric rejection.
The initial two proposals were unexecuted; review fixes signed-zero membership
and complete zero enumeration before testing. No model output chooses a bound.

Numerical workspace reserves48MiB, including the qualified32MiB affine workspace,
bounded rotary/SiLU caches and local temporaries. Python metadata reserves64MiB
separately; the later complete reader's numeric reserve is62MiB, with mapped
source/native/model bytes and runtime counted separately. These are prospective
bounds, not observed complete process peaks. All14 controls pass in0.597s; formatting, both strict Clippy configurations,
default457/Metal556 tests and the release Metal library test build pass
(20/181 ignored; suites overlap). Astra verifies6877 current input/output
identities and all seven closed groups with clean endpoints.

The isolated NumPy2.3.2/Accelerate worker has authenticated module origins
164→304 and non-system native images15→34. OS maximum RSS is100,007,936 bytes;
ten direct samples observe lifetime physical-footprint59,343,496 bytes. These
observations do not measure individual workspace reservations or real-model
performance. System shared-cache image paths are observed rather than hashed;
full-binary64 arithmetic remains the qualified affine backend premise.
[Exact evidence](../benchmarks/results/2026-10-29-lfm2-conformance-math-v1.json)
pins all sources, actual controls, runtime, cold receipts and Astra audit.
Real capture/state/text/model execution and whole-app/phone/full-quality
qualification remain pending.
