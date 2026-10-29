# Borrowed strided Half products on Mac

B4d4a qualifies a small reusable product for B*x and C*z in the selected LFM2
composition. B/C/x are offset/strided views of one physical `[T,3H]` BCx parent;
C*z also reads the separate convolution result. Inputs may share a readonly
parent. Output and sticky status cannot alias either input or one another.
The operation adds no activation, clamp, transpose or state publication.

The bounded profile admits H256/1024 and1..32 rows. Checked UInt32/host arithmetic
resolves all offsets, strides and minimum extents before encoding. The output
includes its prefix, complete row padding and seven final guards. Minimum view
lengths are not an allocation sum: the consumer counts actual physical parents
once. The representative real B*x test owns262,204 physical operand bytes,
including its shared parent, output and status. Pipeline admission, metadata,
framework and diagnostic/caller owners are separate; this is not model fit.

Each lane computes a FP32 product of two stored Half values and one RNE Half
store. Finite Half products have at most22 significant bits and an exponent
range inside FP32's normal domain. Both active inputs are checked even when the
other is zero; nonfinite operands/products/stores set sticky status. Failed
output belongs to the caller's discard policy. Encode allocates no operands,
uploads no values, resets no status and performs no host wait.

The source precedes target outcomes: five cases/17 arrays/738,248 bytes contain
67,072 MAIN and67,072 WIDE words. Eight controls, exact four-file replay and
Astra's independent integer-dyadic/RNE reconstruction pass. Source layouts
cover shared parents, nonzero offsets, partial rows, guards, ten exact store
witnesses and eight nonfinite/overflow controls. The unchanged MAIN gate and
diagnostic WIDE remain separate from exact word/state requirements.

Both optimized Mac GPU jobs pass independent saved-raw review:133 raw records,
128 submissions and342,784 fixed MAIN/WIDE comparisons are exact. The reader
also reconstructs4,088 finite invalid-neighbor words and512 changed-tail words,
for347,384 finite outputs. Eight invalid output words are retained as evidence
and are not a numerical oracle. All349,141 full output words include1,749 exact
canaries. Readonly physical hashes cover10,529,267 word observations, with the
shared parent counted once in78 records; these are repeated observations, not
unique model storage. Eleven schedules/96 boundaries, sticky/reset phases,
inactive poison, causal prefix and writable-alias/short-buffer controls pass.
Ten release encodes report zero Rust allocations; driver/capture work is excluded.

Full checks pass formatting, strict default/Metal all-target Clippy,450 default
tests,541 Metal-feature tests and optimized release compilation. No existing
shader changes. The two jobs retain2,138,703 raw log bytes below the prospective
2,713,662-byte bound and16MiB per-stream cap. Both groups close with no endpoint
error. The exact tested release executable is retained for subsequent rebuilds.
Usage samples21/1 are sparse observations, not complete process peaks.

[Machine evidence](../benchmarks/results/2026-10-29-strided-half-product-v1.json)
pins the source, reviews, checked executable and complete raw qualification.
The leaf remains test-only until the concrete resident graph consumes it.
The [LFM2 plan](lfm2-hybrid-plan.md) next freezes the independent Half tiny graph,
then composes whole-step KV/history/logit/token publication and recovery. Real
model, context/app/adoption-quality, performance and mobile work remain open.
