# Complete reference arrays for the real tied head

The accepted independent Gemma mathematical trace contains the full four-token
input and output for the original tied embedding/head matrix. This task extracts
those arrays into a small frozen artifact before adding the native full-head
diagnostic. It verifies extraction integrity; it does not reevaluate the model or
close an inference, source-parity, quality or device-fit gate.

The input is all 6144 FP32 values of staged `final_norm`, shape `[4, 1536]`. The
expected result is all 1048576 FP32 values of staged `head`, shape `[4, 262144]`,
before logit softcapping. Tokens are `[2, 100, 257, 100]` at positions 0–3. The
acknowledged mathematical contract uses original packed4/group64/BF16 weights
reconstructed to FP32, captured mathematical FP32 input, FP64 projection
reductions and output rounded to FP32. It does not substitute native activations
or claim equality to unrounded ideal FP64 or original BF16 backend execution.

The producer authenticates the accepted mathematical receipt, complete original
JSON and 103403520-byte raw trace, compact witness and mathematical recipe
source. It validates exact selected descriptors and streams both arrays into an
exclusive 4218880-byte binary, with an 8 MiB cap. Source offsets, hashes, shapes,
precision and original provenance are explicit. Every write attempt precedes
publication; failure retains the complete first array or partial binary and
independently attempts report/manifest publication while preserving the original
exception.

An independent checker imports no producer, engine, GPU or NumPy. It separately
authenticates the original sources, checks compact and full trace identities,
compares every extracted byte, verifies complete finite FP32 values and exact
typed schemas and manifests. Two captures are byte-identical and each passes
replay. Seventeen actual-capture mutation/reader tests and three failure-retention
tests pass, including coherent corrupted arrays, truncation/trailing bytes,
wrong precision or softcapped source, boolean substitution, late descriptor
failure and primary-error preservation after publication failures.

All six owned processes close successfully. Each had a separate 512 MiB
RSS/physical allowance and 600-second deadline. Maximum observed RSS was
83017728 bytes and peak physical footprint 39469704 bytes. These are extraction
and verification process observations, with no inference performance claim.
No Rust, shader, prior test or protected release executable changes; this task
needs reference, content/link/whitespace checks rather than another engine or GPU
suite. The [receipt](../benchmarks/results/2026-10-25-gemma-head-reference-v1.json)
binds sources, copied scripts, raw outputs and the three frozen artifacts.

Next, the [resident chunk owner](gemma-partitioned-affine.md) must execute the actual 262144-row head with all
sixteen 16384-row chunks resident simultaneously, with
aggregate admission including Context, caller output, both CPU staging arrays,
original source/loading overlap and diagnostic resources. Complete output,
repeat/partition/resume and buffer/allocation identity checks remain necessary.
The existing single-owner geometry rejection is unchanged. Large per-layer
table gather, resident graph/KV/scheduling, historical source/cross-precision and
long-rotary discrepancies, performance, app/iPhone admission and 131072 output
tokens plus nonempty prompt remain open.
