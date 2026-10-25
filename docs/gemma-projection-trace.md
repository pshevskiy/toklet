# Complete real Gemma projection reference inputs

This reference freezes all complete inputs and rounded mathematical outputs
for the 276 four-token Gemma E2B projections that fit the existing standalone
operator. It contains 552 arrays / 26,333,184 bytes. The full tied head remains
an explicit geometry rejection. No projection, native graph or GPU executes
in this reference task; it prepares a later real-checkpoint qualification.

## Source and numerical contract

Authenticate the pinned checkpoint identity, accepted
[full mathematical trace](gemma4-text-math.md), complete 1426-array ledger,
[coordinate/codebook evidence](gemma-affine-admission.md) and
[L1 reference](gemma-affine-l1.md). The trace uses IDs `[2,100,257,100]` at
positions 0..3. The acknowledged mathematical source is
`tools/reference/gemma4_text_math.py` at SHA256
`6c04a81b93c95edfad4a4de646da3731b71a6dd175163fb856d2009bd103b096`.
These are complete arrays, not the compact fixture's sampled coordinates.

Expected outputs are the raw rounded staged-FP32 linear outputs over
reconstructed FP32 weights, before the following norm, scale, residual or
softcap. The original linear implementation tiles weights and accumulates
with NumPy/BLAS FP64; this is not a new exact-rational or ordered-scalar dot
oracle. It must not be confused with the separate ideal-history graph.
Per-layer model projection maps to `ple_projection`; q/k, attention output,
FF down, PLE projection and PLE input gate map to their pre-norm linear states.
A later computed native ordered reference must be labeled separately.

## Actual input reconstruction and ownership

There are 206 captured-state reconstructions: 171 direct/context inputs and
35 residual inputs. Contexts preserve their head-major `[1,8,4,D]` source
layout while producing contiguous `[4,8*D]` rows. Residuals preserve the two
sequential FP32-rounded additions, including signed zeros.

Seventy additional inputs derive from the acknowledged mathematical GELU
recipe, rather than newly observed native/upstream activations. Compute the
original FP64 `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x**3)))`, retaining Python
`x**3`. Round the activation to FP32, multiply by the captured up/PLE value,
then round to FP32 again. Gate/up order and zero signs are preserved.
These actual recipe products replace sufficient envelopes only for this
new reference; earlier envelope records remain unchanged.

The producer resolves output names from the accepted invocation inventory.
The independent checker separately enumerates all 277 names, shapes, source
recipes and pre-norm output mappings, including the layer-15 FF-width change.
It uses exact rational FP32 rounding for residual/product boundaries and
preserves IEEE zero signs. Both implementations use the acknowledged platform
FP64 tanh recipe; this does not supply a new independent transcendental or
upstream BF16-source qualification.

Every actual input is checked with the current max-first/L1 proof. Of the
276 fitting calls, 269 satisfy the original max check and seven require L1.
The 262144-row full head remains above the unchanged 16384-row owner limit;
its record has no admitted array payload. Both whole tables still require
separate storage/gather/chunking mechanisms.

The binary stores 11,128,832 input bytes and 15,204,352 expected-output bytes,
within a concrete 32 MiB cap. Each array has at most 196,608 bytes. Per-record
shapes, offsets, lengths and hashes form one contiguous complete ledger.
Frozen artifacts stay outside Git to avoid adding a model-specific 26 MiB
binary to ordinary source checkouts. Their immutable paths and identities are
bound in the public receipt; a later ignored native probe must admit that
explicit frozen directory and model identity.

## Verification and remaining work

Portable checks cover the independent full mapping, byte inventory and FF
boundary, output placement, context axis, both residual roundings, two-stage
GELU/product rounding, malformed/nonfinite values and signed zeros. Actual-path
failure checks retain earlier completed arrays and the current attempt after
a late read failure. An injected report-write/flush failure after a complete
synthetic capture retains the full binary, partial report, failed manifest and
original error. The eleven methods need no model download or Apple GPU.

Two fresh captures match byte for byte, and both independent replays pass all
277 records / 552 arrays. Two additional retention methods and six owned
command/resource/cleanup closures pass. Maximum successful-process RSS is
62,603,264 bytes; maximum physical footprint is 44,384,904 bytes, each below
the separate 512 MiB reference-process allowance. This is not model residency
or inference performance. Exact scripts/interpreter are bound; a full
standard-library/native-runtime closure is not claimed. Numerical Rust code
is unchanged, so unrelated engine/GPU suites are not rerun for this task. The
[receipt](../benchmarks/results/2026-10-25-gemma-projection-trace-v1.json) binds
the exact sources, first captures, frozen directory, checks and exclusions.

The accepted artifacts are frozen before any native edit. A later native
qualification must load bounded original packed/BF16 matrix spans, upload one
owner at a time and compare every complete result under unchanged gates.
One-owner-at-a-time qualification will not establish whole-model residency.

Existing source/cross-precision failures, head/table chunking, native activation
parity, full resident graph/KV/scheduling, quality/performance, app/iPhone fit
and 131072 output tokens plus nonempty prompt remain open. No capability or
earlier failed numerical result changes.
