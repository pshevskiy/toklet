# Gemma3 Half RMS with FP32 scales

29 October 2026. B3c1 qualifies one internal Metal operation on Apple M1 Pro.
It is test-only until the resident Gemma3 graph uses it; this does not enable
Gemma3 loading or Chat. The [CPU graph](gemma3-cpu-composition.md) and
[remaining checklist](../TODO.md) define the surrounding work.

## Precision and ownership

Gemma3 stores offset-gamma norms. Loading widens the BF16 gamma, computes
`FP32(1 + gamma)` once and retains those effective FP32 coefficients. Narrowing
them to Half would lose information before normalization. This operation reads
Half activations, accumulates squared values in sequential compensated FP32,
applies epsilon `FP32(1e-6)` and the FP32 coefficients, then stores Half.
Contraction is disabled and the optional library uses safe, precise Metal math.

The checked plan accepts rows 1–128, widths 32/64/256/640 and independent
input/output row strides. Only logical columns are read or written. Each
allocation includes seven guard elements. FP32 coefficients are held in the
existing aligned `Buffer<u32>` storage as raw FP32 words. Status is one atomic
word plus seven guards. Composite indices and rounded dispatches are bounded
before encoding; aliases, short buffers, mismatched devices and unretained
command resources are rejected.

The operation owns its compiled pipeline and queried execution width. The
caller owns input/output buffers, immutable scales, shared sticky status and
command completion. Encoding borrows those owners, copies five scalar arguments
and creates an encoder without uploading operands or waiting. Two operations
can be encoded into one command with one completion wait. Session integration
must preserve these lifetimes and synchronize CPU access explicitly.

Numerical operand bytes are the sum of guarded Half input/output and guarded
FP32 scales/status, counting shared coefficients once at graph integration.
For one 32-wide row with strides 35/37, that is 360 bytes. The constructor also
checks a separate 64 MiB compiler allowance plus its Rust owner before compiling.
That allowance is an admission reserve, not measured pipeline residency or a
whole-process fit claim. Caller, context, driver and diagnostic allocations are
separate. A future graph should compile this pipeline once, not per layer.

## Independent reference and acceptance

The standalone Python producer imports no target graph. Main rounds every
prescribed scalar step to FP32; wide evaluates the same stored inputs in FP64.
Both store Half. Export and replay are byte-identical. Six cases contain 54,272
finite output values per profile, covering zero rows, mixed signs, actual
256/640-wide operands and maximum finite Half inputs. Padding is quiet NaN so
an accidental padded read cannot silently act as valid data.

The fixed per-case MAIN gate is `atol=max(1e-4, 2*envelope), rtol=1e-3`, with
wide diagnostic only. Exact controls supplement it:

- The FP32 scale witness requires `0x3fa7`; narrowing scales gives `0x3fa6`.
- Positive/negative input subnormals `0x0001/0x8001` require `0x03e8/0x83e8`.
- Invalid active NaN, infinity and Half-output overflow must set sticky status.
  Their output bit patterns are retained but are not an acceptance gate.

The consumer retains all output/status words and immutable input/scale hashes
before checking values. It checks every padding/guard word, exact duplicated
encoding, status recovery and explicit reset, four short-buffer errors and a
true equal-extent input/output alias error.

## Checks and measured outcome

Formatting, strict default/Metal Clippy, 420 default tests, 300 Metal-library
tests, two targeted cold RMS controls and the existing Gemma3/packed graph
regressions pass. Counts overlap. Three explicit release GPU tests pass on M1 Pro:
108,544 twice-encoded source comparisons have zero absolute error and zero wide
diagnostic exceedances. Exact scale and subnormal words, sticky status, reset,
padding, guards and input immutability pass.

All twelve measured encoding calls allocate zero times through the calling
thread's Rust global allocator in release. This excludes Objective-C/Metal
driver allocations. The first debug GPU attempt failed the unchanged zero gate
with five allocations per encode. Astra's dependency inspection localizes a
source-grounded explanation in debug-only Objective-C signature parsing; no
allocation stack trace was captured. That debug failure remains preserved and
rejected. The release retry changes only build profile, preserving source and
all numerical/allocation assertions.

An earlier collector expected Cargo's `lib` artifact kind instead of this
package's `rlib/staticlib`. All cold commands had passed; a separately recorded
metadata-only continuation froze the existing binary. Neither failure changes
the numerical policy.

Owned jobs have finite deadlines, retain raw logs and close their process
groups. Process-memory samples remain direct-process samples, not a guaranteed
instantaneous process-tree peak. No latency, throughput, complete resident
graph, real model, app, iPhone or full-context qualification is claimed.

See [machine-readable evidence](../benchmarks/results/2026-10-29-gemma3-half-rms-v1.json)
for identities and private receipt locations. B3c2 adds the missing Half GELU
and embedding-scale operations; rolling cache and complete composition follow.
