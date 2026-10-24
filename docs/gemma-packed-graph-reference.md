# Complete tiny packed Gemma text graph

The [receipt](../benchmarks/results/2026-10-25-gemma-packed-graph-reference-v1.json)
freezes a complete token-to-logit packed-weight witness before native packed
model/session composition. The earlier [dense graph](gemma-text-graph-reference.md)
and [packed projection/gather contracts](gemma-packed-reference.md) retain their
original scope. This task reads no real checkpoint payload.

## Concrete graph and stored weights

The synthetic graph uses hidden64, vocabulary17, four alternating local/full
layers, Q2/KV1, local head64/rotary64 and global head128/rotary32, PLE64,
MLP64/128 and a four-token local window. The first two layers own K/V; the next
two reuse the respective producers. Embedding and head share the same packed
table. PLE lookup/projection, direct learned norms, unlearned V norm, residual
order, GeGLU, layer scalar and final softcap are all present. Every matrix input
width is divisible by64, including PLE output and both attention O projections.

The deterministic source defines literal low-bit-first code patterns separately
from their semantic reconstruction equations. Per-row/group scales and biases
are rounded to BF16, with zero/negative/unequal scales. Embedding coefficients
use a larger explicit seed scale to exercise output capping. These seeds were
defined and captured before the first source trial and remain unchanged.

The [ledger](../tests/fixtures/gemma-packed-graph-v1/math.json) binds a
[4,723,548-byte binary](../tests/fixtures/gemma-packed-graph-v1/math.bin): original
packed U32 words/BF16 coefficient bits, promoted-FP32 physical module parameters,
small expanded FP32 oracle weights and both mathematical graph contracts.
The expansions belong only to this bounded reference producer. The native
packed model should retain borrowed packed matrices instead of expanded weights.

## Independent math and source execution

The oracle reconstructs FP32 rows from BF16 coefficients, with FP64 affine
multiply/add rounded once. Its ideal graph uses FP64 stable sums and Decimal80
rotary frequencies/angles. Its staged graph uses ordered FP64 linear reductions
and explicit FP32 operator boundaries, including separate rotary products and
division/tanh softcap stages. An independent standard-library replay checks all
253,188 expanded weight values,41,238 physical parameter values and180 math
arrays/567,424 values. Staged results match exactly; the largest ideal difference
is1.3322676295501878e-14 under the unchanged1e-13 absolute/relative replay gate.

Authenticated original class bodies and compiled decorators run with actual
MLX0.32.3 modules. Actual `nn.quantize` converts the constructed graph; exact
physical parameter names, shapes, U32/FP32 dtypes and values are then admitted.
This source execution explicitly promotes BF16 coefficient values to FP32 and
uses FP32 activations. It does not test the original-BF16 whole graph.

Each backend runs all32 positions in full prefill, one-token decode and mixed
chunks. Separate pristine and traced models make41 calls each. Traces check
shared producer/offset identity, normal cache updates, sliding wraps and immutable
weights. Each backend retains3,731 arrays/852,768 values in3,411,504 bytes and
7,462 ideal/staged comparisons.

CPU passes all7,462 comparisons under unchanged `1e-5 + 1e-4×abs(reference)`
gates. Maximum ratios are0.852418 for ideal and0.593479 for staged math.
GPU completes the same observations but rejects15 intermediate comparisons,
with maxima1.641661/1.382700. They include layer2 per-layer normalization/output
and layer3 input/output. The worst comparison is layer3 output at decode
position10. All164 final-logit comparisons per backend pass, without waiving
any rejected intermediate result. GPU graph qualification remains open.

## Ownership, checks and remaining gates

Ten portable tests cover exact geometry, packed code/group/metadata semantics,
embedding/PLE/tied boundaries, local/shared attention, rotary halves/tails,
future-token causality and strict binary/JSON admission. Three owner tests and
three runtime-admission tests cover source substitution, failed/raised children,
external identities and import resolution. Export and both source trials finish
as six owned commands. Completion is recorded separately from numerical success.
Completed raw arrays precede dtype/shape/finiteness rejection; partial failure
records and source/runtime/output endpoint checks remain in place.

This qualifies a bounded promoted-FP32 mathematical/source-CPU witness for the
next native packed model/session. Actual E2B geometry/payload admission, general
generation/prefix/stop semantics, resident Metal, original-BF16 execution,
checkpoint quality, performance, Mac/phone memory fit and app integration remain
open. The earlier dense graph's15 GPU discrepancies and8 long-rotary discrepancies
remain distinct, unwaived records. No engine ABI, app limit or eligibility changes.
