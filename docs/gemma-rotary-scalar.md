# Scalar proportional rotary mathematical reference

`cpu::proportional_rotary` and `ProportionalRotaryShape` add a standalone G2
correctness operation from the previously frozen
[mathematical reference](gemma-rotary-math.md). No model graph uses it yet.
The [eight upstream diagnostic disagreements](gemma-rotary-precision.md) remain
open; mathematical fixture agreement does not waive them.

Input/output are contiguous FP32 `[tokens,heads,head_dim]`. For active pair
`i < rotated_dims/2`, the inverse frequency is `base^(-2*i/head_dim)/factor`.
Coordinate i pairs with i+head_dim/2, using the full head dimension for both
exponent denominator and pair distance. Frequencies, absolute angles and
sine/cosine use FP64. Sine/cosine then round to FP32 and pair products/sums use
FP32, matching the existing scalar rotary precision style. Frequencies are
hoisted outside token/head loops. This is a correctness reference, with no
optimization or upstream approximation-emulation claim.

Inactive coordinates copy bit for bit, including negative zero. Zero rotated
dimensions copy all input. Tokens/heads/head dimension must be nonzero; head and
rotated dimensions are even, with rotated dimensions no greater than the head.
Base/factor are finite positive FP32. Checked absolute last positions must be
at most2^53, avoiding integer-position rounding. An empty sequence makes no
operation calls. Checked dimension products, exact lengths and finite admission
of all input coordinates precede output writes. Inputs remain immutable.

The operation initially copies input, then writes successful active pairs.
Rotation overflow rejects with an output index; callers discard the entire
partial output and may replay from unchanged input. The function allocates
nothing, owns no session state and requires no extra alignment beyond FP32.
It does not grant a model context limit or change existing rotary behavior.

Six offline tests authenticate the mathematical receipt's literal hash before
fixture identities, compare every nonempty saved output under the unchanged
`1e-5 + 1e-4*abs(expected)` policy, and check inactive coordinates exactly. Other
cases use hand literals and negative zeros, compare full rotation against the
existing scalar operation, enforce exact chunk offsets with multiple heads,
reject invalid geometry/lengths/nonfinite inputs before writes, and cover both
overflow coordinates plus replay. Extreme finite parameters and the2^53
position boundary check finiteness/admission only; they do not establish
numerical accuracy or model quality beyond the frozen cases.

The [receipt](../benchmarks/results/2026-10-24-gemma-rotary-scalar-v1.json) binds
exact source, mathematical fixture, raw check logs, prior engine sources and
protected binaries. Astra reviews source and final evidence before commit.
Formatting, strict default/Metal Clippy, 217 CPU tests and239 Metal-feature
tests (15/107 ignored respectively), plus both iOS cross-checks pass. These
cross-checks compile the library; they are not physical device qualification.
Builds use a separate copy-on-write compiler cache; existing accepted targets
and protected benchmark executables remain unchanged.

Remaining G2 work includes local/shared KV semantics, value normalization,
per-layer paths, decoder ordering and logit capping. Graph/session/Metal,
independent upstream agreement, real-model quality, capacity admission and
Gemma4/131072-output device/app qualification remain open in the
[plan](gemma4-implementation-plan.md). No native ABI or execution cap changes.
