# Independent proportional rotary mathematical reference

This G2 witness defines a mathematical correctness reference before a native
scalar target. It does not qualify upstream parity. The separately retained
[precision diagnostic](gemma-rotary-precision.md) still has eight failing
comparisons under the unchanged policy; none are waived by this fixture.

Inputs, base and factor must be exactly representable finite FP32 values.
For active pair i<R/2, the frequency is `factor*base^(2*i/D)` and coordinates
i and i+D/2 rotate together. Decimal arithmetic with80 significant digits
constructs frequency and absolute-position angle. The angle then rounds to
FP64 for standard-library sine/cosine and FP64 pair products/sums. Stored
frequencies round independently to FP64 and are not reused to construct angles.
This is Decimal80 construction followed by FP64 output, not an80-digit output
oracle. The eventual scalar target's FP32 trig/output rounding will be explicit.

The layout is contiguous `[tokens,heads,D]`. Inactive coordinates copy exactly,
including signed zero. Zero rotated dimensions copy the input; zero tokens make
no numerical target calls. The bounded reference allows0–8 tokens,1–4 heads,
positive even D≤512, even0≤R≤D and at most8192 input elements. Base is in[1,1e6]
and factor in[0.125,8]. Non-Boolean integer positions and the last absolute
position must lie in[0,2^53]. These are mathematical reference bounds, not model
or device context capabilities.

Eight cases cover empty input, position zero, tiny half-head pairs, full rotation,
zero rotation, actual D512/R128 geometry at short offsets and positions ending
at131071, and a non-unit frequency factor. Deterministic FP32 inputs and FP64
frequencies/outputs are frozen in the
[fixture](../tests/fixtures/gemma-rotary-v1/math.json). Six portable tests cover
hand literals distinguishing denominator/pair distance, exact inactive signed
zeros, multiple heads, exact chunk offsets, norm preservation, invalid
geometry/positions/precision and strict externally hashed bounded JSON.

The producer uses Python3.12.8 and standard-library Decimal/math only; no NumPy,
MLX or model code executes. Recorded interpreter/framework/math/Decimal binary
and selected source identities are a minimal reproducibility record, not a
claim of a complete standard-library runtime closure. MLX-LM source revision
and hash identify the intended mathematical semantics, not execution or parity.
An independent replay with90-digit logarithm/exponential frequency construction
checks the saved mathematical arrays before acceptance.

The [receipt](../benchmarks/results/2026-10-24-gemma-rotary-math-v1.json) binds
the exact producer/tests, six passing tests, export, fixture, replay and the
retained upstream diagnostic. Astra reviews code and final evidence before
commit. Existing numerical engine sources and protected binaries are unchanged.

Next is the separate standalone scalar reference with FP64 frequency/angle,
then FP32 sine/cosine and rotation arithmetic, matching the existing CPU rotary
precision style. Whole-graph and Metal precision decisions, upstream agreement,
real-model quality, context/device admission and Gemma4/131072-output capability
remain open. A passing mathematical operator will not close those gates.
