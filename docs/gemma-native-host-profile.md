# Gemma4 host arithmetic on the inspected arm64 Mac build

**Design only.** This identifies the host arithmetic for an independent
four-token reference for the proposed [Mac 512 conformance policy](gemma4-native-conformance.md).
The 11/27/4/8 historical failures retain their original verdicts. No new graph
comparison, provider qualification or app admission follows from this inspection.

## Release evidence

The inspected executable is
`artifacts/gemma-projection-environment-2026-11-04/actual-v2/release-driver`.
Addresses below are linked Mach-O virtual addresses before ASLR. Resolve direct
branches against the arm64 indirect-symbol table. A binary-wide import list
alone cannot identify the provider used by an individual operation.
The exact executable is bound by the [pipeline environment evidence](../benchmarks/results/2026-11-04-gemma-projection-environment-v1.json).

| Operation | What the optimized instructions establish |
| --- | --- |
| [RMSNorm](../src/cpu/reductions.rs) | Function at `0x1006228bc`. FP64 squares feed ordered scalar additions. Divide by width, add the widened FP32 epsilon and execute hardware `fsqrt d2,d2` at `0x100622cb0`. Divide by that denominator, multiply by the direct scale, then store FP32. |
| [Gated GELU](../src/cpu/gelu.rs) | At `0x1006220f0`, widen the gate input `x`. Compute `((c*x)*x)*x`, add `x`, multiply by `k`, then call `_tanh` at `0x10062210c`. Multiply `0.5*x` by `1+tanh(...)`, cast to FP32 and multiply by the FP32 partner. The FP64 bit patterns are `c=0x3fa6e4e26d4801f7` and `k=0x3fe9884533d43651`. |
| [Final softcap](../src/cpu/softcap.rs) | Function at `0x10042c078`. Widen input and cap, divide in FP64, call `_tanh` at `0x10042c15c`, multiply by the cap in FP64 and cast once to FP32. |
| [Proportional RoPE](../src/cpu/proportional_rotary.rs) | Call `_pow` at `0x100202568` for the frequency, divide by the factor, multiply by absolute position, and call `___sincos_stret` at `0x10020258c`. Cast both returned doubles to FP32. Four separate FP32 multiplications and one subtraction/addition produce the half-head pair; no fused multiply-add occurs. Copy inactive dimensions unchanged. |
| [Span attention](../src/cpu/span_attention.rs) | Function at `0x10038eb9c`. Accumulate ordered FP64 QK products, scale and store FP32 scores. The included-key denominator and probabilities use scalar FP64 `_exp` calls (e.g. `0x10038fcfc`). Store probabilities in FP32, then accumulate weighted V in logical key order in FP64 before an FP32 store. Masked probabilities are exact zero. |

## Proposed independent implementation

Use explicit Python `float` operands for binary64 operations and a single FP32
rounding helper at each specified store. For FP32 rotary pairs, round each
product **before** the final sum or difference. Construct the two GELU constants
from their bits. Implement ascending reductions with explicit addition.
Python 3.12's built-in `sum`, `fsum` and vectorized reductions have different
reduction semantics. Preserve the [PLE scale and merge boundaries](../src/cpu/per_layer.rs)
and producer/consumer order from the [packed graph](../src/model/gemma_packed/forward.rs).
Derive coupled expected rows from authenticated weights and IDs alone.

The pinned CPython 3.12.8 arm64 math extension imports `_exp`, `_pow`, `_tanh`
and `_sqrt`. Use the first three on admitted finite operands. Check
`math.sqrt` independently against hardware binary64 square root on the
positive RMS domain. For paired trigonometry, use one small batched C wrapper
for the Darwin SDK's `__sincos(double, double*, double*)` and return both
rounded FP32 coefficients. Verify the compiled wrapper calls the paired ABI.

**Before a coupled run**, bind the release binary, CPU/OS build, Python
executable and math extension, compiler flags and wrapper binary. Verify live
provider addresses and loaded image UUIDs, round-to-nearest-even and subnormal
handling in the actual worker. Check signed zeros, halfway stores, cancellation,
softcap single-rounding, masked softmax and inactive rotary dimensions using
independent exact/high-precision fixtures and selected host primitive witnesses.
Keep any unresolved provider or contraction detail blocking. Start with four
tokens and measure before moving a 512-position host operation into C.

## Paired primitive and actual-process witness

The explicit C/Python helper now batches up to65,536 binary64 angles within
±512 and returns interleaved FP32 sine/cosine pairs. It validates the complete
batch before writing, checks RNE and preserved FP32 subnormals, and calls the
compiled Darwin paired ABI. Typed FFI arrays have separate aligned ownership.
Live symbol lookup records provider paths, addresses and loaded Mach-O UUIDs
for the five named operations. Four controls pass in the sealed worker:
signed zeros,100-digit trigonometry/midpoint controls, bounded inputs, live UUID
stability,200-digit sqrt and selected libm comparisons. Source/runtime hashes
are unchanged at both endpoints; Astra approves this primitive/witness scope.

UUIDs identify this worker's live lookup images. Future coupled callers must
repeat the provider/environment checks and retain their import evidence.
Sampled accuracy controls cover the tested inputs. The independent host graph,
selected native witnesses and coupled comparisons remain open.
[Witness evidence](../benchmarks/results/2026-11-04-gemma-native-host-v1.json).
