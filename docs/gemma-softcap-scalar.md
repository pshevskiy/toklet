# Scalar in-place logit softcap

A small standalone CPU function now applies the
[frozen tanh-logit contract](gemma-softcap-reference.md) in place. It reuses the
caller’s logits buffer without allocating a second full-vocabulary array.
The [receipt](../benchmarks/results/2026-10-25-softcap-scalar-v1.json) binds
source, accepted fixture, commands, raw logs and executable provenance.

`logit_softcap(&mut [f32], cap)` accepts a nonempty contiguous FP32 slice and a
positive finite FP32 cap. It admits the cap and every input value before writing;
late nonfinite input or invalid cap leaves every bit intact. Division, tanh and
multiplication use FP64, followed by one FP32 output rounding. For any finite
FP32 input and positive finite FP32 cap, the quotient fits FP64 and the result
magnitude is bounded by cap, so valid inputs cannot overflow output. Signed zero
is preserved. No allocation, extra alignment, dynamic dispatch or unsafe code
is introduced. Empty sequences make no call. A future model skips the operation
when its cap is disabled; cap0 is not an identity mode.

Four [methods](../src/cpu/softcap/tests.rs) authenticate the accepted reference
before fixture parsing and compare all five nonempty cases against ideal/staged
contracts with stable buffer pointers. Late invalid values/caps preserve bits.
Native-only min-subnormal through MAX caps check finite, odd, monotone and bounded
outputs, exact zero signs and scaling counterexamples. Independent chunks reproduce
full-vector bits with unchanged allocation/capacity. Those extreme-cap checks do
not qualify additional upstream checkpoint configurations or source zero-bit parity.

The [implementation](../src/cpu/softcap.rs) passes formatting, strict default/
Metal Clippy,264 CPU tests,286 Metal-feature tests and both iOS library compile
targets. The eight-command trial first cleans only copied Toklet package artifacts
inside its new private target, preventing inherited absolute CLI executable paths.
Seven CLI-spawning test binaries point into the current target; the eighth,
cfg-disabled default MLX-CLI harness, has an owned zero-test listing. Separate
CPU/Metal executable snapshots are retained. Ignored hardware suites stay
unexecuted; no timing or performance claim follows from build-cache use.

The numerical operator is available; the Gemma model graph does not call it yet.
Quantized/BF16 checkpoint, disabled-cap routing, full-vocabulary graph, session/
Metal, real model, device fit, quality, performance and long-reply qualification
remain open, including the [upstream rotary gate](gemma-rotary-precision.md).
No ABI, app cap, protected binary or installed package changes.
