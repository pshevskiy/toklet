# Isolated scaled Metal rotary qualification

M8 B1c1, 28 October 2026. Retained after prospective policy review, CPU/build
checks, explicit Apple M1 Pro GPU checks and independent Astra readback review.
See the [source-bound evidence](../benchmarks/results/2026-10-28-llama3-metal-rotary-v1.json)
and [live checklist](../TODO.md). This operation remains **test-only** until the
next task composes it with model loading, memory planning and structural safety
admission. The scaled-artifact rejection guard remains in force.

## Mechanism and ownership

An optional private `Operation` owns its precise, noncontracting pipeline and
one resident FP32 inverse-frequency buffer. Canonical Llama3 frequencies are
computed in FP64 once at construction, validated after FP32 conversion and
written directly into the guarded shared allocation. No temporary host table
is retained. Allocation/device/budget checks precede initialization/compilation.
Existing ordinary shaders and the default pipeline inventory are unchanged.

The encoder borrows caller-owned contiguous Half input/output and status buffers
and a retained command. It checks dimensions, composite UInt32 indices, rounded
dispatch, buffer extents and operand aliasing before encoding. Absolute positions
are restricted to 0–511 for this task. FP32 position-times-frequency, precise
trig and separate pair products/sums precede checked Half storage. Invalid or
overflowing output sets status; a caller must reject publication on that status.
Encoding performs no buffer allocation or host wait. Callers retain resources
and exclude CPU access until command completion. Model/session integration is
not supplied by this isolated test operator.

## Gates frozen before GPU outcomes

Astra selected two simultaneous per-element gates, with independent Decimal80
mathematical and explicitly rounded FP32/Half profile witnesses:

- Mathematical error bound: `scale * (8 * 2^-23 * (abs(angle) + 1) + 0.0005) + 1e-5`,
  where `scale = abs(a) + abs(b)`. This is the unchanged engineering envelope
  from the existing ordinary high-position rotary test.
- Rounded-profile error bound: `8 * 2^-23 * scale + 2 * Half_spacing(expected)`.
  Spacing is the larger finite adjacent Half spacing, including binade boundaries
  and zero/subnormals. Each multiplication and add/sub rounds separately in the
  oracle; no fused pair arithmetic is assumed.

Both gates are mandatory. A bare Half-ULP-only gate is unsuitable near
cancellation because FP32 trig/product error depends on input magnitude.
These are prospective engineering criteria, not vendor guarantees, MLX source
parity or substitutes for full-model loss/task gates. The deterministic fixture
and both formulas were frozen and independently reviewed before target outcomes.

## Results and limits

Seven frozen cases cover widths 8/16/64, actual candidate Q/K head counts,
all frequency bands, identity scaling, offsets 0/5/final 511 and cancellation
dyadic Half inputs. Two repeats and partitions of one/two rows match bit for bit:
17,408 fixture output comparisons pass. Maximum mathematical error is
0.0009690819352, at most 0.82448 of its bound; maximum rounded-profile error is
1.1920928955e-7, at most 0.07003 of its bound. Guards are exact, inputs/frequencies
stay immutable, and controlled Half overflow signals status before a fresh
invocation succeeds. The separate unchanged ordinary regression passes its
13,824 values within its existing historical scope.

Formatting, strict default/Metal Clippy, 383 default / 261 Metal-library tests,
targeted 9/10 Llama tests and release test compilation pass; counts overlap.
Astra authenticates all source/binary/fixture/output bindings and ten closed
command groups. A private check-driver rewrite initially raised `NameError`
before starting children; it is retained. Exact replacement and syntax/main
preflight corrected the launcher. No numerical gate or fixture changed.

Measured hardware is Apple M1 Pro, macOS 26.6.2/build 25G83. The two isolated test
processes peak at 21,364,736 bytes RSS and 10,502,720 bytes physical footprint.
These are tiny test-process measurements, not real-model memory or speed claims.
No new model, Chat, larger scaled context, iPhone or source-profile quality
qualification follows. B1c2 must account for the shared frequency allocation and
optional pipeline, validate actual dense strides/index contracts, compose the
scaled Half graph/session and qualify it before removing the loader guard.
