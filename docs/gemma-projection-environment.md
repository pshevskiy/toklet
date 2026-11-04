# Gemma projection pipeline environment on M1 Pro

The fresh production precise projection pipeline reports SIMD width32 on the
Apple M1 Pro. Its constructor checks Safe/Precise math and Metal2.3 settings.
Six literal probes run twice with exact raw operand/upload/output/status guards
and unchanged712-byte owner accounting. The independent checker verifies the
complete inventory and normal controls; subnormal observations stay descriptive.

| Probe | IEEE result | Observed output |
| --- | --- | --- |
| Smallest positive FP32 subnormal ×2^24 | 2^-125 | +0 |
| Smallest negative FP32 subnormal ×2^24 | −2^-125 | +0 |
| Two normal2^-126 inputs ×0.5 | 2^-126 | +0 |
| One normal2^-126 input ×0.5 | 2^-127 | +0 |
| Normal cancellation requiring FMA residual | 129×2^-30 | Exact |
| Positive zero control | +0 | Exact |

All27 authored width1536 projection outputs also match the independent CPU
SIMD replay exactly at the queried width, on both repeats. The full existing
normal GPU fixture passes its unchanged mathematical/admission/state gates.
PeakRSS is34,963,456bytes for that fixture and17,399,808 for the probes.
Required fmt/default+Metal strictClippy/default550/Metal658 tests and release
build pass. Two cold controls reject coherent residual corruption, changed
operands/guards/width and incomplete inventory. Astra approves this evidence.

The first launcher misidentified Cargo's library target kind before execution;
its initial script/logs remain. A sandboxed attempt could not access a Metal
device; a separate owned run with GPU access passed and exited cleanly. Required
checks and the numerical recipe stayed fixed.

These probes characterize the selected operands on this pipeline. They do not
justify a general flush emulator. The independent [CPU primitive](gemma-native-projection.md)
retains IEEE arithmetic. The first [coupled reference](gemma4-native-conformance.md)
must fail closed on unsupported subnormal inputs, products, residuals and
compensation intermediates, retaining the first affected operation. Begin with
the four-token graph. Four bounded workers may evaluate independent row tiles
only after floating-point-environment, deterministic single/four-worker and
memory checks. Full512 propagation and selected heads follow if it passes.
Historical rejections and app/phone qualification remain unchanged.
[Evidence](../benchmarks/results/2026-11-04-gemma-projection-environment-v1.json).
