# Frozen real LFM2 reference metadata

This is source-only qualification for
`mlx-community/LFM2-350M-4bit@18dc72abf3b2337f9123cfd6eeeb58dfa7947066`.

- [protocol.json](protocol.json) is the byte-identical prospective protocol,
  fixed before source outcomes. Historical no-execution fields describe its
  design stage; the qualified result records the later actual source execution.
- [components.json](components.json) is the byte-identical exported component,
  counter and saved-invariant record.
- [gates.json](gates.json) freezes all 3,102 literal MAIN comparison coefficients,
  routes, shapes and diagnostic envelopes, with exact source-manifest/array pins.
  It contains coefficients only; the numerical centers remain in the retained,
  authenticated source arrays.

The complete 48,049,152-byte array capture, 2,247,239-byte manifest and raw journal
are retained under the owned experiment, with exact four-file export/replay.
The [result record](../../../benchmarks/results/2026-10-29-lfm2-real-reference-v1.json)
pins both runs, their runtime/cleanup observations and independent audit.
Use the [producer](../../../tools/reference/lfm2_real_half.py) with this exact
protocol and authenticated model bytes to reproduce the reference under the
documented isolated NumPy runtime; ordinary CPU tests require no model download.

The audit checks every descriptor, finite value, gate and state invariant. It
reconstructs selected local nonprojection operations and first/last coordinates
of every captured affine invocation using an independent exact rounding method.
Remaining projection coordinates and uncaptured teacher/producer formulas have
the limitations stated in the [report](../../../docs/lfm2-real-reference.md).
This is not native/runtime/Chat, full-context, phone or adoption-quality evidence.
