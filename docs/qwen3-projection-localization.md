# Qwen3 C512 same-input projection diagnosis

The [rejected native C512 trial](qwen3-context512-native.md) has identical captured
state across chunk schedules but exceeds its unchanged canonical FP32 gates.
This follow-up distinguishes a local projection error from inherited input
differences. It changes no numerical kernel, precision, gate or capacity.

Before computing dots, a frozen plan selects 14 original-case layer27 Q/K/V
coordinates: two canonical failures and two passing controls per projection,
plus the two V features at position65 with disjoint CPU-reference intervals.
All selected byte ranges are authenticated against the full native/CPU captures
and original checkpoint. Astra independently checked their journal, array and
safetensors-header offsets and historical gates.

Only selected rows are read. The original little-endian U32 rows contain four
LSB-first eight-bit codes per word, with BF16 scale/bias per64 columns. Each
affine weight is rounded once to FP32. For each of the native, CPU-FP32 and
CPU-FP64 saved Half input rows, an exact rational1024-term dot is rounded once
to nearest-even Half. This tests the local projection boundary without loading
the full model graph or retaining expanded weights. A separate publication
replay uses binary search over representable values instead of the diagnostic's
three-candidate rounding algorithm.

All **42 own-input results** reproduce their saved Half outputs exactly:
14 native,14 CPU-FP32 and14 CPU-FP64. There is no extra local arithmetic error
beyond correct final Half rounding at these coordinates. The six selected
canonical failures and two additional V coordinates still fail the original
gates; all six passing controls pass. The exact decomposition is:

`native_saved - fp32_saved = native_local + inherited_input + reference_local`

The local terms are final Half rounding errors. The inherited term is the exact
dot of the weight row with the difference between native and FP32 saved inputs.
For example, V feature25 at position65 is0.68115234375 natively versus
0.4248046875 in canonical FP32 and0.76513671875 in the FP64 diagnostic. Each is
the correctly rounded dot for its own different input. This supports keeping
the Q/K/V kernels unchanged; it does not locate the first upstream divergence
or qualify unobserved coordinates, a full model, source parity or quality.

The31 cold tests pass, including nine new exact-rounding, packed-layout,
decomposition, schema, authentication and retained-partial-result controls.
On the same M1 Pro/16GiB Mac, external diagnostic physical/RSS peaks are
288,441,208/271,482,880 bytes, within512MiB. The two diagnostic groups close.
The initial controller's reference-environment rejection and the first
publisher's access-time endpoint mistake are preserved; the corrected cold
run and publication reuse unchanged numerical sources/data. Production Rust,
shaders and protected release binaries remain unchanged.

See [complete public evidence](../benchmarks/results/2026-10-27-qwen3-projection-localization-v1.json).
The existing C512 qualification remains rejected. Source/held-out quality,
broader model context and independent phone qualification remain open.
