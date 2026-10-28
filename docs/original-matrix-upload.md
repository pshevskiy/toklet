# Original packed Matrix upload

29 October 2026. B3c4a shares one cold byte-copy operation between the existing
Metal affine Context and a Matrix entry for resident graph composition. It
preserves the validated original packed codes and F16/BF16/F32 coefficient
words. No projection/gather shader, precision policy or model admission changes.

## Representation and ownership

The immutable Matrix supplies already-narrowed original byte slices. Nested row
views therefore copy their selected rows without reopening or rebasing a parent.
Unaligned source addresses need no typed loads. Destination packed weights have
their exact logical byte extent; coefficient owners round their byte lengths to
four-byte alignment, zeroing only the final zero to three padding bytes.

The helper checks full layout equality, all three destination lengths and the
existing GPU numerical domain before any write. Scalar-valid coefficients can
still be rejected by that domain, including FP32 subnormal metadata, tiny
reconstructed magnitudes and overflow for possible packed codes. Rejection
preserves every destination byte. No expansion or coefficient conversion occurs.

Context keeps its pre-allocation Matrix/domain/size checks and the same three
resident allocations. The copy boundary checks the numerical domain again for
direct callers. This is cold work; no new steady-state decode work is added.
The new Packed Matrix entry stays test-only until the resident graph consumes
it. Existing audited Source upload, including fused row selection, remains
byte-identical. Compile, product, gather and accounting functions are unchanged.

Callers exclusively own cold destination buffers and must exclude outstanding
GPU access. The source borrow ends with copying; it does not become resident
ownership. Apple little-endian destination bytes match the existing u32 upload
representation. No temporary complete model copy or new persistent buffer is
introduced.

## Checks and measured scope

Formatting, strict default/Metal Clippy, all 424 default tests across library and
integration suites, 313 Metal-library tests and three targeted debug/release
controls pass. Counts overlap. The new cold controls cover 162 combinations of
bit width, coefficient dtype, group width and parent/selected/nested row views;
they also check every destination-length error, same-size wrong-layout rejection
and scalar-valid/GPU-invalid coefficients.

Seven explicit frozen-release tests pass on the Apple M1 Pro Mac. The new
physical allocation/readback test retains original, uploaded and rejected bytes
for all 18 bit-width/dtype pairs. Astra independently reconstructs all 3,894
original bytes, 2,352 resident bytes and 2,352 unchanged bytes after rejection. The existing strided projection/gather test
now consumes that upload entry with its exact arithmetic and guards unchanged.
Four existing Context tests retain 54 storage cases, 864 completed affine/gather
calls and 390,096 exact value assertions, plus midpoint, nonfinite recovery,
odd metadata padding and subnormal controls. Those numerical values are checked
inside the existing Rust consumers, not retained as full external raw arrays.
The remaining test checks existing tiny Qwen session publication/replay behavior.

All processes finish within bounded ownership and source/runtime guards. Sparse
direct-process memory samples are retained; they do not establish execution
peaks or whole-model admission. This task claims no speed improvement, complete
Gemma3 GPU graph, real-checkpoint result, app integration or iPhone qualification.

See [machine-readable evidence](../benchmarks/results/2026-10-29-original-matrix-upload-v1.json)
and [the live checklist](../TODO.md). The next composition tasks freeze the
independent Half graph reference and build the concrete resident Model/Session.
