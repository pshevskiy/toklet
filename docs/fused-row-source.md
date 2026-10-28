# Checked fused row-span source loading

B2a adds a storage mechanism for the selected [Phi3 composition](phi3-fused-source.md).
It does not yet admit or execute that architecture. Existing scalar arithmetic,
whole-tensor readers and numerical kernels are unchanged.

`affine::Layout::row_span` validates a nonempty output-row range and returns the
selected layout and original tensor-relative byte ranges. Linear weights remain
row-major `[out,in]`; groups run across input columns. Codes, scales and affine
biases retain their original packing and row order. `Matrix::row_range` borrows
the same three owners using this shared checked plan.

`weights::Source::visit_raw_range` checks both absolute seek endpoints before
seeking, then streams through the existing 64 KiB source buffer. There is no
per-range allocation or typed pointer cast; unaligned safetensors data starts
are valid. The caller owns its destination and must discard partial output after
a read or consumer error. An unaudited source can explicitly reread after error.

`qwen::Source::read_affine_rows` checks all three destination sizes before any
write, copies the selected original bytes and validates selected finite
coefficients. `visit_float_rows` checks a rank-two F16/BF16/F32 selection and
streams finite little-endian values. It dispatches dtype outside scalar loops.
Unselected metadata is deliberately not validated by a partial read.

A range read grants no whole-tensor content identity, including when it covers
the complete tensor. Every range attempt on an audited source rejects before
seek/callback and clears and sticky-poisons its identity. Existing whole audited
readers retain their proof path. These contracts prevent later fused consumers
from accidentally claiming a whole-artifact audit from selected bytes.

Formatting, strict default/Metal all-target Clippy and all eight owned check
groups pass. Full default tests:397 passed; Metal library:278 passed; targeted
row:18, raw range:4 and Metal Llama:9 passed. Suites overlap; explicitly ignored
hardware/model tests were not run. Controls cover all six CPU affine bit widths,
three coefficient dtypes, groups32/64/128, span/offset overflow, unaligned input,
buffer crossings, partial failure/retry, selected NaNs and sticky audit rejection.
The initial development command repeated `--lib` and Cargo rejected it before
compilation; its corrected targeted invocation passes.

Astra reviewed the proposal before application and the final source/checks
before commit. No full Phi payload, GPU execution, performance or model-quality
result is claimed. B2b next composes one cold logical-to-physical row plan into
CPU/Metal loading, strict Phi configuration and independent tiny graph checks.
Physical source bytes must be counted once; resident allocation padding must be
counted for the actual logical owners. See [evidence](../benchmarks/results/2026-10-28-fused-row-source-v1.json),
[TODO](../TODO.md) and [improvements](improvements.md).
