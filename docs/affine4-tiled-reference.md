# Bounded four-bit rows for independent checkpoint references

The [receipt](../benchmarks/results/2026-10-25-gemma4-affine-tiles-v1.json)
binds the offline checks and explicit real-data probe for the independent
`affine4_tiles` helper. It supports only packed4/group64 weights with
little-endian BF16 scales and affine biases. This is reference tooling outside
the Rust engine; it changes no architecture, ABI or model eligibility.

## Representation and precision

The caller authenticates a seekable source descriptor and its container/schema,
and owns its lifetime. Each matrix describes three nonoverlapping contiguous
row-major file spans. All sizes and ranges are bounded before reading; at most
128 rows and 12,288 columns are reconstructed at once. There is no whole-table
expansion or memory mapping. Repeated and nonmonotonic row lookups return
independent contiguous FP32 rows.

The helper decodes low-first nibbles from bytes and BF16 metadata from exact
bits. It multiplies and adds in separate FP64 operations, then rounds each
weight once to FP32. Both affine endpoints must remain finite in FP32, including
unused codes. Negative scales, subnormals and signed zero are covered.

Projection uses NumPy/BLAS FP64 accumulation over those FP32-reconstructed
weights, with explicit FP32 or FP64 output. This differs from exact-affine FP64
weights: an authored cancellation example distinguishes them without a tolerance
floor. BLAS may reassociate FP64 sums; comparisons use the existing unchanged
`1e-5 + 1e-4 * abs(expected)` gate against independently decoded scalar `fsum`
and sequential contracts. No bitwise scalar projection equivalence is asserted.

## Resource boundary

Projection admits 64 MiB of conservative NumPy-array overlap plus its output
before source conversion or row reads. Output is bounded at 32 tokens by
262,144 rows: 64 MiB for FP64, making the maximum array reservation 128 MiB.
The overlap includes old/new tiles, raw packed/metadata buffers, BF16 conversion,
FP64 reconstruction and dot operands/results, FP32 rows and activation conversion.
An allowance one byte below the requirement rejects before any source reads.

Interpreter/package residency and private BLAS workspace need additional caller
admission. The cold process admits external package and native-runtime manifests before
NumPy imports, checks actual imported modules and mapped non-system images,
disables user-site imports and uses an exclusive fresh bytecode-cache prefix.
It parses the first bounded captured bytes and rejects changed recaptures.
Three capture regressions and three existing runtime-admission regressions
cover these boundaries. The real-data controller uses one thread and a separate 256 MiB
process allowance, including runtime/BLAS, metadata, hash/read and capture
reserves. Its completed lifetime resource report checks both maximum resident
memory and peak physical footprint. This reference allowance is not inference
or app memory admission and cannot imply iPhone fit.

## Checks and real-data scope

Eight offline NumPy tests cover frozen small and actual embedding widths,
independent scalar sums, the 128-row tile boundary and tail, signed zero and
widest rows, precision cancellation, invalid extents/types/overlap, unused-code
and output overflow, truncation/oversized/read failures and descriptor ownership.
Actual driver failure/interruption/endpoint tests retain completed raw arrays,
binary hashes and the original error. The first preliminary run retains a
rejected receipt-field lookup error; the corrected seven-test run and later
eight-test source/evidence remain separately available.

The explicit Gemma4 probe authenticates the pinned checkpoint already admitted
by the [acquisition task](gemma4-checkpoint.md). It compares three rows per packed
text matrix against a separate word/nibble scalar decoder. It also checks two
complete two-input projections: layer0 K `[256,1536]` and layer15 MLP down
`[1536,12288]`. Raw scalar and NumPy arrays and per-row hashes remain retained.
All834 selected rows match bitwise, and all eight projection comparisons pass.
The reference process reports161,611,776 bytes maximum resident memory and
76,759,832 bytes peak physical footprint, separately below256 MiB. The first
real attempt is retained as failed: runtime admission rejected an unlisted
Python app executable before any model reads, and the sandbox separately denied
process-resource reporting. The retry explicitly pins that executable and uses
the reviewed resource-query access. Numerical and memory gates are unchanged.
This is selected-row coverage across the matrices, not every stored element.

The real E2B graph, selected states from every layer, final logits, tokenizer,
chat/generation, resident Metal, quality, representative engine performance,
phone admission and app integration remain open. The existing GPU and
long-rotary failures remain unwaived. No original-BF16 graph or output/context
limit is qualified by the reference helper or matrix probe.
