# Streamed exact-affine reference for LFM2

B4e2a supplies an independent source operation for the real LFM2 reference.
It does not load a model into Toklet or qualify real-model inference.
The reviewed thirteen synthetic controls and all six repository checks pass on
Apple M1 Pro/macOS26.6.2 with Python3.12.8 and NumPy2.3.2. See the
[machine-readable result](../benchmarks/results/2026-10-29-lfm2-exact-affine-reference-v1.json).
Astra independently audits the saved controls, actual NumPy/Accelerate origins,
source identities, all six Cargo results and seven closed process groups.
Audit SHA256: `339cfb205e2aaeddc12a3eab5fce7a1b23aed191f9be2030c4626fc5866a3860`.

The reference reconstructs each original Q4/group64 coefficient with separate
FP32 multiplication and addition of the stored BF16 scale and bias. It accepts
finite stored Half inputs and rounds the exact dot directly to Half or FP32,
using nearest ties to even. Exact cancellation produces positive zero; a
negative nonzero result that underflows preserves its sign. Overflow fails.
No source center depends on a native output.

The implementation streams at most 128 output rows and 4608 input columns per
stripe, with at most 64 input token rows. It factors an exact power-of-two grid
from each coefficient/input row. A checked absolute-product bound at most 2^53
permits binary64 arithmetic with exact products and every partial sum. A larger
bound at most 2^63−1 permits checked signed-int64 arithmetic. Otherwise fixed
64-element subtotals are combined with exact Python integers; a block that
cannot be certified uses a bounded exact per-element fallback. Every branch
has the same mathematical result and direct destination rounding.
Exact binary64 arithmetic assumes the pinned backend executes ordinary
full-binary64 operations. NumPy reports Apple Accelerate with an unknown library
version; system shared-cache image paths are observed, rather than content-hashed.

The original-byte reader checks absolute offsets, row spans, overlap and short
reads. It handles unaligned metadata and returns owned decoded stripes. The
caller must authenticate the full artifact/header, count the complete mapping
including its header, and release streamed stripes. This helper neither
authenticates a checkpoint nor retains an expanded full model or Python integer
per model weight.

Synthetic controls use an independent Fraction/finite-word search oracle.
They cover certificate boundaries, stale certificates, cancellation, subnormal
and midpoint rounding, double rounding, zero multiplied by extreme operands,
the fallback paths, original nibble order and row/token partitions. These
controls remain independent of downloads and GPU availability. Actual real
activation certificate coverage, source duration, model memory fit, native
arithmetic, text, app integration and phone execution remain later tasks.

The explicit numerical owner reserve is 32 MiB; the synthetic worker's OS maxRSS
is 67,846,144 bytes below its 512 MiB ceiling. Five direct samples separately record
resident/physical-footprint observations. These are synthetic-process results,
not model fit or inference performance. The runner authenticates 2,785 runtime
files and checks actual imported modules/non-system native images after NumPy
import and at endpoint. Its 600 s job and 1 MiB-per-stream limits pass with normal
exit, clean process-group teardown and no endpoint errors. The initial two owner
versions were never run; the successor adds runtime-origin/nonblocking-open/
interpreter checks without changing helper or test bytes.

Formatting, strict all-target Clippy in default and Metal configurations, full
default 450/Metal 545 tests and the release Metal test build pass. Counts overlap;
20/181 explicitly ignored checks remain ignored. No GPU test is inferred from
compilation. Original Rust/Metal numerical code and previous reference fixtures
are unchanged.
