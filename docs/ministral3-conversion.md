# Bounded Ministral 3 text conversion

Status: complete source authentication, all144 genuine controls, full conversion,
saved-byte audit and86 portable controls perPythonmode pass. Combined Astra
review approves scoped retention with no findings; publication evidence is linked below.

The candidate uses `xtools-at/Ministral-3-3B-Reasoning-UNCENSORED-heretic` at
`46fb9a42c0b66b0e762d08c1ea13a26d7ef915f4`. Both its complete BF16 source and
the pinned original source are acquired without constructing a model. Complete
byte authentication is distinct from ordinary source-package admission. The
stock index's optional parameter count disagrees with the strict physical-count
policy; its rejection is preserved. The modified index passes that check.
No index is repaired and the source marker/ordinary engine loader is unchanged.

The frozen text recipe contains 130 affine4 matrices, 53 affine8 matrices and
53 unchanged BF16 norms. The tied embedding/output, every output projection and
every down projection use eight bits. The packed embedding serves the tied head;
no second head is added. Groups contain 64 contiguous input features per output
row. Codes are little-endian U32, low bits first; scales and biases are separate
BF16 tensors. Output names retain `language_model.model.*`.

The planned child contains 602 physical tensors and 2,662,004,736 body bytes.
Its sorted compact MLX header has a 77,576-byte prefix including length/padding,
making the complete planned file 2,662,082,312 bytes. Prefix SHA256 is
`669cf6cf4405148e8c9ad3896be700a3dd765c2de352a1c5d476aa0865300413`.
These are layout facts, not quantization-quality or full-model fit evidence.

The producer bounds each tile to 224 complete BF16 rows and 9,216 columns:
4,128,768 source bytes at the largest shape. Actual tail rows are 32, 64, 128 and
160. Existing Granite and MiniCPM5 shape contracts remain unchanged. A new strict
source-span validator accounts for all 458 BF16 tensor declarations, including
the excluded rank-four vision patch convolution. Exactly 218 vision-tower and
four projector tensors are excluded from conversion; their bytes still
contribute to complete source hashes and captured exclusion records.

Every matrix tile must complete upstream GPU quantization/readback, then match
an independent NumPy FP32 code/metadata decoder byte for byte against completed
upstream MLX dequantization. The complete campaign requires 4,252 tiles and
3,428,843,520 reconstructed matrix values. Norms copy exactly. Every output
component is read back and hashed; whole source objects, child, metadata and
runtime endpoints must close before a candidate is qualified. Failed output
remains unqualified and uses fresh output/evidence paths on retry.

The saved-byte auditor independently rebuilds the expected layout, rehashes
actual source/child objects, components, tiles, norms and exclusions, and checks
strict ordered journal coverage and complete positive receipts. It reconciles
the captured independent/upstream FP32 witness hashes; it does not recompute
floating-point values or qualify runtime execution from those receipts alone.

Both tools reuse the unchanged bounded runtime inventory/admission mechanism.
The repository `.venv`, exact NumPy/MLX versions, complete package/source files,
actual module/native-image origins and endpoint identities must match. Cold
snapshotting hashes at most 1 MiB per read. Numerical imports and full-model
owners remain separate from source acquisition. Audit/setup commands retain the
512 MiB cap including 32 MiB helper reserve; the engine ceiling stays 6 GiB.
Sampled observations and terminal RSS do not establish a continuous allocator cap.

Retired suffix-only compiler cleanup increases observed free space by 6.55 GB;
all retained executables/shared libraries, model/raw evidence and the active
dependency cache remain available. Acquisition reserves both complete parents,
one planned child, the 2 GiB disk floor and 256 MiB of work space. Two network
streams stay inside one guarded acquisition cohort. Global free-space changes
may also reflect other host activity.

The new portable CI job runs the same four test modules in normal and optimized
Python without numerical packages, model downloads or Apple GPU availability.
Formatting, strict CPU/Metal Clippy and previous Rust/GPU results are reusable
only because 4,641 existing engine/Cargo/fixture/example/reference input files
match the checked `5c0eee96` snapshot. New numerical and actual-artifact checks
remain separate.

No codec, complete graph, quality, physical-fit, request latency, O1 overhead or
adoption decision follows from conversion. The original graph gates and complete
10,000-target/100-task quality comparisons remain mandatory before admission.
iPhone and interactive capture remain deferred.

[Frozen prospective recipe](../artifacts/ministral3-conversion-2026-11-13/prospective-conversion-contract-0.json),
[cleanup](../artifacts/ministral3-conversion-2026-11-13/cache-cleanup-closure-1.json),
[index policy check](../artifacts/ministral3-conversion-2026-11-13/original-index-inspection-1.json),
[portable controls](../artifacts/ministral3-conversion-2026-11-13/portable-combined-result-1.json),
[Rust check reuse](../artifacts/ministral3-conversion-2026-11-13/rust-check-reuse-1.json).

The full modified-source run passes all4,252 tiles and3,428,843,520 matrix values
with exact independent/upstream FP32 bytes. The child SHA256 is
`fe0acffd3bcac67f1c4bcf4e396f310dba94687e24c2a6a798e288ef48f4f7be`.
Observed sampled/terminal charges including reserve are below512MiB throughout:
source acquisition uses115MB, genuine controls490MB, conversion430MB and the
corrected saved-byte audit145MB. Owned cleanup is empty. Conversion completes in
95.03seconds; its phase sums include4.52s source reads,8.23s completed upstream
quantize/readback,26.07s independent FP32 decoding,5.92s completed upstream FP32
decoding and2.04s component write/readback. These diagnostic phases omit other
setup/hash/metadata work and do not measure model inference performance.

The first saved-byte audit rejected config/ancillary descriptors because its
strict schema omitted the producer's before/after identity fields. The corrected
checker validates all known fields against held descriptors, rejects unknown or
missing fields and detects same-byte replacement. Forty-one checker regressions
and the fresh complete audit pass; conversion bytes and numerical captures are
unchanged. The initial wrong environment-root invocation and CLI expectation
are retained too. The actual CLI reports `unknown`/`unknown_architecture` on both
backends; the expectation now preserves that explicit unknown state. No loader
or source-index policy changes.

[Source authentication](../artifacts/ministral3-conversion-2026-11-13/source-authentication-closure-1.json),
[genuine controls](../artifacts/ministral3-conversion-2026-11-13/probe-output-1/observation.json),
[full conversion](../artifacts/ministral3-conversion-2026-11-13/full-conversion-1/manifest.json),
[saved-byte audit](../artifacts/ministral3-conversion-2026-11-13/full-audit-2/audit.json),
[scalar report](../benchmarks/results/2026-11-13-ministral3-mixed-affine-v1.json),
[review/publication](../artifacts/ministral3-conversion-2026-11-13/publication-checks.json).
