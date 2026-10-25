# Gemma4 cold payload storage

The [receipt](../benchmarks/results/2026-10-25-gemma4-payload-storage-v1.json)
records the cold storage mechanism following the
[E2B metadata adapter](gemma4-config-layout.md) and
[bounded native packed graph](gemma-packed-model-scalar.md). This task does not
download or numerically qualify a real checkpoint. Public success still needs
an authenticated real payload; the small success tests use private synthetic
plans and cannot admit a new public geometry.

## Admission and ownership

`mlx::gemma4::payload::load` consumes a Rust `Read` stream plus bounded config
and index bytes. The caller must supply an independently trusted container byte
count and binary SHA256, a memory allowance and explicit modality policy. A hash
read from that same untrusted stream is not authentication. The loader reads the
actual length prefix and at most1MiB of header, then passes that same header to
the existing strict E2B inspector. No Hugging Face repository code executes.

The immutable `PayloadStore` owns only text tensor storage and its metadata
plan. Packed matrices retain separate little-endian U32-code, BF16-scale and
BF16-bias byte vectors. BF16 norm/scalar vectors become FP32 once. Cold named
accessors build borrowed `Matrix` views or return borrowed norm slices; callers
resolve these before numerical execution and retain the store for their lifetime.
No self-referential handle, storage trait hierarchy, whole expanded matrix,
second retained model copy or executable architecture registration is added.

Ordered spans cover the entire payload. A single streaming pass hashes the
length prefix, original header, text bytes and excluded audio/vision bytes.
Excluded bytes use reusable64KiB scratch and are never retained or numerically
interpreted. Exact payload length follows from the admitted header coverage;
a one-byte EOF probe rejects appended data. Digest mismatch, truncation, I/O
failure or value rejection returns no partially populated store.

Every converted vector and packed coefficient must be finite. Codes0 and15
are reconstructed using the existing FP64-affine-to-FP32 convention. Both
endpoints must remain finite, which bounds every intervening four-bit code.
This intentionally stronger admission also checks unused codes. It handles
negative scales and both overflow directions without expanding a matrix.

## Memory reservation

Before large payload buffers, derive retained packed bytes plus FP32 norm bytes,
the largest raw BF16 vector conversion overlap,64KiB streaming scratch and a
64MiB cold metadata reservation. The reservation includes live caller
config/index/header, parser structures, tensor maps/names/descriptors, vector
owners, SHA state and small bookkeeping. It is conservative logical accounting,
not measured process memory or a device allowance. Cold metadata itself is
admitted before header allocation. Actual payload/vector capacities and live
conversion overlap are rechecked, with the largest admitted reservation retained
in the report, including allocators that round capacity upward.

For the pinned E2B metadata, retained tensor bytes are2,604,553,356 after
571,974 bytes of BF16 vectors become FP32. The largest raw vector is3,072 bytes;
the derived load reservation is2,671,730,828 bytes. These are metadata-derived
requirements; this task supplies no real-payload measurement or memory-fit claim.
Sessions, tokenizer/application storage and general process memory need their
own admission. Phone memory policy is independent of the desktop6GiB policy.

## Checks and remaining gates

Five offline tests cover fragmented reads, authenticated excluded bytes,
borrowed views, finite metadata/vectors, negative scales, overflow in later
metadata entries and both directions, wrong digest, truncation, trailing data and I/O
failure. They verify exact-budget success and rejection before payload reads.
The public loader consumes the actual frozen E2B header then rejects a small
allowance without reading payload. Explicit modality rejection and malformed
length-prefix/header cases remain fail-closed. The private synthetic success
contains one matrix, one vector and deliberately uninterpreted excluded bytes.

Formatting, strict default/Metal Clippy, default/Metal tests and both iOS compile
checks are bound in the receipt. Earlier preliminary source/output is retained;
peak-reservation reporting was strengthened before final freeze. Astra reviews
the final source and evidence before commit. No debug timing is a performance
claim. Authenticated real acquisition/load, actual E2B geometry execution,
tokenizer/generation/chat, resident Metal, quality/performance, Mac/iPhone fit and
app integration remain open. Existing packed/dense GPU and long-rotary
rejections remain unwaived; no ABI, eligibility or output/context limit changes.
