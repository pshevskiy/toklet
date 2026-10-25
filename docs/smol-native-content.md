# Native Smol source consumption evidence

`mlx::qwen::Source::open_smol_audited` adds an explicit audited loading path for
one pinned SmolLM2 checkpoint. Its private-field immutable `SourceContentIdentity`
is historical raw-byte evidence. Opening authenticates metadata; identity remains
unavailable until every required physical range has been consumed successfully.
This does not construct or bind a live model. The
[receipt](../benchmarks/results/2026-10-25-smol-native-content-v1.json) records source,
checks, actual reads and completed process resources.

## Trusted content and metadata

The [independent range witness](smol-load-identity.md) fixes
`mlx-community/SmolLM2-135M-Instruct-8bit` at
`0f0d9b8218915bc34d401e1a340b8c049d300d5e`. The native audit authenticates that
checked-in witness with a literal digest. Callers cannot supply replacement
expected hashes or construct a content identity.

Config and index are verified and parsed from the same bounded byte buffers.
The exact eight-byte length prefix and raw header are verified from the admitted
shard handle before parsing its tensor metadata. Qualified opening uses the
already locked Unix libc version for no-follow/nonblocking flags, then requires
regular handle metadata. No-follow covers the final path component; parent
directories may be linked. Other platforms reject this explicit opening path.
Legacy opening behavior and unqualified identities remain unchanged.

The audited source retains its original shard handle. Every tensor read checks
its trusted offsets, shape and dtype, and the handle's exact file extent before
and after delivery. A persistent appended tail or truncation rejects. Replacing
the path does not replace that handle or the admitted metadata.

## One raw delivery boundary

A bounded sorted vector holds 694 trusted ranges, their digests and a seen mask.
A detached stack ticket hashes each raw chunk before that same borrowed chunk
reaches its consumer. Floating conversion chooses the dtype once and reuses this
raw boundary; packed weight/scale/bias uploads use it too. Exact count and digest
must pass before a range becomes complete. Every repeated read verifies again.
There is no growing per-read history and no numerical hot-loop hashing.

Starting a raw operation provisionally prevents proof publication. Only that
same successfully completed operation may clear the provisional state. Read,
seek, extent, consumer, digest or unwind failure leaves the audit permanently
unable to publish proof. The public floating/affine operations also poison after
post-read validation failure or invalid arguments. A subsequent valid attempt
cannot restore publication; reopen a fresh source instead. Because chunks reach
the consumer before the final range digest, callers must discard partial
unpublished destinations on any error.

The identity means all required metadata and payload ranges matched their trusted
reference when consumed. Its container digest is the **authenticated reference
anchor**; combining per-range hashes does not recompute whole-file SHA-256.
There is no new sequential whole-container digest claim. This is not an atomic
current-disk snapshot or proof that a caller assembled a model successfully.
Prefer obtaining the value only at the final source-consumption boundary. A copy
owns no file/path reference and remains historical evidence after later path
changes, source drop or a subsequent failed read.

## Admission and checks

Qualified opening requires the existing 64 MiB metadata plus 64 KiB stream budget
before witness parsing. Retained ledger capacities and names/shapes are counted
inside that metadata policy; temporary parsing remains within the cold metadata
allowance. No second model or checkpoint payload is retained by the audit.

Thirteen raw-source methods and four public-consumer methods pass offline. Tests
cover complete/incomplete and repeated reads, mutation/restoration despite equal
endpoint hashes, historical copies after path replacement, callback failure and
panic, nonfinite conversion, exact extent, offsets/dtype/shape, checked counters,
unknown names, short destinations and post-read affine metadata validation.
A range larger than 64 KiB is truncated after its first callback: the next
`read_exact` produces `UnexpectedEof`, and restoring the file does not resurrect
the audit. Regular, final-symlink, directory and FIFO cases are explicit. The
invalid-handle test does not claim a distinct seek-stage fault injection.

Formatting, default/Metal strict checks, all 329 CPU/352 Metal-feature tests,
both iOS targets and release repeats of the 13/4 new methods pass. The first
preliminary strict check failed on a deprecated dtype-size API and a needless
borrow; its exact sources/logs are retained uncredited, followed by corrected
passing checks. No warning or tolerance was waived. Seventeen controller schema/
counter/authority corruptions reject. The regression wrapper proves successful
owned execution only, without an invocation-failure retention claim.

The explicit Mac source diagnostic consumes all 272 logical tensors: 211 packed
matrices and 61 floating vectors. Two additional embedding reads exercise six
repeated weight/scale/bias visits, for 700 verified raw reads across all 694
physical ranges. Payload coverage is 142,955,136 bytes in the 143,030,401-byte
container. Identity survives dropping the source. Retained ledger ownership is
121,326 bytes and stays unchanged during reads.

Source admission is 210,129,536 bytes, with a separate 67,108,864-byte diagnostic
reserve inside a 512 MiB process policy. On M1 Pro/MacBookPro18,3 with 16 GiB RAM
and macOS 26.6.2, completed RSS/physical peaks are 36,372,480 / 35,094,912 bytes.
The 1.40-second combined source observation is not an inference benchmark or
phone-memory qualification. The diagnostic admits per-tensor destinations and
discards them rather than retaining a full model.

## Remaining boundary

A later engine task must carry this source evidence through successful resident
model construction, then bind the separately consumed tokenizer/profile and
current device/session admission before granting capabilities. No native model
handle, ABI, app eligibility, graph, numerical context limit, quality, GPU or
phone qualification changes here. Gemma4 and 131072-output requirements remain
open in the [active plan](gemma4-implementation-plan.md).
