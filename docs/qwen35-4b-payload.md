# Qwen3.5-4B authenticated CPU storage

Q1.B is implemented and Astra-approved in its CPU storage scope. It extends the [pinned cold recipe](qwen35-4b-cold.md)
with a sequential CPU payload loader for the selected single-shard MLX source.
It provides no hybrid graph, recurrent state, codec, Metal route or adoption.

The loader requires the fixed config/header and complete source SHA256 before
returning a store. It accepts no caller-selected expected payload digest.
Original affine4/group64/BF16 codes/scales/biases stay packed in249immutable
matrix owners. The153BF16 vectors and24depthwise convolution weights decode exactly to
FP32; convolution storage keeps the original `[channels,4,1]` channel/tap order.
The unaligned source origin is consumed as bytes, without typed mappings.
Cold names resolve borrowed views for a future concrete executor; nothing
selects a kernel or looks up tensors in a numerical loop.

Finite metadata and finite endpoint reconstruction are required. Existing MLX
negative and zero affine-scale semantics remain valid. Vectors and convolutions
reject nonfinite BF16 values. Truncation, trailing data, source corruption,
missing ownership or admission failures return no store or content receipt.
The receipt describes consumed historical bytes, not a current path or codec.

Requested retained tensors total2,369,040,384bytes. The load allowance adds a
64MiB metadata reservation,64KiB streaming allowance and64KiB maximum raw BF16
conversion overlap, totaling2,436,280,320bytes. Actual Vec capacities are checked
as owners are acquired; the caller's limit is capped at6GiB. These are CPU
storage/load owners, excluding tokenizer, sessions, Metal and framework memory.
A future GPU performance loader need not retain this CPU reference copy.

Six independent small controls cover low-first packed nibbles, exact BF16 bits
and signed zero, four-tap ordering, negative/zero scales, finite overflow,
budgets, corrupt/truncated/trailing streams and ownership failures. Fresh624CPU/
756Metal tests (36/298ignored), formatting and strict CPU/Metal all-target Clippy
pass. Five Python controls and six damaged-evidence checks pass. Initial compile
type mismatch and model-specific shared-reader labels are fixed; earlier logs
remain intact. Final3,910source/test/Cargo pins match the frozen release binary.

On Apple M1 Pro/16GiB, two fresh release CPU captures independently match all
924retained components and747sampled affine rows (2,698,752FP32values each).
Both row bodies have the same hash. Positive lifetime physical peaks are
2,381,564,208B and2,381,596,976B; a reader-only last-byte corruption is rejected
before store publication, with2,377,206,064B peak and no source-file mutation.
Load-only observations are8.243s and7.905s. Gemma jobs ran concurrently; there
are no latency, overhead, sustained, graph or device-fit performance claims.
The independent reader authenticates source endpoints and validates original
packed hashes, exact small BF16 expansion and full selected row bits.

Astra approves with no findings. [Verification](../artifacts/qwen35-4b-payload-2026-11-08/verification.json)
links final checks, raw captures, source proofs and controls.
[Checks/review/commit](../artifacts/qwen35-4b-payload-2026-11-08/publication-checks.json)
records publication separately.

Hybrid numerical/state fixtures, source lineage/quality, tokenizer/stops,
session/cancellation/admission and request metrics remain separate gates.
Neither declared context nor payload size qualifies actual model/device fit.
iPhone runtime remains deferred.

[Contract](../artifacts/qwen35-4b-payload-2026-11-08/acceptance-contract.json),
[source authentication](../artifacts/qwen35-4b-payload-2026-11-08/model/checks.json),
[iteration](../artifacts/qwen35-4b-payload-2026-11-08/iteration-notes.json).
