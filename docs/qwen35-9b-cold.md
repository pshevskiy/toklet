# Pinned Qwen3.5-9B cold metadata

Q2.A adds a separate exact metadata inspector for
`lukey03/Qwen3.5-9B-abliterated-MLX-4bit` at
`8ff61b6a3af49b4f3973a5ed8ff0937de812c0d3`. It describes the text graph and
source storage before any payload read, dequantization, numerical execution,
model/session/GPU allocation or device admission. Required Rust checks and
release measurements and fresh package checks pass; Astra approves scoped retention
with no actionable findings. No
managed route is added.

| Role | Source tensor | Logical shape | Packed U32 shape | BF16 scales and biases, each |
| --- | --- | --- | --- | --- |
| Embedding | `language_model.model.embed_tokens.weight` | `[248320,4096]` | `[248320,512]` | `[248320,64]` |
| Output head | `language_model.lm_head.weight` | `[248320,4096]` | `[248320,512]` | `[248320,64]` |

These are distinct physical owners with nonoverlapping spans. The9B `Plan`
exposes `embedding()` and `output_head()` without a tied-head alias. The existing
4B inspector, tied-head API, payload loader and strict source pins remain intact;
matching a model-family name does not admit9B through the4B path.

The cold recipe has32layers:24delta/convolution owners and8full-attention KV
owners. It records H4096/I12288,16query/4KV heads with D256 and64rotated dimensions,
16delta key/32value heads with128-dimensional keys/values, four convolution taps,
and original BF16 vectors. RoPE sections remain `[11,11,10]` at theta1e7. Cold
shape/config/header validators are shared between these two concrete sources;
there is no dispatch change in numerical loops.

The captured header contains927physical tensors:250affine matrices,153BF16
vectors and24rank-three convolutions. Reconstructing logical tensor-element
counts from that header yields8,953,803,264, including the small vectors and
convolutions. The optional index reports8,953,801,728; its count is not used as
an ownership ledger. Payload bytes are4,476,370,944codes +559,546,368affine
metadata +549,888vectors +1,572,864convolutions =5,038,040,064. The complete
container is5,038,160,953bytes, starting its payload at the deliberately unaligned
absolute offset120,889. No typed alignment is inferred from that offset.

`inspect` requires exact raw config/header lengths and SHA256 identities, the
fixed container length and explicit text-only policy. It separately validates
all graph options, names, dtypes, shapes and complete gap-free/nonoverlapping
span coverage. It rejects duplicate/unknown/missing fields, tied-head substitutions,
changed packing, aliases, invalid spans and cross-source metadata. The declared
262,144position limit grants no capacity: implemented/qualified positions remain
`None`. No whole-file authentication or tensor-finiteness claim follows from this API.

The small fixture copies immutable upstream config/header bytes. Its inventory
is reconstructed independently from physical descriptors without Toklet code;
ordinary tests use no model download or Apple GPU. The8new controls cover source
identity, distinct head storage, hybrid owners, unaligned large offsets, malformed
graph/tensor metadata, duplicates/overflow and cross-source rejection. The
explicit ignored probe reads only config plus the length/header bytes and records
cold inspection/drop timing, not loading5GB of weights or serving a request.

Required formatting, strict all-target Clippy in CPU/Metal configurations and
652CPU/786Metal tests pass (42/320ignored). The original check driver completed
all five commands but redundantly recomputed source hashes in its finalizer;
that parent alone was stopped after its children exited. Independent finalization
verifies every log hash and all3,946unchanged inputs; the original exit143/logs
are retained. Two controlled actual4B loads and a nonmutating last-byte corruption
reader pass. Independent checks authenticate all924retained components and
747sampled rows/2,698,752FP32 values in each positive capture, with identical row
bits. Maximum loader process footprint is approximately2.380GB. This verifies storage and
admission behavior, not the rejected4B numerical trajectory.

On Apple M1Pro16GiB, Rust1.90 optimized release metadata cohorts use three warmups
and ten recorded samples per process. Actual9B medians are2.488/2.554ms;
IQRs0.159/0.054ms and physical process peaks4.932/4.998MB. Each inspection reads
123,320metadata bytes and zero tensor-payload bytes. These peaks are for the
metadata-only process and cannot establish a loaded-model fit.

The same4B probe compares old and new binaries with matched profiles, fixture
bytes and child allocator controls. Initial pair changes are+13.03%/+2.88%; the
first has a much wider candidate IQR. Four additional predeclared reversed-order
pairs yield−1.02%,+1.06%,+1.62%,+0.02%. All original samples remain; pooling
60samples per version gives2.438→2.495ms (+2.31%). The initial large difference
does not repeat; no repeated5% cold regression is established. This is no1%
bound, optimization gain, request-overhead, sustained or tail qualification.
AC power and before/after thermal-warning snapshots are recorded. Source and
binary endpoints match. All14cohorts/42warmups/140measured samples are retained.

Fresh native59 builds all three arm64 slices (macOS15, iOS18 and iOS18simulator),
with46exports per slice, strict C linking/platform checks and a verified
XCFramework. Tiny CPU and Metal C ABI execution smokes pass on the Mac.
This adds no9B C route or iPhone runtime qualification, and leaves the app's
native46 framework link unchanged. Package inputs and retained output hashes
close without changes. Astra independently verifies the diff, all inputs, fixture
inventory/ranges, raw measurements and recovered logs, and approves cold metadata
inspection only; no later qualification follows.

The [complete source-byte verification](../artifacts/qwen35-9b-payload-2026-11-09/verified.json)
was performed separately. Its success does not qualify a payload reader, untied
numerical head binding, the reference reader above4GiB,6GiB cold/steady fit,
codec/template/stops, numerical trajectory, request/overhead metrics or quality.
The [4B128/256rejection](qwen35-context256.md) and all original gates stay binding.

[Contract](../artifacts/qwen35-9b-cold-2026-11-09/acceptance-contract.json),
[fixture provenance](../artifacts/qwen35-9b-cold-2026-11-09/fixture-checks.json),
[initial targeted checks](../artifacts/qwen35-9b-cold-2026-11-09/targeted-tests-1.log),
[required checks](../artifacts/qwen35-9b-cold-2026-11-09/required-checks-1/checks.json),
[4B source controls](../artifacts/qwen35-9b-cold-2026-11-09/4b-payload-checks.json),
[independent source/row checks](../artifacts/qwen35-9b-cold-2026-11-09/4b-independent-checks.json),
[metadata measurements](../artifacts/qwen35-9b-cold-2026-11-09/metadata-summary.json),
[native59 package](../artifacts/qwen35-9b-cold-2026-11-09/native59-publication.json),
[Mac C smokes](../artifacts/qwen35-9b-cold-2026-11-09/native-smoke-59/checks.json),
[Astra](../artifacts/qwen35-9b-cold-2026-11-09/astra-review.json),
[checks/review/commit](../artifacts/qwen35-9b-cold-2026-11-09/publication-checks.json),
[cache recovery](../artifacts/qwen35-9b-cold-2026-11-09/cache-recovery.json).
The second cache cleanup ended short of its5.25GiB target after exhausting its
generated Toklet candidates; all57protected source/output files are unchanged
and5.18GiB remained free. The2GiB build guard holds;
[reconciliation](../artifacts/qwen35-9b-cold-2026-11-09/cache-recovery-2-reconciliation.json).

Next is separate pinned CPU payload ownership and full-stream authentication,
with physical storage measurements and unchanged4B eligibility. Untied numerical
execution and every later gate remain separate;
[next-scope analysis](../artifacts/qwen35-9b-cold-2026-11-09/next-scope-analysis.json).
