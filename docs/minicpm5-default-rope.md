# S1.A: explicit default-RoPE Llama exports

Implementation, required Rust/GPU checks, native package/smokes and Astra
review pass; the publication record below carries the commit and integration evidence. This adds source syntax for the existing Llama graph. It qualifies no
MiniCPM5 payload, numerical trajectory, codec, physical fit, quality or adoption.

Before this change, the release CLI rejected the accessible MiniCPM5 export's
`rope_parameters` field. A nested-only control instead failed because the flat
`rope_theta` field was missing. The selected evaluation candidate is the publisher-labelled
uncensored [MLX8bit conversion](https://huggingface.co/bumblebuttpow/MiniCPM5-2B-heretic-abliterated-MLX-8bit/tree/50ab1b226834c290cdf68e460eb899e5cff7c61d).
The previous mondk4bit conversion now returns401 from pinned/HEAD/config
requests; its failure and historical metadata remain preserved.

## Source interpretation

For Llama dense admission, the nested declaration must be an object with exactly
`rope_type: "default"` and an explicit numeric `rope_theta`. It supplies the
ordinary positional plan's existing theta; no missing value becomes10000.
When the flat field is also present, both parsed arbitrary-precision JSON number
representations must match exactly. Differently encoded duplicates are rejected
without f32/f64 coercion. A nested default cannot accompany non-null legacy
`rope_scaling`; legacy flat and Llama3 scaling paths retain their behavior.
Other architectures retain their existing source schemas.

Interpretation happens once in an owned configuration value before the ordinary
finite-positive theta, geometry, dtype, packing, context and memory validations.
Publisher files are untouched. Numerical kernels, dispatch, buffers, session
state, text/stop semantics, telemetry defaults and managed profiles do not change.
The Qwen-only `Source::open` remains strict; this uses explicit dense admission.

## Metadata evidence and limits

The pinned header and index independently agree on973physical tensors:
296affine matrices, with separate packed words/scales/biases, and85small BF16
tensors. These become381logical source descriptors. The expected container is
2,674,327,346bytes, with an unaligned110386-byte payload origin. Both theta
declarations say5000000; the42-layer/2048-hidden/16-query/2-KV/128-head graph has
2,516,756,480logical parameters and separate embedding/head tables.

Only config, index and exact header bytes are committed as fixtures. Tests create
a sparse file of the declared length; its holes are zeros, not publisher weights.
This exercises cold admission without a download, GPU or model allocation.
The2,741,391,360-byte source admission is a planning estimate. Scalar expansion
exceeds6GiB and must reject before allocation. A compatible metadata report still
requires payload, runtime/session, text and qualification checks.

The conversion's declared insraq parent is accessible at `8d83eaf4`. Its config
saysBF16 while the card saysFP16; parent headers and complete tensor bytes remain
unverified. Full publisher/parent authentication and all later model gates stay
open. The6GiB Mac ceiling,512/32 pilot,512MiB codec allowance, existing4B/9B
deeper-context rejections and deferred physical iPhone scope remain binding.

## Checks and measurements

New CPU checks exercise both alias forms against the unchanged independent tiny
Llama stages/upstream logits and cached-prefix/reset/overflow cases. Invalid,
unknown, ambiguous, missing, coercible-but-different and out-of-range declarations
must reject. A separate explicit Apple GPU check covers the tiny Half oracles,
offsets, three cache partitions and valid-prefix preservation. These are not
real MiniCPM5 execution checks. Required formatting, CPU/Metal strict Clippy,
670 CPU/806 Metal tests pass (46/331 ignored); the five explicitly run GPU tests
include 60 new alias cache schedules and 3,840 logits. Numerical fixture arrays
and their gates are unchanged. Fresh native63 builds all three arm64 slices,
checks46exports each and strict C links/package hashes. Mac CPU and Metal ABI
smokes pass callbacks/reentrancy, score, continuation, cancellation, reset and
release. No app link or ABI changes; simulator/device builds qualify no physical
iPhone runtime. Astra approves scoped source-admission retention and commit;
its timing-consumer finding is resolved below.

The actual release CLI returns identical cold reports for legacy flat, nested
only and matching dual declarations. Four separate controls reject conflicting
theta, zero context, insufficient source budget and scalar expansion over6GiB.
Accepted metadata remains `compatible_untested`, requiring payload, device,
session, text and qualification checks.

The release probe measures completed metadata open/validate/drop over64opens per
batch, three warmups and ten measured batches, with warm filesystem cache and
raw hardware/power boundaries. It does not load weights or measure inference.
Before-change medians are2.072/2.831ms per open;36.6% between-run variation already
exists with the same binary. All samples are retained. No optimization,
continuous stability, tail, model-fit or1% telemetry-overhead claim follows.
After-change medians are2.118/2.118ms for legacy controls and2.145/2.107ms
for the modern dual declaration. Legacy changes are+2.17%/−25.20% against the
respective before runs, so the repeated≥5% regression trigger does not fire.
IQRs are0.019–0.022ms for after legacy and0.013–0.016ms for after modern.
All six cohorts retain60measured/18warm batches (4,992metadata opens); all twelve
AC/nominal/LowPower-off boundaries pass. Probe lifetime RSS peaks at10,076,160B,
which says nothing about real-model physical fit. Source compatibility is the
reason to retain this change; these timings establish no engine optimization. The unmeasured first probe confused
physical and logical counts; its source/binary are retained, and v2 names the
381logical descriptors explicitly.

[Contract and evidence](../artifacts/minicpm5-default-rope-2026-11-10/acceptance-contract.json),
[metadata preflight](../artifacts/minicpm5-heretic8-preflight-2026-11-10/preflight-summary.json),
[current-parser reproduction](../artifacts/minicpm5-default-rope-2026-11-10/current-parser-before.json),
[before timings](../artifacts/minicpm5-default-rope-2026-11-10/before-timing-summary.json).

[Required checks](../artifacts/minicpm5-default-rope-2026-11-10/required-checks-1/checks.json),
[explicit GPU checks](../artifacts/minicpm5-default-rope-2026-11-10/gpu-checks-1/checks.json),
[release CLI controls](../artifacts/minicpm5-default-rope-2026-11-10/cli-after-checks-1/checks.json),
[before/after comparison](../artifacts/minicpm5-default-rope-2026-11-10/metadata-comparison-v3.json).

Screen-capture qualification is deferred by the user's10November engine/CLI
priority. No screen-sharing interaction is needed for this task. Existing UI
WIP, failed capture evidence and unqualified first-visible metrics are preserved
separately; no UI claim blocks these console checks.

[Native package](../artifacts/minicpm5-default-rope-2026-11-10/native63-publication.json),
[CPU/Metal C smokes](../artifacts/minicpm5-default-rope-2026-11-10/native-smoke-63/checks.json).

Astra found that the original collector bounded each batch independently but
failed to enclose their sum within the completed process lifetime. The corrected
offline consumer requires positive integer monotonic owned timestamps and bounds
all13disjoint batches, including warmups, by that interval.10optimized-Python
methods pass, including resealed119-second-per-batch damage and warmup/exact-bound
controls. Revalidation accepts all six original cohorts and reproduces their
statistics;30raw evidence files and the original frozen collector are unchanged.
No new benchmark, model or Cargo run is needed for this consumer-only correction.
[Consumer correction](../artifacts/minicpm5-default-rope-2026-11-10/metadata-readback-v3-contract.json),
[offline revalidation](../artifacts/minicpm5-default-rope-2026-11-10/metadata-readback-v3-result.json).

[Astra review](../artifacts/minicpm5-default-rope-2026-11-10/astra-review-result.json),
[checks/review/commit](../artifacts/minicpm5-default-rope-2026-11-10/publication-checks.json).
