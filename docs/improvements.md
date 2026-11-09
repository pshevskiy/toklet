# Toklet improvement index

Updated **9 November 2026**. This is the compact index of retained capabilities,
measured gains and material restrictions. The [full dated history](archive/2026-11-07/improvements.md)
preserves every previous entry, source revision, failed check and evidence link.
Read it only when the current task needs that detail.

Measurements apply to the named hardware, route, workload and timing boundary.
Gains from different experiments cannot be added together. Current task and
qualification scope live in [TODO](../TODO.md); acceptance policy is in
[PLAN](../PLAN.md#validation-and-performance-policy).

## Retained performance decisions

| Change | Measured result and scope | Evidence |
| --- | --- | --- |
| Ordered CPU projection tile | Warm 128/128 TTFT improves about 77.6% and request time about 52% in two pairs. Qualified opt-in; scalar reference preserved. | [Retention](experiments/2026-10-12-cpu-tiled-retention.md), `0abbe60` |
| Exact committed-prefix reuse | On Q4 0.5B, editing position 128 of a 512-token prompt reduces first-token latency about 21.6%; editing only position 511 about 93%, in paired runs. No KV copying; not Q4 quality qualification. | [Continuation](experiments/2026-10-13-prefix-continuation.md), `340c965` |
| Selected Half/Q4 and Online attention | M5/M6 retained only their documented configurations and quality/memory/context scopes. FP16 KV retained after the I8 evaluation. | [M5 release](experiments/2026-10-15-m5-release.md), [M6 release](experiments/2026-10-19-m6-release.md) |
| Four-row native affine8 prefill | Pinned Qwen2.5-0.5B MLX8 on M1 Pro: 1,024-token prefill improves 13.69%, complete fixed-token request 6.22%. Mac opt-in; managed adoption remains separate. | [M9 decision](m9-mac-affine-decision.md) |
| Gemma BF16 G2 last-row prefill | Official BF16 Mac route: both primary runs improve prefill about 61% and request time about 28%; retained in native45. | [Retention](../benchmarks/results/2026-11-06-gemma-bf16-g2-retention-v1.json) |
| Exact guard validation reduction | Same complete guard coverage with automatic integer vectorization. Two Mac BF16/G2 primary runs reduce request time25.0–26.6%;626 sustained requests confirm26.3%,512 deeper decode pairs bit-identical. Required checks/Astra pass;6GiB preserved. No UI, memory-saving or broader-model claim. | [Scope](gemma-guard-validation.md), [checks/review/commit](../artifacts/gemma-guard-validation-2026-11-08/publication-checks.json) |
| Qwen2 prompt lookup | Repeated contract prompt gains about 53% decode, but the code prompt misses the 5% gate. Explicit nonstreaming API retained; default adoption rejected. | [Timing](../benchmarks/results/2026-11-06-m9-qwen2-lookup-timing-v1.json) |

## Existing correctness, memory and observability foundations

- Readable scalar CPU implementations and independent numerical fixtures remain
  correctness references. Real-model, reduced-precision and quality results have
  separate frozen scopes; [model coverage](model-coverage.md).
- Immutable weights and session-owned KV/scratch preserve committed versus pending
  token history, exact-prefix reuse, capacity admission and cancellation/replay;
  [session evidence](../benchmarks/results/2026-10-09-qwen-session.md),
  [Metal engine](experiments/2026-10-13-metal-engine.md).
- Resident weights, streamed loading, explicit layouts and bounded scratch avoid
  redundant full-model ownership. Zero steady-state Rust numerical allocations
  are measured only for their recorded routes; native/application allocations
  remain separately scoped; [Metal engine](experiments/2026-10-13-metal-engine.md).
- Generation reports, completed Metal timings, bounded calibrated GPU profiles,
  Swift callback metrics and runtime planning already exist;
  [current source inventory](observability-performance-proposal.md#evidence-already-available).
- Memory reporting separates model/session owners, allocated KV capacity,
  planning/admission allowances and measured footprint/conditional loading peaks;
  [owner evidence](../benchmarks/results/2026-11-02-session-owner-memory-v1.json),
  [KV limits](../benchmarks/results/2026-11-03-kv-preview-apply-v1.json),
  [load checkpoints](../benchmarks/results/2026-11-03-load-memory-checkpoints-v1.json).
- Managed Mac profiles and official Gemma public download/offline Chat have scoped
  app evidence; [journeys](mac-hub-journeys.md),
  [official Gemma](../benchmarks/results/2026-11-07-gemma-public-resume-mac-v1.json).
  Further iPhone work remains deferred.

## Restrictions that remain binding

| Result | Current decision |
| --- | --- |
| Historical failed quantization/precision trials | Keep failed evidence and gates; a later passing configuration does not rehabilitate an earlier candidate. [Full history](archive/2026-11-07/improvements.md#rejected-or-inconclusive-experiments). |
| Qwen3 fidelity and Gemma3 coupled intermediates | Preserve rejected trajectories and distinguish native conformance from original adoption quality. [Qwen3](qwen3-native-profile.md), [Gemma3](gemma3-native-conformance.md). |
| Split-KV and Llama Online | Do not adopt rejected arithmetic. [Split-KV](../benchmarks/results/2026-11-05-m9-split-kv-session-v1.json), [Llama](todo-implementation-plans.md#llama-online-correctness). |
| Gemma QAT task quality | Full-task loss exceeds policy despite passing loss/state/fidelity checks. [Decision](../benchmarks/results/2026-11-06-gemma-qat-full-quality-v1.json). |
| Gemma G3 assistant | Longer guard regresses about 21%; keep test-only, retain ordinary G2. [Decision](../benchmarks/results/2026-11-06-gemma-assistant-mtp-retention-v1.json). |
| New device/context/default adoption | Requires its own applicable checks. Mac execution, metadata support or admission estimates do not establish unrestricted support. |

## Console reference comparison

All78 requests pass across two3warm+10 cohorts for each of3engines on M1Pro16GiB.
Official BF16/G2 Toklet request medians2.184/2.202s; verified uncensoredMLX4bit
Ollama121/119ms and vLLM-Metal147/149ms. Precision, weights, cache hits and timing
boundaries differ; no engine speedup ratio or adoption qualification. All observed
memory guards and observed external lifetime sums stay below6GiB. Ollama0.40.15 and
local vLLM0.33.0/Metal0.33.0 are installed/verified. Original BF16 external runs
fail tested admission. The ordinary native affine4 inspector remains strict;
the mixed artifact uses the separate diagnostic below.
Offline report/document checks pass; Astra approves scoped publication.
[Comparison / limitations](console-engine-comparison.md),
[full evidence](../artifacts/engine-comparison-2026-11-07/console-comparison-report.json).

## Current observability work

[O1 Gemma summaries](o1-gemma-request-summaries.md) and
[O2.A native phase/reporting](o2-gemma-cost-report.md) are delivered with Astra
approval and required checks. Native46 preserves the legacy ABI and adds a
completed-report copy; summary schema2 carries phase/work counters. Matched warm
reports identify decode as the dominant phase; no kernel speedup is claimed.
The ≤1% overhead bound remains unqualified, so collection stays opt-in/default
disabled. [O2.B1 numerical coverage](o2-gemma-workloads.md) now adds repeated
load/continuation comparisons and 601 seconds of sustained work. Both batches
pass token/state and counter checks; maximum child-process lifetime peak is
4.924 GB, below 6 GiB. Six new portable tests, strict standalone Swift and Astra
review pass; unchanged production checks are reused by verified source pins.
No optimization gain or UI/whole-app/tail qualification follows.
[O2.B1 evidence](../benchmarks/results/2026-11-07-o2b-gemma-workloads-v1.json).
[O2.B2.A publication metrics](o2-ui-publication.md) add bounded Send preparation,
callback TTFT and controller handling/publication. Six real Gemma controls,
52 UI and 52 headless paired requests, fresh Swift/app builds and 29 portable
checks pass; Astra approves after a typed state/work join correction. Both overhead tracks remain
unqualified/default-disabled. Visible-frame qualification remains open; selected O3 scope is delivered below.
A bounded display-frame lab is now in validation: strict clock/pixel fixtures,
offline corruption checks, actual renderer crop inspection and fresh platform
builds pass. User selected the primary display; first attempt failed strict viewport visibility
before capture. Captured-frame/resource evidence and Astra remain pending; [current evidence](../artifacts/o2-ui-frames-2026-11-07/iteration-notes.json).
O3 cold/live explanation and first bounded349-event profiles pass inventory,
full-logit parity and failure/state controls. Fresh604CPU/726Metal tests, both
strict Clippy configurations and104 exact overhead requests pass. Observed
request-median capture changes−0.28%/+0.98% (ordinal2); default None changes
+0.22%/−0.09%. No1% bound or optimization claim. Exact admission5.814GB and
4.919GB live process peak fit6GiB. Three native slices/46 exports each and Mac C smoke pass; local native-lifetime fix/stress control and final Astra approval pass;
[scope/evidence](o3-gemma-runtime.md).
[Evidence](../benchmarks/results/2026-11-07-o2a-gemma-phase-report-v1.json).

The [R1 offline regression job](gemma-regression-job.md) now rejects incomparable
or reused collections and separates regression/noisy/speed signals over disabled
caller timings. Local named CI checks pass17 tests; Astra approval passes after
three fixes. It never accepts an optimization or qualifies missing quality,
memory, UI, thermal or tail gates.
[Checks/review/commit](../artifacts/r1-gemma-regression-2026-11-07/checks.json).

## Latest documentation work

Preferred uncensored Gemma cold-format planning is delivered with Astra approval. A separate pinned cold
recipe preserves277affine4/group64 matrices and one BF16 PLE projection, with
explicit unused-KV/media exclusions. Fresh610CPU/735Metal tests, strict CPU/Metal
Clippy and formatting pass; two3warm+10 actual header cohorts pass at about6.2ms.
No numerical execution, model quality,
memory fit or adoption follows from metadata inspection.
[Scope/checks/review](gemma-heretic-mlx.md),
[commit receipt](../artifacts/gemma-heretic-mlx-2026-11-08/publication-checks.json).

The [selected model plan](model-observability-plan.md) now makes observability
the first deliverable: defined latency/throughput/memory/state/outcome metrics,
bounded collection and paired overhead checks before optimization. Order:
Gemma 4 E2B, Qwen3.5-4B, Qwen3.5-9B; five strictly under-4B candidates follow.
Uncensored MLX is preferred with per-artifact quality/admission gates; conversion
gaps and an over-budget F32 derivative are explicit. Content/link/whitespace
[checks pass](../artifacts/model-observability-plan-2026-11-07/checks.json), with
binding acceptance policy unchanged. This adds no qualified runtime capability
in those documentation-only commits; later O1/O2 deliveries are listed above.

The user stopped O1 before integration/checks/Astra review and requested
[current model research](models-under-30b-2026-11-07.md). The ten-model shortlist
distinguishes total from active parameters and capability from local memory fit.
Documentation [checks](../artifacts/models-under-30b-2026-11-07/checks.json) pass;
those research commits claimed no new engine capability or device qualification.
O1 was subsequently completed as recorded above.

The [observability/performance proposal](observability-performance-proposal.md)
was committed as `338c4e5b`. It proposes correlated request summaries, a bounded
overhead budget, explainable selected operations and one measured performance
change at a time. Existing code/evidence and primary ClickHouse sources were
reviewed; [documentation checks](../artifacts/observability-performance-proposal-2026-11-07/checks.txt)
pass. It is a proposal, not an implemented capability or a new speedup claim.

The user then requested archiving earlier material to reduce context. Full
snapshots are preserved in the [archive](archive/2026-11-07/README.md), with
rebased links and [hash manifest](archive/2026-11-07/manifest.json). Live indexes
retain active work, acceptance policy and important limitations. Routine document
bytes fall about 97%; all 32 milestone criteria, four binding policy sections and
both open items are preserved. Integrity/link/content/whitespace
[checks pass](../artifacts/context-archive-2026-11-07/checks.json). This is a
documentation-only change, with no new runtime or performance claim.

Native uncensored Gemma has an explicit Mac diagnostic API.
Original277affine matrices plus the BF16 projection load through matched byte
proofs; all-row/G2 logit/state/fault-replay controls and512 deeper bitwise pairs
pass, observed control peak3.104GB. Final610CPU/742Metal checks, strict Clippy,
seven GPU regressions, three native slices/46exports and Mac C smokes pass.
Paired caller time improves26.2%; sustained322/434requests over600s each confirm
26.18% and stay below2.947GB. Astra approves scoped implementation/retention;
no held-out quality or managed-model adoption. [Scope](gemma-heretic-native.md),
[checks/review/commit](../artifacts/gemma-heretic-runtime-2026-11-08/publication-checks.json).

Independent mixed-source numerical/quality qualification is in progress. The
modified BF16 parent is separately authenticated and all1094consumed components
reproduce exactly. Independent32/64/512context checks pass5.77Mvalues/22rows
against FP64/stagedFP32 under the original gate; no greedy mismatches. Required
checks and combined Astra review pass with no findings; full held-out quality/adoption remain open.
[Scope](gemma-heretic-fidelity.md),
[checks/review/commit](../artifacts/gemma-heretic-quality-2026-11-08/publication-checks.json).

G1.Cb quality probes and rejection decision complete; Astra approves.611CPU/745Metal,27Python,
fmt/strictClippy, nine GPU controls, native52 three slices/46exports and both
Mac C smokes pass. Both full1,025fidelity gates reject despite exact sampled
IDs. Both10kNLL gates pass; isolated tasks score original52/modified50/
quantized43 of100. Reject quantized adoption:9pp loss versus original,7pp
versus parent exceed2pp. Both raw replays match; VM restoration verified.
Public512/6GiB and gates unchanged; typed aggregate-budget preflight fixed.
[Scope](gemma-heretic-quality-probes.md),
[checks/review/commit](../artifacts/gemma-heretic-quality-probes-2026-11-08/publication-checks.json).

Q1.A cold Qwen3.5-4B metadata validation is complete in its scoped recipe.
The separate pinned recipe describes24gated-delta/four-tap and8full-attention
layers,249affine matrices and177unquantized BF16 owners. Tensor extents retain
an unaligned byte origin.618CPU/750Metal, fmt/strict Clippy and two metadata
cohorts pass; Astra approves with no findings. No runtime, model-fit or adoption claim.
[Scope](qwen35-4b-cold.md),
[checks/review/commit](../artifacts/qwen35-4b-cold-2026-11-08/publication-checks.json).

Q1.B bounded CPU payload storage is complete in its scoped prerequisite: original packed matrices,
exact small BF16 decoding, immutable views and explicit admission accounting.
624CPU/756Metal, strict Clippy/fmt, five Python and six damaged-evidence
controls pass. Two source proofs match924components/747rows each;2.382GBpeak.
Astra approves with no findings. No graph/runtime/quality/adoption claim.
[Scope](qwen35-4b-payload.md),
[checks/review/commit](../artifacts/qwen35-4b-payload-2026-11-08/publication-checks.json).

Q1.C packed scalar Qwen hybrid model/session is complete in its scoped reference: distinct state
owners, atomic chunk publication, explicit memory/completed-work observations
and independent ideal/staged fixtures.630CPU/762Metal, strict Clippy/fmt,14Python
and five damaged-capture controls pass. Two actual32-layer CPU one-input
captures match, physicalpeak2.638GB, zero warmed allocations. Astra approves with no findings;
no independent real coupled/GPU/codec/quality/adoption qualification.
[Scope](qwen35-hybrid.md), [checks/review/commit](../artifacts/qwen35-hybrid-2026-11-08/publication-checks.json).

Combined reviewed G1.Cb/Q1.A–C source passes631CPU/765Metal, formatting and both
strict Clippy configurations. Native53 retains three slices/46exports and both
Mac C smokes pass. Code/fixtures exactly match the union of approved revisions;
only documentation required conflict resolution. [Integration checks](../artifacts/model-integration-2026-11-08/checks.json).

Q1.D independent actual-source Qwen oracle is complete in its one-input scope. Bounded row views
preserve the odd payload origin and original scalar arithmetic; all tiny fixture
bytes regenerate exactly. Nineteen Python/nine damaged-evidence controls pass;
both native captures pass32.135M full-array comparisons each, maxratio0.239,
no greedy mismatch. Closed reference peak0.540GB;3,923Rust inputs unchanged,
required/native receipts reused. Astra approves with no findings;
contextual/GPU/quality/adoption gates remain open. [Scope](qwen35-real-fidelity.md),
[checks/review/commit](../artifacts/qwen35-real-oracle-2026-11-08/publication-checks.json).

Q1.E real17input/offset37 replay is complete in its scoped prerequisite; Astra approves. The contract covers
nonzero recurrence/history, causal KV, full final-stage/logit comparisons,
chunk equivalence and failure/cancel/retry with completed-work counters.
27Python/12negative controls and fresh631CPU/765Metal, fmt/strictClippy pass.
Final native bits/recovery and32.667M comparisons pass, maxratio0.447, greedy22
matches. Reference/native peaks0.668/2.655GB. Earlier setup/report/comparison
corrections remain in evidence; no128/512/GPU/quality/adoption qualification.
[Scope](qwen35-context-fidelity.md),
[checks/review/commit](../artifacts/qwen35-context-2026-11-08/publication-checks.json).


Q1.F FP32 Metal projection diagnostic is complete in its isolated scope; Astra approves.
All 249 original Qwen matrices pass batches 1/17 in two owned processes:
24.846M independent comparisons each, no failures, max ratio 0.000580; peak
5.299 GB. Four cohorts reduce completed sequence medians from 8.047–8.149 s
scalar to 0.191–0.202 s Metal; retain this operator candidate. Required
631CPU/765Metal, fmt/both strict Clippy, 20Python/20evidence controls and the
four-case/88-step GPU transaction witness pass. Astra’s reader-admission P2 is
fixed with independently derived owner reconciliation/footer agreement; original
native receipts remain unchanged. Ordinary engine/package inputs
are unchanged outside test modules; no coupled GPU model, overhead or adoption
claim. [Scope](qwen35-gpu-projections.md),
[checks/review/commit](../artifacts/qwen35-gpu-projections-2026-11-08/publication-checks.json).


Q1.G concrete Metal projection/gather executor passes both matched performance
cohorts; Astra approves scoped retention, no findings. It consumes authenticated CPU matrix
ownership, retains original GPU affine4/G64/BF16 and small CPU tensors, and shares
CPU numerical/state transactions through static dispatch. One/context17 pass
64.802M independent comparisons combined, maxratio0.447; peak3.425GB. Required
631CPU/765Metal, fmt/both strict Clippy, native54/Csmokes,18Python/20coupled+6performance
evidence controls and tiny GPU recovery pass. Coupled forward medians fall97.6–97.8%
(138.03s→3.31s/138.02s→3.10s); matched physicalpeak5.307GB. Explicit GPU poison repair admits overlap
and preserves committed prefix. No codec/full-GPU/request/adoption qualification.
[Scope](qwen35-metal-executor.md),
[checks/review/commit](../artifacts/qwen35-metal-executor-2026-11-08/publication-checks.json).

Q1.H bounded text codec passes100source records/165stream schedules against two
frozen witnesses; exact whitespace, NFC/ByteLevel, flags and bothEOS IDs preserved.
Controlled638CPU/772Metal, fmt/bothClippy and native56/Csmokes pass. Default13load
reconstruction rejects512MiB at686MB; explicit process-only `MallocLargeCache=0`
baselines pass at261MB, construction360/358ms. App/library defaults stay unchanged;
inheritedNano0 retry is child-isolated; regular opens are nonblocking/no-follow.
Astra approves scoped retention with no findings; no controller,
deeper-context, request or adoption qualification.
[Scope](qwen35-text-codec.md),
[investigation](../artifacts/qwen35-text-codec-2026-11-08/memory-investigation.json),
[checks/review/commit](../artifacts/qwen35-text-codec-2026-11-08/publication-checks.json).

Q1.I generation state bridge is in progress: pending sampled/input history stays
separate from committed hybrid state; exact whole-prefix reuse and conservative
edit/shortening replay feed the existing generation controller. Detailed memory
owners and last numerical work stay observable.644CPU/778Metal, strict Clippy/fmt,
native58/Csmokes and tiny GPU controls pass. Actual17 state/logits and failure/
repair/retry bits match, peak2.765GB. Two release baselines pass: generation
2.985/2.950s versus direct2.968/2.961s, no speed/overhead claim;8Python/36resealed
evidence controls pass. Astra approves scoped retention with no findings; no managed
route or quality qualification.
[Checks/review/commit](../artifacts/qwen35-generation-state-2026-11-09/publication-checks.json).
[Scope](qwen35-generation-state.md),
[contract](../artifacts/qwen35-generation-state-2026-11-09/acceptance-contract.json).

Q1.J's closed128/256-input evaluation rejects numerical qualification:
13,505failures/76.98M comparisons, maxratio34.066, with matching greedy IDs.
The two original references have3,132disjoint checkpoint tolerance intervals;
no threshold or profile requirement changes. Native replay/recovery passes at
2.765GB; the reference peaks at810MB. Required644CPU/778Metal, fmt/bothClippy,
26Python/24native/30reference evidence controls pass; native58 reuse is audited.
Astra approves scoped retention with no findings. Source/storage failures are
retained; no full-request,
quality, managed route or context expansion follows.
[Scope](qwen35-context256.md),
[decision](../artifacts/qwen35-context256-2026-11-09/decision.json),
[checks/review/commit](../artifacts/qwen35-context256-2026-11-09/publication-checks.json).

The separate [reference-step prototype](../artifacts/qwen35-context256-2026-11-09/recurrent-vector-experiment-summary.json)
now matches all38.49M originalFP32 values/2,102arrays bitwise, after tiny/real17
checks. Eight unit/32actual evidence controls pass at722MBpeak. Isolated recurrence
cohorts improve, but full collection timings are not comparable and no model
speedup or collector promotion follows; original source and failed controls remain.
[Full parity](../artifacts/qwen35-context256-2026-11-09/reference-graph-vector-experiment/full256-parity.json).

[Persistent-codec cache stress](../artifacts/qwen35-context256-2026-11-09/codec-cache-stress-2/validated-final.json)
passes four owned processes and15resealed evidence controls on22,000 distinct
near-limit words. Singleton peaks are289MB default/260MB controlled, within512MiB.
Whole50KB decode rejects the existing conservative bound; ASCII chunks verify
roundtrips. Astra approves this additional singleton scope; prior repeated-
construction rejection and aggregate fit restrictions remain.

[MiniCPM5 metadata preflight](../artifacts/minicpm5-preflight-2026-11-09/preflight-summary.json)
confirms the pinned973-tensor header/index and2.517B logical count. The declared
modified BF16 parent's current API access returns401; its lineage and every
payload/runtime/quality gate stay open. FullFP32 expansion exceeds6GiB; no source
payload was downloaded and selected implementation order is unchanged.
The separate [LFM3 BF16 preflight](../artifacts/lfm3-preflight-2026-11-09/preflight-summary.json)
identifies a3.458GB uncensored candidate with pinned BF16-parent metadata and a
0.973GB official Q4 reference. Header geometry and manifest hash links pass;
source bytes, new graph/codec, numerical/quality/fit and adoption remain open.
The earlier over-budget GGUF-to-F32 artifact stays rejected.
The [Granite/Ministral/Qwen2B audit](../artifacts/small-model-metadata-2026-11-09/preflight-summary.json)
passes53metadata/12header checks with explicit stale-index/declared-size
rejections retained. It identifies config/YaRN and Qwen norm-precision work;
no payload authentication, runtime qualification or policy/cap change follows.
The [9B source](../artifacts/qwen35-9b-payload-2026-11-09/verified.json) now passes
independent5.038GB byte/header/config/index verification after guarded cache
recovery. No9B parser, runtime/fit/quality or route claim follows; implementation
order and6GiB stay fixed.

Q2.A's [pinned9B cold recipe](qwen35-9b-cold.md) is retained with Astra approval. It represents
the distinct embedding/output-head owners,250affine matrices,153BF16 vectors
and24convolutions, with full metadata shape/dtype/span checks. Strict4B APIs
remain separate; no tensor payload, numerical backend or model/device admission
is added. Required652CPU/786Metal, fmt/both strictClippy, two actual4B loads/
independent924component/747row checks and corruption rejection pass. Two9B
metadata cohorts give2.488/2.554ms; all14cohorts include the noisy4B comparison
and four confirmation pairs. No request gain/overhead or model-fit claim.
Fresh native59 three slices/46exports each and Mac CPU/Metal Csmokes pass;
Astra approves scoped retention with no findings.
[Contract](../artifacts/qwen35-9b-cold-2026-11-09/acceptance-contract.json),
[checks/review/commit](../artifacts/qwen35-9b-cold-2026-11-09/publication-checks.json).

Q2.B's separate [9B CPU payload store](qwen35-9b-storage.md) is retained with Astra approval.
It authenticates the full stream before publishing original affine owners and
exact promoted small values, preserving strict4B APIs. Required661CPU/795Metal,
fmt/both strictClippy,17Python/native60/Csmokes and actual4B/9B byte/row/corruption
checks pass. Independent9B verification covers927components/750rows/3.858Mvalues
per capture at5.053GBpeak;8actual evidence controls pass. Astra's reader timing P2
is fixed with explicit consumer-only reuse; no remaining findings. Six loading cohorts/
60samples pass:9B22.731/22.675s; matched4B−1.02%/−0.18%, with no repeated5%
regression. Astra approves scoped retention; no model/GPU/codec/request/quality/adoption
qualification follows, and OS cache residency is not inferred from warmups.

[Q2.B checks/review/commit](../artifacts/qwen35-9b-storage-2026-11-09/publication-checks.json).

Q2.C's [explicit untied numerical head](qwen35-untied-head.md) preserves tied4B
interfaces and adds a diagnostic9B CPU factory. Required667CPU/801Metal,
fmt/both strictClippy,3Python, independent tiny CPU/Metal graph/state and
native61/Csmokes pass. Both old4B capture scopes are bitwise unchanged;60release
samples show4B+0.13%/−2.43% and9BCPU15.480/15.546s at5.31GBpeak. No speedup,
full9B fidelity or managed adoption claim; deeper4B rejection stays binding.
Astra approves scoped retention with no findings;
[checks/review/commit](../artifacts/qwen35-9b-head-2026-11-09/publication-checks.json).
