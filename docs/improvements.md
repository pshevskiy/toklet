# Toklet improvement record

Updated 14 October 2026. This records implemented capabilities, measured benefits,
and the decisions that limit their use. Detailed reports retain exact revisions,
artifact hashes, raw samples, checks and Astra reviews. M0–M4 are functionally
complete; M5.1, M5.3, M5.4 and M5.5 are complete in their recorded scopes. M5 remains
in progress. CPU/scalar remains the default. Speedups below
apply to the named workload and comparison, and cannot be added together.

## Measured performance improvements

Measurements use Apple M1 Pro, 16 GiB, macOS 26.6.2 and optimized release builds.
Each linked report specifies its power observations and timing boundary.

| Change | Measured benefit | Decision and evidence |
| --- | --- | --- |
| Ordered FP64 CPU projection tile | Warm TTFT for 128 input / 128 fixed output tokens improves **77.6400% / 77.6009%** in two confirming pairs; complete request time improves **52.0024% / 52.0992%**. | Qualified opt-in `--cpu-projection tiled`; scalar reference/default preserved. Full loss, 100 tasks, memory, eight workloads and application checks pass. [Retention report](experiments/2026-10-12-cpu-tiled-retention.md); commit `0abbe60`. |
| Longest exact committed-prefix reuse | On Q4 0.5B, editing position 128 of a 512-token prompt reduces first-token latency **21.586% / 21.785%**; editing only position 511 reduces it **92.983% / 93.010%**. | Retained CPU/Metal session mechanism; no KV copying. Two paired runs, identical final distributions and fixed buffers. This does not qualify Q4 model quality. [Continuation report](experiments/2026-10-13-prefix-continuation.md); `340c965`. |
| Original eight-row Q4 prompt tile | 512-input / one-output engine TTFT improves **35.1% / 35.2%** against Original Q4 projection. | Experimental `Q4Tiled`, not default. Refinement passes, but both paths fail the expanded independent offset decoder gate. [Experiment](experiments/2026-10-13-q4-prefill.md); `cc05939`. |
| Accurate eight-row Q4 prompt tile | Larger-model 512-input / one-output engine TTFT improves **25.4128% / 25.6841%** in two paired runs, with identical final-logit bits. | Reviewed and retained as an explicit experimental component. Both fixed Q4 configurations remain quality-rejected. [Experiment](experiments/2026-10-13-accurate-q4-prefill.md); `aef86c9`. |

The CPU tile reuses contiguous weights/activations across four token rows and
four output features while preserving feature-order FP64 accumulation. The
following accepted workload medians document its broader effects. Only the
short row has the confirming paired primary experiment; other ratios compare
separate historical runs and are descriptive.

| CPU workload, 128 fixed outputs | Scalar request seconds | Tiled request seconds | Request reduction |
| --- | ---: | ---: | ---: |
| 128 input | 128.561790 | 61.706524 | 52.0024% |
| 512 input | 253.195493 | 97.892741 | 61.3371% |
| 1,024 input | 428.750184 | 142.368874 | 66.7944% |
| 1,920 input, capacity boundary | 749.021123 | 246.854774 | 67.0430% |
| 512 cached + 64 appended | 111.628890 | 63.166372 | 43.4140% |
| Fully cached 512-token prompt | 90.000207 | 54.425495 | 39.5274% |
| Cached 512 + one pending token | 90.500725 | 54.797608 | 39.4506% |
| Changed 512-token prefix | 253.054154 | 93.038050 | 63.2339% |
| Three-turn application scenario | 263.230241 | 129.141379 | 50.9398% |

[Full retention evidence](experiments/2026-10-12-cpu-tiled-retention.md) links
every workload. All application token IDs, text hashes, stops and states match.
Several uninterrupted sequences exceed ten minutes. Near-floor fully cached
first-token timings have overlapping ranges; they do not support a percentage
ranking. No external-runtime speed lead follows from these internal comparisons.

## Correctness and usable engine capabilities

| Implemented change | Benefit and qualification | Evidence / implementation commit |
| --- | --- | --- |
| Checked scalar CPU kernels | Linear/bias/embedding/residual operations, stable FP64 reductions, RMSNorm, softmax, SiLU and gated MLP; shape/range checks and independent fixtures. | [Reference documentation](../tools/reference/README.md); M1 commits `c28df92`, `eea790e`. |
| Complete deterministic decoder | Rotary pairing/absolute offsets, causal masking, GQA head mapping, two-layer tiny decoder, tied output weights and deterministic greedy ties. | [M1 plan](../PLAN.md#m1-build-the-tiny-fp32-decoder); `5c413fd`, `41af02f`. |
| Validated real-model loading | FP32/FP16/BF16 and indexed safetensors shards; names/shapes/config/tied weights checked before use; tensor-wise conversion bounds temporary memory. | [Qwen loading](../benchmarks/results/2026-10-09-qwen-load.md); `8b9d892`. |
| Independent Qwen forward parity | Full short numerical arrays against pinned Transformers, rather than judging generated prose; source/config/tokenizer identities frozen. | [Qwen forward](../benchmarks/results/2026-10-09-qwen-forward.md); `66630c5`. |
| Bounded transactional CPU sessions | Persistent separate K/V, reusable scratch, absolute positions, exact append/full-cache reuse, capacity admission and replay. Sampled history is separate from committed KV. | [Session qualification](../benchmarks/results/2026-10-09-qwen-session.md); `5f58a6e`. |
| Streaming CLI and interactive sessions | Qualified tokenizer/chat template, stop/EOS, pending Unicode, append/continue/reset, bounded JSON input, Ctrl-C and recoverable request errors; generation statistics. | [CLI qualification](../benchmarks/results/2026-10-10-qwen-cli.md); `dd3f5e6`. |
| Root FP16 Metal decoder | Complete tiny and pinned Qwen forward/generation: MPS projections, embedding, RMS, rotary, separate KV writes, causal/cached GQA, MLP, residuals and tied logits. | [Resident engine](experiments/2026-10-13-metal-engine.md); `876a5f7`. |
| Metal failure and cancellation state | Commit positions only after a successful full chunk; finite overflow/status errors preserve the valid prefix, completion failure requires replay, invalid inputs fail before mutation. | [Resident engine](experiments/2026-10-13-metal-engine.md) and [continuation](experiments/2026-10-13-prefix-continuation.md). |
| Exact-prefix edits and shortening | Reuse the longest committed match; reconstruct shortened cached distributions by replaying the final required token; preserve recovery provenance across repeated cleanup. | [Continuation report](experiments/2026-10-13-prefix-continuation.md); `340c965`. |
| Larger exact model geometry | Admits the preselected 28-layer, hidden-1536 Qwen2.5-1.5B checkpoint; larger CPU and Half/Q4 numerical references verified independently within 6 GiB. | [CPU oracle](experiments/2026-10-13-larger-independent-oracle.md), [GPU geometry](experiments/2026-10-13-q4-compensated-storage.md), [passing Q4 numerical qualification](experiments/2026-10-13-compensated-rms.md); `58dc39a`, `dee4bc3`, `8e76c3d`. |
| Compensated Q4 projection | Explicit FMA residual plus two-component accumulation and round-to-odd Half storage fixes a localized double-rounding error without changing Q4 reconstruction. Expanded tiny gates and signed midpoint cases pass. | [Numerical investigation](experiments/2026-10-13-q4-compensated-storage.md); `dee4bc3`. |
| Compensated RMS reduction | Compensated summation of exactly represented Half squares isolates and fixes 66 larger first-layer rounding differences. Passes 43,008 exact primitive values, 4,550 tiny stage comparisons and 969 selected larger comparisons under unchanged gates. | [RMS qualification](experiments/2026-10-13-compensated-rms.md); `8e76c3d`. |
| Explicit compensated resident Half projection | Shape-independent per-row accumulation, bias last, midpoint-preserving Half storage and nearest FP32 vocabulary output pass1,668 primitive outputs,2,520 tiny captured arrays and969 selected larger arrays. | [Numerical qualification](experiments/2026-10-14-half-compensated.md), [short consistency](experiments/2026-10-14-half-scorer.md) and [full loss](experiments/2026-10-14-half-full-loss.md) pass. [Full task quality](experiments/2026-10-14-half-task-quality.md) now passes both checkpoints; [release decision](experiments/2026-10-14-m5-performance-decision.md) withholds replacement. |
| Explicit compensated Half CLI | Opt-in mode, actual ready/done execution metadata and complete logical admission; six bounded native workers preserve cached/pending/error/reset/stop/cancellation and independently finalized raw/chat text. | [CLI qualification](experiments/2026-10-14-metal-cli-half.md); later [full qualification and decision](experiments/2026-10-14-m5-performance-decision.md) record quality, workloads and repeatability; default/replacement remains unaccepted. |
| Frozen controlled Metal worker/collector | Completed waits, exact consumed token-byte pins and full-history/state/memory checks preserve eight contract workloads on both fixed checkpoints, including explicit changed-prefix replay and the2048-position boundary. | [Functional qualification](experiments/2026-10-14-metal-controlled.md);16workers/62events pass. both [full matrices and repeats](experiments/2026-10-14-m5-performance-decision.md) are now complete. |
| Bounded larger FP32 task-reference worker | Raw prepared IDs, capacity562/chunk8 and post-exit tokenizer construction fit all100 original admissions without a simultaneous tokenizer/model allocation. Synthetic maximum/EOS/state checks and three real short probes pass. | [Reference prerequisite](experiments/2026-10-14-cpu-task-reference.md); [full100responses/literal scores](experiments/2026-10-14-half-task-quality.md) now complete; not an application baseline. |
| Backend-aware Half task collection | Actual Metal execution/memory admission, exact frozen case grammar and independent finalized-text comparison; qualified receipt replacement rejects before launch. Six real short probes/55events pass. | [Collector qualification](experiments/2026-10-14-metal-tasks.md); [full100responses and literal grading](experiments/2026-10-14-half-task-quality.md) now complete. |

Compensated projection/normalization are explicit options. Their numerical
qualification does not establish acceptable end-to-end quantization quality.

## Memory, traffic and scheduling improvements

| Change | Concrete effect | Scope and evidence |
| --- | --- | --- |
| Resident immutable weights and session-owned KV/scratch | Load once; preallocate admitted buffers/views/pipelines; reuse them at steady state. CPU and Metal numerical decode have zero measured Rust allocations after warmup. | [CPU sessions](../benchmarks/results/2026-10-09-qwen-session.md), [Metal engine](experiments/2026-10-13-metal-engine.md). Native frameworks and application allocations are separately scoped. |
| Streamed Half loading | No redundant full FP32 model retained with GPU weights; bounded conversion stripes and a single tied embedding/output allocation. | [Metal engine](experiments/2026-10-13-metal-engine.md); `876a5f7`. |
| One completed command per Metal chunk | Dependent operations encoded without a host wait per operator/layer; resources remain owned until completion. Real GPU captures and calibrated counters validate scheduling. | [Trace report](experiments/2026-10-13-metal-trace.md); `ede1d45`. No isolated before/after speed claim. |
| Versioned source-bound Q4 artifacts | Group-64 affine Q4 format, explicit bit/group layout and oracle; provenance, overflow, padding, source replacement and full load/session admission checked. | [Format](q4-v1.md), [artifact qualification](experiments/2026-10-13-q4-artifact.md); `6ec620b`, `a0263f9`. |
| Fused Q4 reconstruction/projection | Dequantizes within the dot product; no full-model or per-token expanded weights. 0.5B resident weights are **476,307,372 bytes**, versus Half **988,069,596** and FP32 **1,976,131,072**. | [Execution](experiments/2026-10-13-q4-decode.md); `b99d334`. Weight payload only; quality rejected. Larger Q4 payload is **1,223,992,516 bytes**. |
| Required-position vocabulary projection | Ordinary prefill computes only its final-row logits, using a precreated checked view; diagnostic paths retain all rows. | [Qualification](experiments/2026-10-13-required-position-logits.md); `d93264e`. Float scratch is unchanged; no measured latency/memory gain claimed. |
| Explicit GPU greedy token readback | Exact finite FP32 ordering and lowest-ID ties select in the completed command; eight cached-ID bytes replace two host distributions, with a 32-byte guarded GPU buffer. Larger-session admission falls **1,215,448 bytes**. | [Selection task](experiments/2026-10-13-metal-greedy.md). Full logits remain default; diagnostics preserve full arrays. No quality or speed acceptance. |
| Explicit Metal chunks through 1,024 | Checked admission and a shared-model four-chunk harness pass exact capacity/budget boundaries and larger offset/append/edit final-logit bit refinement. | [Chunk task](experiments/2026-10-13-metal-chunks.md). Default stays 128; this does not qualify every mode or full release workload. |
| Scoped M5.4 evaluation closure | All four named mechanisms have checked, Astra-reviewed and committed decisions. Only explicit functional token-ID readback remains; all three fusion prototypes were removed. | [Synthesis](experiments/2026-10-14-m5-dispatch-closure.md). No retained fusion latency gain or release/default acceptance. |
| Scoped M5.3 prefill closure | Required-position logits, independently qualified accurate Q4 prompt tile and completed chunk sweep satisfy the named prefill mechanisms; default chunk stays 128. | [Synthesis](experiments/2026-10-13-m5-prefill-closure.md). Experimental compensated path only; both Q4 quality failures and M5 release gates remain open. |
| CPU caller-owned all-row logits | Bounded external chunk output supports teacher-forced scoring without full-prefix traces or duplicate weights; normal final-row execution remains unchanged. | [All-row logits](experiments/2026-10-13-cpu-all-row-logits.md); `3dd5485`. |
| Streamed independent storage oracles | Cache one Half tied table, stream current tensors/Q4 stripes, pin exact bytes before parsing, publish only after completed native memory qualification. | [Oracle report](experiments/2026-10-13-streamed-storage-oracles.md); `962f206`. Evaluation memory benefit, not inference latency. |

## Measurement and verification improvements

| Change | What it establishes | Evidence |
| --- | --- | --- |
| Offline Rust package, CI and frozen fixtures | CPU tests need neither model downloads nor Apple GPU; locked dependencies, formatting and strict Clippy, independent numerical data and causal/session regressions. | [Plan M0](../PLAN.md#m0-establish-the-project-and-measurement-contract), [reference](../tools/reference/README.md). |
| Frozen benchmark/quality contracts | Named input/output/cache workloads, artifact/token hashes, raw completed-work timings, three warmups/ten repetitions and second-run retention policy. | [Benchmark contract](../benchmarks/README.md). |
| Internal and external comparison adapters | Separate engine/application timing, teacher-forced and practical representation tracks, native MLX and llama.cpp evidence without combining incompatible precision/cache rows. | [External matrix](../benchmarks/results/2026-10-10-external-matrix.md), [scalar closure](experiments/2026-10-11-m2.5-scalar-comparison.md). |
| Owned bounded experiment transport | Exclusive evidence files, bounded nonblocking pipes/timeouts, owned process groups, cleanup/reaping and endpoint identity checks; failed/paused/overlapped attempts remain excluded. | [Continuation collector](experiments/2026-10-13-prefix-continuation.md), [loss collector](experiments/2026-10-13-larger-metal-loss.md). |
| Native GPU trace and memory accounting | Paired CPU/GPU clocks, completed stage counters, diagnostic profiling separated from production; explicit allocated/physical/app/transient accounting under 6 GiB. | [Trace](experiments/2026-10-13-metal-trace.md), [larger admission](experiments/2026-10-13-cpu-row-loss.md). |
| Frozen 10,000-target loss and 100-task grading | Causal target/block coverage, stable loss, all raw responses and literal grading; no generated-code repair. CPU scalar/tiled NLL and task results are identical. | [Loss](experiments/2026-10-11-held-out-tiled-loss.md), [tasks](experiments/2026-10-11-held-out-task-quality.md), [isolated grader](experiments/2026-10-11-linux-literal-controller.md). |
| Matched larger references | Exact preselection before quality evaluation; larger FP32 mean NLL **2.485958031019153**, independent geometry/oracles and bounded native residency. | [Preselection](experiments/2026-10-13-larger-preselection.md), [FP32 collection](experiments/2026-10-13-larger-fp32-loss.md); `4f226ef`, `095dd91`. |
| All-layer Half consistency localization | Reproduces all six scored rows, captures451 stage pairs/56 complete KV prefixes, and replays the first divergent projection with identical operands and exact dyadic rounding witnesses. | [Investigation](experiments/2026-10-14-half-consistency.md). Layer0 Q MPS shape-dependent rounding is localized; no arithmetic, quality or speed acceptance. |
| Explicit compensated Half scorer | Both preselected checkpoints pass the unchanged short consistency guard with zero NLL difference and exact scored-row logits/target bits across chunk1/128. | [Scorer task](experiments/2026-10-14-half-scorer.md).12 scored observations per model; [full loss now passes](experiments/2026-10-14-half-full-loss.md), [full task quality](experiments/2026-10-14-half-task-quality.md) now passes; release replacement remains unaccepted. |
| Compensated Half full held-out loss | All10,000 targets per fixed model pass the unchanged0.05 NLL-increase gate:0.5B mean2.8836406 vs FP32 2.8837251;1.5B mean2.4859246 vs FP32 2.4859580. | [Full-loss qualification](experiments/2026-10-14-half-full-loss.md). Corpus-specific quality gate; [full task quality and repeated measurements](experiments/2026-10-14-m5-performance-decision.md) now complete. Overall M5 remains open. |
| Astra code/evidence review per task | Review findings addressed before task commits; receipts bind sources, binaries, fixtures, checks and saved results. | Each linked task report records its review and acceptance scope. |

Full task grading admission now preserves actual backend/model/modes and requires
all100 original cases/caps, complete native replay, owned successful exit, full
memory and independent finalized text. Exact backend inventories and executable
hashes anchored in pinned qualification receipts reject omitted dependencies and
replacement; nine targeted/all410 Python checks pass. Published short probes cannot
be scored. This is admission only; no full task score or VM execution is claimed.
[Admission report](experiments/2026-10-14-task-grading-admission.md).

A separate backend-aware literal grading window now connects full qualified task
collections to the unchanged executor. It preserves literal programs and unresolved
outcomes, requires all three matched models to exit, checks fresh PID/group absence
immediately before VM startup and owns stopped restoration.22 mocked/all432 Python
checks pass; subsequent [full native grading](experiments/2026-10-14-half-task-quality.md) resolves all300 outcomes.
[Window report](experiments/2026-10-14-task-grading-window.md).

The frozen authored application trace now supports explicit compensated Half Metal
on both selected checkpoints. Exact36/76/117 prompt IDs, scripted history/reset,
actual memory/modes and first-token/nonempty-text/done/finalization boundaries pass
seven targeted/all439 Python checks and six short native turns/26event replay.
Both model-plus-driver charges fit6GiB; short probes establish functionality only.
[Application report](experiments/2026-10-14-metal-application.md).

Full task response collection now covers all100 original cases for both Half
checkpoints and the larger FP32 reference.18,027 native events, full provenance,
state/memory and independent finalized text pass; all three workers exit cleanly
and fit6GiB. Larger visible IDs match99/100 cases. This is response qualification;
subsequent [matched literal scores](experiments/2026-10-14-half-task-quality.md) pass; latency replacement remains unaccepted.
[Full response report](experiments/2026-10-14-full-task-responses.md).

Full matched Half task quality now passes both checkpoints:2/100 versus2/100 small
FP32 and11/100 versus11/100 larger FP32, with0pp loss under the unchanged2pp gate.
Three native literal windows resolve300 outcomes/120 code programs, restore the VM
stopped and replay exactly. Absolute code passes are0/40 per model, without repair.
Together with full Half loss this closes frozen Half quality; later repeated
measurements and sustained scopes are complete, while the Q4-blocked M5 exit stays open.
[Matched quality report](experiments/2026-10-14-half-task-quality.md).

Full small-model compensated Half controlled measurements now complete all eight
workloads with three warmups and ten samples,187raw events and clean owned exits.
Median shortTTFT/request are0.484318/12.192120seconds. The ten boundary requests
total1096.033071seconds qualify sustained-duration scope with recorded AC/thermal
endpoints; continuous temperature, energy and system pressure remain unavailable.
Maximum admission1,262,602,960/samplednativejointcharge1,166,575,656bytes fit6GiB.
Independent raw/state/timing/memory replay and six evidence-mutation checks pass.
Astra cleared the exact replay source after provenance/phase/memory/byte-binding
fixes; completed evidence review is recorded in the [full data report](experiments/2026-10-14-small-half-controlled-full.md).
An initial account usage-limit failure in automatic approval review blocked post-run
inspection. The user's retry restores access: fresh observations find no controlled
worker/collector commands and a stopped/not-starting machine. Astra accepts the
completed small-model evidence before commit; historical PID/group IDs were not retained.

Full larger-model compensated Half measurements complete the same eight workloads/
24warmups/80samples and187raw events. Median shortTTFT/request2.037336/28.398590s;
boundaryrequest264.684586s. Seven rows exceed ten measured minutes, with recorded
AC/thermal endpoints. Admission3,396,602,720/nativejointcharge3,301,821,896bytes fit
6GiB. Independent replay/sixmutation checks and recorded9PID/eightworker-group idle
proof pass. Both full matrices now complete; see the [larger report](experiments/2026-10-14-larger-half-controlled-full.md).
Astra accepts completed larger-model evidence before commit; no speed/default/Q4 adoption.

Full compensated Half application traces complete both fixed models with3warmups/
10three-turn scenarios,78total turns and60measured. Medians20.445239/43.550813s;
admissions1,443,266,720/3,577,266,480/nativejointcharges1,331,447,848/3,467,005,408
bytes fit6GiB. Independent finalized text/state/receipt-boundary/memory replay passes.
Astra found audit timestamp/nullability gaps; corrected source and15invalidvariant
checks pass. Small second-turn14tokens differs fromFP3218, and neither measured
scenario sum reaches600s; no purebackend speed or extra10min claim. See the
[full application report](experiments/2026-10-14-half-application-full.md). Final
Astra accepts completed application evidence before commit; confirming/release/Q4 gates remain open.

Independent fixed Half primary repeats now retain secondthreewarmup/tensample
short128/128collections on bothmodels.26requests/30events replay; TTFTmedian
changes+0.003433%/−0.168474% and requestchanges+0.087202%/+0.005098%. These are
repeatability, not a new paired optimization gain. Complete states/memory/inputs/
endpoints and clean exits pass; caller source/fivefailurechecks cleared. See the
[primary confirmation report](experiments/2026-10-14-half-primary-confirm.md); final
Astra accepts final data before commit, applicationconfirmation/release/Q4 gates open.

Independent full Half application repeats now match all39turn response/state
signatures permodel to the initial runs, with medians20.456533/43.572448s and
changes+0.055240%/+0.049678%.78turns/60measured/7165events, memory/endpoints/
owned-exit/idle proofs pass. This is repeatability, retaining differentFP32outputs,
sub600s/application-duration and unavailableend-power/sensor/timing limits. See
[application confirmation](experiments/2026-10-14-half-application-confirm.md). Final
Astra accepts final data before commit; finalrelease/Q4 gates remain open.

The final [M5 evidence and decision report](experiments/2026-10-14-m5-performance-decision.md)
consolidates all completed Half quality, eight-row matrices, sustained scopes and
confirming runs. It retains explicit compensated Half and records all Q4/fusion/
chunk/readback decisions without changing defaults. Historical CPU FP32 and
external FP32/BF16 are different representation tracks; small Half boundary decode
is slower than historical CPU, and application token counts differ. No matched
external ranking or performance replacement follows. M5.2/.6 remain open;
record completion is separate from milestone acceptance. Saved arithmetic,
links/whitespace and protected executable checks pass; Astra accepted after two
wording corrections with no remaining findings.

A separate [Q4 v2 group8 candidate](q4-v2.md) now has a checked [source-bound
packer and independent CPU reconstruction oracle](experiments/2026-10-14-q4-v2-format.md).
It uses8effectivebits/weight including Half scale/bias before padding, with distinct
version/quantizer identities and immutable admitted format. V1 default/bytes and
rejected quality remain. Manual cross-language fixtures, group/alignment tails,
source replacement, output ownership, overflow and version mixing pass. V2 GPU
loading rejects before allocation until qualified execution is implemented.
No v2 quality or latency gain is claimed; Astra accepted design, code, checks
and evidence with no findings before commit.

The [Q4 v2 GPU task](experiments/2026-10-14-q4-v2-gpu.md) adds two static
compensated kernels and immutable format dispatch without weight expansion.
Independent tiny graph9100stage comparisons,4032exactprimitiveoutputs and16
midpoint/overflow witnesses pass; both session modes preserve cancellation,
pending/cache/failure/admission and zero warmed Rust allocations. Existing v1
independent regression4550arrays passes. The oracle now rejects missing/duplicate
inventories and file replacement using hashes of consumed bytes and endpoint checks.
This qualifies tiny execution only; real-model quality/memory/performance stay open.
Astra accepted code, checks and native evidence after oracle fixes, with no findings.

## Rejected or inconclusive experiments

These findings prevent unsuccessful experiments from being presented as improvements.

| Experiment | Outcome and current constraint |
| --- | --- |
| Core ML / ANE supported slices | CPU slice/state behavior qualified; actual ANE acceleration unobserved. FP16 projection fails its frozen gate under both policies. No ANE backend adopted. [Closure](experiments/2026-10-11-ane-feasibility-closure.md). |
| Dynamically indexed Original Q4 tile | **28.5% slower**; rejected in favor of the bounded fixed-index experiment. [Report](experiments/2026-10-13-q4-prefill.md). |
| Residual/post-attention RMS fusion | All numerical/state checks pass, but complete 512/128 Fused requests are **0.553034% slower**. Prototype removed; Split retained. [Negative experiment](experiments/2026-10-14-residual-rms.md). |
| Rotary/cache-write fusion | All numerical/state checks pass, but complete 512/128 request gain is only **0.014593%**, below the 5% gate and Split dispersion. Prototype removed; Split retained. [Negative experiment](experiments/2026-10-14-rotary-cache.md). |
| MLP SiLU/gating fusion | Exact Half storage and all numerical/state checks pass, but complete512/128 request gain is only **0.006032%**, below dispersion and5%. Prototype removed; Split retained. [Negative experiment](experiments/2026-10-14-mlp-gating.md). |
| GPU greedy latency | Larger512-input/128-output median is **0.2890% slower** than full-logit CPU selection. Explicit ID-only capability retained for verified host storage/traffic; no performance default or speed acceptance. [Result](experiments/2026-10-13-metal-greedy.md). |
| Larger Q4 chunk-size default changes | 256/512 improve the named 1,024-input/one-output median only **4.0506% / 2.6294%**, below the frozen 5% threshold; 1,024 is **8.4712% slower**. No larger default adopted. [Sweep](experiments/2026-10-13-metal-chunks.md). |
| Original and Original-tiled expanded Q4 tiny decoder | Both fail the frozen offset17 independent gate. The compensated candidate passes; no tolerance was loosened. [Original report](experiments/2026-10-13-q4-prefill.md), [compensated report](experiments/2026-10-13-q4-compensated-storage.md). |
| 0.5B Q4 held-out quality | Mean NLL **3.057166039390558** vs FP32 **2.883725149929484**; increase **0.173441 > 0.05**, rejected. [Result](experiments/2026-10-13-metal-held-out-loss.md). |
| 0.5B Original Half full-loss collection | Short batch/single NLL guard **0.004681 > 0.001** fails; collection stops before full loss. No full Half quality conclusion. [Result](experiments/2026-10-13-metal-held-out-loss.md). |
| Larger Original Half full-loss collection | Short batch/single target NLL guard **0.001523869 > 0.001** fails at the first probe. Full scoring never starts; no full Half quality conclusion. [Result](experiments/2026-10-14-larger-half-loss.md). |
| Larger Q4 held-out quality | Mean NLL **2.588645591385506** vs matched FP32 **2.485958031019153**; increase **0.10268756036635285 > 0.05**, rejected despite short numerical parity. [Result](experiments/2026-10-13-larger-metal-loss.md); `a082be5`. |
| Native macOS generated-code isolation prototype | Fixed resource admission fails; not adopted. Literal grading instead uses the separately qualified Linux controller. [Native rejection](experiments/2026-10-10-code-isolation.md), [Linux qualification](experiments/2026-10-11-linux-literal-controller.md). |

The fixed quality gates, quantizer, checkpoint selection and corpus remain
unchanged. Full compensated Half loss/task quality passes both selected models.
Both confirming primary/application runs are complete. The [final decision](experiments/2026-10-14-m5-performance-decision.md)
records performance baseline gaps, withholds default replacement and keeps the Q4-blocked M5 exit open.
Long-context attention/KV experiments and physical iPhone work belong to M6/M7.
This record will be updated with each reviewed implementation or measurement
decision; unmeasured changes will remain explicitly marked.
