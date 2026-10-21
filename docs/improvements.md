# Toklet improvement record

Updated 19 October 2026. This records implemented capabilities, measured benefits,
and the decisions that limit their use. Detailed reports retain exact revisions,
artifact hashes, raw samples, checks and Astra reviews. M0–M6 are complete in their
qualified scopes: M5 selects the small compensated Half and frozen larger group8
Q4 v2 configurations with full quality, repeated workload/application, sustained
and memory evidence. M6 adds explicit Online attention, qualified4K/8K/16K/32K
context cases and final FP16 KV retention; M7 remains pending. CPU/scalar and the existing GPU/readback/
chunk defaults stay unchanged. Speedups below apply to the named workload and
comparison, and cannot be added together. See the [M5 release decision](experiments/2026-10-15-m5-release.md).

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

Both [Q4 v2 artifacts and real-model references](experiments/2026-10-14-q4-v2-real-oracles.md)
now match the frozen layout on the two pinned checkpoints. Streaming conversion
uses≤17,253,048bytes recorded native charge. Sixcases/342fullarrays permodel freeze
independent reconstructed-math gates beforeGPU/heldout, with new17token/offset
shape probes and explicit unavailable-source metadata. Reference admissions/
complete native charges fit6GiB. Explicit stripe temporary release closes a
memory-accounting overlap, verified with a lifetime regression.51Python checks
and portable Rust integrity pass; Astra accepted final evidence with no findings.
This is evaluation infrastructure/reference qualification, not inference speed
or model quality acceptance.

Both [real Q4 v2 GPU geometries](experiments/2026-10-14-q4-v2-real-geometry.md)
now pass all 14,892 full-element stage-slice comparisons across frozen vector/
tile modes, whole/irregular chunks, cached prefixes and offsets, with no greedy
mismatches. Measured native lifetime charges758,334,616/1,910,425,496 bytes and
admitted numerical+validation874,062,494/2,029,218,550 bytes fit6GiB. Capacity27
is graph qualification, not full-context/application/quality/speed acceptance.
Frozen held-out and repeated performance checks remain.

The [explicit Q4 v2 scorer](experiments/2026-10-14-q4-v2-scorer.md) now reports
actual format/quantizer/mode and resident admission, preserving legacy calls.
Both unchanged short batch/single loss guards pass before any full-corpus worker.
Exact-byte probe admission and both-model prelaunch tuple checks close Astra's
provenance findings;22portable and104ordinaryMetal-feature checks pass. This
qualifies integration/shortguards only; full loss/task/speed remain open.

The [frozen Q4 v2 full-loss evaluation](experiments/2026-10-14-q4-v2-full-loss.md)
completes all10,000targets for both models. Larger meanNLL2.51426472772287 versus
matchedFP322.485958031019153 increases0.02830669670371666, passing0.05.
Small increases0.05129570579114473 and remains rejected. All20,268events,
owned exits, state, unchanged endpoints and≤6GiB budgets replay cleanly.
This is larger loss acceptance only. Per the plan's explicit Half fallback,
continue larger task-quality/release evidence while preserving small rejection.

The [explicit v2 CLI](experiments/2026-10-14-q4-v2-cli.md) now requires a
paired artifact/mode selection, rejects v1 before events, and publishes actual
version/group/quantizer/dtype/modes while preserving legacy metadata/defaults.
Eight native workers/109events qualify tiny and both real-model cached/pending/
cancel/stop/text behavior within6GiB.95CPU/107Metal/13portable checks pass.
The preflight fixture failure is retained/excluded; transitive helper provenance
uses prior committed qualification/snapshots plus explicit endpoint proof.
CLI functionality does not repair small v2 rejection or qualify tasks/speed.

The [larger v2 task collector](experiments/2026-10-14-q4-v2-task-collector.md)
now admits the exact reviewed loss-qualified CLI/source/artifact and preserves
all frozen task/stop/reset/finalization rules. A prelaunch binary/source snapshot
race is closed with reasserted pins and zero-launch mutation tests.14portable
checks and the3category/24event native probe pass, with clean exit/post-exit
text replay and≤6GiB accounting. Full100tasks, grading and speed remain open.

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
| Small Q4 v2 group8 held-out loss | MeanNLL2.9350208557206288 vsFP322.883725149929484; increase0.05129570579114473 exceeds0.05. Rejected with no tuning/waiver. [Decision](experiments/2026-10-14-q4-v2-full-loss.md). |
| Native macOS generated-code isolation prototype | Fixed resource admission fails; not adopted. Literal grading instead uses the separately qualified Linux controller. [Native rejection](experiments/2026-10-10-code-isolation.md), [Linux qualification](experiments/2026-10-11-linux-literal-controller.md). |

The fixed quality gates, quantizer, checkpoint selection and corpus remain
unchanged. Full compensated Half loss/task quality passes both selected models.
Both confirming primary/application runs are complete. The [final decision](experiments/2026-10-14-m5-performance-decision.md)
records the historical performance baseline gaps and withheld default replacement. Its open-exit conclusion is superseded by the [15 October M5 release decision](experiments/2026-10-15-m5-release.md).
Long-context attention/KV experiments and physical iPhone work belong to M6/M7.
This record will be updated with each reviewed implementation or measurement
decision; unmeasured changes will remain explicitly marked.

The [larger v2 full response collection](experiments/2026-10-14-q4-v2-full-tasks.md) completes100original cases at256/32/16caps. Clean natural model exit,101ordered native-memory observations, exact raw/state/post-exit text and unchanged source/artifact endpoints qualify transport. Literal grading and task-quality/performance acceptance remain pending.

The [larger v2 full task gate](experiments/2026-10-14-q4-v2-task-quality.md) passes11/100 against matchedFP3211/100 with0pp precision loss. Exact v2 collection/inventory/pin admission and cold grading-window hooks preserve original literal/text semantics;43portable checks and the complete100-outcome/40code native window pass with zero unresolved infrastructure and stopped-VM restoration. Combined with its full loss pass, larger v2 quality is qualified; small v2 rejection and performance/release gaps remain.

The [v2 benchmark workers](experiments/2026-10-14-q4-v2-bench.md) publish actual format/group/quantizer/mode/residency and truthful result-schema representation while preserving legacy defaults/timing.23portable/95CPU/107Metal checks pass. Eight controlled probes/31events qualify fullcontext/state/limits;13paired512/1 requests preserve exact fullvocabulary bits/greedy/state. Complete memory/endpoints/owned cleanup pass. A confirmed useful gain and final release measurements remain pending.

The [v2 application adapter](experiments/2026-10-14-q4-v2-application.md) preserves the frozen scripted turns/formatting/policies and reports actualv2 execution.14portable checks cover quality/source/pin/parser admission, timers/state/stop/finalization and owned cleanup. The3turn/13event cap2nativeprobe passes exactpost-exit text with clean exit/unchanged endpoints/≤6GiB memory. Full application measurements and final release acceptance remain pending.

The [v2 affected-shape harness](experiments/2026-10-14-q4-v2-shapes.md) now measures128reset andactual64append after512cachedtokens, preserving the512/1primary. Bounded exact input sizes/SHAs, both consumed append arrays, actual cache/state metadata and fullbits qualify39pairs/45events.13portable/oneRustreader checks pass. Identicalsingle-token dispatch and unchanged otheroperations support scoped regression analysis. A label-only correction is recorded without rewritingevidence; overlapped preparatory512timings are entirelyexcluded. Freshprimaryconfirmation andfullrelease workloads remain pending.

The v2 controlled collector now accepts an explicit subset of frozen workloads for confirmation runs. Invalid, empty or duplicate selections fail before input admission or worker launch; the default still runs all eight rows. `selected_workloads` records the scope, and only a successful full canonical measurement sets `full_matrix_completed`. Fifteen portable controlled checks, formatting and strict Metal Clippy pass. This permits a separate short confirmation without repeating the full matrix; it changes no numerical worker, timing boundary, quality gate or default. Final measurements and M5 acceptance remain pending.

The [fresh v2 primary confirmation](experiments/2026-10-14-q4-v2-primary.md) completes two same-binary512/1 collections,26pairs/30events with exact full logits/greedy/state. Tile gains23.691277% and22.202400% exceed5%, nonoverlapping measured ranges and between-run median shifts1.867689%/3.855253%. Complete admission/native memory/endpoints/cleanup pass. This accepts the named prefill refinement only; full release matrix/application/sustained evidence remains. The contaminated preparatory512timings remain wholly excluded.

The [full larger v2 matrix and short confirmation](experiments/2026-10-15-q4-v2-controlled.md) complete117requests/202events at3warmup/10perrow. Exact cache/history/boundary2048/overflow preservation, source endpoints and≤6GiB admission/nativecharge pass. Completed matrix inference totals10,742.769647s includingwarmups (8,256.699921s measured-only), qualifying sustained duration without setup/hash gaps. Short medianTTFT/request change−3.349513%/−1.463285%; repeatability only. Full application/release acceptance remains pending.

The [larger v2 full application and confirmation](experiments/2026-10-15-q4-v2-application-full.md) complete78turns/6086events, with39exactID/text/stop/state signatures repeated acrossruns,3warmup/10measuredscenariosperrun. Scenario medians41.267054271/41.301725271s change+0.084016%; outputcounts86/128/14 preserveEOS/capsemantics. Naturalexit, independentpostexittext, completepins/parserinventory/clockbounds and≤6GiB memorypass. Applicationmeasureddurations412.726713/412.866698s are not sustainedqualification; controlledsequence suppliesit. Finalrelease synthesis remains.

The [15 October M5 release decision](experiments/2026-10-15-m5-release.md) supersedes the earlier open-exit conclusion: M5.1–M5.6 are complete. The selected small Half configuration and larger frozen group8 Q4 v2 configuration pass unchanged quality gates and full repeated/state/memory workloads; larger v2 confirms22.20–23.69% useful512/1prefill gains and the controlled sequence exceeds600s. Every optimization retains its measured scope, including slower cross-representation complete requests, rejected smallv2/v1, removed fusions, explicit GPUgreedy capability and unchanged defaults. Baseline gaps remain disclosed without externallead/defaultreplacement; M6/M7 remain pending.

M6 implementation has started. The [bounded online attention component](experiments/2026-10-15-m6-online-attention.md)
adds explicit separate decode/two-row prefill paths with16-key online softmax,
GQA sharing and no depth-sized score matrices. Independent FP64 primitive checks
cover202,048 outputs through32K depth;81 adversarial cases and173,184 existing
tiny-stage values pass unchanged gates, along with session/failure/readback and
zero warmed-allocation checks.95 CPU/107 ordinary Metal tests, formatting and
strict linting pass. Calculated allocated scratch falls by about28MiB small/24MiB larger
at capacity2048/chunk128 (exact byte accounting in the report). Retained as an
explicit component; real-model quality, performance, longer-context admission and
support, and the measured KV decision remain pending. Existing defaults stay.

The [longer-context admission component](experiments/2026-10-15-m6-long-admission.md)
permits explicit model-bounded Metal capacities through32K and adds
`--metal-attention reference|online`, preserving2K/Reference defaults and the CPU
session limit. Checked allocation/absolute positions/output reservation, exact
budgets and state preservation pass.97 CPU/110 ordinary Metal checks and six
native CLI workers/122 events pass;6912 independent high-position rotary values
pass their predeclared arithmetic envelope through32767. Structural/tiny evidence
is retained separately from the pending real-model long-prefix quality,
performance, memory and supported-depth qualification.

The [M6 Online quality collector](experiments/2026-10-15-m6-quality-collector.md)
binds actual Online selection, compiled source/executable, model/corpus/baseline
and packed payloads, reasserts them before worker launch, and preserves the
unchanged short/full loss gates and incomplete/rejected flags.26 portable checks,
97 CPU/110 ordinary Metal checks and eight real-model short-guard workers/68
events pass; single/batch NLL differences are0 small and at most0.00001075 larger.
Full10000-target loss/task quality and long-context release remain pending.

The [M6 Online full-loss collections](experiments/2026-10-15-m6-online-full-loss.md)
pass the unchanged10000-target gate for both selected representations: small
mean2.8835716762 (−0.0001534737 vsFP32), larger v2 mean2.5142232720
(+0.0282652410≤0.05). Ten serial workers/20268 raw events pass exact state,
source/artifact/corpus and complete≤6GiB memory checks. The block policy remains
near1K; task quality, long contexts, performance and M6 closure stay pending.

The [paired M6 context worker](experiments/2026-10-15-m6-context-worker.md)
shares immutable weights once, admits both session/counter resource sets before
allocation, and separates ordinary samples, paired vocabulary/loss checks and
same-depth decode profiles. Five portable protocol checks,97 CPU/110 ordinary
Metal checks and two native tiny workers/262 events pass exact cache/state,
rejection, memory and profiling checks. A byte-pinned4K–32K stress contract was
frozen before long real-model evaluation. Native collectors, long quality,
performance and the measured FP16 KV decision remain pending.

The [M6 context collector](experiments/2026-10-15-m6-context-collector.md)
binds reviewed compiled/runtime/model/contract bytes before launch and after
owned exit, validates exact paired quality/state/memory, and separates ordinary
timing from matching-command GPU profiles. The512/128 primary and three/ten
paired policy are explicit. Ten collector/seven controlled/seven transport tests
and two selected-model short probes/six events pass. Astra accepted all311
endpoint bindings and raw replay after three collector fixes. Long quality,
performance, application admission, task gates and the KV decision remain open.

The [M6 Online task adapter](experiments/2026-10-15-m6-tasks-collector.md)
preserves the original100-case caps/stops/literal rules, requires reviewed full
loss, binds actual Online execution and reasserts the complete qualified CLI
and consumed input inventory before launch. Separate full-collection admission
retains raw replay, natural exit, independent text and the existing isolated
grading window. Nine new/seven legacy/seven v2/seventeen window checks and two
native probes/55events pass. Astra accepted code/evidence; full100-task
collection and grading remain pending.

The [M6 maximum-context application adapter](experiments/2026-10-15-m6-application-collector.md)
checks the actual tokenizer-owning Online CLI's32K storage/admission, complete
published-ID/output-limit/pending semantics, reset/finalization and independent
text after owned exit. Separate decoder charge and honest derived-history-hash
scope are recorded. Four portable checks and two native full-capacity allocation
probes/tenevents pass, with admissions1,791,516,840/3,122,841,472bytes and sampled
joint native charges1,617,987,816/2,960,116,952bytes. Astra accepted code/evidence;
full32640+128 generation and M6 qualification remain pending.

The [full M6 Online task gate](experiments/2026-10-15-m6-online-task-quality.md)
passes for both selected representations:2/100 small and11/100 larger, matching
their FP32 baselines with0pp loss. Two serial collectors/12221events and two
isolated literal windows resolve all200outcomes, including80 unchanged
SyntaxError code failures and120text grades. Exact original caps/state/text,
complete≤6GiB admission/native charge, source/published archives and fresh stopped
VM restoration pass. Astra accepted all411bindings and independent replay;
absolute task scores stay disclosed. Full loss/task gates are now closed;
long quality, profiles/performance, max-context generation and the KV decision
remain before M6 completion.

The [M6 bounded-wait adapter](experiments/2026-10-15-m6-long-budget.md)
preserves completed collector identities while giving slow higher-depth
collections explicit 2/6/12-hour per-event read bounds under the existing 24-hour
worker lifetime. Four new/ten existing collector and seven transport checks pass;
two serial selected-model probes/six events validate 21,600/43,200 second settings,
exact state, endpoint identities and natural cleanup. No numerical gate or
performance/support claim changes; the incomplete 16K attempt is retained for
retry, and M6 long qualification remains open.
Astra accepted the final code and evidence after independent six-event replay and
313 binding checks.

The [M6 short-workload comparison](experiments/2026-10-15-m6-short-performance.md)
completes both eight-row matrices and independent 512/128 primary confirmations.
All four workers/300 events pass exact state, identity, native memory and cleanup
checks. Complete-request median reductions repeat at 76.26/76.24% small and
73.02/73.03% larger, with disjoint observed ranges. All primary timing medians
improve by at least 5%; small prefill/TTFT ranges overlap, so no uniform prefill
gain is claimed. Every original workload has a lower complete-request median.
The saved-data auditor fixes exact-buffer identity and extra-event gaps; 35
portable checks and eight-record/568-event/327-binding replay pass. The stricter
all-metric range-separation diagnostic remains false and disclosed. Astra
accepted final source/evidence after byte-identical replay, independent statistics
and 23 receipt/publication bindings; long qualification and the KV decision remain open.

The README now distinguishes the unchanged 2K CPU/default limit from explicit,
model-bounded experimental Metal admission through 32K, and documents the Online
attention selector with its reviewed short-workload scope. PLAN records the
accepted matrices/confirmations while leaving every M6 long-context/KV checkbox
open. This corrects documentation; it changes no runtime default or support gate.
Astra accepted the three-file documentation review without findings; changed
links/anchor, open-task content and whitespace checks pass.

The [M6 long-quality synthesis checker](experiments/2026-10-15-m6-long-quality-audit.md)
requires all four exact depths and 512 observations per model, preserves both
per-depth and pooled loss gates, and checks saved winner/logit/tie consistency.
Changed predictions receive two-candidate reversal accounting and remain open
for separate investigation. Fourteen portable checks pass; partial exact replay
of three records/2,747 events/115 bindings also validates 256 existing real winner
rows. The first small-model 32K profile completes three repetitions per mode
with exact committed state and bounded memory. Online attention's 97.2638%
median contribution triggers KV investigation; it does not prove DRAM traffic
or support an ordinary timing ranking. Astra accepted code/evidence after
independent byte-identical replay, 14 passing tests, all 115 endpoint bindings,
15 receipt pins and 13 published copies. Full long quality, other profiles,
boundary/application runs and the KV decision remain open.

The [M6 I8 KV component contract](../benchmarks/m6-kv-component-v1/manifest.json)
is frozen before component collection, following the reviewed small-model 32K
profile trigger. It specifies signed packed bytes, per-32-value Half scales,
nearest-even rounding, independent reconstruction/arithmetic checks, distinct
resident layer inputs, three warmup/ten alternating measured pairs and separate
confirmations under complete 6 GiB admission/native limits. Its payload is
53.125% of FP16. A negative result can reject this candidate; a positive result
still requires integration, cache-write, state, quality and end-to-end evidence.
This records the experiment design only: implementation checks and measurements
are pending, with no I8 adoption or M6 completion claim.

The [M6 context coverage checker](experiments/2026-10-15-m6-context-scope-audit.md)
requires all 32 model/depth/quality/profile/timing/confirmation roles, exact saved
replay, unchanged quality gates and frozen repetition counts. Six portable tests
pass; four-record compatibility replays 5,652 raw events and rejects incomplete
coverage. The larger 32K profile also completes three samples per mode with exact
state, clean exit and bounded memory. Its 95.6989% Online attention median joins
the small-model trigger for KV investigation. Profile stage time is not DRAM
traffic or an ordinary timing ranking; full context/application evidence and the
KV decision remain pending.
Astra accepted source and partial evidence after independent six-test and
byte-identical replay checks, 318 bindings, eight receipt pins and six published
copies, with larger-profile state/memory/intervals independently recomputed.

The [isolated M6 I8 KV experiment](experiments/2026-10-15-m6-kv-component.md)
passes the unchanged independent arithmetic gate, exact packed-format witnesses,
immutable-input checks and complete component memory limits across four serial
workers/116 events. Small synthetic decode medians improve 5.08%/5.56%; larger
improves 3.20%/2.67%, below the frozen 5% component usefulness threshold. I8 uses
53.125% of FP16 payload. These mixed component results do not qualify engine
adoption or model quality; FP16 remains in use, with full context/application
qualification and final KV review pending.
Astra accepted the code, four-worker evidence and mixed-benefit non-adoption
decision after byte-identical 116-event/6,579-binding replay, 29 receipt pins,
20 published copies and independent timing/memory/input checks. Eleven portable
tests, four packing tests, formatting and strict all-target Metal lint pass;
the engine and protected CPU binaries remain unchanged.

Both [full M6 maximum-context applications](experiments/2026-10-15-m6-full-application.md)
complete 32,640 prompt IDs plus 128 ordinary generated outputs with the real
tokenizer-owning Online CLI. Exact published text/IDs, output-limit state,
32,767 committed positions plus one pending token, reset, independent post-exit
decode, unchanged identities and clean exits pass. Complete admissions are
1,791,516,840/3,122,841,472 bytes and joint native lifetime maxima are
1,615,907,072/2,960,706,752 bytes, within 6 GiB. Saved-data replay covers 262
events and 322 bindings. This is application state/memory evidence under a
cyclic fixed-ID input; input tokenization is excluded, and no latency ranking
or natural long-document quality is claimed. M6 context/retention closure
remains pending.
Astra accepted the full application evidence after byte-identical replay of both
records, all 322 bindings/262 events and eight published copies, with exact
requests, state, output, memory and clean exits independently verified.


The [M6 producer-typed quality path](experiments/2026-10-16-m6-typed-quality.md)
corrects shortest-FP32 JSON interpretation without changing raw events, FP64
statistics, numerical gates or inference. The original larger32K attempt stays
failed because the collector exception prevented natural successful completion.
A separate typed collector/replay and mixed-format dispatcher preserve the old
successful route. Twenty-five targeted portable tests pass; the broader40-test
selection also passes independently under Astra. Seven original cases replay
896 rows/917 events/334 bindings, with both larger4K winner changes still open.
A fresh typed Small4K probe exits0, reproduces all original comparison events
and quality exactly, and replays131 events/112 bindings within6GiB. Astra accepted
code/data after byte-identical independent replay, all6source/26evidencepins and
26exactcopies, with native state/memory/cleanexit verified. Fresh larger32K retry,
winner investigations and full M6 completion remain pending.

The [M6 full-distribution diagnostic](experiments/2026-10-16-m6-winner-investigation.md)
reproduces all original larger4K comparison events and four complete distribution
hashes for the two changed winners. Independent ordering, FP32 candidate bits,
margins, vocabulary errors and target losses verify both top-two reversals.
Fifteen portable Python tests, two Rust example tests, formatting/strict Clippy
and 12 replay-admission mutation checks pass. Exact saved replay covers133 events
and6,843 bindings. Admission2,445,934,228/native2,322,762,608 bytes and separate
post-exit driver75,105,024 bytes fit6GiB; exit0/cleanup/endpoints/identities pass.
Astra independently accepted code/data and manual investigation of only these
two exact SHA/prefix cases after byte-identical replay and all6source/16evidence
pins/16copies; original unresolved automatic flags remain. Full long-context
qualification/M6 completion stay open.

The [complete M6 long numerical quality](experiments/2026-10-16-m6-full-long-quality.md)
now covers all four depths for both models:128 targets/depth and512/model.
Every depth and pooled model mean passes the unchanged +.05 NLL limit; pooled
increases are -0.00001851999728900924 small/-0.0001262955813478074 larger.
Fresh typed larger32K v2 exits0 with no flips, unchanged identities/clean
endpoints and4,442,652,308-byte admission/4,297,527,400-byte native maximum.
Exact8-record replay covers1,048 events/334 bindings and independently matches
under Astra. Only the two previously reviewed exact larger4K reversals remain
flagged automatically; their manual resolution is recorded separately. Failed
v1 and later interrupted ordinary-timing attempts remain unqualified. Astra accepted the final publication after all5source/6evidencepins and5copies
verified; full long numerical quality is qualified. Boundary coverage and final
KV/M6 completion stay open.

The [M6 long-run supervisor](experiments/2026-10-16-m6-supervised-runs.md)
keeps the collector attached to its GPU worker across chat-tool interruptions.
An observed interrupted ordinary32K attempt is preserved as unqualified; its
orphan was stopped and absence verified, while all13 completed roles remain.
A detached owned serial queue resumes only10 unfinished roles, with a fresh
small32K ordinaryv2 destination. Graceful group interruption joins collector
cleanup, includes metadata/file-close failures and preserves primary errors.
Eight CPU tests pass independently under Astra, including a real three-process
fixture proving the separate-session child is reaped before supervisor exit.
No numerical request, quality gate or collector timeout changes. Astra accepted publication after3source/11evidencepins and11exactcopies, with
interruption preservation/role counts/report verified. This is prelaunch CPU
supervision qualification; numerical boundary coverage/M6 remain open.

The [complete M6 context qualification](experiments/2026-10-19-m6-context.md)
closes all32 frozen roles across both models and4K/8K/16K/32K total tokens.
Root and Astra replay22,752 events/429bindings byte identically, preserving the
original automatic flags and only the two exact manually reviewed winner cases.
All ordinary/independent boundary runs finish full-capacity recomputation and
state-preserving overflow checks with clean exits and complete<=6GiB memory.
Max admission/native bytes are2,770,507,468/2,518,502,184 small and
4,576,870,036/4,357,935,304 larger.105Python/97CPU/110Metal-feature tests,
formatting and strict Clippy pass; protected CPU binaries stay unchanged.
Long timings remain descriptive, profiles measure completed GPU attention
fractions, and cyclic stress does not establish natural long-document ability.
Astra accepted code, data and final publication after all4source/126evidence
pins and107exactcopies verified. Separate KV/M6 retention synthesis remains pending.

The [M6 release decision](experiments/2026-10-19-m6-release.md) completes
M6.1–M6.3 for the selected small compensated Half and larger frozen Q4 v2 models
on Apple M1 Pro. It joins the unchanged full10,000-target loss/100-task gates,
all32 context roles, both actual32K applications and a separate final KV decision.
Retain FP16 after complete<=6GiB context/application qualification; preserve
small positive I8 component gains and larger sub5% gains, with no I8 production
adoption or waived cache-write/fullquality/state/end-to-end gates. Repeated
short512/128 numerical request reductions76.26/76.24% small and73.02/73.03% larger
remain scoped to their measurements; long timing is descriptive and natural
long-document ability remains unqualified. Defaults/baseline limits/rejections
stay unchanged. Astra accepted final synthesis/publication after independently verifying all497
bindings/11exactprerequisites/421localtargets and the final FP16 decision.
M6.1–M6.3 are complete in this qualified scope; M7 stays pending.


The first [M7 native boundary component](experiments/2026-10-19-m7-native-abi.md)
adds a versioned CPU raw-ID C ABI with retained immutable models, session-owned
KV/scratch, dedicated-thread operations and separate concurrent atomic cancellation.
Callback reentry returns Busy; failed output preserves pending history and prior
successful emission counts. Rust panics poison affected sessions for recreation.
Eight ABI tests plus a real compiled C client match the independent tiny oracle
and cover ownership, continuation, cancellation, errors/reset/release. Formatting,
strict Metal Clippy,105CPU/118Metal-feature tests and staticlib build pass. Astra
accepted code/data/publication after39source/8evidencepins/4copies and all26
exported tensors independently verified.
No numerical/default/mobile-budget change or M7 completion is claimed.


The [explicit M7 Metal bridge](experiments/2026-10-19-m7-metal-abi.md) adds
compensated Half and frozen Q4 v2 native models with Online attention/FP16 KV,
retained ownership and per-request static dispatch. Both selected real models
match direct-session IDs and7/6/1 pending state; a compiled C Metal client matches
the independent tiny oracle. Combined logical admission counts shared weights
once and stays within6GiB; no new latency/native-footprint/mobile claim follows.
107CPU/119Metal-feature tests, formatting, strict Clippy, six bounded-controller
checks and release builds pass. The timed-out debug attempt stays preserved and
unqualified; fresh release validation passes all244 endpoint bindings with clean
exits. Astra accepted source, native evidence and final publication after all7source/
16evidencepins/244validationbindings,201frozenpayloads and14exactcopies verified.
M7 iOS packaging, Swift/frontend and physical-device qualification remain open.

The [M7 Apple package task](experiments/2026-10-19-m7-apple-package.md)
extends production Metal/native target gates to arm64 iOS and guards precise-math
APIs with macOS15/iOS18 minimums. Separate Mac/device/simulator static-library
slices, C linkage/export/deployment checks and XCFramework file verification
prepare native integration without changing numerical kernels or defaults.
The builder admits a controlled environment and absent Cargo configurations,
uses isolated temporary/build outputs, and drains owned descendant processes.
Ten package and seven integration checks cover cleanup, environment, frozen
provenance and package-integrity failures. Three release slices compile/link with
exact exports/platform/minimums; CPU107/Metal119 tests, fmt and strictClippy pass.
The bounded simulator timeout witness is stopped and verified Shutdown.
Physical-device qualification and overall M7 completion remain pending.
Astra accepted the final task after verifying41source/115evidencepins/94copies,
protected binaries, historical source identities, all slices and native/shutdown
results. M7.1 is complete in this scope; Swift and physical M7 remain open.

The [M7 native history task](experiments/2026-10-19-m7-native-history.md) adds
an allocation-free caller-owned snapshot of exact sampled history, including
pending IDs. It removes cancellation/replacement-prompt ambiguity for Swift
replay while retaining owner/busy/poison guards and existing ABI record layouts.
A tiny recreation test distinguishes identical history from eagerly committed
KV and checks independent continuation IDs; the C client also verifies history.
Physical-device lifecycle and overall M7 qualification remain pending.

Astra accepted final code/data/publication after verifying4source/19evidencepins/
14copies, protected binaries, clean native workers and all check counts.

The [M7 Swift worker](experiments/2026-10-19-m7-swift-worker.md) serializes native
owners on a dedicated OS thread with bounded queued work and polled raw-ID events.
Cancellation and flag lifetime share the queue lock; callbacks have explicit
ARC lifetime, and command autorelease pools drain temporary framework objects.
Transactional replay retains the exact prefix through precancellation; suspend
retains immutable weights, while unload clears replay state. Optimized strict
Swift compilation and actual MacCPU/Metal tiny state/cancel/replay/unload/close/
weak-release checks pass. Swift UI/iOS and physical M7 qualification remain open.

Astra accepted the worker component after verifying30source/evidencebindings,
18copies, the exact compiler command and four clean executions.

The [M7 SwiftUI harness](experiments/2026-10-19-m7-swift-harness.md) adds the
minimal raw-ID native app, shared self-test and arm64 Xcode project. Separate
simulator/unsigned-device release builds verify deployment, resources, library
provenance and simulator signing. Eight builder/five runner checks reject
mutated inputs and preserve cleanup failures. The actual simulator self-test
passes all ten native states, exact initial[47]*4 IDs, replay/cancel/resume,
unload/close/deallocation; installed files/endpoints and Shutdown verify.
Visual QA fixes persistent field labels and locks loaded-model settings.
M7.2 software integration is complete; M7.3 remains blocked by phone Developer
Mode/signing setup and lacks physical memory/thermal/sustained/lifecycle evidence.

Astra accepted task6/M7.2 with no remaining findings after checking current and
historical bindings,58copies, installed files, ten states, clean shutdown,
helper checks, final UI/lifecycle wording and redacted device status.
M7.3 and overall M7 remain pending physical qualification.

The [physical setup follow-up](../benchmarks/results/2026-10-19-m7-physical-setup-v1.json)
distinguishes the newly connected physical iPhone14Pro from the earlier
wireless ProMax and all simulators. The wired Pro's Developer Mode and developer
services are ready; owner account sign-in and Personal Team availability are
verified, while development signing remains open. The harness
guide now explains initiating Xcode pairing before the hidden Developer Mode
setting appears and assigning a Personal Team for local testing. Private
inventories remain outside committed evidence. No engine/app code, numerical
gate, supported phone limit or milestone completion changes; physical runs
have not started.

Astra accepted this setup documentation task with no remaining findings after
verifying all12 pins,354 relative link targets and private-to-redacted facts.
No physical workload or engine change is included.

The [signed physical tiny integration](experiments/2026-10-20-m7-physical-tiny.md)
adds private device/team admission, strict signed Release/profile verification,
streamed hashes for seven admitted files and independently verified phone-process
cleanup. Preparation and exact-install reuse avoid rebuilding while waiting for
owner trust/unlock. Failed cleanup and nested private identifiers cannot be
hidden by a later successful command. Sixteen targeted and52 related isolated
checks pass; both SDK Release builds pass strict Swift checks (inherited Xcode
orientation/AppIntents warnings remain). Actual iPhone14Pro CPU and Metal checks
pass exact initial IDs, ten states, replay/cancel/resume, unload, close and weak
owner release, with matching before/after file proofs and verified process absence.
Earlier failed attempts and intermittent host timeout-probe failures remain
recorded. No numerical gate/default changes occur. Real-model memory, limits,
performance, thermal and actual OS lifecycle evidence remain pending; M7.3 and
overall M7 remain open. Astra accepted code and publication with no remaining
findings after verifying198 pins,69 copies, privacy and363 local links.

Physical evidence storage now explicitly preserves the two console logs' mixed
line endings. A post-commit blob check caught automatic normalization; the
original raw bytes and hashes are retained through path-specific Git attributes.

The [worker timing capture](experiments/2026-10-20-m7-worker-timing.md) preserves
bounded native elapsed events and monotonic callback timestamps, and adds a
non-UI completion wait. Strict Mac CPU/Metal checks retain exact state/ID/release
behavior and validate timing order/reset; both SDK builds and16 controller
regressions pass. Callback timestamps include lock acquisition delay and caller
completion timestamps include wakeup delay. Nine commands/seven events per
backend validate the mechanism; no phone throughput or memory claim follows.
Astra accepted this task and publication with no remaining findings after
verifying110 evidence pins,29 exact copies and all stated build/runtime checks.

The [authenticated measurement runner](experiments/2026-10-20-m7-measurement-runner.md)
adds streamed local artifact identity, explicit native/process trial budgets,
SDK-layout-checked own-process physical footprint/lifetime peak, bounded raw
request samples, separate warmups and optional sustained accumulation. iOS
interruption notifications invalidate foreground trials and request serialized
cleanup. Failure-path artifact proofs preserve original output errors. Twelve
functional cases pass on actual Mac CPU/Metal, retaining26 timed samples;
27 related checks and both strict SDK builds pass. Changed headers/archives/
helpers prevent every owned child. Earlier endpoint/provenance/parser gaps and
executed sources remain recorded. These mechanism checks do not qualify phone
memory/performance or complete M7.3. Astra accepted code and publication with
no remaining findings after verifying164 evidence pins, six code pins, five
historical mappings,50 copies,27 checks and364 local link targets.

Raw Swift-version evidence also preserves its tool-emitted trailing space through
a path-specific Git attribute; engine/source whitespace checks remain active.

The [physical measurement controller](experiments/2026-10-21-m7-physical-measurement.md)
admits exact signed installation and frozen inputs, stages immutable hard links,
checks remote inventory and validates ordered raw token/state/memory/timing output.
Transfer-only preparation and authenticated launch-only reuse avoid rebuilding or
copying after owner unlock. Forty-four related portable checks and15 Mac cases
pass. Actual phone failures exposed filesystem alias containment and startup
activation ordering; both are fixed while retaining child-symlink rejection,
strict foreground checks and checked deadlines. Both SDK builds and actual
physical CPU/Metal runs pass26 raw samples overall, exact tiny IDs/state,
installed/model proofs, close/release and independent process absence. Observed
tiny lifetime peaks are21,546,656/23,037,600bytes; these do not qualify model
budgets or sustained performance. Failed attempts and executed sources remain
bound. Astra accepted source and publication after checking815 distinct paths,
35 historical mappings,106 exact staged copies, both console replays, privacy,
links and whitespace. M7.3 remains open.

The [filesystem observation](experiments/2026-10-21-m7-filesystem-observation.md)
adds a read-only own-filesystem free-byte report before loading, with checked
Foundation numeric conversion and an explicit instrument/monotonic window.
Forty-five portable checks,16 actual Mac cases and both SDK builds pass; Astra
cleared source. Actual iPhone14Pro observation reports32,768,000,000bytes before
13 exact tiny requests, with matching artifact proofs and clean close/release/
process absence. The earlier Locked failure remains recorded. Astra accepted261 bound paths,
38 exact staged copies, independent phone replay, privacy, links and whitespace.
Transfer admission and device limits remain separate; M7.3 stays open.

The [storage admission](experiments/2026-10-21-m7-storage-admission.md)
binds fresh same-device/signed-app observations to checked full payload bytes,
2GiB reserve, both host clocks and a private boot proof. Freshness is rechecked
after hashing directly before every copy, and no-copy retry replays original
admission. Fifty-five portable checks pass, including the stat/hash replacement
fix and expiration with independent cleanup. Small-model transfer and13 actual
short-oracle requests pass exact IDs/state/proofs/release. A near-file-sized
footprint increase during artifact proof is recorded for separate memory work;
larger inference is deferred. Larger packed preparation passes after an earlier
freshness expiration blocks the job write and preserves clean cleanup. The fresh
retry completes all copy checks within84s. Astra accepted978 bound paths,30 exact
staged copies, proof/state/admission replays, privacy, links and whitespace.
M7.3 stays open.

The [bounded hashing fix](experiments/2026-10-21-m7-hash-memory.md) drains
Foundation read buffers after each256KiB SHA update in both model and installed
bundle proofs. The independently hashed256MiB regression fails before loading
with old code (280,724,248byte after-proof footprint) and passes under192MiB with
the pools, preserving hashes, limits and cleanup. Astra cleared source;84 portable
checks,17 actual Mac cases/39 samples and both SDK builds pass. Two same-job
small phone confirmations retain26 exact ID/state samples; after-proof memory
falls from1,013,664,064 to18,990,728/18,826,888bytes and final lifetime peaks from
2,097,663,088 to1,016,990,064/1,016,858,992bytes. Sandbox timeout-test restriction,
existing-process preparation failure and missing-snapshot reuse refusal remain
recorded. Budgets/defaults remain unchanged. Final reclamation varies; these
short trials qualify neither full mobile quality nor model/context limits.
M7.3 stays open.
Astra accepted622 distinct paths,130 exact staged copies, seven historical
mappings, baseline/fixed and storage replays, privacy, links and whitespace.

The [larger short phone oracle](experiments/2026-10-21-m7-larger-oracle.md)
passes two runs/26 requests with frozen `[614,264,3405]`, exact pending state,
before/after model and installed proofs, weak release and verified process
absence. Fresh full-payload storage admission and no-copy reuse pass. Observed
lifetime peaks1,825,098,624/1,824,607,104bytes fit unchanged trial policies;
post-proof footprints stay near19MB. This evidence-only task retains the signed
app/native package and all numerical defaults. Full quality, supported limits,
context/state/lifecycle/sustained and base iPhone14 qualification remain open.
Astra accepted619 bound paths,14 exact staged copies, console/launch/boot and
original admission replays, privacy, links and whitespace, with no findings.
