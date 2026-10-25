# Toklet improvement record

Updated 25 October 2026. This records implemented capabilities, measured benefits,
and the decisions that limit their use. Detailed reports retain exact revisions,
artifact hashes, raw samples, checks and Astra reviews. M0–M6 are complete in their
qualified scopes: M5 selects the small compensated Half and frozen larger group8
Q4 v2 configurations with full quality, repeated workload/application, sustained
and memory evidence. M6 adds explicit Online attention, qualified4K/8K/16K/32K
context cases and final FP16 KV retention. M7 is complete for the amended
iPhone14Pro target and selected trials. CPU/scalar and the existing GPU/readback/
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

The [iPhone context matrix](experiments/2026-10-21-m7-phone-context.md) adds two
repeats of all small/larger128/512/1024 raw-ID rows:156 requests/19,968 events,
identical128-ID sequences per row, exact committed/pending state and end-to-end
exit/storage/install/source closure. Peaks1,114,245,536/1,944,701,920bytes fit the
explicit trial budgets. Heating to serious state is retained; no controlled
speedup, quality, supported-limit or M7 release is claimed. One verified initial
preflight reset precedes all testing;47 other absences are natural. The old
analyzer's all-natural refusal is retained, then narrowly corrected/reviewed with
12 checks plus six driver checks. Astra accepted completed exploratory evidence
with no findings. Full physical quality/lifecycle/remaining sustained/base
iPhone14 gates stay open.

The [committed-distribution score API](experiments/2026-10-21-m7-score-api.md)
adds a read-only48-byte diagnostic for target/greedy FP32 logits and compensated
FP64 loss, enabling later phone teacher forcing without full-logit C copies.
Final-row, exact-prefix, pending/reset and cancellation semantics remain explicit;
owner/busy/poison/pointer guards preserve state and failure outputs. Three new
independent CPU tests,111 ordinary CPU/123 Metal-feature tests, strict Clippy/fmt,
both actual selected-model Metal comparisons, three Apple build/link targets and
Mac C CPU/Metal score clients pass. All14 exports and original CPU binaries are
checked. Astra accepted source, controller and completed evidence with no blocking
findings after3,590 distinct pins,74 exact staged copies, four code pins,
command/test/platform/C replays,379 links, privacy and whitespace. No numerical kernel/default/gate changes or full phone quality/
performance/device-limit qualification follows; M7.3 remains open.

The [Swift score adapter](experiments/2026-10-21-m7-worker-score.md) keeps native
scoring on the owning thread and exposes only bounded validated scalars. Accepted
commands and failures clear earlier scores; completion suppresses publication
when lifecycle cleanup or close supersedes it. Deterministic CPU/Metal barriers
cover both races, and an isolated former-predicate control fails the intended
assertion. Seven portable checks,40 score-state rows/four races,20 legacy worker
rows, strict Swift and both SDK builds pass. A first strict-compile failure and
its exact old source remain recorded. Astra accepted source/regressions and completed evidence after156 bound paths,
66 exact staged copies, one historical mapping, all runtime/SDK/negative
replays, privacy,376 links and whitespace, with no remaining findings. No full physical quality or M7.3 completion follows.

The [teacher-forcing coordinator](experiments/2026-10-21-m7-teacher-forcing.md)
adds bounded reset/score-before-append sequencing, exact committed-prefix checks,
final-target commitment and compensated loss aggregation. Failed append/output
cannot return a complete summary. Seven simulated protocol cases include the
full10000-target frozen block layout; these are not full model-quality results.
Actual tiny CPU/Metal each pass three targets with independent CPU fixture loss,
clean close/release, seven portable checks and strict Mac/both iOS module builds.
Astra accepted source and completed evidence after100 pins,25 exact staged
copies, four code pins, independent oracle/runtime/module replays,379 links,
privacy and whitespace, with no findings. App admission and
full physical corpus/task collection remain; M7.3 stays open.

The [sealed quality harness](experiments/2026-10-21-m7-quality-harness.md) adds
strict config/corpus SHA admission, exact target sequencing, foreground and
interruption checks around each command, per-target process peaks and verified
close/release/endpoint proofs. Shared measurement artifact rules and chunk pools
remain, with parent sources retained. All109 portable M7 checks pass;19 native
cases include six actual tiny CPU/Metal targets and five raw expected failures.
The existing17-case/39-request measurement and256MiB bounded-hash regression,
strict Swift and both complete SDK app builds pass. Exact marker-order and load
bracket parser findings are fixed and tested. Astra accepted source and completed evidence after196 pins,58 exact staged
copies, historical/helper mappings, all native/regression/SDK replays,109-test
record,382 links, actual private-identifier scan and whitespace, with no findings. Full physical corpus/task quality and M7.3 remain.

The [physical quality controller](experiments/2026-10-21-m7-physical-quality.md)
adds sealed score-package/corpus jobs, exact corpus-inclusive transfer/storage
admission, separate receipt types, authenticated no-copy reuse and full raw
prefix/FP32/aggregate/memory/environment validation. Twelve offline methods include
a synthetic10000-target stream and mocked complete prepare transport;121 direct
M7 tests and10 direct package tests pass. Two wrapper-only process-group EPERM
failures, a temporary-root test alias and shell-glob refusal remain recorded;
no cleanup rule or gate changes. Astra accepted source and completed component evidence after58 pins,28 exact
staged copies, both parent mappings, source/test/failure correspondence,385links,
actual private-identifier scan and whitespace, with no findings. No new physical command/model run or full quality/M7.3 acceptance occurs.

The [bounded task-response coordinator](experiments/2026-10-21-m7-task-response.md)
keeps128-token worker commands while supporting256-token raw tasks through exact
continuation. Whole-request preflight, hidden-EOS budget/state, callback order,
persistent cancellation and failed command/sink behavior are checked. Astra's
stale stop metadata finding is fixed with successful idle reason2 publication;
failed/lifecycle commands clear it. Eleven protocol cases/backend and actualtiny
CPU/Metal onecall/chunked/hidden-stop checks pass, as do128 portable methods,
worker/score regressions and both unsigned iOS app builds. Two prior EPERM suite
failures remain recorded. Astra accepted completed code/evidence with no findings. Physical100-task
collection/grading remains; M7.3 is open.

The [sealed raw task harness](experiments/2026-10-21-m7-task-harness.md) adds exact
bounded task/config admission, foreground/cancellation enforcement, raw chunk
clocks/admission/memory observations and per-case reset acknowledgements, with
endpoint proofs and explicit release.31 native test cases/four tiny completions,
12 parser checks,27 clean owned commands including quality/hash regressions and
both iOS app builds pass. All140 portable methods pass across15 isolated modules;
three combined EPERM failures and four corrected-discovery import failures stay
recorded. Astra's clock/consumed-buffer findings and a preflight tiny-test prompt
mistake are corrected with retained old evidence. Astra accepted completed code
and evidence with no findings. Frozen physical100-task collection/grading and
M7.3 remain open.

The [literal task-ID compatibility fix](experiments/2026-10-21-m7-task-id-compat.md)
preserves original labels such as HumanEval/0. All100 frozen IDs contain slash;
Swift now accepts that literal character within the unchanged bounded/unique ID
policy, without using IDs as paths. Twelve parser methods, actualCPU/Metal label
roundtrip,31-case native checks and27 clean owned commands/both iOS apps pass.
Original task IDs/prompts/caps/numerics/gates stay unchanged. Astra accepted source and completed evidence with no findings. Full physical task collection/grading remains.

### M7 physical held-out loss — 21 October 2026

Hypothesis: the selected desktop-qualified paths preserve the frozen loss policy on the wired iPhone. The [full device report](experiments/2026-10-21-m7-phone-loss.md) accepts 10,000 scored targets per model with unchanged +0.05 nats/token gate: small −0.000153474, larger +0.028265241. Both means exactly match their frozen M6 Online results. Fourteen sealed stages, 20,012 total scored targets, 216 clean commands, independent tiny CPU reconstruction and 28 natural absence checks pass; four driver and ten analysis regressions pass. Full peaks 1,028,770,136/1,835,338,624 bytes fit the trial cap; thermal/power conditions and failed preparation/reuse attempts are retained. Retain the numerical evidence without a performance or energy claim. Physical tasks, actual OS lifecycle, small sustained work, supported-limit synthesis, the original base-device requirement and overall M7 remain open.
The [sealed physical task collector](experiments/2026-10-21-m7-physical-tasks.md)
keeps100 literal IDs/prompts/caps and original EOS, with full-payload storage
admission and ordered raw chunks, hidden-stop/state/reset/clocks/peaks/proofs.
Astra's pin-persistence and tiny-oracle findings are fixed; exact previous
sources remain.153 methods/16 isolated modules/16 clean owned commands and an
explicit unchanged frozen-input audit pass. Astra accepted source and completed evidence with no findings.
Phone task collection/grading and M7.3 remain open.

### M7 physical frozen task quality — 21 October2026

Hypothesis: selected desktop-qualified paths preserve the original task policy on the phone. The [full task report](experiments/2026-10-21-m7-phone-tasks.md) accepts100cases/model with zero baseline drop: small2/100,larger11/100. Every visible ID, hidden stop and independently decoded text matches;120text grades recompute and80exact literal-program/test outcomes replay from pinned historical grading, without fresh generated-code execution.14stages/206cases/218clean commands pass;12analyzer/13quality regressions pass. The audit accurately records27natural absences plus1owned pre-copy reset and retains the failed strict analysis/service guard attempts. Retain selected-path task numerical evidence without speed/energy/general-mobile claims; remaining M7 gates stay open.

### M7 dedicated small-model phone duration — 21 October2026

The [sustained phone report](experiments/2026-10-21-m7-phone-small-sustained.md) closes the≥600-second measured-native observation:3warmups+24measured requests total620.566seconds, all128IDs identical. Peak1,114,180,024bytes and fair/serious thermals stay explicit; first/last-five decode medians slow15.511→8.386IDs/s.42clean commands/83sources/3456events replay, with5natural absences and1verified initial owned reset.6driver/16analyzer tests pass. Retained failed analyses corrected cleanup/count/producer-sum assumptions without tolerances, threshold changes or numerical reruns. Retain duration/thermal/memory evidence without controlled speedup, constant-rate, energy or overall device claims.

The [completed-prefix diagnostic](experiments/2026-10-21-m7-worker-prefix.md)
adds opt-in bounded saved-ID snapshots from the owning thread; ordinary callers
retain no prefix copy. The first native check caught stale active counts after
close; fixed terminal publication keeps the strict zero-state oracle unchanged.
Exact failing sources/receipt remain. Astra's race-witness/state-matrix findings
are fixed.26 states/backend, four actual publication witnesses,34 native/build
commands/both unsigned apps and159 portable methods/17 modules pass. Astra accepted source and
completed evidence with no findings. Actual phone lifecycle and M7.3 remain open.

The [bounded lifecycle core](experiments/2026-10-21-m7-lifecycle-core.md)
adds64-event/1152-ID weak-owner traces with before/requested states, separate
observation/record clocks, command/provenance/footprint and completed-close
detach. Astra's cleanup-after-trace-failure and pre-action-bound findings are
fixed. Fourteen events/13 scenarios perCPU/Metal backend,43 native/build commands
and165 portable methods/18 modules pass. The initial fixture-alias parser
failure remains recorded. Astra accepted source and completed evidence with no findings. Actual app/OS/phone
lifecycle qualification and M7.3 remain open.

The [lifecycle app integration](experiments/2026-10-21-m7-lifecycle-app.md)
shares scene/warning actions, preserves original callback times, requires active
foreground READY, and tests exact original-prefix replay against uninterrupted
decode without a second model. Astra findings and the initial test-thread trap
are fixed with old sources retained.51 native/build commands,173 portable methods
and19 Mac synthetic events pass. Retain for physical collection; actual OS/device
qualification and M7.3 remain open.

The [sealed physical lifecycle controller](experiments/2026-10-21-m7-physical-lifecycle.md)
adds sealed jobs, active READY, Settings/foreground activation without replacement,
exact observed/saved/rebuilt IDs and owned cleanup. Astra launch-race and prefix
findings are fixed.190 portable methods/20 modules and two local commands pass;
no native model or device was used. Initial mock/wrapper failures remain. Retain
for actual OS collection; lifecycle/device qualification and M7.3 remain open.

### M7 bounded state diagnostics — 21 October 2026

Hypothesis: the native session preserves exact usable prefixes through cache reuse, active prefill cancellation and rejected capacity requests. The [state app report](experiments/2026-10-21-m7-state-app.md) adds one-model diagnostics and independent fresh references, exact scalar before/after rejection checks and real partial-prefill evidence. It also fixes successful close publishing stale error/admission/emission fields, with direct error-to-close and generated-output-to-close regressions. Astra findings and failed attempts remain pinned.59 native/build commands,206 methods across21 modules and15 focused tests pass; small Mac Metal covers512-token prefill cancellation and1152 fully committed positions. Retain for phone testing, without mobile/performance/energy claims.

### M7 physical state orchestration — 21 October2026

The [sealed state controller](experiments/2026-10-21-m7-physical-state.md) reuses admitted storage/signing/cleanup transport, adds explicit --state mode, separates pure result replay from component-only error-close tests, and admits the complete reviewed current app closure for both state and lifecycle jobs.217portable methods/22modules,11focused parser/admission/shared-controller tests and one owned local two-model inventory/three-native-raw replay child pass. Exact job/SHA, no-copy reuse, absent auxiliary payloads and cleanup after combined console/input failure are verified. Retain for physical collection; no new native model/device/build/performance claim.

### M7 physical activation argv fix — 21 October2026

The [activation boundary fix](experiments/2026-10-21-m7-lifecycle-cli-fix.md) corrects the first actual lifecycle attempt's exit64: device/output options now precede the positional app bundle. The regression fake enforces the real devicectl parsing boundary; both Settings and original-process activation arrays are checked.217methods/22modules and18focused tests pass. The failed phone receipt/source remain pinned; no app/kernel/gate change or physical qualification is claimed.

### M7 phone state and actual scene replay — 21 October2026

The [physical state/lifecycle report](experiments/2026-10-21-m7-phone-state-lifecycle.md) passes both selected models against exact cached/changed-prefix, pending append, actual partial-prefill cancellation, full1152-position commitment and overflow-preservation oracles. Actual Settings/return callbacks interrupt generation in the same process; retained-prefix replay matches uninterrupted continuation.13accepted stages and all sealed sources/raw endpoints replay, with two failed attempts retained and two owned preflight resets disclosed. Synthetic warning-handler coverage stays separate from actual UIKit warning delivery/jetsam. Retain bounded mechanism evidence on iPhone14Pro; base-device scope, energy/limit synthesis and M7.3 remain open.

### M7 bounded energy instrumentation observation — 21 October2026

The [energy availability report](experiments/2026-10-21-m7-phone-energy.md) records two bounded actual PowerProfiler attempts against the exact owned process, using PID then guarded-name selection. The instruments fail with exits21/19 without a trace; energy stays null/unqualified. Both separate native diagnostics pass with independent natural release. Source reviews close parent-cleanup and evidence-binding gaps;11regressions per version cover threaded failure cleanup and exact byte/hash capture. Retain reproducible availability evidence without joules/efficiency claims or production changes.

### M7 joined iPhone14Pro trial limits — 21 October2026

The [device report](experiments/2026-10-21-m7-iphone14pro.md) joins six accepted evidence components with exact quality policies, immutable model identities and33unchanged numerical-source files. Both selected paths retain full loss/task quality; named repeated contexts and sustained sequences fit explicit mobile trial budgets. The largest joined uninstrumented peak is1,944,701,920bytes; state/replay passes through1,152committed positions. Energy remains unset after both bounded instrument failures. Full-loss/task capacities stay distinct from context/state limits. Pro qualification does not silently replace the original base-device M7 requirement; Astra accepted the final Pro-only synthesis with no findings.

### M7 completion after explicit device-target amendment — 21 October2026

The user instructed “Amend M7 to iPhone 14 Pro.” The [M7 closure](experiments/2026-10-21-m7-release.md) updates the plan target and completes M7.1–M7.3 against the already accepted Pro report and six evidence components. The original base-device requirement is superseded for this milestone, not reported as physically tested. A separate closure receipt binds the immutable pre-amendment Pro qualification; historical flags, failed attempts, quality gates and all numerical/runtime bytes remain unchanged. Energy remains null and actual warning delivery/jetsam remain unqualified.

Astra accepted the amended M7 closure and current documentation with no findings. All528documentation links, six accepted prerequisite hashes,43prior evidence pins,33unchanged numerical sources, protected CPU binaries and whitespace checks pass.

### Full-plan audit: grading provenance regression repair — 21 October 2026

The [fixture repair](experiments/2026-10-21-grading-fixture-audit.md) restores two provenance tests that stopped at an obsolete mocked Rust-source inventory. The helper preserves historical identities and synthesizes only missing current Rust sources; a new missing-Cargo.lock regression still requires exact production rejection. Production admission, historical qualifications and numerical gates remain unchanged. Ten focused tests and all 887 Python tests across 127 modules pass, alongside Cargo formatting, CPU/Metal-feature Clippy and 111/123 Rust tests. The initial fixture failures and a separate audit-invocation import failure remain retained. Astra accepted the source and completed evidence after independent hash, source-pin and suite-count replay.


### Full-plan audit and current documentation reconciliation — 21 October 2026

The [M0–M7 audit](experiments/2026-10-21-full-plan-audit.md) maps all 32 tasks and exit criteria to implementation and accepted scoped evidence. It closes stale ABI/backend/device descriptions, marks README and PLAN chronology as historical, and points the early CPU profiling probe to completed matrix/application/retention evidence. The separately reviewed fixture fix is committed as `ff8ba28`. All 887 Python tests, 111 CPU and 123 Metal-feature Rust tests, formatting and strict Clippy pass; ignored hardware/model tests remain explicit. The final audit records Astra review and link/content/hash/whitespace checks. No additional required implementation gap remains within the accepted plan scopes; energy availability, real warning/jetsam, other devices and rejected/deferred mechanisms retain their stated limits. No runtime, defaults, tolerances, quality policy or historical receipt changes.


### Final implementation and performance summary — 21 October 2026

The [final summary](final-summary.md) consolidates implemented M0–M7 capabilities, selected-path quality, measured M1 Pro and iPhone 14 Pro latencies/memory, sustained phone slowdown and remaining limitations. It separates numerical, native and application timing, paired mechanism gains, descriptive long-context trials and independent mobile budgets. It references immutable accepted reports without adding benchmarks, runtime changes or expanded qualification claims. Astra verified the numerical transcriptions and scopes; all 278 local links and whitespace checks pass.


### Native Rust MLX affine storage foundation — 21 October 2026

Following the user's explicit switch from Swift MLX integration to native Rust model support, the [affine reader/reference](mlx-affine.md) adds borrowed MLX U32 packing for 2/3/4/5/6/8 bits, groups 32/64/128 and F16/BF16/F32 metadata. It preserves negative/zero scales, separate biases, unaligned-safe decoding, checked sizes and failure-atomic scalar matvec output. Fifty-four pinned MLX CPU/GPU fixtures preserve every code and separate FP32 reconstruction from native-dtype rounding; 31 native-dtype CPU/GPU differences remain visible. Seven focused tests, 118 CPU tests, 130 Metal-feature tests, formatting, strict Clippy and offline fixture hashes pass. The initial expected-value failure used FP32 powi for a BF16 subnormal; exact expected bits fix the test without a parser/tolerance change. The uncommitted Swift prototype/setup failure and its investigation remain under ignored artifacts. Complete-model loading, architecture expansion, GPU execution, discovery/downloads and physical device qualification remain open in the [M8 plan](mlx-implementation-plan.md).

Astra accepted the affine source and completed evidence with no blockers after independent fixture replay, source/raw hashes, failed-attempt correspondence and check-count verification. This acceptance covers the storage primitive only.

### Native MLX local Qwen artifact source — 21 October 2026

The [local source](mlx-qwen-source.md) validates MLX affine Qwen2 config, single/indexed safetensors, mixed floating/packed tensors, per-module overrides/exclusions and exact tied-head sharing. It streams into caller-owned buffers without a full-model expansion and explicitly separates metadata opening from payload validation. Eight focused tests, 126 CPU tests, 138 Metal-feature tests, formatting, strict Clippy and offline affine hashes pass; protected CPU binaries retain their hashes. Two initial failures were synthetic fixture-construction mistakes; their source/logs are retained and test setup was corrected without changing loader behavior or numerical gates. Admission remains a source-payload/metadata estimate, not GPU/session or device qualification. Astra cleared source correctness; complete numerical execution, architecture expansion, app integration, downloads and physical qualification remain open.

Astra accepted the final source, tests and completed evidence with no blockers after independent pin/raw/hash/count and retained-failure verification. Acceptance covers the local artifact source, not executable model or device support.

### Native MLX resident affine operators — 22 October 2026

The [Metal operator component](mlx-metal-affine.md) keeps original packed weights and separate metadata resident, caches bounded dtype/width pipelines, and implements matrix products plus embedding gather with reusable buffers and checked completion/output guards. The M1 Pro passes 54 pinned storage cases, 864 completed calls and 390,096 exact value comparisons; four explicit GPU tests additionally cover odd Half/BF16 metadata padding, shrinking batches, signed midpoint rounding, failed-result reuse, Half subnormals and the exact signed FP32-normal residual floor. The first actual GPU trial exposed FP32-subnormal flushing; the original failed expectation/source/logs are retained. GPU upload now conservatively rejects subnormal metadata or any possible nonzero fused weight below 2^-79/above finite range, without changing the broader scalar reader or numerical gates. All eight final owned commands pass: formatting, strict CPU/Metal-feature Clippy, 126 CPU tests, 140 Metal-feature tests, offline fixture hashes, two focused admission tests and four release GPU tests. Protected CPU binaries remain unchanged. Astra's requested padding, signed tie-side and residual-floor coverage is included. This is operator qualification only; complete model execution, quality, app integration, discovery/downloads, broader architectures and phone qualification remain open.

Astra accepted the resident operator source and final evidence with no blockers, independently verifying all current/raw identities, retained subnormal failure, numerical counts and domain proof. Acceptance remains restricted to the documented operator domain on M1 Pro.

### Native MLX full-model scalar reference — 22 October 2026

The [diagnostic import](mlx-reference.md) expands one packed tensor at a time into the existing FP32 scalar graph, with checked admission for complete expanded weights, largest packed temporary and metadata/stream estimates. It preserves tied ownership and publishes no model after malformed/nonfinite payload failure. A new mixed 4-bit/8-bit/excluded-projection tiny snapshot uses actual pinned MLX quantization, independent NumPy reconstruction/full traces and actual MLX CPU/GPU logits under unchanged tiny-v1 gates. All 195 NumPy arrays and eight native-logits arrays pass; cached offset/prefix/reset execution, budget boundaries and payload failures pass. Four focused tests, 130 CPU tests, 144 Metal-feature tests, formatting, strict Clippy and both offline MLX fixture checks pass. Astra found an initial exporter provenance gap; the original export/source are retained, and pre-execution source/selected-producer admission plus endpoint verification fixes provenance. Regenerated numerical arrays and model checkpoint remain byte-identical. Protected CPU binaries are unchanged. This explicitly expanded reference is not the production performance/device-fit backend; resident full-graph execution, quality, app/discovery/architecture coverage and physical checkpoint trials remain open.

Astra accepted the corrected reference source and final evidence with no blockers, independently checking all fixture/current/historical identities, 5,248 native-logit values, unchanged gates and check counts. The expanded scalar reference remains diagnostic-only.

### Native MLX Half-activation oracle frozen before GPU composition — 22 October 2026

The [independent graph oracle](mlx-half-oracle.md) freezes 199 arrays/190,336 bytes with original reconstructed packed projections/tied output, Half exclusions/gather/activations, and independent FP32/FP64 accumulation. It retains existing tiny Q4 and Half-storage envelope policies before full-model GPU evaluation. Seven portable tests pass: complete manifest/gate identity, exact coverage/layouts/finite values, actual saved MLX CPU/GPU logits against the storage envelope, causal prefix bits, byte-exact regeneration, source/immutability rejection, and stdlib-only verification. Sixteen output dots verify the deliberate original-FP32 tied-output boundary differs from Half-rounded weights. Both frozen four-step token series are [47,47,47,47]. The owned exporter and all three verification commands pass with unchanged Rust/shader sources and protected CPU binaries. No new target full-model GPU, quality, performance or mobile claim; graph composition remains open.

Astra accepted the frozen oracle and final evidence with no blockers after independent identity/layout/gate replay, including all 16 tied-output differences, recorded checks and unchanged runtime sources.

### Native MLX resident complete tiny graph and sessions — 22 October 2026

The [native graph integration](mlx-metal-model.md) streams packed MLX tensors directly into resident GPU allocations, retains one tied embedding/output allocation and uses explicit MlxAffine sessions with Half activations/exclusions and original packed FP32 vocabulary reconstruction. Strided row views and final-row offsets use the existing completed chunk command. On M1 Pro, 525 diagnostic snapshots/74,784 values pass pre-frozen independent gates; both attention/readback choices, 16 selected IDs, pending/changed prefixes, actual partial prefill, loader/mode rejection, gather overflow and transaction fault injection pass. Completion failure is injected after real staged work, not an observed command failure. Four prior affine tests still pass 54 cases/864 calls/390,096 exact values after the ABI change; the existing Half decoder regression passes 2,520 stage arrays and warmed allocation/state checks. Formatting, strict CPU/Metal-feature Clippy, ordinary Rust tests, all three offline MLX fixtures and seven portable oracle tests pass; protected CPU binaries remain unchanged. Retain this selected tiny numerical/state mechanism. Real-checkpoint quality, process peaks/performance, CLI/app selection, broader architectures, downloads and iPhone MLX qualification remain open.

Astra accepted the graph source and completed evidence with no blockers after independently checking all current/raw identities, unchanged gates, exact coverage/counts and qualification limits. Both owned runners pass all 13 commands, including ten explicit GPU tests; ordinary totals remain 130 CPU and 144 Metal-feature tests.

### Explicit native MLX CLI selection — 22 October 2026

The [CLI integration](mlx-cli.md) adds explicit `--metal-projection mlx-affine` routing to the resident native Rust loader and compensated normalization. It preserves existing defaults and rejects backend/Q4 conflicts. Readiness/statistics identify mixed MLX affine/Half storage and per-tensor quantization without inventing a global group or Toklet Q4 version. Ten explicit M1 Pro subprocesses cover greedy/hidden-stop and persistent cache/continuation/rejection/reset behavior with exact IDs/state and logical accounting. Astra requested stronger state assertions after the first passing version; both sources/raw runs remain retained, and the strengthened suite passes without a runtime/gate change. Formatting, strict CPU/Metal-feature Clippy and all ordinary Rust tests pass; historical CPU binaries are unchanged. Retain the explicit CLI surface. Native Apple app selection, real checkpoint quality/performance/peaks, other architectures, downloads and iPhone MLX execution remain open.

Astra accepted the CLI source and final evidence with no blockers after verifying current/raw/historical source correspondence, unchanged numerical identities and exact subprocess coverage. Final ordinary totals are 131 CPU and 145 Metal-feature tests; all six final owned commands pass.

### Native MLX C bridge entry point — 22 October 2026

The [additive native bridge](mlx-native-bridge.md) exposes resident MLX loading with the existing ABI-v1 layouts and ownership/error contract. Native sessions explicitly select MlxAffine/Compensated/Online/FullLogits; unsupported builds reject without CPU fallback. Mac Rust and actual C callers pass frozen tiny IDs, direct scoring, shared-model retention/release, exact pending/cancellation/rejected-input/reset state and existing callback/ABI checks. iOS Rust compilation also passes; app UI/package and physical phone execution remain separate. The first new GPU test incorrectly expected ENGINE_ERROR for an out-of-vocabulary request; retained failed source/logs show the unchanged boundary correctly returns INVALID. Correcting only that expectation preserves the exact state assertion. All ten final owned commands pass: formatting, strict CPU/Metal-feature Clippy, ordinary Rust tests, explicit GPU ABI test, native library build, C link/run and iOS check. Protected CPU binaries remain unchanged. Retain the native integration boundary; real checkpoint quality/performance/process peaks, harness selection, broader architectures, downloads and iPhone MLX qualification remain open.

Astra accepted the native bridge source and completed evidence with no blockers after verifying all source/raw/binary identities, the exact retained failure/correction, final checks and qualification limits. Ordinary totals are 132 CPU and 146 Metal-feature tests; both actual Mac callers and iOS compilation pass.

### Explicit MLX Apple worker and UI selection — 22 October 2026

The [Apple worker/UI integration](mlx-apple-worker.md) adds explicit native MLX routing, rejects CPU/MLX/separate-Q4 conflicts before loading, and preserves existing defaults and lifetime rules. Mac CPU, Half Metal and native MLX workers pass identical exact token/state/replay/cancellation/cleanup oracles; the MLX path also rejects conflicting settings and a missing artifact before recovering with valid data. Strict arm64 iOS18+ source typechecking passes. Both passing source/raw runs remain retained: the final UI refinement clears MLX when choosing the bundled floating Tiny fixture. All eight final owned commands and source/archive/fixture/protected-binary endpoints pass; Rust/shader bytes are unchanged. Retain for new app packaging and physical collection. Linked app builds, sealed real-model job selection, real checkpoint quality/memory/performance and actual iPhone MLX testing remain open.

Astra accepted the worker/UI source and completed evidence with no blockers after verifying current/historical/archive/raw identities, exact state/timing matrices, unchanged numerical sources and the limited qualification scope.

### MLX-enabled native package and linked Apple apps — 22 October 2026

The [package integration](mlx-apple-package.md) fixes source provenance that previously bound only the original Metal shader. The builder now binds all Rust and embedded Metal sources; a regression catches both actual libraries and a nested future shader. All 11 focused portable methods pass. A fresh three-slice arm64 package passes 21 native build/link/platform/symbol/bundle commands, then the exact Astra-admitted receipt produces linked release simulator and unsigned device apps in nine commands. The known generated library alias switch is recorded; the old M7 package remains immutable. Actual Mac C execution passes CPU, Half Metal and native MLX tiny ABI/state/score/cleanup checks. Rust/shader bytes, ABI/math policy and protected historical binaries are unchanged. Retain the package and linked apps for signed physical testing. Development signing/install, sealed real-model jobs, checkpoint quality/memory/performance and physical iPhone MLX qualification remain open.

Astra accepted the packaging repair and completed native/app/Mac evidence with no blockers after verifying all current/raw/archive/app/alias identities, the preserved prior package and unchanged numerical inputs. All 36 recorded commands are clean, including outer wrappers; physical and real-checkpoint claims remain open.

### Authenticated MLX diagnostic jobs and pinned trial data — 22 October 2026

The [versioned job integration](mlx-apple-jobs.md) keeps exact legacy schemas and adds explicit true/Metal/no-separate-Q4 MLX selection across measurement, quality, task, state and lifecycle runners. Forty malformed variants reject; authenticated tiny jobs pass 13 measurement requests, three scoring targets, two task responses and bounded state checks. Legacy CPU/Half checkers pass 17/19/31 case labels, strict iOS typechecking and fresh linked app builds pass, and numerical sources/protected binaries remain unchanged. The injected READY failure closes/releases owners but exposes an existing active-trace detach cleanup error; its raw negative result is retained and repair is explicitly next before physical trials. Separately, reviewed data-only acquisition publishes the immutable public four-bit Qwen0.5B snapshot after three portable checks and eleven exact-identity downloads. Tokenizer ID semantics match the frozen source, while file/template identities differ and are not silently accepted. Real numerical references, quality/performance/phone qualification and app discovery/download implementation remain open.

Astra accepted the job-routing source and completed evidence for the stated positive scope, including the authenticated execution bindings and exact acquisition/tokenizer identities. The observed lifecycle trace cleanup remains explicitly unqualified and is the next separate repair before phone collection; no real-model inference or physical/performance claim is made here.

### Lifecycle trace cleanup after external coordinator failure — 22 October 2026

The [trace-cleanup repair](mlx-lifecycle-cleanup.md) ends a remaining healthy trace label only after verified complete worker shutdown and signal detachment. It preserves the original error and all already failed trace evidence. Strengthened MLX and Half tests actually reach READY and retain the injected failure with empty cleanup/endpoint errors and post-close trace detachment; CPU records the distinct absent-active-window failure with clean closure. Both legacy app unit runs pass 34 labels and both low-level trace runs pass 13 labels, including sticky failures and resource release despite exhausted diagnostics. Authenticated MLX positive components, strict iOS source checks and fresh linked apps also pass. All 12 owned outer/nine app child commands complete with unchanged numerical sources and protected CPU binaries. Retain the cleanup fix; the earlier failing trace receipt remains immutable. Actual OS/device behavior and real-model reference/quality/memory/performance remain open.

Astra accepted the trace-cleanup repair and completed evidence with no blockers after independently verifying all current/raw/prior/app identities, distinct observed negative branches, unchanged primary failures, complete detachment and sticky-trace guards. This closes the observed diagnostic defect while OS and phone qualification remain separate.

### Independent selected real-checkpoint MLX reference — 22 October 2026

The [public-checkpoint oracle](mlx-real-oracle.md) freezes 346 full CPU graph arrays before any target GPU run. Independent unsigned nibble/signed Half affine reconstruction feeds the unchanged Half-storage FP32/FP64 graph, with Half gather and original FP32 tied-output weights. Six prompt/position cases and four teacher-forced steps yield matching reference IDs; selected producer and 19 NumPy binary identities, exact input descriptors, full arrays and frozen numerical gates are bound. Seven new portable methods plus seven existing tiny-oracle methods pass; all four owned commands and numerical/protected-binary endpoints are clean. Logical admission is 1.762 GB and observed lifetime peak physical charge is 0.661 GB. Astra's source review required exact schema, finite tolerance arithmetic, single-descriptor verification and external manifest admission; those fixes precede export. Retain the reference for target trials. The largest independent stage allowance is 3.0 and the logits allowance about 0.2653; these do not qualify full loss/task quality. Real target parity, Mac/phone quality/performance, physical app execution, architecture expansion and discovery/download product work remain open.

Astra accepted the real-reference source and final evidence with no blockers after independently verifying all 346 slices/gates, exact saved IDs, 84 provenance/output/protected pins, four owned commands and the disclosed qualification limits.

### Public MLX Qwen checkpoint native Mac parity — 22 October 2026

The [real native Mac tests](mlx-real-metal.md) pass the independently frozen public-checkpoint reference: 816 exact-inventory snapshots / 13,771,520 values, six prompt/position cases, one/three/full chunks and selected layers. Eight full vocabulary distributions and 16 exact greedy selections pass Reference/Online attention and FullLogits/token-only contracts, with maximum logit difference 0.0288893. Cached prefixes, invalid-token failure atomicity, changed-prefix fresh equality, reset and committed partial-prefill reuse pass. Resident logical weights are 277,997,982 bytes and maximum diagnostic session admission is 387,662,292 bytes; no whole-process peak or performance claim follows. Six final ordinary/build commands and two explicit GPU tests in one command pass, with unchanged numerical/Cargo sources and protected binaries. Astra's requested mandatory logits/absence checks and distribution count precede target execution; the first passing ordinary suite/source remain retained. Retain selected native execution for app and full-quality trials. Held-out quality, actual app/phone memory/performance/lifecycle, template compatibility and broader M8 scope remain open.

Astra accepted the real Mac parity source and final evidence with no blockers after independently verifying 89 current/historical/reference/binary pins, all 34 raw files, recomputed coverage, all 13 current/retained owned commands and the qualification limits.

### Shared MLX job staging and real Mac app transport — 22 October 2026

The [shared raw-ID transport](mlx-mac-jobs.md) stages exact version-two tiny/public MLX jobs for Mac and iOS and routes the Mac through the app's existing coordinators. Eight actual Mac jobs pass: 13 four-ID requests per model, three scored targets, complete short task responses and all 26 component state rows. Full task/chunk/hidden-EOS accounting reuses the physical validator with an explicit Mac scope while preserving physical defaults; complete state validation is reused unchanged. Eight staging, 17 task and 15 state methods pass. All 19 owned collector commands satisfy their expected outcomes, including four deliberate CLI rejections, plus five supplemental source checks. Public native process peaks range from 293.36 to 384.78 MB under the explicit trial cap; these do not qualify representative performance or phone fit. Astra required persistent package/header admission, same-buffer log evidence and full response/state validation before collection. The existing-process phone preparation refusal remains retained; an identity-checked reset and 13-command signed installation then pass with zero inference launches. Retain for full-quality/performance and physical trials. Numerical sources/protected binaries and frozen quality gates are unchanged; broader M8 gaps remain open.

Astra accepted the shared-job source and final evidence with no blockers after independently verifying 430 pin checks, all 104 raw files, exact task/state/score/measurement streams, 40 portable methods, source correspondence and private-identifier/qualification boundaries.

### Guarded native MLX app execution on iPhone 14 Pro — 22 October 2026

The [physical MLX transport](mlx-phone-jobs.md) authenticates the existing signed app, immutable staged jobs, fresh same-device storage proof and exact copied inventories before launch. Bounded consumed buffers bind metadata and raw console evidence; fast ready reuse avoids recopying the model. Ownership/device checks and cleanup survive source mutation. All 96 portable methods and ten physical stages pass, with 104 successful transport commands (94 devicectl and ten host boot queries). Cleanup has 19 natural absence endpoints and one verified initial owned-app preflight reset, all ending absent before measurement or at cleanup. The public Qwen checkpoint completes 13 exact four-ID generation requests, three scored targets and one original task with an eight-token short cap. Observed public native lifetime peaks are 307.97 MB for measurement, 306.28 MB for scoring and 397.64 MB for tasks; tiny observations remain below 21.52 MB. Astra's source requests for auxiliary inventory, consumed-byte proof, cleanup identity and strict schema types precede collection. Numerical/Swift sources, frozen quality gates and protected CPU binaries are unchanged. Retain this selected physical functional path. Full quality, representative/sustained performance, physical state/lifecycle, tokenizer-template compatibility and broader M8 architecture/discovery work remain open.

Astra accepted the completed evidence with no remaining blockers, verifying 484 pins, exact source copies, six console replays and storage decisions. The corrected publication retains the initial accounting versions, distinguishes host boot queries and the verified preflight reset, and passes all 126 final raw inventory bindings and document links.

### Full public MLX checkpoint loss checks on Mac and iPhone — 22 October 2026

The [full scoring validator](mlx-full-loss.md) adds explicit Mac proof scope while preserving physical defaults and every target/block/state/logit/memory/aggregation gate. All 54 relevant portable methods pass. The unchanged public four-bit checkpoint completes 10,000 frozen targets on each device; all paired greedy IDs, target logits and losses agree exactly. Both mean losses are 3.0829338398049995 versus frozen FP32 2.883725149929484, an increase of 0.19920868987551543 nats/token that **fails** the unchanged 0.05 gate. Retain that rejection and the shared validator; do not adopt this candidate or infer same-source quantization attribution from its model card. Native lifetime peaks are 306,840,368 bytes on M1 Pro and 320,014,176 bytes on iPhone 14 Pro, with Mac thermal state zero, phone states zero/one and Low Power Mode off. All 45 transport commands (41 devicectl/four host boot queries), five outer commands and four portable commands pass. Cleanup records seven natural absence endpoints and one verified pre-copy owned-app reset, all ending absent. Astra required consumed-receipt/source/raw closure checks through exclusive publication; sources and protected CPU binaries remain unchanged. Full task quality, representative/sustained performance, physical state/lifecycle, a loss-qualified MLX candidate and broader M8 product work remain open.

Astra accepted the completed rejected-checkpoint experiment with no remaining blockers after replaying both 10,000-target streams and verifying 322 pins, 99 raw files, 70 source copies, all commands, conditions, privacy and 221 links. Acceptance applies to the validator/evidence task; both numerical loss gates remain failed.

### Independent upstream-runtime diagnosis of the MLX loss rejection — 22 October 2026

The [local diagnostic](mlx-upstream-loss.md) evaluates the identical immutable checkpoint and all 10,000 frozen targets with static trusted MLX-LM Qwen2 classes, completed GPU work and independent stable FP64 loss scoring. Upstream mean 3.083056082158314 also fails the unchanged gate (+0.19933093222882992 > 0.05), differing from Toklet's mean by only +0.0001222423533144834. This reproduces the application-family rejection upstream without attributing it to an unproven precise source or changing native acceptance. Forty-four greedy IDs differ; maximum target-logit/loss differences and the Half-upstream/FP32-native output boundary remain explicit, with no invented parity tolerance. Upstream process peak is 620,070,376 bytes under its 6 GiB reference cap; no speed/memory ranking follows. All 14 portable methods and both final owned hardware/reference commands pass. Astra's initial shard-inventory and failure-journal findings have actual zero-load/partial-row regressions. The agent caught an omitted separate Metal-library pin during the first trial, stopped only its identity-verified driver/worker, retained all 7,449 partial rows and interruption/absence evidence, then reran with all non-cache MLX files bound. Retain the diagnostic and failed artifact decision; native product defaults/sources and numerical gates are unchanged. A loss-qualified MLX candidate, full task/performance/physical-state/lifecycle evidence and wider M8 architecture/discovery work remain open.

Astra accepted the diagnostic with no remaining blockers after independently verifying 1,145 current pins, all 105 raw files, 57 source copies, 10,000 target rows and descriptive comparisons, 14 portable checks, the excluded interruption, privacy and 222 links. No checkpoint adoption or numerical-policy change follows.

### Immutable higher-precision public MLX candidate — 22 October 2026

The [eight-bit acquisition](mlx8-checkpoint.md) pins the off-the-shelf public Qwen0.5B snapshot at revision6474d4c1a05bd7bd634a9c993560e4e7272eda3d as an explicit next candidate, with no held-out calibration or gate change. Three portable methods and eleven exact HTTPS/size/Git/LFS download commands pass; ten files total536,580,527bytes are published exclusively. Metadata has628 tensors/169 packed matrices with eight-bit/group64 Half metadata and the expected Qwen2/tied structure. Frozen vocabulary/merges/added IDs match, while tokenizer files/template differ and remain unqualified. Sources and protected CPU binaries match their recorded identities. Retain the immutable data for a separately frozen numerical reference before any target inference. No native execution, device fit, quality/performance, product discovery/download or broader architecture acceptance follows; the four-bit rejection remains intact.

Astra accepted data acquisition only, verifying 99 pins, all ten Git/LFS file identities, 84 raw files, 51 source copies, exact metadata/tokenizer checks, all command outcomes and 50 unchanged numerical/app sources. Privacy, 223 links and whitespace checks pass; no inference or adoption qualification follows.

### Independent eight-bit Qwen graph oracle frozen before target trials — 22 October 2026

The [eight-bit CPU reference](mlx8-real-oracle.md) independently reconstructs four unsigned octets per U32 with signed Half metadata and preserves original FP32 tied-output weights, Half gather/exclusions/activations/KV and the established FP32/FP64 graph. Fourteen portable methods and all four owned suite/export/stdlib-verification commands pass. Six authored prompt/offset cases and four greedy distributions freeze346arrays/38,461,888bytes; FP32/FP64 greedy IDs both[4460,311,1855,264]. The unchanged pre-target envelope policy yields largest stage allowance1.375/logits0.2790870666503906, explicitly separate from quality gates. Logical admission2,009,108,622bytes and observed lifetime physical peak629,425,784bytes stay within6GiB. Astra's wrapper-log preflight finding is fixed before execution. Selected producers/19NumPy binaries, every array and protected CPU/source endpoints remain bound. Retain this oracle for the next native trials; existing four-bit references/rejection remain unchanged. Eight-bit native app/device/quality/performance and wider M8 scope remain unqualified.

Astra accepted the frozen CPU reference after independently checking 98 pins, 72 raw files, 59 source copies, every array's value/layout/hash/gate and exact greedy IDs, all four owned commands and 14 portable methods. Privacy, 227 links and whitespace checks pass; no target or quality acceptance follows.

### Native eight-bit public Qwen numerical/session qualification — 22 October 2026

The [eight-bit Mac tests](mlx8-real-metal.md) pass the independently frozen oracle across816snapshots/13,771,520values, six cases, chunking/offsets and selected layers. Eight full distributions and16exact IDs pass both attention and readback choices; maximum logit difference is0.02828359603881836, with unchanged gates. Pending/rejected-input/reset/changed-prefix and committed partial-prefill reuse checks pass. Resident logical weights524,978,590bytes and maximum diagnostic session admission634,642,900bytes are allocation counts, not physical peaks. The initial copied four-bit300MiB assertion stopped graph checks before comparison; its exact source/raw failure remain retained. The reviewed correction uses exact payload plus floating-tensor guards, changes no numerical gate, and explicitly labels the overwritten old executable as superseded. Six final ordinary/build commands and both GPU tests pass with clean source/model/oracle/protected-binary endpoints. Retain selected native execution for app and full-quality trials; full quality, representative performance, physical state/lifecycle, chat-template compatibility and broader M8 product scope remain open.

Astra accepted the source and completed evidence with no remaining blockers after verifying92pins/82rawfiles/44sourcecopies, all final commands and comparison coverage, retained failure/disclosure and qualification limits. Privacy,228local links and whitespace checks pass.

### Eight-bit native app jobs on Mac and iPhone 14 Pro — 22 October 2026

The [shared app integration](mlx8-app-jobs.md) adds explicit immutable eight-bit recipes while preserving corpus, task, prompt, capacity and stop policies and the existing tiny/four-bit configurations. Fresh storage admission and ready-proof replay cover every admitted non-tiny candidate; regressions catch bypass for both public artifacts. All 101 portable methods and seven owned commands pass. Four actual Mac app jobs and ten signed-phone stages pass, including 13 exact four-ID generation requests, three scores and one short task per device, plus full Mac component state checks. Compiled app/native sources, packages, the signed bundle and protected binaries remain unchanged. Native physical peaks range from 539.38 to 630.80 MB on Mac and 552.73 to 643.99 MB on phone. All 104 transport commands comprise 94 devicectl operations and ten host boot queries; 19 natural absence endpoints plus one verified initial preflight reset end absent. Six phone console replays agree exactly, with no cleanup or endpoint errors. Retain selected short integration for full-quality trials. Full loss/task quality, representative and sustained performance, physical state/lifecycle, chat-template compatibility and broad architecture/discovery work remain open.

Astra accepted the source and completed evidence with no remaining blockers after verifying 438 pins, 145 raw files, 83 source copies, all console replays, actual storage/reset/launch evidence and command counts. Privacy, 231 links and whitespace checks pass.

### Eight-bit full loss acceptance on Mac and iPhone 14 Pro — 22 October 2026

The [full frozen loss trial](mlx8-full-loss.md) identifies a selected native MLX candidate that passes the unchanged application-family loss gate on both devices. Each scores all 10,000 original targets with mean 2.884953696657687, an increase of 0.0012285467282029394 over FP32 baseline 2.883725149929484, below the 0.05 limit. Every paired greedy ID, target logit and loss agrees exactly. The four-bit rejection remains retained; model-card provenance does not prove exact prequantization source bytes or controlled quantization attribution. Observed native peaks are 553,075,648 bytes on Mac and 566,265,888 on phone. Low Power Mode is off; thermal states are zero on Mac and zero/one/two on phone, with no performance qualification inferred. Five collection commands and 45 transport commands pass (41 device operations/four boot queries); seven natural absence endpoints and one verified initial preflight reset finish absent. Both complete validators replay, all gate decisions recompute, and source/private/protected-binary endpoints remain clean. The evidence retains 108 raw files and 73 source copies, alongside the current 101-method portable suite. Retain selected loss acceptance for the next full task-quality and performance trials; broader M8 scope remains open.

Astra accepted full loss quality for the selected checkpoint with no remaining blockers after independently verifying both complete streams, every paired target, 262 pins, 108 raw files, 73 source copies, storage/reset/cleanup and command/test totals. Privacy, 234 links and whitespace checks pass. Full task quality and adoption remain separate.

### Repeated remote idle admission for task grading — 22 October 2026

The [grading-window hook](mlx-task-grading-window.md) lets a physical-response adapter check remote app identity/absence at all five existing idle boundaries while retaining mandatory local process/group and input checks. Default callers, literal controller/classifier and VM ownership/restoration remain unchanged. Existing files and dangling window/probe/restoration outputs reject before work. Five new mocked methods cover repeated checks, remote failure before and after startup, local-check precedence and exclusive outputs; all 49 methods in four related suites pass. Broader checks exposed a stale synthetic legacy test inventory, now extended only with backend-appropriate current Rust/shader names while preserving historical identities and strict missing-input rejection. Initial environment/fixture failures and exact intermediate sources remain retained. No actual device query, VM run, generated-program execution, new full task-quality or numerical/performance qualification follows. Retain the mechanism for the next full native-app response adapter; broader M8 remains open.

Astra accepted the source and completed evidence after verifying 124 pins, 120 raw files, 85 source copies, exact historical failure correspondence and all 49 methods. Mandatory local checks and owned restoration remain intact; numerical/Swift/protected inputs, links and whitespace pass. Actual remote response admission and full task grading remain open.

### Full eight-bit native-app task response collection — 22 October 2026

The [full raw collection](mlx8-full-task-collection.md) completes all 100 original task responses on both M1 Pro and iPhone 14 Pro, preserving every literal prompt, full output cap and stop policy. Each produces 123 chunks, 6,520 visible IDs and 66 hidden stops; all paired visible IDs, hidden stops and sampled/committed/pending/chunk counts agree exactly. Both complete validators replay from captured bytes. Observed native peaks are 631,620,640 bytes on Mac and 644,466,768 on phone; power/thermal observations remain recorded without performance acceptance. Five collection commands and 45 transport commands pass, with seven natural absence endpoints and one verified preflight reset ending absent. The publisher binds 389 inputs/evidence files, retains 94 raw files and 75 source copies, and preserves numerical/Swift/signed-app/protected inputs. No tokenizer, VM or generated program runs in this collection phase. Retain raw evidence for reviewed post-exit decoding and isolated grading; full task-quality acceptance remains open.

Astra accepted raw collection after verifying all 389 pins, 94 raw files, 75 source copies, both complete streams and exact paired responses, storage/reset/launch/absence evidence and command/method totals. Privacy, 236 local links and whitespace checks pass. No tokenizer or task-outcome qualification follows from this task.

### Post-exit native-app literal response admission — 22 October 2026

The [response adapter](mlx-task-responses.md) authenticates both full raw collections, replays all original cases, and decodes with the frozen evaluation tokenizer only after fresh owned-host/group and exact signed-phone absence checks. It preserves literal fences/spacing, hidden stops, sampled/committed state and original app JSON substrings in explicit derivatives. All100 paired decoded dictionaries agree exactly; 71 portable methods pass, including16 new adapter checks. Source review adds separate command/capture-cause retention and per-boundary profile expiry checks. The initial temporary-path test mismatch, superseded passing suites, hardware-details schema failure and live-phone refusal remain retained with exact historical sources. A separately reviewed identity-checked one-SIGTERM test-app reset proves absence; the adapter itself stays read-only. The final retry passes two idle observations/eight read-only commands, with no VM or generated program execution. Publication binds540 files,142 raw files and78 current source copies while preserving numerical/Swift/package/signed-app/protected inputs. Retain literal derivatives for isolated task grading; task-quality acceptance and broader M8 remain open.

Astra accepted response admission after independently verifying540 pins,142 raw files,78 source copies,71 methods, exact historical correspondence and all200 pinned-tokenizer decoded dictionaries. Actual reset/idle ordering, unchanged inputs, privacy,238 links and whitespace pass. No task-quality acceptance follows yet.

### Full eight-bit native-app task quality — 22 October 2026

The [full task qualification](mlx8-full-task-quality.md) passes the unchanged gate on Mac and iPhone 14 Pro: 2/100 each versus FP32 2/100, zero drop against the two-percentage-point limit. Each result is 0/40 code, 1/30 prose and 1/30 continuation; the low original baseline and zero-pass boundary remain explicit. The adapter independently reconstructs both literal derivatives after fresh host/phone absence. One qualified Linux window executes 40 original source-test programs; both devices replay exact program/test/completion hashes and recompute 60 text outcomes, with zero unresolved infrastructure. Six local/remote idle observations and 24 read-only commands pass. The VM returns to stopped with helper ownership finished.

Astra's pre-launch review removes a finite parent cleanup deadline that could kill the owner during restoration and adds original private failure ledgers plus late-guard retention. All 81 portable methods and seven commands pass, including cooperative interruption with a simulated 30-second helper drain and running/foreign-machine refusal. Fixture errors, the unexecuted rejected parent, initial live-phone refusal, exact-app reset and restricted publication attempt remain retained; the latter's unavailable original query row is disclosed rather than reconstructed. The corrected publisher binds the final post-exit query before assertions and retains 722 pins, 177 raw files and 112 current source copies. Numerical/Swift/package/signed-app/protected inputs and the four-bit rejection remain unchanged. Retain selected full loss/task acceptance for representative performance and physical state/lifecycle trials; broader M8 remains open.

Astra accepted the selected full-task gate after independently verifying all 722 pins, 177 raw files, 112 source copies, 81 methods, exact outcome replays and original baseline/gate. All 40 code cases resolve SyntaxError with clean removal; runtime identities, 12 VM control captures, six idle observations, reset/private-ledger/post-exit evidence, privacy, 241 links and whitespace pass. Performance and broader M8 qualification remain open.

### Strict Mac native-app measurement admission — 22 October 2026

The [Mac measurement counterpart](mlx-mac-measurement.md) preserves completed serial/callback/native clocks, state/admission, monotonic process peaks and first-eligible duration accounting, while requiring absent UIKit fields and zero synthetic installation proofs. Existing physical defaults stay unchanged. Typed vocabulary, exact repeated greedy IDs and unknown-marker rejection strengthen this explicit Mac scope. All 36 portable methods and three owned commands pass. One optimized native app trial completes three warmups plus ten measured original 128/128 requests on M1 Pro, with every 128-ID sequence identical. Descriptive medians are 0.746537 seconds native TTFT, 31.8685 IDs/second decode and 4.731781 seconds caller completion; observed native lifetime peak is 639,337,456 bytes. The load/session observation is separate, cache state is explicit, AC power/Low Power Mode off/thermal zero are recorded, and no cold-cache or speedup claim follows. All ten collection/observation commands and source/capture/cleanup endpoints pass. Publication replays the full stream and retains 763 pins, 146 raw files and 114 source copies without numerical/Swift/package/protected-input changes. Retain the mechanism for the full matrix, second confirmation and sustained Mac/phone trials; broader M8 remains open.

Astra accepted the Mac measurement mechanism and descriptive trial after independently verifying 763 pins, 146 raw files, 114 source copies, all 13 requests/1,664 events, the regenerated frozen recipe, exact medians/peak and all 36 methods/13 commands. All 51 numerical/Swift/Cargo sources, protected binaries, power/thermal observations, privacy, 242 links and whitespace pass. Full performance qualification remains open.


### Repeated and sustained native MLX app performance — 22 October 2026

The [selected eight-bit matrix](mlx8-performance.md) completes all 128/512/1024 primary/confirmation workloads and dedicated 512/128 sustained sequences on M1 Pro and iPhone 14 Pro. All 256 requests/32,768 native events replay exact IDs/state, with matching cross-device and confirmation responses. Mac 512 primary/confirmation decode medians are 22.1772/22.2205 IDs/s and native TTFT 3.1133/3.1134 seconds; phone results are 15.2318/13.6681 IDs/s and 13.3074/15.1906 seconds. These are descriptive observations. Mac sustained 68 measured requests cover 600.685220793 native seconds; phone 26 cover 628.915998376 seconds, all excluding three warmups. First-five to last-five decode changes are −0.866%/−22.324%, with nominal Mac and nominal/fair/serious phone thermals. Maximum native process peaks of 639,157,232/651,888,720 bytes remain distinct from logical admission and general memory guarantees; energy is unmeasured.

The reviewed data-only publisher binds canonical recipes/native/prepared/ready/proof identity, exact storage/copy freshness and raw installation/process/signal response identity; final intervention rejects. All 53 saved-data methods and the existing two orchestration methods pass, with 68 outer and 154 phone commands, 29 natural absence observations and three verified preflight resets. The cooperative collector retains original errors and has no finite parent deadline that could kill cleanup. The first publication's self-mutating stdout inventory is retained and explicitly unaccepted; fresh v2 output avoids the inventory, repeats checks/replay and retains 1,306 pins, 427 raw files and 114 current source copies. Numerical/Swift/package/signed-app/protected inputs and quality gates remain unchanged. Retain selected performance observations; physical MLX state/lifecycle and wider M8 architecture/discovery/product scope remain open.

Astra accepted v2 and the documentation with no remaining blockers after independently verifying all 1,306 pins, 427 raw files, 114 source copies, 51 unchanged numerical/Swift/Cargo sources, fourteen complete sequences, all launch/storage/process/signal replays, 222 clean collection commands and all 55 current check methods. Statistics, privacy, 261 local links and whitespace pass. Acceptance covers selected-checkpoint repeated/sustained raw-ID observations only.


### Native MLX physical state witnesses — 22 October 2026

The [selected phone state trial](mlx8-phone-state.md) passes all 33 rows on iPhone 14 Pro using the unchanged signed app and native Rust/Metal package. Explicit state-512 recipes reuse the frozen 512-token contract, capacity 1152 and chunk 128. The new host validator preserves the unchanged state oracle and strict signed physical/foreground admission. Actual cancellation leaves 128 committed positions and the exact 384-token input tail with no emitted IDs; resume matches fresh evaluation. Cached primer/scalar, pending-token append, changed-prefix references, full 1,152-position commitment, all three overflow rejections, recovery and zero-resource close pass. Observed process peak is 651,921,488 bytes, nominal thermal state and Low Power Mode off; these are state witnesses, not performance or general memory limits.

All 100 related Python methods, four orchestration and 22 data-replay methods pass, plus Rust 1.90 formatting, strict all-target Metal Clippy and 148 ordinary Metal-feature passes (74 explicit ignored cases). The unrelated ambient Rust 1.99 lint failure remains retained; no engine repair or lint waiver is introduced. A source-review timeout mismatch is corrected before replay, with its unexecuted draft retained and independent 3600/7200 substitution mutations. All four actual stages/42 transport commands pass, with seven natural absences and one verified initial reset, no final intervention and no unresolved cleanup/endpoints. Publication independently replays every state/launch/storage/ownership row and retains 1,498 pins, 207 raw files and 128 source copies. Exactly four changed historical Python paths use their original accepted copies; numerical/Swift/Cargo/package/signed/protected inputs stay unchanged. Retain selected physical state observations; actual OS lifecycle, architecture expansion, chat compatibility and discovery/downloads remain open.

Astra accepted the source, completed phone evidence and documentation with no remaining blockers after independently verifying all 1,498 pins, 207 raw entries, 128 source copies, 51 unchanged numerical/Swift/Cargo sources, both console streams and all storage/launch/ownership/state replays. Test counts, retained failure/draft, privacy, 266 links and whitespace pass. Acceptance covers this selected checkpoint’s physical state diagnostic only.


### Native MLX actual scene interruption and replay — 22 October 2026

The [selected phone lifecycle trial](mlx8-phone-lifecycle.md) closes the remaining selected local-Qwen M8.2 witness. An explicit lifecycle-512 recipe prepares the unchanged signed app; a separate cooperative owner requires natural preflight absence and returns from Settings to the same live process without forced relaunch. Nineteen actual trace events preserve ten sampled IDs and the exact 522-token interrupted prefix; replayed continuation matches fresh reference IDs, close releases ownership and final absence needs no intervention. The synthetic warning action remains distinct from actual UIKit warning delivery/jetsam. Observed whole-process lifetime peak is 663,799,936 bytes, nominal thermal endpoints and Low Power Mode off. No numerical, shader, Swift, package, signed-app, quality-gate or default-backend change is made.

Astra's source review adds deterministic shared proof dependencies without losing per-invocation source guards, composed standalone preparation/actual-ready admission tests, and immutable original/late failure journals with consumed log hashes. A later review requires explicit device-details outcome/type at both endpoints. Fresh checks pass 137 Python methods, four current orchestration methods and 16 saved-data methods, plus Rust 1.90 formatting, strict all-target Metal Clippy and 148 ordinary Metal-feature passes (74 explicit ignored cases). The final four-stage phone run has 44 clean commands, eight natural absences and no signals. Independent replay retains 1,849 bindings, 259 raw entries and 132 current source copies; all numerical/Swift/Cargo/package/signed/protected inputs stay unchanged.

The initial sandbox boot-query denial issued no device command. The successful pre-hardening run is superseded under exact original collector/test copies, and the hardened rerun alone qualifies. Two early mock-only failures and the old auditor's unavailable original source snapshot are explicitly disclosed without reconstruction or current-check credit. Retain selected native MLX lifecycle support and close M8.2's local-Qwen scope; broader architecture implementation, chat compatibility, discovery/downloads and wider M8 qualification remain open.

Astra accepted the code, completed evidence and documentation with no remaining blockers. Independent verification covers all 1,849 bindings, 259 raw entries, 132 source copies, 51 unchanged numerical/Swift/Cargo sources, exact historical mappings and both strict streams. All transport/storage/process/activation/launch/ledger replays, 157 current portable methods, Rust totals, privacy across 265 public files against nine actual identity/signing/path tokens, 269 local links and whitespace pass. M8.2 closes only the stated selected local-Qwen scope; M8.3–M8.5 remain open.


### Pinned language/vision architecture reference coverage — 22 October 2026

The [M8.3 inventory](mlx-architectures.md) records every Python model source in pinned MLX-LM and MLX-VLM trees: 137/1,399 modules and 126/232 static root Model exports, plus literal loader aliases. These are reference counts, not new native support or unique graph-family counts. Recursive Git tree reconstruction, exact selected path/blob/mode equality, bounded gzip-before-tar parsing and AST-only inspection preserve source identity without importing upstream code. Ten parser and six mocked ownership methods pass. All 1,403 immutable vision-source transfers finish with no retry or cleanup/capture error; the 120-second incomplete whole archive stays excluded. Language subset assembly records permission normalization and never follows the unrelated archive symlink. The collection binds 5,691 inputs/evidence files and preserves all 50 numerical/Swift/Cargo sources/protected binaries; prior Rust checks are explicitly inherited, with no new numerical or hardware claim. Retain this bounded reference inventory to guide independently specified dense, expert, recurrent/hybrid and multimodal implementations. M8.3–M8.5 remain open.

Astra accepted the code, reference inventory and documentation with no remaining blockers after independently checking all 5,691 bindings, two source copies, both reconstructed Git roots, 1,544 selected files, 1,403 owned transfers, 16 test results and 50 unchanged numerical/Swift/Cargo sources. Privacy, 270 local links and whitespace pass. Acceptance covers static reference coverage only; native architecture expansion and M8.3–M8.5 remain open.


### Independent dense Llama oracle — 22 October 2026

The [first Llama task](mlx-llama-oracle.md) freezes809 tiny arrays before native target changes, covering bias-free GQA, full causal masking, split-half ordinary RoPE, tied and deliberately distinct untied heads, FP32 and mixed-Half storage, five cases and four greedy steps. Actual pinned MLX CPU/GPU full-prefix and two-plus-two cached logits pass fixed FP32 gates. Independent ordered FP32/FP64 reductions freeze mixed-Half tolerances before target evaluation. The wrong-tied-head witness changes the selected ID.

Astra's review requires the complete non-cache MLX runtime closure, externally pinned manifest identity, mandatory finite non-Boolean gates and exact greedy/witness replay. All12 portable methods pass, including mutations for missing gates, overflow, versions, producer omission and metadata tampering. The owned export/check receipt preserves existing numerical/Swift sources and protected binaries; prior Rust checks are inherited. Retain this independent oracle for native Llama implementation. Native target execution, real models, quality/performance and phone qualification remain open.


### Native bias-free Llama source and scalar decoder — 22 October 2026

[Strict Llama loading](mlx-llama-scalar.md) adds explicit dense admission and diagnostic FP32 execution with full causal GQA, ordinary split-half RoPE and independently owned untied output. Packed/floating tensor validation and bounded streaming are reused; source and expanded head bytes count before loading, tied output keeps one embedding allocation, and empty bias vectors represent actual bias absence. All three scalar output paths select the correct head at the operation boundary with static numerical dispatch.

Six CPU-only task tests pass390 frozen stage comparisons, saved actual upstream CPU/GPU logits, chunk/offset/reuse/reset/overflow contracts, bias/head constructor validation and memory capacity accounting. Astra identifies premature Llama admission through the shared reader; explicit open_dense/load_dense now opt in, while original source/Metal/Qwen entry points retain the original Qwen parser. The first full CPU run exposes the changed old error-text regression, retained without credit or an original source snapshot. Fresh Rust1.90 fmt, strict all-target Metal Clippy,140 CPU passes/eight ignored and154 Metal-feature passes/74 ignored pass. Protected binaries and the frozen oracle remain unchanged. Retain native tiny scalar Llama; GPU, real checkpoints, quality/performance and iPhone qualification remain open.


### Resident tiny Llama with qualified Reference attention — 22 October 2026

The [Metal Llama task](mlx-llama-metal.md) retains original resident affine embedding/projection/head buffers, optional immutable QKV biases and one reusable session zero buffer. An independent untied head counts exactly2,560 additional bytes; tied output keeps one embedding allocation. Head selection and architecture/capability selection stay outside numerical loops, with no per-token allocation or full-model expansion. Geometry is explicitly limited to the frozen tiny Llama shape.

The first two release trials expose the existing shader's hardcoded Qwen theta1e6 rather than Llama theta1e4. The repaired rotary ABI carries validated FP32 theta bits at all three dispatch sites;13,824 independent values across both bases pass the unchanged operator envelope. A later Online candidate fails the frozen offset-case second-layer attention-projection gate; its precise numerical cause is unproven. Reference-only Llama is retained, explicit Online requests reject before session allocation, and existing Qwen Online remains available. The C ABI explicitly selects Llama Reference before session construction rather than retrying a failed mode. Failed logs and exact source correspondence are retained; only the comment-only header uses a disclosed exact-SHA-verified reconstruction. No tolerance is changed.

The accepted M1 Pro run passes1,050 graph snapshots/149,568 values, eight full distributions and16 exact selected IDs, plus cache/pending/offset/replay/boundary/budget checks and explicit Online rejection. All11 release GPU/integration cases pass, including Qwen MLX, ordinary/compensated Half, CLI and the Rust C-ABI caller. Fresh Rust1.90 fmt/strict all-target Metal Clippy,140 CPU passes/eight ignored and155 Metal-feature passes/77 ignored pass. The receipt binds3,639 files and76 current source copies; oracle and protected binaries stay unchanged. Retain tiny native Reference-attention Llama. Real Llama checkpoints, wider shapes, Online attention, held-out quality/performance, a new iOS package and physical phone execution remain open; M8.3 is not complete.


### Immutable real Llama-family candidate — 22 October 2026

[SmolLM2-135M eight-bit acquisition](mlx-smollm-checkpoint.md) pins a public LlamaForCausalLM graph at revision0f0d9b8218915bc34d401e1a340b8c049d300d5e. Nine whitelisted files/147,875,570 bytes pass immutable Git/LFS identities and exclusive bounded HTTPS publication. Three portable methods pass. Exact data-only inspection validates694 tensors/211 packed matrices/61 normalization vectors, absent projection biases, tied output, contiguous payload spans and index equality. The30-layer576-hidden graph exercises nine-query/three-KV-head GQA. Retain as the real-model candidate; download size does not establish phone fit. No numerical, device, quality, performance or chat claim follows. Rust/shader/Swift and protected binaries are unchanged; prior Rust checks are inherited. Freeze the independent real-model oracle before target changes and preserve separate SmolLM tokenization/quality evidence.


### Independent real Llama-family numerical gates — 22 October 2026

The [SmolLM2 CPU oracle](mlx-smollm-oracle.md) freezes349 arrays/13,945,632 bytes before target edits: six prompt/offset/future/tile cases across layers0/1/29, four greedy distributions and three scoring-prefix distributions. The independent unsigned8bit/F16 reconstruction retains original FP32 tied-output weights, Half gathers/exclusions/activations and nine-query/three-KV mapping. An optimized bias-free graph is cross-checked against the ordered scalar tiny implementation. FP32/FP64 greedy IDs agree exactly at339,3543,719,1891; captured tokenization and BOS/EOS identities are SmolLM-specific.

Nine portable methods and three owned commands pass. Numerical formula remains the predeclared real-graph FP32/FP64/equivalent-prefix policy; actual maximum absolute allowance32 and logits allowance0.207672119140625 are disclosed, with no held-out quality claim. Logical reference admission1,195,800,705 bytes and measured lifetime physical peak185,025,592 bytes stay under6GiB; neither establishes phone fit. The receipt binds579 files and58 current source copies, with unchanged Rust/shader/Swift/protected endpoints and inherited Rust checks. Retain this frozen CPU reference for real target qualification.


### Real Llama-family resident Mac execution — 22 October 2026

[Selected SmolLM2 native execution](mlx-smollm-metal.md) admits only the inspected30-layer576/1536 nine-query/three-KV shape, theta100000, epsilon1e-5 and context<=512. Export switches have exact-value admission; Reference-only Llama/default native routing and Qwen paths remain unchanged. Original packed affine weights stay resident with one tied table and142,955,990 allocated weight bytes; no shader or gate change.

All816 frozen selected-stage snapshots/4,988,928 values pass, plus four full greedy distributions, three scoring prefixes and eight exact selected IDs. Maximum stage error16 is inside the independently frozen allowance32; maximum observed decode/scoring logit error0.0589752197265625. A separate512-capacity/chunk128 witness covers128 committed/384 pending discard/resume, overflow preservation, cached no-op and exact fresh-run full distribution. It does not claim asynchronous cancellation or independent long-context numerical parity. Logical session admission peaks at254,860,338 bytes, not a process/phone limit.

Astra requested and reviewed the actual512 witness before GPU execution. Rust1.90 fmt, strict all-target Metal Clippy,142 CPU passes/eight ignored and158 Metal-feature passes/80 ignored pass; all14 explicit release GPU/integration cases pass. The corrected test-only Clippy allocation and sandbox no-device trial remain retained with exact source correspondence. The receipt binds3,425 files and77 current source copies; frozen oracle/protected binaries are unchanged. Retain selected real-model Mac Reference execution. Rebuilt app, actual phone execution, held-out quality, Online, chat and wider architecture scope remain separate.


### Real Llama-family app and physical execution — 23 October 2026

[Selected SmolLM2 app/device integration](mlx-smollm-app-device.md) keeps native Rust/Metal execution with resident original eight-bit affine weights and Reference attention. Explicit tokenizer-specific raw-ID recipes use49152 vocabulary/EOS2, independent scoring prefixes and two authored cases; Qwen quality datasets are not reused. Complete explicit package/prepared contexts preserve legacy defaults and original storage replay. A distinct physical component-state route preserves the legacy prefill witness requirement. Exact-current-installed-app reset has strict command/device/PID/executable/signal admission, same-buffer captures, preserved original/late causes, cooperative interruption and zero-command substitution tests.

Fresh Mac/device/simulator archives build/link; strict Swift coordinator compilation/iOS type-check and the actual signed iPhone Release install pass. All232 portable methods, fmt and strict all-target Metal Clippy pass; numerical tests are inherited unchanged from the reviewed real-model Mac task. Both devices pass primary/confirmation13-request generations, three exactly paired scalar scores satisfying independent frozen-logit intervals, two matching authored task responses and26 exact state/scalar rows. Thirteen physical stages complete130 owned commands and26 natural absences, without trial cleanup intervention; two initial exact-app resets remain separate.

Short4/4 raw-ID native TTFT medians are39.471/39.950ms Mac and63.240/63.359ms phone; decode interval medians18.986/19.201ms and27.102/27.615ms. Maximum component process peaks156,730,016/169,854,720 bytes remain separate from512MiB numerical/1GiB process trial caps. These are functional short observations, not representative/sustained throughput or comparative speedup claims. Retained setup failures cover the old generated link, a later existing process and missing exact reuse snapshots; no failed attempt receives inference credit. The publication binds4,790 files and298 current source copies; all14 saved-data replay/mutation methods pass. Retain selected real-model app execution on both devices. Full quality/chat, Online, independent long-context parity, actual interruption and broader M8.3–M8.5 scope remain open; phone probes currently exercise32 positions.


Astra accepted the completed code, physical evidence and documentation with no remaining blockers. Independent verification covers all4,790 bindings/298 source copies, exact command/storage/signing/console replay, both separate resets, paired outputs/state/scores,232 portable and14 saved-data methods. Privacy across16 changed/new public files against seven actual identity/path/container tokens,285 local links and whitespace pass. Acceptance remains bounded to this selected SmolLM2 raw-ID app scope.


### Exact text profile for one Toklet app — 23 October 2026

The user requested the friendly InferencePhone interface inside Toklet Harness,
with its current controls retained in Expert mode. The neighboring app belongs
to a separate earlier runtime; it is a visual reference for this consolidation.
The first [text boundary](app-chat-profile.md) validates the selected SmolLM2
tokenizer/config identities and implements its exact bounded user/assistant
template outside numerical kernels. Independent Python Jinja/tokenizers cases
freeze32/68-token Unicode/multi-turn rendering, IDs and decoding. An additive
immutable C handle provides caller-owned query/encode/decode buffers with explicit
thread ownership, a separate64MiB text allowance and no model lifetime borrow.

Astra's review identifies empty null-buffer validation and relative-directory
drift. Both are fixed; absolute origin uses the actual load path and moves into
one exact-capacity allocation, with native bridge allowance raised4→20KiB.
Full-prefix decoding replaces displayed text rather than concatenating isolated
token strings. Rust1.90 fmt, strict all-target Metal Clippy,147 CPU/163 Metal
ordinary passes, two explicit local tokenizer/C-handle methods, independent
Python replay and one actual Mac GPU public C lifecycle test pass. The
[receipt](../benchmarks/results/2026-10-23-app-chat-profile-v1.json) binds103 files
including source copies; protected CPU binaries stay unchanged. A preliminary
wrapper path exception is observed in tool output; only preliminary fmt logs
are retained, with no test credit. Astra accepts this bounded task after the
final source/evidence audit. Retain the exact text
profile; shared-worker/UI and phone text integration follow separately.


### Friendly Chat and preserved Expert mode in one app — 23 October 2026

The [interface consolidation](app-interface.md) addresses the confusing split
between the earlier InferencePhone UI and Toklet's diagnostic app. Toklet Harness
now opens friendly Chat, with local model selection, bounded saved history/drafts
and settings; the original raw-token form remains under Expert mode. Both use
one Toklet worker and the existing Rust/Metal engine. The selected SmolLM2 text
profile supplies formatting/decoding; no second numerical runtime or implemented
Hugging Face download flow is implied.

Astra review found a queued-chat cancellation race before dequeue: cleanup could
replace an accepted command without returning a terminal result. The worker now
publishes cancellation for that request ID and retains terminal results through
cleanup/Close. Deterministic suspend/unload/close checks cover the race. Review
also tightened failed-read recovery, pending folder cancellation, security-scope
release, result routing across conversations, terminal worker recreation and
rejected-load identity. The final stale-published-state regression verifies that
a rejected load cannot label an already loaded model as another selection.

The final strict Swift6 Mac runner passes all six commands: SDK query, compile,
workspace/controller regressions, real SmolLM2 text and original raw CPU/Metal
checks. Both SDK app builds pass; final development signing/install passes on
iPhone14Pro. An earlier exact-source physical smoke matches Mac text/token IDs
for first/second turns and explicit session recreation, with native cancellation,
context rejection, release, signed correspondence and natural process absence.
Original physical CPU/Metal worker checks pass. Historical source differences
are retained as exact-SHA-verified copies; the final store-only fix is covered
by the fresh Mac regression and app builds/install. Numerical Rust/shaders stay
unchanged from the reviewed text-profile task, so its Rust checks are inherited.

Visible checks cover Simulator navigation, Files opening/cancel, original Expert
generation, terminal Close/Back and draft entry; physical dark appearance was
inspected. The final Simulator capture preserves the draft after reinstall.
Physical GUI Send/Stop, successful folder selection and large-text/keyboard
layouts remain unqualified. No performance improvement or broader quality claim
is made. The tiny model's arithmetic reply is incorrect; this is functional text
support only. A proposed new collector was rejected in review and never executed;
existing reviewed physical helpers and a bounded manual smoke provide the device
evidence. The [receipt](../benchmarks/results/2026-10-23-app-interface-v1.json)
records checks, scope, historical correspondence and remaining limitations.
Retain the one-app interface; M8.4 search/downloads and wider chat profiles remain
open.

Astra accepts the final interface task for commit with no remaining blockers:
412 audited evidence bindings/75 exact source copies, six final Mac commands, both SDK
builds, final signing/install, historical physical correspondence and all304
documentation links independently pass. Protected binaries remain unchanged.

The final staged-file whitespace check additionally catches one blank line at
EOF in the new chat store. Removing that newline changes no Swift tokens; the
receipt retains the exact executed source and maps the final whitespace-only
version explicitly. Functional checks are inherited unchanged.
Astra clears this amendment after verifying all413 bindings,75 current source
copies and the exact one-newline correspondence.


### Immutable public-model search/download core — 23 October 2026

The [model hub core](model-hub.md) separates searchable public MLX repositories
from the single currently qualified SmolLM2 chat download. The hypothesis is that
a pinned, verified app-owned snapshot can replace manual model copying without
claiming arbitrary search results are compatible. A cold Swift actor admits the
immutable revision and six trusted file sizes/SHA-256 hashes, streams sequentially,
reserves storage and publishes an authenticated local receipt only after full
verification. Existing snapshots are preserved. Networking/hashing stay outside
the numerical worker; Rust, shaders, native package and protected CPU binaries
remain unchanged.

Astra found path-based prechecks before open and a missing active-URLSession
cancellation witness. Verification now uses no-follow/nonblocking descriptors,
`fstat` admission, bounded reads and the same opened inode for parsing/hashing;
openat binds children to an opened directory. Ancestor symlinks reject before
mkdir. A live delayed URLProtocol sends1MiB through the actual sink, then
cancellation stops the transfer before its continuation returns and prevents
later writes. Exact cancellation/publication boundaries, storage changes,
corruption, FIFO/link/path/revision/receipt errors, duplicate requests and
exclusive publication collisions have deterministic regressions.

The final v6 strict Swift6 compile, deterministic suite and iOS type-check pass.
Retained earlier failures include test-autoclosure compilation, URL directory
identity and explicit Apple `/tmp`/`/var` alias handling; no acceptance gate was
loosened. Models-screen integration is the next separately reviewed task.

The actual Mac URLSession search/download succeeds for all146,607,552 required
bytes, followed by full saved-snapshot verification. Independent Python hashes
match all six files against the previously qualified local source. This task
establishes transport/integrity behavior only; UI, numerical loading and physical
use remain separate. The [core receipt](../benchmarks/results/2026-10-23-model-hub-core-v1.json)
retains exact source and result identities. Retain the bounded download core.
Astra accepts Task1 for commit with no remaining blockers after independently
verifying all67 bindings, three current source copies and the actual network
result. Links and staged-file whitespace checks pass.


### Hugging Face Models screen and offline Chat opening — 23 October 2026

The [Models screen](model-hub.md) now offers public MLX search, an explicitly
qualified pinned SmolLM2 download, byte progress, Cancel and offline opening.
Other search results explain that engine/text qualification is missing. Runtime
admission reads current iOS process memory allowance and unified-GPU capability;
Mac checks use the plan’s configured allowance minus current process footprint,
not free RAM. Downloading and verification stay off the UI/native worker. Saved
models are rehashed before the existing Chat worker is asked to load them.

Astra identified a late-open race: an awaited verification could override a
newer model selection or Expert transition. Model-changing actions now invalidate
the operation and cancel it; competing controls are disabled while busy, and
completion checks the operation identity and current mode. Delayed-verification
regressions prove the normal callback runs once and that Expert/new-model/unload
choices prevent the late callback. Other controller checks cover failed/duplicate
requests, late progress, missing snapshots, device rejection and cancellation.

All seven final strict Mac commands pass, including controller tests, ordinary
workspace tests, real chat and original raw CPU/Metal checks. Both SDK builds
and final signing/install pass. Real Mac execution revalidates the previously
transferred snapshot, then passes native text checks and actual HubStore →
FriendlyChatStore → worker generation using a temporary history and a network
handler that refuses every request. The 32-ID controller reply retains the exact
native eight-ID prefix and releases all model resources. No new quality or
performance claim follows.

Retained setup attempts include an existing-app preflight refusal, exact-app
resets and a cached phone run whose native/controller checks succeed but whose
fresh-transfer requirement correctly fails. The saved model is preserved. A
separate explicit fresh-download diagnostic uses an exclusively created UUID
temporary tree, rejects reuse and removes only its own data before success.
The one-off phone verification wrapper was tightened before execution to check
selected device/outcome/command, exact query/signal ownership, exclusive outputs,
both diagnostic streams and same-buffer hash/parse correspondence. Earlier
uncleared wrapper drafts were never executed and receive no test credit.

The first isolated phone transfer was still progressing when its 1,400-second
remote test bound expired. The strict wrapper retained the failed trial and
confirmed the exact test process absent after its scoped SIGTERM. It receives no
fresh-download credit. The observed file growth does not establish the cause of
the slow transfer. A read-only Mac capacity-query probe was fast and did not
justify changing storage checks. The resource timeout is now a bounded two hours;
the 60-second request timeout, cancellation, storage and integrity checks stay
unchanged. Astra accepted the narrow adjustment, and final-source checks repeat
before a new isolated trial with longer, still bounded test deadlines. A killed
trial can leave its temporary files until the OS purges them; process-kill
recovery remains unqualified.

The final fresh iPhone 14 Pro trial passes public search, a new six-file transfer,
all snapshot hashes, native text checks and the actual offline Chat controller
path. It explicitly reports `reusedSnapshot=false`; native and full 32-ID
controller outputs match the Mac, and the app exits naturally after releasing
resources and removing its owned temporary tree. The normal model cache and
owner conversations remain separate. A preceding retry refused a missing exact
test-helper source copy before any device command or app launch; that retained
preflight failure was resolved by restoring the hash-matching copy.

Final-source core checks pass all three commands, controller/chat/raw checks pass
all seven commands, both SDK builds pass and the signed final app is installed.
The [app receipt](../benchmarks/results/2026-10-23-model-hub-app-v1.json) binds
the final source, historical inputs, native package, signed installation and
retained failures. These are functional checks; the new search/download tap flow
and keyboard/accessibility layouts remain visually unqualified. No throughput,
answer-quality or broader M8.5 qualification follows.

Astra accepted this task for commit after independently verifying all 584
bindings, 81 current source copies and 16 historical mappings, final builds and
signed phone proofs, fresh-transfer result, native/full-controller Mac equality,
natural exit and retained failures. All 318 local links, staged whitespace and
task-file privacy checks pass. The decision is to retain this bounded feature;
wider downloadable/text profiles remain open.

### Qwen native text profile and profile-specific stop IDs — 23 October 2026

The existing Qwen2.5-0.5B eight-bit numerical checkpoint needs its own text assets,
system message, stop IDs and memory admission before the friendly app can use it.
The independent Python/Jinja export was frozen before target edits. Exact model
configuration and tokenizer hashes now select one of two bounded text profiles,
then validate the loaded geometry. The numerical kernels and raw-ID APIs remain
unchanged. Qwen uses 128 MiB of separate logical text allowance; SmolLM2 retains
64 MiB. A coarse immutable C query supplies the complete profile stop set before
generation, avoiding SmolLM2’s stop IDs being reused for Qwen.

Astra found a repository-label typo in the initial Qwen reference export. The
original producer/fixture and executed checks are retained. The correction changes
only that metadata label to the pinned Qwen2.5-0.5B repository; all rendered text,
IDs, decoding and other metadata compare exactly. It grants no numerical or
quality waiver. Astra also requested explicit stop-query bounds and thread/pointer
coverage before final evidence acceptance. The bounded real C bridge check loads
Qwen, rejects insufficient text allowance, generates/decode text and proves a
forced stop produces no visible token. Native model release does not invalidate
the immutable text handle or session. Swift/catalog integration follows separately.

All seven final commands pass: Rust 1.90 formatting, strict all-target Metal
Clippy, ordinary CPU and Metal suites, both pinned local codecs, native real GPU
text/stop checks and independent Python replay. The [Qwen text receipt](../benchmarks/results/2026-10-23-qwen-app-chat-profile-v1.json)
retains source copies and earlier checks. Protected CPU binaries remain unchanged.

Astra accepted this native task after independently verifying all 157 bindings,
47 current source copies and six historical mappings, all three seven-command
receipts and the metadata-only correction. Final logs contain 148 CPU passes
(11 ignored), 164 Metal-feature passes (85 ignored), two explicit codec checks
and three native text checks. Whitespace, 237 local links and public-task privacy
checks pass. The decision is to retain the bounded native extension.

### Qwen Models integration and per-profile admission — 23 October 2026

The native profile commit `7fd8c4c` enables Qwen’s bounded text boundary. Models
now exposes the pinned Qwen2.5-0.5B eight-bit and existing Llama-family SmolLM2
profiles separately, including exact download identities, supported-result
selection, per-profile memory admission and offline opening. It does not enable
the rejected Qwen four-bit candidate. Qwen’s numerical/text budgets are 1 GiB and
128 MiB, with a conservative current 1.5 GiB process allowance; Smol’s existing
policy remains. These reservations are not physical memory measurements.

The worker queries native stops once at load, retains them across suspension and
clears them on unload. This prevents Qwen token 0 from being hidden as a SmolLM2
stop. Local Files configuration identification is bounded and follows security
scope acquisition, then fresh memory admission. Unknown/changed configurations
reject before worker loading. Profile selection cannot change during an operation;
refresh and open retain the intended profile, and earlier cancellation guards
remain intact. Old Smol receipts preserve their exact schema and file identities.

Strict integration/core checks, both SDK builds and added profile regressions
pass. The regressions cover old receipt compatibility, exact/changed/linked local
configs and selected Qwen refresh/open followed by Smol refresh. Astra cleared
the runtime source for actual trials and requested explicit Qwen token-0 parity
and a Smol re-open with the new package. A phone preparation attempt correctly
refused an existing app session it did not own; the user was asked to close it
while independent Mac work continued. No owner process was terminated.

The fresh real Mac Qwen transfer passes complete pinned-file verification,
native multi-turn/state checks and the actual offline controller. The first
eight IDs match the committed C bridge exactly, including Qwen token 0, and all
native/controller text independently decodes with the pinned Python tokenizer.
Smol’s old saved snapshot verifies and passes native/controller generation with
the new package. The [Qwen app receipt](../benchmarks/results/2026-10-23-qwen-model-hub-app-v1.json)
binds these sources/builds/checks and the retained phone refusal. Phone Chat,
physical text-memory peaks, GUI taps and performance/quality remain unqualified
by this task. Mistral is separate work.

Astra accepted the Mac-functional app task after independently verifying all 311
bindings, 84 current copies and nested package/build/check closures: 21 package,
nine SDK, seven integration, three core, two profile and two real Mac commands.
Independent byte-level decoding matches every Qwen and Smol native/controller
reply. The phone refusal contains three read commands and no install, launch or
signal. Whitespace, 323 local links and 19 public-task privacy checks pass.
Retain the feature; actual physical Qwen Chat remains pending.

### Mistral reference tools and Mac focus — 23 October 2026

The user temporarily turned off the iPhone and requested Mac focus. All further
phone operations are paused. The selected next candidate is the public pinned
Mistral-7B-Instruct-v0.3 four-bit full-attention snapshot; other Mistral variants
and sliding attention remain unsupported. The [reference preparation](mlx-mistral-oracle.md)
adds bounded independent reconstruction, an untied-head graph and original
instruction-template fixtures before native target work.

Astra requested actual Matrix/projection composition across the1,024-row
boundary and adapted reference-verifier mutation coverage. All nine portable
tests now pass, with the numerical policy unchanged. The exporter observes and
checks its lifetime footprint after each case and generation/scoring pair; this
is an acceptance check, not an OS allocation limit. Chat-template replay,
syntax and whitespace checks also pass. The [tools receipt](../benchmarks/results/2026-10-23-mistral-reference-tools-v1.json)
records exact tested inputs. Full numerical export, target support, app routing,
real download/chat and performance remain separate work.

The evidence retains an ignored inspection-directory collision, an incomplete
serial acquisition and subsequent exact owned-process absence checks. No partial
checkpoint was published. The new range acquisition has its own directory,
validated HTTP ranges and complete weight-hash admission before publication.

### Bounded multi-gigabyte download prerequisite — 23 October 2026

Shared explicit64-bit admission limits now allow at most4GiB per file and6GiB
per snapshot, preparing the existing streaming core for the selected Mac
candidate. Catalog support and numerical budgets are unchanged. The two-snapshot
storage reserve and per-stream byte bounds remain checked, along with existing
hash, ownership, cancellation and publication rules.

Astra cleared the arithmetic and boundary tests. Strict Swift compilation, the
complete core checks and iOS SDK type checking pass without contacting the
phone. The [limit receipt](../benchmarks/results/2026-10-23-model-hub-large-limits-v1.json)
distinguishes large-size request admission from an actual large payload transfer;
that qualification and Mistral engine/chat support remain pending.

### Complete immutable Mistral acquisition — 23 October 2026

The [pinned Mistral checkpoint](mlx-mistral-checkpoint.md) is now admitted as data:
all 244 disjoint ranges reconstruct the exact 4,077,480,215-byte weight hash, all
eight files match the immutable inventory, 252 owned commands finish cleanly and
all 252 process groups are observed absent. Astra independently verifies the
complete file/diagnostic/range closure before CPU-reference admission. The
[receipt](../benchmarks/results/2026-10-23-mistral-checkpoint-v1.json) preserves the
incomplete serial acquisition and ignored inspection collision; neither receives
qualification credit.

The CPU reference export has started with the accepted acquisition SHA and
unchanged reviewed numerical policy. The separate app-core URLSession transfer
is still running. No Mistral native execution, app catalog support, physical fit,
performance or quality is claimed by acquisition alone. The record's heading
now reflects M7's completed amended iPhone14Pro scope, and the Apple guide names
the current Qwen-capable native package. Phone work stays paused.

### Frozen independent Mistral numerical reference — 23 October 2026

The streamed CPU exporter completes all **349 arrays / 35,025,920 bytes**, with
matching four-step FP32/FP64 selections and full vocabulary distributions for
decode and scoring. The [oracle report](mlx-mistral-oracle.md) records the
externally frozen manifest/array hashes before native Mistral target edits.
The unchanged arithmetic/prefix policy yields maximum absolute allowance 0.5
and logits allowance 0.08046722412109375, plus relative 1e-3. These measured
envelopes grant no held-out quality waiver.

Both owned export/stdlib verification commands pass with unchanged source,
package and data endpoints and clean process completion. Logical admission is
5,486,766,359 bytes, counting the complete clean source mapping; lifetime
charged physical peak is 293,752,696 bytes. The charged counter can omit clean
file pages, so it does not replace the logical budget or qualify mobile fit.
The [receipt](../benchmarks/results/2026-10-23-mistral-real-oracle-v1.json) binds
this evidence. Native decoder, Chat/catalog integration and actual app transfer
qualification remain separate. Phone work remains paused.

### Selected native Mistral decoder on Mac — 23 October 2026

The [strict Rust/Metal decoder](mlx-mistral-metal.md) now executes the pinned
Mistral 7B four-bit full-attention snapshot. Shared static bias-free dispatch
preserves resident packed weights, separate KV state and existing Qwen-only API
contracts. Explicit null sliding-window admission rejects omitted windows and
Llama-only metadata. Reference attention is qualified; Online remains rejected.

Final formatting, strict Clippy, CPU/Metal baselines, release budget and native C
checks pass. Real M1 Pro GPU checks pass 816 exposed snapshots / 19,302,400 values,
four complete greedy distributions, three scoring prefixes, eight selections
and a separate 512-capacity state test. Frozen gates are unchanged. Maximum
logical session admission is 4,267,272,802 bytes; physical peak, held-out quality,
performance and long-context numerical qualification remain open. The
[receipt](../benchmarks/results/2026-10-23-mistral-metal-v1.json) retains the
pre-fix baseline and final source evidence. Chat/catalog integration continues
separately. Phone operations remain paused at the user's request.

### Native Mistral instruction text on Mac — 23 October 2026

The [Mistral text profile](app-chat-profile.md#mistral-text-profile--23-october-2026)
adds the exact pinned default instruction template and EOS-only stopping to the
native bridge. Its bounded tokenizer configuration and separate 64 MiB text
reservation preserve the numerical boundary. All 771 tokenizer control literals
are rejected after JSON decoding. SmolLM2 and Qwen semantics remain unchanged.

All six baseline/tokenizer/reference commands pass. A retained sandboxed trial
fails GPU model loading; the unchanged code passes all four real/native text
checks with Mac GPU access, including Mistral greeting, Qwen's unchanged token-0
reply, ownership and hidden stops. Independent Python decoding agrees. The
[receipt](../benchmarks/results/2026-10-23-mistral-app-chat-profile-v1.json) keeps
those scopes distinct. App catalog/package integration continues next; phone
operations stay paused and no new speed or physical-memory claim is made.

### Bounded retries for large Mac downloads — 23 October 2026

A real near-complete Mistral full-body transfer failed on a lost connection;
its incomplete staging was removed and no snapshot was published. The
[range transport](model-hub.md#bounded-large-file-transport--23-october-2026)
now bounds large streams to four 16 MiB chunks with strict response identities,
three transient-only attempts, joined cancellation and a two-hour operation
deadline. Small-file behavior and final whole-file SHA admission remain intact.

Astra requested fatal close-error composition and canonical length validation;
both now have deterministic regressions. Final strict Swift/core/SDK checks pass.
The [receipt](../benchmarks/results/2026-10-23-model-hub-ranges-v1.json) preserves
the failed full-body and sandboxed trials plus intermediate source snapshots.
The fresh actual range transfer is separate, as are numerical fit and app Chat.
Phone operations remain paused.

### Mistral Models and offline Chat on Mac — 23 October 2026

The [app catalog](mlx-mistral-app.md) adds exact pinned Mistral downloads/search
selection, vocabulary and separate numerical/text/process admission. The chat
header now names the loaded model. Older saved profiles and Smol/Qwen behavior
remain compatible. A fresh three-slice package and final shared SDK, integration,
core and profile checks pass without contacting the phone.

Real local-checkpoint execution passes multi-turn text, context rejection,
cancellation, suspend/replay and the friendly offline controller. Qwen and Smol
saved-cache regressions pass with the new package. Guarded independent Python
decoding agrees with every reply. The single completed Mistral process observes
a 4,226,780,712-byte charged lifetime peak; logical accounting, clean mappings,
sustained behavior and performance remain separate.

Two retained trials exposed stale admission before a long verification await.
Refresh now samples afterward and clears stale allowed state on observation
failure, with tests in both memory directions. Budgets stay unchanged. The
[receipt](../benchmarks/results/2026-10-23-mistral-model-hub-app-v1.json) distinguishes
these failures and local seed from fresh network transfer. Actual range-download
qualification continues separately. All physical phone work remains paused.

### Preserve range cleanup failures during cancellation — 23 October 2026

A final review found that cancellation or a deadline could hide a simultaneous
close/removal failure even though tasks were joined and publication was already
blocked. The [narrow fix](model-hub.md#preserve-cleanup-errors-during-interruption--23-october-2026)
retains those causes through URLSession completion, chunk retry, both task groups
and assembly cleanup. Every cancelled sibling is drained explicitly.

New slow-protocol cases pass active cancellation plus close failure and deadline
plus close failure, preserving interruption context without retry or later
writes. Final strict Swift/core/SDK checks pass. Astra also requested typed
removal errors and a bounded deadline fixture; both are included. The
[receipt](../benchmarks/results/2026-10-23-model-hub-cleanup-errors-v1.json) records
historical failed/intermediate inputs. Completed transfer and numerical paths
are unchanged, and no new phone or performance qualification is claimed.

### Bounded Mistral raw-ID observations on M1 Pro — 23 October 2026

Two fresh processes complete the unchanged Mac observation protocol: three
warmups each, 82 measured requests each, and 602.477/602.576 measured native
seconds. The [fixed four-input/32-output probe](mlx-mistral-mac-observations.md)
observes first-token medians 536.028/535.985 ms and decode medians 4.551/4.549 IDs/s.
Charged lifetime peaks are 4,119,186,864/4,119,039,408 bytes under the existing
4 GiB trial gate. Frozen four-ID oracles, full-sequence repeatability, clocks,
state, ownership, source endpoints and complete artifact proofs pass.

This adds measured Mac data without changing a kernel or admission gate. A
concurrent public range download is an explicit confounder, and the single AC
query is a pre-run observation only. The
[receipt](../benchmarks/results/2026-10-23-mistral-mac-observations-v1.json) makes
no full performance, text/UI, held-out quality or phone qualification claim.

### Retry metadata and smaller downloads safely — 23 October 2026

The actual ranged Mistral trial failed with timeout code -1001; incomplete
staging was discarded and no final snapshot remained. Its old diagnostic does
not identify the failed file/range or final byte count. Inspection exposed a
separate asymmetry: metadata/smaller files had one attempt versus three for
ranges. The [whole-file retry change](model-hub.md#whole-file-retries-and-safe-failure-context--23-october-2026)
adds three transient-only attempts under one deadline, closed exclusive sink
replacement, fatal cleanup errors and joined interruption. Network diagnostics
now exclude URL-bearing NSError metadata while retaining code, attempt, range,
actual delivered byte count and monotonic elapsed time. Chunk tuning and
cross-launch persistence are unchanged.

Astra caught a false-pass path in a new nontransient test and requested a real
URL-bearing injected error for privacy coverage; both were fixed. Two diagnostic
fixture trials assumed two queued bytes must reach the delegate before immediate
failure; URLSession reported zero. The test now admits exactly zero or two,
without altering production size/hash gates. Final strict Swift fixture checks
and SDK type checking pass; the earlier sandbox storage failure and all fixture
trials remain in the [receipt](../benchmarks/results/2026-10-23-model-hub-whole-retries-v1.json).
No numerical kernels changed. Fresh Mistral network completion remains separate;
physical phone work stays paused.

### Fresh Mistral network snapshot and reopening on Mac — 23 October 2026

The [actual new Mac download](mlx-mistral-hub-download.md) now passes public
search, pinned catalog admission, all six files and independent full-file SHA256
checks, totaling 4,079,633,318 bytes. The download reports no reuse; ordinary
reopening of that actual cache reports reuse and passes native multi-turn/state,
cancellation/suspension/release and offline friendly-controller Chat. All four
reply ID/text sequences match the previously independently decoded local test.
The weight transfer observes 3,667.422 seconds on this link; it is not an engine
performance result or promised ETA. The first full-body loss and ranged timeout
remain retained failures, without inferring their cause from a later success.
The frozen app source predates the separate Mac window add-on. Physical phone,
visible GUI, wider checkpoint, quality and full performance scopes remain
separate.

### Add native Mac Chat and Models — 23 October 2026

The [Mac add-on](mac-app.md) provides a separate native app entry point while
sharing Chat, Models and the original Expert form with the iPhone target. It
adds native folder/Finder access, Command-Return Send/Stop, stable local storage,
and joined close/quit with draft recovery and safe worker recreation. Review
fixes cover close/reopen races, save failures, bounded build diagnostics and
receipt/header admission. The numerical kernels and native package are unchanged.

Final strict coordinator/real-model checks, six builder negative tests and both
local iOS SDK app builds pass. The visible Mac trial passes fresh Smol download,
independent hashes, all three real-model greetings, Mistral Stop, Expert Tiny
`47,47`, and draft-preserving close/reopen/quit. Its stderr is empty. Earlier
watchdog/automation and SwiftUI-warning trials remain recorded without attributing
the warning reduction to one change. Astra reviewed the code and evidence.
The [receipt](../benchmarks/results/2026-10-23-macos-window-v1.json) separates
visible observations, saved-state proofs and automated checks. Phone work stays
paused; no new engine-performance or wider compatibility claim is made.

A staged whitespace check then exposed one extra EOF blank line in the new Expert
file. A formatting-only follow-up removes it; the Mac receipt maps the original
executed source to its immutable build copy. Numerical and UI behavior are unchanged.

### Recover rejected Chat turns and preserve model identity — 23 October 2026

The [controller recovery fixes](app-recovery.md) restore drafts after validation
or known-unstarted lifecycle cancellation, without overwriting later edits even
when intentionally cleared. Partial generation remains a terminal history turn.
New/delete-last keeps a live model usable, unrelated folders are rejected before
unloading, and delayed verified Open cannot target another conversation. Actual
worker admission precedes selection changes. No numerical kernel, context gate,
model identity gate or persisted history schema changed.

A real-model baseline reproduced the forbidden-literal failure. Exact-source
regressions cover correction/retry, context failure, six gated lifecycle cases,
actual partial cancellation, history/model identity and stale verification.
Strict existing controller suites and iOS SDK builds pass; Astra reviews the
code and evidence before commit. Physical button/layout fixes and the final
phone audit are recorded separately when completed.

### Isolate Expert buttons and repair app presentation — 23 October 2026

The ongoing [iPhone function/button audit](iphone-app-audit.md) reproduced a Form
row bug: Tiny fixture also fired Documents/Model. Explicit per-action styling now
isolates all three multi-button Expert rows, with vertical accessibility layout.
The friendly view keeps qualified model controls above search results, exposes
folder errors, treats Files cancellation normally, owns deletion confirmation in
the visible History sheet, and follows latest chat activity while respecting
manual scroll-back/gestures. Initial restored scroll position and full accessible
model headings are handled explicitly. Numerical code is unchanged.

SDK and Mac build checks and Astra review are recorded in the
[UI receipt](../benchmarks/results/2026-10-23-iphone-ui-controls-v1.json).
The corrected signed app was installed in place. Physical checks confirm isolated
Expert actions and lifecycle transitions, public search, Qwen download/load/chat,
active Stop with partial text, long-response follow/scroll-back, Settings choices,
Files cancellation and visible History confirmation. After action-time approval,
only the new interrupted synthetic chat was deleted. The detailed checklist
retains unexecuted controls; the full sweep remains open after another Mac lock.
Existing owner chats and model caches were neither deleted nor exported.

### Plan modular engine expansion and freeze Qwen3 reference — 23 October 2026

The [active queue](engine-expansion-plan.md) orders graph mechanisms, measured
optimizations and general Hugging Face admission/download/chat. Architecture
modules own their config, weight mapping and graph rules; shared operators remain
separate from tokenizer/UI/download code. Capability, device fit and physical
qualification remain distinct. Existing pinned profiles stay supported while
compatibility grows; the complete architecture program remains open.

The [first Qwen3 task](mlx-qwen3-oracle.md) freezes an independent tiny graph with
hidden32/query64/KV32 and per-head Q/K RMSNorm before RoPE. Both tied/untied outputs,
mixed packed/floating weights, offset/causal/capacity cases, actual upstream
CPU/GPU/cached logits and wrong-norm witnesses pass. The fixture contains 891
arrays; all 15 portable tests pass. Native Rust/Metal Qwen3 execution, real models,
chat/thinking semantics, quality/performance and iPhone testing remain pending.
Numerical target sources and protected release binaries are unchanged. Astra
reviews the code and completed evidence before commit.

### Add modular Qwen3 dense scalar execution — 24 October 2026

The [native scalar task](mlx-qwen3-scalar.md) adds a separate Qwen3 config module
and reuses the checked dense graph with explicit per-head Q/K normalization.
Query width can differ from hidden width; projections, scratch, traces and O
weights use the correct dimensions. Norm weights are immutable, allocated
capacities are counted, and normalization reuses rotary scratch. No numerical
inner-loop architecture dispatch or new steady-state allocation is introduced.
Legacy Qwen entrypoints and Toklet Q4 geometry stay constrained; Qwen3 Metal
rejects explicitly until independently implemented and qualified.

All 430 frozen stage comparisons pass. Nine targeted Metal-feature tests cover
malformed configs/tensors, exact admission, over-reserved norm storage, zero
allocation decode, cache offsets/reuse/reset/overflow and pending/edit/cancel
recovery. Rust1.90 formatting, strict Clippy in both feature configurations,
161 CPU/179 Metal-feature ordinary tests and five explicit existing Qwen/Llama
GPU regressions pass. Protected release binaries, shaders, Swift and installed
apps are unchanged. Astra reviews the completed source/evidence before commit.
Real Qwen3 models, resident Metal, thinking/chat, quality/performance and phone
qualification remain open in the active queue.

### Run resident tiny Qwen3 on Metal — 24 October 2026

The [resident task](mlx-qwen3-metal.md) composes existing Metal operators with
independent hidden/query layouts and immutable per-head Q/K norm vectors shared
across the respective head types. It reuses rotary scratch and submits one
completed command per chunk. Reference attention is explicit; profiling admits
the additional norm encoders. Source review caught and fixed profile capacity.
No shader or numerical-policy change was needed.

The Mac passes 1,170 stage snapshots/204,288 values, eight full distributions,
sixteen exact selections, four profiled executions and exact session/diagnostic
budget boundaries. Both Qwen3 output variants pass raw CLI/native bridge routing,
stops, scoring, retained ownership and cancel/pending state. The three GPU tests
also pass in release. Rust1.90 formatting, both strict Clippy configurations,
161 CPU/179 Metal-feature passes and five existing Qwen/Llama GPU regressions
pass. Legacy Half/Q4 release allocation gates pass unchanged.

The extra legacy Half debug allocation check fails identically on the parent
and current source (6,320/20steps); zero-allocation release checks pass. This
baseline result and an initial incomplete baseline snapshot are retained, without
claiming a traced root cause or weakening the assertion. Astra reviews the final
code and evidence before commit. Protected binaries, Swift and installed apps
are unchanged. Real Qwen3 models, thinking/chat, quality/performance, iPhone and
general Hugging Face admission remain open.

### Acquire and inspect the real Qwen3 candidate — 24 October 2026

The [pinned eight-bit checkpoint](mlx-qwen3-checkpoint.md) contains ten files,
649,377,414 bytes at revision `11de96878523501bcaa86104e3c186de07ff9068`. All 48
bounded owned transfers pass immutable identities and exclusive publication.
Data-only inspection validates 704 tensors/197 packed matrices/113 norm vectors,
contiguous exact payload coverage and index equality. Unlike earlier selected
checkpoints, all 507 floating tensors use BF16. The graph has independent
hidden1024/query2048 widths and 56 per-head Q/K norm vectors of width128.

Nine acquisition and five inspector portable methods pass. Astra's review
required retaining all diagnostic streams when descriptor cleanup also fails,
and authenticating the same weight descriptor consumed by the header reader;
both have deterministic regressions. No repository code or template executes.
Numerical sources, old receipts and protected CPU binaries stay unchanged; prior engine
checks are inherited on identical numerical/Swift sources. Retain the acquired
data for a separately frozen real numerical reference. Native real-model
geometry, tokenizer/thinking/chat, device fit, quality/performance, Mac/iPhone
trials and broader Hugging Face product admission remain open.

### Freeze the independent real Qwen3 reference — 24 October 2026

The [real-model CPU oracle](mlx-qwen3-real-oracle.md) reconstructs eight-bit
packed weights with exact BF16 metadata bits, preserving an original FP32 tied
output table separately from Half gathers, norms and activations. A separate
Qwen3 graph composes existing primitives with independent query width and
per-head Q/K normalization before RoPE. Its unequal-width portable graph matches
the ordered independent implementation. Bounded owned stripes fix Astra's
mapped-view failure-path finding; a malformed-reader regression retains the
original error and proves cleanup.

All 50 portable methods, actual export and external-SHA stdlib verification
pass. Six cases, selected layers 0/1/27, four greedy and three scoring distributions
freeze 385 arrays/41,115,648 bytes before any target change. Both arithmetic paths
select `[1096,374,264,3491]` on the declared FP64 teacher history. Unchanged
pre-target policy yields maximum stage allowance 2.0 and logits allowance
0.07747364044189453. Logical reference admission 2,597,950,130 bytes and observed
completed-process lifetime physical peak 866,552,040 bytes stay below 6 GiB.

One initial portable run fails because an existing test assumes no real cache;
the isolated test fixture fixes that assumption while preserving production
collision rejection. Failed logs and original bytes remain retained. Twelve
new methods, seven producer sources and 445 package files are bound. The one raw
text tokenization witness does not qualify templates or thinking/chat. Numerical
engine/Swift sources and protected binaries are unchanged. Native real Qwen3,
quality/performance, app/device fit and wider architecture/discovery work remain
open.

### Qualify real Qwen3 Reference execution on Mac — 24 October 2026

The [selected real-model task](mlx-qwen3-real-metal.md) widens exact Qwen3
geometry admission while reusing the existing graph, kernels and BF16 affine
support. Immutable packed weights/QK norms and tied ownership remain resident;
sessions own KV and scratch. No shader, numerical-policy or default routing
change is needed. Unqualified geometry and Online attention reject explicitly.

On M1 Pro, all 912 exposed snapshots/14,276,352 values pass the unchanged
reference gates. Four greedy and three scoring distributions pass, with eight
exact full/token-only selections and maximum logit error 0.03388547897338867.
A separate actual 512-capacity/chunk128 witness discards/resumes its pending
input tail and matches a fresh full distribution exactly. This is not generation
cancellation or an independent long-context numerical oracle. Exact session and
diagnostic limit-minus-one checks pass on fresh sessions for both readbacks.
Resident weights total 633,366,062 bytes; maximum short diagnostic and 512-state
logical admissions are 747,182,964 and 853,205,778 bytes, without a process-peak
or phone-fit claim.

Rust 1.90 formatting, both strict all-target Clippy configurations, 162 CPU and
181 Metal-feature passes, and all 18 optimized explicit GPU tests pass. The
regressions include tiny Qwen3, Qwen/Llama, legacy Half/Q4 allocation checks,
shared native bridge and CLI. Protected binaries, Swift and shaders are unchanged.
Retain selected real Qwen3 Mac Reference execution. Quality/performance, actual
generation cancellation, tokenizer/thinking/chat, app integration, iPhone and
broader architecture/Hugging Face qualification remain open.

### Verify real Qwen3 generation callback recovery — 24 October 2026

The [tests-only follow-up](mlx-qwen3-real-generation.md) exercises the public
generation API with the actual selected checkpoint. Both readbacks preserve
sampled-versus-committed state after callback cancellation and output failure,
resume exactly like uninterrupted same-recipe generation, hide stop IDs from
callbacks and preserve evaluated output after invalid requests. Full-logit
recovery distributions match exactly and pass the unchanged frozen gates;
token-only requires an available correct selection and no full logits.

The optimized GPU witness, formatting, both strict Clippy configurations,
162 CPU and 181 Metal-feature tests pass. Astra tightened two optional-token
comparisons to require a real selected ID; initial checks and intermediate bytes
remain retained alongside final checks. Production sources and existing real
GPU test bodies are unchanged, so prior numerical/session evidence is inherited.
Retain callback cancellation and recovery for selected real Qwen3 on Mac.
Prefill/mid-GPU and physical cancellation, quality/performance, app/thinking/chat,
iPhone and wider architecture/Hugging Face work remain open.

### Share local MLX compatibility reporting — 24 October 2026

The [H1 foundation](mlx-compatibility.md) provides an engine-owned metadata report
for implemented Qwen2, Qwen3, Llama and Mistral graphs. The same captured bounded
configuration, tensor admission, scalar expansion estimate and exact Metal
geometry rules drive inspection and loading. No second architecture parser or
numerical-loop abstraction is added. Compatible metadata, load admission,
backend availability and pending payload/device/session/text/qualification
checks stay separate. Affine metadata dtype is explicit; packed words remain U32.

Astra required private geometry construction after validation and an overflow
regression; both are included. Offline family/budget/malformed/nonfinite probes
and the real Qwen3 metadata witness verify report boundaries. Final captured
checks pass: formatting, both strict Clippy configurations, 171 CPU tests,
191 Metal-feature tests, 19 optimized GPU regressions and one real metadata
probe. The task receipt binds raw evidence and prior changed source versions.
Retain this shared foundation. CLI/native/app consumers, actual Metal resource
planning and receipt-based qualification remain open, alongside general Hub
downloads/Chat and the remaining graph inventory. No new architecture, app,
quality, performance or phone claim follows from inspection.

### Expose compatibility to the CLI and native callers — 24 October 2026

The [H1 consumer follow-up](mlx-compatibility-consumers.md) adds an offline JSON
inspection command and immutable, thread-owned C report handles. Dedicated
versioned inspection options preserve the exact numerical budget. A bounded
64 KiB writer serializes once; size queries, repeated copies, short buffers and
source mutation preserve a stable report. JSON ownership has its own explicit
logical allowance. Existing text queries reuse the unchanged shared copy helper.

Astra requested explicit closed-pipe exit semantics and native/text regression
checks; both are included. Formatting, both strict Clippy configurations, 178 CPU
and 198 Metal-feature tests pass. A fresh static library links/runs the real C
caller, all 23 exports match the header, the local text bridge passes, and actual
real Qwen3 CLI metadata is correct. The initial environment-only C link failure
and successful explicit-temporary-directory retry remain captured. Retain these
consumers. Swift app integration, actual runtime memory planning, qualification
receipts, general Hub/Chat and the remaining engine inventory stay open.

### Integrate compatibility on the app worker — 24 October 2026

The [Swift adapter](mlx-compatibility-swift.md) inspects MLX metadata on the
existing inference thread before actual loading. It captures an immutable report,
checks the exact request/schema and reason/admission/backend tuple, and releases
native report ownership before numerical work. Actual payload/device/session
loading and independent pinned text profiles remain authoritative. No numerical
source or token/context limit changes. The extra application allowance is stated
as a requirement, separately from numerical admission.

Astra found contradictory-wire acceptance and a queued load→Suspend stale-summary
race. Exact tuple checks and clearing on Suspend fix them. Independent mutations
and dequeue-gated idle-state assertions before Close cover both regressions.
The strict Swift wrapper, C boundary, portable checker and existing worker checks
pass: 89 portable methods, strict default/Metal Swift wrappers, existing CPU/
Half-Metal/MLX workers, C caller and strict iOS/simulator modules. Three real Mac
Chat profiles also pass short cancellation/replay/context/owner regressions from
immutable snapshots. The fresh native package builds/links on all three Apple
targets. The
initial old-system-Python failure and explicit-interpreter retry are retained.
Retain this adapter; runtime planning, qualification receipts, general Hub/Chat
and the remaining engine inventory stay open. Parallel mobile UI work owns its
presentation changes and evidence separately.

### Extend bounded Chat output to 256 tokens — 24 October 2026

The [Chat output extension](chat-output-limit.md) widens codec/native/worker
visible output to256 while retaining the 512-token total context, 64 KiB text
bound, existing templates/stops and Swift raw/Expert128-token command limit.
Output/timing storage is bounded and separate from numerical admission. The
worker publishes the actual C emitted count and clears it for unstarted requests.

Astra required full-prefix preservation checks rather than count-only witnesses.
The real Qwen run consumes the independently frozen65-ID prompt and emits256,
with321 history/320 committed/one pending. Cancellation at128/129, exact fresh
replay and rejection without changing saved history all pass. Portable256/257
decode/buffer tests, formatting, both strict Clippy configurations,180 CPU and
200 Metal-feature tests pass. The explicit Swift/C/worker/platform checks pass;
fresh Mac/iOS/simulator libraries retain the same23 exports. All three short
profile regressions match the previous replies and flags exactly. Retain this
bounded extension. UI options/persistence belong to the parallel UI task; phone
qualification at256, larger context and wider engine/Hub scope remain open.

## 24 October 2026 — mobile interface stability and presentation

The UI task investigated model/search controls blocked by automatic full-file
verification, the 64-token initial reply cap, and missing Expert help/code
formatting. Background checks are now replaceable and cancellation joins them
without publishing stale choices. Model loading is explicit; reopening a live
model preserves native and UI history. Supported search matches are actionable.

The parallel engine commit `0b8945b` supplies the bounded 256-token text bridge.
Chat offers 32/64/128/256 with a 128-token initial default, preserved saved
settings, visible limit stops and the existing 512-position context preflight.
Expert keeps its raw 128-ID bound. Inline Markdown, fenced code, lexical
highlighting/Copy, action descriptions, examples and readable stop states add
presentation without changing tokenizer input or numerical kernels.

Strict Swift 6/concurrency/platform checks, stale/cancelled verification/Open
regressions and real SmolLM2 load/Send/idempotent Open/invalid-folder checks pass.
Full Release iOS and simulator apps pass against the matching package; ordinary
Rust formatting/Clippy/CPU checks also passed, with final engine checks recorded
separately. Light/dark highlighted-text samples were inspected; ImageRenderer
cannot verify native scroll/button surfaces. Live Device Hub inspection stalled,
so full tap-through and physical-iPhone behavior remain unqualified. No latency,
quality or memory-limit improvement is claimed. Retain the functional fixes.
See the [interface report](mobile-interface-fixes.md) and
[bound receipt](../benchmarks/results/2026-10-24-mobile-interface-v1.json).

## 24 October 2026 — shared runtime Metal resource planning

Hypothesis: one concrete layout/admission plan can serve both constructors and
cold compatibility inspection without parallel formulas or changing numerical
execution. The new Rust-only `metal::planning::inspect_mlx` shares the loader's
MPS-padded floating/affine layouts and the session's exact KV/scratch/history/
output/table totals. Constructors consume those plans directly. Every numerical
buffer uses the same device/index limit helper as allocation. Source load,
resident load and session admission remain overlapping, separately named totals.

Before extraction, 42 M1 Pro actual allocation reports were frozen across six
tiny artifacts, chunks1/3/32, tied/untied output, mixed storage, independent query
width, both readbacks and qualified Qwen2 Online. New plans and actual owners
match every report. Exact/minus-one budgets, offsets, per-buffer bounds and
overflow checks pass. Real Qwen3 retains633,366,062 resident bytes; capacity32/
chunk3 admissions are707,649,798 FullLogits and706,434,350 GreedyToken, matching
actual owners. Its independent final-row numerical/diagnostic budget gate passes.

Formatting, strict Clippy,180 CPU and203 Metal-feature tests pass; both iOS
library targets compile. Sixteen explicit release GPU tests pass, including
Half/Q4/MLX numerical/state and warmed-allocation regressions. Initial test-only
Clippy comparisons were corrected without loosening policy. An accidental debug
GPU run reached the Half zero-allocation assertion with4640 allocations and
failed; its complete evidence is retained. The established release invocation
passes that unchanged gate; allocation evidence is scoped to release.

Retain the shared planner. Driver/pipeline setup still allocates, the64MiB
allowance remains an estimate, and a fit flag reserves no memory or payload
validity. No latency, quality, free-memory or physical-iPhone qualification is
claimed. The runtime CLI/native/app consumer and qualification receipts remain
open. See [resource planning](metal-resource-planning.md) and its
[receipt](../benchmarks/results/2026-10-24-metal-resource-planning-v1.json).

## 24 October 2026 — runtime planning CLI and native consumers

The shared Metal plan is now available through explicit `toklet plan` and the
additive ABI-v1 `toklet_mlx_plan_create`. The56-byte C options record selects all
session geometry/modes and separate exact budgets, without bridge deduction.
Version/size/reserved/mode/range validation precedes runtime initialization.
Budget-fit failures return immutable reports; unavailable builds or unqualified
modes reject. The existing metadata command/API keep their CPU-only behavior.
Both report kinds share bounded JSON copying, owner/thread checks and release.

Formatting, strict Clippy,184 CPU/205 Metal-feature tests pass (14/105 ordinary
ignored). Explicit release checks compare20 native and4 CLI reports with the
direct planner; fit=false, bad modes, wrong-thread calls, changed-source snapshots
and broken pipes pass. A fresh native package builds all three Apple slices with
24 symbols. New C callers link on all targets and run on Mac; strict Swift6
imports/layouts pass on Mac/iOS/simulator. Existing C/Swift metadata callers and
21 wire-mutation/three queued-cleanup checks pass. Real Qwen3 CLI totals exactly
match the preceding resource receipt for both readbacks.

Initial test-literal Clippy and CLI position-type compiler failures are retained;
explicit byte-array/type corrections pass the unchanged policies. No numerical
kernel, shared planner, schema of metadata reports or app load path changes.
Retain these consumers. Runtime initialization still allocates driver/pipeline
resources, fit flags reserve nothing, and loading must revalidate. No physical
phone, task-quality or timing result follows. Swift/app integration and exact
qualification receipts remain open. See [consumer contracts](metal-resource-consumers.md)
and the [receipt](../benchmarks/results/2026-10-24-metal-resource-consumers-v1.json).

## 24 October 2026 — native factory resource policy

App preflight no longer needs to reconstruct the engine's attention choice:
explicit runtime attention2 selects the native factory policy after graph and
geometry admission. The actual factory and inspector share one exhaustive helper;
Qwen2/non-MLX Online and selected Llama/Mistral/Qwen3 Reference behavior is
unchanged. Explicit modes0/1 and runtime report schema keep their behavior.
New graph enums require an explicit policy decision.

The existing20,480-byte native model/session-owner allowance is now a shared
Rust/header constant. Checked subtraction still rejects whole budgets at/below
that value; allowance+1 yields one numerical byte. Runtime inspection takes exact
numerical budgets, so callers can plan the actual factory budget without a
second hidden deduction. This is logical accounting, not a memory reservation.

Formatting, strict Clippy,185 CPU/207 Metal-feature tests pass (14/106 ignored).
Five explicit release GPU tests pass: four-family plan/factory comparisons with
frozen parent memory and exact/minus-one load/session admission, existing20
explicit native reports,42 frozen plans/budget tests and4 CLI direct reports.
The fresh three-slice package retains24 exports. C callers link all SDKs and run
four families on Mac, verifying plan+allowance equals actual session admission.
Existing Swift6/importer and metadata21-mutation/three-cleanup checks pass; both
complete real Qwen3 CLI reports remain equal to the preceding receipt.

Retain the shared policy and allowance. No kernel, numerical memory formula,
text profile, qualification label or phone behavior changes. Swift worker runtime
preflight and qualification-receipt evaluation remain open. See
[native resource policy](native-resource-policy.md) and the
[receipt](../benchmarks/results/2026-10-24-native-resource-policy-v1.json).

## 24 October 2026 — Swift app runtime preflight

The app worker now consumes the engine's bounded runtime plan before MLX
numerical allocation. It selects native policy2 and subtracts the shared owner
allowance once for inspection, while passing the original whole budget to the
actual factory. The Swift adapter validates request/modes, checked memory sums
and independent fit flags; it releases the C report on its creating thread.
Consistent non-fitting reports remain visible and reject before numerical owners.

New loads and all cleanup paths clear metadata/runtime summaries. A cold
post-inspection check prevents a superseded load from publishing or allocating
after queued cleanup; the actual uncancellable C load still revalidates source
and device state. Two-summary logical application allowance is1,311,232 bytes,
not a reservation or process-peak measurement. Ten native-check helper dependency
lists and the app project include the new adapter.

Strict Swift6 default/Metal checks pass25 wire mutations, budget boundaries,
plan/factory admission, a runtime-only budget rejection, three queued cleanups
and source mutation after inspection. Existing21-mutation, legacy CPU/Half/MLX
and89 helper regressions pass. Complete real SmolLM2/Qwen/Mistral Chat JSON and
the prior Qwen256-output/cancellation/replay result remain equal. Both SDK modules
and full Release iOS/simulator/Mac app builds pass. The first test compile's
mixed-integer array inference error is retained; an explicit UInt64 array fixes
it without changing policy. Rust/shader/header bytes match the accepted native
policy package and inherit its185 CPU/207 Metal-feature evidence; a fresh
default archive independently checks unavailable-Metal behavior.

Retain the preflight. No larger output/context, new model eligibility, numerical,
quality, timing or physical-phone claim follows. Qualification-receipt evaluation
and broader acquisition/chat remain open. See [the contract](swift-runtime-preflight.md)
and [bound receipt](../benchmarks/results/2026-10-24-swift-runtime-preflight-v1.json).

## 24 October2026 — single-reply128K plan and decoding witnesses

The user clarified131,072 output tokens in one Chat/Expert reply, with prompt
positions additional. The [engine plan](long-reply-engine-plan.md) now separates
independent text fixtures, bounded streaming, native capabilities, a concrete
long-context graph, integration and device qualification. Existing E1–E6/H1–H4
requirements remain open. Current executable limits are unchanged.

Before target streaming code,33 complete-prefix text cases and eight exact
131,072-ID synthetic cases are frozen from three pinned original tokenizers.
Compact decode-only fixtures preserve every checked result. Mistral's invalid
fallback byte can rewrite an entire preceding ASCII byte run; split Unicode and
literal replacement characters also require explicit state/terminal behavior.
The independent Python producer imports no Toklet implementation. Export/verify
reproduction and eight reference checks pass with all source/runtime identities
recorded. Exploratory assumptions about0xFF availability and single-ID replacement
characters were corrected before freezing, without changing expected decoder
semantics. Retain the oracle; no numerical generation or128K app/device claim.
See [the contract](chat-stream-oracle.md) and
[receipt](../benchmarks/results/2026-10-24-chat-stream-oracle-v1.json).

## 24 October2026 — physical model-switching fix

The parallel mobile task committed8a5ecff after Astra review and checks. Models
now offers direct Use model/Download and use actions with an explicit Active
label. Unchanged snapshots no longer publish at every poll: a separate worker
revision counter avoids retaining old snapshot arrays while preserving callback
and cleanup updates. Presence-only inventory avoids automatic full-model hashing;
Use still verifies every file. Replacement admission checks fresh available memory
after old-owner release, with initial-load/download gates preserved. Mac Finder
recovery remains reachable.

Strict Swift6/controller/callback/revision, real Smol→Qwen→Smol post-release
admission and both SDK/Release builds pass. Physical iPhone14Pro checks repeat
Smol↔Qwen with actual replies and no earlier60-second animation-idle wait, select
256-token Chat replies, inspect highlighted code/visible Copy controls, run
Expert47,47 and return to Chat. Copy visibility/enabled state was checked; no
clipboard-content claim. Owner history bytes stayed unchanged. The checked app
is installed with immutable native policy package2d7; later decoder Rust is not
compiled into it. No broader model,128K, quality, throughput or process-memory
claim follows. See [the mobile report](mobile-model-switching.md) and
[receipt](../benchmarks/results/2026-10-24-mobile-model-switch-v1.json).

## 24 October2026 — bounded Rust response decoding

The [L2 decoder](chat-stream-decoder.md) is checked against frozen witnesses. It emits
new bounded chunks, preserves ByteLevel UTF-8 and whole-token behavior, and holds
Mistral fallback runs until exact text is stable. It never reconstructs a whole
response prefix per token; token lookup still allocates temporary strings.
Failures poison the decoder and preserve counters for successfully returned text.
Current native/app generation and input limits are unchanged.

At the requested stop, formatting/strict Clippy,191 CPU/213 Metal-feature
tests, release checks against all233,585 IDs of the three real tokenizers and the
iOS cross-check passed. The simulator cross-check was stopped safely; raw results
and cleanup records are preserved. After explicit testing resumption, the
unchanged source passed that check in a fresh capture, with all original
source/input/log identities authenticated before inheritance. Astra reviews the
code and evidence before commit. See the
[receipt](../benchmarks/results/2026-10-24-chat-stream-decoder-v1.json).
This does not implement128K numerical/app support. Native event ownership and
integration remain separate tasks. The architecture/Hugging Face plan stays open.

## 24 October2026 — independent recurrent-attention witnesses

The first gated-delta mathematical reference is frozen before its native target:
FP32 inputs, independent FP64 NumPy normalization/gates/recurrence/readout,
explicit grouped-head mapping and immutable initial state. Five tiny geometries
include zero/epsilon-dominated normalization, saturated gates and 128-wide state.
The bounded stdlib format reader requires external manifest/binary identities.

Twenty-three portable and four orchestration tests pass. Fresh export and
verification pass; pinned upstream CPU and GPU each pass all 36 comparisons at
the unchanged source policy. Raw prefix/chunk states, outputs, runtime/source
identities and process ownership are retained. The initial sandbox Metal import
failure and earlier draft-source captures remain separate from the final pass.
Astra reviews the source/evidence before commit. See
[the contract](gated-delta-oracle-preparation.md) and
[receipt](../benchmarks/results/2026-10-24-gated-delta-oracle-v1.json).

Source review corrected overflow, publication/import ownership and failure
retention defects before acceptance. Cooperative interruption has a bounded
parent cleanup deadline; hard-kill loss remains explicit. Masks/vector gates and
the upstream masked-readout discrepancy stay outside this first oracle. No native
recurrence, Qwen3.5 graph, real checkpoint, 128K generation, quality, performance
or mobile qualification is claimed. E1–E6/H1–H4 remain open.

## 24 October2026 — checked scalar recurrent update

The first native gated-delta step consumes normalized Q/K and scalar gates,
preserves contiguous grouped-head mapping and updates a separate caller-owned
candidate matrix with FP32 reductions. Inputs and the previous state are checked
before output writes. Arithmetic failure leaves the previous state intact;
candidate/output must be discarded. No numerical allocation or session commit
occurs in this standalone operator.

The literal accepted reference receipt is authenticated before its fixture
hashes. Six offline regressions cover all frozen step states/outputs, grouping,
every nonfinite input class, invalid gates, malformed/overflow dimensions,
pre-write preservation and correction/candidate/readout failure/replay. The first
check run was stopped to apply Astra's fixture-binding correction; raw snapshots
and interruption are retained. Final corrected checks pass formatting, strict
default/Metal Clippy, 197 CPU/219 Metal-feature tests and both iOS cross-checks.
See [the contract](gated-delta-scalar.md) and
[receipt](../benchmarks/results/2026-10-24-gated-delta-scalar-v1.json).

This is an operator foundation, not Qwen3.5 or long-context generation. Raw Q/K
normalization, gate construction, convolution, hybrid/session commit, GPU state,
model admission, quality, performance and app/device qualification remain open.

## 24 October2026 — recurrent raw Q/K normalization

The separate raw normalizer uses sum-of-squares epsilon and folds one readout
scale into Q. FP64 intermediate squares/sums avoid FP32 overflow, with one FP32
output rounding. Checked dimensions, lengths, finite inputs and epsilon precede
all writes; the operator allocates no buffers and changes no session state.

Three regressions authenticate the frozen receipt and compare raw normalization
and composed recurrent states/output. Hand cases distinguish sum/mean epsilon,
one Q scale, zero/epsilon-dominated heads and maximum finite FP32 input. Astra
strengthened the analytic epsilon witness so an erroneous zero cannot pass the
unchanged tolerance. Invalid inputs preserve both outputs. Formatting, strict
default/Metal Clippy, 200 CPU/222 Metal-feature tests and both iOS cross-checks pass.
See [the contract](gated-delta-normalization.md) and
[receipt](../benchmarks/results/2026-10-24-gated-delta-normalization-v1.json).

The previous recurrent update body is unchanged. Gate construction, convolution,
hybrid model/session execution, GPU state and real-model/long-context/device
qualification remain open. No app capabilities or numerical capacity were raised.

## 24 October2026 — scalar recurrent gate construction

The native gate constructor uses stable FP64 softplus/sigmoid and exponential
decay with one FP32 output rounding. Shape/finiteness admission precedes writes;
nonfinite intermediate rates reject and partial outputs must be discarded. No
storage or numerical state is owned by the operator.

Three new regressions cover frozen gates and the complete raw
normalization→gates→recurrence state/output path, hand values/bias/saturation,
tiny nonzero beta, invalid input classes and intermediate overflow. The accepted
reference and numerical policy are unchanged. Formatting, strict default/Metal
Clippy, 203 CPU/225 Metal-feature tests and both iOS cross-checks pass. Earlier
normalization and update bodies are byte-identical. See
[the contract](gated-delta-gates.md) and
[receipt](../benchmarks/results/2026-10-24-gated-delta-gates-v1.json).

This completes a scalar operator chain, not the hybrid graph. Convolution, gated
output normalization, config/weight mapping, model/session state commit, GPU
execution and real-model/long-context/app qualification remain open. Current
native/app limits and eligibility are unchanged.

## 24 October2026 — mobile monitoring and dialog controls

Commit `abe661e` adds measured process memory/allowance, response timing and
throughput, device-fit recommendations, a local repeated benchmark and confirmed
single/all dialog deletion. Fit recommendations describe supported capacity,
not model quality. Failed dialog saves restore the previous workspace and expose
recovery; injected full-disk failures exercise retained history. Simulator checks
cover cancellation and confirmed deletion, navigation and visible metrics.

Astra accepted the exact final source and
[receipt](../benchmarks/results/2026-10-24-mobile-dashboard-v1.json), including
strict Swift checks, Release iOS/Simulator builds, a passing Simulator UI test and
separate real Mac benchmark/controller checks. This does not qualify a physical
iPhone benchmark or new model/128K capability. The installed phone remains on its
previous app source/native package until a separately admitted update is tested.

## 24 October2026 — independent causal convolution witnesses

The new mathematical convolution/SiLU reference fixes oldest-to-current tap
order, per-channel weights and oldest-to-newest raw-input history. It uses FP64
NumPy math over FP32 inputs with no Toklet arithmetic. Exact bounded artifact
admission and a separately authenticated runtime precede numerical comparison.
Selected pinned upstream assignments invoke actual MLX Conv1d/SiLU with the
explicit converted weight layout; no whole model execution is implied.

Three orchestration, nine mathematical/format/source-selection and three runtime
admission tests pass. CPU and GPU each pass 154 comparisons and retain 154 raw
arrays for full, prefix and chunked calls under the unchanged numerical policy.
Astra reviews source/evidence before commit. See
[the contract](causal-conv-reference.md) and
[receipt](../benchmarks/results/2026-10-24-causal-conv-reference-v1.json).
Empty input is a mathematical no-op; kernel-one source cache behavior, masks and
length-dependent caches remain excluded. Native convolution, gated output norm,
hybrid graph/session/Metal execution and real-model/app/device/long-context
qualification remain open. No capability or execution limit was raised.

## 24 October2026 — checked scalar causal convolution

The scalar step consumes one raw token with oldest-to-current per-channel taps,
accumulates FP64 products/sums and rounds once to FP32. It validates geometry,
lengths and finite inputs before writes. Previous history stays immutable; only
a fully successful output prepares the separate candidate by shifting history
and appending raw input. No buffers or session state are allocated/owned.

Four regressions authenticate the frozen reference receipt and check every
convolution/SiLU output and exact copied history, hand tap/channel order,
kernel-one/empty policy, intermediate cancellation, all invalid input classes,
output overflow and replay. The previous recurrent kernels remain unchanged. Formatting, strict default/Metal
Clippy, 207 CPU/229 Metal-feature tests and both iOS cross-checks pass.
Astra reviews code and evidence before commit. See
[the contract](causal-conv-scalar.md) and
[receipt](../benchmarks/results/2026-10-24-causal-conv-scalar-v1.json).

This is an operator foundation. Gated output normalization, hybrid model/session
state commit, Metal execution and real-model/chat/long-context/device quality
and performance gates remain open. No model eligibility or execution limit was
raised.

## 24 October2026 — physical mobile dashboard and Gemma 4 rejection checks

Commit `e4dcefb` records four passing physical iPhone 14 Pro UI tests on the
reviewed `abe661e` app and unchanged native `2d7d35b0` package. Monitoring,
device-fit explanations, supported-model switching/generation, code Copy,
256-token notice, Expert fixture, dialog deletion and public Gemma 4 unavailable
labels pass. The owner's history is preserved and normal app launch succeeds.

The SmolLM2 local benchmark completes one warm-up and three 64-token trials;
rounded UI samples show 17.8–18.0 decode tokens/s and 361–364 ms to first token.
These session observations do not establish controlled performance, quality or
long-context capability. Config-only official/MLX Gemma 4 probes safely reject
unknown architecture without creating a model handle. No Gemma weights,
generation or memory-fit claim follows. The official candidate's 131072 context
positions also do not establish 131072 output tokens plus a prompt.

Astra accepted the exact source and
[receipt](../benchmarks/results/2026-10-24-mobile-gemma4-phone-v1.json). See
[device coverage and limitations](mobile-gemma4-testing.md). The initial
certificate-trust failure and fresh successful run remain distinct. The full
engine/Hugging Face and long-reply plans stay open.

## 24 October2026 — gated output-normalization witness and active priorities

A separate independent FP64 reference fixes mean RMS normalization, direct scale
and subsequent stable SiLU gating over FP32 inputs. Five bounded cases include
epsilon-dominated rows, asymmetric heads, saturation and tiny negative gates.
The exact JSON reader rejects Boolean/invalid geometry and non-FP32 inputs.
Three ownership, eight math/format/failure and three runtime-admission tests pass;
pinned upstream CPU/GPU each pass all12 comparisons with immediate raw retention.
See [the contract](gated-norm-reference.md) and
[receipt](../benchmarks/results/2026-10-24-gated-norm-reference-v1.json).
No native composition, graph, quality, performance or phone qualification follows.

The human now prioritizes [Gemma4 and131072-token replies](gemma4-implementation-plan.md).
After finishing this small witness, the next task is pinned source/config/tensor/
tokenizer mapping and real memory/context feasibility. Gemma4 E2B's declared
131072 total positions do not establish131072 output tokens plus a prompt.
The installed passing phone baseline remains preserved until a committed,
qualified native/app candidate and full handoff evidence exist. The rest of the
architecture/Hugging Face plan stays open.

## 24 October2026 — Gemma4 E2B source and artifact mapping

The first pinned candidate inspection maps all540 logical text tensors to1096
physical tensors in the actual2511-entry header/index. It checks BF16/U32 shapes,
per-matrix affine metadata, spans and component coverage without downloading any
tensor payload. Nine final bounded owned HTTPS commands capture metadata and
header ranges; the initial redirect/framing/cancelled/order-correction attempts
remain separately retained. No full-shard integrity or native execution is claimed.

The text weights occupy2603981382 stored bytes, including1321205760 bytes for
per-layer embeddings. Proposed FP16 shared/local KV at131072 total positions
adds811597824 bytes; their sum is already about3.181GiB before other owners.
This is a lower-bound derivation, not phone admission. The tokenizer's decoder
has no leading-space Strip, and official/converted/nested stops differ; template
and text execution need independent fixtures. Exact graph order, shared-cache
producers, GELU, proportional rotary and per-layer paths are mapped in
[the inspection](gemma4-e2b-inspection.md) and
[receipt](../benchmarks/results/2026-10-24-gemma4-inspection-v1.json).

G1 mapping is complete; G2 starts with independent tanh-GELU/GeGLU witnesses.
Real weights, graph/session/Metal/Chat, quality and device fit remain open. The
unchanged131072 total-context configuration cannot meet131072 output tokens plus
a prompt, so no capability was raised and no new phone package was supplied.

## 24 October2026 — independent Gemma4 GELU/GeGLU witness

The first G2 witness freezes FP64 tanh-GELU and gated-product expectations over
FP32 inputs. Hand tests distinguish erf-GELU, SiLU and reversed gate order;
additional tests cover chunk independence, symmetry, tails/tiny values, invalid
inputs, exact bounded format and preservation of a completed observation when a
later call fails. Three ownership, seven portable and three runtime-admission
methods pass. CPU/GPU each pass eight arrays/360 values under unchanged policy.

Astra corrected owner filename references before the first capture; the
unexecuted proposal is retained. Exact source/runtime, raw arrays and independent
replay are bound by [the receipt](../benchmarks/results/2026-10-24-gemma-gelu-reference-v1.json)
and [contract](gemma-gelu-reference.md). No native function, whole graph, quality,
performance or phone fit follows. The scalar activation target is next; remaining
Gemma4 and131072-output requirements stay open and current caps are unchanged.

## 24 October2026 — native scalar tanh-GELU and GeGLU

Two allocation-free CPU functions add the activation needed by Gemma4. FP64
tanh-GELU rounds to FP32; GeGLU then multiplies by FP32 up, preserving separate
activation/product rounding. All admission checks precede writes. A gated
product overflow rejects with partial output to discard; immutable inputs allow
replay. Existing kernels, graphs and capabilities remain unchanged.

Four tests authenticate the independent receipt before fixture parsing, compare
all nonempty frozen arrays, enforce exact composed rounding, distinguish tanh
from erf/SiLU, exercise tiny/extreme values and invalid inputs, and check
overflow/replay. Astra's review strengthened wrong-length output preservation
assertions before the checked source freeze. Formatting, strict default/Metal
Clippy, 211 CPU/233 Metal-feature tests and both iOS cross-checks pass. See
[the contract](gemma-gelu-scalar.md) and
[receipt](../benchmarks/results/2026-10-24-gemma-gelu-scalar-v1.json).

This is a correctness operator, with no performance or full-model claim. G2
continues with proportional rotary; remaining Gemma4 graph/session/Metal,
real-model/Chat/device and131072-output gates stay open. The installed phone
baseline and all output/context/byte caps remain unchanged.

## 24 October2026 — Gemma4 rotary precision gap made explicit

A bounded source/operator diagnostic on this M1 Pro captures nine positions on
separate cold-admitted MLX CPU/GPU processes before implementing proportional
rotary. It verifies the width512 denominator and pair distance256 for128 rotated
dimensions. All observations complete; independent stdlib replay authenticates
every raw array and all eighteen numerical comparisons. At sampled positions
0,1,7,31,127 both backends meet the existing FP64 policy. Both fail at511,1023,
8191,131071; the largest error/policy ratios at131071 are642.26 CPU/314.82 GPU.

Saved-frequency reconstructions help isolate FP32 frequency/phase effects but
cannot admit an independent oracle; GPU cause remains incompletely explained.
Astra strengthened ownership/capture/collision handling before execution.
See [the full diagnostic](gemma-rotary-precision.md) and
[receipt](../benchmarks/results/2026-10-24-gemma-rotary-precision-v1.json).
The tolerance is unchanged, failing comparisons stay recorded, and no native
rotary, context/model/phone/performance qualification is claimed. G2 continues
with an explicit independent frequency/phase contract and broader witnesses.

## 24 October2026 — independent proportional rotary math frozen

A pure standard-library producer constructs frequencies/angles with Decimal80
from exact FP32 inputs and uses FP64 trig/pair output. Eight cases freeze empty,
zero-position, half-pair, full/zero rotation, non-unit factor, actual D512/R128
geometry and positions ending at131071. Six offline methods pass hand literals,
signed-zero copying, chunk offsets, multiple heads, norm preservation and invalid
geometry/position/precision/external-format checks. Independent90-digit replay
uses logarithm/exponential construction to validate the saved arrays.

See [the mathematical contract](gemma-rotary-math.md) and
[receipt](../benchmarks/results/2026-10-24-gemma-rotary-math-v1.json).
This is frozen before its scalar target; the previous eight upstream numerical
failures remain explicit and unwaived. No graph, Metal execution, quality,
performance, model/context/phone qualification or capability change follows.

## 24 October2026 — scalar proportional rotary correctness reference

The standalone CPU operation fixes full-width exponents and half-head pairing
for proportional rotation. It uses FP64 inverse frequencies/absolute angles and
trig, then explicit FP32 sine/cosine and rotation arithmetic, while inactive
coordinates copy bit for bit. Admission validates geometry, position and size
overflow, lengths and all finite inputs before writing. Arithmetic overflow
leaves discardable output and immutable inputs for replay. No buffers are
allocated and existing rotary remains unchanged.

Six tests authenticate the prior mathematical receipt, compare every nonempty
fixture output under unchanged policy, check hand values/signed-zero copies,
full-rotation equivalence, multiple heads/exact chunks, invalid admission,
overflow/replay and finite-parameter/position boundaries. Formatting, strict
default/Metal Clippy, 217 CPU/239 Metal-feature tests and both iOS cross-checks
pass. Astra reviews exact source and evidence before commit. See
[the contract](gemma-rotary-scalar.md) and
[receipt](../benchmarks/results/2026-10-24-gemma-rotary-scalar-v1.json).

The eight upstream long-position disagreements remain unwaived. Extreme
parameter/boundary tests establish finiteness/admission, not extra numerical or
model quality. No graph/session/Metal integration, context or output cap, phone
package, real-model/performance qualification follows. G2 continues with
local/sliding attention and shared-KV ownership witnesses.

## 24 October2026 — causal/sliding GQA witness and512-window boundary

Independent FP64 scores, stable masked softmax and context fix the causal/local
window predicate, contiguous grouped-head mapping and explicit Gemma4 scale1.
Eleven cases freeze77 arrays and14640 output scalars, including actual head
geometry and a separate small-width512-window boundary. Seven portable,
three ownership and three runtime-admission methods pass. Pinned upstream CPU
and GPU each match20 raw observations; masks are exact and maximum context
error/policy ratios are about0.00810/0.00487 under unchanged tolerance.

Astra reviewed the added512 boundary after the initial small-window capture;
the preliminary result remains separate from the complete fresh capture.
Independent replay validates all saved arrays and comparisons. See
[the contract](gemma-local-attention-reference.md) and
[receipt](../benchmarks/results/2026-10-24-gemma-local-attention-reference-v1.json).
The numerical engine remains unchanged. Scalar local attention, physical
ring/shared-KV ownership and failure recovery, whole graph, rotary parity,
quality/performance and app/device/capability gates remain open.

## 24 October2026 — scalar causal/sliding-window GQA

The new allocation-free CPU operation consumes token-major queries and
head-major KV capacity with an explicit valid prefix and logical origins. It
preserves contiguous head grouping, explicit scale, causal/window boundaries
and exact masked zero probabilities. Unused capacity is never read; callers
still provide initialized Rust slices. All geometry, length, position, finite
input and fully masked query checks precede output writes.

FP64 dot/scaling stores FP32 scores, stable FP64 softmax stores FP32 probabilities,
and context accumulates those probabilities times V in FP64 before FP32 output.
All valid-key scores, including masked ones, remain diagnostic and may reject
overflow. Failed arithmetic leaves discardable outputs and unchanged inputs for
replay. The initial formatting error in test literals was fixed before capture
and its proposal retained. Six tests compare every frozen nonempty array plus
capacity padding, mask zeros, invalid/prewrite paths, overflow/replay and exact
multi-head chunks. Formatting, strict default/Metal Clippy, 223 CPU/245
Metal-feature tests and both iOS cross-checks pass. Astra reviews final evidence
before commit; see [the contract](window-attention-scalar.md) and
[receipt](../benchmarks/results/2026-10-24-window-attention-scalar-v1.json).

The numerical core owns no cache/session and the existing engine paths remain
unchanged. Physical ring mapping, shared-KV ownership/commit/rollback, other G2
mechanisms, full graph/Metal/quality and device/capability gates remain open.
No native ABI, output/context/byte limit or phone package changes.

## 24 October2026 — independent KV staging and ring-state witness

The required attention union, each query's visible keys and the final committed
ring now have separate frozen expectations. Seven cases cover thirty updates
through single-token and multi-token wraps, window1, global cache and the512
boundary. At most two old physical spans plus immutable new tokens preserve
keys needed by early chunk queries. The314 arrays contain23306 values; logical
FP64 context is independent of the source's physical ring phase.

Seven portable, three owned-process and three runtime-admission methods pass.
Pinned upstream CPU/GPU each retain101 raw arrays and pass30 context comparisons;
K/V and masks are exact, with maximum context error/policy ratios about0.00193
and0.00230. Independent replay verifies the saved state and observations. The
initial wrong-interpreter preflight is retained separately with no child run.
Astra reviews exact final code/evidence before commit. See
[the contract](gemma-kv-reference.md) and
[receipt](../benchmarks/results/2026-10-24-gemma-kv-reference-v1.json).

Abort/retry remains a required native policy, not tested native/upstream
rollback. Shared producer reuse, native cache/session and ring-aware attention,
full graph/Metal, rotary parity, quality/performance and device/capability gates
remain open. Existing numerical engine, native ABI, app limits and phone package
are unchanged.

## 24 October2026 — native KV owner with checked batch commit/discard

A standalone FP32 cache now owns fixed global or local-ring K/V buffers and
exposes borrowed old spans plus immutable new-token staging. It preserves old
keys needed by early chunk queries until commit, without copying the whole
cache. Private frames bind owner identity, generation and offset; constructor
identity reservation adds no decode-step atomics. Actual capacities and owner
metadata are reported separately from borrowed staging and numerical scratch.

Every batch frame is validated before writes; commit copies only new retained
rows and advances positions once. Six tests cover all thirty frozen updates,
exact spans/rings, discard/retry, a failed last frame preserving all owners,
missing/reordered/duplicate/stale/foreign frames, signed zeros, oversized local
chunks, global limits, invalid arithmetic and moved owners. Formatting, strict
default/Metal Clippy,229 CPU/251 Metal-feature tests and both iOS cross-checks
pass. Astra reviews final code and evidence before commit; see
[the contract](window-cache-scalar.md) and
[receipt](../benchmarks/results/2026-10-24-window-cache-scalar-v1.json).

This establishes cache batch commit/discard, not whole-model rollback or shared
producer/consumer integration. Ring-aware attention, resident Metal and Gemma
graph/session/weights/quality/device qualification remain open. No ABI, app
limits, protected binaries or installed phone package change.

## 24 October2026 — shared-KV routing and current-step identity witness

An independent exact routing fixture now records the35-layer Gemma4 producer
map: first15 own KV, then local13/global14 supply the last20 consumers. Selected
unchanged Attention, DecoderLayer and text-model control-flow bodies execute
with strict shape/identity stand-ins. Consumer cache arguments are None, each
receives the current-step producer's exact KV objects and offset, and only
producers append. Four cases/fifteen repeated calls observe226 layer forwards,
108 appends and118 consumers, including the512-window transition.

Seven portable and four owned-process methods pass; four owned commands and
independent replay bind all routing observations. Source constructors/numerical
operators are explicitly replaced, with PLE/MoE/learned scalars disabled. Astra
reviews final exact source/evidence before commit. See
[the contract](gemma-shared-kv-reference.md) and
[receipt](../benchmarks/results/2026-10-24-gemma-shared-kv-reference-v1.json).

Native shared-frame/stale-input checks, numerical/full-layer/Gemma graph,
rotary parity, whole-model rollback, quality/performance/memory/device gates
remain open. Engine/shader sources, ABI, app limits, protected binaries and
installed phone package remain unchanged.

## 24 October2026 — task-based mobile model ranking and search controls

The reviewed mobile commit `f739fabe` adds Balanced/Coding/Fast ranking and
readable score explanations after Toklet compatibility/memory filtering. Scores
are estimated preferences, not measured quality/speed or a grant of support.
Fresh load admission remains authoritative. A Search button dismisses the
keyboard so results remain usable; disclosure labels and picker waits were
corrected after two retained failed phone suites.

Eight local checks, signed build/signatures and the final five physical iPhone14
Pro methods pass, including Fast→Smol, Coding→Qwen, real Qwen reply, unsupported
Gemma search, model switching,256 selection and Expert controls. Dialog data
remains byte-exact; failed suites include verified rollback and normal launch.
Astra accepted the exact nine-file packet before commit. See
[the mobile record](mobile-model-ranking.md) and
[receipt](../benchmarks/results/2026-10-24-mobile-model-ranking-v1.json).
Native `2d7d35b0` and Chat/Expert/context limits remain unchanged; this is selected
functional evidence, not Gemma4 or131072-output qualification.

## 24 October2026 — native shared-KV routing and ordered publication

An immutable concrete routing plan creates only producer caches and resolves
shared layers to same-kind owners with matching geometry. Consumer views borrow
the actual producer spans/new input; no private cache, copy or append occurs.
Every committed owner's mode/window/capacity/position/geometry and offset/
generation is checked, including unpublished owners. Ordered prefix publication
allows earlier attention before later producers exist; future proposed lengths
are checked when published. Final commit requires the complete valid batch
before writes. Model/session/token-history provenance stays caller-owned.

Eight methods compare226 frozen layer views/108 producer updates and exercise
geometry, last-owner state/token mismatch, missing/reordered/duplicate/foreign/
stale publication, sequential availability, partial commit, discard/retry,
moved identity and exact memory accounting without duplicated owner metadata.
Formatting, strict default/Metal Clippy,237 CPU/259 Metal-feature tests and both
iOS cross-checks pass. Astra reviews final source/evidence before commit. See
[the contract](shared-kv-scalar.md) and
[receipt](../benchmarks/results/2026-10-24-shared-kv-scalar-v1.json).

Ring-aware numerical attention, full Gemma/session/Metal, rotary parity,
quality/performance/device fit and capability gates remain open. No ABI, app
limits, protected binaries or installed package changes follow from this task.

## 24 October2026 — scalar attention directly over borrowed ring spans

A concrete three-segment CPU operation reads two old ring spans and the new
chunk directly, with explicit per-segment head strides and logical origins.
It preserves the contiguous scalar's reduction order and precision, avoiding
whole-KV linearization, allocation or dynamic dispatch. Cache views derive the
physical slices; immutable borrowing prevents concurrent commit.

Eight methods compare frozen scores/probabilities/original FP64 contexts and
bitwise contiguous outputs, then run all thirty KV-witness updates through
actual ring views with Hkv2 wrap, pointer identity and512-window coverage.
Empty spans, NaN/negative-zero padding, late invalid/overflowing metadata before
writes, fully masked later queries, arithmetic failure/replay and no-commit
recovery are checked. Formatting, strict default/Metal Clippy,245 CPU/267
Metal-feature tests and both iOS cross-checks pass. Astra reviews final evidence
before commit; see [the contract](span-attention-scalar.md) and
[receipt](../benchmarks/results/2026-10-24-span-attention-scalar-v1.json).

Actual head widths and full window transitions are separate bounded cases;
maximum-usize arithmetic checks do not grant model/context support. Full Gemma
layer/session/weights/Metal/rotary/quality/performance/device gates remain open.
No ABI, app limit, protected binary or installed phone package changes.

## 24 October2026 — per-layer token lookup and projection witness

Independent ideal FP64 and staged FP32 contracts now freeze raw main embedding
scale, lookup scale, projection/reshape/per-P learned RMS, addition and final
combination scale. Rounding boundaries stay explicit; a portable test detects
illegal algebraic scale cancellation. Hand literals distinguish the norm axis,
direct unequal learned weights and lookup sqrt(P). Seven cases include batches,
projected-only helper, empty policy and separately bounded H1536/L35-P256 shapes.

The173 arrays contain253823 values. Seven portable, three ownership and three
runtime-admission methods and all six owned commands pass. Separately evaluated
real FP32 operator probes and unchanged source helper outputs retain71 raw
arrays per CPU/GPU, each passing142 comparisons against both contracts. Maximum
error/policy ratios are about0.02092/0.02354 CPU and0.04990/0.05188 GPU. Independent
scalar replay verifies all arrays; staged FP32 matches exactly. Astra reviews
final exact source/evidence before commit. See
[the contract](gemma-per-layer-reference.md) and
[receipt](../benchmarks/results/2026-10-24-gemma-per-layer-reference-v1.json).

Probes are separate executions, not helper-internal/fusion qualification.
Harness ID rejection, projected-only helper and empty policy have explicit
limits; BF16/quantized checkpoint, nearest-token fallback, native composition,
full Gemma/session/rotary/quality/performance/device gates remain open. No ABI,
capability, app limit, protected binary or installed package changes.

## 24 October2026 — one-tap live Hugging Face discovery

The reviewed mobile commit `17fbc4f` gives Find best model one44pt tap target
and bounded live public Hugging Face discovery, while pinned Chat compatibility
and fresh memory admission remain authoritative. Inventory refresh/preference
changes no longer cancel or restart discovery. Four bounded metadata tasks
retain partial network errors but drain every child and preserve fatal close/
removal cleanup causes through cancellation. Focused regressions cover those
paths; an interrupted phone attempt is retained without passing credit.

Eight local checks, transport checks, real68-model discovery, signed build and
strict signatures pass. The final six physical iPhone14Pro tests pass, including
ordinary/edge/repeated taps, live links/results, resident Qwen and keyboard/draft
preservation plus prior controls. Owner chat bytes remain exact; the updated
app normally launches and a verified rollback is saved. Astra accepted the
exact nine-file packet before commit. See [the mobile record](mobile-model-ranking.md)
and [receipt](../benchmarks/results/2026-10-24-mobile-hf-discovery-v1.json).
This grants no new engine eligibility, speed/quality or131072-output support.

## 24 October2026 — native per-layer input composition

Two plain CPU functions now prepare scaled main embeddings and finish raw
backend projections with per-P direct RMS and optional selected lookup rows.
Projection/table ownership stays outside the mechanism, allowing later
quantized backends without an expanded FP32 model copy. All FP32 scale/add
boundaries stay explicit; optional lookup dispatch is outside element loops.
Caller scratch/output are reused, with complete prewrite admission and
discard/replay semantics for arithmetic overflow.

Six methods compare full scalar paths/stages against both frozen contracts,
exact scaled-main bits/selected-row order, hand normalization, late validation,
intermediate overflow, replay/immutable inputs, exact chunks, buffer reuse and
empty/signed-zero/size boundaries. Formatting, strict default/Metal Clippy,251
CPU/273 Metal-feature tests and both iOS cross-checks pass. Astra reviews final
source/evidence before commit; see [the contract](per-layer-scalar.md) and
[receipt](../benchmarks/results/2026-10-24-per-layer-scalar-v1.json).

Quantized/BF16 execution, per-layer gate/residual/scalar order, full Gemma/session/
Metal/rotary/quality/performance/device fit remain open. No ABI, app limit,
protected binary or installed package changes follow from this numerical task.

## 24 October2026 — independent dense decoder order

The new Gemma witness freezes exact direct-RMS/residual/FFN/PLE/scalar order
before native composition. Authenticated unchanged DecoderLayer and MLP calls
retain the compiled GeGLU helper and use real FP32 MLX modules. An explicit
external attention result isolates this task; normalized input and all incoming/
returned state identities are checked. Every PLE guard term, disabled modules,
small distinct FFN widths and synthetic scalar absence are separately covered.

Twelve cases contain540 arrays and75,720 values. An independent stdlib replay
verifies all values, with exact staged FP32 agreement. Eight portable, three
owner and three runtime-admission methods and all six owned commands pass.
Each CPU/GPU source comparison retains201 raw arrays and passes402 ideal/staged
comparisons. Maximum error/policy ratios are0.04864/0.05881 CPU and0.01746/0.02760
GPU under unchanged gates. Astra reviews the final source/evidence before commit.
See [the contract](gemma-decoder-order-reference.md) and
[receipt](../benchmarks/results/2026-10-24-gemma-decoder-order-reference-v1.json).

Separately evaluated probes do not claim internal helper-stage/fusion coverage.
Constructor/config/double-wide selection, attention/KV/rotary, MoE, BF16/quantized
weights, native layer/session, full model/device fit, quality, performance and
long replies remain unqualified. The eight long rotary discrepancies remain
unwaived. Existing numerical sources, ABI, app limits, protected binaries and
installed native package remain unchanged.

## 24 October2026 — native dense decoder finish

A plain concrete FP32 function now composes post-attention RMS/residual, GeGLU
FFN with pre/post norms, optional complete PLE gate and final learned scalar.
Input normalization/attention and all cache/session ownership stay outside.
Borrowed immutable weights and five caller-owned scratch buffers avoid numerical
allocation, weight clones and dynamic inner-loop dispatch. Coherent optional
PLE admission validates even skipped weights before writes; arithmetic overflow
requires discarding output/scratch and replaying unchanged inputs.

Six methods compare all11 nonempty frozen cases and final scratch stages against
both independent contracts, then cover late invalid inputs/scalars/scratch,
missing input/disabled PLE/scalar absence, signed zero, exact repeated chunks,
buffer reuse, immutable operands, sizes and epsilon. Seven overflow scenarios
prove successful prerequisites and the exact failing operator before replay.
Formatting, strict default/Metal Clippy,257 CPU/279 Metal-feature tests and both
iOS compile targets pass. Astra reviews final source/evidence before commit.
See [the contract](dense-decoder-scalar.md) and
[receipt](../benchmarks/results/2026-10-24-dense-decoder-scalar-v1.json).

This is dense FP32 reference composition. Packed quantized/BF16 weights,
whole-layer attention/cache graph, session/model/Metal, device fit, quality,
performance and long replies remain open; existing caps/native package and
protected binaries are unchanged. Upstream long-position rotary discrepancies
remain unwaived.

## 24 October2026 — independent unlearned V normalization

The Gemma V witness fixes each D-wide head row's mean-square RMS with epsilon,
without learned scale. Distinct anisotropic heads, hand literals, D1, zeros,
epsilon-dominated values and D256/D512 expose identity, wrong-axis, sum/L2 and
epsilon mistakes. Independent stdlib replay exactly reproduces all24 arrays and
7,350 values, including staged FP32.

The actual unchanged pinned source helper runs on MLX CPU/GPU. Seven portable,
three owner and three runtime-admission methods and all six owned commands pass.
Each device retains seven raw outputs and passes14 ideal/staged comparisons;
maximum error/policy ratios are below0.001066. Partial outputs, failures, child
ownership and source/protected-binary guards are retained. Astra reviews final
source/evidence before commit. See [the contract](gemma-v-norm-reference.md) and
[receipt](../benchmarks/results/2026-10-24-gemma-v-norm-reference-v1.json).

Native mapping, V projection/storage, KV/attention/model/session/Metal graph,
constructor/config, BF16/quantized checkpoint, fit, quality, performance and long
replies remain open. Future scalar mapping can reuse RMS with a persistent exact-
one D vector, counted once and owned by the adapter. No numerical source, ABI,
capability, protected executable or installed package changes in this task.

## 25 October2026 — scalar unlearned V-norm recipe

The existing RMS kernel now passes the accepted V fixture with persistent
immutable unit weights. Three methods authenticate the reference receipt/fixture,
compare seven nonempty cases against both contracts and exact staged FP32 bits,
then qualify independent head-row chunks, reused pointers and immutable input/
scale data. Native-only finite extremes/subnormals/signed zero and prewrite
invalid-input/size/epsilon cases remain separate from upstream fast RMS scope.

Formatting, strict default/Metal Clippy,260 CPU/282 Metal-feature tests and both
iOS compile targets pass. Astra reviews exact final evidence before commit.
See [the recipe](gemma-v-norm-scalar.md) and
[receipt](../benchmarks/results/2026-10-24-v-norm-scalar-v1.json).

The initial copied-cache suite is retained as failed: a reused CLI test still
embedded the previous target's executable path. Its actual binaries and failure
logs are preserved. Cleaning only copied Toklet package artifacts and checking
all fresh CLI-spawning binary paths fixes qualification provenance without
changing code or tolerances. A second failed collection expected a binary path
in a cfg-disabled empty harness; owned zero-test listing closes that evidence
gap. Only its four successful checks are inherited before completing the other
three with unchanged source guards; both failed attempts remain unaccepted. Future copied-cache trials must prevent this absolute-path reuse.

No production numerical/shader code changes. A future Gemma adapter must own and
account for its shared exact-one vector; V projection/storage, graph/cache/
session/Metal integration, device fit, quality, performance and long replies
remain open. No installed package, app cap, protected binary or ABI changes.

## 25 October2026 — independent Qwen3 plain Chat reference

The separately reviewed reference commit `6edb777` freezes original non-thinking
Qwen3 Chat strings/IDs, original NFC behavior, response prefixes and all26 control
literal IDs/flags. Twelve controls are non-special and remain visible on decode.
The guarded runtime admits exact Python/Jinja/MarkupSafe/tokenizers files before
imports; independent capture/replay,11 admission tests and4 owner failure tests
pass. Preliminary failed assertions/attempts are retained, with unchanged original
conversation/Unicode values. See [the witness](qwen3-chat-reference.md) and
[receipt](../benchmarks/results/2026-10-25-qwen3-chat-reference-v1.json).
No fourth runnable app profile, native package or phone fit is granted; cold
profile, memory, package and device qualification follow separately.

## 25 October2026 — independent final-logit softcap

A tiny compiled-source witness now freezes Gemma's cap30 tanh-logit transform,
plus explicit synthetic smaller caps. Separate ideal/staged contracts expose
quotient, activation and multiplication rounding without claiming helper-internal
captures. Four staged divisions overflow for finite extrema/small caps while
external outputs remain finite; counts are retained without serializing infinity.

Independent replay verifies all18 arrays/585 values, exact staged bits and zero
signs. Seven portable, three owner and three runtime-admission methods and all
six owned commands pass. Each MLX CPU/GPU source capture retains five outputs
and passes ten comparisons under unchanged gates; maximum error/policy ratio
is below0.00125. Astra reviews final source/evidence before commit. See
[the contract](gemma-softcap-reference.md) and
[receipt](../benchmarks/results/2026-10-25-gemma-softcap-reference-v1.json).

Native transform, complete vocabulary/model graph, BF16/quantized checkpoint,
constructor/disabled-cap routing, device fit, quality, performance and long replies
remain open. Upstream signed-zero bit parity is unqualified; mathematical/native
zero semantics are checked separately. No numerical source, ABI, app cap,
protected executable or installed package changes.

## 25 October2026 — native in-place logit softcap

A plain FP32-slice transform now applies tanh logit capping without a second
vocabulary-sized array. Full prewrite admission preserves invalid-input bits;
FP64 division/tanh/product gives finite bounded output for every positive finite
FP32 cap and finite logit, including domains where staged FP32 division overflows.
Signed zero remains explicit. No allocation, dynamic numerical dispatch or unsafe
code is added.

Four methods compare five frozen nonempty cases against both contracts, then
cover late invalid values/caps, storage reuse, native-only extreme/subnormal
caps, odd/monotone/bounded properties, exact zero signs and chunk equivalence.
Formatting, strict default/Metal Clippy,264 CPU/286 Metal-feature tests and both
iOS compile targets pass. Package-only cleanup in the private copied target and
all CLI-spawning executable paths are verified, with a separate zero-test proof
for the disabled harness. Astra reviews final source/evidence before commit.
See [the contract](gemma-softcap-scalar.md) and
[receipt](../benchmarks/results/2026-10-25-softcap-scalar-v1.json).

Model integration, quantized/BF16 checkpoint, Metal, fit, quality, performance,
phone and long replies remain open; upstream rotary discrepancies stay unwaived.
No ABI, app capability/limit, protected executable or installed package changes.

## 25 October2026 — checked native Qwen3 plain Chat profile

Commit `40bf67f` adds the exact immutable Qwen3-0.6B eight-bit text profile to
native Chat. It preserves the original non-thinking prefix, EOS151645, all26
reserved input literals and the visible behavior of12 non-special added tokens.
It adds no numerical kernels or ABI declarations. The conservative256MiB text
reservation is separate from numerical memory; no whole-app peak follows from it.

Astra accepted the guarded [shared-tree integration receipt](../benchmarks/results/2026-10-25-qwen3-native-chat-integration-v1.json):
266 CPU/288 Metal-feature tests, strict default/Metal Clippy, formatting, both iOS
compile targets, four real codecs, four native C API Chat trials and five frozen
Qwen3 GPU state/admission tests pass. All17 commands, complete test/fixture/model/
external-array inputs, raw logs, actual binaries and process-group cleanup are
bound. The narrower isolated evidence and stale Mistral test-boundary failure
remain explicitly recorded. See [scope and behavior](qwen3-native-chat.md).

Chat256/raw128/total512 limits stay unchanged. New package/app/iPhone admission,
whole-app memory, held-out quality, performance, Gemma and long replies are
separate tasks; this native commit alone adds no installed eligible model.

## 25 October2026 — complete tiny Gemma text graph reference

A complete synthetic token-to-logit fixture now composes embedding scaling,
per-layer inputs, local/global GQA, shared KV, exact dense layer order and the
tied capped output head. Independent replay verifies4,358 deterministic weight
values,32 token IDs and180 stage arrays/68,224 values. Staged FP32 values match
exactly. Three schedules cover every position0..31 and repeated sliding wraps.

All7,462 source CPU comparisons pass. The source GPU run completes but rejects15
intermediate comparisons under unchanged gates; its maximum ratio is1.35576321.
All164 final-logit comparisons per device pass, which does not override that
rejection. Original failed GPU command/raw arrays remain retained. Captured-stage
reconstruction supports accumulated drift and normalization/residual sensitivity
without establishing a specific GPU kernel root cause. Weights and tolerances
remain unchanged. Nine portable, three owner/failure and three runtime-admission
methods pass; Astra reviews exact code/evidence and the CPU-only decision before
commit. See [the witness and limitations](gemma-text-graph-reference.md) and
[acceptance/rejection receipt](../benchmarks/results/2026-10-25-gemma-text-graph-reference-v1.json).

This freezes a short synthetic CPU reference, not a Gemma checkpoint or completed
G2. Native graph/session, GPU intermediate parity, original long-position rotary,
quantized storage, real quality/fit/performance and app/device work remain open.
No Rust/shader, ABI, app cap, protected executable or installed package changes.

## 25 October2026 — bounded native Gemma scalar graph/session

A named reference module now composes the exact hidden8/vocabulary17/four-layer
Gemma fixture using existing statically dispatched scalar operations. Immutable
FP32 weights stay separate from model-bound sessions, producer K/V staging,
shared/rotating caches and reusable scratch. Both producer frames remain alive
through the shared consumers and final norm/head/softcap. Only then do all caches,
committed token history and public logits advance. Earlier failure preserves
their exact previous state; reset reuses buffers and invalidates old proposals.

Five graph methods check all three schedules against both frozen contracts,
including selected intermediate states of every layer, producer staging/physical
KV positions, final logits, immutable weights, pointer reuse and reset/replay.
Five injected late boundaries and a real finite scalar overflow qualify failure
atomicity. A separate reset method covers stale frames and generation exhaustion.
The scoped allocator test records zero Rust allocations across96 warmed decode
steps, three resets/accounting and a rejected memory admission. Logical reference
memory is build-specific and does not qualify a process or phone budget.

Formatting, strict default/Metal Clippy,272 CPU/294 Metal-feature tests and both
iOS compile targets pass. Complete source/test/fixture/helper guards, raw logs,
fresh CLI-spawning executable identities and protected binaries are retained.
Astra reviews final code/evidence before commit. See [behavior and limits](gemma-text-graph-scalar.md)
and [receipt](../benchmarks/results/2026-10-25-gemma-text-graph-scalar-v1.json).

This is a32-position synthetic G3 foundation. Actual checkpoint/config/quantized
integration, remaining generation/session semantics, Metal, real quality/fit/
performance and app/device work stay open. Neither upstream GPU discrepancy is
waived. No ABI, installed package or app generation-limit change is included.

## 25 October2026 — named Gemma4 E2B config/layout adapter

A separate cold module now validates the audited E2B wrapper/text semantics and
derives540 logical tensors/1096 physical MLX descriptors, using the existing
four-bit/group64 affine layout and BF16 metadata. Explicit text-only policy
separates known modality components; unknown text options, private consumer KV,
tensor names/layouts, index mismatches and invalid/gapped/overlapping spans reject.
Bounded duplicate-free JSON and checked arithmetic precede metadata planning.
The existing executable dense-family/compatibility registry is unchanged.

Seven offline methods authenticate original G1 config/index/header bytes and
exercise graph/representation/shape/index/offset/overflow/duplicate/buffer-limit
mutations. An initial parser comparison mistook equivalent JSON exponent spelling
for a semantic difference, and one negative test used the wrong excluded name;
both failed source/log records are retained. Finite float semantics are corrected
without changing fixture bytes or any numerical tolerance.

Formatting, strict default/Metal Clippy,279 CPU/301 Metal-feature tests and both
iOS compile targets pass. Astra reviews final code/evidence before commit.
See [scope and mapping](gemma4-config-layout.md) and
[receipt](../benchmarks/results/2026-10-25-gemma4-config-layout-v1.json).
The2,603,981,382-byte text count is a header-derived layout, not a payload-value,
allocation/fit or performance result. Real payload/graph/tokenizer/Metal/quality,
device and app gates remain open, with no new eligibility, ABI, output limit or
upstream GPU waiver.

## 25 October2026 — independent packed table/projection precision

Before native packed Gemma graph composition, freeze three-row four-bit/group64
tables with BF16 coefficients at widths64,128,1536 and8960. Literal packed codes,
unequal group/row coefficients and repeated/reordered lookup IDs distinguish
FP32 reconstruction from BF16 once-rounded and separate-rounded alternatives.
Non-BF16-exact FP32 inputs exercise the input-rounding boundary. Independent
ideal/ordered-FP64 contracts and exact raw ledgers precede any new native target.

Actual MLX embedding/linear/tied-head calls complete on both Mac CPU and GPU.
All28 required FP32/tied comparisons per backend pass. Each captures24 arrays /
85,600 values; BF16 diagnostics remain separate: CPU gathers match separate
product/add rounding, GPU gathers match once-rounded reconstruction, while28
diagnostic comparisons per backend reject unchanged gates. No BF16 checkpoint,
CPU/GPU equivalence or complete graph/device qualification is granted.

Seven portable, three owner and three runtime-admission tests plus export and
both backend captures pass as six owned commands. Sequential observation capture
preserves completed raw bytes before dtype/shape/finiteness rejection and on
later failure/interruption. Source copies and actual imported module origins
are bound to admitted runtime identities. The initial insufficient width64
rounding witness and pre-correction source snapshots remain unaccepted;
non-dyadic seeds are corrected before any fixture freeze, without a gate change.

See [contracts and observed differences](gemma-packed-reference.md) and the
[receipt](../benchmarks/results/2026-10-25-gemma-packed-reference-v1.json).
Rust/shader/header inputs remain unchanged. Payload admission, native packed
composition, quality, performance, Mac/phone fit and app integration remain open;
the existing15 short-GPU and8 long-rotary discrepancies are not waived.

## 25 October2026 — native packed row fetch

Add `Matrix::copy_row` to fill caller-owned FP32 storage without expanding a
quantized table. Dtype/bit dispatch occurs once, group metadata is hoisted, and
all selected reconstruction values are preflighted before any output write.
The second reconstruction pass preserves failure atomicity without a temporary
row. Existing scalar value/matvec precision and behavior remain unchanged.

Two focused tests pass all54 prior affine layouts and the newly frozen Gemma
gather/projection bytes at small and actual embedding widths. They check
repeated/reordered rows and output storage reuse,18 unaligned metadata/bit
combinations and signed-zero bits, and unchanged outputs on bad row/length or
reconstruction overflow in a later metadata group. The first preliminary
two-test pass and its source are retained; final required checks include the
strengthened later-group overflow case.

Formatting, strict default/Metal Clippy,281 CPU/303 Metal-feature tests and both
iOS compile targets pass in an isolated package-cleaned target. Astra reviews
the exact final source/evidence before commit.

See [API and scope](mlx-affine.md#native-row-fetch) and the
[receipt](../benchmarks/results/2026-10-25-mlx-affine-row-fetch-v1.json).
This is a reference primitive for G3, with no BF16, payload/graph, quality,
performance, device or app-capability qualification. All existing upstream
precision gaps remain open.

## 25 October2026 — complete tiny packed Gemma graph reference

Freeze a group64-compatible four-layer packed graph with tied embedding/head,
PLE64, local64/global128 attention, two shared K/V producers, MLP64/128,
direct norms, residual/scalar ordering and softcap. Literal packed patterns and
separate semantic equations retain original BF16 coefficient bits, explicitly
promoted-FP32 source parameters and independent ideal/staged graph contracts.
The earlier dense witness stays frozen.

Actual original class bodies, compiled helpers and MLX quantized modules execute
all32 positions in three schedules, both pristine and traced. Each backend
captures3,731 arrays/852,768 values. CPU passes all7,462 comparisons; GPU completes
but rejects15 intermediate comparisons under unchanged gates. All164 final-logit
comparisons per backend pass, without waiving the rejected states. This execution
uses promoted-FP32 coefficient values and activations; original-BF16 graph parity
remains unqualified.

Independent standard-library replay verifies253,188 expanded weights,41,238
physical values and180 mathematical arrays/567,424 values. Staged FP32 is exact;
ideal maximum difference is1.3322676295501878e-14. Ten portable, three owner and
three runtime-admission tests, export and both source captures pass as six owned
commands. Numerical rejection stays separate from capture completion. Source,
runtime, raw output, routing/cache identity and immutable parameter checks bind
the evidence.

See [scope and comparisons](gemma-packed-graph-reference.md) and the
[receipt](../benchmarks/results/2026-10-25-gemma-packed-graph-reference-v1.json).
Next is native borrowed packed model/session composition. Real payload/value,
generation/tokenizer, Metal, quality, performance, Mac/phone fit and app gates
remain open; the prior15 dense-GPU and8 long-rotary failures remain unwaived.

## 25 October2026 — native borrowed packed Gemma graph/session

Compose packed embedding/PLE, four local/full/shared-KV decoder layers and the
tied capped head using borrowed matrices. A bounded public constructor admits
the frozen H64 profile; a private layout drives dimensions and scratch planning.
Weights are never expanded into a retained matrix. Sessions reuse23 scratch
buffers, persistent producer staging and two caches. All-cache validation and
publication follow final normalization/head/cap; history and prior logits remain
valid on failure. Reset reuses the same storage.

Four unit tests cover all three schedules and selected states from every layer,
physical cache slots, buffer identities, seven late fault/retry boundaries,
invalid inputs/capacity, real scalar overflow and model-boundary admission. A
separate integration test checks96 warmed decode steps/three resets/accounting
with zero Rust allocations and rejected admission before session allocation.
The receipt records required default/Metal checks, both iOS compile targets and
Astra review before commit.

See [scope and ownership](gemma-packed-model-scalar.md) and the
[receipt](../benchmarks/results/2026-10-25-gemma-packed-model-scalar-v1.json).
This is a synthetic32-position scalar G3 foundation. Actual payload/value and
E2B geometry, generation/tokenizer, Metal, quality/performance, Mac/phone fit and
app stages remain open; no ABI/catalog limit is raised. All existing upstream
GPU and long-rotary discrepancies remain unwaived.

## 25 October2026 — cold Gemma4 payload ownership and value admission

Add a streaming immutable text-only payload store behind the strict E2B
metadata adapter. Hash the consumed prefix/header and every payload byte,
including discarded audio/vision components. Reject trailing data, digest/read
failures and nonfinite vectors/coefficients; code0/15 reconstruction bounds all
four-bit codes without expanding weights. Cold borrowed views keep explicit
store/model/session lifetimes and numerical loops free of name lookups.

Admit packed data, converted norm storage, conversion overlap, streaming scratch
and cold metadata before large allocations. Recheck actual vector capacities
and peak live overlap. Five focused offline tests distinguish private synthetic
success from public real-header budget rejection and cover fragmented reads,
excluded-byte integrity, failure cleanup, exact budgets and signed overflow.
Required Rust/default/Metal/iOS checks and Astra review are recorded before
commit; preliminary source and outputs remain retained.

See [storage contract and accounting](gemma4-payload-storage.md) and the
[receipt](../benchmarks/results/2026-10-25-gemma4-payload-storage-v1.json).
The real container is not downloaded by this task. Acquisition/load and E2B
geometry, generation/Metal/quality/performance, Mac/phone fit and app stages
remain open. No new family/ABI/catalog limit is granted and all upstream
precision discrepancies remain unwaived.

## 25 October 2026 — downloadable Qwen3 Chat on iPhone

Add the pinned Qwen3-0.6B eight-bit non-thinking profile as the fourth catalog
entry, with six immutable files totaling 644,927,148 bytes. Discovery queries five
bounded public Hugging Face pages. Preserve graph/format/tokenizer eligibility,
download integrity, estimated ranking and fresh memory admission after releasing
the previous model. The signed app uses the accepted 40bf67f native package;
parallel Gemma Rust changes are outside that immutable package.

All seven physical UI cases pass on iPhone 14 Pro / iOS 27.0.1, including repeated
Qwen3/Qwen2/Smol switching, real replies after background/re-selection, colored
Python with successful Copy feedback and existing Expert fixture controls.
A separate fresh transfer, offline controller generation, prefix rejection,
cancellation/resume and handle release pass. All three eight-token native
replies match Mac IDs. Owner saved-chat bytes are unchanged; the updated app is
normally launched and a separately signed/pinned rollback copy is preserved.

The fresh process lifetime physical peak is 893,274,528 bytes, within its 2,304 MiB
reservation. Three 64-token measured benchmark runs report 840/846/846 ms TTFT and
9.3 tok/s each; these are short UI observations, not controlled performance or
quality claims. Ten owned local commands, eight interface checks, signed
app/runner checks and ten publication fault tests support the source-bound
receipt. Four earlier locator-failed phone trials retain raw evidence and
successful owner-byte/rollback checks.

See the [model and verification report](mobile-qwen3-downloads.md) and
[phone receipt](../benchmarks/results/2026-10-25-mobile-qwen3-phone-v1.json).
Chat/raw/context limits remain 256/128/512. Gemma 4 real-model/device admission and
131,072 output tokens plus prompt/history remain separate prioritized engine
gates; this mobile addition does not qualify them.

## 25 October2026 — immutable Gemma4 checkpoint and Mac storage admission

Add a bounded data-only downloader for the pinned Gemma4 E2B four-bit candidate
and an explicit native cold-load example. Nine files total3,583,087,091 bytes.
Range/response checks, exact Git/LFS identities and the whole-container hash
precede atomic publication. Retain the failed initial transfer and all four
rejected/canceled rows; reviewed recovery reuses195 verified ranges and fetches
17missing ranges, with226 actual HTTP attempts across both runs.

The fresh native process admits all540 text views,278 packed matrices and262
FP32 vectors. Tensor capacity is2,604,553,356 bytes; conservative load admission
is2,671,730,828 bytes. On the M1 Pro Mac, observed maximum resident memory is
1,520,435,200 bytes and peak physical footprint is2,616,527,296 bytes. A single
debug storage validation takes183.83 seconds after file hashing; these are
storage observations, without inference or controlled performance qualification.

Eleven portable acquisition tests and four recovery fault regressions pass.
Formatting, strict default/Metal Clippy,291 CPU and313 Metal-feature tests, both
iOS compile targets and the example build pass; numerical fixtures/tolerances
are unchanged. Astra reviews source and final bound evidence before commit.
See the [scope and result](gemma4-checkpoint.md) and
[receipt](../benchmarks/results/2026-10-25-gemma4-checkpoint-v1.json).
Real E2B geometry/execution, tokenizer/chat/generation, Metal, quality/performance,
phone fit and app support remain open. All existing GPU/long-rotary failures
remain unwaived; no ABI/catalog/context limit changes.

## 25 October 2026 — bounded multi-producer packed Gemma staging

Replace the fixed two-owner proposal sequence with a validated private producer
prefix and persistent session staging vector. Keep up to 15 borrowed proposals
in an inline container through all shared consumers and the final head. All-cache
validation precedes public KV/history/logit commit; failure preserves the prior
prefix. Account actual staging capacity and every K/V allocation. The checked
inline container uses 976 transient stack bytes separately from persistent memory.

Three small structural tests cover 35 layers / 15 producers, routes 13/14, chunk
agreement, physical cache slots, reset, 38 failure/retry boundaries, zero Rust
allocations on failure/retry and 32 decode steps, independent producer bounds
and exact budget/capacity accounting. Existing four-layer numerical fixtures,
tolerances and allocation checks remain in force. Required Rust/default/Metal/iOS
checks and Astra source/evidence review precede commit.

See [ownership and verification](gemma-producer-staging.md) and the
[receipt](../benchmarks/results/2026-10-25-gemma-producers-scalar-v1.json).
The public constructor remains the frozen four-layer profile. Reused synthetic
weights give logical accounting, not unique physical residency or E2B parity.
Real checkpoint execution, generation, Metal, quality/performance, phone fit and
app gates remain open; no ABI/catalog/limit changes or waived precision failures.

## 25 October 2026 — bounded independent four-bit matrix reference

Add independent file-row lookup and tiled projection for packed4/group64/BF16
weights. Reconstruct once to FP32, then accumulate projections in NumPy/BLAS
FP64 with explicit output precision. Keep exact-affine and reconstructed-FP32
contracts distinct with an authored cancellation case. Validate extents, unused
codes and array reservations before reading/expanding weights.

Eight offline tests, three captured-output regressions and three cold-runtime
admission tests pass. On the actual pinned Gemma4 checkpoint, all834 selected
rows from278 text matrices match independent scalar decoding bitwise. Two full
projections, including the12,288-column MLP, pass all eight scalar-fsum/sequential
comparisons under unchanged tolerances. The reference process reports
161,611,776 bytes maximum resident memory and76,759,832 bytes peak physical
footprint, below its separate256MiB allowance.

Authenticate cold runtime files, actual imported/native origins and first-captured
output buffers. Retain the initial receipt-field error, rejected unexecuted
controller and failed first real runtime-admission/resource-query attempt. The
reviewed retry preserves numerical and memory gates. Rust engine, tests,
fixtures, examples and protected binaries remain unchanged fromab408f4.
See [precision, resources and scope](affine4-tiled-reference.md) and the
[receipt](../benchmarks/results/2026-10-25-gemma4-affine-tiles-v1.json).
Full E2B graph, tokenizer/chat/generation, Metal, quality/performance, phone fit
and app support remain open; no ABI/catalog/limit changes or waived failures.


## 25 October 2026 — complete short E2B mathematical reference

Implement independent bounded text-graph math over file-backed packed matrices,
with named tiny/E2B geometry, explicit precision boundaries, independently
computed rotary angles, stable reductions, and bounded softcap temporaries.
Preserve all180 frozen tiny-state comparisons while streaming real states
instead of retaining an expanded model or complete live graph.

Eight graph and eight established affine methods pass. The actual immutable E2B
checkpoint produces713 states per precision contract for four raw input IDs.
All1,426 changed-final-token prefix checks match exact typed bytes. The collector
independently replays713 cross-precision comparisons using scalar arithmetic;
five arrays reject57 values, with maximum error-to-bound fraction1.54940274476271.
Retain both contracts, full raw observations and the compact68,448-byte probe
fixture. No tolerance or quality gate changes.

The successful owned reference process reports270,467,072 bytes maximum resident
memory and101,827,400 bytes peak physical footprint under its separate512MiB
allowance. Admit the cold dependency/native closure before NumPy import; retain
first-captured output buffers and verify source/model/binary endpoint guards.
Keep the early per-contract count assertion failure, contaminated source-guard
attempt and first real controller variable-shadowing failure unaccepted. Test
the actual corrected verification block and final guard before repeating it.

See [arithmetic, failures and scope](gemma4-text-math.md) and the
[receipt](../benchmarks/results/2026-10-25-gemma4-text-math-v1.json).
Rust engine/tests/examples stay unchanged fromcb2aeec. Native E2B execution,
original-BF16/source qualification, generation, Metal, quality/performance,
iPhone fit, app eligibility and131072-output-token support remain open.


## 25 October 2026 — named borrowed short E2B scalar adapter

Resolve the exact admitted Gemma4 E2B text names/geometry in a small cold module
and reuse the existing packed graph. Keep immutable store -> borrowed model ->
borrowed session lifetimes explicit, with no expanded full matrices or hot-path
name lookup. Replace logical borrowed tensor bytes once with the same owner's
actual capacity plus retained metadata; count source scalar vectors/capacity
excess and exclude load-only conversion overlap. Keep the tiny constructor's
behavior/accounting unchanged.

Separate512 physical local cache slots from32 executable positions. Offline
checks cover exact real geometry, bounded scratch, backing admission/overflow,
all15 owner snapshots on late failure/token33 rejection and persistent buffer
addresses. Existing tiny numerical gates and allocation checks remain intact.
Required formatting, default/Metal strict checks,297 CPU/319 Metal-feature tests,
both iOS targets and the explicit release probe build pass;16/109 ignored
hardware/model tests remain explicit.

The real immutable checkpoint executes four raw IDs in batched, single-token
and mixed schedules. All three full logit arrays are bitwise identical. Each
passes1,048,576 staged-FP32 comparisons under unchanged gates, with maximum
error-to-bound fraction0.0013644315853741284. Each retains27 FP64-ideal failures
(maximum1.2599434792233821) from the existing reference gap. Overall numerical
acceptance remains false; no precision/quality gate is waived.

Model accounting including its store is2,671,706,548 bytes and actual session
capacity93,555,912 bytes. The owned Mac process reports2,174,353,408 bytes maximum
resident memory and2,717,239,840 bytes peak physical footprint, below6GiB. Its
67.96-second combined load/three-schedule/output observation is not a latency
benchmark. Diagnostic copies/output are outside numerical accounting and inside
the process measurement.

See [ownership, numerical results and limitations](gemma4-short-scalar.md) and the
[receipt](../benchmarks/results/2026-10-25-gemma4-short-scalar-v1.json).
This is short scalar raw-logit execution, not completed G3 or app support.
Original-BF16/source qualification, precision gaps, tokenizer/chat/generation,
resident Metal, quality/performance, phone fit and131072 output tokens remain
open; no ABI/family/catalog/limit changes.


## 25 October 2026 — bounded packed generation-state integration

Connect the existing packed scalar graph to InferenceSession and the shared
greedy generator without modifying either numerical kernels or generation
policy. Keep sampled/input history separate from committed KV, expose only the
final evaluated vocabulary row and validate full appends/replacements before
mutation. Reuse an entire exact committed prefix; replay changed/shortened
prompts conservatively from position0.

Invalidate/reset logical state immediately and defer fallible physical reset
until new numerical work. Actual generation-exhaustion and late-forward failures
preserve explicit replay/pending state and hide stale distributions. Requests
cancelled at entry or rejected during validation preserve the entire old state;
later cancellation/callback failure preserves valid progress and pending sampled
tokens. Stop IDs remain pending and un-emitted, and continuation evaluates them
exactly once when requested.

Admit model+raw-session+wrapper bytes before allocation and verify actual
capacity afterward. Count both fixed histories, full32-row logit capacities,
cache/staging buffers and the same retained store allowance once. Two cold
accounting fields add16 bytes of model metadata; smaller requested capacities
still admit unchanged physical32-position storage.

Five new state methods pass in default, Metal-feature and release builds.
Required formatting/strict checks,302 CPU/324 Metal-feature tests and both iOS
targets pass, with16/109 ignored hardware/model tests explicit. Ninety-six warmed
decode steps and three fork replays/resets/discards allocate zero Rust heap
storage. Preserve the initial test-helper visibility compile failure without
test-pass credit; the corrected helpers retain their original numerical bodies.

See [state and accounting rules](gemma-generation-state.md) and the
[receipt](../benchmarks/results/2026-10-25-gemma-generation-state-v1.json).
Real E2B continuation, tokenizer/chat and full G3 qualification remain open.
Earlier FP64/source/GPU/long-position failures, Metal, quality/performance, phone
fit, app support and131072 output tokens remain unqualified. No ABI/catalog or
position-limit changes.

## 25 October 2026 — independent real E2B greedy continuation witness

Freeze four raw greedy selections before testing native generation. Traverse
all713 states per graph while retaining only complete final-position norm,
head and logits vectors. Staged FP32 chooses the history; FP64 is explicitly
teacher-forced on that history. Record first-index argmax, exact promoted
top-two margins and ties independently for both contracts. Flush raw selected
arrays before finite rejection, retain incomplete steps on failure, and publish
sampled IDs only after both complete traversals and captures succeed.

The actual pinned checkpoint selects107,100,45518,107 with unique maxima and
no selection disagreements. All12 unchanged last-row precision gates pass
over2,103,296 pairs; maximum error-to-bound fraction0.5103206474158878.
All six step0 vectors match the earlier accepted four-token reference bitwise.
Retain24 complete arrays/25,239,552 bytes locally and1,152 bytes of compact
probes in the committed fixture. Final state is seven committed IDs plus one
pending sampled107, ready for native continuation comparison.

Four new offline methods and eight existing graph/eight affine methods pass,
including actual-driver nonfinite and partial write/interruption/endpoint
retention. The cold process reports249,626,624 bytes maximum resident memory
and75,465,424 bytes peak physical footprint within a separate512MiB reference
allowance. The90.08-second combined reference observation is not a benchmark.
The collector independently replays all raw values, selections, margins,
prefixes and memory gates. Rust code/tests/examples remain unchanged.

See [contracts, measurements and limits](gemma4-greedy-reference.md) and the
[receipt](../benchmarks/results/2026-10-25-gemma4-greedy-math-v1.json).
This qualifies an independent short raw witness, not native generation or G3.
All earlier full-graph FP64/source/GPU/long-position failures remain unwaived;
tokenizer/chat/stop metadata, Metal, quality/performance, phone fit, app support
and131072 output tokens remain open. No ABI/catalog/position-limit changes.

## 25 October 2026 — real E2B raw generation and continuation checks

Add an explicit scalar generation example over the immutable checkpoint and
existing model-bound generation session. Stream complete vocabulary rows at
the sampling boundary before append through a statically dispatched diagnostic
observer. Preserve ordinary callback-time logits unavailability. Bound capture
to16 rows/16MiB with a64KiB writer, small descriptors and no full-row clones.
Keep kernel, session and shared generation-policy code unchanged.

The four real Mac scenarios cover uninterrupted output, two-plus-two
continuation, cancellation after two visible tokens then resume, and explicit
caller-stop107 followed by continuation. All select107,100,45518,107 and finish
with seven committed IDs plus one pending sampled107. Caller stop emits only
the final three IDs; it does not qualify model EOS metadata. Every captured
prefix, emitted sequence, report counter/reason and final memory/state invariant
passes independent replay. All16 distributions are bitwise identical across
schedules. All32 unchanged precision gates pass over8,388,608 values, maximum
error-to-bound fraction0.5103206474158878 against FP64 and0.0013656471300556837
against staged FP32. Earlier full-prompt/intermediate failures remain unwaived.

Three new capture tests verify exact finite/nonfinite raw bytes, hashes,
offsets, prewrite limits and an actual read-only-file write error. Required
formatting/strict checks,302 CPU/324 Metal-feature tests, both iOS targets,
release example and five release state methods pass. Numerical admission is
2,765,262,700 bytes. The actual process reports2,143,485,952 bytes maximum
resident memory and2,710,440,456 bytes peak physical footprint, below6GiB.
Its138.13-second combined diagnostic observation is not a performance benchmark.

See [scenarios, ownership and remaining gates](gemma4-raw-generation.md) and the
[receipt](../benchmarks/results/2026-10-25-gemma4-raw-generation-v1.json).
This is bounded short raw continuation evidence; full G3, tokenizer/chat/model
stop semantics, source-BF16, existing precision/GPU/long-rotary gaps, resident
Metal, quality/performance, phone fit, app support and131072 output tokens remain
open. No ABI/catalog/position-limit changes.

## 25 October 2026 — pinned Gemma4 basic-text reference

Freeze exact upstream tokenizer and authenticated template behavior before
adding a native text profile. Admit a strict bounded non-thinking user/assistant
subset, reject advanced options and all24 control literals, and explicitly
resolve the selected artifact's stop IDs1/106/50. Preserve identity
postprocessor/no-automatic-BOS behavior under both encode modes. Keep raw text,
normalization, Unicode/whitespace, multiple turns and all control-token observations
alongside complete/prefix byte-fallback decodes and diagnostic-only macro cases.

Two fresh captures produce identical97-record/36,960-byte witnesses. Manually
assembled turn layout matches all six rendered prompts, and an independent
scalar vocabulary decoder replays all Replace/ByteFallback/Fuse outputs and
prefixes. Invalid UTF-8 can rewrite the entire fallback run; leading spaces are
not stripped. This informs the native stream design without granting append-only
streaming or a separately implemented BPE oracle.

Cold admission authenticates1,970 Python/package/native files, source/native
loaders before import, actual origins afterward and exact endpoint lengths and
hashes. Seven offline methods pass, including source shadowing, actual partial
driver failures and signal restoration after publication failure. Retain the
initial unresolved-temporary-path test failure without acceptance credit; the
corrected five-method and final seven-method attempts pass with clean guards.
Separate reference-process resident peaks649,592,832/671,023,104 bytes and
physical peaks380,700,976/391,661,968 bytes fit1GiB. The2.10/1.94-second combined
observations are not performance benchmarks. Rust sources/tests/examples remain
unchanged; no unrelated engine build is rerun.

See [subset, source policy and decoding evidence](gemma4-text-codec-reference.md)
and the [receipt](../benchmarks/results/2026-10-25-gemma4-text-codec-reference-v1.json).
Native text/streaming, formatted continuation, full G3 and all earlier numerical,
source, Metal, quality/performance, device/app and131072-output gates remain
open. No ABI/catalog/position-limit changes.

## 25 October 2026 — dedicated native Gemma4 text codec

Add a small concrete text module outside kernels and existing app profiles.
Admit a conservative1GiB codec allowance before any asset access; authenticate
small profile files first and hash/parse the same captured tokenizer buffer.
Count tokenizer construction overlap, persistent tables, metadata, precomputed
token lengths and bounded operation buffers within that allowance. Caller-held
responses/history and numerical allocations remain separate owners.

Render typed duplicate/unknown-safe basic messages/options with exact manual
turn layout, reserved-literal/C0 rejection, canonical UTF-8 size and exact output
admission. Keep raw control-token observations separate from Chat rejection.
Validate IDs and conservative decode bytes before tokenizer output allocation,
using cold u16 lengths rather than repeated vocabulary clones. Separate raw,
rendered and decoded64KiB bounds and the diagnostic codec ID cap from numerical
position/output admission. Expose selected stops explicitly without changing
generation policy. No thinking stripper or append-only stream is added.

Five offline methods and all97 real-tokenizer records/285 prefix observations
pass against the frozen witness. Formatting, strict checks,307 CPU/329
Metal-feature tests, both iOS targets, release example and five release methods
pass. Retain the initial strict-style failure with no test acceptance credit.
Three actual-controller mutations of complete inventories keep parity false.
On the named Mac, resident/physical peaks515,948,544/488,866,560 bytes fit the
separate1GiB codec gate; the1.73-second combined observation is not a benchmark.

See [ownership, bounds and parity](gemma4-native-codec.md) and the
[receipt](../benchmarks/results/2026-10-25-gemma4-native-codec-v1.json).
Formatted model continuation, text streaming, full G3, earlier numerical/source
gaps, Metal, quality/performance, combined/mobile admission, app support and
131072 output tokens remain open. No Profile/ABI/catalog or numerical-limit
changes.

## 25 October 2026 — bounded mobile generation consumer foundation

Commit `d96fbffbea66256e3f01a6947adb9ab36c4cc35c` adds strict versioned Swift
per-mode capability metadata and an ordered bounded response consumer. Immutable
model identity, request-relative generated counters, full session state,
caller-owned UTF-8 storage and terminal sealing are explicit. Metadata admission
does not authorize model loading or generation.

Astra's three findings are closed: replay cannot claim history beyond the
request, sampled output requires the whole prompt committed, and pre-Foundation
numeric grammar admits exact checked UInt64 digits without rounded fractions or
exponents. Five same-closure version/build/iOS/simulator/synthetic checks pass.
A synthetic131072-token-count/4MiB text run retains8,404,960-byte Array capacity;
Mac resident/physical observations are73,515,008/67,420,664 bytes. These exercise
the consumer only and do not qualify real generation or phone fit.

See the [mobile contract](mobile-generation-contract.md) and
[actual-results receipt](../benchmarks/results/2026-10-25-mobile-generation-consumer-v1.json).
Astra accepted the exact seven-file packet and refreshed evidence. Existing app
profiles, UI, header/native ABI, packages and installed application are unchanged.
The engine publisher, owned cursor, combined admission, real model/device checks
and131072-output capability remain open.

## 25 October 2026 — formatted Gemma4 basic-hello reference

Add an explicit closed `basic-hello` profile to the independent greedy witness;
the original four-step raw default and accepted history remain intact. Admit
only the authenticated text receipt/fixture's ten-token prompt and stops1/106/50,
16 sampled IDs and at most25 evaluated positions. Both precision contracts
traverse all713 finite states per prefix. Publish samples only after both full
traversals/captures; stops remain pending and hidden, while output-limit tokens
remain pending and visible. Keep capture integrity separate from numerical and
selection/stop diagnostics.

Six offline greedy methods plus eight graph/eight affine methods pass, including
first/final stop, no-stop limit and partial retention between precisions. Actual
controller tail tests accept four valid cases and reject nine corruptions. An
earlier narrower controller draft missed sample-count shadowing in the inner
numeric loop; it was corrected before model execution and its source/results
remain explicitly superseded, without execution acceptance credit.

The actual reference samples11 IDs and stops naturally at106. Ten visible IDs
decode to “Hi! How can I help you today? 😊”; committed length20/pending1 preserve
the complete valid prompt/reply prefix. All33 selected-row gates pass over
5,784,064 values, maximum fraction0.7812509635026432, with zero selection/stop
membership disagreements. Scalar replay authenticates all66 arrays/69,408,768
bytes, exact maxima/margins/ties and the decoded output. Eight probes per vector
form a3,168-byte portable witness. Resident/physical peaks343,932,928/152,011,976
bytes fit the separate512MiB reference allowance;366.02 seconds combined is not
a representative latency measurement.

See [workload, state and evidence](gemma4-basic-chat-reference.md) and the
[receipt](../benchmarks/results/2026-10-25-gemma4-basic-chat-math-v1.json).
Rust numerical/test/example sources remain unchanged from `e20e601`. Native
formatted generation and streaming are next. Full G3, earlier precision/source
failures, Metal, quality/performance, combined/mobile admission, app integration
and131072-output qualification remain open; no ABI/catalog/position limits change.

## 25 October 2026 — native formatted Gemma4 generation

Extend the existing diagnostic with an explicit closed `basic-hello` profile;
keep the raw default and its four schedules. Compose one pinned codec/model and
reset session across uninterrupted and split2+remaining replies. Reuse selected
stops1/106/50 on every call, resume only after output limit and decrement the
16-token budget by sampled count. Preserve hidden pending natural stops,
completed earlier reports and valid state on later failure. Capture the full
logit row before sample append, stream bounded diagnostics and decode only the
visible IDs after generation. Keep primary capture I/O failures visible.

Before asset access, reserve1GiB codec and16MiB diagnostic/text ownership from
6GiB; pass the remaining numerical budget consistently and count model/store
once. Seven offline/release methods cover both raw/new capture bounds, closed
profile/pre-open memory rejection, early/final stops, no-stop output limit and
second-call output failure. Both strict checks,307 CPU/329 Metal-feature tests,
both iOS targets and release compilation pass. Retain the earlier six-method
preliminary result before adding direct formatted capture-bound coverage.
The full controller-verifier regression rejects six corrupt packets and records
a finite numerical failure without falsifying capture success.

Actual Mac uninterrupted and split runs match all11 frozen sampled IDs, emit
“Hi! How can I help you today? 😊” and finish with20 committed IDs/pending106.
All22 full rows match bitwise across schedules;44 unchanged reference gates
pass over11,534,336 pairs (max FP640.46041798867955425/FP320.0013559174399768907).
Combined admission is3,855,781,740 bytes; resident/physical process peaks are
2,273,132,544/2,900,626,216 below6GiB. The188.52-second combined diagnostic is
not a representative latency benchmark.

See [workload, ownership and checks](gemma4-basic-chat-generation.md) and the
[receipt](../benchmarks/results/2026-10-25-gemma4-basic-chat-generation-v1.json).
A single formatted reply advances native basic-text execution only. Append-only
streaming, broader quality/full G3, earlier precision/source failures, Metal,
performance, mobile fit/app integration and131072 output tokens remain open.
No public app Profile/ABI/catalog or numerical position/output limits change.

## 25 October 2026 — bounded Gemma4 no-Strip text streaming

Split a private owned ResponseState from the borrowed tokenizer facade and
select ByteFallback/leading-space behavior in cold pinned factories. Reuse the
existing byte-level and Mistral paths; Gemma retains leading spaces. Preserve
whole fallback runs across skipped special tokens, finish on visible stop/cancel
prefixes and poison failed mutations without leaking partial new chunks. No
full-response IDs/text or per-token full-prefix decode is retained. A Gemma
factory separately admits 4 MiB before allocation and reports checked actual
capacities, counting the borrowed codec once. Per-token temporary String
allocations remain; delayed fallback output has no per-token latency guarantee.

Freeze 317 actual sparse vocabulary entries and decode-only expectations before
decoder changes: 59 upstream cases plus generated/edge cases and five synthetic
131072-ID patterns, including a late invalid byte after a long valid run.
The isolated source/data extractor fits its 1 GiB gate with 370,819,072 resident/
281,084,696 physical bytes. Preserve pre-change input identities and runtime
alias/target byte pins; no encoder or numerical capacity follows from this data.

All 310 CPU/332 Metal-feature tests, both strict checks, both iOS targets,
22 release text methods and four probe retention methods pass. Preserve the
initial compile/type/test-call failure and later preliminary snapshots without
upgrading their scope. Probe prefix mismatch, expected decoder failure, partial
framing and publication I/O failures now retain actual outputs, active stage and
original error. Existing real profiles pass 233,585 vocabulary IDs/33 short/eight
long cases. The actual controller/classification rejects seven corrupt packets.

The real Gemma text probe checks 268 small case/partition records, 472 prefix
chunks, 15 long records and all 262,144 isolated IDs. All 7,675,150 captured bytes
replay against the independent scalar vocabulary oracle. Actual maximum stream
ownership 2,097,419 bytes (single-ID 4,361) fits 4 MiB; native resident/physical
544,669,696/489,227,032 fit 1,094,713,344 combined codec/stream/diagnostic admission.
The 3.87-second combined observation is not an LLM performance benchmark.

See [ownership, behavior and limits](gemma4-native-stream.md) and the
[receipt](../benchmarks/results/2026-10-25-gemma4-native-stream-v1.json).
This qualifies the pinned text mechanism, with broader quality/full G3,
earlier precision/source failures, Metal/performance, mobile fit/native ABI/app
integration and 131072 LLM output tokens still open. No Profile/catalog/position
or app output limit changes.


## 25 October 2026 — short Gemma reply and stream composition

Add an explicit basic-hello-stream diagnostic, preserving prior raw/basic-hello
request policies and reserves. Admit another 4 MiB before asset reads, keep one
stream across a split OutputLimit, deliver only visible callback IDs and flush
once at actual terminal completion. Cancel after the second successful delivery
without continuation. Copy each new chunk before mutation/comparison; retain
numerical pending state, completed reports, partial delivery and original errors.
Bound the diagnostic ledger to 16 callbacks plus one terminal entry.

All 310 CPU/332 Metal-feature tests, both strict checks, both iOS targets and
12 release methods pass. Five new methods check the actual scheduling helper,
including a whole fallback run across the split, cancelled-prefix finish,
second-request callback failure, sealed-terminal failure and admission before
asset access. Decoder correctness remains supported by its independent fixtures.
The actual controller rejects 12 corrupt packets and keeps finite numerical
failure distinct from successful process capture. The owned regression's
source/log/cleanup proof covers success only; preserve the first identical result
without standalone invocation/log credit. Synthetic owner sizes grant no memory
or model qualification.

Actual Mac runs match the frozen eleven samples and exact full reply in both
full schedules. The third run stops at “Hi!” after two visible IDs and retains
11 committed plus one pending ID. Every text prefix matches the independent
scalar oracle. All 24 complete logit rows / 25,165,824 bytes match across schedules;
48 unchanged gates pass over 12,582,912 values. Stream ownership stays 65,816 bytes
within 4 MiB. Combined admission is 3,859,976,044 bytes; resident/physical peaks
2,054,062,080 / 2,776,222,504 fit 6 GiB. The 236.63-second combined observation is
not a controlled latency benchmark.

See [ownership, checks and limits](gemma4-stream-generation.md) and the
[receipt](../benchmarks/results/2026-10-25-gemma4-stream-generation-v1.json).
This closes one short diagnostic composition, with full G3/quality, earlier
precision/source/GPU/rotary gaps, resident Metal/performance, qualified model/
codec identity, native ABI/app, phone fit and 131072 numerical output still open.
No profile/catalog/ABI/capability or numerical-limit change.


## 25 October 2026 — loaded Gemma content identity

Retain exact config/index SHA-256 and the completed consumed container SHA-256/
byte count in a private-field immutable identity. Hash the same metadata buffers
admitted by inspection; count successful chunks with checked arithmetic,
including excluded modalities. Publish only after EOF, digest/count equality,
finite values and capacity checks. The named E2B model copies the store evidence
only after construction succeeds; tiny/manual constructors remain unbound.
No file reopen, numerical kernel change or per-token hashing is added.

The 104-byte store value stays within the existing 64 MiB metadata reservation;
the arm64 model's optional copy adds 112 bytes through existing size accounting.
All 312 CPU/334 Metal-feature tests, both strict checks, both iOS targets and
seven payload/15 model release methods pass. Exact-byte synthetic metadata,
copied-value lifetime, off-by-one/overflow and prior late-failure checks pass.
The controller rejects eleven corrupted packets, with successful owned execution
provenance and no invocation-failure claim for the regression wrapper.

Actual Mac cold loading resolves 540 views and checks model/store identity
against exact metadata and the fixed full-container pin. Model bytes are
2,671,706,676; cold-load admission remains 2,671,730,828. Resident/physical peaks
1,602,158,592 / 2,616,641,984 fit 6 GiB. The 14.71-second combined observation is
not a controlled latency benchmark; no forward/tokenization/generation occurs.
Historical generation measurements keep their original memory scope.

See [content proof, accounting and limitations](gemma4-content-identity.md) and
the [receipt](../benchmarks/results/2026-10-25-gemma4-content-identity-v1.json).
Content evidence grants no repository provenance, tokenizer pairing, live
admission or app capability. Full G3/quality, earlier precision/source/GPU/rotary
gaps, resident Metal/performance, native ABI/app, phone and 131072-output gates
remain open, with no catalog/profile/ABI/limit change.

## 25 October 2026 — maintained Smol consumed-byte witnesses

Add separate offline producer/replay tools and frozen exact metadata/range
witnesses for the pinned eight-bit SmolLM2 checkpoint. Compute whole-container
and per-range digests from the same sequentially consumed chunks; independently
reconstruct coverage and reverse-order reads without importing the producer.
This supplies a reference for later native load-time identity, including six
repeated tied-weight reads, rather than inferring identity from a model path.

A fresh relocated execution passes twelve portable and sixteen runner methods,
two captures and two independent replays. All 694 ranges cover 142,955,136 payload
bytes; the five generated data files equal the frozen fixture byte for byte.
All six children exit successfully and are reaped/drained. Completed maximum RSS
is 32,849,920 bytes within 128 MiB; physical footprint is separately recorded.
The first rejected runner and preliminary newline-check failure remain uncredited.
Interpreter/scripts are pinned without a full standard-library/native closure.

See [design, evidence and limits](smol-load-identity.md) and the
[receipt](../benchmarks/results/2026-10-25-smol-load-identity-v1.json).
No loader, ABI, capability, app, phone, Gemma4 or long-reply qualification changes.
The next producer boundary still requires hashing the bytes actually converted
or uploaded before publishing a private immutable loaded-model identity.

## 25 October 2026 — resident standalone Metal proportional rotary

Add a fixed-shape owner with cached host FP64 coefficients, one separate precise
Metal pipeline and resident guarded buffers. Pair opposite half-head coordinates,
use full-dimension exponents and prevent product/sum contraction. Copy inactive
FP32 bits exactly. Validate bounded geometry and owned capacity before allocation;
confirm command completion before CPU access. Pre-submission errors remain
reusable; later errors poison the owner and preserve caller output.

Independent rational rounding and eight-query-head local/global witnesses were
frozen before target implementation. Actual M1 Pro execution passes eleven cases
twice, 86,640 comparisons and twelve bitwise-equal token partitions, plus rounding,
guards, negative-zero/subnormal copy and completed overflow/recreate checks.
Maximum absolute error is 5.582909308188277e-7 under unchanged tolerances. Largest
observed owner is 100,116 bytes; context's separate 64 MiB allowance is policy,
not measured residency. Completed test RSS/physical peaks are 37,715,968 /
26,460,760 bytes within a separate 1 GiB diagnostic ceiling.

All 312 CPU/335 Metal-feature tests, both strict checks, both iOS targets and
release geometry pass. Sixteen controller corruptions reject. The first GPU
child passed but its controller rejected the prefixed first record; that attempt
remains retained as failed, followed by a fresh reviewed successful run. The
initial independent rounding-generator assumption failure is also retained.

See [layout, ownership, checks and limits](gemma-metal-rotary.md) and the
[receipt](../benchmarks/results/2026-10-25-gemma-metal-rotary-v1.json).
This is a G4 mathematical building block. It does not waive source/precision gaps
or qualify resident graph execution, driver failure recovery, mobile fit,
representative performance, app capabilities or 131072 numerical output.

## 25 October 2026 — native Smol consumed-range source identity

Add an explicit audited source path for the frozen Smol checkpoint. Authenticate
metadata from the same parsed buffers and retain the original regular shard
handle. Hash each raw chunk before conversion/delivery, verify all 694 trusted
ranges and every repeated visit, and publish historical source evidence only
when complete. Sticky failure/unwind state and public post-read validation prevent
later successful calls from resurrecting proof. Identity has private fields and
grants no successful model assembly or app capability.

Seventeen new source/consumer methods pass, including actual second-chunk
UnexpectedEof, mutation/restore, repeated reads, path replacement, callback panic,
post-affine validation and invalid public arguments. All 329 CPU/352 Metal-feature
tests, both strict checks, both iOS targets and release 13/4-method repeats pass.
The initial strict-check failure is retained without credit; no warning waived.
Seventeen controller corruptions reject under the actual typed-schema verifier.

Actual source consumption checks 272 logical tensors/700 raw reads, including six
additional embedding role reads. Ledger ownership stays 121,326 bytes. Source
admission plus separate diagnostic reserve fits 512 MiB; completed RSS/physical
peaks are 36,372,480 / 35,094,912 bytes. The 1.40-second observation is not model
inference performance. The container digest is an authenticated reference anchor,
not a newly computed sequential runtime digest or current-disk snapshot.

See [ownership, evidence and limits](smol-native-content.md) and the
[receipt](../benchmarks/results/2026-10-25-smol-native-content-v1.json).
Live-model binding, codec/profile pairing, native ABI/capabilities/app integration,
phone qualification, full Gemma graph and 131072-output gates remain open.

## 25 October 2026 — audited resident Smol model binding

Add an explicit audited Metal model import that reuses existing upload/assembly.
Initialize every constructor with absent identity; attach copied historical source
evidence only after every upload and final assembly succeeds. An incomplete audit
cannot publish the model. Legacy imports remain unbound, and no public setter can
retrofit proof. Move padded-weight admission ahead of device compilation while
keeping queried buffer-length validation before weight allocation.

All 329 CPU/353 Metal-feature tests, both strict checks, both iOS targets and the
release admission method pass. Zero-selection targeted attempts remain uncredited;
corrected debug/release filters each run the intended method once against unchanged
binaries, with the full Metal suite already covering it. Six separate actual
legacy/audited processes pass
the existing graph, generation/state and 512-capacity witnesses with unchanged
numerical gates. Each graph run checks 816 snapshots/4,988,928 values; observed
stage/logit errors remain 16 / 0.0589752197265625. Audited models report all exact
reference hashes and 694 verified ranges/reads; legacy models report no identity.
Nineteen controller corruptions reject across twelve valid prefix/mode cases.

The inline option is 168 bytes within the metadata policy; model structure is
832 bytes and resident weights remain 142,955,990 bytes. Maximum audited completed
RSS/physical peaks are 260,472,832 / 249,332,408 bytes within the existing 1.5 GiB
diagnostic policy. These regressions add no new raw-array, pairwise-bitwise,
quality, long-context numerical or representative performance qualification.

See [publication boundary, checks and limits](smol-model-content.md) and the
[receipt](../benchmarks/results/2026-10-25-smol-model-content-v1.json).
Codec/profile/device admission, native ABI/capabilities and app integration remain
separate. Earlier Gemma/source/GPU/precision and 131072-output gates remain open.
