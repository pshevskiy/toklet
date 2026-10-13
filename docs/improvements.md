# Toklet improvement record

Updated 13 October 2026. This records implemented capabilities, measured benefits,
and the decisions that limit their use. Detailed reports retain exact revisions,
artifact hashes, raw samples, checks and Astra reviews. M0–M4 are functionally
complete; M5.1, M5.3 and M5.5 are complete in their recorded scopes. M5 remains
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
| Explicit Metal chunks through 1,024 | Checked admission and a shared-model four-chunk harness pass exact capacity/budget boundaries and larger offset/append/edit final-logit bit refinement. | [Chunk task](experiments/2026-10-13-metal-chunks.md). Default stays 128; this does not qualify every mode or full release workload. |
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
| Astra code/evidence review per task | Review findings addressed before task commits; receipts bind sources, binaries, fixtures, checks and saved results. | Each linked task report records its review and acceptance scope. |

## Rejected or inconclusive experiments

These findings prevent unsuccessful experiments from being presented as improvements.

| Experiment | Outcome and current constraint |
| --- | --- |
| Core ML / ANE supported slices | CPU slice/state behavior qualified; actual ANE acceleration unobserved. FP16 projection fails its frozen gate under both policies. No ANE backend adopted. [Closure](experiments/2026-10-11-ane-feasibility-closure.md). |
| Dynamically indexed Original Q4 tile | **28.5% slower**; rejected in favor of the bounded fixed-index experiment. [Report](experiments/2026-10-13-q4-prefill.md). |
| Larger Q4 chunk-size default changes | 256/512 improve the named 1,024-input/one-output median only **4.0506% / 2.6294%**, below the frozen 5% threshold; 1,024 is **8.4712% slower**. No larger default adopted. [Sweep](experiments/2026-10-13-metal-chunks.md). |
| Original and Original-tiled expanded Q4 tiny decoder | Both fail the frozen offset17 independent gate. The compensated candidate passes; no tolerance was loosened. [Original report](experiments/2026-10-13-q4-prefill.md), [compensated report](experiments/2026-10-13-q4-compensated-storage.md). |
| 0.5B Q4 held-out quality | Mean NLL **3.057166039390558** vs FP32 **2.883725149929484**; increase **0.173441 > 0.05**, rejected. [Result](experiments/2026-10-13-metal-held-out-loss.md). |
| 0.5B Half full-loss collection | Short batch/single NLL guard **0.004681 > 0.001** fails; collection stops before full loss. No full Half quality conclusion. [Result](experiments/2026-10-13-metal-held-out-loss.md). |
| Larger Q4 held-out quality | Mean NLL **2.588645591385506** vs matched FP32 **2.485958031019153**; increase **0.10268756036635285 > 0.05**, rejected despite short numerical parity. [Result](experiments/2026-10-13-larger-metal-loss.md); `a082be5`. |
| Native macOS generated-code isolation prototype | Fixed resource admission fails; not adopted. Literal grading instead uses the separately qualified Linux controller. [Native rejection](experiments/2026-10-10-code-isolation.md), [Linux qualification](experiments/2026-10-11-linux-literal-controller.md). |

The fixed quality gates, quantizer, checkpoint selection and corpus remain
unchanged. GPU greedy/fusion evaluations, full GPU workload/task
and application comparisons, sustained qualification and M5 exit remain open.
Long-context attention/KV experiments and physical iPhone work belong to M6/M7.
This record will be updated with each reviewed implementation or measurement
decision; unmeasured changes will remain explicitly marked.
