# M5 performance evidence and release decision, 14 October 2026

Retain compensated Half as an explicitly selected configuration with passing
frozen loss/task quality, bounded memory and repeated workload/application
measurements on both pinned checkpoints. Keep CPU scalar as default, GPU Original
projection, FullLogits readback and chunk128 defaults. M5 is **not complete**:
both unchanged Q4 v1 loss gates fail. This report closes the synthesis task,
not the M5.2 quality or M5.6 performance-release acceptance tasks.

## Configuration and quality

Measurements use Apple M1 Pro / MacBookPro18,3 /16GiB /macOS26.6.2(25G83),
Rust1.90.0 and separately pinned optimized executables. The [full small matrix](2026-10-14-small-half-controlled-full.md)
and [larger matrix](2026-10-14-larger-half-controlled-full.md) bind exact checkpoint,
compiler/build, numerical source, tokenizer and workload bytes. Both select
Half weights/KV, HalfCompensated projection, Original normalization, FullLogits,
capacity2048/chunk128. Controlled requests use128 fixed teacher-forced outputs
and ignore EOS; application uses greedy/EOS with output cap128.

| Model | Half / matched FP32 mean NLL | Increase, nats/token | Half / FP32 task passes |
| --- | ---: | ---: | ---: |
| Qwen2.5-0.5B-Instruct |2.8836405658255675 /2.883725149929484|−0.000084584104|2/100 /2/100|
| Qwen2.5-1.5B-Instruct |2.485924560227056 /2.485958031019153|−0.000033470792|11/100 /11/100|

Both [10,000-target loss](2026-10-14-half-full-loss.md) and
[100-task quality](2026-10-14-half-task-quality.md) pass unchanged0.05-nats and
2-percentage-point gates. Three literal windows resolve300 outcomes/120 code
programs without repair or unresolved infrastructure. Code passes remain0/40
per model; this is relative qualification on the frozen suite, not broad quality.
Original Half fails the short consistency gate; these results qualify only the
explicit compensated variant.

Small Q4 mean NLL3.057166039390558 increases0.173440889461074 overFP32;
larger Q4 mean2.588645591385506 increases0.10268756036635285. Both exceed0.05.
The [small rejection](2026-10-13-metal-held-out-loss.md) and
[larger rejection](2026-10-13-larger-metal-loss.md) remain binding. Numerical
oracle parity and a faster Q4 tile do not establish acceptable model quality.
No checkpoint, quantizer, corpus or acceptance threshold is changed.

## Completed workload and repeatability evidence

Each full matrix has eight rows, three warmups and ten measured requests per
row,104 requests/187 raw events with owned clean exit and complete state,
completed-work timing, memory and endpoint replay. Units are seconds except
decode steps/s, which counts127 post-first-event intervals. These are same
configuration measurements, not optimization comparisons. All raw samples,
ranges and MAD remain in the linked reports; ten samples do not qualify tails.

| Model | Workload | TTFT median, s | Request median, s | Decode steps/s | Ten measured requests, s |
| --- | --- | ---: | ---: | ---: | ---: |
| small | short | 0.484317792 | 12.192120 | 10.847458 | 121.886869 |
| small | medium | 2.359421125 | 31.840999 | 4.308656 | 319.499685 |
| small | long | 5.939004230 | 59.266852 | 2.380557 | 597.909383 |
| small | boundary | 15.210350396 | 109.611879 | 1.345305 | 1096.033071 |
| small | append | 0.528379479 | 32.838057 | 3.930478 | 328.360961 |
| small | cached | 0.000000687 | 29.379901 | 4.322683 | 293.759972 |
| small | pending | 0.208862854 | 29.636882 | 4.315615 | 296.332626 |
| small | changed | 2.350358125 | 31.715965 | 4.324617 | 317.138613 |
| larger | short | 2.037335896 | 28.398590 | 4.817349 | 283.565480 |
| larger | medium | 9.233068459 | 75.446195 | 1.917664 | 754.185183 |
| larger | long | 21.623262209 | 141.158441 | 1.062459 | 1411.587778 |
| larger | boundary | 51.331213646 | 264.684586 | 0.595285 | 2646.948165 |
| larger | append | 1.821654770 | 74.724694 | 1.741866 | 747.214432 |
| larger | cached | 0.000000750 | 66.278119 | 1.916168 | 662.712967 |
| larger | pending | 0.470977312 | 66.832540 | 1.913733 | 668.301719 |
| larger | changed | 9.275988521 | 75.584861 | 1.915350 | 755.834220 |

Cached TTFT is a near-floor retained-distribution event without numerical work;
no percentage ranking follows. Cached prefill throughput is absent/null; decode rates remain meaningful. Cache preparation occurs outside request timing;
pending-token evaluation is inside it. Changed-prefix rows time explicit reset
and replay. Boundary checks commit all2048 positions and reject another token
before mutation. These rows preserve the frozen workloads rather than relabeling
longest-prefix continuation as reset-and-replay.

The [second primary runs](2026-10-14-half-primary-confirm.md) repeat the short
row with three warmups/ten samples per model. Small TTFT/request medians change
+0.003433%/+0.087202%; larger−0.168474%/+0.005098%. The larger TTFT ranges remain
1.856106–2.040837s initially and1.850053–2.039807s on confirmation. Variation
is retained without a causal attribution; these are not paired candidate gains.

The [full application](2026-10-14-half-application-full.md) and
[confirmation](2026-10-14-half-application-confirm.md) each complete78 turns,
60 measured, with three warmups/ten three-turn scenarios per model. Every39-turn
ID/text/stop/state signature per model matches between runs. Initial/confirming
scenario medians are20.445239/20.456533s small and43.550813/43.572448s larger
(+0.055240%/+0.049678%). Prompts have36/76/117 IDs, start empty and use scripted
assistant history; final reset flushes text. Driver timing includes validation
and memory instrumentation. Native/driver components overlap and are not summed.
Only token receipt timestamps and separate-text phase bounds are archived;
full event-receipt vectors were not saved. Nullable absent text timing stays null.

## Memory, sustained scope and baseline gaps

Maximum controlled admissions are1,262,602,960/3,396,602,720bytes (small/larger).
Across initial and confirmation runs, maximum sampled joint native charges are
1,176,504,144/3,311,848,736bytes. Application admissions are
1,443,266,720/3,577,266,480bytes; sampled joint charges across both runs reach
1,332,742,160/3,467,005,408bytes. All fit6GiB. Each report separately accounts
for weights, allocated KV, scratch, native resources and driver/tokenizer state.
Native lifetime charge counters are not complete shared physical allocation
totals; a charge difference between collectors is not a numerical allocation
regression. No redundant full CPU/GPU model copy is retained.

Small boundary requests total1096.033071s, qualifying that sustained duration;
small long requests total597.909383s and do not reach ten minutes. Larger seven
rows exceed600s; short does not. Neither application's measured scenario sum
reaches600s. Controlled runs retain AC/low-power-mode-zero and thermal endpoints;
application records start power and thermal endpoints, with no end-power record.
Continuous temperature/clocks, energy and full system pressure are unavailable.
Source hashing warms storage, so setup is not cold-loading evidence. No builds,
tests, other owned numerical/VM work or large hashes overlap retained timed runs.
Owned exits and current idle evidence are reviewed in each data receipt;
historical IDs are claimed only where retained.

The [historical small CPU FP32 tile](2026-10-12-cpu-tiled-retention.md) uses
compatible frozen token/cache workloads but different weight/KV representation
and separate historical runs. Its short TTFT/request are8.993484/61.706524s,
versus Half0.484318/12.192120s. This describes two practical configurations;
it is not an alternating, matched-representation backend gain. Boundary request
is246.854774s CPU versus109.611879s Half, but boundary decode is2.103946 versus
1.345305 steps/s: Half is slower in that phase. Do not claim improvement in
every phase, infer the cause, or adopt a new default from the short result.
Simple attention remains scoped for M6.

Historical small FP32 application median129.141379s emits128/18/128 tokens;
Half emits128/14/128. Different output prevents an identical-output speedup
claim. The larger standard FP32 CLI application exceeds6GiB; its separately
bounded raw-ID quality worker is not an application latency baseline.

The [external FP32-storage matrix](../../benchmarks/results/2026-10-10-external-matrix.md)
and [native BF16 MLX baseline](../../benchmarks/results/2026-10-09-mlx-bf16-512-128.md)
remain separate representation tracks. llama.cpp uses Half GEMM operands despite
FP32 storage, lacks full held-out quality qualification and has incomplete
mapped/shared physical peak accounting. There is no matched FP16 external
matrix with full quality/memory and application qualification. No external speed
lead or same-quality practical ranking is claimed. Baseline gaps remain explicit;
repeatability of Toklet alone does not fill them.

## Retain/reject decisions and remaining scope

| Mechanism | Decision |
| --- | --- |
| Compensated resident Half and explicit CLI | Retain passing opt-in configuration on both fixed models; no default/replacement acceptance. |
| Q4 v1 artifact and fused resident projection | Retain correct experimental mechanism, no full expansion; reject both model quality configurations. |
| Accurate Q4 eight-row prefill tile | Retain experimental25.4128%/25.6841% paired512/1 gain; quality rejection still applies. |
| Required-position logits and exact-prefix continuation | Retain qualified functionality and the separately measured continuation tradeoff. |
| Chunk256/512/1024 | Keep128 default; best named gains4.0506%/2.6294% miss5%,1024 is8.4712% slower. |
| GPU greedy token readback | Retain explicit storage/traffic capability;512/128 is0.2890% slower, FullLogits default stays. |
| Residual/RMS, rotary/cache and MLP gating fusion | Reject/remove prototypes:0.553034% slower,0.014593% and0.006032% gains below gate/dispersion. |

The [prefill closure](2026-10-13-m5-prefill-closure.md),
[dispatch closure](2026-10-14-m5-dispatch-closure.md),
[continuation](2026-10-13-prefix-continuation.md) and
[improvement ledger](../improvements.md) retain hypotheses, code differences,
checks, paired measurements and rejected experiments. M5.1/.3/.4/.5 stay complete;
M5.2 and M5.6 remain open. Completing a report does not waive Q4's explicit exit.
A new Q4 candidate would need its own prior design, independent reconstruction
and frozen full quality/performance acceptance, rather than tuning to these
held-out results. Matched comparison work must address the disclosed baseline
and phase regressions before a replacement decision. M6 and M7 remain pending.

The [saved-data synthesis](../../benchmarks/results/2026-10-14-m5-performance-decision-v1.json)
recomputes medians/ranges/MAD, measured-duration flags and confirming changes,
binding exact accepted analysis/quality receipts. The [exact producer](../../benchmarks/results/2026-10-14-m5-performance-decision-producer-v1.py.txt)
and [qualification receipt](../../benchmarks/results/2026-10-14-m5-performance-decision-qualification-v1.json) retain replay/check provenance. Content, local links,
whitespace and unchanged protected CPU executable checks apply to this
source-unchanged documentation task; no new engine suite or benchmark is needed.
Astra accepted after correcting normalization-mode and cached-prefill wording,
then independently verifying all187 bindings,289 local links,16 table rows,
quality/repeatability arithmetic, duration scopes and byte-identical producer replay.
No remaining findings. M5 completion/default replacement/external
speed-lead acceptance remain explicitly false in the synthesis.
