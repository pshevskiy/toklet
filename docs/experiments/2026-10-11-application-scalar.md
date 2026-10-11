# Full scalar application trace, 11 October 2026

The frozen three-turn coding/tool-result transcript completed three warmups
and ten uninterrupted measured scenarios on the Apple M1 Pro. Median scenario
latency was **263.230241 s** (minimum 260.727219 s, maximum 266.067983 s).
All thirty measured turns have identical token IDs, visible text and stop
reasons for their respective prompt. This qualifies the scalar application
trace; full held-out quality and M2.5/M3 acceptance remain pending.

| Authored turn | Prompt tokens | Emitted tokens | Stop | Median first token / visible text (s) | Median finalized response (s) | Committed / pending |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| 1 | 36 | 128 | Output limit | 11.606069 | 100.282415 | 163 / 1 |
| 2 | 76 | 18 | EOS 151645 | 24.114077 | 36.626189 | 94 / 1 |
| 3 | 117 | 128 | Output limit | 36.922609 | 125.966325 | 244 / 1 |

The hidden EOS is sampled and remains pending, so the second turn has nineteen
sampled tokens and eighteen emitted tokens. Each turn starts with an empty
numerical cache and ends with explicit reset; the model and buffers stay
resident. Later turns use scripted assistant messages rather than generated
replies. This is latency evidence, not task-quality scoring, live tool
execution or an external application-speed ranking.

## Identity, timing and memory

The launch was clean at `caabef4c5641aab939fac4618fe0652612878f85`.
The historical frozen scalar CLI has SHA256
`5250694e814ea112148a52cfdeaf5bc6286ff2804b3acfd8f89e80bc73122a59`;
this repository launch revision does not claim a new binary build.
The exact [driver snapshot](../../benchmarks/results/2026-10-11-application-scalar-driver-v1.py.txt)
has SHA256 `9bcee63ef402a416d5a8cf743a81abfd7061add578544a383b633762418b57da`.
Qwen/tokenizer revision is `7ae557604adf67be50417f59c2c2f167def9a775`,
with FP32 weights/KV, context capacity 2,048 and prefill chunk 128.
Rust is 1.90.0; Python is 3.12.8. The record binds compiler/build flags,
source helpers, source weight/tokenizer identities and transcript identity.

Driver timestamps span submission through complete JSON-line receipt, including
transport and dispatch. First visible text is the first nonempty payload;
its median equals the first-token median in each turn here. Finalization
includes explicit reset and any remaining Unicode suffix. Native numerical,
sampling, input formatting/tokenization, detokenization and publication
components overlap native totals and must not be added to them. Summing
per-turn medians is not the scenario median.

New-process submission-to-ready was 2.063503 s. Native model loading was
1.902842 s, including 1.179785 s of payload conversion; tokenizer loading
was 0.121991 s. These setup costs are retained separately from warm requests.
Source hashing warmed files, so no cold-storage claim is made.

Maximum own-child `auxiliary.phys_footprint_peak` was **2,128,758,896 bytes**;
conservative admission was **2,353,012,560 bytes**, below 6 GiB. The complete
logical inventory includes 1,976,131,072-byte weights, 50,331,648-byte allocated
KV, 10,436,608-byte scratch, history/logits, metadata and tokenizer/application
allowances. Its sum equals admission. No system-pressure or energy measurement
is available.

Hardware was `MacBookPro18,3`, Apple M1 Pro, 16 GiB RAM, macOS 26.6.2
(`25G83`). Launch power was AC, battery 80%, not charging; AC low-power mode
was zero. Start and end thermal queries reported no recorded thermal or
performance warning. These are endpoint observations, not continuous thermal,
clock, power or system-pressure qualification. The ten samples support median
and dispersion reporting, not reliable tail percentiles.

## Completion and verification

The fresh launcher started at 08:23:53 UTC. Its exact owned chain was
65037 / 65046 / 65060 (launcher / driver / CLI). Session 16633 returned
exit zero and the launcher's natural-completion message by 09:20:58 UTC.
The [completion receipt](../../benchmarks/results/2026-10-11-application-scalar-session-completion-v1.json)
retains that actual tool result. The
[idle receipt](../../benchmarks/results/2026-10-11-application-scalar-idle-v1.json)
proves all three PIDs absent at 09:21:14 UTC and records matching end hashes
for both frozen binaries and all eight model/tokenizer assets. The original
interrupted orchestration remains held; it was not resumed for this trace.
The previously restored Podman machine was stopped before launch. Tests,
builds, VM/native probes and model/binary hashing stayed outside this run.
Only small source/document preparation ran alongside it.

The saved [analysis](../../benchmarks/results/2026-10-11-application-scalar-analysis-v1.json)
and exact [producer](../../benchmarks/results/2026-10-11-application-scalar-analysis-v1.py.txt)
replay all ten raw/sample pairs, thirty response/state records, ordered receipt
times, text hashes, deterministic IDs, independent summaries, complete memory
accounting, source identities and launch/owned/exit bindings. The
[record](../../benchmarks/results/2026-10-11-application-scalar-v1.json) and
[raw checkpoints](../../benchmarks/results/2026-10-11-application-scalar-v1.json.samples.jsonl)
are byte-exact copies. Launcher/idle producers and owned-process metadata are
preserved with hashes.

The [observed driver stdout](../../benchmarks/results/2026-10-11-application-scalar-observed-driver-stdout-v1.txt)
retains the observed three rounded warmup and ten sample summaries. It is an
observed driver-output subset, not a complete raw CLI protocol log. Detailed
warmup states, individual text chunks and native per-token protocol lines were
not retained by this driver. The measured record retains token receipt times,
final text, native per-turn statistics and derived first-visible-text times.

Astra accepted the completed evidence task with no findings after independent
byte-identical replay and state, timing, memory, identity and completion checks.
All 118 local links in this report, the benchmark README and plan resolve;
observed sample summaries and whitespace checks pass. No Rust engine code or
qualified driver changed; executor drafts prepared during the
run remain separate, unexecuted work and are not adopted by this task.
