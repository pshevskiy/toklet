# Full scalar task response collection, 11 October 2026

All 100 frozen scalar responses are collected and replayed: 40 code, 30 prose
and 30 continuation cases. The qualified CLI emits 6,393 visible tokens across
6,594 native events. Sixty-eight responses stop on EOS and 32 reach their
category output limit; none has empty visible text. This establishes complete
response collection, not a task score. Generated code execution, the tiled
100-case collection, pass-rate comparison and M2.5 acceptance remain pending.

The [collection](../../benchmarks/results/2026-10-11-held-out-task-scalar-v1.json)
and [native journal](../../benchmarks/results/2026-10-11-held-out-task-scalar-v1.json.native.jsonl)
retain every prepared request, token event, finalized response, generation state
and reset acknowledgement. Original dataset text remains in the ignored frozen
payload. The [analysis](../../benchmarks/results/2026-10-11-held-out-task-scalar-analysis-v1.json)
replays all cases with the protected task semantics and verifies journal order,
exact finalized text and IDs, EOS/output limits, sampled versus committed history,
one pending token, independent resets and conservative memory accounting.
No generated program is compiled or executed by this replay.

The clean engine revision is `e65c77d695dc3c1890a69c88acc470460e1923bc`.
The executable is the existing qualified isolated CLI, SHA-256
`8e1be695229fb4a46e49725f06050b3296f6fb398e718fca8567bf0c6b400b43`.
It uses scalar projections, capacity 2,048, prefill chunk 128, raw prepared
token IDs, greedy sampling and enabled EOS. Output caps remain 256/32/16 for
code/prose/continuation. Weights and KV retain the qualified precision; the
numerical source and executable are unchanged. This request policy differs
from the chunk-one full loss comparison and requires its separate task evidence.

| Memory charge | Bytes |
| --- | ---: |
| CLI logical admission | 2,353,012,560 |
| Maximum CLI own-process physical peak | 2,125,318,256 |
| Maximum driver own-process physical peak | 29,442,648 |
| Maximum conservative joint physical peak | 2,154,760,904 |

Every recorded joint charge is the exact sum of positive CLI and driver peak
values and remains below the unchanged 6 GiB gate. These are conservative
high-water charges, not instantaneous simultaneous measurements.

The [launch receipt](../../benchmarks/results/2026-10-11-held-out-task-scalar-launch-v1.json)
records 13:26:33.140373 UTC. Its collector and
[completed checks](../../benchmarks/results/2026-10-11-held-out-task-collector-checks-v1.json)
are bound to the checked commit: 172 offline Python tests, formatting, locked
offline Clippy with warnings denied and 76 Rust tests, using the isolated target.
The [owned process observation](../../benchmarks/results/2026-10-11-held-out-task-scalar-owned-v1.json)
binds launcher 91589, driver 91598 and CLI 91610, their parents, session groups
and full command texts. Collection ran with no concurrent model, VM, test or
build work. Bounded journal observations and inactive ignored source drafts
were the only preparation performed during the run.

The [actual terminal completion](../../benchmarks/results/2026-10-11-held-out-task-scalar-session-completion-v1.json)
was observed at 15:42:22 UTC: native session 1132 exited zero with the natural
completion marker. There was no watchdog expiry, group termination, cleanup
error, trailing output or surviving descendant. The
[idle proof](../../benchmarks/results/2026-10-11-held-out-task-scalar-idle-v1.json)
at 15:42:39.360415 UTC verifies all three PIDs absent and both owned groups empty,
the Linux VM stopped and the original interrupted orchestration still held.
Root release binaries and all protected model/tokenizer assets are unchanged.
Receipt times describe execution provenance; this is not a timing comparison.

The [exact replay producer](../../benchmarks/results/2026-10-11-held-out-task-scalar-analysis-v1.py.txt)
and [validator snapshot](../../benchmarks/results/2026-10-11-held-out-task-scalar-validator-v1.py.txt)
are retained with the exact launch, ownership, driver and archiver sources.
The replay verifies the complete eleven-file archive inventory before loading
the validator, the ordered preflight checks and isolated target, fixed clean
revision, process graph, chronological receipt order, binary/compiled source
and helper bindings, frozen model/contract/assets, all responses and journal
hashes. It explicitly reports `quality_completed: false` and no aggregate.

Astra independently replayed the exact archived producer to a fresh output and
confirmed byte-identical analysis. The raw collection, journal, stderr and
launch copies also match exactly. Astra accepted the complete source, provenance,
response, memory and documentation task with no remaining actionable findings.
All 144 local links across this report, README and PLAN resolve; content and
whitespace checks pass. No quality threshold or milestone status has changed.
