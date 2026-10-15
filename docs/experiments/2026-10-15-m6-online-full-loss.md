# M6 Online attention full held-out loss

15 October2026. Both selected M5 representations pass the unchanged full
10000-target loss gate with explicit Online attention. Task quality,
long-context behavior, performance and M6 completion remain pending.

The [reviewed collector](2026-10-15-m6-quality-collector.md) runs both original
short single/batch guards before each full collection. The fixed corpus,
independent1025-token block policy, all-row causal scoring, FP32 baselines,
checkpoint/config, Q4 v2 artifact,128-row chunks and maximum+0.05-nat mean loss
increase are unchanged. No held-out tuning occurred.

| Selected configuration | FP32 mean NLL | Online mean NLL | Increase | Decision |
| --- | ---: | ---: | ---: | --- |
| Small compensated Half / FP16 KV | 2.883725149929484 | 2.8835716761956602 | −0.0001534737338237946 | Pass |
| Larger frozen Q4 v2 / FP16 KV | 2.485958031019153 | 2.5142232720120155 | +0.028265240992862317 | Pass |

The [full-loss receipt](../../benchmarks/results/2026-10-15-m6-online-full-loss-qualification-v1.json)
binds both archived summaries, raw logs and stderr. Ten serial owned workers
complete20268 exact events, including20000 full-corpus scored targets plus the
short guards. Every full collection includes10 reset blocks and88 completed
chunks, including the irregular final chunk. Natural exits, cleanup, state and
compiled-source/model/corpus endpoints pass.

Maximum complete admission/joint native lifetime charge is
1,377,627,122/1,305,102,376 bytes small and2,189,857,482/2,144,864,992 bytes larger.
Both fit6GiB. Native charge may omit shared/clean pages and does not replace
resident-allocation accounting. All-row loss buffers are part of admission.
These scopes reset near1K and do not qualify long-prefix memory or behavior.

Collections use revision1e71caf and the immutable quality executable/source
snapshot from the collector task. A new, untracked `metal_context` example and
bounded frozen context arrays were prepared offline during scoring; they are
outside the quality executable's source closure. No competing build, test,
model execution or large file hashing ran during the native collections.

Astra accepted the full saved-data replay and bound evidence with no findings.
Retain both Online configurations for further qualification. Existing M5
production defaults remain; small Q4 v2 and Q4 v1 rejections remain unchanged.
A full loss pass does not establish the100-case task gate, natural long-document
ability, a speed advantage or M6 closure.
