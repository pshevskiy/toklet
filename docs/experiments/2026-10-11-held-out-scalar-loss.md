# Full scalar held-out loss, 11 October 2026

The complete unquantized scalar reference scores all 10,000 frozen held-out
targets with mean negative log likelihood **2.883725149929484 nats/token** and
total **28,837.25149929484 nats**. This establishes the scalar loss baseline.
The separate 100-task pass-rate evaluation, full candidate loss and workload
qualification, and M2.5 acceptance remain pending.

The [completed record](../../benchmarks/results/2026-10-11-held-out-scalar-loss-v1.json)
retains every target label, target logit, normalization value and loss. The
[native journal](../../benchmarks/results/2026-10-11-held-out-scalar-loss-v1.json.native.jsonl)
contains all 10,012 ordered events: admission, 10,000 losses, ten block summaries
and completion. The
[saved replay](../../benchmarks/results/2026-10-11-held-out-scalar-loss-analysis-v1.json)
independently sums each block and the complete corpus and checks every event
against the native stream. Its
[exact producer](../../benchmarks/results/2026-10-11-held-out-scalar-loss-analysis-v1.py.txt)
also verifies the frozen contract, source, helper, binary, launch, check and
completion identities.

The frozen context policy uses independent blocks of at most 1,025 tokens,
with one-token overlap and every target scored exactly once. The first nine
blocks score 1,024 targets and commit 1,025 tokens; the final block scores 784
targets and commits 785. Every block finishes with zero pending tokens.
The worker uses capacity 1,025, prefill chunk one, FP32 weights and FP32 KV.
The pinned Qwen2.5-0.5B-Instruct checkpoint is converted from its source BF16
weights once at loading. The corpus and policy match the
[benchmark contract](../../benchmarks/contract-v1/manifest.json).

Peak own-worker physical charge is **2,009,548,792 bytes**, below admitted
**2,136,856,460 bytes** and the 6 GiB limit. Admission includes weights, allocated
KV, scratch, history, logits, metadata and the fixed metadata/application
allowances. The diagnostic logits buffer is exposed separately and covered by
the application allowance. This physical measurement is the numerical worker's
process charge, not a joint host-driver measurement or an energy measurement.

The run uses clean engine revision
`7bc7e6376ea5224964d25c35aa5f7930d09694eb` on MacBookPro18,3, Apple M1 Pro,
macOS 26.6.2 (25G83). The isolated release worker SHA-256 is
`18c2b95fc6eab4622c9a088f522fd04e681cf148ec648d50bd72f116d0bc591a`.
Power admission reports AC power with its low-power mode off; the two thermal
queries report no recorded warning. These are endpoint observations. This
single loss pass does not establish repeated latency or sustained-power results.

Before scoring, formatting, offline locked Clippy with warnings denied, all
76 ordinary Rust tests, and three `quality_bench` example tests passed. The
driver then copied the qualified isolated target cache and built the release
example in a fresh isolated target. All six ordered commands and their raw
outputs are retained in the completed record. Root release binaries and
protected numerical helpers are unchanged. Builds, tests, other model runs
and native VM work remain outside scoring; only bounded progress observations
and cold source preparation occurred concurrently.

The [actual launcher](../../benchmarks/results/2026-10-11-held-out-scalar-loss-launch-v1.json)
starts at 09:36:19.144391 UTC. The
[observed owned process graph](../../benchmarks/results/2026-10-11-held-out-scalar-loss-owned-v1.json)
binds launcher 71230, driver 71239 and worker 71731. Session 56539
[exits naturally with code zero](../../benchmarks/results/2026-10-11-held-out-scalar-loss-session-completion-v1.json)
at the observed 11:35:03.094 UTC. The
[post-exit receipt](../../benchmarks/results/2026-10-11-held-out-scalar-loss-idle-v1.json)
verifies all three PIDs and both owned groups absent at 11:35:19.295131 UTC,
the original interrupted orchestration still held, the VM stopped, and every
protected endpoint identity unchanged. The unchanged
[driver](../../benchmarks/results/2026-10-11-held-out-scalar-loss-driver-v1.py.txt),
[worker](../../benchmarks/results/2026-10-11-held-out-scalar-loss-worker-v1.rs.txt),
[launcher](../../benchmarks/results/2026-10-11-held-out-scalar-loss-launch-v1.py.txt),
[owned observation](../../benchmarks/results/2026-10-11-held-out-scalar-loss-owned-v1.py.txt)
and [idle archiver](../../benchmarks/results/2026-10-11-held-out-scalar-loss-idle-v1.py.txt)
are retained exactly.

Astra independently replayed the complete saved analysis byte-for-byte and
verified counts, numerical totals, state, memory, identities, ordered checks
and actual completion/idle evidence without new model or native execution.
Astra accepted the complete evidence/report task with no actionable findings.
All 140 local links across this report, README and PLAN resolve, and
documentation whitespace checks pass.
