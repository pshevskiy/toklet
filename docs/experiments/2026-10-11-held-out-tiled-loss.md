# Full tiled held-out loss, 11 October 2026

The tiled candidate matches the complete scalar reference exactly on all
10,000 frozen held-out targets. Mean negative log likelihood is
**2.883725149929484 nats/token** in both modes, with total
**28,837.25149929484 nats**. All 10,000 complete-logit hashes, target-logit FP32
bits and loss events agree. Mean increase and maximum absolute target loss
difference are both zero. The unchanged **0.05 nats/token** loss gate passes
for this declared context policy. The 100-task pass-rate evaluation, remaining
candidate workloads and M2.5 acceptance remain pending; scalar stays default.

The [completed tiled record](../../benchmarks/results/2026-10-11-held-out-tiled-loss-v1.json)
and [native journal](../../benchmarks/results/2026-10-11-held-out-tiled-loss-v1.json.native.jsonl)
retain all 10,012 ordered events: admission, 10,000 target losses, ten block
summaries and completion. The
[comparison replay](../../benchmarks/results/2026-10-11-held-out-tiled-loss-analysis-v1.json)
and [exact producer](../../benchmarks/results/2026-10-11-held-out-tiled-loss-analysis-v1.py.txt)
validate every native event, independently sum all blocks and targets, replay
the [complete scalar baseline](2026-10-11-held-out-scalar-loss.md), and compare
all aligned targets without selecting or dropping discrepancies.

The [frozen contract](../../benchmarks/contract-v1/manifest.json) uses independent
blocks of at most 1,025 tokens, one-token overlap and every target scored once.
Nine blocks commit 1,025 tokens and score 1,024 targets; the final block commits
785 tokens and scores 784. Every block finishes with zero pending tokens.
Both runs use capacity 1,025, **prefill chunk one**, FP32 weights and FP32 KV.
This loss comparison covers that single-token execution policy; it does not
replace quality checks for task prompts using prefill chunk 128. Source BF16
Qwen weights are converted once during loading.

The tiled worker's own-process physical peak is **2,009,532,408 bytes**, below
admitted **2,136,856,460 bytes** and the 6 GiB limit. Admission includes weights,
KV capacity, scratch, history, logits, metadata and fixed allowances. The
separate diagnostic-logit buffer is covered by the application allowance.
This is an own-worker process charge, not joint driver memory or energy.

The run uses clean revision `5ceee7d38157973466d6829a8e22d7223ff98d14` on
MacBookPro18,3, Apple M1 Pro, macOS 26.6.2 (25G83). The isolated release worker
has the same SHA-256 as the scalar loss worker:
`18c2b95fc6eab4622c9a088f522fd04e681cf148ec648d50bd72f116d0bc591a`.
Admission reports AC power with AC low-power mode off; endpoint thermal queries
report no recorded warning. This single loss pass does not establish repeated
latency, sustained-power qualification or a performance ranking.

Before scoring, formatting, locked offline Clippy with warnings denied, all
76 ordinary Rust tests and three `quality_bench` example tests passed. The
driver copied the qualified isolated target cache and built the release
example in a fresh target. All six ordered commands and raw outputs are saved.
Root release binaries and protected numerical helpers are unchanged. Builds,
tests, other model runs and VM work remained outside scoring; concurrent work
was limited to bounded observations and inactive source drafts.

The [launcher receipt](../../benchmarks/results/2026-10-11-held-out-tiled-loss-launch-v1.json)
records admission at 12:02:20.848048 UTC. It binds the completed scalar idle and
latest authored Linux idle proofs. The
[owned process graph](../../benchmarks/results/2026-10-11-held-out-tiled-loss-owned-v1.json)
binds launcher 85088, driver 85096 and numerical worker 85473. Session 50862
[exits naturally with code zero](../../benchmarks/results/2026-10-11-held-out-tiled-loss-session-completion-v1.json)
at the observed 13:15:13 UTC. The
[post-exit receipt](../../benchmarks/results/2026-10-11-held-out-tiled-loss-idle-v1.json)
verifies all three PIDs and both groups absent at 13:15:27.595318 UTC, the
original interrupted orchestration still held, the VM stopped and all protected
endpoint identities unchanged. The exact
[launcher](../../benchmarks/results/2026-10-11-held-out-tiled-loss-launch-v1.py.txt),
[owned observer](../../benchmarks/results/2026-10-11-held-out-tiled-loss-owned-v1.py.txt),
[driver](../../benchmarks/results/2026-10-11-held-out-tiled-loss-driver-v1.py.txt),
[worker](../../benchmarks/results/2026-10-11-held-out-tiled-loss-worker-v1.rs.txt)
and [archiver](../../benchmarks/results/2026-10-11-held-out-tiled-loss-idle-v1.py.txt)
are retained unchanged.

Astra independently reproduced the complete comparison byte-for-byte, verified
all numerical, provenance, completion and idle evidence, and accepted the full
report task with no actionable findings. All 145 local links across this report,
README and PLAN resolve; documentation whitespace checks pass.
