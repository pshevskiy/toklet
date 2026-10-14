# Frozen Half application trace adapter

The explicit qualified Half CLI now runs the frozen authored three-turn application
trace for both selected checkpoints. Exact bounded transcript/template bytes and
pinned tokenizer/version independently reproduce36/76/117 prompt IDs before model
startup. Scripted assistant history, empty numerical cache per turn, greedy hidden
EOS and explicit reset/Unicode finalization preserve the original application policy.
No CPU application adapter, numerical kernel, default, model or quality gate changes.

The strict application response grammar checks actual Metal/HalfCompensated/Original/
FullLogits, complete persistent memory, token indices/EOS/caps, committed/pending
counts, finite ordered native timers and attributed formatting/tokenization/text/
publication. Driver timers record submission to first token, first nonempty text,
done and finalized reset; native components overlap and are not summed. Setup and
independent oracle decoding are outside warm timers. Scenario time includes driver
validation and per-turn memory instrumentation; it should not be interpreted as
kernel-only time. Source hashing warms files; no cold-storage or energy claim.

Check mode uses two outputs per turn, zero warmups and one scenario, and cannot
publish completed measurements. Full mode keeps128 outputs per turn, three warmups
and ten samples. Owned bounded transport validates event order and final coverage,
records separate worker/driver lifetime native charges, and exits/reaps before
independent finalized text comparison. Exact binary/source/contract/asset identities
are bound initially and at endpoints. Astra caught final thermal observation failure
losing the summary; guarded endpoint reporting now preserves primary errors/raw
samples, invalidates success and always writes the result.

Seven targeted tests and all439 offline Python checks pass. They cover frozen turn
IDs/caps, truthful modes/state/EOS/reset, attributed and receipt timers, text first
appearing at reset, bounded pre-parse identity, owned failure cleanup and final
thermal failure after both successful and failed workers. Two sequential actual
workers on Apple M1 Pro16GiB/macOS26.6.2 pass all three turns each, two visible outputs
per turn,26 raw events, exit0/cleanup/endpoints and independent finalized text.

| Checkpoint | Complete logical admission | Sampled worker-plus-driver native lifetime charge |
| --- | ---: | ---: |
|0.5B|1,443,266,720 bytes|1,331,464,136 bytes|
|1.5B|3,577,266,480 bytes|3,465,186,736 bytes|

Both fit6GiB. These charges are conservative lifetime-counter sums, not a shared
physical total. Short scenario observations are excluded from latency acceptance.
Larger FP32 raw task responses remain separate from application latency: simultaneous
resident tokenizer plus the standard FP32 CLI exceeds the initial memory policy.

Source parent `29d2705`; immutable binary remains
`artifacts/m5-cli-half-native-binary-v1/toklet`. Source snapshot
`artifacts/m5-metal-application-source-v1/` and [receipt](../../benchmarks/results/2026-10-14-metal-application-qualification-v1.json)
bind the helper and checks. [Small probe](../../benchmarks/results/2026-10-14-small-half-application-check-v1.json),
[small raw](../../benchmarks/results/2026-10-14-small-half-application-check-v1.raw.jsonl),
[larger probe](../../benchmarks/results/2026-10-14-larger-half-application-check-v1.json)
and [larger raw](../../benchmarks/results/2026-10-14-larger-half-application-check-v1.raw.jsonl)
retain responses/timings and exact inputs. Full trace collection omits `--check`:

```sh
.venv/bin/python tools/experiments/metal_application.py --model small --check --output benchmarks/results/application-small-probe-rerun.json
.venv/bin/python tools/experiments/metal_application.py --model larger --check --output benchmarks/results/application-larger-probe-rerun.json
```

Use fresh output paths and sequential native access. Full task responses/literal
scores, repeated controlled/application latency and sustained release evidence are
still outstanding. Both frozen Q4 full-loss rejections still block the explicit Q4
exit; M5 remains open.
