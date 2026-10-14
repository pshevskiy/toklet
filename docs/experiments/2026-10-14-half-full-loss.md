# Both fixed checkpoints pass compensated Half full loss

The explicitly selected compensated Half configuration passes the unchanged full
mean-NLL increase limit0.05 on both preselected checkpoints. Each collection scores
all10,000 frozen targets once, after both unchanged short consistency probes pass.
This qualifies full loss for this configuration only; task quality, controlled
workloads, application comparisons and performance release acceptance remain open.

| Checkpoint | Matched FP32 mean NLL | Compensated Half mean NLL | Half minus FP32 | Frozen limit |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B-Instruct |2.883725149929484|2.8836405658255675|-0.0000845841039165407|0.05|
| Preselected Qwen2.5-1.5B-Instruct |2.485958031019153|2.485924560227056|-0.00003347079209703452|0.05|

The small negative deltas are descriptive results on this fixed corpus, not a
general claim of better model quality. No checkpoint, quantizer, corpus, token IDs,
block policy or gate changes follow the results. Both Q4 v1 quality failures remain
separate and prevent the unchanged Q4 release exit from passing.

## Configuration and complete coverage

The [reviewed scorer](2026-10-14-half-scorer.md) and immutable release binary are
unchanged. Both models select `HalfCompensated` projection, Original RMS, Half KV,
all-causal-row FP32 logits, capacity1025 and full-run chunk128; compatible queried
widths are32/32. Full collection begins only after the two frozen four-token probes
pass at chunks1/128. All six scored probe rows per checkpoint preserve NLL, target
bits and complete row hashes exactly across the paired modes.

Each full worker covers10 blocks and88 completed chunks under the frozen
1024-target policy. Adjacent blocks overlap one input token; every corpus target
has exactly one causal label, row and block. At final completion the last785 inputs
are committed with pending0. Each model has five clean workers and10,134 replayable
raw events, including34 short-probe events and10,100 full events. Both corpus SHA
identities remain `f53c5b0c446959164c4d2265166040123f95be5b6d9eec784895fb45a8215beb`.

## Memory, provenance and qualification boundary

| Checkpoint | Maximum admitted bytes including application allowance | Sampled conservative worker-plus-driver native lifetime charge |
| --- | ---: | ---: |
| 0.5B |1,392,321,514|1,319,700,328|
| 1.5B |3,512,638,586|3,449,327,120|

Both fit6GiB. One worker/model is resident at a time; the two model collections run
sequentially. Builds, tests and large independent hashes do not overlap native GPU
execution. All10 workers exit0 with empty cleanup lists. Both initial/final identity
inventories are unchanged, and endpoint error lists are empty. AC power,
AC low-power mode0 and stable endpoints are recorded. Continuous thermal, energy
and system-pressure evidence is unavailable; no latency/sustained ranking follows.

The immutable scorer remains
`artifacts/m5-half-scorer-native-binary-v1/metal_quality`, with its complete
source/helper snapshot at `artifacts/m5-half-scorer-source-v1/`. Numerical source
is reviewed in `b7768f6`; scorer integration in `0c8fbb2`, the source parent for
these collections. No code changes or builds occur during the collections.
Previous checked code remains valid; this data/documentation task requires saved
event/aggregate/provenance replay and link/whitespace checks, with Astra evidence
review before commit. Protected CPU binaries remain unchanged.

[Small result](../../benchmarks/results/2026-10-14-small-half-compensated-held-out-loss-v1.json),
[small raw events](../../benchmarks/results/2026-10-14-small-half-compensated-held-out-loss-v1.raw.jsonl),
[larger result](../../benchmarks/results/2026-10-14-larger-half-compensated-held-out-loss-v1.json),
[larger raw events](../../benchmarks/results/2026-10-14-larger-half-compensated-held-out-loss-v1.raw.jsonl),
and [receipt](../../benchmarks/results/2026-10-14-half-full-loss-qualification-v1.json)
bind each completed target, stable aggregation, actual configuration, model/config,
frozen contract, matched references, binary and source identities.

## Remaining release work

Full Half loss is accepted only after this evidence review. CLI explicit-mode
qualification, all eight controlled workloads, application trace, frozen100-task
collection/literal grading and larger matched FP32 task admission remain required.
Final-configuration sustained/profile evidence and compatible comparisons follow.
Defaults are unchanged. M5.2/M5.6 and overall M5 remain open; a passing Half fallback
does not satisfy the plan's explicit Q4 quality exit requirement.
