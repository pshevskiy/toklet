# Larger Half loss qualification stops at consistency guard

The preselected larger unquantized Half configuration fails the unchanged short
batch/single consistency guard. Full 10,000-target collection was requested but
never begins. This is a batch-consistency rejection, with no full Half NLL or
release-quality conclusion.

The checkpoint remains Qwen2.5-1.5B-Instruct revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, with the same source/config/tokenizer
and matched larger FP32 reference used for the Q4 evaluation. No corpus, source,
quantizer, tolerance or default changes. Original MPS Half projections and
Original RMS, Half KV and all-causal-row FP32 logits are selected explicitly by
the existing collector. CPU/scalar remains default.

## Guard result and coverage

The first frozen `future_changed` probe is `[9707, 11, 358, 498]`. Each completed
worker scores its three causal targets and commits all four inputs, with no
pending token. Capacity is1025; configured chunks 1/128 process one/four active
rows. Maximum permitted absolute target NLL difference remains 0.001.

| Target index | Single-row NLL | Batched NLL | Batch minus single |
| --- | ---: | ---: | ---: |
| 0 | 1.719989728653862 | 1.719973365330723 | -0.000016363323139 |
| 1 | 0.914020482612138 | 0.914559386708132 | 0.000538904095994 |
| 2 | 11.185940008934150 | 11.184416139882632 | -0.001523869051518 |

Maximum difference is **0.001523869051518 > 0.001**. The collector stops
before the second short probe and before any full-corpus worker. The unprocessed
probe metadata is not a larger-model measurement. The small-model Half guard
failure remains separate; neither establishes a full larger Half mean loss.
Selected-stage numerical envelopes previously pass, but they do not override
this independent consistency condition or identify the cause.

## Collection and provenance

Both owned workers exit 0, with empty cleanup lists. All 17 raw events replay
exactly, including causal labels, chunk/state transitions and stable aggregation.
Endpoint identities are unchanged and endpoint errors are empty. Maximum
allocation admission is 3,512,638,586 bytes; sampled conservative worker-plus-driver
native lifetime charge is 3,362,491,920 bytes. Both fit 6 GiB. AC power, low-power mode
zero and unchanged endpoint observations are recorded; continuous thermal,
energy and system-pressure measurements are unavailable. This is no latency
comparison.

A fresh release binary is retained at
`artifacts/m5-larger-half-loss-native-binary-v1/metal_quality`; complete sources
and helper scripts are retained in `artifacts/m5-larger-half-loss-source-v1/`.
The source parent is `9132db6`. Cargo uses the isolated target
directory and default release flags. Current numerical code is unchanged from
the checked post-discard implementation: formatting, strict Clippy in both
configurations, 87 CPU/94 Metal ordinary tests and 174 Python tests pass. The current
release scorer rebuild succeeds. Protected CPU binaries remain unchanged.

[Summary](../../benchmarks/results/2026-10-14-larger-half-held-out-loss-v1.json),
[raw events](../../benchmarks/results/2026-10-14-larger-half-held-out-loss-v1.raw.jsonl),
[matched FP32 reference](2026-10-13-larger-fp32-loss.md), and
[receipt](../../benchmarks/results/2026-10-14-larger-half-loss-qualification-v1.json)
retain this failed prerequisite. Astra independently reviews saved-data replay,
provenance, bounds and the stated rejection scope before commit.

## Reproduction and next task

```sh
CARGO_TARGET_DIR=artifacts/m5-checks cargo build --offline --release --features metal --example metal_quality
.venv/bin/python tools/experiments/metal_quality_larger.py --binary artifacts/m5-larger-half-loss-native-binary-v1/metal_quality --output benchmarks/results/2026-10-14-larger-half-loss-rerun.json --fp16 --full
```

Use a fresh output path; the guard failure is retained and the collector exits 1.
A bounded all-layer batch/single capture and identical-input replay should
localize the first differing operation before any arithmetic candidate is
implemented. Added diagnostics must reproduce the scorer's logit bits/NLL or
their perturbation must be recorded. No guard is loosened and no full-corpus
retry of unchanged arithmetic is justified. M5.2/M5.6 and overall M5 remain open
under both fixed Q4 loss failures and this unqualified Half fallback.
