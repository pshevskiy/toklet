# Full frozen loss checks for the public MLX Qwen checkpoint

The [scoring stream validator](../tools/experiments/m7_physical_quality.py) now
admits an explicit Mac proof scope. Its physical default still requires signed
installation proofs, foreground UIKit and valid battery observations. Mac
callers must explicitly select a Boolean `require_installation=False`, provide
no signed bundle or installation rows, and claim no UIKit observations. Both
scopes retain the complete target order, corpus SHA, block positions, finite
FP32 logits, stable loss aggregation, lifetime memory chronology and close/release
checks. Three new portable methods cover short/full Mac streams and malformed
scope, target, memory and summary variants; all 54 relevant regression methods pass.

The full workload uses the unchanged 10,001 token corpus / 10,000 scored targets,
blocks of up to 1,024 targets with one-token overlap, capacity 1,025 and chunk one.
The [public checkpoint](mlx-real-oracle.md), native library, Swift app sources and
frozen quality thresholds remain unchanged. A captured immutable FP32 baseline
and frozen contract precede execution. The prior compiled Mac runner is admitted
by its binary and complete compiled native/Swift source identities; current
Python validators are bound separately. Phone collection uses a fresh tiny
observation and the [guarded transfer/ready transport](mlx-phone-jobs.md).

The comparison is the frozen original Qwen-family FP32 application-quality gate.
The public quantized snapshot's model card does not prove the exact revision of
its prequantization source, so this is not a controlled same-source attribution
of quantization error. Successful collection and passing the numerical loss gate
are separate results. A loss failure must remain a failed gate; it cannot be
converted into acceptance by the independent stage tolerances or a threshold change.

The completed M1 Pro run scores all 10,000 targets. Mean negative log likelihood
is **3.0829338398049995**, compared with the frozen FP32 mean
**2.883725149929484**: an increase of **0.19920868987551543 nats/token**.
It **fails** the unchanged maximum increase of **0.05**. The functional execution
and full stream checks pass, while this checkpoint remains rejected for adoption
under the application loss policy. This result alone does not attribute the
difference to native execution, quantization or an unproven source revision.

The iPhone 14 Pro (iPhone15,2, iOS27.0.1) also scores all 10,000 targets and
produces exactly the same mean, increase and **failed** loss-gate decision.
All 10,000 paired greedy IDs, target logits and negative log likelihoods match
between devices. This is observed cross-device agreement for this workload,
not acceptance of the failed application-family gate.

| Device | Mean loss | Increase over frozen FP32 | Loss gate | Peak physical bytes |
| --- | ---: | ---: | --- | ---: |
| M1 Pro / 16 GiB | 3.0829338398049995 | +0.19920868987551543 | Fail | 306,840,368 |
| iPhone 14 Pro | 3.0829338398049995 | +0.19920868987551543 | Fail | 320,014,176 |

Mac samples observe thermal state zero (normal); phone samples observe states
zero and one (normal/fair). Both keep Low Power Mode off. These native
lifetime physical-footprint peaks are workload observations, distinct from RSS
and clean/shared-page accounting. Neither run exceeds the explicit 2.5 GiB
process trial cap. Quality collection is not a representative speed benchmark.
The phone path completes four stages and 45 transport commands (41 devicectl
commands and four host boot queries). Eight cleanup endpoints end absent;
seven are natural absences and one is an identity-checked initial preflight
SIGTERM reset before copying or scoring. Five outer commands and four portable
commands also pass. Rust, shader and Swift app bytes and protected CPU binaries
remain unchanged; no new app signing or build is required.

The [qualification](../benchmarks/results/2026-10-22-native-mlx-full-loss-v1.json)
and [raw inventory](../benchmarks/results/2026-10-22-native-mlx-full-loss-raw-v1.json)
retain the successful execution and failed loss decisions separately. All
consumed receipt/source/private/raw identities are checked through exclusive
publication, and both decisions are recomputed from the bound baseline and
contract. Retain the shared validator and these rejected-checkpoint results.
The numerical policy and existing qualified non-MLX configurations are unchanged.

Full task quality, representative/sustained performance, physical state and OS
lifecycle, chat-template compatibility, other architectures and in-app discovery
and downloads remain separate requirements. This work does not establish overall
checkpoint adoption or M8 completion.

Astra accepted this as a completed rejected-checkpoint experiment after replaying
both full streams and checking 322 pins, 99 raw files, 70 exact source copies,
54 portable methods, storage and cleanup, conditions, memory and privacy.
All 221 local document links and whitespace checks pass. The failed loss
decisions remain unchanged.
