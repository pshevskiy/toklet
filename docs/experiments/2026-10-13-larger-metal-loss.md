# Matched larger Metal held-out loss

The numerically qualified larger Q4 configuration completes the frozen 10,000
targets but fails the unchanged quality gate. The collector code is checked;
this is a rejected quality result, not a passing release configuration.

| Configuration | Mean negative log likelihood |
| --- | ---: |
| Matched larger FP32 CPU reference | 2.485958031019153 |
| Larger Q4 v1 + Half exclusions/KV, compensated projection/RMS | 2.588645591385506 |
| Increase | 0.10268756036635285 |
| Maximum permitted increase | 0.05 |

The selected checkpoint remains Qwen2.5-1.5B-Instruct revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Model/config/tokenizer and unchanged
packed Q4 manifest are pinned to the previously qualified identities. The corpus
SHA is `f53c5b0c446959164c4d2265166040123f95be5b6d9eec784895fb45a8215beb`.
The matched FP32 reference is the completed larger result, never the 0.5B loss.

## Code and policy

The existing Metal row scorer gains an explicit final `larger` argument. Larger
Q4 selects the qualified compensated modes; old 0.5B arguments and Original
execution remain unchanged. Readiness reports actual geometry and modes.
The shared owned transport accepts a single readiness callback and optional
extra arguments at its coarse admission boundary; scoring, event bounds,
timeouts, group ownership, kill/reap and pipe cleanup remain unchanged.

The new larger collector verifies the exact bounded baseline bytes, all 10,000
causal labels and completed aggregation. It pins corpus, source, config,
tokenizer, Q4 artifact and the admitted contract before launching workers, then
checks all bound files again at the endpoint. The admitted contract's hash is
reasserted in the later identity snapshot to reject replacement between parsing
and collection. Evidence files are exclusive and never overwrite prior results.

Short probes use the same frozen token IDs with chunks 1/128 and the unchanged
1e-3 NLL consistency guard. Matched larger CPU probe rows replace informational
small-model logits. Full collection requires `--full`; ordinary probe admission
does not claim full quality. The full limit remains 0.05 nats/token. This metric
assesses the entire mixed-precision configuration; it does not identify a sole
cause for the loss increase.

## Completed evidence

Both short probes pass; maximum batch/single NLL difference is
0.000008572458672917449. The full worker scores every target once across 10
independent blocks and 88 chunks, including each final committed target whose
next distribution is not scored. All five workers exit 0, with empty cleanup
errors. Source endpoint identities are unchanged and endpoint errors are empty.
Maximum observed worker-plus-driver lifetime native charge is 1,606,355,152 bytes,
below 6 GiB; allocation admission also passes. Clean/shared pages may be omitted
from this native charge, so allocation admission remains a separate bound.

Commands use the isolated target directory:

```sh
CARGO_TARGET_DIR=artifacts/m5-checks cargo build --release --features metal --example metal_quality
.venv/bin/python tools/experiments/metal_quality_larger.py --binary artifacts/m5-checks/release/examples/metal_quality --output benchmarks/results/2026-10-13-larger-q4-held-out-loss-v1.json --full
```

The collector exits 2 because its fully collected quality result is rejected.
Formatting, strict Clippy in both feature configurations, 87 CPU tests,
94 Metal-feature tests and 162 experiment Python tests pass. Five new tests cover
geometry/modes/residency, matched reference and causal mutations, initial identity
replacement, owned admission failure cleanup and exclusive destinations.
Protected CPU binaries are preserved.

See the [result](../../benchmarks/results/2026-10-13-larger-q4-held-out-loss-v1.json),
[raw events](../../benchmarks/results/2026-10-13-larger-q4-held-out-loss-v1.raw.jsonl),
[matched reference](2026-10-13-larger-fp32-loss.md), and
[qualification receipt](../../benchmarks/results/2026-10-13-larger-metal-loss-qualification-v1.json).

The 0.5B rejection remains unchanged. This larger checkpoint and Q4 recipe are
also quality-unqualified. No gate, corpus, quantizer or default changes follow
this result. Task pass-rate quality, performance/sustained checks and the M5 exit
remain open. Further M5 implementation may be checked independently; a failing
quality configuration cannot be presented as a qualified performance release.

Astra accepted the collector code and rejected-result record after verifying all
63 bindings and independently replaying all 10,134 raw events. No actionable
findings remain; this review does not accept Q4 quality or M5 completion.
