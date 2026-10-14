# Both compensated Half short consistency guards pass

The explicitly selected compensated Half scorer passes the unchanged `1e-3`
batch/single NLL guard on both preselected checkpoints. All six scored rows per
checkpoint have identical NLL, target FP32 bits and full-logit SHA256 across
chunk1/128 pairs. Full-corpus scoring was excluded from this task; there is no full
mean NLL, quality or performance acceptance.

## Integration and fixed policy

The [qualified numerical mechanism](2026-10-14-half-compensated.md) is exposed only
through an explicit `half-compensated` scorer argument after `small`/`larger`
geometry. Packed/Q4 input rejects before loading. Legacy calls keep their modes
and defaults. Both collectors validate actual worker projection, Original RMS,
typed positive compatible queried widths, exact geometry and resident weight bytes.
The small collector additionally pins the known config and binds the exact bounded
contract bytes before parsing, then reasserts their SHA at initial snapshot.

Frozen checkpoints, tokenizer-independent input IDs, corpus, CPU references, block
policy, short guard and full mean-NLL increase limit0.05 remain unchanged. One
owned worker at a time uses resident weights/KV/scratch and all-causal-row FP32
diagnostics. The shared transport retains bounded nonblocking pipes, per-request
and lifetime watchdogs, idle memory handshakes and explicit cleanup. No second live
FP32 model is needed.

Small collection gains `--probes-only`; its default continues to request full
scoring. Larger collection already requires `--full` explicitly. Probe completion
is separate from full collection and quality completion. Offline control-flow
checks demonstrate that guard failure launches no full worker, probes-only cannot
claim quality, and a failed full worker keeps the result failed. A contract replaced
after parsing is rejected before worker launch.

## Executed probe evidence

Each model runs the two frozen four-token probes, `future_changed` and `short`,
with capacity1025 and configured chunks1/128. Every worker commits four inputs,
scores three causal targets and ends with pending0. Each probe pair contains17 raw
events; each model has34 events, four workers and12 scored observations. The
probes share prefixes; this is not24 distinct targets or full-corpus coverage.

| Checkpoint | Maximum batch/single NLL difference | All six scored-row hashes/target bits | Maximum admission | Sampled joint native lifetime charge |
| --- | ---: | --- | ---: | ---: |
| Qwen2.5-0.5B-Instruct |0.0| Exact |1,392,321,514 bytes|1,230,210,944 bytes|
| Preselected Qwen2.5-1.5B-Instruct |0.0| Exact |3,512,638,586 bytes|3,361,557,984 bytes|

Both run Original RMS and Half KV, with actual queried reduction widths32/32.
All eight workers exit0 and have empty cleanup lists. Endpoint identities are
unchanged with no endpoint errors. Admissions and sampled conservative
worker-plus-driver lifetime charges fit6GiB. AC power, AC low-power mode0 and stable
endpoints are recorded; continuous thermal, energy and system-pressure evidence is
unavailable. No timing comparison is inferred.

The default Original Half failures remain separately recorded:
[small rejection](2026-10-13-metal-held-out-loss.md) and
[larger rejection](2026-10-14-larger-half-loss.md). A finite short consistency pass
does not override either fixed Q4 full-loss rejection or establish acceptable full
Half loss. It permits the next unchanged full-corpus collection.

## Checks, provenance and reproduction

Source parent is `b7768f6`. Frozen source/helper and fresh release binary are retained
at `artifacts/m5-half-scorer-source-v1/` and
`artifacts/m5-half-scorer-native-binary-v1/`. All Cargo work uses
`CARGO_TARGET_DIR=artifacts/m5-checks`; protected CPU binaries remain unchanged.
Formatting, strict Clippy in both configurations,87 CPU/97 Metal-feature ordinary
tests and182 Python tests pass. Astra reviews source and independently replays the
saved probe evidence before commit.

[Small summary](../../benchmarks/results/2026-10-14-small-half-compensated-probes-v1.json),
[small raw events](../../benchmarks/results/2026-10-14-small-half-compensated-probes-v1.raw.jsonl),
[larger summary](../../benchmarks/results/2026-10-14-larger-half-compensated-probes-v1.json),
[larger raw events](../../benchmarks/results/2026-10-14-larger-half-compensated-probes-v1.raw.jsonl),
and [receipt](../../benchmarks/results/2026-10-14-half-scorer-qualification-v1.json)
retain code, binary, source/contract/reference identities, checks and observations.

```sh
.venv/bin/python tools/experiments/metal_quality.py --binary artifacts/m5-half-scorer-native-binary-v1/metal_quality --output benchmarks/results/half-small-probes-rerun.json --fp16 --half-compensated --probes-only
.venv/bin/python tools/experiments/metal_quality_larger.py --binary artifacts/m5-half-scorer-native-binary-v1/metal_quality --output benchmarks/results/half-larger-probes-rerun.json --fp16 --half-compensated
```

Use fresh output paths and run sequentially. For subsequent full collection omit
small `--probes-only` or add larger `--full`; the unchanged short probes still run
first. M5.2/M5.6 and overall M5 remain open pending full quality and release evidence.
