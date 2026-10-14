# Explicit compensated Half CLI qualification

`--backend metal --metal-projection half-compensated` now exposes the independently
qualified resident Half path. CPU/scalar and Metal Original remain defaults.
Readiness and completed-request statistics publish the actual weight/KV dtypes,
projection, normalization, readback and prefill chunk, plus every logical memory
component and tokenizer/application allowance. Selection happens at construction;
numerical kernels and shared input/output behavior are unchanged.

The existing CPU projection field remains compatible. Invalid, missing, duplicate
or CPU-only use of the Metal option rejects before model access. CPU equivalence
checks validate actual metadata before comparing scalar/tiled semantics. Both
feature configurations pass formatting, strict Clippy and ordinary tests:89 CPU,
98 Metal-feature and189 Python checks. Ordinary suites are separate from explicit
native qualification.

## Native functional evidence

Astra cleared production source and the bounded harness before launch. Six owned
workers run sequentially on Apple M1 Pro16GiB/macOS26.6.2 with fresh optimized
binary, queried native device access and completed GPU work. Tiny authored cases
cover default/explicit Original/HalfCompensated, four frozen greedy IDs, fully
cached prefixes, pending continuation, rejected input preserving state, reset,
hidden stop47, active SIGINT cancellation followed by continuation, and idle
SIGINT exit130. Other workers exit0. Actual readiness/statistics, full admitted
memory and valid committed/pending state are checked throughout.

The real fixed0.5B source runs raw Unicode input and the first frozen31-token chat
case, each with eight-output admission. Successful done and both reset
acknowledgements are required. Token text plus deferred text events equals
independent Python batch detokenization after each reset. Valid immediate EOS
remains distinct from an error. This covers actual finalized output; forced split
UTF8 behavior remains covered by unchanged shared decoder fixtures, not by an
invented native output claim.

All79 raw events replay, six workers clean up with empty error lists, and endpoint
identities remain unchanged. Maximum admitted bytes1,309,679,844 and sampled
worker-plus-driver lifetime native charge1,187,547,056 fit6GiB. Native counters do
not replace logical admission. AC power/low-power settings are recorded; thermal
status is unavailable. No comparative timing, task-quality or release claim follows.

The first sandbox attempt has no Metal device, fails before readiness for all six
workers, and is retained/excluded. It is an environment failure, not numerical
qualification. Fresh v2 with approved native device access passes. Earlier source
review caught a false-success text harness and Python name shadowing; offline
regressions require successful done/reset and reaching the first owned child after
fixed asset admission. Frozen full tokenizer/config/generation assets and bounded
preparse chat manifest are pinned before launch and rebound at completion.

## Provenance and next gates

Source parent`c837821`; immutable code and binary snapshots:
`artifacts/m5-cli-half-source-v1/` and`artifacts/m5-cli-half-native-binary-v1/`.
All Cargo work uses`CARGO_TARGET_DIR=artifacts/m5-checks`; protected CPU binaries
are unchanged. [Native summary](../../benchmarks/results/2026-10-14-metal-cli-half-v2.json),
[raw events](../../benchmarks/results/2026-10-14-metal-cli-half-v2.raw.jsonl),
[excluded sandbox attempt](../../benchmarks/results/2026-10-14-metal-cli-half-v1.json)
and [qualification receipt](../../benchmarks/results/2026-10-14-metal-cli-half-qualification-v1.json)
retain source/binary/asset/check identities and state/memory evidence. Astra reviews
the completed evidence before commit.

```sh
.venv/bin/python tools/experiments/metal_cli.py --binary artifacts/m5-cli-half-native-binary-v1/toklet --output benchmarks/results/metal-cli-half-rerun.json
```

Use a fresh output and native Metal access. This completes explicit CLI functional
qualification only. Controlled workloads, application comparisons, matched100-task
quality and final sustained/profile evidence remain. Both fixed Q4 full-loss
rejections still block the explicit M5 exit; full Half loss passes separately.
