# Qualified CPU CLI application adapter, 10 October 2026

The [application driver](../../tools/bench/application.py) accepts
`--cpu-projection scalar|tiled` and an optional completed CLI qualification.
Tiled measurements require that qualification before contract or model reads.
Both modes can use the same isolated, qualified CLI executable; the default
command continues to use the frozen scalar CLI with its original arguments.

The [admission helper](../../tools/bench/cpu_cli.py) verifies the completed CLI
receipt, complete compiled Rust source inventory, Cargo metadata, CLI tests,
numerical qualification, compiler/flags, build/check completion, binary bytes,
and frozen Qwen fixture identity. It requires all four functional roles, all
three frozen offset-zero input cases, independent first-token IDs and exact
scalar/candidate event/session agreement. Each case must contain one generated
token, a complete request, one pending-token continuation, another complete
request and reset, with correct evaluated/history/pending counters. All current
mode readiness/memory accounts agree and native child peaks/admission fit 6 GiB.
Receipt fields do not replace independent reference or state requirements.

The selected binary resides in isolated artifact storage. The live process must
acknowledge its selected mode at readiness and on every request; qualified runs
cannot silently fall back to scalar. Only the old default scalar CLI may omit
the new metadata. Receipt/source/binary and application helper identities are
rechecked after the worker exits, outside scenario timing. Records retain mode,
qualification hash, build provenance and every measured native/visible-text
component. No numerical kernel, precision, tokenization policy, authored
transcript, stop policy, sample count or timing scope changes.

Six [offline tests](../../tools/bench/test_cpu_cli.py) cover complete admission,
changed source/binary/compiler/flags/reference/qualification, failed checks,
missing modes/cases, incorrect output/state/ownership, legacy scalar metadata
and required live acknowledgement. Missing and supplied incomplete receipts are
rejected before contract/model readers run. They use tiny local synthetic artifacts and
mock only the already separate numerical qualification boundary, without
model downloads, GPU access or executing binaries.

All 28 offline benchmark tests pass. The
[functional qualification](../../benchmarks/results/2026-10-10-cpu-tiled-application-qualification-v1.json)
runs legacy scalar, qualified scalar and qualified tiled in separate sequential
processes. The accepted CLI code was committed as `c52af3a` first; the
[executed procedure](../../benchmarks/results/2026-10-10-cpu-tiled-application-qualification-v1.py.txt)
checks every qualified compiled source against HEAD before running checks or
model probes. The matrix parent remains held, with its boundary transport and
numerical worker already exited. Checks ran from 04:42:31 to 04:45:25 UTC;
all commands exit zero without timeout, and frozen binaries remain unchanged.
Existing root formatting/Clippy/76 Rust checks belong to the unchanged compiled
engine and accepted CLI task; they are not redundantly rerun for this Python
adapter change.

Each role runs one three-turn scenario, capped at two tokens per turn. All
nine turns agree on tokens, visible text and complete non-timing generation
state. Prefill lengths are 36, 76 and 117; each turn resets its cache and ends
with two sampled/emitted tokens and one pending token. Final reset belongs to
the authored response and completes its text before the next turn.

| Turn | Token IDs | Final visible text |
|---|---|---|
| 1 | `[1249, 9593]` | `To validate` |
| 2 | `[40, 686]` | `I will` |
| 3 | `[785, 5666]` | `The operation` |

Logical ownership and admission are identical in all three roles:
`2,353,012,560` admitted bytes. Maximum child physical peak is `2,128,513,136`
bytes for legacy scalar; qualified scalar/tiled peaks are `2,123,892,872` and
`2,123,729,056` bytes. All remain below 6 GiB. Each result and its complete
scenario JSONL are retained alongside the qualification and independently
hash-bound. Records preserve every turn's timing components, mode, executable,
source/helper, model/tokenizer, transcript and CLI receipt identities.

The qualification SHA-256 is
`4a5d4b71e4bf6dc68e295111d95f49cab9ad8b1d50ca32fec1cf8471555d8e04`;
its executed producer is
`b1cd97372937608c82b76f6da59109d5d804c8e2c922ce47ee9697317e13a666`.
Astra accepted the completed adapter task without findings after independently
checking the final code, command outcomes, exact raw/evidence copies, all nine
turns' semantics, selector/receipt/source identities and memory figures.

These probes establish wiring and visible-text equivalence for the short
scenario. Their single-scenario timings do not qualify application performance.
Complete three-warmup/ten-sample application measurements and full M2.5/M3
acceptance remain pending.

```sh
.venv/bin/python tools/bench/application.py --probe --cpu-projection tiled --cpu-qualification benchmarks/results/2026-10-10-cpu-tiled-cli-qualification-v1.json --output benchmarks/raw/application-tiled-probe.json
```
