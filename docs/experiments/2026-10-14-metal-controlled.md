# Frozen controlled Metal workload worker

A separate concrete resident Half worker mirrors the frozen CPU teacher-forced
benchmark timing without changing historical CPU code or protected binaries.
HalfCompensated/Original RMS/FullLogits, capacity2048/chunk128 and separate64MiB
application allowance are explicit. Every numerical evaluation completes through
GPU projection and host FP32 logits readback before its token timestamp. Admission
and bounded event allocation precede timing. Tokenization, sampling, detokenization,
JSON and output remain outside it; staging the final pending fixed ID is included.

## Frozen workloads and validation

The collector reuses the contract's exact arrays and request policy. Three warmups
and ten measured samples produce the existing result schema; `--check` instead
runs one request per workload, publishes no benchmark results and cannot claim
measurement completion. Initial overcapacity rejection preserves an empty state.
The changed row explicitly resets inside the timer, preserving the original
reset-and-replay comparison rather than silently using the separate M5.5 reuse
optimization. Cache preparation resets/evaluates outside timing.

| Workload | Reused | Prefill | Final committed | Pending |
| --- | ---: | ---: | ---: | ---: |
| Short128 |0|128|255|1|
| Medium512 |0|512|639|1|
| Long1024 |0|1024|1151|1|
| Boundary1920 |0|1920|2047|1|
| Append512+64 |512|64|703|1|
| Fully cached512 |512|0|639|1|
| Cached512+pending2077 |512|1|640|1|
| Changed512 reset/replay |0|512|639|1|

All request outputs are the frozen128 teacher-forced IDs, ignoring EOS. Complete
history SHA/state,128 monotonic timestamps, timer boundaries, completed last-command
metadata and persistent memory inventory are checked. Fully cached prefill
throughput is null. Boundary follow-up commits all2048 positions and rejects the
next token without mutation. This numerical follow-up is outside request timing.

Source and collector reject unknown fields, invalid tokens/capacity, out-of-range
geometry/modes, incompatible queried widths, invalid counts, nonfinite timers,
mutable inventories and incomplete finalization. Owned processes have bounded
nonblocking pipes,64KiB lines, request/lifetime/output limits and explicit reap/close
cleanup. Idle worker-plus-driver native charge is checked alongside full logical
admission. A bounded exact-byte token read is hashed against the frozen manifest
before unpacking; initial path hashes and endpoints reassert every admitted array.
All helper implementations, including aliased endpoint functions, are bound.

Offline tests cover all eight states, forbidden changed-row prefix reuse, typed
mode/geometry/budget, complete history/timing mutations, exact repetition count,
null cached throughput, token replacement after verification, every initial pin,
and owned-worker cleanup after malformed readiness. Numerical kernels are unchanged;
this task creates measurement/admission code, not a new arithmetic candidate.

## Functional check evidence

All16 sequential native workers pass: eight workloads per fixedcheckpoint,31 raw
events per checkpoint, with identical admitted inventories and valid complete
histories. Every worker exits0 with empty cleanup errors. Endpoint identities remain
unchanged without errors. Actual queried widths are32/32 on Apple M1 Pro16GiB,
macOS26.6.2. Both full-capacity follow-ups commit2048 tokens and reject the next.

| Checkpoint | Maximum complete admission | Sampled worker-plus-driver native lifetime charge |
| --- | ---: | ---: |
| Fixed0.5B |1,262,602,960 bytes|1,158,891,392 bytes|
| Fixed1.5B |3,396,602,720 bytes|3,295,317,520 bytes|

Logical admission and sampled native charge fit6GiB; counters do not establish a
complete shared physical allocation total. AC/low-power and thermal endpoints are
recorded, with unavailable continuous thermal/energy/system pressure left explicit.
These are check-only single requests, not performance acceptance. No benchmark
result rows or completed-measurement flag are published.

## Checks, provenance and remaining gates

Source parent`fdcd314`. Formatting, strict Clippy in both configurations,89 CPU,
98 Metal-feature ordinary tests and196 Python checks pass. Fresh release build uses
`CARGO_TARGET_DIR=artifacts/m5-checks`; protected CPU binaries remain unchanged.
Frozen source/helper and binary snapshots are retained at
`artifacts/m5-controlled-source-v1/` and`artifacts/m5-controlled-native-binary-v1/`.
Astra reviews source before native and independently replays completed evidence
before commit.

[Small check](../../benchmarks/results/2026-10-14-small-half-controlled-check-v1.json),
[small raw events](../../benchmarks/results/2026-10-14-small-half-controlled-check-v1.raw.jsonl),
[larger check](../../benchmarks/results/2026-10-14-larger-half-controlled-check-v1.json),
[larger raw events](../../benchmarks/results/2026-10-14-larger-half-controlled-check-v1.raw.jsonl)
and [receipt](../../benchmarks/results/2026-10-14-metal-controlled-qualification-v1.json)
bind code, exact inputs, compiled bytes, checks, memory and full-state observations.

```sh
.venv/bin/python tools/experiments/metal_controlled.py --binary artifacts/m5-controlled-native-binary-v1/metal_bench --model small --check --output benchmarks/results/half-controlled-small-rerun.json
.venv/bin/python tools/experiments/metal_controlled.py --binary artifacts/m5-controlled-native-binary-v1/metal_bench --model larger --check --output benchmarks/results/half-controlled-larger-rerun.json
```

Run sequentially with fresh outputs and native Metal access. Omit`--check` for the
required three warmups/ten samples; do not relabel these checks as that collection.
This completes worker/collector functional scope only. Performance repetitions,
application,100-task quality and sustained/profile evidence remain open. Both
unchanged Q4 loss rejections still block the explicit M5 exit.

The next larger FP32 task-reference prerequisite can use frozen raw IDs at fixed
capacity562/chunk8 and64MiB separate application allowance: planned admission
6,343,713,560 bytes leaves98,737,384 bytes below6GiB. The tokenizer's113,554,896-byte
allowance cannot coexist within that plan. Construct/finalize tokenizer output only
after the owned model worker exits; native admission and strict response/state
qualification remain necessary. This is a task-quality reference design, not a
qualified application-latency path or an implementation claim.
