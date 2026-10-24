# MLX compatibility CLI and native consumers

This H1 follow-up exposes the [shared local report](mlx-compatibility.md) through
an offline command and an additive C ABI. No numerical kernel, graph policy,
qualification label or installed download profile changes.

## Command

```sh
toklet inspect --model /path/to/model --backend metal \
  --context-capacity 512 --memory-limit-bytes 1610612736
```

The command writes one schema-versioned JSON report. A completed report exits 0,
including unknown architectures, unsupported artifacts and exceeded budgets.
Malformed command arguments exit 2 with stderr diagnostics and no report. Output
errors exit 1, except a closed pipe exits 0. `toklet inspect --help` documents the
options. The default backend is Metal; the default 6 GiB budget is the engine
ceiling, never an inferred phone allowance. `scalar-reference` selects the
existing scalar expansion estimator. This command creates no tokenizer, model
expansion, GPU, session or cancellation/signal machinery.

## Native owner

`toklet_mlx_inspect_create` receives a bounded UTF-8 path and its own versioned
`TokletCompatibilityOptions`: backend, context and exact numerical load budget.
These options deliberately avoid the model/session bridge's 20 KiB deduction.
Unknown or unsupported inspection results return an ordinary report handle;
invalid boundary arguments fail and clear a valid output handle.

One bounded writer serializes JSON once, with a 64 KiB maximum. The owner retains
immutable bytes and its creating thread. `toklet_mlx_inspect_json` supports a
NULL+zero byte-count query, exact copying, and short-buffer failure without
partial output. Repeated reads stay identical after source files change. Wrong
thread calls reject, and release must happen once on the creating thread.
JSON is UTF-8 without a NUL terminator. Other errors clear a valid byte count.

`TOKLET_COMPATIBILITY_JSON_ALLOWANCE_BYTES` reserves two maximum buffers plus
256 bytes for the handle: 131,328 logical bytes, separately from the reported
numerical budget. A compile-time assertion bounds the handle record. This is
logical ownership, excluding allocator overhead and temporary bounded metadata
parsing. There is no process-peak or session-memory claim.

The report is historical metadata. It is never permission to load later without
revalidating files, budgets, payloads, runtime device capabilities and session
requirements. App/download consumers still need those later gates.

## Verification and remaining scope

Formatting, strict default/Metal Clippy, 178 CPU and 198 Metal-feature tests pass;
ordinary suites leave 14 and 101 tests ignored respectively. New portable checks
cover all four families/backends, exact budgets, unknown/rejected requests,
version/backend/path/null failures, immutable capture, bounded writing,
short/exact/query outputs and wrong-thread ownership.

An actual C caller compiles, links and runs against the fresh Mac static library.
Its fixed-width options layout and all 23 public symbols match the header. The
existing explicit local text bridge test passes encode/decode/stop queries and
failure/thread boundaries after the output-copy helper moves to shared native
scope. The actual real Qwen3 CLI probe reports its 633,364,480-byte source payload,
pending runtime planning and unavailable Metal backend in the default build;
compiled-Metal fixture reports also pass in the ordinary feature suite.

The first C link attempt fails because the restricted environment cannot create
its default temporary file. The retry uses an explicit workspace temporary
directory and passes without a production code change. Failed captures remain
alongside the successful retry in the
[consumer receipt](../benchmarks/results/2026-10-24-mlx-compatibility-consumers-v1.json).

Retain the CLI/native consumers. No numerical behavior changed, so the existing
GPU execution results remain scoped to their earlier checked sources; this task
does not rerun or requalify GPU execution. Swift app consumption, actual Metal
resource planning, exact qualification receipts and H2–H4 general Hub/Chat flows
remain open. No new architecture, quality, performance or phone qualification
is claimed.
