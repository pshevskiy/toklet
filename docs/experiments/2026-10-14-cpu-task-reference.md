# Bounded larger FP32 raw-ID task reference

A dedicated CPU Tiled worker preserves the qualified FP32/ordered-FP64 numerical
path while admitting the frozen100-task inputs without a live tokenizer. Fixed
capacity562/chunk8, complete session allocations,64MiB numerical allowance and
separate64MiB worker/driver allowance total6,343,713,560 bytes. This is a raw-ID
quality-reference path, not a substitute for the capacity2048 CLI or application
latency. CPU/scalar and Metal defaults remain unchanged; no numerical kernels change.

## Memory and response ownership

Code prompt lengths88–306 plus output256 determine capacity562; prose maximum is
153+32=185 and continuation108+16=124. All100 original IDs/order/caps remain fixed.
Each case resets independently, uses unchanged EOS151643/151645, samples greedily
from complete FP32 logits and retains its final sampled token pending. The
collector rejects errors/cancellation, visible EOS, invalid token/index/time order,
cap/state/history SHA differences, missing resets and changed inventories.

The existing tokenizer allowance113,554,896 bytes would raise model-phase admission
to6,457,268,456 bytes, beyond6GiB. The new worker never instantiates it. The driver
retains bounded raw IDs/reports and constructs the pinned tokenizer only after
successful process exit and reaping. Final text is batch-decoded then hashed, with
no claim of streamed text timing. Separate decoder-phase admission180,663,760 bytes
includes the tokenizer and64MiB application allowance, with model residency0.
Lifetime peaks from different phases are not summed as simultaneous allocations.

Both readiness and response state publish actual modes, geometry and complete
memory. Bounded pipes,64KiB lines,256 worker requests, output/event/time watchdogs
and owned cleanup keep infrastructure failures separate from model-answer failures.
Exact frozen payload/contract/source/config/tokenizer/EOS identities are bound before
launch and at endpoints. Full collection is separate from `--check`: the latter
uses first fixed code/prose/continuation cases at eight-output probes and cannot
claim100-case completion or task-quality results. Literal graders are not run here.

## Completed qualification

A source-reviewed synthetic zero-weight CPU test independently forces immediate
hidden EOS,256-token cap exhaustion with306-token prompt, commitment of all562
positions, invalid-request prefix preservation, reset and next-token rejection.
No model download or Apple GPU is needed. Five offline Python tests reject wrong
representations/budgets, resident tokenizers, cancellation/error/state/mode/reset
mutations and probe/full-cap confusion; tokenizer construction is gated on exit,
and malformed readiness preserves the primary error and reaps the owned worker.

One actual larger FP32 worker then passes three fixed short protocol probes:
HumanEval/0 emits8 and reaches its probe cap; truthfulqa/row-0 emits1 then samples
hidden EOS151645; lambada/row-0 emits8 and reaches its probe cap. Initial overflow
and three invalid-token requests preserve complete state hashes. All28 raw events
replay; exit0, cleanup/endpoint errors empty, identities unchanged. Tokenizer
construction follows successful exit/reaping, and all finalized strings match
independent pinned batch decoding. These outputs are excluded from100-task scoring.

Actual admission6,343,713,560 and sampled model-phase worker-plus-driver lifetime
native charge6,237,181,496 bytes fit6GiB. Decoder driver lifetime native peak is
74,777,176 bytes in its separate phase. Native counters do not establish total
shared/clean mapped physical residency. No latency, energy or quality score is
inferred from this protocol check.

## Checks, provenance and reproduction

Source parent`1a332a5`. Formatting, strict Clippy in both configurations,
96 CPU/108 Metal-feature all-target Rust tests and201 Python checks pass. These
include example tests previously outside ordinary test counts. Fresh default-feature
release build uses`CARGO_TARGET_DIR=artifacts/m5-checks`; protected CPU binaries are
unchanged. Source and compiled snapshots:
`artifacts/m5-cpu-task-reference-source-v1/` and
`artifacts/m5-cpu-task-reference-native-binary-v1/`. Astra reviews source before
native and independently replays completed evidence before commit.

[Check summary](../../benchmarks/results/2026-10-14-larger-cpu-task-reference-check-v1.json),
[raw events](../../benchmarks/results/2026-10-14-larger-cpu-task-reference-check-v1.raw.jsonl)
and [receipt](../../benchmarks/results/2026-10-14-cpu-task-reference-qualification-v1.json)
bind source/compiled bytes, frozen inputs, checks, phase ownership and observations.

```sh
.venv/bin/python tools/experiments/cpu_task_reference.py --binary artifacts/m5-cpu-task-reference-native-binary-v1/cpu_task_reference --check --output benchmarks/results/cpu-task-reference-check-rerun.json
```

Use fresh output paths. Omit`--check` for all100 exact cases, then apply the unchanged
literal graders in their separately qualified isolation window after all model
workers exit. Full reference responses, matched Half task results, application,
performance repetitions and sustained/profile evidence remain. Both fixed Q4 loss
rejections still block the explicit M5 exit.
