# Backend-aware Half task-response collection

The frozen task collector now supports the explicitly qualified Half Metal CLI
without changing the protected CPU collectors or literal executor. Metadata remains
truthful: backend Metal, CPU projection `none`, HalfCompensated projection, Original
RMS, FullLogits, FP16 weights/KV, capacity2048/chunk128 and complete admitted memory.
No CPU/scalar label is substituted to bypass admission.

## Frozen response and artifact policy

The original100 IDs/order, prepared prompt IDs, code/prose/continuation caps256/32/16,
lowest-ID greedy ties, hidden EOS151643/151645, empty state per case and explicit
reset/finalization remain fixed. The Metal-specific parser preserves the existing
response grammar and checks actual CLI statistics/admission. It rejects missing or
reordered tokens/resets, visible EOS, wrong modes/memory, repeated text tails,
cancellation/errors, nonfinite times, changed caps and invalid committed/pending
counts. Existing task admission/scoring functions and CPU response code are unchanged.

`--check` uses first fixed indices0/40/70 at reduced eight-output probes, with an
explicit effective cap in the same strict parser. It cannot claim100-case completion,
quality completion or scores. Full mode retains each original cap. Before launch,
the exact bounded qualification receipt bytes are hashed before parsing and rebound
at initial inventory. Its immutable CLI binary and compiled snapshots are checked;
current numerical/CLI source must agree with that qualified binary. All actual
source/config/tokenizer/EOS/payload/contract/helper bytes are pinned/rebound.

Owned bounded nonblocking transport records native events and idle worker-plus-driver
charges. Reset finalizes each response before the next case. After successful
exit/reaping, independent pinned batch decoding must match every assembled native
string and text hash. No new numerical code or default is introduced.

## Completed code-task checks

Seven offline tests cover actual Metal metadata, original/full versus probe caps,
EOS/reset/timing/state/error mutations, deferred independent text comparison,
unqualified executables, qualification replacement after parse preventing launch,
and owned cleanup preserving primary errors. All208 Python checks and whitespace
checks pass. No unrelated engine rebuild/test is needed for these Python-only
changes; the immutable Astra-qualified release CLI and current unchanged numerical
source provide the executable prerequisite.

Six actual short probes pass on Apple M1 Pro16GiB/macOS26.6.2. Small emits8/8/8;
larger emits8/1/8, with its prose result ending in hidden EOS151645. All31 small and
24 larger raw events replay. Both workers exit0, finalize/reset all three cases,
clean up without errors and preserve endpoint identities. Probe responses are
excluded from100-task scoring.

| Fixed checkpoint | Complete logical admission | Sampled model-phase worker-plus-driver native lifetime charge |
| --- | ---: | ---: |
|0.5B |1,443,266,720 bytes|1,270,007,752 bytes|
|1.5B |3,577,266,480 bytes|3,406,483,080 bytes|

Both admissions/charges fit6GiB. Counters do not establish total shared physical
allocation, and no comparative latency, energy or task pass-rate claim is made.

## Provenance and next work

Source parent`59716ad`; collector/helper snapshot`artifacts/m5-metal-tasks-source-v1/`.
Executable remains`artifacts/m5-cli-half-native-binary-v1/toklet`, bound to its earlier
[CLI qualification](2026-10-14-metal-cli-half.md). Protected CPU binaries are unchanged.
Astra reviews source before native and complete saved evidence before commit.

[Small check](../../benchmarks/results/2026-10-14-small-half-tasks-check-v1.json),
[small raw](../../benchmarks/results/2026-10-14-small-half-tasks-check-v1.raw.jsonl),
[larger check](../../benchmarks/results/2026-10-14-larger-half-tasks-check-v1.json),
[larger raw](../../benchmarks/results/2026-10-14-larger-half-tasks-check-v1.raw.jsonl)
and [receipt](../../benchmarks/results/2026-10-14-metal-tasks-qualification-v1.json)
retain actual metadata, exact inputs, native state/text, compiled bytes and checks.

```sh
.venv/bin/python tools/experiments/metal_tasks.py --model small --check --output benchmarks/results/half-small-tasks-check-rerun.json
.venv/bin/python tools/experiments/metal_tasks.py --model larger --check --output benchmarks/results/half-larger-tasks-check-rerun.json
```

Use fresh outputs and run sequentially with native Metal access. Full collection
omits`--check`; it still requires separate grading admission and the existing
qualified literal executor after every model worker exits. Matched larger FP32/full
Half100-case responses and scores, application, repeated performance and sustained
release evidence remain. Both fixed Q4 loss rejections continue to block M5 exit.
