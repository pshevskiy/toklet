# MiniCPM5 complete quality-input contract

The original held-out workload is now prepared for MiniCPM5: **10,000 loss
targets and all 100 tasks**, with exact tokenization, source identities and
unchanged scoring policies. A separate cold source replay reproduces every
payload byte. Checks and Astra review pass; model scoring and adoption remain
open.

## Workload and independent source identities

The [prospective contract](../artifacts/minicpm5-quality-inputs-2026-11-10/prospective-contract-1.json)
was frozen before any loss/task/native quality outcomes. It retains the
original WikiText test corpus and first 40 HumanEval, 30 TruthfulQA and 30
LAMBADA cases from the pinned [source manifest](../benchmarks/sources.json).
The model-independent task recipe and original 256/32/16 output caps are
unchanged. All HumanEval prefixes/tests remain text during preparation.

Complete corpus tokenization produces 288,006 IDs; the first 10,001 define
exactly 10,000 targets in ten contiguous blocks with one shared boundary token.
The 65,536-byte UTF-8 prefix verifies tokenization; it does not replace the loss
corpus. Original NLL/task-loss thresholds remain 0.05 nats/token and two
percentage points. Strict grading, greedy lowest-ID ties, task resets, hidden
stop accounting and stop union `[1,130073]` remain explicit.

Two comparisons must eventually pass separately:

1. Original stock BF16 checkpoint versus modified BF16 parent.
2. That exact modified parent versus selected native MLX8/Half execution.

The stock metadata is pinned to `openbmb/MiniCPM5-2B` at
`42433e90dad95123ab87729fd7011e7110629556`. Its declared single weight object is
5,033,557,096 bytes, SHA-256
`14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c`.
Eight public metadata files (10.15 MB total) pass HTTPS and API/LFS/Git-blob
binding. **No stock weight payload was downloaded or authenticated.** The
modified parent/selected child retain the distinct pinned identities and
bounded reconstruction evidence from the [matrix audit](minicpm5-matrix-lineage.md).
[Stock metadata receipts](../artifacts/minicpm5-quality-inputs-2026-11-10/stock-source-receipts-1.json).

Stock and child tokenizer core JSON structures are identical. Stock padding is
null; the child retains BatchLongest-left padding. Single-case encoding produces
identical IDs without clearing either policy. Their wrapper files and template
tool branches differ. Under explicit no-tools, thinking-disabled, generation-
prompt settings, both original templates render all 100 task strings exactly
and both tokenizers produce identical task IDs. Full corpus IDs also match.
These observations qualify only the complete basic-text input recipe.

## Capacity and bounded parser observations

Loss blocks require 1,025 positions. HumanEval/32 has a 303-token prompt plus
256 output slots, requiring 559; it is the sole task above 512. The complete
suite therefore requires 1,025. Current engine capacity 512/chunk 32 is
unchanged. Wider numerical/context qualification must precede complete native
quality scoring; truncating this workload would violate the contract.

Preparation and replay use external hashes for all 3,643 Python/parser/native
runtime files before Tokenizers/Jinja/NumPy/PyArrow work. The source parser has a
separate 1 GiB process allowance; this does not raise the 512 MiB Rust codec,
6 GiB engine or 2 GiB free-disk policies. No GPU model is allocated. Owned child
RSS and sampled libproc physical counters describe these parser processes;
they exclude their parent controllers and do not establish inference fit.

| Phase | Final child peak RSS | Largest sampled child physical counter | Result |
| --- | ---: | ---: | --- |
| Prepare | 546,193,408 bytes | 279,069,992 bytes (lifetime) | Complete inputs, owner closed |
| Cold replay | 546,324,480 bytes | 284,247,312 bytes (lifetime) | Every payload byte matches, owner closed |
| Altered-input controls | 556,761,088 bytes | 295,847,256 bytes (lifetime) | Controls pass, owner closed |

The controls admit one extra worker source file (3,644 total). Final physical
peaks are unavailable. Sampling has no errors and does not establish continuous
hard-cap coverage. These are source-process observations, without warmed
inference cohorts, speed, overhead or memory-saving claims.
[Prepare owner](../artifacts/minicpm5-quality-inputs-2026-11-10/prepare-checks-1.json),
[replay owner](../artifacts/minicpm5-quality-inputs-2026-11-10/replay-checks-1.json),
[controls owner](../artifacts/minicpm5-quality-inputs-2026-11-10/controls-checks-1.json).
The exact guarded commands are retained in the
[preparer controller](../artifacts/minicpm5-quality-inputs-2026-11-10/run-prepare-1.py),
[replay controller](../artifacts/minicpm5-quality-inputs-2026-11-10/run-replay-1.py) and
[controls controller](../artifacts/minicpm5-quality-inputs-2026-11-10/run-controls-1.py).

## Reproduction and checks

The [versioned manifest/token payload](../benchmarks/minicpm5-quality-v1/README.md)
binds full local task/prefix files, all source data, comparison order, exact case
hashes, scoring, input-equivalence observations and capacity requirements. The
producer rejects changed source assets and partial workloads. The consumer
requires an external manifest digest, validates typed counts, blocks, caps,
hashes, token IDs and qualification flags, then reconstructs all sources and
compares every payload byte. The replay shares the frozen model-independent
recipe; it does not constitute independent numerical or native-codec parity.

Seven portable tests use independent small structural fixtures and require no
parser extensions, network, model or GPU. One positive and 28 actual-input
mutations pass: 20 structural rejections and eight complete-source replay
rejections after coherent payload/self-hash repair. Those eight cover changed
loss IDs, original tests/code prefixes/answers, rendered strings, prompt IDs
and verification text. The controls use one cold admitted source reconstruction
and the CLI's same validator/byte-equality predicate, not one new CLI per case.
Original capture bytes remain unchanged.
[Full cold replay](../artifacts/minicpm5-quality-inputs-2026-11-10/input-replay-1.json),
[altered inputs](../artifacts/minicpm5-quality-inputs-2026-11-10/input-controls-1.json).

Rust, kernels, model/session/ABI code and existing fixtures are unchanged.
All 4,031 required Cargo/fixture inputs, 421 native sources and 10 native67
package outputs match the previously reviewed checks: 679 CPU/821 Metal tests,
both strict Clippy configurations, three native slices and C smokes. Formatting
passes freshly; new Python checks and source replay are fresh.
[Input reuse proof](../artifacts/minicpm5-quality-inputs-2026-11-10/required-native-reuse-1.json),
[portable tests](../artifacts/minicpm5-quality-inputs-2026-11-10/portable-tests-1.json).
Astra approves scoped retention and commit with no actionable findings. Its
fresh guarded cold replay independently reproduces all four payload hashes;
review-parser final RSS is 537,264,128 bytes, below the separate 1 GiB allowance.
Administrative documentation closure follows with reviewed code/data/contracts
and original observations frozen.
[Review](../artifacts/minicpm5-quality-inputs-2026-11-10/astra-review-result-1.json),
[commit/integration](../artifacts/minicpm5-quality-inputs-2026-11-10/publication-checks.json).
No independent model numerical,
task-quality, default allocator, managed adoption, full-window, performance,
summary-overhead, UI or physical-iPhone gate closes from input preparation.
The full improvement goal continues with the original deeper/quality gates.
