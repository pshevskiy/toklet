# Gemma E2B quality-input contract

This task freezes evaluation inputs before inference or grading. It does not
qualify Gemma quality, app eligibility, performance or the model's full context
window. The [active Gemma plan](gemma4-implementation-plan.md) keeps G2.1/G5 and
the [model-driven window requirement](long-reply-engine-plan.md) open.

## Candidate and comparison identities

The candidate is `mlx-community/gemma-4-e2b-it-4bit` at
`238767527555cb75a05732a84dff5d6ba0dd6809`, with original container SHA-256
`038e39a37a7667373d2c3991375446b10c96ae1d717a68674870343db376b76e`.
Its pinned conversion README identifies `google/gemma-4-E2B-it` source revision
`70af34e20bd4b7a91f0de6b22675850c43922a03`. Use that conversion lineage for
the unquantized baseline, rather than substituting a later upstream revision.

The [official pinned inventory](https://huggingface.co/api/models/google/gemma-4-E2B-it/revision/70af34e20bd4b7a91f0de6b22675850c43922a03?blobs=true)
advertises a 10246621918-byte `model.safetensors` with LFS SHA-256
`2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550`.
At contract freeze this was authenticated metadata, with no downloaded, verified
or executed weight payload. The subsequent
[complete acquisition](gemma-unquantized-checkpoint.md) now verifies that exact
file. Frozen preparation flags and workload bytes remain unchanged. A bounded
reference route now has [complete short graph preparation](gemma-bf16-text-reference.md)
with a separate diagnostic memory policy. Full independent graph/upstream BF16
agreement and complete 1025-position capacity still require qualification;
file size alone is not a runtime memory estimate.

The official tokenizer payload matches the candidate's pinned tokenizer JSON.
Git blob identities also match the candidate's template and generation config;
the entire official and candidate `text_config` dictionaries are equal.
Tokenizer-config wrappers differ, so both comparisons use the same direct
tokenizer JSON, frozen token IDs and rendering policy instead of wrapper
defaults. Generation defaults do not become hard output limits.

Two comparisons remain distinct: quality against the unquantized checkpoint,
and engine precision parity against the same four-bit source. Four-bit source
agreement cannot replace the required unquantized quality baseline.

## Frozen workload and capacity

The [benchmark policy](../benchmarks/README.md) and original pinned dataset
identities remain unchanged. The Gemma tokenizer supplies its own first 10001
WikiText-2 raw test IDs, scoring all 10000 next-token targets. Ten independent
blocks contain up to 1024 targets and 1025 input tokens, with only the preceding
token shared between blocks. Reset positions at each block, and aggregate by
target count rather than equally weighting unequal block means.

The 100 equally weighted cases are the first 40 HumanEval, first 30 TruthfulQA
and first 30 English LAMBADA rows. Code/prose use the pinned basic nonthinking
template with exactly one BOS. LAMBADA and loss are raw without added BOS.
Greedy selection breaks ties by lowest token ID. Output caps remain 256/32/16
for code/prose/continuation. Hidden stops consume a sampled output slot.
Scorers preserve append-only code, exact trimmed uppercase answer letters and
the first exact case/punctuation-sensitive continuation word. No dataset or
generated code is executed during preparation.

Mean NLL increase must be at most 0.05 nats/token; task pass-rate loss must be
at most 2 percentage points against the unquantized baseline. All selected
sources/cases remain excluded from calibration and tuning. No gate is changed
to accommodate the current engine.

The frozen contract records each full prompt, requested output and required
shared capacity. Its 1025-position loss requirement exceeds the current
32-position Gemma diagnostic. Neither prompts, caps nor blocks may be shortened
to claim this quality suite passes. Capacity expansion is a prerequisite to
scoring the full candidate. Passing this suite will still not qualify the
model's declared 131072-position shared input/output window.

## Implementation and verification

[Preparation](../tools/bench/prepare_gemma_quality.py) authenticates the four
original datasets, complete tokenizer/template/config identities and baseline
lineage. The [independent checker](../tools/bench/check_gemma_quality.py)
reconstructs source selectors, expected answers, manual template rendering and
all token IDs. It uses Parquet batches independently of the producer's full
column read. Both cold processes admit exact runtime bytes before importing
NumPy/PyArrow/tokenizers and check origins and loaded native images afterward.

The [Rust codec driver](../examples/gemma4_quality_contract.rs) loads no model
weights. It checks all 100 complete rendered prompts and the 10001 loss IDs with
Toklet's actual Gemma codec. The corpus verification prefix is only a bounded
tokenizer input: its first 10001 IDs must equal IDs from the complete original
corpus. It does not replace the loss workload with a shorter one. The Rust
driver relies on the separately authenticated manifest/source replay for
dataset and checkpoint lineage.

Recipe controls, coherent source/content corruption controls, structural
controls and Rust negative cases exercise the boundaries. Coherent corruptions
update dependent hashes; rejection therefore cannot rely solely on stale
hashes. Quality scores and native numerical execution remain absent.

Raw task text, expected answers, corpus prefix, runtime inventories and logs
stay in ignored artifacts. Git records the manifest, small loss-ID array and
preparation/checker sources. Preserve source attribution and licenses when
redistributing derived text.

## Actual capture, 26 October 2026

The [frozen manifest](../benchmarks/gemma-quality-v1/manifest.json) is 28428 bytes,
SHA-256 `c4e6fc5cce765a30c43c9ed2d3ee4bfda58a4b23bbfc3284e2160431e6fb6506`.
The 40004-byte loss array has SHA-256
`877cbe18574b47e4b6a8d1b58fb79621c37e0531bd1f9504e50ce63d186f8c46`.
Full task payload is 188152 bytes; the manifest records its digest. The 65536-byte
verification prefix encodes to 16025 IDs, with its first 10001 matching full
corpus tokenization. Both independent Python replay and Rust replay pass.

The largest task request is HumanEval/32: 311 rendered prompt tokens plus 256
requested output tokens, requiring 567 positions. Category maxima are 567 code,
167 prose and 132 continuation. The full loss/task suite therefore requires
1025 positions. These are complete workload demands, not smaller product caps.

Seven recipe controls, eight structural controls, six coherent source replay
controls, five Rust example unit tests and two coherent Rust negative cases
pass. The complete CPU suite passes 356 tests with 16 explicitly ignored;
formatting and strict default/Metal all-target Clippy pass. No new GPU or
iPhone execution is claimed. All 17 development/collection command groups close.

On Apple M1 Pro/16 GiB/macOS 26.6.2 (25G83), completed-process observations are:

| Process | Peak physical footprint, bytes | Maximum RSS, bytes |
| --- | ---: | ---: |
| Cold preparation | 419383648 | 634142720 |
| Independent source replay | 407931160 | 584794112 |
| Six source corruption controls | 399723040 | 976879616 |
| Rust codec replay | 487506688 | 478789632 |

Python phases each fit their 1 GiB diagnostic policy. Rust fits its separate
1 GiB codec plus 32 MiB data policies. These are input-preparation/codec memory
observations, not model fit, latency benchmarks or whole-app memory results.
The [public evidence](../benchmarks/results/2026-10-26-gemma-quality-contract-v1.json)
retains exact source/runtime/input/output identities, commands and controls.
Protected release binaries remain unchanged.

The first collection attempt failed an existing CLI test because the copied
integration-test executable embedded an absolute path to the earlier target's
CLI. Its original transcript and failed receipt remain retained. The successful
retry rebuilt CPU tests in a fresh target; no test, engine behavior or gate was
changed to pass. The reviewed release codec driver was reused only after exact
source/build continuity checks.

## Next reference and context work

Astra's geometry audit identifies the main unquantized-reference opportunity:
the per-layer embedding table has expected BF16 shape `[262144,8960]`, or
4697620480 bytes, but one selected row is only 17920 bytes. Expected selected
text weights are 9257138758 bytes; excluding that table leaves 4559518278.
These were deductions at contract freeze. Subsequent
[official header/layout inspection](gemma-unquantized-layout.md) confirms those
exact tensor extents, including60 unused shared-consumer K/V parameters beyond
the selected inventory. The subsequent complete acquisition verifies full
payload identity. Positive row/layout binding, arithmetic and model fit remain
unproved; header inspection does not qualify reference execution.

Use the authenticated official payload/header to qualify positive row/layout
binding and the bounded reference. Prefer explicit bounded row reads/conversion ownership over assuming
mapped pages stay within budget. The tied `[262144,1536]` embedding/head still
needs all 805306368 BF16 bytes for a full vocabulary projection. Independently
validate layout, same-consumption identity, tied weights, arithmetic precision,
short graph/source agreement and whole-process fit. BF16 weights read into FP32
arithmetic do not automatically reproduce the upstream BF16 runtime.

For numerical context expansion, separate total positions, allocated cache
capacity and maximum chunk rows. Current scratch sizes full-logit and quadratic
attention arrays by total capacity; changing only the 32-position constant would
require about 2.15 GB of logits at 1025 positions, and is unsuitable for the full
model window. Keep bounded activation/projection scratch, then verify the whole
32 boundary, local 511/512/513 transitions and 1024/1025 boundaries, including
shared KV, reset, cancellation and failed-step preservation. The complete frozen
quality suite precedes broader model-window/app qualification; neither step
waives the existing BF16/GPU/long-rotary discrepancies.
