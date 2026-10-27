# Qwen3 native 4096-position diagnostic

This experiment applies the approved [native arithmetic profile](qwen3-native-profile.md)
to the next boundary in the [full-window plan](qwen3-full-window-plan.md). It uses
the pinned eight-bit Qwen3-0.6B model on Apple M1 Pro, MlxAffine projections,
Compensated normalization, Reference attention, FP16 activations/KV and FP32
logits. The ordinary app remains at its qualified partial 512-position window.
Historical trajectory-fidelity rejections and weak absolute instruction quality
remain separate facts. A diagnostic pass does not qualify the declared 40960
positions, another device or performance.

## Frozen protocol

The [inputs](../benchmarks/qwen3-context4096-v1/README.md) were prepared and
independently reconstructed before the new model runs. A repeated structural
fixture supplies exact schedule/state controls; a distinct last-token
perturbation supplies the causal control. Quality uses a new, unrepeated,
prospectively frozen WikiText span and exactly 1024 targets at depth 3072–4095.
The mean native NLL increase over the original BF16 model must remain at most
0.05. No trajectory envelope, near-tie waiver or altered quality threshold is
introduced.

The finite native recipe commits 37376 tokens in nine phases: current 512-token
archive anchor; 4096-token chunk32 run plus prescribed reset replay; chunk128
schedule and future-perturbation runs; three ordinary 4064-input/32-output
generation runs; one sampled-history replay; and one quality run. It checks all
finite logical row hashes, exact schedules, cached prompts, committed/pending
state, discard/reappend, final flush and atomic rejected 4097-token requests.
Generation compares visible IDs between chunk32/128 and FullLogits/GreedyToken.
The ID-only path compares the final selection; full-logit paths also compare the
final distribution hash. The raw diagnostic uses explicit empty stop IDs.

Independent analytical controls cover exact uniform attention at 1024/2048/4096
keys and three dominant-key cases with actual 16:8 GQA and a masked future NaN
value. The unchanged high-position rotary control covers positions 4093–4095
under its original primitive envelope. Neither control substitutes for model
quality.

The original unquantized BF16 reference retains its modules, parameters, dtype,
RoPE and full history. Ordinary cache instances preallocate 4096 positions;
evaluation barriers bound lazy update graphs. Before quality, three sequential
same-chunk executions require exact stock/adapted outputs for all 1025 rows,
all-layer valid K/V at 255/256/257/1024/1025, and six masked-head rows. Full
eight-row projection precedes selecting scored rows. All 18 retained BF16/FP32
row pairs are independently checked by integer bit widening.

All numerical processes have a 6 GiB measured lifetime-memory gate. Conservative
source admission reserves 5927671248 bytes during loading and 5299813096 during
inference. These reserves do not establish full-window fit. Owned processes,
immutable file/runtime identities and raw failure evidence surround each phase.

## Retained integration failure

The first attempt stopped in stock-cache metadata validation at the call from
505 to 513 positions, before native 4096 or new quality execution. The validator
incorrectly assumed allocated capacity was rounded from the current offset.
Pinned MLX instead trims to the previous valid prefix before growth for uneven
chunks, producing capacities 761, 1017 and 1273 in this schedule.

Astra investigated and independently confirmed the cause. The correction tracks
the exact allocation recurrence from prior expected capacity and frozen chunk
boundaries; it still requires exact shapes, offsets, steps and dtypes for all
28 layers. Literal regression cases reject the old incorrect capacity. No model
arithmetic, schedule, input, comparison or acceptance threshold changed. The
original producer/test bytes and complete failed attempt remain retained.

The continuation reuses authenticated passing builds, operator controls and
inputs; it repeats the corrected source tests and complete source equivalence
before proceeding to the previously unexecuted phases. Completed numerical work
is not silently discarded or relabeled as a pass.

## Result

The finite diagnostic passes on Apple M1 Pro / 16 GiB / macOS26.6.2(25G83), using
an isolated release build with Homebrew Rust1.90.0. The
[result receipt](../benchmarks/results/2026-10-28-qwen3-context4096-v1.json)
retains phase identities, raw evidence hashes, timings and resource observations.
Astra independently reviewed the code and completed evidence and accepts this
scoped diagnostic before the task commit.

All 37376 prescribed native commits complete. The current 512-row anchor matches
its archive; all 4096 chunk32/chunk128 row hashes agree, and changing the last
input leaves the first 4095 rows unchanged. Cached-prompt, reset/prefix,
pending/discard/reappend, sampled replay and final flush checks pass. All 14
prescribed append/set-token overflows return the exact expected error and leave
state unchanged. All three generation paths return the same 32 visible IDs;
FullLogits paths also agree on the final distribution, and GreedyToken agrees on
the final selection. This structural continuation repeats the fixture and is not
an instruction-following result.

The new 1024-target comparison records:

| Profile | Mean NLL |
| --- | ---: |
| Native eight-bit weights / Half state | 1.9171930208055907 |
| Original unquantized BF16 source | 1.9167905279642214 |
| Increase | 0.0004024928413692841 |

The increase passes the original inclusive 0.05-nat gate. The complete historical
10000-target/100-task result remains separately qualified at its original scope;
this supplement does not add task-score evidence. Absolute instruction quality
remains weak, and the historical trajectory-fidelity contracts remain rejected.

Formatting, strict default/Metal Clippy, 372 default tests, 247 Metal-feature
library tests, five example tests, three input controls, 13 corrected source
controls and the two explicit deep GPU controls pass. Counts overlap. Source
equivalence passes all 1025 rows, every layer's five cache endpoints and six
masked-head rows. The reader independently verifies all 18 raw widening pairs,
1024 target identities and aggregate mean/delta arithmetic from retained
per-target losses. Full quality logits are not retained for independent per-row
rescoring. All 308 final bindings and both
model inventories authenticate. All 27 test-owned groups close, including the
retained failed attempt; the separate publication reader also closes cleanly.

| Phase | Peak physical bytes | Maximum RSS bytes |
| --- | ---: | ---: |
| Largest native process: B/chunk128 | 1426802176 | 1435074560 |
| BF16 equivalence | 1834469080 | 1417461760 |
| BF16 quality | 1841317520 | 1243398144 |
| Independent readback | 20529656 | 32358400 |

Every numerical process stays below 6 GiB, and the readback stays below 512 MiB.
These are individual process lifetime observations, not a combined working set
or a full40960 fit proof.

### Ordinary completed-call planning observations

Diagnostics and per-encoder profiling are disabled for these three runs. Loading,
pipeline setup and session creation are recorded separately in the receipt.

| Chunk / readback | 4064-input prefill calls, seconds | 31 decode calls, seconds | Final flush, seconds |
| --- | ---: | ---: | ---: |
| 32 / FullLogits | 456.640868 | 206.097792 | 6.569717 |
| 128 / FullLogits | 166.720088 | 206.353619 | 6.454386 |
| 32 / GreedyToken | 455.824168 | 206.553624 | 6.686795 |

These are sums of completed engine-call wall times, including encoding, waiting
and engine readback. They exclude sampling, journal work, tokenization and UI
delivery. There are no benchmark warmups/repetitions here, and power/thermal
conditions were not independently recorded. No speedup, throughput, tail-latency
or performance qualification follows. The serial Reference work is sufficiently
costly to justify the next bounded scheduling experiment before8192; it does not
justify changing arithmetic or its gates.

The improvement is checked native4096 diagnostic behavior, held-out relative
quality and whole-process fit. The app remains at512. Full40960, usable Mac
streaming/admission, final full-window quality and iPhone qualification remain
open under the full-window plan.
