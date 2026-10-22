# Toklet: implementation, measured performance and remaining limits

21 October 2026. Repository state: `9749861`, following the completed M0–M7 audit.
All 32 plan tasks are complete within their accepted scopes. M7 uses the explicitly
amended **iPhone 14 Pro** target. Performance numbers below come from earlier
immutable, optimized-build collections, not a new benchmark of this documentation
revision. The linked reports retain executable/source hashes, raw samples and
precise acceptance policies.

## M8 native MLX update — 22 October 2026

The M0–M7 closure below predates the active [native MLX expansion](mlx-implementation-plan.md). Toklet now loads and executes selected MLX affine Qwen artifacts directly in Rust/Metal. The public eight-bit Qwen0.5B snapshot passes full 10,000-target loss and 100-task preservation gates on Mac and iPhone 14 Pro (mean loss 2.884954; tasks 2/100 versus FP32 2/100). The public four-bit candidate remains rejected. M8.2 local Qwen execution is complete in this selected scope. Broader architectures, model discovery/downloads and chat-template compatibility remain open.

The [selected MLX phone state trial](mlx8-phone-state.md) passes all 33 diagnostic rows, including actual 128/384 committed/pending prefill cancellation and the full 1,152-position boundary. The [actual MLX scene trial](mlx8-phone-lifecycle.md) also passes: the same app process survives Settings/return, retains ten sampled IDs and replays the exact 522-token prefix, with matching fresh-reference continuation and natural close. Its observed process peak is 0.6638 GB. Real UIKit warning delivery and jetsam remain unqualified.

The [new app performance report](mlx8-performance.md) records all 128/512/1024 primary/confirmation workloads and dedicated 600-native-second sustained runs. At 512/128, Mac primary/confirmation native TTFT medians are 3.1133/3.1134 seconds and decode 22.1772/22.2205 IDs/s; phone 13.3074/15.1906 seconds and 15.2318/13.6681 IDs/s. Maximum process peaks are 0.6392 GB Mac/0.6519 GB phone. Sustained phone decode falls 22.324% from first-five to last-five medians under changing thermals. These raw-ID timings do not establish an external speed lead, steady phone rate or text/UI latency. Earlier Half/Q4 lifecycle qualification below does not carry over automatically to MLX.

## What is implemented

| Area | Delivered behavior |
| --- | --- |
| Rust engine | One Cargo package, library, CLI and C-compatible static library. Readable scalar decoder with FP32 storage, qualified FP64 accumulation, independent numerical fixtures and deterministic greedy selection. |
| Model loading | Validated local Qwen safetensors/config loading, FP32/FP16/BF16 conversion, indexed shards, tied weights and bounded conversion memory. Qualified checkpoints are Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct. |
| Persistent sessions | Separate immutable weights, session-owned per-layer KV and reusable scratch; exact-prefix reuse, append, fully cached prompts, pending sampled tokens, reset, cancellation and explicit replay. Admission and overflow checks preserve valid state. |
| CLI | Local tokenizer and pinned chat-template/stop semantics; raw IDs, prompts/messages, streaming text, persistent JSON-line interaction and separate timing/state/memory statistics. |
| CPU optimization | Opt-in ordered FP64 projection tiling with preserved scalar reference behavior and measured workload/application improvement. |
| Metal execution | Resident weights, FP16 KV and bounded scratch; cached pipelines; full embedding/projection/normalization/rotary/GQA/MLP/output path, dependent command scheduling and explicit failure/ownership boundaries. |
| Quantization and prefill | Versioned Q4 packers/reconstruction oracles, fused dequantization/projection and separate tiled prefill. The selected larger model uses frozen group-8 Q4 v2 with Half exclusions. |
| Long context | Explicit Online attention avoids a full attention-score matrix. Selected Mac configurations qualify 4K/8K/16K/32K total-token cases, quality/state/overflow and maximum-context application generation. |
| Native integration | Checked C ownership/thread/callback/cancellation/error contracts, Apple device/simulator/Mac library packaging, and a Swift raw-token diagnostic harness with background inference and bounded events. |
| Phone lifecycle | Installed physical-device execution, cached/changed-prefix and pending-token checks, actual partial-prefill cancellation, reset/release, and same-process scene interruption followed by exact retained-prefix replay. |
| Verification | Frozen loss/task policies, benchmark contracts, source/artifact identity checks, raw-data replay, retained failures and an improvement/retain/reject record. |

Defaults remain CPU scalar, GPU Original projection/Reference attention,
FullLogits, chunk 128 and capacity 2,048. Optimized configurations require explicit
selection; there is no automatic fallback. CPU context remains capped at 2,048.
See the [plan audit](experiments/2026-10-21-full-plan-audit.md),
[native ABI](native-abi.md) and [improvement record](improvements.md).

## Selected numerical configurations and quality

The selected paths on both measured devices use **Online attention, FP16 KV,
FullLogits and chunk 128** for context/generation trials:

- Small: Qwen2.5-0.5B-Instruct, compensated Half projections, Original normalization.
- Larger: Qwen2.5-1.5B-Instruct, frozen group-8 Q4 v2 compensated tiled projections,
  Compensated normalization; tied embedding/output, norms and biases remain Half.

Q4 v2 has four-bit codes plus Half scale/bias per eight weights: approximately
**8 effective bits per packed weight before padding**, with additional Half
exclusions. It is not a universal four-bit model or an MLX/GGUF-compatible format.

| Model | Online mean NLL | Matched FP32 mean NLL | Increase, nats/token | Frozen task passes, selected / FP32 |
| --- | ---: | ---: | ---: | ---: |
| Small | 2.883572 | 2.883725 | −0.000153 | 2/100 / 2/100 |
| Larger | 2.514223 | 2.485958 | +0.028265 | 11/100 / 11/100 |

Each model passes the unchanged 10,000-target loss gate of at most +0.05
nats/token and 100-task gate of at most a two-percentage-point drop. Phone loss
means exactly match the accepted desktop Online means, and phone task output
bytes match the desktop results. Phone code outcomes replay the pinned historical
literal grading for identical bytes; they were not executed again on the phone.
Absolute quality is low, including **0/40 code-task passes per model**. These are
relative preservation gates, not broad capability claims. Full-loss/task trials
retain their own capacities and chunks, distinct from the context trials.
Sources: [M6 release](experiments/2026-10-19-m6-release.md),
[phone qualification](experiments/2026-10-21-m7-iphone14pro.md).

## Measured Mac performance

**Apple M1 Pro, MacBookPro18,3, 16 GiB RAM, macOS 26.6.2 (25G83),
Rust 1.90.0 optimized builds.** Retained collections record power/thermal endpoints;
these are named-hardware results, not estimates for other Macs.

### Confirmed short-workload Metal result

The primary workload is **512 prompt + 128 controlled/teacher-forced output IDs**,
capacity 2,048. The initial collection and its independent confirmation each
have three warmups and ten alternating measured Reference/Online pairs.
Numbers are medians in seconds; GPU work is completed before timing ends.
Loading, tokenization, sampling and visible-text publication are excluded.

| Model | Collection | Reference complete request | Online complete request | Reduction | Online numerical TTFT | Online decode phase |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Small | Initial | 32.338 | 7.677 | 76.26% | 2.388 | 5.290 |
| Small | Confirmation | 32.351 | 7.687 | 76.24% | 2.392 | 5.294 |
| Larger | Initial | 78.517 | 21.184 | 73.02% | 8.586 | 12.588 |
| Larger | Confirmation | 78.504 | 21.172 | 73.03% | 8.572 | 12.600 |

This compares attention implementations with the same selected weights. All
eight workload rows preserve their state contracts; seven non-primary rows have
smaller exploratory sample counts. Small-model prefill ranges overlap, so no
uniform every-request prefill gain follows. These are numerical request gains,
not a measured application speedup or external-runtime lead.
[Primary measurements, MAD, ranges and eight-row screen](experiments/2026-10-15-m6-short-performance.md).

### Other retained improvements

| Mechanism | Accepted measured result | Scope |
| --- | --- | --- |
| CPU projection tile | Small-model 128/128 TTFT: 40.221 → 8.993 s in the first pair; reductions repeat at 77.64% / 77.60%. Complete-request reductions repeat at 52.00% / 52.10%. | Same-executable scalar/tiled primary; CPU tile remains opt-in. |
| Larger Q4 v2 tiled prefill | 512-input/one-output TTFT: 13.056 → 9.963 s and 13.300 → 10.347 s; 23.69% / 22.20% reductions. | Same Q4 v2 artifact, vector versus tile; separate from the Online-attention experiment. |

These gains cannot be added together. Q4 v2 is not faster than Half in every
practical configuration: all eight historical larger-model Q4 v2 complete-request
medians were 2.69–10.14% slower than historical Half in separate collections.
That comparison also changes representation/normalization and does not isolate
the cost of quantization. Sources:
[CPU retention](experiments/2026-10-12-cpu-tiled-retention.md),
[M5 release](experiments/2026-10-15-m5-release.md).

### Long-context timing and memory

Online complete numerical request times below include prompt processing and 128
controlled output positions. Prompt length is total capacity minus 128. Ordinary
runs have one warmup and two measured pairs; confirmations have one measured
pair. These are descriptive observations, not tail-latency estimates or a new
performance-ranking qualification. The input is frozen cyclic stress text.

| Total tokens | Small ordinary / confirmation, s | Larger ordinary / confirmation, s |
| --- | ---: | ---: |
| 4,096 | 48.499 / 48.497 | 117.912 / 117.788 |
| 8,192 | 118.460 / 118.741 | 263.807 / 263.851 |
| 16,384 | 331.797 / 333.624 | 651.138 / 651.196 |
| 32,768 | 1,073.044 / 1,064.595 | 1,813.159 / 1,815.157 |

Maximum complete admission across context roles is **2.771 GB small / 4.577 GB
larger**; sampled joint native lifetime charges are **2.519 / 4.358 GB**.
Separate tokenizer-owning maximum-context applications have admission
**1.792 / 3.123 GB** and sampled native maxima **1.616 / 2.961 GB**. All fit the
6 GiB admitted budget. GB here is decimal; admission is logical accounting and
native charges are separately sampled process lifetime maxima, not total system
RAM. Ordinary 32K application generation also passes, but supplies no natural
long-document ability or application-latency claim.
[Full context evidence](experiments/2026-10-19-m6-context.md),
[application and memory closure](experiments/2026-10-19-m6-release.md).

## Measured iPhone 14 Pro performance

**iPhone 14 Pro (`iPhone15,2`), iOS 27.0.1, wired, Low Power Mode off.**
Context/state capacity is **1,152 total positions**, with 128 output IDs reserved
and chunk 128. The phone does not inherit the Mac's 32K qualification.

Each row below has two runs, each with three warmups and ten measured requests.
Paired values are **run 1 / run 2 medians**, not min/max sample ranges. Requests
reset and emit 128 raw IDs. Native TTFT includes prefix preparation/prefill and
first sampling; decode rate counts the 127 intervals between first and last
native token events. Load/session creation and transfers are excluded. Swift
callback timing is separately retained; visible-text latency is not measured.

| Model | Prompt tokens | Native TTFT, s | Decode, IDs/s | Completed request, s |
| --- | ---: | ---: | ---: | ---: |
| Small | 128 | 1.356 / 1.512 | 21.378 / 20.739 | 7.296 / 7.637 |
| Small | 512 | 10.846 / 15.058 | 15.638 / 12.756 | 18.962 / 25.017 |
| Small | 1,024 | 39.929 / 39.880 | 8.844 / 8.835 | 54.288 / 54.255 |
| Larger | 128 | 11.768 / 13.062 | 4.049 / 3.722 | 43.157 / 47.180 |
| Larger | 512 | 49.883 / 48.466 | 3.613 / 3.649 | 84.940 / 83.332 |
| Larger | 1,024 | 95.401 / 95.627 | 3.427 / 3.426 | 132.851 / 132.635 |

Observed load/session creation ranges are **3.263–3.360 s small** and
**37.926–42.723 s larger**. All 156 context requests preserve exact output/state
semantics. The sequence heats the device, reaches serious thermal state and
imposes no cooldown. These observations characterize the tested setup; they do
not establish controlled speedups or a direct Mac-versus-phone ranking.
[Repeated phone matrix and raw timing definitions](experiments/2026-10-21-m7-phone-context.md).

### Sustained behavior, memory and lifecycle

The dedicated small 512/128 sequence completes **24 measured requests over
620.566 native seconds**, excluding three warmups. First-five versus last-five
medians show the slowdown explicitly:

| Metric | First five | Last five |
| --- | ---: | ---: |
| Native TTFT | 10.876 s | 23.227 s |
| Decode | 15.511 IDs/s | 8.386 IDs/s |
| Completed caller request | 19.064 s | 38.371 s |

Observed thermal states are fair/serious; a controlled thermal cause is not
isolated. Larger-model 512-token runs contain **839.172 / 827.973 measured native
seconds**, and 1,024-token runs contain **1,344.731 / 1,329.965 seconds**. Duration
coverage does not imply constant throughput.
[Sustained observations](experiments/2026-10-21-m7-phone-small-sustained.md).

The largest joined uninstrumented whole-process lifetime peak is
**1,944,701,920 bytes (1.945 GB)**. Context-matrix peaks are **1.114 GB small /
1.945 GB larger**. Numerical admission caps are **1.5 GiB / 2 GiB**; the separate
whole-process trial cap is **2.5 GiB** for either model. These successful trial
budgets are not permanent iOS memory guarantees.

Both models pass full 1,152-position commitment and state-preserving overflow,
actual partial-prefill cancellation/resumption, and Settings/return scene
interruption with exact same-process retained-prefix replay. Synthetic warning
handling is tested; actual UIKit warning delivery and jetsam recovery are not.
[Joined device qualification](experiments/2026-10-21-m7-iphone14pro.md),
[M7 closure](experiments/2026-10-21-m7-release.md).

## Remaining limitations

- **Hardware scope:** qualification covers the named M1 Pro and iPhone 14 Pro/OS
  versions and selected artifacts. Base iPhone 14, Pro Max, iPad, other Macs and
  other OS versions need separate evidence. Mobile limits are independently
  bounded and do not extend to Mac 32K capacity.
- **Quality:** preservation gates pass, but absolute task scores remain low.
  Cyclic 32K stress/state coverage does not establish natural long-document
  reasoning. This is a small inference engine and diagnostic harness, not a
  broadly validated assistant product.
- **Performance:** no full quality/memory-qualified external FP16 comparison
  matrix, external speed lead, p99 guarantee or universal GPU-over-CPU claim.
  Mac numerical and phone native/callback/application timing boundaries differ.
  Source verification warms storage; loading figures do not establish cold-disk
  performance. Phone sustained slowdown remains material.
- **Memory/energy:** process observations are not whole-system RAM or universal
  allocation guarantees. Continuous pressure/clocks/thermal telemetry is
  incomplete. Two actual phone Power Profiler attempts failed without a usable
  trace; no attributable joules/watts/battery-life result is available.
- **Mobile lifecycle:** actual scene interruption/replay passes; real system
  memory-warning delivery, jetsam recovery and cleanup while OS-suspended remain
  unqualified. The app is a raw-token harness; tokenization/chat rendering,
  downloads and tool execution stay outside the kernels.
- **Rejected/experimental paths:** Q4 v1 and small Q4 v2 miss frozen quality gates.
  Non-beneficial fusion prototypes were removed. GPU greedy is explicit and was
  0.289% slower in its measured comparison. Chunk 128 stays after larger chunks
  failed the useful-gain policy. I8 KV remains a component experiment, with mixed
  benefit and no production engine integration; FP16 KV is retained. No ANE
  backend is adopted because actual acceleration was unobserved in the bounded
  public-runtime experiment.
- **Product scope:** stochastic sampling, speculative decoding, paged KV,
  multi-session serving, model discovery/downloads, GGUF import, broader model
  families and polished application UI are deferred or conditional work.
- **Verification scope:** the final audit passed 887 Python tests across 127
  modules, 111 CPU tests and 123 Metal-feature tests, formatting and strict
  all-target Clippy. Rust ignored counts are 8 CPU / 59 Metal-feature; hardware
  and real-model suites are explicit, separately collected evidence. Configured
  Linux CPU CI is not proof of a remote CI run. This summary adds no runtime
  change or new numerical/device/performance qualification.

The [full-plan audit](experiments/2026-10-21-full-plan-audit.md) distinguishes these
limits from incomplete required tasks. The [improvement record](improvements.md)
retains measured benefits, rejected experiments and links to accepted evidence.
