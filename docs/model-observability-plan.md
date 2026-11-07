# Model roadmap: observability before optimization

Updated 7 November 2026. Selected order: **Gemma 4 E2B → Qwen3.5-4B →
Qwen3.5-9B**. Qwen2.5-1.5B remains a historical regression reference, not the
next adoption target. The [O1 opt-in pilot](o1-gemma-request-summaries.md) now
implements available worker-boundary observations. The additive
[O2.A phase API/report](o2-gemma-cost-report.md) now exposes completed native
phases. [O2.B1 numerical coverage](o2-gemma-workloads.md) adds two matched
cold/continuation batches and a ten-minute sustained run; UI/device/other-workload
gaps remain explicit and the 1% summary overhead bound is unqualified.
[O2.B2.A publication metrics](o2-ui-publication.md) correlate Send with controller
handling; visible-frame qualification remains separate. [Current coverage](model-coverage.md) describes
what actually runs. [PLAN](../PLAN.md#validation-and-performance-policy) retains
the binding numerical, quality, memory and measurement gates.

## Delivery order

| Stage | Delivery | Acceptance before advancing |
| --- | --- | --- |
| O1 | Versioned request summaries on the existing managed Gemma 4 E2B BF16 Mac route; disabled/summary modes and bounded JSONL export | Metric definitions below; authoritative success/rejection/error/Stop/replay state; clock and memory scope; collector pressure and exporter-failure checks; telemetry overhead qualified or explicitly opt-in |
| O2 | Gemma baseline, phase breakdown and offline comparison report | Exact artifact/build/workload manifests, raw samples, missing-field reasons and state counters; separate cold load, warm generation, continuation and sustained runs; identify the largest measured cost |
| G1 | Gemma 4 performance work and preferred uncensored MLX artifact evaluation | One measured change at a time; preserve qualified BF16/G2 reference. New weights/quantization get independent fidelity, task-quality and memory gates before managed adoption |
| Q1 | Qwen3.5-4B, preferred uncensored affine MLX 4-bit | Inspect actual graph/packing first; qualify loader, hybrid recurrent/attention state, template/stops, capacity and cancellation; expose the same request metrics before optimization |
| Q2 | Qwen3.5-9B using the qualified mechanisms from Q1 | Separate artifact/shape/quality/admission evidence; repeat baseline and overhead checks. Keep the 6 GiB ceiling; a failed fit does not justify silent paging or a budget increase |
| S1 | Five under-4B candidates below, one at a time | Measured quality/latency/memory shortlist, then explicit individual adoption decisions. If Q2 is memory-blocked, record it and evaluate this smaller-model track |

The remaining [under-30B candidates](models-under-30b-2026-11-07.md) are research
options for later: Gemma 12B/26B, Qwen3.8-27B, Devstral Small 2, LFM2.5-24B,
Ministral 14B, gpt-oss-20b and Granite 8B. They do not displace the selected order
or expand the current working-set budget. iPhone remains deferred.

## Preferred MLX artifacts

Prefer **uncensored/abliterated MLX** deployments, with a reproducible lineage
and a verified numerical/quality result. These are publisher descriptions, not
an assurance of capability or refusal behavior. Preserve the original checkpoint
as the reference; record the modified checkpoint as a distinct model identity.

| Main route | Candidate, inspected revision prefix | Format and adoption work |
| --- | --- | --- |
| Gemma 4 E2B first | [deadbydawn101/gemma-4-E2B-Heretic-Uncensored-mlx-4bit](https://huggingface.co/deadbydawn101/gemma-4-E2B-Heretic-Uncensored-mlx-4bit), `cde01efc` | Affine 4-bit, group 64; derived from p-e-w's Heretic ARA checkpoint. Evaluate after O1/O2 on the qualified official BF16 route. Existing Gemma Q4/QAT quality failures remain binding for their artifacts; they neither qualify nor automatically reject this different artifact |
| Qwen3.5-4B second | [avan-ag/Qwen3.5-4B-Uncensored-MLX-4bit](https://huggingface.co/avan-ag/Qwen3.5-4B-Uncensored-MLX-4bit), `fb98fbc5` | Affine 4-bit, group 64; DavidAU fine-tuned lineage, not simply stock Qwen quantization. Preserve this distinction in quality and timing comparisons |
| Qwen3.5-9B third | [lukey03/Qwen3.5-9B-abliterated-MLX-4bit](https://huggingface.co/lukey03/Qwen3.5-9B-abliterated-MLX-4bit), `8ff61b6a` | Affine 4-bit, group 64, text-only configuration. Inspect exact packed tensors and memory before admission; do not transfer multimodal claims from the official model |

Full inspected revisions/configuration are in the
[artifact record](../artifacts/model-observability-plan-2026-11-07/artifact-candidates.json).
At implementation freeze every consumed file hash, tokenizer/template, converter
version, quantization mode/group/exclusions and source revision. Prefer direct
conversion from high-precision modified weights over re-quantizing GGUF-derived
weights. An MXFP4/NVFP4 model is not interchangeable with affine Q4.

Qualify two differences separately: the modified high-precision checkpoint versus
the original on held-out tasks, then quantized/native execution versus that exact
modified checkpoint. Keep the existing loss/task gates; report intentional text
behavior changes separately without using them to excuse unrelated capability
loss. An MLX repository is an artifact/external-baseline candidate, not evidence
of Toklet loader, kernel or managed-app support. No blanket family adoption.

## Five candidates strictly below 4B total parameters

Ranked research shortlist for capability, local efficiency and useful model
coverage, not a common-harness leaderboard. Counts below are full unquantized
checkpoint tensor-element counts from the publisher's Hub metadata, including
stored embeddings and modality tensors; neither active nor effective counts.
The [metadata receipt](../artifacts/model-observability-plan-2026-11-07/model-metadata.json)
and artifact record preserve revisions. Recheck unique tensor ownership/aliases
and every derivative's actual shapes at admission.

| Rank | Official reference | Total parameters | Reason to evaluate | Uncensored / MLX path |
| --- | --- | ---: | --- | --- |
| 1 | [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) | 2,516,756,480 | Strong small reasoning, coding and tool-use candidate; ordinary Llama-style model type | [mondk uncensored MLX 4-bit](https://huggingface.co/mondk/MiniCPM5-2B-Abliterated-Uncensored-MLX-4Bit), `8ebe35e0`; affine/group 64, independently qualify |
| 2 | [LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) | 1,728,801,536 | Recent compact agent/instruction-following model; useful performance experiment with a new architecture | [mchen04 uncensored MLX](https://huggingface.co/mchen04/LFM3-1.7B-Uncensored-MLX), `095df4e8`, exists but expands quantized GGUF into F32. About 6.44 GiB of weights alone: do not adopt under 6 GiB. Find/convert a high-precision uncensored source to bounded MLX quantization; artifact gate remains open |
| 3 | [Granite 4.2 3B](https://huggingface.co/ibm-granite/granite-4.2-3b) | 3,659,737,600 | Dense reasoning/tool-use baseline with adjustable thinking effort | [Heretic source](https://huggingface.co/saidutta69/granite-4.2-3b-heretic) found; uncensored MLX conversion pending. [Official MLX Q4](https://huggingface.co/ibm-granite/granite-4.2-3b-q4-mlx) is a reference fallback, not uncensored |
| 4 | [Ministral 3 3B Reasoning](https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512) | 3,849,090,048 | Compact STEM and vision reasoning; count includes vision encoder | [Heretic source](https://huggingface.co/xtools-at/Ministral-3-3B-Reasoning-UNCENSORED-heretic) found; uncensored MLX conversion pending. [MLX 4-bit reference](https://huggingface.co/mlx-community/Ministral-3-3B-Reasoning-2512-4bit), `2cd2087a`, uses affine/group 64 |
| 5 | [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | 2,274,069,824 | Small vision/reasoning candidate that can reuse mechanisms qualified for Qwen3.5-4B/9B | [TheCluster uncensored MLX BF16](https://huggingface.co/TheCluster/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-MLX-bf16), `78efacde`; inspect its changed tensor count/lineage before conversion, then evaluate affine Q4 for deployment |

Gemma 4 E2B's effective name does not make it under 4B total; it remains the
main first model. Qwen3.5-4B is also outside this strict tier. Nanbeige4.2-3B was
investigated but has 4,169,800,704 stored parameters at `b82e54bd`, so is excluded.
Do not use quantized container word counts to claim a smaller parameter class.
Availability of a publisher's conversion is not a passing local benchmark.

## Metric contract

O1 must make each handled request observable before performance changes. Reuse
[`generation::Report`](../src/generation.rs), completed
[`metal::Timing`](../src/metal/mod.rs), worker events and owner-memory reports.
Use a schema version, request/attempt/session IDs and owner generation. Attach
immutable manifests once: engine/build, device/OS/power mode, artifact/revision/
hashes, model lineage, tokenizer, quantization/KV format, backend/kernel choices,
context/chunk/output capacity, cache state, prompt/token hashes in benchmark
captures, sampling/seed, thinking mode/budget and instrumentation mode.

| Metric fields | Definition / unit | Collection and interpretation |
| --- | --- | --- |
| `request_total_ms`, `native_total_ms` | User submission to terminal UI state; native generation start to terminal worker outcome, respectively | Monotonic host clock; nested intervals, not additive. Record outcome and observed partial duration on errors/cancellation |
| `ttft_event_ms`, `ttft_visible_ms`, `first_answer_visible_ms` | Submission to first emitted token event, first nonempty displayed text, and first displayed final-answer text | Reasoning text may precede the answer; preserve thinking mode. No token/text means null with reason. A callback is not a render acknowledgment |
| `queue_ms`, `tokenization_ms`, `output_decode_ms`, `ui_delivery_ms` | Submission→worker start; formatting/tokenization; token→text decoding; correlated publication→render acknowledgment | Separate boundaries; missing render/phase probes remain null. Never infer a UI or GPU clock from another clock without calibration |
| `load_ms`, `conversion_ms`, `compile_ms`, `ready_ms` | Completed model acquisition/load phases and process start→ready | Separate download from disk load; new process is not proof of cold filesystem cache. Label cache state and component overlap |
| `prefill_ms`, `prefill_tokens_per_s` | Completed prefill wall duration; newly evaluated prompt rows divided by prefill seconds | Exclude reused prefix rows from numerator; report reuse and evaluated rows separately. Fully cached/no-evaluation case is not infinite throughput |
| `decode_ms`, `decode_tokens_per_s` | Completed decode phase duration; `(N-1)/(last_token_time-first_token_time)` for N emitted model-token events >1 | Follow PLAN's formula, excluding first-token prefill from the rate. Same event source and stop policy; N≤1/zero interval gives null. Count thinking/answer tokens separately where tokenizer semantics support it |
| `token_interval_ms`, `ui_update_interval_ms` | Successive model-token event gaps; successive visible update gaps | Distinct distributions; chunked callbacks cannot establish per-token latency. Bounded aggregates/histograms, no unbounded token trace |
| `sampling_ms`, `host_encode_ms`, `gpu_completed_ms`, `host_wait_ms`, `readback_ms` | Existing phase scopes; GPU start→completion on its own clock | Host sampling or cached-ID fetch only; fused GPU selection stays charged to evaluation. GPU execution overlaps host wait; do not sum them as exclusive phases |
| `prompt_tokens`, `reused_prefix_tokens`, `prefill_rows`, `decode_evaluations`, `sampled_tokens`, `emitted_tokens`, `committed_tokens`, `pending_tokens` | Authoritative work/state counts, with prompt/decode scopes documented | Commit only after successful work; no fabricated KV advancement on partial failure. Record cache action/replay reason and replayed rows |
| `weights_owned_bytes`, `kv_allocated_bytes`, `kv_valid_positions`, `scratch_bytes`, `telemetry_bytes`, `app_allowance_bytes` | Unique physical owners and separately labelled admission allowances | Count shared storage once; preserve capacity versus valid state; include hybrid recurrent/convolution state and file-backed PLE explicitly |
| `footprint_current_bytes`, `process_peak_bytes`, `load_peak_bytes`, `memory_pressure` | OS-measured process observations with instrument/scope/time | Lifetime peak is not request peak. Sampled maxima are lower bounds; unavailable isolated load peak is null. Admission estimates are not measured footprint |
| `command_buffers`, `dispatches`, `host_waits`, `readback_bytes`, `buffer_growths`, `pipeline_builds`, `steady_state_allocations` | Existing counters or cheap operation-boundary counters | Mark unsupported counters unavailable; distinguish Rust numerical allocations from native/app allocations. No per-element events or extra GPU waits |
| `terminal_status`, `failed_phase`, `error_category`, `cancel_to_terminal_ms`, `replay_required` | One authoritative outcome for success/Stop/cancel/reject/error; cancellation signal→terminal acknowledgment | Offline success/failure/rejection/cancellation rates use attempted-request denominator. Process death is incomplete, not a fabricated terminal event |
| `telemetry_dropped`, `export_errors`, `collector_peak_bytes`, `summary_overhead_pct` | Record completeness/resource counters; `100*(median_summary/median_disabled-1)` separately for TTFT and total time | Bound both collection and retention; dropping telemetry never changes inference/session state |

Every observation includes availability and instrument/scope; **unknown is not
zero**. Counters and stable enums at request/chunk boundaries stay outside
numerical hot loops. No routine prompts, output text, token arrays, string maps,
per-token JSON or synchronous disk writes. Keep IDs in records, not aggregate
metric labels. Optional energy/J per token and device-bandwidth counters require
an identified, qualified instrument; modelled bytes are not measured DRAM traffic.

## Measurement and acceptance gates

1. **O1 correctness and bounded overhead first.** Check terminal outcomes,
   empty/no-output and single-token cases, pending/committed state, fully cached
   prompts, reset/replay, counter overflow, ring wrap, dropped records and failed
   export. Freeze field units and reconcile counters with authoritative state.
   Maximum 16 KiB collector/request, 1 MiB process ring and 4 KiB serialized
   summary; account for these bytes. No new steady-state numerical allocation,
   GPU command or wait. Qualify ≤1% median TTFT and request-time overhead in paired
   disabled/summary trials; if noise cannot resolve it, collect more evidence or
   keep summary mode opt-in, not declared qualified.
2. **Baseline before optimization, on every selected artifact.** Release builds,
   named Mac and power/thermal conditions; three warmups, ten measured trials,
   raw samples, median and dispersion, then an independent confirmation run.
   Token-matched numerical and user-visible app comparisons are separate tracks.
   Record tokens and wall time together so reduced reasoning output cannot be
   misreported as a faster kernel. Use MLX as an external reference, not proof
   that Toklet can execute the same artifact.
3. **Admitted workload matrix.** Start the existing Gemma pilot at its qualified
   context/chunk/reply limits, including a 384-input/128-output case only if its
   actual admission passes. For each new model run 128/512/1,024-input plus
   128-output cases only when admitted; report unsupported cases, never truncate.
   Include empty-session warm runs, exact append, fully cached/pending tokens,
   prefix edits/replay, boundary rejection, Stop, failed execution/output and
   separate new-process loading. Continue sustained evidence for at least ten
   minutes where required. No automatic 128K/256K context expansion.
4. **Useful reports first.** JSONL plus offline table of phase share, TTFT,
   first-visible/final-answer time, prefill/decode rate, request/continuation time,
   memory and outcomes. Report p50 with sample count; ten requests do not qualify
   p95/p99. Collect a separately sized tail study and uncertainty before making
   tail claims. Diagnostic operator capture is opt-in and excluded from acceptance
   timing; capture coverage/overflow is explicit.
5. **Retain one measured improvement.** Declare primary metric and hypothesis
   before changing code; target ≥5% beyond noise and confirm separately, with no
   unexplained required-workload regression. Preserve quality, numerical/session
   semantics and the 6 GiB engine ceiling including load transients. Uncensored
   and quantized variants need their own gates; prior BF16 results do not transfer.
   Record retain/revert decision and exact evidence/commit.
6. **Required implementation checks and review.** Cargo fmt/clippy/tests with
   relevant supported features, changed model/session/GPU checks, Swift/app
   lifecycle checks, real Mac baseline and overhead trials; retain full logs.
   Submit one combined diff/invariants/check-results request to Astra; resolve
   findings before code completion/commit. The original planning commit was
   documentation-only; the O1 implementation has its own linked check/review
   receipt and does not qualify missing native/UI metrics.

Source/metadata inspection and documentation checks are recorded in
[checks](../artifacts/model-observability-plan-2026-11-07/checks.json). No new
model, speed, overhead or memory qualification is claimed by this plan.
