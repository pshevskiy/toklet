# Mac coverage and performance queue

5 November 2026. The user explicitly postpones physical iPhone work and asks for
engine model coverage and performance. Mac work proceeds independently; missing
phone checks and all historical numerical/quality rejections remain recorded.
M8 is not retrospectively declared fully qualified on iPhone.

Current priority, clarified by the user: **new models first**. Qwen1.5 and
SmolLM3 and Granite3.3 real Mac512 execution now pass their scoped checks.
Astra selects Granite3.1-1B-A400M sparse-MoE next: authenticate its exact artifact
and memory ledger before router/selected-expert blocks. Performance
work follows demonstrated needs; the draft P0 worker is preserved unreviewed
outside Cargo in `artifacts/m9-p0-deferred-2026-11-05/`.

Astra recommends these bounded deliveries:

1. **P0: current native MLX8 baseline.** Use the pinned Qwen2.5-0.5B eight-bit
   artifact, Half activations/F16 KV, Compensated normalization, Online attention,
   FullLogits, chunk128/capacity1152. Reuse frozen128/512/1024-token prompts and
   128 teacher-forced output IDs,3warmups/10samples, release builds and completed
   work. Collect uninstrumented timing separately from counters; identify every
   traced prefill chunk offset and explicitly sampled decode positions. Verify
   profiled/unprofiled outputs, count shared weights once and preserve memory
   limits (1.5GiB numerical /2.5GiB process trial). Rank normalization/projection
   before selecting any optimization.
2. **Coverage: Qwen2.5-1.5B-Instruct-8bit.** Pin the conversion, inspect exact
   tensors/config/tokenizer and full memory admission before acquiring weights.
   Reuse dense shared blocks; change code only for a demonstrated missing case.
   A real Mac512 diagnostic must cover formatted input/short greedy reply,
   chunk versus incremental operation, selected stock-MLX rows/decisions, reset
   and overflow preservation. The original1.5B checkpoint is already local.
   A second executed geometry is coverage; it does not establish managed Chat
   discovery or unrestricted support for every Qwen export.
3. **P1/P2: one useful optimization.** Choose parallel compensated normalization
   or a four-token affine prefill tile from P0's measured cost budget. Keep the
   existing fallback; apply unchanged numerical/quality/state/memory gates and
   independently repeat end-to-end measurements. Use1.5B as a secondary shape
   if qualified. Revert complexity without useful measured gain.

No concurrent model execution during acceptance timings. CPU/scalar references,
independent fixtures and static operation-boundary dispatch remain. No new
framework, speculative MoE layer or Gemma precision redesign belongs to this
queue. A future concrete mechanism needs its own fitting artifact and criteria.
Each code task gets checks, combined Astra review, a commit and concise progress
updates. Start with the1.5B metadata/memory inventory, then real execution.

The [M9 performance contract](next-performance-milestone.md) supplies measurement
and acceptance details; [coverage](model-coverage.md) remains the support index.
Phone qualification resumes later as a separate device task, not an obstacle to
this newly authorized Mac queue.
