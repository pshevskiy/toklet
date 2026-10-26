# M8 delivery correction and research — 26 October 2026

The human requested research with Astra and correction of the active plan, then
replaced the fixed 131072-output requirement: **Chat and Expert should support
the input/output context window of the model being used.** The model's total
window is shared by the rendered input and generated continuation. A separate
documented output constraint also applies. Supporting the full window remains
the target; a smaller checked runtime/device limit is partial support and must
be displayed as such.

This review changes sequencing and acceptance criteria, not executable limits.
M0–M7 receipts remain scoped. M8, broader architectures and general Hugging Face
compatibility remain incomplete. The [engine plan](engine-expansion-plan.md),
[Gemma queue](gemma4-implementation-plan.md) and
[model context plan](long-reply-engine-plan.md) carry the active work.

## What the investigation established

The existing [short scalar E2B adapter](gemma4-short-scalar.md) already executes
the real checkpoint at 32 numerical positions. Raw generation, the independently
formatted basic-hello reply and [stream delivery](gemma4-stream-generation.md)
have selected Mac witnesses. Describing all scalar graph/session and real
generation work as absent was stale. Those diagnostics do not establish broad
quality, product readiness or full-window support.

The [forward implementation](../src/model/gemma_packed/forward.rs) separates
projections from CPU normalization, rotary, attention, activations and KV
transactions. The [GPU projection sessions](gemma-projection-session.md) and
lookup sessions now reuse immutable packed weights. Our engineering conclusion
is to compose these into a short hybrid Mac diagnostic before writing every
remaining resident attention/elementwise operation. Count synchronization and
readback explicitly. This is a route to whole-graph feedback, not a performance
claim or permission to expose Gemma in the app early.

The [complete storage reference](gemma-resident-storage-reference.md) is
committed at `da16ea9`. Native bundle code is in progress: eight pure admission/
integrity tests pass, but the new GPU storage path has not run. Its provisional
requested retained ownership is 2604665212 bytes and conservative load policy
2821589444 bytes. Neither figure is a measured footprint. The retained size alone
exceeds the tested [Qwen3 phone profile](mobile-qwen3-downloads.md)'s 2304 MiB
whole-process policy (2415919104 bytes). That policy is not a universal iOS limit;
it prevents inheriting the old phone fit claim. Gemma phone feasibility needs
its own decision before package/install work.

## Primary sources and resulting decisions

| Source checked | Finding | Plan consequence |
| --- | --- | --- |
| [Official pinned Gemma E2B config](https://huggingface.co/google/gemma-4-E2B-it/raw/3e22461f65e89153144f8adb70e3b8c2cc9845a7/config.json) | 131072 total text positions; local and full attention with shared KV. | Target that model window, including input and output together. The superseded request for 131072 output plus a prompt is no longer an acceptance condition. |
| [Official pinned Qwen3.5-0.8B config](https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/eb706f593d2d43c90a10271199c10b07ced7569a/config.json) and [pinned MLX text implementation](https://github.com/ml-explore/mlx-lm/blob/0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d/mlx_lm/models/qwen3_5.py) | 262144 positions; recurrent convolution/gated-delta layers plus full attention. | Optional future architecture, not a prerequisite for Gemma or the corrected requirement. Its remaining full-attention KV still grows with context. |
| [MLX Qwen3.5 candidate repository](https://huggingface.co/mlx-community/Qwen3.5-0.8B-4bit/tree/main) and [observed commit](https://huggingface.co/mlx-community/Qwen3.5-0.8B-4bit/commit/da28692b5f139cb0ec58a356b437486b7dac7462) | A quantized artifact is listed. Immutable config/index retrieval was unavailable through the research browser. | Candidate only; no tensor/header admission, payload acquisition, native execution or fit claim. Inspect exact revision and headers before selection. |
| [Hugging Face download contract](https://huggingface.co/docs/huggingface_hub/guides/download) | Downloads support full commit revisions and filtered file selection. | Resolve discovery to a pinned required-file manifest; reuse the existing downloader instead of downloading arbitrary repositories or executing repository code. |

Model configuration is evidence of declared semantics, not proof of numerical
execution, usable long-context quality or device fit. Generation defaults such
as `max_new_tokens` are not automatically architectural output limits. Resolve
conflicting config/model-card declarations explicitly per adapter; never infer a
universal field or use an MLX tag as compatibility.

## Next deliverables and dependencies

Each row is a coherent deliverable. Split code when a separately reviewable
mechanism requires it; every code task still needs Astra review, its applicable
checks and a commit. Reuse frozen witnesses and existing controllers. Add new
fixtures only for uncovered behavior. Do not insert unrelated ownership
refactors, model families or optimization prerequisites into this queue.

| Order | Deliverable | Exit evidence |
| --- | --- | --- |
| 1 — G4.1 | Complete the current immutable Gemma storage loader. | All original components and guarded buffers match the independent inventory; actual capacity/admission and whole-load peak are recorded; failed candidates preserve prior owners and original errors. No forward/app capability follows from storage alone. |
| 2 — G4.2 | Complete short Mac hybrid graph/session: GPU packed projections/gathers, existing CPU graph operations and transactional KV. | Tiny and real 32-position stage/logit comparisons, existing raw/formatted/streamed schedules, stop/cancel/failure/pending-token checks, complete memory admission. Avoid a second full CPU packed model; any bounded vector copy is counted. |
| 3 — G2.1/G5 | Investigate first divergent operations and freeze the checkpoint-specific product precision/quality decision. | Separate staged-FP32, ordered/math and original-BF16 results; preserved failures; held-out quality under the existing policy. No tolerance widening or silent contract substitution. Product acceptance stays blocked by failed required gates. |
| 4 — G4.3/L3/L4 | Compose resident execution and model-owned capacity admission; progressively qualify context. | Measure the hybrid bottlenecks, move costly stages without host waits per operation, qualify window wrap/shared producers/absolute positions, chunked prefill, cancellation and total memory. Start with a 512-position candidate before larger boundaries; this does not close the full-window target. |
| 5 — G6/L5 | Mac Chat/Expert and Hugging Face integration for the admitted checkpoint. | Search → inspect → pinned download → verify → load → stream → cancel → reopen offline; engine-owned model/codec/capability identity, truthful declared versus usable windows, existing profiles preserved. Short support may ship explicitly as partial. |
| Device branch — G7 | Gemma phone feasibility, then a separately qualified native/signed candidate. | A concrete representation/loading strategy fits an independently chosen whole-process budget; then physical generation, boundary/state/lifecycle/UI and memory checks. Preserve the passing installed baseline. An infeasible route stays open/rejected, not complete. |

L3's capability schema can be developed against already supported graphs without
waiting for all Gemma shaders. Full-window L4/L6 qualification continues after a
short Mac candidate. Generic Hugging Face H1–H4 follows each supported model; it
does not wait for every architecture E1–E6. Qwen3 plain Chat and selected physical
phone evidence already exist in [the current Qwen3 record](mobile-qwen3-downloads.md).

## Precision and acceptance ledger

| Evidence contract | Current state | Required action |
| --- | --- | --- |
| Frozen staged FP32 | Selected complete short scalar and projection/gather comparisons pass. | Reuse exact gates for hybrid/resident composition; compare intermediate states and complete distributions, not argmax alone. |
| Ordered/mathematical | [Real graph reference](gemma4-text-math.md) has 57 cross-precision failures across five arrays; each original raw scalar schedule rejects 27 ideal-logit values. | Locate first divergence and quantify its propagation; preserve original evidence and thresholds. |
| Original upstream BF16/GPU | Earlier dense and packed tiny GPU traces each have 15 discrepancies; eight rotary comparisons remain open. | Diagnose separately by dtype, rounding/order and position. Passing selected later rows does not waive these failures. |
| Product quality | The basic-hello example is a functional witness. | Freeze checkpoint/tokenizer-specific held-out quality and actual-context tests before adoption; apply the existing loss/task policy. |

A separately named FP32 execution profile may be investigated only with a
reviewed numerical and quality contract. It cannot retroactively turn rejected
BF16 or mathematical evidence into a pass. The product decision must identify
exactly which contract is required and why, with remaining differences visible.

## Full-window completion and sizing

Full support requires model-specific input/output/position semantics, successful
native admission and execution through the declared boundary, bounded text/UI
ownership, long-context quality and independent device evidence. A local sliding
window is not the model's total position limit. Fixed recurrent state does not
make positions or output text unbounded. Device-limited support remains useful,
but must be distinguished from achieving this requirement in full.

Astra's low-confidence engineering sizing is 2–5 focused working days for
storage plus a first hybrid whole-graph result, and roughly 2–6 weeks for a short
Mac product candidate if precision/quality discrepancies can be resolved. These
are effort ranges, not delivery promises or an ETA for full model windows. Replan
after each complete graph result. Full-window and phone dates require actual
fit, long-context correctness and measured throughput first. Broader architecture
coverage remains an ongoing program with named missing mechanisms.

This is documentation-only research and planning. It grants no new runtime,
model download eligibility, installed app, performance or device qualification.
