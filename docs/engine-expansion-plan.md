# Engine coverage and Hugging Face implementation plan

Requested 23 October 2026. This is the active ordering for the remaining
[M8 work](mlx-implementation-plan.md), following [PLAN.md](../PLAN.md)'s
correctness, quality, measurement and memory policies. Earlier qualification
records remain scoped to their original code and checkpoints.

## Outcome and boundaries

Toklet should run a growing, explicit set of LLM graph families in native
Rust/Metal, then find, download and use compatible Hugging Face models from its
Mac and iPhone apps. Support is determined by the actual configuration, tensors,
tokenizer and chat semantics, rather than a repository name or an MLX tag.

“All known architectures and optimization techniques” is an ongoing coverage
program, not a finite compatibility promise. The [pinned source inventory](mlx-architectures.md)
is the initial reference universe. New upstream graphs enter the backlog after
inspection. Every known missing mechanism stays visible. Neither listing a
family nor sharing one operator qualifies its variants or every checkpoint.

Use separate evidence states for graph implementation, artifact admission,
numerical parity, chat correctness, device fit and physical qualification.
Downloads may be enabled for a validated compatible configuration before it has
a full performance report, but must say so. Rejected representations, including
the previously rejected Qwen four-bit snapshot, remain rejected.

## Modular implementation contract

Requested explicitly by the user: adding an architecture must be easy to review,
maintain and understand. Keep architecture-specific config admission, weight
mapping and forward-order semantics in small named modules once they exist.
Share artifact reading, affine layouts and numerical kernels where their actual
semantics match. Keep tokenizer, chat formatting, downloads and UI outside the
numerical graph. Avoid a growing central parser containing every family's rules.

Each implemented graph documents its shapes/layout, norm/position/activation
order, immutable weights, mutable session state, memory plan and rejected options.
A new family should add its own checked description and fixtures, then compose
existing operators or introduce one explicitly tested missing mechanism. Common
validation helpers must preserve family-specific rejection rules. Choose the
graph/backend at a coarse boundary using concrete types or a bounded enum; avoid
trait objects or architecture switches inside numerical inner loops. Do not
scaffold empty modules or build a general tensor framework for hypothetical use.

The engine owns compatibility facts so app and download code need not repeat
architecture interpretation. Refactor an existing module only when a concrete
second use establishes the shared mechanism, with existing fixtures preserved.

## Coverage before E1

This table records the plan's starting point. Completed Qwen3 work is tracked in
the status below; its tiny fixtures do not qualify real checkpoints.

| Area | Implemented at plan start | Work identified at plan start |
| --- | --- | --- |
| Dense graphs | Qwen2; selected bias-free Llama and full-attention Mistral | Qwen3, independent attention width, Q/K normalization, rotary/window variants, other normalization and activation graphs |
| Representation | Validated MLX affine U32 packing, selected four/eight-bit artifacts and mixed floating exclusions; existing Toklet Half/Q4 paths | Additional formats and schemes need their own reconstruction oracle, quality and kernel gates |
| Other graphs | No native MoE, recurrent/hybrid, MLA or media execution | Routing/expert layouts, recurrent state, compressed attention, encoders and real media inputs |
| Execution | Scalar reference, resident Metal weights/KV/scratch, existing selected prefill/decode and attention paths | Port or qualify optimizations per new graph; compatibility does not follow automatically |
| App | Mac/iPhone Chat, Models, Expert; public Hugging Face search; three pinned download/chat profiles | Config-driven compatibility and resource admission, broader snapshot acquisition and chat profiles |
| Devices | Selected Mac M1 Pro and iPhone 14 Pro evidence | Every new graph/configuration has separate checks; Mistral 7B is denied on the phone's current budget |

## Execution queue

Each row is a sequence of small code tasks, each reviewed by Astra and committed
after relevant checks pass. Freeze an independent oracle before its target
implementation. Do not combine an unrelated optimization with a graph change.

| Order | Work package | Concrete tasks and completion evidence |
| --- | --- | --- |
| E1 — active | Qwen3 dense | Independent tiny upstream/scalar oracle with Q/K per-head RMSNorm and query width different from hidden width; checked scalar loading/execution; resident Metal graph; session/cache/cancellation/memory tests; immutable small real checkpoint and tokenizer/chat probes; Mac then iPhone trials. Start Metal with Reference attention; qualify Online separately. Qwen3 MoE is separate. |
| E2 | Dense graph variants | Inspect and implement sliding/local attention, supported rotary scaling/partial rotary and independent head geometry; then Gemma- and Phi-family graphs one variant at a time, including their actual norm, activation, embedding/logit scaling or soft-capping rules where required. Each unsupported option rejects explicitly. |
| E3 | Expert models | Freeze deterministic routing/tie/normalization fixtures; bounded top-k routing and expert layouts; scalar expert accumulation, resident GPU execution and memory planning; one small real MoE checkpoint. Shared experts and routing variants are separate cases. |
| E4 | Compressed and recurrent/hybrid attention | MLA-style projection/cache semantics and recurrent/state-space or linear-attention updates in separate tasks; exact rollback/replay and prefix validity, independent state witnesses, bounded state memory and a real checkpoint for each implemented graph. |
| E5 | Multimodal | Start with one vision-language graph: image decoding/preprocessing, encoder, connector, positions and insertion; real-image reference and app tests on both devices. Audio, video and other modality graphs remain separate backlog items. |
| E6 | Remaining inventory | Map every pinned loader entry to its actual mechanisms and evidence or a named missing mechanism. Implement additional useful families in that order. Audit new inventories periodically during requested implementation work; no automatic monitoring is created. |
| H1 — after E1 native compatibility | General compatibility probe | One engine-owned, bounded config/tensor admission report with machine-readable reasons, representation and resource requirements. The app consumes this report instead of maintaining a second architecture allowlist. Distinguish unknown, unsupported, compatible but untested and qualified. |
| H2 | General Hugging Face discovery and acquisition | Resolve a selected public result to an immutable commit; inspect bounded config, index and tensor headers; admit supported graph/format/tokenizer and exact required files; storage/load/session budgets; stream/hash shards; joined cancellation/retry; atomic snapshot publication and revalidation. Never execute repository code. Gated/private access needs a separate explicit account flow. |
| H3 | Downloaded chat | Bounded, explicitly supported tokenizer/chat-template evaluation and stop semantics; ordinary offline loading and multi-turn Chat; clear reasons and recovery for unsupported templates, missing files, changed budgets, network failure and corrupt caches. Qwen3 thinking/control tokens and templates need their own probes; do not inherit Qwen2 formatting. Generic manifests must retain existing pinned-profile receipt compatibility. Preserve existing pinned profiles as regressions. |
| H4 | End-to-end app qualification | Search → inspect → download → verify → open → chat → stop → reopen offline on Mac and iPhone. Cover unsupported models, insufficient memory/storage, interruptions, backgrounding and ownership-safe cancellation. Publish exact model/device coverage. |

E2–E6 do not block H1–H4 once a concrete graph is implemented. The user's engine
first dependency is enforced per model: search never upgrades an unsupported
graph into an executable one. Existing search/download/chat remains usable while
the engine grows.

The first Qwen3 target is justified by an observable missing mechanism, rather
than an alias: the official [0.6B config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)
declares hidden width 1024, 16 query heads and head width 128. The query projection
therefore has width 2048. At plan start, the native invariant equated query width
with hidden width. E1's scalar and tiny Metal tasks now support independent
widths and per-head Q/K normalization. Their real-checkpoint qualification still
needs independent checking
against the [Qwen3 implementation](https://github.com/ml-explore/mlx-lm/blob/0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d/mlx_lm/models/qwen3.py).
These are graph requirements, not measured performance claims.

## Optimization coverage

Maintain a named experiment record, not an “all optimizations enabled” switch.
Select candidates from measurements on the implemented graph and device.

| Technique group | Work and acceptance |
| --- | --- |
| Weight formats and precision | Add only explicit layouts/dtypes with reconstruction, exclusions and rounding oracles; fuse unpacking with products; no full-model expansion per step. Apply unchanged loss/task gates to lossy adoption. |
| Projection and prefill | Profile decode bandwidth/dispatch and prefill reuse separately; evaluate available Apple matrix primitives, tiling and batching for real shapes before substantial custom GEMM. Retain only an end-to-end benefit. |
| Attention and context | Qualify online/tiled attention, local windows, grouped-query traversal, KV layout/quantization and chunk sizing per graph; causal/offset/capacity and memory witnesses are mandatory. |
| Scheduling and fusion | Reuse pipelines/resources, reduce synchronization/readback and evaluate fusion with bounded scratch/register pressure; preserve valid-prefix failure/cancellation semantics. |
| Prompt reuse and generation | Exact-prefix reuse and pending tokens first; speculative decoding, prefix sharing/branching or continuous batching require separate ownership, correctness and end-to-end experiments before adoption. |
| Device execution | Capability-selected Metal paths and separately bounded CPU/ANE or newer API experiments; no assumption that a desktop win or API availability transfers to iPhone. |

Every retained optimization records hypothesis, bytes/operations/access pattern,
implementation difference, exact hardware/build/model/workload, correctness,
quality, timing, memory, and retain/revert decision. Use release builds, three
warmups and ten measured samples, plus a second run for a promising gain; the
existing default retention threshold is at least 5%, beyond observed variation.
Quality baselines must match the checkpoint and tokenizer; Qwen2 results cannot qualify a new family. Functional probes do not satisfy this performance gate. Energy stays unavailable
unless a qualified instrument supplies units and scope.

## Per-task checks and publication

For Rust changes: formatting, strict Clippy and CPU tests; supported Metal feature
checks and explicit hardware tests when affected. Loader changes include
malformed/overflow/budget fixtures. Session changes include exact-prefix,
fully-cached, pending-token, cancellation and capacity recovery. CPU tests remain
offline. Reference-tool changes run their portable and explicit producer checks.
UI/download changes include strict controller builds and targeted integration.

For each completed task, publish the exact scope and remaining gaps, preserve raw
failures and source/model identities, update [improvements.md](improvements.md),
obtain Astra's code/evidence acceptance and commit. Existing immutable receipts
continue to describe their original snapshots. No full-family, quality,
performance or physical-device claim follows from a tiny fixture alone.

H1 foundation: [local engine-owned compatibility reporting](mlx-compatibility.md)
now shares bounded artifact admission, scalar estimates and exact Metal geometry
policy with loading. The [CLI/native consumers](mlx-compatibility-consumers.md)
also pass. The [shared Swift worker adapter](mlx-compatibility-swift.md) adds
metadata preflight and safe summary invalidation. The explicit Rust
[runtime Metal resource planner](metal-resource-planning.md) now shares the
loader/session layouts and accounting. Its [CLI/native runtime consumers](metal-resource-consumers.md)
now expose immutable plans. A shared [native factory policy](native-resource-policy.md)
resolves attention and publishes its fixed bridge allowance for app callers.
The Swift/app runtime consumer and receipt-based
qualification remain open. No installed catalog is broadened. The [bounded Chat output extension](chat-output-limit.md)
adds 256-token replies within the unchanged 512-token app context; Swift raw
diagnostic commands stay capped at 128.

Status: E1 has an [independent tiny Qwen3 oracle](mlx-qwen3-oracle.md); [native scalar execution](mlx-qwen3-scalar.md) now passes its tiny gates. [Resident tiny Metal execution](mlx-qwen3-metal.md) also passes its frozen gates. An [immutable real Qwen3 candidate](mlx-qwen3-checkpoint.md) is acquired and structurally inspected. Its [independent real numerical reference](mlx-qwen3-real-oracle.md) is frozen, and [selected real Mac execution](mlx-qwen3-real-metal.md) passes the frozen numerical/session gates. [Public generation callback recovery](mlx-qwen3-real-generation.md) also passes. Quality, chat/app integration and iPhone qualification remain next. E1–E6 and
H1–H4 remain open until their stated evidence exists. No reliable whole-program
ETA is assigned to an open-ended architecture inventory.
