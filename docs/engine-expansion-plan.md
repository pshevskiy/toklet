# Engine coverage and Hugging Face implementation plan

Latest Gemma precision work: the [packed table/projection witness](gemma-packed-reference.md)
passes required FP32/tied CPU/GPU comparisons while retaining BF16 backend
differences and rejected projection diagnostics. It grants no new artifact,
download/Chat or device capability.

The [native affine row fetch](mlx-affine.md#native-row-fetch) now supplies a
bounded FP32 packed-table primitive for later graph composition, with unchanged
artifact/app eligibility.

A [complete tiny packed Gemma graph](gemma-packed-graph-reference.md) now supplies
the independent reference for packed scalar model/session composition. Its source
CPU passes all comparisons, with15 new GPU intermediate discrepancies retained.

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

## Current user priority: Gemma4 and single-reply128K

The user prioritized Gemma4 and131,072 tokens in one reply, for Chat and Expert.
Finish the current small [normalization reference](gated-norm-reference.md), then
follow the [Gemma4 priority queue](gemma4-implementation-plan.md) before unrelated
architecture expansion. The first [Gemma4 E2B inspection](gemma4-e2b-inspection.md)
maps source/config/headers and names missing mechanisms. The
[tanh-GELU/GeGLU witness](gemma-gelu-reference.md) is frozen; its standalone
[scalar target](gemma-gelu-scalar.md) passes offline and Rust/iOS checks. G2
continues with [proportional rotary precision](gemma-rotary-precision.md), whose
long-position diagnostic disagreements remain open. The
[standalone scalar math reference](gemma-rotary-scalar.md) passes its frozen
witnesses and required checks. The
[local-attention witness](gemma-local-attention-reference.md) is frozen before
the [checked scalar target](window-attention-scalar.md). The
[KV staging/ring witness](gemma-kv-reference.md) now passes independent logical
state and pinned upstream CPU/GPU checks. Its [cache owner](window-cache-scalar.md)
now passes batch commit/discard checks; full model transactions and shared
producer ownership remain open. The [shared routing witness](gemma-shared-kv-reference.md)
qualifies selected-source forwarding/identity only. The
[native routing/publication plan](shared-kv-scalar.md) now passes ordered-prefix
and cache-batch checks. [Borrowed-span scalar attention](span-attention-scalar.md)
now qualifies numerical ring views. The
[per-layer lookup/projection witness](gemma-per-layer-reference.md) freezes
scales and normalization axes. Its [native FP32 composition](per-layer-scalar.md)
now passes the frozen contracts. The [dense decoder-order witness](gemma-decoder-order-reference.md)
freezes norm/residual/PLE/scalar order with an explicit external attention
boundary; its [native FP32 finish](dense-decoder-scalar.md) passes the frozen
contracts. The [unlearned V-norm witness](gemma-v-norm-reference.md) freezes
the per-head norm; its [scalar recipe](gemma-v-norm-scalar.md) now passes
the accepted fixture. The [logit-softcap witness](gemma-softcap-reference.md)
freezes its separate contracts; the [native in-place transform](gemma-softcap-scalar.md)
now passes them. Quantized backend and full graph/session
integration remain open.
Qwen3 completion and the
broader inventory remain open. The
[long-reply engine plan](long-reply-engine-plan.md) sequences bounded text
streaming, engine-owned capabilities and a qualified long-context graph within
E2/E4/H1/H3/H4. It does not close E1–E6 or broaden current runnable limits. The
[mobile preparation](long-reply-mobile.md) remains a separately scoped UI task.

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
| E1 | Qwen3 dense | Independent tiny upstream/scalar oracle with Q/K per-head RMSNorm and query width different from hidden width; checked scalar loading/execution; resident Metal graph; session/cache/cancellation/memory tests; immutable small real checkpoint and tokenizer/chat probes; Mac then iPhone trials. Start Metal with Reference attention; qualify Online separately. Qwen3 MoE is separate. |
| E2 — priority G1–G7 | Dense graph variants | Inspect and implement sliding/local attention, supported rotary scaling/partial rotary and independent head geometry; then Gemma- and Phi-family graphs one variant at a time, including their actual norm, activation, embedding/logit scaling or soft-capping rules where required. Each unsupported option rejects explicitly. |
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
The [Swift worker runtime consumer](swift-runtime-preflight.md) now rejects
non-fitting plans before numerical allocation and preserves factory revalidation.
Receipt-based qualification remains open. No installed catalog is broadened. The [bounded Chat output extension](chat-output-limit.md)
adds 256-token replies within the unchanged 512-token app context; Swift raw
diagnostic commands stay capped at 128.

E4/L4 foundation: the independent [unmasked gated-delta reference](gated-delta-oracle-preparation.md)
is frozen before its [normalized scalar recurrence](gated-delta-scalar.md).
The operator writes a separate candidate state and preserves its immutable input
on failure. Its separate [raw Q/K normalization](gated-delta-normalization.md)
also passes, and [scalar gate construction](gated-delta-gates.md) completes the
raw-input recurrence operator chain. The separate [causal convolution reference](causal-conv-reference.md)
checks tap/channel order, SiLU and carried history against pinned upstream CPU/GPU
operators before its [checked scalar step](causal-conv-scalar.md). The step keeps
old history immutable and prepares raw-input history separately; SiLU remains an
existing operation. Gated output normalization, masked policy, hybrid graph and
session, Metal state, real-checkpoint and long-context/device gates remain open.

Status: E1 has an [independent tiny Qwen3 oracle](mlx-qwen3-oracle.md); [native scalar execution](mlx-qwen3-scalar.md) now passes its tiny gates. [Resident tiny Metal execution](mlx-qwen3-metal.md) also passes its frozen gates. An [immutable real Qwen3 candidate](mlx-qwen3-checkpoint.md) is acquired and structurally inspected. Its [independent real numerical reference](mlx-qwen3-real-oracle.md) is frozen, and [selected real Mac execution](mlx-qwen3-real-metal.md) passes the frozen numerical/session gates. [Public generation callback recovery](mlx-qwen3-real-generation.md) also passes. The exact [cold native plain Chat profile](qwen3-native-chat.md) now passes guarded shared-tree codec/C API checks; whole-app/package/iPhone admission and broader quality remain separate. E1–E6 and
H1–H4 remain open until their stated evidence exists. No reliable whole-program
ETA is assigned to an open-ended architecture inventory.

Gemma's [complete tiny graph reference](gemma-text-graph-reference.md) now covers
token embedding through the tied capped head at every position0..31, with
independent math and passing source CPU comparisons. The original GPU comparison
retains15 failed intermediate checks; final logits alone do not grant a pass.
This is partial G2 evidence, with G2/long rotary, native graph/session, Metal,
checkpoint, app and device stages still open.

The corresponding [bounded scalar graph/session](gemma-text-graph-scalar.md)
now executes that exact synthetic geometry, with borrowed immutable FP32 weights,
shared/rotating KV and atomic publication after final logits. It is a G3
foundation with32 total positions, without checkpoint admission, quantized
fallback, new catalog eligibility or a GPU/long-context gate waiver.

The separate [Gemma4 E2B config/layout adapter](gemma4-config-layout.md) now
validates the inspected real text graph and exact MLX physical tensor mapping.
It handles bounded metadata only, requires explicit known-modality exclusion,
and leaves executable-family and compatibility eligibility unchanged. Payload
integrity/values, real numerical graph, resources and device/app gates remain.

The [native packed Gemma graph/session](gemma-packed-model-scalar.md) now composes
borrowed matrices and reusable atomic session state for the frozen synthetic
H64 profile. A private layout separates concrete geometry from execution;
arbitrary public geometry is not admitted. It is a G3 foundation, with actual
E2B payload/geometry, general generation, resident Metal and real-model/app/device
qualification still open. Numerical/GPU/long-position gates remain unchanged.

Gemma's [cold payload storage](gemma4-payload-storage.md) now keeps borrowed-model
backing owners separate from execution, with complete-stream integrity and
text-only residency admission. Its offline synthetic success and public E2B
metadata rejection do not qualify real acquisition/load, numerical execution,
quality, performance, device memory or compatibility eligibility.

The [immutable Gemma4 checkpoint and Mac storage load](gemma4-checkpoint.md) now
admit the actual packed text payload and540 borrowed views after complete-file
integrity/value checks. The failed first transfer is retained separately from
the successful recovery. This is acquisition/storage evidence only: real E2B
execution, tokenizer/chat/generation, resident Metal, quality, representative
performance and device/app gates remain open.

The [packed graph proposal mechanism](gemma-producer-staging.md) now supports a
validated private prefix of up to15 producers with persistent stage owners and
inline borrowed proposals. A35-layer structural test covers routing and atomic
failure/retry without admitting public E2B execution. All real numerical,
generation, Metal, quality, device and app gates remain open.

The [independent tiled four-bit reference](affine4-tiled-reference.md) avoids
whole-table expansion for large embeddings and projections. Eight offline tests
and a bounded actual Gemma4 matrix probe pass, with scalar bitwise row checks
and unchanged projection tolerances. This advances real-checkpoint reference
tooling only; full numerical graph, generation, Metal, quality, device and app
qualification remain open.


The [complete four-token E2B mathematical reference](gemma4-text-math.md) now
streams all713 states in each of two precision contracts over the actual pinned
checkpoint. Every1,426 future-prefix comparison preserves exact typed bytes.
Independent scalar replay verifies complete cross-precision observations and
compact probes within a separate512MiB reference-process allowance. Five arrays
reject57 cross-precision values under unchanged tolerances; that gap remains
explicit. This advances real-shape reference evidence only. The named native
adapter, checkpoint numerical parity, generation, Metal, quality, device and app
qualification remain open, with all earlier precision failures unwaived.


The [named short native E2B adapter](gemma4-short-scalar.md) now borrows the
admitted payload into the existing packed scalar graph. Physical512-slot local
rings are separate from the unchanged32-position execution limit, and retained
store capacity/metadata are counted once. Real four-token batched, one-token
and mixed schedules produce identical logits and pass every staged-FP32 value.
Each retains27 FP64-ideal logit failures from the reference precision gap;
overall numerical acceptance remains false. Native short raw-logit execution
is now measured on Mac, with generation/state, source-BF16, precision gaps,
Metal, quality/performance, phone fit and app integration still open. G3 is not
marked complete and no capability/catalog/position limit changes.


The [bounded packed generation-state wrapper](gemma-generation-state.md) now
connects the same model-bound raw session to the existing greedy generator.
Separate pending history, exact entire committed-prefix reuse, conservative
fork/shortening replay, deferred physical reset failures and final-row sampling
are checked offline. All302 CPU/324 Metal-feature tests, both iOS targets and
five release state methods pass; warmed state operations allocate nothing.
This does not qualify real E2B continuation or close G3. Independent continuation
witnesses, tokenizer/chat, precision/source, Metal, quality/performance, device
and app gates remain open, with no catalog/ABI/position changes.

The [independent four-step greedy reference](gemma4-greedy-reference.md) now
freezes raw selected IDs107,100,45518,107 before native continuation testing.
All24 complete selected-row arrays and12 unchanged cross-precision gates are
independently replayed; FP64 is teacher-forced on staged-FP32 history. Step0
matches the earlier four-token reference bitwise. This advances continuation
witnesses only: native generation is next, and all earlier full-graph precision,
source, tokenizer/chat, Metal, quality/performance, device/app and long-output
gates remain open. No G2/G3/G4 completion or capability/limit change.

The [real short raw continuation probe](gemma4-raw-generation.md) now checks
uninterrupted, split, cancellation/resume and explicit caller-stop/continue
schedules on Mac. All32 complete-logit gates pass over8,388,608 pairs and all
four schedules preserve bitwise-identical distributions, seven committed IDs
and one pending sampled token. Required Rust/default/Metal/iOS checks pass and
the measured process fits6GiB. This advances short raw generation/state only.
Tokenizer/basic-text formatting/model-stop witnesses are next; all earlier
full-graph precision/source failures and Metal, quality/performance, device/app
and long-output gates remain open. G3 is not marked complete.

The [pinned tokenizer/basic-text witness](gemma4-text-codec-reference.md) now
freezes97 exact records in two fresh byte-identical upstream captures, with
manual prompt-layout and scalar complete/prefix-decode replay. The strict
user/assistant subset, all24 control literals, no-Strip byte-fallback behavior
and selected-artifact stops1/106/50 are explicit. Seven offline methods and
separate1GiB reference-process checks pass. Native text-profile implementation
and formatted model continuation are next; no app capability, streaming, G3,
numerical/source, quality/performance, device or long-output gate is closed.

The [dedicated native basic-text codec](gemma4-native-codec.md) now matches all97
frozen records and285 prefix decodes, without adding an app Profile or numerical
capability. Same-buffer pinned loading, duplicate-safe rendering and preallocated
decode-byte admission pass offline and real-tokenizer checks. All307 CPU/329
Metal-feature tests, both iOS targets and the release codec checks pass; the
isolated text process fits its separate1GiB allowance. Formatted model execution
and append-only text ownership are next. Numerical/source gaps, full G3, Metal,
quality/performance, combined/mobile admission, app and long-output gates remain
open, with the32-position numerical limit unchanged.


The [formatted basic-hello mathematical reference](gemma4-basic-chat-reference.md)
now composes the frozen ten-token text prompt with the complete713-state graph.
Both precision paths select the same11 IDs, ending naturally at hidden pending
stop106; ten visible IDs decode to “Hi! How can I help you today? 😊”. All33
selected-row gates pass over5,784,064 pairs. Independent replay checks exact
prompt/selection policy, complete arrays, text and committed/pending/stop state;
the separate512MiB reference-process gate passes. This advances a frozen
formatted continuation witness. Native formatted execution, streaming, full G3,
earlier numerical/source gaps, Metal, quality/performance, combined/mobile fit,
app integration and131072-output qualification remain open. No limits change.


The [native formatted basic-hello diagnostic](gemma4-basic-chat-generation.md)
now composes the pinned codec, named scalar E2B model and generation owner with
combined admission. Uninterrupted and split2+remaining schedules produce the
same eleven samples and exact reply, naturally stopping at hidden pending106.
All22 logit rows are bitwise identical between schedules and all44 reference
gates pass over11,534,336 pairs. Required307 CPU/329 Metal-feature tests, both
iOS targets and seven release diagnostic methods pass. The whole Mac process
fits6GiB with3,855,781,740 combined admitted bytes. Append-only streaming is next;
broader text quality, full G3, earlier numerical/source gaps, resident Metal,
representative performance, mobile fit/app integration and131072-output gates
remain open. No app profile, ABI, catalog or numerical-limit change.


The [pinned no-Strip Gemma text stream](gemma4-native-stream.md) now reuses a
private owned decoder state behind a borrowed tokenizer facade. A separately
admitted 4 MiB stream owner preserves whole fallback runs, skipped specials,
terminal flushing and poisoned-state semantics. All 310 CPU/332 Metal-feature
tests, both iOS targets, 22 release text methods and four failure-retention
methods pass. Existing real profiles retain parity across 233,585 vocabulary IDs.
The actual Gemma probe matches 262,144 isolated IDs, 67 small cases/four partitions
and five synthetic long patterns/three partitions; all 7,675,150 raw bytes replay
against the scalar oracle. Text-process admission passes. This closes the
bounded text-stream mechanism only; full G3, quality/precision/source gaps,
Metal, representative performance, mobile fit/ABI/app integration and 131072
LLM-output qualification remain open, with no numerical-limit changes.


The [short native Gemma streamed reply](gemma4-stream-generation.md) now composes
the accepted scalar generation and bounded text owners in an explicit diagnostic.
Uninterrupted and split runs stream the same exact reply with hidden stop 106;
cancellation after two visible IDs flushes “Hi!” once and preserves 11 committed
plus one pending ID. All 24 full rows match bitwise across applicable schedules;
48 unchanged gates pass over 12,582,912 values. All 310 CPU/332 Metal-feature tests,
both iOS targets and 12 release methods pass. Combined admission is 3,859,976,044
bytes and the measured Mac process fits 6 GiB. This closes short diagnostic
composition only. Full G3, quality/precision/source gaps, resident Metal,
representative performance, qualified identity/ABI/app integration, mobile fit
and 131072-output gates remain open, with no capability or numerical-limit change.


[Loaded Gemma content identity](gemma4-content-identity.md) now retains hashes of
the exact admitted config/index and completely consumed container, and copies
that evidence into the named short model. Late failures publish no store;
tiny/manual constructors retain absent identity. All 312 CPU/334 Metal-feature
tests, both iOS targets and seven payload/15 model release methods pass. A fresh
Mac cold load checks all 540 views and actual store/model identity without forward
execution; its measured process fits 6 GiB. This is content binding only. Trusted
repository/tokenizer pairing, native capabilities/ABI/app integration and the
remaining G2/G3/G4, quality/performance, phone and long-output gates stay open.

## Consumed-byte identity reference

The [Smol range witness](smol-load-identity.md) provides maintained independent
offline producer/replay tools and exact frozen metadata for a later qualified
native loader. Fresh relocated captures/replays and 28 synthetic methods pass;
694 ranges and repeated tied-weight reads are checked without duplicating the
checkpoint. Loader publication, trusted profile/tokenizer binding and native
capability/app integration remain separate reviewed tasks. No inference or
eligibility change is granted by this reference.
