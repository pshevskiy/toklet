# Model-driven input, output and context support

**Requirement corrected by the human on 26 October 2026:** Chat and Expert should
support the input/output context window of the selected model. This replaces
the earlier fixed requirement for 131072 output tokens in one reply. The file
path is retained for existing links; previous fixtures and receipts retain their
original scope. See the [Astra-assisted plan review](m8-plan-review-2026-10-26.md)
and [active engine plan](engine-expansion-plan.md).

The target is each admitted model's full supported window. A smaller implemented,
qualified or device-admitted window is partial support, with the actual bound
and reason visible. Do not silently substitute a smaller limit and call the
full requirement complete. Do not offer the declared maximum as runnable before execution,
text ownership and memory are qualified. It can still be shown as model metadata.

## Capacity contract

The engine reports separate facts for the exact model/tokenizer revision and
execution profile: declared total positions and any distinct input/output
constraints; implemented and numerically qualified bounds; device-admitted
capacity for this session; rendered input tokens, requested output and remaining
capacity. Unknown or conflicting source limits need explicit adapter decisions.
Generation defaults are defaults, not automatically hard model limits.

For ordinary shared-window causal models, conservatively admit:

`rendered_prompt_tokens + requested_max_new_tokens <= usable_total_positions`

Rendered input includes system text, chat template/control tokens, retained
history and the actual user input. Apply a distinct documented output/input cap
where the model has one. Raw Expert still uses the same numerical position and
memory contract; Chat adds its own rendering costs. Exact-prefix reuse, fully
cached prompts and pending sampled tokens must not double-count history or
incorrectly create free capacity. Specify hidden stop and final pending-token
semantics. Reject overflow or over-budget requests before state mutation; any
history truncation requires a visible policy rather than silent loss of input.

For sliding/hybrid/recurrent graphs, distinguish absolute positions from
retained KV/recurrent state and input/output token counts. Sliding retention
does not extend a model's declared positions or erase its full-attention layers.
Any context extension or KV precision change needs its own numerical and quality
qualification. A supported maximum permits generation up to that limit; normal
EOS/cancellation still ends a response earlier.

## Current state and next tasks

The installed baseline still uses 256 Chat output, 128 raw Expert output, 512 app
context positions and 64 KiB conversation/decoded text. General engine 32768 and
selected native MLX 512 ceilings are separate qualified scopes. Gemma's current
Mac scalar diagnostic has 32 positions and no app capability. These are current
limits to remove through qualification, not the corrected product target.

The [Gemma quality-input contract](gemma-quality-contract.md) now freezes its
complete 1025-position loss/task demand, independently replayed through source
and Rust tokenizers. That is a next numerical-capacity dependency, not a new app
limit or proof of the model's full window. Context and chunk scratch must be
separated before expanding the current diagnostic; preserve complete requests.

The [native host chunk-scratch prerequisite](gemma-chunk-scratch.md) now sizes
activations, producer proposals and private/published logits by per-call rows;
attention scratch scales with chunk rows times current context. Tiny32-prefix
and synthetic GPU admission checks pass. The
[session-capacity mechanism](gemma-session-capacity.md) now admits actual
context/cache/history/routes at C1–32 independently of chunk rows. Its tiny
numerical/state, two-head logical/segmented KV and exact-budget checks pass.
Default/real E2B/hybrid/GPU ceilings remain32, and real whole32 remains
unqualified; this does not close L3–L6 or real full-window support.
The [pre-target tiny1025 full-sequence oracle](gemma-long-witness.md) now
freezes independent long expectations for subsequent cached/native checks. Its
W4 graph and selected boundaries do not raise native capacity or qualify real
E2B window512. The separately qualified [FP64 matrix-dot reference attention](gemma-reference-attention.md)
passes original numerical gates and two reference-operation timing runs.
[L3 app integration](context-app-integration.md) now passes scoped admission/state checks;
next L4 qualifies real Qwen3 C512 numerics. The [explicit scalar witness context](gemma-scalar-long-context.md)
now passes C<=1025 with bounded32-row calls, frozen numerical/argmax/state
checks and structural two-head W512 wraps. Default/real E2B/hybrid/GPU32 and
real whole32/long BF16/1025 quality remain separate.

L3.1 [engine context facts and shared request preflight](context-admission.md) now
passes Rust/native/C and tiny Metal checks. Same-document declared positions,
global loader/loaded/session bounds, process-local model/session identity and
pending-history arithmetic remain separate from qualification/device fit.
L3.2 [app context integration](context-app-integration.md) now consumes these fixed
records in Chat and Expert, with explicit codec/text/event policy, fresh instance
checks, rendered shared-window preflight and draft/prefix-preserving rejection.
Strict Mac/iOS source checks, both app builds and Mac real admission/lifecycle
regressions pass. The385+127 versus128 workload cancels after one token; no new
512-position numerical qualification follows. Current ranges stay fixed. The [independent Qwen3 C512 reference](qwen3-context512-reference.md)
now has two byte-identical complete cold captures and two readbacks. Old17,
causal/mask/argmax checks pass, but7282 scalar comparisons exceed unchanged
FP32/FP64 short-context gates; numerical context qualification remains false.
Next localize failing coordinates/positions/margins and disjoint target-acceptance
intervals before any precision-policy decision or native context upgrade.
Real Gemma whole32/long BF16/full1025 precision/quality/fit remain required.

| Task | Mechanism and acceptance | Status |
| --- | --- | --- |
| L1 | Independent decoding/terminal witnesses for existing tokenizers, including incomplete UTF-8, controls, whitespace and fallback rewrites. | [Frozen witnesses](chat-stream-oracle.md); historical synthetic 131072-ID cases remain decoder evidence only. Gemma has separate [stream composition](gemma4-stream-generation.md). |
| L2 | Bounded incremental decoding/event delivery: token, byte, carry and batch bounds, terminal flush, backpressure/failure state, no per-token full-prefix decode or full-output clone. | [Rust decoder checked](chat-stream-decoder.md); whole native/app ownership and every model profile remain separate. |
| L3 — next capacity task | Engine-owned model/runtime/device capability schema and checked request admission, exposed through native bridge to both app modes. Resolve exact config/position rules, raw versus rendered input, sampled/committed history, stops and replay. | [Engine facts/preflight](context-admission.md) and [both app modes](context-app-integration.md) checked for current profiles and admission/state scope. Qualification/device-fit statuses remain explicitly unevaluated; fuller context/profile coverage remains open. |
| L4 | Qualify each model's declared window progressively: short baseline, useful context, window/chunk/rotary/cache boundaries, near-maximum and exact maximum, plus rejected maximum+1. Use bounded chunked prefill and correct absolute offsets; test long input and long continuation independently and together. | Open; integrate Gemma through G4, without requiring an unrelated new long-context architecture first. |
| L5 | Chat/Expert controls and bounded text/history delivery driven by L3. Display model maximum, usable device limit, remaining tokens and limiting reason. Test model switching, stale events/settings, early stop, cancel/resume and offline restore. | [Initial facts/policy UI](context-app-integration.md) checked at current bounds. Larger bounded text/codec/event delivery and model-driven full-window controls remain open. |
| L6 | Real checkpoint/context quality, completed prefill/decode/visible timing, weights/KV/state/scratch/text/UI/conversion memory and lifecycle on Mac, then independently iPhone 14 Pro. | Open per checkpoint, execution profile and device; a smaller usable window remains partial qualification. |

Capacity arithmetic tests and synthetic text IDs do not prove full-context
numerical execution. At the admitted boundary, preserve complete raw evidence
and compare independent numerical/state witnesses; use held-out long-context
quality tasks as well as short loss/task gates. Run a bounded output-limit
witness without treating forced no-EOS generation as ordinary model behavior.
Freeze workloads and expected results before target trials.

Working future text proposals of 8 MiB per reply/conversation JSON and 64 MiB
persisted UI history are not universal model limits or qualified budgets. Derive
and qualify bounded token/byte/storage policies for each supported tokenizer;
if a byte/UI limit prevents the full model window, show that restriction and
keep full-window support open. Count conversion overlap and copying/backpressure.

## Model selection after the requirement correction

The [pinned Gemma E2B config](https://huggingface.co/google/gemma-4-E2B-it/raw/3e22461f65e89153144f8adb70e3b8c2cc9845a7/config.json)
declares 131072 total positions. It is a valid declared target for the corrected
shared-window requirement; it never implies 131072 input plus another 131072
output. Its present short diagnostic does not qualify that window.

[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/eb706f593d2d43c90a10271199c10b07ced7569a/config.json)
is an optional 262144-position hybrid candidate. The
[mechanism investigation](mlx-qwen35-design.md) remains useful for later E4 work;
it is no longer needed to satisfy an artificial fixed-output target before
Gemma delivery. New checkpoint selection still requires immutable artifact,
tokenizer, numerical and resource admission. There is no reliable full-window
or phone ETA before actual fit, boundary checks and throughput measurements.
