#ifndef TOKLET_H
#define TOKLET_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define TOKLET_ABI_VERSION 1u
#define TOKLET_OK 0u
#define TOKLET_INVALID 1u
#define TOKLET_LOAD_ERROR 2u
#define TOKLET_ENGINE_ERROR 3u
#define TOKLET_BUSY 4u
#define TOKLET_WRONG_THREAD 5u
#define TOKLET_PANIC 6u
#define TOKLET_POISONED 7u
#define TOKLET_CALLBACK_ERROR 8u
#define TOKLET_UNSUPPORTED 9u
#define TOKLET_CANCELLED 10u
/* Ordinary TokletOptions budgets include this fixed logical owner allowance.
   Actual load/session admission subtracts it; no reservation or peak is implied. */
#define TOKLET_MODEL_SESSION_BRIDGE_BYTES 20480u
/* All records use native C alignment. All nonnull pointers must refer to live,
   initialized storage for their full stated extent. Inputs and outputs/error
   records must not overlap. Counts are element counts; null+zero is permitted
   for token/stop arrays and the documented history-size query. Handles come only from this library.
   Model/session/chat operations and release stay on their creating dedicated thread.
   Sessions and audited chats retain the exact model allocation after external
   model release. Keep model memory admitted until those owners end; ordinary
   legacy chat handles retain no model.
   Callbacks run synchronously there, may not throw/unwind, and may not destroy
   active resources. Reentrant session calls return BUSY. Rust panic poisons a
   session: release/recreate it. Empty-candidate preparation only reads current
   and does not poison it on candidate panic. OOM abort/foreign exceptions are not contained.
   Cancellation alone may run concurrently; release its flag only after every
   user returns. NULL error suppresses messages. Release NULL is a no-op. */
typedef struct TokletModel TokletModel;
typedef struct TokletSession TokletSession;
typedef struct TokletCancel TokletCancel;
typedef struct TokletChat TokletChat;
typedef struct TokletCompatibility TokletCompatibility;
typedef struct { uint32_t code; uint8_t message[256]; } TokletError;
typedef struct {
    uint32_t abi_version, struct_size, capacity, chunk;
    uint64_t memory_limit_bytes;
} TokletOptions;
/* Cold metadata inspection; numerical budget is reported exactly, without the
   model/session bridge deduction. Backend: 0 scalar reference, 1 Metal.
   Context/memory outside engine bounds produce an invalid_request JSON report. */
typedef struct {
    uint32_t abi_version, struct_size, backend, context_limit;
    uint64_t memory_limit_bytes;
} TokletCompatibilityOptions;
/* Metadata-only native MLX position policy; no requested capacity or budget.
   load_profile: 0 ordinary native MLX; 1 frozen audited Smol metadata.
   These select existing distinct load entrypoints, not additional architectures.
   reserved must be zero. Native C alignment/16 bytes. */
typedef struct {
    uint32_t abi_version, struct_size, load_profile, reserved;
} TokletNativeLimitsOptions;
/* Explicit runtime planning, native C alignment/56 bytes. Normalization:
   0 Original, 1 Compensated; readback: 0 FullLogits, 1 GreedyToken;
   attention: 0 Reference, 1 Online, 2 native model factory policy.
   Value2 resolves after graph admission and reports actual Reference/Online.
   Normalization/readback remain explicit. Projection is always MlxAffine.
   reserved must be zero. Budgets are exact, without bridge deduction;
   load budget must be positive, session budget may be zero (fit=false).
   capacity+offset must fit context_limit; chunk is1..min(1024,capacity). */
typedef struct {
    uint32_t abi_version, struct_size, context_limit, capacity, chunk;
    uint32_t position_offset, normalization, readback, attention, reserved;
    uint64_t load_memory_limit_bytes, session_memory_limit_bytes;
} TokletRuntimePlanningOptions;
/* Same-model old-plus-candidate accounting, native alignment/64 bytes.
   expected IDs must match the live model and current session; reserved=0.
   Aggregate budget includes both sessions and reserves, may be zero, <=6GiB.
   Replay/text/application reserves are caller claims, not measured ownership.
   Existing candidate TokletOptions and its individual whole budget are unchanged. */
typedef struct {
    uint32_t abi_version, struct_size;
    uint64_t expected_model_instance, expected_session_instance;
    uint64_t transition_budget_bytes, replay_reserve_bytes;
    uint64_t text_reserve_bytes, application_reserve_bytes, reserved;
} TokletTransitionPlanningOptions;
/* Idle model/session must retain the exact same immutable model allocation.
   No source files, candidate, numerical work, mutation or reservation. A valid
   out is cleared on error. Uses existing inspection JSON copy/release functions.
   Report individual candidate fit separately from aggregate old+candidate fit.
   Metadata/driver and bridge allowances for both sessions remain conservative;
   actual candidate capacities must be rechecked before any future publication.
   Fit=false is an OK report. Wrong IDs/Rc/options: INVALID; unsupported retained
   diagnostic/profile/mode owners: UNSUPPORTED. Busy/poisoned/wrong-thread rules
   are unchanged. This snapshot cannot authorize later creation or publication.
   No context, text, whole-app/device or performance qualification is evaluated. */
uint32_t toklet_session_transition_plan_create(const TokletModel *, const TokletSession *,
    const TokletOptions *, const TokletTransitionPlanningOptions *, TokletCompatibility **, TokletError *);
/* Prepare an explicitly EMPTY same-model session; does not adopt/release/change
   current, replay history or reopen weights. Capacity may be below old history.
   Uses the same transition request and rechecks actual candidate allocation
   against both individual and aggregate budgets. Caller reserves are unverified.
   Required cancellation flag is sampled before construction and before return,
   never cleared. Synchronous construction is not interruptible. Cancellation
   after the final check can race with success; callers must still reject stale
   results. CANCELLED=10 returns NULL; all other failures also clear out.
   Candidate-only Rust unwinding returns PANIC without poisoning current; OOM
   abort/foreign exceptions remain uncontained. Failed attempts may consume IDs.
   Success transfers an ordinary new session handle, with empty history/KV
   validity and shared model identity; release it normally if not adopted. */
uint32_t toklet_session_prepare_empty(const TokletModel *, const TokletSession *,
    const TokletOptions *, const TokletTransitionPlanningOptions *,
    const TokletCancel *, TokletSession **, TokletError *);

#define TOKLET_MAX_CHAT_OUTPUT_TOKENS 256u
#define TOKLET_MAX_COMPATIBILITY_JSON_BYTES 65536u
/* Literal for the Swift importer; checked against 2*maximum+256 by C smoke. */
#define TOKLET_COMPATIBILITY_JSON_ALLOWANCE_BYTES 131328u
/* Reserve this separate logical JSON allowance (two maximum buffers plus the
   handle). Allocator overhead is excluded; temporary
   metadata uses the existing bounded artifact parser. No weights are expanded,
   GPU created, session admitted or text/qualification receipt evaluated.
   Additive ABI v1 entrypoints. Handles follow the creating-thread rule.
   Unsupported/unknown models return OK with their machine-readable JSON state.
   Invalid backend/options/path arguments fail; a valid out is cleared on error.
   Captured metadata is historical: later loading must revalidate files, budgets
   and runtime requirements. This handle never authorizes inference. */
uint32_t toklet_mlx_inspect_create(const uint8_t *, size_t,
    const TokletCompatibilityOptions *, TokletCompatibility **, TokletError *);
/* Additive metadata-only native MLX context limits, no GPU/model/session.
   Returns native_mlx_context_limits schema v1 through the existing JSON handle.
   Reports validated declared total, global ceiling, optional graph cap and
   nullable ordinary factory upper; the upper is not a budget-fitting capacity,
   loaded-instance fact, artifact identity or qualified context. Unsupported or
   unknown source metadata returns OK with upper:null. Unknown profile/reserved
   bits, bad ABI/path/pointers return INVALID and clear a valid out.
   Smol profile authenticates frozen metadata; complete consumed payload evidence
   remains required at toklet_model_load_mlx_smol_audited. Ordinary inspection
   does not authenticate that identity. Both routes require payload, runtime and
   session revalidation; budget/qualification/text/whole-app fit are unevaluated.
   Explicit Qwen3 diagnostics, specialized Gemma/LFM and scalar-reference paths
   are outside this report. No numerical arrays are allocated; redundant tied
   heads may be streamed by the existing bounded source inspector.
   Uses the same JSON allowance and creating-thread copy/release rules. */
uint32_t toklet_mlx_context_limits_create(const uint8_t *, size_t,
    const TokletNativeLimitsOptions *, TokletCompatibility **, TokletError *);
/* Additive ABI v1 runtime inspection: initializes Metal/MPS/precise pipelines.
   Returns the separate runtime schema v1 (session_memory and fit flags), not
   metadata status/reason. No numerical weight/KV/scratch/logit allocation.
   The same bounded JSON allowance, copy/release and creating-thread rules apply.
   Budget-fit failures return OK reports. Invalid arguments/options: INVALID;
   unavailable/unqualified backend/mode: UNSUPPORTED; rejected artifact: LOAD_ERROR;
   other runtime failures: ENGINE_ERROR. A valid out is cleared on error.
   Driver memory uses the existing allowance estimate; fit flags reserve nothing.
   Captured plans remain historical and require revalidation before loading. */
uint32_t toklet_mlx_plan_create(const uint8_t *, size_t,
    const TokletRuntimePlanningOptions *, TokletCompatibility **, TokletError *);
/* Additive ABI v1 numerical capacity inspection. Same options, thread, error,
   bounded JSON allowance and Metal/MPS initialization as toklet_mlx_plan_create.
   Returns {"resources":<unchanged runtime schema>,"capacity":<range report>}.
   capacity.minimum is requested chunk; maximum is requested capacity. Selected
   is the largest fitting point, or null; first_excluded is selected+1 (or the
   minimum if no point fits), or null when maximum fits. Modes and offset stay
   fixed. Every point reports budget and per-buffer fit. A fitting point does NOT
   override false source/resident load flags in resources. Neither OK nor a
   selected point reserves memory, widens factory eligibility, discovers declared
   model context, or grants context/text/whole-process/device qualification.
   Current allocations and all limits are revalidated at actual construction. */
uint32_t toklet_mlx_capacity_plan_create(const uint8_t *, size_t,
    const TokletRuntimePlanningOptions *, TokletCompatibility **, TokletError *);
/* UTF-8 bytes, no NUL, <=65536 bytes. NULL+0 queries required byte count. Short
   buffers return INVALID, set required count and leave output untouched. Other
   failures clear a valid count. Capacity >65536 rejects before copying.
   count/output/error/handle must be disjoint; repeated reads return identical
   bytes even if source files change. Release once on creating thread. */
uint32_t toklet_mlx_inspect_json(const TokletCompatibility *, uint8_t *, size_t, size_t *, TokletError *);
uint32_t toklet_mlx_inspect_release(TokletCompatibility *, TokletError *);
typedef struct { uint32_t token_id, index; double elapsed_seconds; } TokletTokenEvent;
typedef struct {
    uint32_t abi_version, struct_size, reason, stop_token;
    uint32_t history_tokens, evaluated_tokens, pending_tokens, emitted_tokens;
    uint64_t admitted_bytes;
} TokletReport;
/* Read-only scalar diagnostic; no full-logit copy. Native C alignment,48 bytes.
   FP32 target/greedy logits are exactly widened to double; loss uses stable
   compensated FP64 log-sum-exp. Tied greedy IDs choose the lowest token ID. */
typedef struct {
    uint32_t abi_version, struct_size, target_token, greedy_token;
    uint32_t vocab_size, evaluated_tokens;
    double target_logit, greedy_logit, negative_log_likelihood;
} TokletScore;
/* Additive scalar execution facts. Instance IDs are process-local object
   continuity, never artifact hashes or persistent identity. Unknown declared
   limits are zero with their known bit clear; defaults do not populate them.
   Numerical session admission is not context qualification or whole-app fit.
   engine_global_position_ceiling is a global loader bound, not this backend's
   usable graph/window limit; loaded_position_limit/session_capacity remain binding. */
#define TOKLET_CONTEXT_DECLARED_TOTAL_KNOWN 1u
#define TOKLET_CONTEXT_DECLARED_INPUT_KNOWN 2u
#define TOKLET_CONTEXT_DECLARED_OUTPUT_KNOWN 4u
#define TOKLET_CONTEXT_NOT_EVALUATED 0u
#define TOKLET_CONTEXT_CPU 0u
#define TOKLET_CONTEXT_METAL 1u
typedef struct {
    uint32_t abi_version, struct_size, known_fields, backend;
    uint32_t context_qualification, whole_app_device_fit, reserved[2];
    uint64_t model_instance, session_instance;
    uint64_t declared_total_positions, declared_input_tokens, declared_output_tokens;
    uint64_t engine_global_position_ceiling, loaded_position_limit, session_capacity;
    uint64_t absolute_position_offset, history_tokens, committed_tokens, pending_tokens;
    uint64_t numerical_and_bridge_admitted_bytes;
} TokletContextInfo;
typedef struct {
    TokletContextInfo context;
    uint32_t operation, reserved;
    uint64_t base_history_tokens, input_tokens, history_after_input_tokens;
    uint64_t requested_output_tokens, required_capacity, remaining_tokens;
} TokletRequestAdmission;
/* Return nonzero for output failure; sampled ID remains pending in history. */
typedef uint32_t (*TokletEmit)(void *, TokletTokenEvent);
uint32_t toklet_abi_version(void);
uint32_t toklet_model_load(const uint8_t *, size_t, const TokletOptions *, TokletModel **, TokletError *);
/* Explicit Metal: compensated Half when packed_len==0, otherwise Q4 v2.
   Sessions select Online/FP16 KV/FullLogits. Unavailable builds fail without fallback. */
uint32_t toklet_model_load_metal(const uint8_t *, size_t, const uint8_t *, size_t,
    const TokletOptions *, TokletModel **, TokletError *);
/* Explicit local supported native MLX dense model: MlxAffine/Compensated/FP16 KV/FullLogits.
   Qwen2 uses Online attention; explicitly admitted Qwen3/Llama/Mistral graphs use Reference attention.
   Original packed tensors remain resident. Unsupported builds fail without fallback.
   Additive ABI v1 entrypoint; existing layouts and load functions are unchanged. */
uint32_t toklet_model_load_mlx(const uint8_t *, size_t, const TokletOptions *, TokletModel **, TokletError *);
/* Explicit audited import of the exact frozen Smol checkpoint. Authenticates
   consumed source content and publishes evidence only after resident assembly.
   Unsupported artifacts/builds reject; no text or app capability is granted.
   Existing options, output-null-on-error and creating-thread rules apply. */
uint32_t toklet_model_load_mlx_smol_audited(const uint8_t *, size_t, const TokletOptions *, TokletModel **, TokletError *);
/* Exact pinned Qwen3 audited source import, attached only after assembly.
   Ordinary geometry/context limits remain unchanged; no text/app grant. */
uint32_t toklet_model_load_mlx_qwen3_audited(const uint8_t *, size_t, const TokletOptions *, TokletModel **, TokletError *);
uint32_t toklet_model_release(TokletModel *, TokletError *);
uint32_t toklet_session_create(const TokletModel *, const TokletOptions *, TokletSession **, TokletError *);
uint32_t toklet_session_release(TokletSession *, TokletError *);
uint32_t toklet_session_state(const TokletSession *, TokletReport *, TokletError *);
/* Read-only live model/session snapshot, valid after external model release.
   No directory reads, new execution, reservation or app eligibility grant.
   Outputs are complete native-aligned records, untouched on failure. Existing
   thread/busy/poison/disjoint-output rules apply. Qualification/whole-app fit
   remain NOT_EVALUATED. All current native factories use absolute offset zero. */
uint32_t toklet_session_context(const TokletSession *, TokletContextInfo *, TokletError *);
/* Same operation/token/stop semantics as generate, including zero-output
   prefill. Prompt/Reset use base zero; Append/Continue count sampled history,
   including pending IDs. Arrays <=32768 IDs. This snapshot reserves nothing;
   actual generate revalidates and execution can still fail. remaining_tokens is
   session_capacity-required_capacity AFTER requested continuation. Output is
   untouched on every error. */
uint32_t toklet_session_preflight(const TokletSession *, uint32_t,
    const uint32_t *, size_t, uint32_t, const uint32_t *, size_t,
    TokletRequestAdmission *, TokletError *);
/* Scores only the CURRENT FINAL committed row; earlier prefill rows are unavailable.
   Full-corpus callers score each exact prefix BEFORE appending its target, and
   verify evaluated/history counts after any cancelled request.
   Requires an idle fully committed nonempty distribution. generate with maximum
   zero evaluates prompt/append without sampling. Pending/reset/invalid states
   reject; no prefix, cancellation flag or engine state is changed by this query.
   Target must be in vocabulary. Output is untouched unless TOKLET_OK; the full
   output record must be writable and disjoint from handle/error storage. */
uint32_t toklet_session_score(const TokletSession *, uint32_t, TokletScore *, TokletError *);
/* Exact sampled history, including pending IDs. NULL+0 queries count.
   Small buffers return INVALID with required count and no partial copy.
   Caller owns capacity uint32_t elements; count/output/error do not overlap. */
uint32_t toklet_session_history(const TokletSession *, uint32_t *, size_t, size_t *, TokletError *);
uint32_t toklet_session_reset(const TokletSession *, TokletError *);
/* operation: 0 prompt, 1 append, 2 continue (no input), 3 reset+prompt.
   report reason: 0 query/error, 1 output limit, 2 hidden stop, 3 cancelled.
   Stop IDs are explicit (no implicit EOS); raw-ID boundary owns no tokenizer.
   Error reports may contain recoverable state; other outputs require OK. */
uint32_t toklet_session_generate(const TokletSession *, uint32_t, const uint32_t *, size_t,
    uint32_t, const uint32_t *, size_t, const TokletCancel *, TokletEmit, void *, TokletReport *, TokletError *);
uint32_t toklet_cancel_create(TokletCancel **, TokletError *);
uint32_t toklet_cancel_set(const TokletCancel *, uint32_t, TokletError *);
uint32_t toklet_cancel_release(TokletCancel *, TokletError *);
/* Additive immutable text boundary, same creating-thread rule. Pinned SmolLM2,
   Qwen2.5, non-thinking Qwen3 and Mistral with exact config/tokenizer assets.
   Reserve >=64MiB for SmolLM2/Mistral, >=128MiB for Qwen2.5, or >=256MiB
   for Qwen3 separately from numerical admission.
   No model/session is retained by the returned chat handle. No network access. */
uint32_t toklet_chat_create(const TokletModel *, uint64_t allowance, TokletChat **, TokletError *);
/* Bind exact consumed Smol text assets to the supplied live audited model.
   CPU/legacy/unqualified models reject before text-asset access. Reserve an
   independent >=64MiB (<=6GiB) allowance BEFORE loading the model. This handle
   retains the same model allocation until chat release; shared weights count
   once and remain admitted even after external model release. Individual
   allowance maxima do not qualify combined process/device fit. No app grant.
   Existing encode/decode/stop limits and pointer/thread rules apply. */
uint32_t toklet_chat_create_smol_audited(const TokletModel *, uint64_t allowance, TokletChat **, TokletError *);
/* Bind verified Qwen3 codec assets to the exact live audited model allocation.
   Separately reserve >=256MiB (<=6GiB) text memory before model loading. Retains
   the same model until chat release; combined process/device fit is separate.
   tokenizer_config is verified policy data; special_tokens_map is ancillary.
   Raw token APIs do not prevent IDs from being passed to another session. */
uint32_t toklet_chat_create_qwen3_audited(const TokletModel *, uint64_t allowance, TokletChat **, TokletError *);
/* Read-only historical identity of the exact audited model/chat allocation pair.
   Same owner thread; model and chat must retain the same Rc and source evidence.
   Ordinary/source-only/unpaired handles: UNSUPPORTED; different retained owners:
   INVALID. No path, file read, session borrow, GPU work or allocation reservation.
   out is cleared before validation; publication is atomic on success. Uses the
   existing bounded inspection JSON copy/release functions and their allowance.
   Snapshot survives handle release; instance IDs correlate only this process.
   No preferences, current-disk identity, quality or device fit is authorized. */
uint32_t toklet_model_chat_identity_create(const TokletModel *, const TokletChat *,
                                         TokletCompatibility **, TokletError *);
/* JSON array of alternating {role,content} messages, starting/ending user.
   Max63 messages/64KiB UTF-8, no template control tokens; output max512 IDs.
   NULL+0 queries required count. Small output returns INVALID, sets required
   count, and leaves elements untouched. Outputs/count/inputs must be disjoint. */
uint32_t toklet_chat_encode(const TokletChat *, const uint8_t *, size_t, uint32_t *, size_t, size_t *, TokletError *);
/* Entire visible prefix, <=TOKLET_MAX_CHAT_OUTPUT_TOKENS IDs. Replaces displayed text, never concatenate
   per-token decodes. Special IDs hidden. UTF-8 byte output, no NUL, <=64KiB.
   Query and small-buffer semantics as above. A later prefix repairs split UTF-8. */
uint32_t toklet_chat_decode(const TokletChat *, const uint32_t *, size_t, uint8_t *, size_t, size_t *, TokletError *);
/* Profile stop IDs; use for generation. Capacity <=8. Same query, disjoint
   output/count and creating-thread rules as the other text queries. */
uint32_t toklet_chat_stop_ids(const TokletChat *, uint32_t *, size_t, size_t *, TokletError *);
uint32_t toklet_chat_release(TokletChat *, TokletError *);
#ifdef __cplusplus
}
#endif
#endif
