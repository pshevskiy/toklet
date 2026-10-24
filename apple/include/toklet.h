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
/* Ordinary TokletOptions budgets include this fixed logical owner allowance.
   Actual load/session admission subtracts it; no reservation or peak is implied. */
#define TOKLET_MODEL_SESSION_BRIDGE_BYTES 20480u
/* All records use native C alignment. All nonnull pointers must refer to live,
   initialized storage for their full stated extent. Inputs and outputs/error
   records must not overlap. Counts are element counts; null+zero is permitted
   for token/stop arrays and the documented history-size query. Handles come only from this library.
   Model/session operations and release stay on their creating dedicated thread.
   Callbacks run synchronously there, may not throw/unwind, and may not destroy
   active resources. Reentrant session calls return BUSY. Rust panic poisons a
   session: release/recreate it. OOM abort/foreign exceptions are not contained.
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
uint32_t toklet_model_release(TokletModel *, TokletError *);
uint32_t toklet_session_create(const TokletModel *, const TokletOptions *, TokletSession **, TokletError *);
uint32_t toklet_session_release(TokletSession *, TokletError *);
uint32_t toklet_session_state(const TokletSession *, TokletReport *, TokletError *);
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
/* Additive immutable text boundary, same creating-thread rule. Pinned SmolLM2
   Qwen2.5 and Mistral profiles with exact model config/tokenizer assets. Reserve
   >=64MiB for SmolLM2/Mistral or >=128MiB for Qwen separately from numerical admission.
   No model/session is retained by the returned chat handle. No network access. */
uint32_t toklet_chat_create(const TokletModel *, uint64_t allowance, TokletChat **, TokletError *);
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
