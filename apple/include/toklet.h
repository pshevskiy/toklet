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
typedef struct { uint32_t code; uint8_t message[256]; } TokletError;
typedef struct {
    uint32_t abi_version, struct_size, capacity, chunk;
    uint64_t memory_limit_bytes;
} TokletOptions;
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
/* Explicit local native MLX Qwen: MlxAffine/Compensated/Online/FP16 KV/FullLogits.
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
#ifdef __cplusplus
}
#endif
#endif
