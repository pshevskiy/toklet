# Bounded packed Gemma generation state

`Model::new_generation_session` now connects the existing short packed graph to
Toklet's statically dispatched `InferenceSession` and greedy generator. Numerical
kernels and `generation::generate` are unchanged. This task validates the state
mechanism on offline fixtures; real E2B continuation/tokenizer/chat qualification
remains separate.

## Histories, reuse and replay

The wrapper owns a separate fixed32-token input/sampled history and borrows the
same immutable model as its raw session. Appending a sampled token changes only
that history. It becomes committed KV only after a complete successful chunk.
Pending tokens hide the distribution. A nonempty, fully evaluated history
exposes only the final vocabulary row from the raw session's last published
chunk; it does not return all chunk rows to the sampler.

A replacement prompt may reuse the entire committed prefix only when all IDs
match and replay is not already required. Pending sampled/input tokens never
count as committed cache. Changed or shortened prompts conservatively replay
from position0. This is exact committed-prefix reuse, without a longest-prefix
truncation or ring-recovery optimization.

Reset and fork invalidation hide logits and declare zero evaluated tokens
immediately. Physical cache reset is deferred until the next nonempty numerical
chunk because the shared generator's reset boundary is infallible. A physical
reset error leaves the wrapper explicitly replay-required with pending history
intact; stale backing cannot become valid merely because a later prompt matches
its old IDs. After a successful reset, a failed forward leaves the new empty
committed prefix valid and pending tokens available for retry. Later forward
failures preserve the previous committed prefix and public raw logits. Discard
removes pending tokens; it exposes the prior valid distribution only when no
replay is required.

Stop IDs remain pending sampled tokens and are not emitted. Output-limit,
cancellation and callback-failure behavior comes from the existing generator.
A continued request evaluates a pending sampled token exactly once. A request
already cancelled at entry, or rejected during validation, preserves the current
history and raw cache state. Later cancellation preserves valid progress already
made and any pending sampled token.

## Admission and allocation

Options admit capacities1..32, chunk sizes1..capacity and position offset0 only.
Every append/replacement is fully validated before mutation. A smaller requested
capacity/chunk keeps the raw32-position storage; it does not imply a smaller
physical allocation. Logical model-plus-session admission remains within the
initial6GiB Mac ceiling and is checked before numerical allocation and again
against actual capacities.

The accounting invariant is:

`admitted = model.model_bytes + raw.allocated_bytes + sizeof(wrapper) - sizeof(raw)`.

Categories include actual cache/staging numerical capacities, scratch excluding
the two full `[32,vocabulary]` logit buffers, both inline32-token histories,
model/session metadata and the same store's retained allowance exactly once.
Two new cold model accounting fields add16 bytes of model metadata, included
through `sizeof(Model)`. Numerical tensor contents and kernel arithmetic do not
change. The tiny constructor keeps its existing semantics; historical receipts
retain their original struct-size accounting.

## Checks and limits

Five new methods cover admission and category-sum/exact-budget checks for tiny
and synthetic store-accounting variants; pending/exact-prefix/fork replay and
frozen final rows; actual reset-generation exhaustion and late-forward retry;
stop/cancellation/callback/continuation behavior; and warmed allocation-free
operations. Tests also exercise a pre-cancelled Reset and a rejected generation
request while a sampled token is pending. Ninety-six warmed decode steps and
three fork replays/resets/discards allocate no Rust heap storage.

The first preliminary compile rejected private cross-module test helpers before
any tests ran. That attempt remains failed. The narrow fix changes test-only
visibility; frozen helper bodies and numerical fixtures remain unchanged.
A test-only cache setter injects actual generation exhaustion; it is absent
from production builds. The corrected targeted suite passes.

Full verification is recorded in the
[receipt](../benchmarks/results/2026-10-25-gemma-generation-state-v1.json).
The mechanism does not qualify real continuation outputs or close G3. The
[prior short E2B execution](gemma4-short-scalar.md) retains27 FP64-ideal logit
failures per schedule. Original-BF16/source, earlier GPU/long-rotary failures,
quality/performance, resident Metal, phone fit, tokenizer/chat, app integration
and131072 output tokens remain open. No family, ABI, catalog or position limit
changes here.
