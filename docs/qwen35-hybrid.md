# Qwen3.5 scalar hybrid model and sessions

Q1.C is complete in its scoped scalar reference. It composes the existing packed scalar, convolution,
delta, norm and attention primitives into the Qwen3.5 text graph. The model can
bind the authenticated32-layer [CPU store](qwen35-4b-payload.md); ordinary model
catalog, codec, generation controller and Metal adoption remain separate work.

Names resolve once during model construction. Each repeated block has three
gated-delta/four-tap layers and one full-attention layer. Immutable matrices
remain Q4/group64/BF16 views; small FP32 source vectors are borrowed. The output
head uses the embedding owner. Full attention splits Q/gate within each head,
normalizes Q/K, rotates the declared prefix, applies grouped causal attention
and a sigmoid output gate. The small test's query width deliberately differs
from hidden width. No full FP32 model expansion or numerical-loop map lookup is
introduced.

The [official reference](https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
adds L2 epsilon to the sum of squares. Toklet uses that convention and the
existing grouped delta primitive. The converter-era
[MLX-LM0.31.1 graph](https://github.com/ml-explore/mlx-lm/blob/v0.31.1/mlx_lm/models/qwen3_5.py)
uses a mean-epsilon expression; it is not an equivalent numerical oracle.
The [newer pinned MLX helper](https://github.com/ml-explore/mlx-lm/blob/0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d/mlx_lm/models/gated_delta.py)
explicitly corrects that scaling. The
[full-attention definition](https://github.com/ml-explore/mlx-lm/blob/v0.31.1/mlx_lm/models/qwen3_next.py)
also identifies the per-head gate split. Already-sanitized source norms and
convolution weights receive no extra offset or transpose.

The scalar profile uses existing FP64 projection, RMSNorm, convolution and
gate construction with FP32 storage; recurrence and attention dot/context
reductions follow their ordered FP32 contracts. Softmax is FP64→FP32. Partial
rotary pairs within the rotated prefix, with FP64 frequencies/trig followed by
FP32 rotation arithmetic. This profile does not emulate native BF16 GPU math.

Sessions keep separate full-attention K/V `[kv_heads,capacity,head_dim]`, delta
`[value_heads,value_dim,key_dim]`, and raw convolution history `[3,channels]`.
Each recurrent/history owner has one committed and two alternating candidate
banks, permitting a whole chunk to fail without copying its committed state.
KV writes stay beyond the committed prefix. Only an infallible final ownership
swap publishes candidates, history and staged distributions. Failure or
cancellation preserves the prior state/distribution; retry overwrites stale
candidate/KV tails. Source weights stay immutable.

Admission includes borrowed source capacity, model descriptors, three state
banks, KV capacity, reusable scratch, two chunk-logit buffers, token history and
a64MiB metadata policy. Actual Vec capacities are checked against the original
limit, capped at6GiB. Raw numerical sessions are bounded to512tokens/chunks32;
this does not qualify declared262,144positions or actual context fidelity.
A report covers one model/session; other live sessions and caller buffers need
separate admission, counting shared physical backing once. `Memory` reports
allocated owner scope separately from process observations.
`Work` reports completed projections, gathers, convolutions, recurrence,
attention heads, completed/committed tokens and publication outcome. Wall-clock,
GPU, tokenizer and user-visible measurements require their later adapters.

The independent fixture was frozen before Rust composition: four layers,
grouped heads, four taps, partial rotary, non-square projection, tied head and
offsets0/4,079. Its ideal FP64 and explicit staged FP32 arrays contain1,225,024
values. Three chunk schedules compare3,675,072values under the unchanged
`1e-5 + 1e-4*abs(reference)` gate; maximum ratio0.153, no greedy mismatches.
Further partition/reset tests are bit-identical. Six focused Rust controls
cover107observer failure points, cancellation, stale/nonfinite KV tails,
admission/shape/position/ID/arithmetic rejection and warmed allocation-free
forward/reset/retry. Six Python controls check the independent producer.

Fresh630CPU/762Metal tests (37/299ignored), formatting and strict CPU/Metal
all-target Clippy pass. Fourteen selected Python controls and five damaged
capture controls pass. Exact3917build inputs remain unchanged; a later fixture
README is the only non-numerical inventory addition. The initial JSON-number
representation assertion, mixed Clippy-driver metadata failures and new-probe
expression typo are fixed and retained. Neither fixture bytes nor tolerances
changed.

Two fresh release processes bind the actual32-layer source, allocate a512/32
CPU session, reject admission-minus-one, and execute raw input ID1. Each records
1051arrays/16,067,328finite values; both64,269,312byte bodies share a hash.
Observed and plain replay have identical committed state/logits, lowest greedy
ID6503 and zero warmed Rust allocations. Admission is2,691,890,072B; measured
physical peaks are2,638,285,296B and2,638,318,064B on Apple M1 Pro/16GiB.
Completed work is249projections, one gather,24convolutions/recurrences and
128attention-head steps per input. The reader checks source endpoints, capture
hashes/shapes, embedding bits, four sampled tied-head rows and local first-input
cache/history invariants. It uses native intermediate operands for those local
checks; it is not an independent coupled-model oracle.

Load observations are8.243/8.114s; plain CPU forwards8.364/8.327s and observed
forwards9.095/8.930s include diagnostic I/O. Two other CPU reference jobs were
running. These raw observations have no controlled latency/overhead/sustained
qualification and no tokenizer, generation or user-visible TTFT scope.

Astra approves the combined diff, invariants and checks with no findings.
Real-source coupled fidelity, quality,
codec/stops, GPU execution, request summaries and performance qualification
remain open. iPhone runtime remains deferred. [Verification](../artifacts/qwen35-hybrid-2026-11-08/verification.json)
links full logs, raw samples and scope-specific controls.
[Review/commit](../artifacts/qwen35-hybrid-2026-11-08/publication-checks.json).

[Contract](../artifacts/qwen35-hybrid-2026-11-08/acceptance-contract.json),
[fixture freeze](../artifacts/qwen35-hybrid-2026-11-08/fixture-freeze.json),
[iteration](../artifacts/qwen35-hybrid-2026-11-08/iteration-notes.json).
