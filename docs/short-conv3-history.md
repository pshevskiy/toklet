# Transactional scalar short-convolution history

B4b adds `model::short_conv3::History`, a concrete owner for the two preceding
raw input rows of a three-tap causal depthwise convolution. It reuses the existing
scalar kernel unchanged. This is standalone FP32 state, not LFM2 model admission
or a transaction spanning attention KV, logits and token history.

Model-owned coefficients are `[channels,3]`, ordered oldest to current.
Session history is `[2,channels]`, contiguous across channels. Input is already
`u=B*x`; the leaf returns raw convolution `z`. The caller later forms `C*z` and
projects it. No SiLU, bias, padding or mask is applied. Every supplied row is
active. The existing leaf multiplies and sums in increasing tap order with FP64
arithmetic and rounds once to FP32 output; this is distinct from a future Half
profile.

`prepare_chunk` validates the entire chunk, coefficients, extents and admission
before writing scratch. It alternates caller-owned candidate and work banks,
returning an opaque immutable borrow of the final candidate. Committed history
is unchanged. The work bank can immediately serve another owner while the first
proposal remains alive. Dropping a proposal aborts. Numerical overflow may dirty
scratch/output, which must be discarded, but returns no proposal.

Batch publication validates every owner identity, generation, offset, token count
and frame extent before any copy. Foreign, stale, missing, duplicate and mixed
steps reject atomically. Reset checks all generation increments before zeroing
all committed words and retaining allocations. Empty outer batches are no-ops;
zero-token numerical preparations reject. The future hybrid graph must validate
KV and history together before publishing either; consecutive fallible commit
calls alone would not establish whole-step atomicity.

Only the current prefix is represented. Earlier token changes or shortened
histories require reset and replay from zero. This state cannot inherit the
dense session's arbitrary KV-prefix rewind. Token/model identity, pending sampled
tokens and distributions remain responsibilities of the later hybrid session.

Memory accounting includes committed storage and the owner object; caller
staging, coefficients, input/output and collections are separate. For ten FP32
histories at width1024, committed payload is81,920 bytes, candidates81,920 and
one shared work bank8,192:172,032 bytes total. Candidates plus work add90,112
bytes to committed payload. B4a's hypothetical two-bank equation excluded work.
These are payload equations, not measured whole-process admission.

Six unit controls cover every partition of a five-token exact asymmetric
B/C/x witness, existing independently frozen channel-tail convolution/history
arrays, whole-chunk prevalidation, late overflow/abort/retry, batch identity and
publication, reset/replay and size/position/generation exhaustion. Existing
fixtures and their `1e-5 + 1e-4*abs(reference)` gate are unchanged. A separate
allocation test exercises two width1024 owners after warmup:64 preparations,
32 batch commits, eight resets and eight rejected zero-token preparations report
zero Rust allocations. This is an allocation result, not a latency claim.

Formatting, strict default/Metal all-target Clippy and full tests pass:445 default
with20 ignored;526 Metal-feature with169 ignored. Counts overlap. Astra reviews
the source and saved checks before the task commit. Full evidence is recorded in the
[machine result](../benchmarks/results/2026-10-29-short-conv3-history-v1.json).
The [LFM2 plan](lfm2-hybrid-plan.md) proceeds to tiny hybrid composition; no
payload, GPU, real-model, Chat, full-window or iPhone qualification follows here.
