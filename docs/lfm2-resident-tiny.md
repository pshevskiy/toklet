# Resident tiny mixed LFM2 graph on Mac

B4d4c introduces a concrete Metal Model/Session for the original synthetic
H256/I256/V64/Q4/KV2/D64/C12 Conv→Attention→Conv fixture. It composes the
qualified packed affine/gather, direct RMS, raw Conv3, strided product, ordinary
RoPE, Reference attention/cache, SiLU, gating and residual blocks. No new shader
or generic graph interpreter is introduced. This is not a real LFM2 admission.

The Model uploads18 original packed affine parents and36 original metadata
buffers, nine direct norms and two Conv3 coefficient arrays widened exactly to
FP32. The tied table remains one owner. The required zero-bias vector covers
all768 fused BCx output columns. Borrowed source stores do not survive upload;
cold admission includes the caller's declared complete source overlap and checks
it against raw owner/vector minima.

The Session owns one full K/V pair `[2,12,64]`, separate staged K/V `[2,32,64]`,
two history owners with old/candidate `[2,256]` banks, reusable scratch and
candidate/published logit banks. B/C/x are views into one BCx parent. After Conv3
consumes u and copies its raw suffix to the candidate, the same u scratch serves
as C*z output. Matrix row views are cached cold; ordinary append selects them
without rebuilding descriptors.

All math runs on the KV Set's retained Phase-A command and shared sticky status.
Both history candidates are stamped before producers and validated completely
after the vocabulary head, even when a producer is omitted. A narrow checked
pair-dispatch helper reuses the unchanged Half validator; the existing KV loop
still uses one encoder. Set's begin/view/commit/reset/discard bodies remain.

Only successful math can produce the Set's private Ready. Successful terminal
KV copies precede fixed token copies and history/logit bank swaps. Nothing
fallible remains after that copy. A pre-copy failure preserves the published
prefix; uncertain partial copies poison all getters until terminal reset/replay.
Reset checks every destination before resetting the Set, then zeros only the
active history words and invalidates logit/diagnostic availability. Changed
earlier token prefixes require replay from0.

Optional diagnostics require all-row logits and own57 mathematical snapshots;
availability is invalidated before Phase A can overwrite them. Published logits
are separate, so a failed step cannot expose partially overwritten replies.
The ordinary last-row mode returns only the final row through the low-level API.
Sampling, pending-token handling, text, downloads and UI remain outside this graph.

| Numerical owners | Bytes |
| --- | ---: |
| Model buffers, including original packed parents and zero bias | 762,178 |
| Session core with all-row logits | 293,520 |
| Session core with last-row logits | 277,648 |
| Optional57 GPU snapshots | 938,932 |
| Optional host readback reserve | 1,855,488 |

These are checked numerical owner totals, including guards and the committed
token capacity. Actual Rust/Vec/Set metadata, MPS view objects, caller allocations,
source overlap and a shared framework allowance are separate admission fields.
Half diagnostic allocations round their backing storage to four-byte alignment.
The totals are not process peaks or real-model fit measurements.

Full parent checks pass formatting, strict default/Metal all-target Clippy,
450 default tests,545 Metal-feature tests and optimized test compilation. The
hardware journey has two new graph jobs and seven changed-cache/Gemma regressions,
all passing independent saved-raw qualification under prospective64MiB stream
caps. Its source is the separately qualified [Half reference](lfm2-half-reference.md);
no target result sets its gates. All15 schedules/55 chunks/133 tokens are checked.
Astra gates1,923,728 mathematical and199,936 boundary-state values, with no MAIN
failure or WIDE diagnostic exceedance; maximum MAIN gate fraction is0.030099.
All57 physical snapshot hashes, raw-u suffixes, exact per-head KV copies,
masked zeros, model immutability and numerical owner ledgers pass.

The recovery job covers nine omitted/partial/late/head/copy faults and18 replay
chunks. The first collection correctly stops at an ineffective copy witness:
all128 destination words already equal the staged K inherited from prior replay.
Its command completion, poison and published-prefix checks were correct. The
preserved test-only successor uses a fresh owned Session for FirstCopy, retains
the mutation assertion and verifies the exact128-word copied range, all other K
words and all V unchanged. No graph arithmetic, source, gate or token workload
changes. Both collections and exact executed binaries remain retained.

The successful nine-job collection closes every group and retains163,558,564 raw
log bytes, including24,670,544/30,090,938 for the two new jobs below their frozen
32,624,768/36,216,664-byte bounds. Twelve ordinary graph forwards allocate no Rust
memory; all legacy counters also pass, for93 observed zero boundaries. Driver
and capture work is outside that counter. A saved-reader assumption about a
retained command after discard is corrected to the actual released command plus
retained Phase-A completion evidence; no target or gate changes accompany it.

Frame/Ready rejection remains the sealed Set boundary: the public graph API
accepts tokens, not external frames. Existing method identity and current cache
regressions cover duplicate/stale/foreign frame guards. Runtime address evidence
omits the shared status address; its values, guards and byte accounting are
checked separately. No complete runtime address-inventory claim is made.

The [LFM2 plan](lfm2-hybrid-plan.md) retains real checkpoint/text/memory, context/
app/adoption-quality, performance and separate phone validation before support
claims beyond this tiny numerical graph.

[Machine evidence](../benchmarks/results/2026-10-29-lfm2-resident-tiny-v1.json)
pins source/design reviews, both native proposals, full checks, failed/successful
collections, exact executables and the complete independent raw audit.
