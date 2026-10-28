# Resident Half KV cache transaction

29 October 2026. B3c3c qualifies a concrete internal cache set on Apple M1 Pro.
It reuses [row geometry](window-row-geometry.md), F16 `KvOwner` accounting and
the [frozen copy expectations](gemma3-staged-attention-reference.md). The
[attention leaf](gemma3-staged-attention.md) is separately qualified. Complete
decoder composition and real-model execution remain [open](../TODO.md).

## Ownership and readiness

An immutable Operation owns a bounded Plan and one optional Half-bit validator
pipeline. A session's Set borrows that Operation and owns separate old K/V,
new K/V staging, guarded shared status and at most one command. Shapes are
1–32 physical KV1 owners, D32/256, common C1–544 and full or local W1–C
retention. Fixed metadata arrays avoid per-step collection growth. Graph Q,
attention scratch, logits, token history and weights are separate owners.

`begin` checks state, capacity and epoch, resets status while quiescent, then
encodes an active-region staging fill with byte `0x7e`. Its Half NaN stamp
`0x7e7e` appears before all producers in the same command. An omitted producer
cannot silently reuse an earlier finite staged value. Padding and guards stay
untouched. Batch/Frame objects contain opaque nonborrowing set/owner/generation/
epoch/offset/count metadata. Temporary views borrow the command and resources.
Internal callers must end each encoder and must not retain raw Objective-C
handles for later submission or mutation outside this protocol.

`finish_math` classifies premature submission before metadata rejection, then
checks exact owner count/order and every frame field. It appends K/V validation
for every owner, even if attention was never encoded. The integer kernel reads
Half words and rejects all-ones exponents, accepting signed zero and subnormals
without float-conversion or FTZ assumptions. Only active rows are checked.

One Phase-A submission/wait must finish with Completed, no native error and
zero shared status before Set can create opaque Ready. No caller boolean or
arbitrary completed command can create it. Old KV and its committed metadata
remain unchanged on rejected math. This does not prove that a complete model
graph was encoded: the higher session must encode its full head and publish
candidate logits/history only after successful cache commit.

## Copy, publication and recovery

Ready seals encoding views. Commit rechecks the complete metadata and all copy
ranges before encoding one blit command for every owner. The shared row helper
retains the final min(n,W) local rows in at most two runs per K/V, or one full
run. No old row is overwritten while attention still needs it.

The whole set enters the poisoned state before Phase-B submission. Only
Completed with no native error permits publishing the new offset/generation
together and returning CommitInfo. Submitted-copy uncertainty never rewinds a
partially overwritten ring to a prior nonzero offset. Pending commands prevent
mapped access/reuse/reset. After confirmed terminal completion, reset invalidates
frames, sets offset zero and clears status; a higher session must replay trusted
input. Native completion-error branches are statically reviewed, not reproduced
as real hardware faults by this qualification.

## Memory and checks

Old KV uses shared F16 owner equations, with seven guards per array; add two
guarded 32-row staging arrays per owner and one 32-byte status. Fifteen W512
local plus three full owners at C544/D256 require 10,126,320 buffer bytes plus
status. Set's full inline owner storage and a conservative Batch/Ready/View/copy
metadata reserve are added explicitly. Operation/Plan and one 64 MiB compiler
allowance are separate; graph/application/driver allocations are not hidden in
this count. The 18-owner figure is a checked sizing result, not a measured
whole-model/device fit. Exact/minus-one budgets and device bounds are checked
before owned buffer allocation.

Formatting, strict default/Metal Clippy, 424 default tests, 310 Metal-library
tests and two targeted debug/two release controls pass; counts overlap. The
initial lint failure is preserved; two mechanical fixes leave semantics intact.
Three explicit release GPU tests pass. Astra independently reconstructs all
67 raw snapshots and 6,210,796 physical Half words, including:

- 1,623,152 final source-copy words across eight cases, with an actual 513→514
  chain, a wrapping chunk and a chunk larger than the window.
- 1/32-owner bounds, ten finite Half classes, six NaN/infinity patterns and
  shared graph-status rejection, without requiring attention to run.
- Omitted producers after finite staging, and a seven-row partial producer
  leaving only the last two active K words unwritten.
- Foreign, stale, reordered, duplicate, missing and altered frames/Ready,
  premature enqueue, epoch/generation exhaustion and repeated commit rejection.
- An actual first-K-run-only copy followed by a forced protocol error: one K
  owner changes, V and other owners stay unchanged, offsets remain unpublished
  and the entire set rejects further use. This is a test injection after a
  successful terminal Metal command, not a real Metal failure or rollback.
- Terminal reset to zero followed by ordinary GPU replay chunks [5,7], checking
  1,052 final words against frozen expectations.

The audit authenticates 426 before/after input hash pairs, 6,188 Half guard words
and 469 status guards. All 27 observed complete begin/finish/commit boundary
calls have zero calling-thread Rust allocations in release; the unchanged
required-zero gate covers the 12 source-path calls. These measurements include
command creation and waits, not just borrowed encoding. Driver allocations are
excluded. All raw data and command status/error are retained before assertions,
and snapshot helpers reject in-flight mapped reads.

Memory samples are direct-process observations; two jobs have only one early
sample each and cannot establish execution peaks. No full graph, real checkpoint,
token/model replay, app, phone, quality, latency or throughput qualification is
claimed. Seeded copy states and synthetic KV replay are distinct from a model
prefill. See [machine-readable evidence](../benchmarks/results/2026-10-29-gemma3-resident-cache-v1.json).
Next: consume the blocks in a complete resident Gemma3 Model/Session.
