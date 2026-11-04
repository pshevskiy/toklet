# Gemma original-relative 10,000-target loss gate

The frozen loss gate passes for the diagnostic native Gemma route against the
pinned official CPU eager BF16 baseline. This completes the loss criterion;
the100-task criterion, earlier fidelity rejections and app adoption remain open.

| All10,000targets | Mean NLL |
| --- | ---: |
| Official upstream BF16 | 9.995776113844665 |
| Native diagnostic | 9.264116739444173 |
| Native minus original | **−0.7316593744004916** |
| Allowed increase | ≤0.05 |

The join checks each global index1–10,000 and its exact frozen token ID once.
Ten independent blocks reset positions; the final block has785input IDs and
784targets. Means use stable sums over individual losses, not equal weighting
of unequal block means. These are relative loss results for the named profiles,
not a claim of numerical equivalence or general model-quality improvement.

Both [original](gemma-upstream-loss1025.md) and
[native](gemma-native-loss-block.md) approved block-one receipts are reused.
The remaining source blocks use the unchanged stock text forward and full-width
head in batches≤32positions. Independent BF16 replay verifies every target's
finite distribution, greedy ID and stable NLL; a stdlib exponential-mass anchor
checks every batch. Each≤512MiB temporary capture is removed only after replay
and complete journal/receipt publication. Partial or failed blocks remain
separate from completed ones. The original block-one raw capture is preserved.

Native collection reuses one admitted1,025-position/chunk32session, explicitly
resets and checks empty history per block, and preserves allocation pointers,
weight ownership and source identity. Per-target FP32 statistics/row hashes
cover all targets. The eight complete prior alignment rows and seven independent
scored-anchor reductions come from approved block one; remaining distributions
are authenticated producer statistics, not saved full rows.

Source logical requests and all78,134reads/34,451,577,974bytes for the new process
match the prospective independent chain. Source/runtime/input/binary endpoint
seals, clean owned worker exits and the corrected replay pass. The initial
native replay missed the first block-start marker because libtest attached its
test-name prefix. A separate postprocessor strips only that exact prefix; raw
logs, original failure and initial scripts remain. Numerical workers were not
rerun and no gate changed. An initial unhandled reset Result caught by strict
Clippy was corrected before collection; those logs also remain.

On M1 Pro16GiB, source peak RSS/physical footprint are1,843,511,296/
1,496,664,296bytes; native peaks are2,891,120,640/2,907,065,056bytes. Both remain
below6GiB. Source and native collection overlapped; timings are diagnostic cost,
not a performance comparison. fmt, strict default/Metal Clippy,550default/
658Metal tests, release build and nine focused loss controls pass. Astra approves
the combined code, checks, collection and final join.

Next qualify resident original BF16 weights and session-owned stock KV for the
100-task source reference, then collect both unchanged task suites. The prior
11/27 and1,025-context4/8 fidelity failures remain rejected. This loss result
changes no ordinary model/app/phone eligibility.
[Evidence](../benchmarks/results/2026-11-04-gemma-full-loss-v1.json).
