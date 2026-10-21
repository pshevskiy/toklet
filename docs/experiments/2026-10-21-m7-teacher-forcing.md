# M7 bounded teacher-forcing coordinator

A cold Swift coordinator now applies the frozen teacher-forcing order through
ordinary synchronous native commands. It prepares the phone corpus collector;
this component runs six actual tiny targets, rather than a full model corpus.

The caller owns the immutable bounded token array, admitted worker, deadline/
foreground policy and output sink. The coordinator accepts2..10001 IDs and a
block size1..1024. The production frozen policy uses1024 targets per block,
at most1025 input IDs and one token of overlap; the block size is explicit in
the result. No native owner or full logits cross the command boundary.

For each block it resets, commits the first token without sampling, scores the
next teacher target, then appends that target with zero sampled output. Every
command requires idle live owners, successful status, no emitted IDs/timings and
exact history/evaluated/pending/saved counts. Score identity, current position,
vocabulary and finite loss/logits are checked. The last target is appended too.
A row becomes completed output only after its append succeeds. A failed command
or sink returns no complete summary; the caller retains responsibility for
closing/replaying the actual valid prefix. Successful losses aggregate in target
order with compensated FP64 addition, independently of output storage.

Seven deterministic simulated cases cover all10000 target positions, ten frozen
blocks, the final784-target partial block, explicit shorter blocks, append
failure, a mismatched score prefix, cancelled prompt, sink failure and preflight
bounds. The full-policy simulation checks exact row order, command count and
final785-token prefix; it makes no model-quality claim. Append and sink failures
preserve the expected completed-row count and valid committed prefix. Invalid
policy issues no commands; cancellation does not proceed into scoring/append.

Actual Mac CPU and Metal each score `[7,13,3]` from `[1,7,13,3]` with block size3,
capacity4 and chunk1, then verify the final target is committed and the worker
closes/releases. CPU losses match independent frozen tiny-v1 FP32 prompt logits
within1e-5. The host admits the exact manifest and logits SHA, then reduces each
reference row with Python FP64 `math.fsum`; it does not obtain these reference
losses from the new C API. The three references are4.389499721216301,
5.483734151464388 and5.5483128169001885. Metal establishes functional ordering and
finite score/aggregate checks without inventing a full-quality tolerance.

The [qualification receipt](../../benchmarks/results/2026-10-21-m7-teacher-forcing-qualification-v1.json)
binds exact sources, compiler/package/fixture inputs, generated binaries/modules
and25 published raw copies. Seven portable parser checks reject incomplete or
changed policy/coverage, incorrect row/commit/score identities, nonfinite values,
changed independent loss or aggregation, missing simulation/lifetime evidence
and incomplete/replaced reference inputs. Ten owned native commands complete
with successful exit, no cleanup or endpoint errors. Strict optimized Swift
compilation treats warnings as errors and checks complete concurrency. The
coordinator compiles for Mac arm64 and both device/simulator arm64 iOS18 targets.
The iOS checks emit modules; this task does not wire the coordinator into the
app, boot a simulator or launch a phone.

Astra cleared source before execution. Astra accepted completed evidence with no findings after verifying100 pins,
25 exact staged copies, four code pins, all command/oracle/runtime/module
replays,379 links, privacy and whitespace.
Retain the bounded coordinator and independent checks. Artifact/config admission,
foreground interruption accounting, process-memory checkpoints, streaming raw
phone collection and the full frozen10000-target/100-task quality gates remain
for subsequent components. Numerical kernels, defaults, quality thresholds and
M7.3's device/lifecycle/sustained/base iPhone14 requirements stay unchanged.
