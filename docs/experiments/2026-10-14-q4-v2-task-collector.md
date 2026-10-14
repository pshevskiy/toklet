# Frozen larger Q4 v2 task-response collector

The fixed larger v2 candidate can collect the original100greedy task responses
through the qualified CLI. This task qualifies admission/transport/probe behavior;
full response collection, literal grading and the2percentage-point quality gate
remain separate. Small v2 remains rejected and is not admitted by this collector.

## Mechanism and boundaries

[The new adapter](../../tools/experiments/metal_tasks_v2.py) pins the accepted
CLI and full-loss receipts, isolated executable and compiled source snapshot,
current Rust numerical/CLI bytes, fixed larger source/config/tokenizer/assets,
all v2 artifact bytes and original contract/task inventory. Relevant transitive
Python helpers are included explicitly before and after workers. It requires
larger full-loss acceptance before task launch and reports actual version/group/
weight/projection/normalization/readback. There is no checkpoint search or new
quantization candidate.

The existing [task transport](../../tools/experiments/metal_tasks.py) gains only
optional admission, projection-argument and extra-argument hooks; old defaults
remain. Generation still uses all original raw token IDs, categories, caps and
EOS behavior. The independent response validator requires exact visible token
indices, stop/output counts, committed/pending state, no tokenization of prepared
IDs, completed reset/finalized text, bounded events/bytes and complete memory.
The driver decodes independently only after the owned model exits and is reaped.
No native failure can count as a failed model answer.

## Checks and review

14 portable task checks pass. Seven new checks exercise larger-only v2 metadata/
residency, unchanged full/probe caps/stop state, exact reviewed CLI/binary/snapshot
and loss prerequisites, custom owned-admission cleanup with no live-model decoder,
and exclusive evidence destinations. A legacy post-admission receipt-replacement
regression now mocks its earlier admission step so the unrelated newer CLI source
does not mask the intended replacement rejection; its qualification rejection
checks remain independent. Importing its fixture through a module avoids duplicate
test discovery. Formatting, all-target Metal Clippy and whitespace pass. No Rust
numerical source or executable changed; no unrelated full engine rerun is needed.

Astra found a race between qualified executable/source admission and the later
identity snapshot. The final prelaunch pin comparison now reasserts the admitted
binary and all mapped compiled Rust/Cargo hashes. Whole-run mutations for binary
and source replacement prove that no child launches. Source clearance precedes
the native three-category probe.

The native check completes exactly three original cases (code/prose/continuation)
with cap8,24raw events, clean natural worker exit0 and post-exit independent
text replay. Visible counts8/1/8 preserve EOS and pending state. Complete admission
2,267,080,568bytes and joint native lifetime charge2,096,597,584bytes fit6GiB.
All consumed inputs, source/compiled executable, helper/receipt identities and
endpoints remain unchanged; cleanup is empty. Driver exits0. The original full
caps remain256/32/16; no100-response collection or literal grading occurred.

[The receipt](../../benchmarks/results/2026-10-14-q4-v2-task-collector-qualification-v1.json)
binds exact source/check/receipt/executable inputs, native probe events, state,
post-exit decoding, memory and endpoint/cleanup checks. Astra accepted final code, all337bindings, exact24-event/cap/state/text/memory
replay, source snapshots, checks and links with no remaining findings. Probe completion is not100-case collection or task-quality acceptance.
