# Larger Q4 v2 matched full task quality

The frozen larger Q4 v2 candidate passes the original100-task precision-loss
gate:11/100 versus the matched FP32 reference11/100, a0percentage-point drop
within the unchanged2pp limit. Code/prose/continuation scores are0/40,10/30,
1/30. Absolute scores remain low; this is precision-loss qualification on the
frozen suite, not a broader coding-quality claim.

The new [admission adapter](../../tools/experiments/task_grade_admission_v2.py)
requires the exact full response collection,100ordered original caps, complete
raw/state/stop replay,101ordered native-memory observations, exact producer
inventory and initial/current compiled-source/binary/receipt/artifact pins.
It independently finalizes tokenizer text only after the model exited. The
[v2 window](../../tools/experiments/task_grading_window_v2.py) supplies cold
admission/provenance hooks to the existing window/runner. Legacy defaults and
literal program construction, controller, classifier and text rules remain
unchanged. Small Q4 v2 remains rejected.

43 portable checks pass, including every missing inventory entry, extra
entries, saved/current executable/source/receipt/artifact replacement before
decoding, missing/duplicate/reordered memory labels, malformed caps/modes/
exit/accounting, custom admission failure and late-live process rejection
before VM startup. Successful injected admission forwards the selected
validator/provenance and restores stopped state. Legacy native-record unit
fixtures predate the v2 Rust module; they add a synthetic identity only in
mocked inventory checks. No historical native evidence is re-admitted.

Astra source-cleared the code before the native window. The actual full
collection admits100responses with its exact saved summary/raw identities.
The owned qualified arm64 Linux image grades40literal programs only after
fresh process/group absence and fixed controller/runtime/source admission.
No generated code executes on the host or receives completion repair.
All100outcomes resolve, with zero infrastructure-unresolved cases. All40code
responses fail literal source tests. The window removes its owned containers,
verifies an empty live container inventory, restores the initially stopped
machine and finishes helper pipes. Driver exits0; a fresh final inventory
confirms stopped/not-starting. No restoration receipt is needed because
ordinary in-window restoration completes successfully.

Full quality acceptance combines this task result with the
[larger10,000-target loss pass](2026-10-14-q4-v2-full-loss.md). The small v2
loss rejection and both v1 rejections remain recorded. Performance, sustained
workloads, application/repeats and the named useful Q4 gain remain outstanding;
M5.2/M5.6 and overall M5 stay open.

[The receipt](../../benchmarks/results/2026-10-14-q4-v2-task-quality-qualification-v1.json)
binds the code/tests, immutable grading sources, full collection/raw responses,
frozen matched baseline, original native grading/window and byte-identical
published copies, plus literal/runtime/control identities and fresh VM state.
Astra independently accepted all351bindings, exact copies/source endpoints and
matched baseline,40literal-code/60text replays,11/100versus11/100, zero
unresolved outcomes, stopped-machine restoration and completed helper pipes.
