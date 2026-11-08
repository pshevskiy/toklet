# Qwen3.5 recurrent context replay

Q1.E passes the frozen17-input numerical and state checks at rotary offset37;
Astra approves the combined diff, invariants and checks with no findings. This extends the
[first-input prerequisite](qwen35-real-fidelity.md) to nonzero recurrence,
convolution history and causal attention. It does not qualify128/512/deeper
contexts, generation, codec/stops, held-out quality, GPU execution or adoption.
Capacity512/chunk32 is allocated;17raw input IDs are actually executed.

The exact source remains `avan-ag/Qwen3.5-4B-Uncensored-MLX-4bit`, revision
`fb98fbc59c71a567f3e23435386c9e93951261db`, SHA256
`0dfd6217f56cadcc373c90283d58d5e673833650fed1d6c299313fb536999043`.
The input IDs come from the existing tiny numerical fixture. They make no
tokenizer, natural-language quality or representative timing claim.

The independent producer streams complete stages at final input16, absolute
position53, under the original ideal FP64 and staged FP32 profiles. Each includes
all recurrent state `[32,128,128]` for24layers, raw convolution history
`[3,8192]`, and all17retained K/V positions for eight attention layers. Both
profiles include full logits. No native intermediate values enter the oracle.
The original tiny fixture still regenerates byte-for-byte.

The final native capture and closed reference compare32,667,136values across
2,102profile/array pairs. None fail `1e-5 + 1e-4*abs(reference)`. Maximum gate
ratios are0.447 ideal/0.299 staged; all greedy IDs are22. The65,334,272-byte
native body also exactly matches the earlier separately scoped capture.

Native reset and chunk sizes17/1/3 compare every raw logit with `to_bits`, then
hash committed recurrent/history/KV state and the last distribution. All agree.
A failed observer after a valid five-input prefix preserves that entire prefix;
mid-frame cancellation also preserves it. Retrying the remaining12inputs
reproduces all distribution bits and the clean final state. Warm numerical
forward/reset/retry has zero observed Rust heap allocations. Exact admission
minus one rejects; an oversized33-input chunk rejects without changing state.

Completed work stays per call. The observed17-input call records4,233
projections,17gathers,408convolutions/recurrences and2,176attention-head steps.
The injected failure completes two frames,502projections,49convolutions/
recurrences and256attention heads, with zero committed inputs. Cancellation
completes one projection and one gather. Native code verifies the successful
12-input retry counters before a final chunk rejection resets counters to
Error/zero; the footer's `final_work` belongs to that rejection.

The independent process closes with full source/runtime endpoints at668,026,032B
lifetime physical peak. Final native peak is2,654,112,288B, with admitted
model/session bytes2,691,890,072B and a separate128MiB diagnostic reservation.
Both owned groups are absent. The6GiB ceiling and public limits remain unchanged.
A separate real-model probe measures three resets after warm forward passes:
each records zero allocations, empty history, absent logits and idle work, with
unchanged owner bytes. Its measured peak is2,637,187,592B.
Diagnostic timings include capture/control work and establish no performance,
sustained or telemetry-overhead qualification.

Fresh631CPU/765Metal tests, formatting, both strict all-target Clippy
configurations and27Python controls pass. Twelve re-sealed negative controls
reject weaker bit flags, wrong per-call counts, boolean extents, duplicates,
scope escalation and altered body bytes. Required3,925inputs remain pinned. Native53
package/C-smoke evidence is reused: its sole changed existing path is behind
`#[cfg(test)]`, and both new children are also test-only. Ordinary library, shader,
ABI, dependency, Swift and packaging inputs remain unchanged.

Preliminary output-path and report-snapshot mistakes remain recorded. The first
value-equality capture was never promoted to bit qualification; a fresh native
run proves the stronger gate. The reader's initial retry/final-counter scope
assumption was corrected against native code and sealed counters. A later
producer caption change has identical numerical AST; the closed reference keeps
its original source pin and a verified source copy. No numerical/quality gate
or arithmetic was changed. The first reset collector missed a report prefixed
by Rust's test name; its failed receipt stays intact. A fresh process with an
exact-prefix parser verifies the reset witness under the same binary.

[Contract](../artifacts/qwen35-context-2026-11-08/acceptance-contract.json),
[verification](../artifacts/qwen35-context-2026-11-08/verification.json),
[iteration](../artifacts/qwen35-context-2026-11-08/iteration-notes.json),
[review/commit](../artifacts/qwen35-context-2026-11-08/publication-checks.json).
