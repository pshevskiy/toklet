# Bounded Phi3 dense composition

Toklet now parses ordinary-RoPE Phi3 exports and maps their original fused QKV
and gate/up rows into the existing dense CPU and Metal blocks. No new inference
kernel or numerical dispatch hierarchy is introduced. This completes the tiny
composition checks; the [pinned real checkpoint](phi3-fused-source.md) still
needs independent numerical, session, text and measured-memory qualification.

## Source and ownership contract

A concrete private `ParsedConfig` carries validated architecture, tied-output,
position and source-window facts. Existing families retain their previous
semantics. The Phi parser requires bias-free SiLU, zero dropout and full-head
ordinary split-half RoPE. Non-null scaling, partial rotary, conflicting dtype
aliases and unknown semantic fields reject. `auto_map` is descriptive metadata;
no remote Python executes. Actual tensor dtypes remain authoritative.

The adapter replaces the common separate-projection inventory with physical
QKV/gate-up parents before validating quantization overrides and the header.
Checked original row partitions then feed the [shared cold logical loader](logical-projection-loading.md).
Physical payloads are unique, output tying is preserved, and each actual
resident buffer contributes its own guards/padding. Completed models retain
concrete weights, with no tensor-name lookup in their numerical loops.

The source declares4096 positions and window2047. Initial requests must fit
the declaration, window and512 implementation ceiling; they reject rather than
clamp. Existing checked `offset + capacity` admission enforces the same bound
for CPU and Metal sessions. This bounded full prefix fits the source window;
it does not implement general sliding attention. The optional compatibility
window field describes the source declaration and is omitted for old `None`
reports. Native Phi uses the existing structural dense/index/frequency-domain
checks and Reference attention; Online rejects.

## Independent tiny checks on Mac

The new stdlib producer independently decodes and splits original fused tensors.
It preserves unaligned payload offsets, affine4/group64/F16 coefficients and the
packed FP32 head. Geometry is2 layers, hidden192,2Q/2KV heads, head96, MLP256 and
vocabulary64. Four untied17-token cases cover the prompt, changed future tail,
offset5 and offset495; a separate three-token case checks tied output. Offset495
ends at511 with17 local keys. A512-token KV capacity is also allocated and
checked, with at most21 tokens in the native state controls.

Five cases have195 source stage groups and585 CPU/main/wide arrays. Export and
replay are byte-identical; Astra independently verifies all stored values,
envelopes, extents and source identities. CPU checks compare513,704 values at
the unchanged `1e-5 + 1e-4*abs(reference)` gate; exact cached schedules, causal
prefix, reset, window/budget and native-domain controls pass.

Metal checks observe35 of39 source stages per chunk; scores/probabilities remain
oracle-only. All19 schedules,281 committed rows and3,815 snapshots pass:
1,996,224 values meet the frozen **main-only** gate
`max(1e-4, 2*case-stage-envelope) + 1e-3*abs(main)`.
Maximum gate fraction is0.8143322475570033; maximum absolute error is0.00390625.
Wide errors are diagnostic, with zero exceedances of their reported bounds.
Exact finite numeric comparisons allow equal signed zeros; raw journals retain
original IEEE bits. All-row schedules, causal prefixes, cached reuse, rejection
atomicity, reset/partial replay, both readbacks, raw generation, stop/cancel/
output-error handling, Online rejection and exact session budgets pass.

Resident tiny untied/tied weights are347,590/340,678 bytes, including70 bytes of
norm guards above original payloads. Exact session admissions are68,483,922/
68,477,010 bytes. These are checked numerical allocations/admissions, not measured
whole-process peaks. Raw logs retain per-stage aggregates and raw logits;
passing intermediate arrays are not retained for an external all-stage rescore.

Formatting, strict default/Metal all-target Clippy,410 default tests and288 Metal
library tests pass; suites overlap. The frozen debug test binary passes both
explicit GPU tests on Apple M1 Pro. Initial test-only slice-borrow and JSON-number
spelling failures are preserved and corrected without changing fixtures or gates.
Prospective review also corrected an extra wide fatal gate before any GPU run,
restoring the already-frozen policy.

## Real-artifact metadata and remaining work

Cold inspection authenticates291 logical specs from the455 physical header
entries. Original source payload is2,149,644,288 bytes. Full scalar FP32 weights
would require15,284,318,208 bytes; scalar loading admission is15,406,899,200 bytes
and correctly exceeds6GiB. Metal metadata is compatible-but-untested; it still
requires payload/domain, device and session checks. No real model runs in this
task. No throughput, full512-key numerical,4096/sliding-window, app, iPhone or
held-out quality qualification is implied.

Astra reviewed the source policy, producer, frozen fixture, consumers and
collection controller. Final source/evidence review approves retention and commit, including an
independent rescore of17,984 retained logits. Next is B2c's short real
source/native/session/text and measured-memory protocol.
See [compact evidence](../benchmarks/results/2026-10-28-phi3-dense-composition-v1.json),
[TODO](../TODO.md) and [improvement record](improvements.md).
