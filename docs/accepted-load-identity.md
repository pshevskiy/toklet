# Accepted model/chat identity report

`toklet_model_chat_identity_create` implements a cold, read-only report for the
exact audited Qwen3 model and chat owner pair. It reuses the existing
`toklet_mlx_inspect_json` and `toklet_mlx_inspect_release` functions for bounded
JSON copying and release. [Cold/API verification](../benchmarks/results/2026-10-30-accepted-load-identity-v1.json)
passes. Subsequent [Swift/controller integration](swift-accepted-identity.md)
exercises successful use with an assembled model on Mac.

The getter checks both creating threads, exact retained model `Rc`, eligible
native execution modes and complete model/codec source-receipt equality before
publishing a report. Mismatched owners and unsupported/unpaired profiles reject;
a valid output is cleared on failure. It performs no file reads or session
borrow, so querying it neither changes a session nor invents a busy-session gate.

The stable preference namespace contains the canonical full-manifest identity
and engine-owned backend, load, text and planning semantic versions. Wire schema
version is separate from planning-policy version. Paths, process instance IDs,
loaded limit and current session settings are excluded from the namespace.
Capture-time model/loaded-limit fields are retained separately for correlation.

Six asset roles distinguish original source consumption, tokenizer parsing,
fixed-policy verification and ancillary verification. The report describes
historical matching content and owner association. It does not evaluate current
disk content, resident GPU bytes, accuracy, device fit or preference admission.
Stored settings remain untrusted requests requiring fresh context and memory checks.

All eleven verification commands pass: formatting, strict default/Metal Clippy,
501 default tests, 609 Metal-feature tests, a release Metal library with exactly
35 exports, and C header/link/null-input checks. Seven new portable controls
cover report construction/validation, stable namespace, source/owner mismatch,
thread checks and unchanged unrelated session state. The 21 default and 188
Metal ignored tests stay separate. Ordinary CPU fixtures execute in the suites;
there is no additional cached audited-model or GPU invocation for this task.
Both compiled products are preserved as byte-identical owner-local copies.

Astra independently checks the saved source, logs, products and closed groups.
Pure report-builder success and public CPU/mismatch rejection are distinct from
successful getter use with an assembled model. The subsequent
[Swift controller check](swift-accepted-identity.md) closes that positive Mac
getter gate once, preserving the original cold-only result scope.
This original cold-only task changed no numerical kernel, context policy, app
preferences or selected package. The subsequent coherent Swift integration
selects 35 exports after fresh default/three-Metal builds and consumer checks.
