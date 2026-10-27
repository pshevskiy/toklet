# Engine context facts and request preflight

L3.1 adds read-only engine facts and request arithmetic. Runtime and app limits
are unchanged. Swift consumption and explicit app policy now have separate
[L3.2 evidence](context-app-integration.md); the unused JSON capability proposal
is not treated as a native API.

The MLX dense source retains `max_position_embeddings` from the same bounded,
duplicate-checked configuration document passed to architecture admission,
before requested context narrows `Config.max_positions`. A successfully loaded
Metal model copies that declaration into immutable metadata. Queries use the
retained model; they do not reopen its directory. Legacy paths report the
declaration as unknown. This is validated configuration data, not a new claim
about artifact content authentication or usable/qualified model context.

The additive ABI v1 `TokletContextInfo` record separates:

- Model-declared total positions and optional distinct input/output caps.
- The global loader ceiling, loaded position limit and selected session capacity.
- Absolute offset, sampled history, committed positions and pending tokens.
- Numerical and bridge admission from whole-app/device fit and qualification.

Unknown declared values are zero with their known-field bits clear. No model
input/output caps are inferred from generation defaults. Qualification and
whole-app/device fit are explicitly not evaluated. In particular, the global
32,768-position ceiling does not authorize Qwen3's selected backend above its
smaller loaded graph/session limits.

Process-local instance IDs bind the immutable model and each live session.
Checked exhaustion rejects instead of wrapping; IDs are not artifact digests
or persistent identifiers. A session retains its exact model allocation and
model ID after the external model handle is released. Reset preserves instance
identity; a new session receives a new ID.

`toklet_session_preflight` accepts the same operations, raw or already-rendered
input IDs, requested output count and stop IDs as generation. It shares
`generation::request_admission` with the existing validation entrypoint:
Prompt/Reset use base zero, while Append/Continue include the full sampled
history, including pending tokens. Checked addition rejects overflow and the
whole input/history-plus-requested-continuation budget must fit. Zero output
preserves prefill-only behavior. `remaining_tokens` is capacity remaining after
both the input and requested continuation, not the available output beforehand.

Preflight is an arithmetic/token-bound snapshot. It reserves nothing and does
not guarantee execution succeeds. Actual generation validates again. Context
query and preflight use the existing creating-thread, busy/reentry and poisoned
session rules, and leave caller output unchanged on failure. Fixed-width records
contain ABI version, struct size, known fields and reserved fields; no model,
KV, tokenizer or logits arrays cross the boundary.

Planned validation covers declaration/requested-limit divergence and later disk
mutation, explicit unknown legacy metadata, model release/session survival,
new-session/reset identity, exact and maximum+1 requests, arithmetic overflow,
pending history, all four operations, rejected preflight state/output, stale
snapshot versus intervening generation, wrong thread/busy/poison, native C
layout/link/exports and actual tiny Metal metadata propagation/regression.
## Actual checks, 27 October 2026

Formatting, strict default/Metal all-target Clippy and367 ordinary CPU tests pass;
19 download/hardware/explicit tests remain ignored in that ordinary run. New
unit coverage includes checked instance-ID exhaustion, same-config declaration/
requested-limit divergence, later disk mutation, all four request operations,
exact/+1 and usize overflow, zero-output prefill and pending sampled history.
Three native methods verify lifetime/identity, read-only admission and
null/wrong-thread/busy/poisoned failures with unchanged outputs. Valid context
queries and all five native preflight witnesses allocate zero heap buffers.
An intervening generation makes an old snapshot insufficient; actual generation
revalidates and rejects without changing its new state.

The explicit tiny Metal check retains declared32/loaded8/session8 after changing
the directory's declaration to128 and releasing the external model handle.
Unknown model input/output caps stay unknown, and qualification/device-fit
states remain not evaluated. Three existing independent tiny Qwen3 GPU
numerical/session/profiling regressions pass. These checks do not qualify new
context ranges or real-model/phone behavior.

An actual linked C smoke passes136-byte context/192-byte request layouts and
failure-output checks. All28 header entrypoints exactly match library exports,
including the two new functions. The build uses Mac SDK27.0 with minimum macOS15
on Apple M1 Pro/MacBookPro18,3,16GiB/macOS26.6.2(25G83). The source and protected
release binaries are bound to the evidence; the main release CLI/benchmark
binaries remain unchanged.

The first compile found a missing new field in a private synthetic Source test
wrapper. The corrected Rust/Metal run passed, then C compilation lacked an
explicit SDK sysroot. Both failed receipts remain archived. The tool-only retry
reuses authenticated unchanged source, fixtures, header, library and Rust/Metal
results, and reruns only outstanding C checks with the SDK path. All21 owned
groups are absent. Astra accepts the source and complete scoped evidence.
See the [public receipt](../benchmarks/results/2026-10-27-context-admission-v1.json).

L3.2 [app integration](context-app-integration.md) now consumes these records
in both modes with explicit policy, fresh identities and rendered shared-window
preflight. Its admission/state checks preserve current ranges. Next L4 qualifies
independent real Qwen3 C512 numerical execution before raising capacities.
Real Gemma whole32/long BF16/1025 quality and broader full-window work remain open.
