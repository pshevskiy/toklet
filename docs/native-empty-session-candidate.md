# Empty same-model session preparation

`toklet_session_prepare_empty` prepares a separately owned, explicitly empty
session on an already loaded model. It takes the current session, candidate
`TokletOptions`, the existing transition request and a required cancellation
flag. Success transfers an ordinary new session handle; the caller must release
it if it is not adopted. This API does not replace or release the current
session, replay tokens, reopen weights, sample, or apply a Chat setting.

Expected model and session IDs and the exact retained model allocation must
match. Existing creating-thread, busy, poison, pointer and loaded-position
bounds remain authoritative. The new session has a distinct process-local session instance ID and the
same model ID. Its valid history, committed prefix and pending suffix are empty.
Candidate capacity may be smaller than the old history without changing that
history; no transcript or token suffix is trimmed. Failed attempts may consume
an instance ID. IDs are live process identities, not file authentication.

The creator reuses the ordinary constructor and the
[same-model transition ledger](native-session-transition-report.md). Before
construction, both candidate individual and aggregate coexistence estimates
must fit. After construction, actual candidate buffer lengths and host-vector
capacities replace the planned candidate component, and both budgets are
checked again. Both metadata/driver allowances, both bridge allowances, the
JSON allowance and the same caller reserves remain counted. Caller reserves do
not prove app ownership, and this is not a whole-device fit or OS peak report.

Current is borrowed only for inspection. Candidate errors and Rust unwinding
are handled outside the old session's mutating panic wrapper, so they cannot
poison current. Local ownership releases an unsuccessful candidate and its
model reference. Existing ordinary mutating calls retain their poison behavior;
OOM abort and foreign exceptions remain outside unwinding containment.

The additive `TOKLET_CANCELLED` status is 10. The flag is checked before planning,
before construction and after actual admission immediately before returning the
handle; it is never automatically cleared. Construction is synchronous and is
not interrupted midway. A cancellation after the final check can race with
success. Future worker adoption must still reject stale or cancelled requests.
A valid output pointer is cleared on every reported failure. Existing generation
cancellation still uses its existing report reason.

## Verification

Independent saved review approved the bounded native checks. All 16 fresh
compile/CPU commands passed: formatting, strict default/Metal Clippy, 469 default
tests (20 ignored), 577 Metal tests (186 ignored), release builds, exactly 32
exports in each library and the CPU C smoke. Both separately approved Mac jobs
passed: the tiny used-session candidate test and the Metal C smoke. All 18
successful command groups closed with no endpoint errors; actual binaries,
libraries, C callers and raw logs are retained.

The raw final report accounts for 134,460,044 bytes: 28,364 shared model bytes,
17,472 current-session bytes, 11,776 planned candidate bytes and 134,402,432
explicit reserve bytes. The returned candidate is session ID 12 on the same
model ID 1, capacity 16/chunk 3, with zero history, committed and pending tokens.
Its actual 11,776-byte owned ledger matches the plan component by component.
D2 created that handle. The subsequent D1 inspection records still say
`candidate_allocated: false` because inspection itself creates no additional
candidate; that field does not deny the preceding D2 allocation.

The final raw current state is six history tokens, six committed and zero
pending. Earlier in the completed test, rollback was checked with six history,
five committed and one pending; the test then continued generation and discarded
its pending token to check visible committed logits. Those earlier state and
bitwise-logit comparisons, Rc cleanup, injected failures/cancellation, exact and
one-byte-under budgets and actual excess-capacity rejection are completed
assertions. Full state arrays and every boundary report were not exported, so
the final summary is not an independent reconstruction of those assertions.
Three intentional panic-hook messages appear on stderr; the native boundary
caught them, the test exited zero and its harness reported one passing test.
C smoke outcomes likewise record completed ABI/identity/cancellation assertions,
rather than full independent C context records. Sparse process samples are not
complete peaks or performance measurements.

The initial attempt remains failed: formatting passed, then strict default
Clippy rejected the test-only constructor hook's crate visibility because its
result exposed the private native error. Two groups closed with no endpoint
errors; later jobs were not reached. V2 narrows only that hook to `crate::native`.
It does not widen the error type, suppress a lint or change behavior. The passing
16-command phase is a fresh run after that correction.

The native header and retained libraries have 32 exports. The selected C3
application bridge still has 30 and is not rebuilt or consumed by this task.
No worker adoption, Swift/UI Apply, persistence, real-model execution, new
context eligibility, phone or performance claim follows from these checks.
A coherent bridge and a separately reviewed worker adoption boundary are next;
empty preparation does not itself clear the current cache or alter saved Chat.

[Machine-readable evidence](../benchmarks/results/2026-10-30-native-empty-session-candidate-v1.json)
binds the exact eight source files, completed receipts, raw reports, retained
failed attempt and independent saved review.
