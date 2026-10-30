# Same-model session-transition report

`toklet_session_transition_plan_create` describes the memory needed to retain an
existing session while preparing another session on the same loaded model. It
returns a bounded immutable JSON snapshot through the existing inspection
copy/release API. It does not create a candidate, replay tokens, change the
current session, reserve memory or apply a Chat setting.

The call takes live model and current-session handles, existing candidate
`TokletOptions`, and the additive 64-byte `TokletTransitionPlanningOptions`.
Expected model/session IDs must match, and the session must retain the exact
same model allocation. Equal paths or configurations do not establish this
identity. The existing creating-thread, busy, poison and pointer rules apply.
No model files are reopened. IDs describe live process objects, not artifact
provenance or a lasting authorization token.

The report separates shared model storage, actual current-session storage,
planned candidate storage and reserves. Shared learned tensors, positional
storage and model vector tables count once; tied embedding/output weights remain
one owner. Current CPU accounting uses immutable constructor-recorded cache and
scratch capacities plus actual model/history/logit capacities. Current Metal
accounting reads typed buffer extents and verifies actual `MTLBuffer.length`,
then counts host vector capacities. MPS views do not count their referenced
numerical buffers again. Their tables count separately, with internal objects
covered by the existing allowance.

The candidate uses the same ordinary constructor plan and mode checks, within
the loaded model's immutable position bound. CPU's existing 2048-position limit
and Metal's model/device limits are unchanged. Candidate offset is zero under
the native options contract. Metal reports its largest candidate session buffer
and actual device maximum; CPU has no corresponding device fields. The report
does not allocate the planned KV, scratch, history or logits.

Both existing 64 MiB metadata/driver allowances and both 20 KiB bridge allowances
remain in the aggregate, together with the 131,328-byte JSON allowance and
caller-supplied replay, text and application reserves. These reserves are
unverified claims, not observed app ownership. The aggregate is a checked logical
admission estimate, not an OS resident-set measurement or whole-device fit.
There is no blanket 64 MiB subtraction.

Individual candidate fit and aggregate transition fit are independent fields.
A zero aggregate budget yields a useful false-fit report. Malformed options,
stale identities, different model allocations and overflow reject. Retained
Metal diagnostic buffers, profile/parallel state and non-native modes reject
rather than being omitted. A successful ordinary chunk's last-row diagnostic
marker is not an extra allocation and does not prevent reporting.

Existing `Memory` reports and constructor admission behavior are preserved. The
CPU preallocation formula and Metal mode validation are extracted for reuse;
the native mode resolver is shared with actual session creation. No numerical
kernel, precision or context eligibility changes. A later transaction must
revalidate identity and actual candidate capacities after allocation, before any
publication. Replay policy, atomic replacement, Apply/persistence and the
separate reconciliation of Chat text limits with a chosen capacity remain
future tasks.

## Verification

Independent saved review approved the bounded native accounting and ABI checks.
The original 16-command compile/CPU phase passed: formatting, strict Clippy,
466 default tests (20 ignored), 574 Metal tests (185 ignored), release builds,
exactly 31 exports in each library, and the CPU C smoke. Those outcomes and the
unchanged libraries/C callers are inherited under one ignored-test-file
supersession. Three fresh checks then passed formatting, strict Metal Clippy and
the release test build. Both subsequent Mac jobs passed: the tiny used-session
accounting test and the Metal C ABI smoke. All 21 successful command groups
closed with no endpoint errors; this is not a new full-suite run against V3.

The retained raw Metal report accounts for 134,460,044 bytes:

| Component | Bytes |
| --- | ---: |
| Shared model | 28,364 |
| Current session | 17,472 |
| Planned candidate | 11,776 |
| Explicit reserves | 134,402,432 |

The current session reports six history tokens, five committed and one pending,
with capacity 32 and chunk 3. The planned candidate has capacity 16 and chunk 3.
The test separately creates a second session through the existing constructor;
its actual 11,776-byte session ledger equals the candidate plan component by
component. Both reports retain `candidate_allocated: false`: the reporting API
itself creates no candidate. Shared model and reserve totals were independently
recomputed from the raw JSON.

History/logit preservation, exact and one-byte-under aggregate budgets, and
retained diagnostic/profile rejection are verified by assertions in the
completed test. Full state arrays and the boundary reports were not exported;
the raw summary booleans are not a second independent state reconstruction.
CPU and Metal C smoke results likewise attest completed assertions for the
64-byte options ABI, bounded JSON copy and identity rejection. Sparse process
observations are retained without a complete peak-memory or timing claim.

Two failed attempts remain preserved. The first default-only preflight could
not infer an optional tuple when Metal was absent; V2 adds its explicit type.
The first Mac attempt failed while enabling profiling on the test's separately
created second session: it required 134,253,388 numerical bytes, but the 128 MiB
whole budget less the 20 KiB bridge provided 134,197,248. That attempt emitted no
final report and did not reach the C Metal job. V3 raises only the ignored test's
candidate setup budget to 256 MiB so the existing extra 64 MiB profile allowance
fits. Current-session budget, exact/under checks, numerical behavior and all
production admission rules are unchanged. Neither failed attempt is relabeled
as passing.

The new native interface has 31 header exports. The selected C3 app bundle still
has its historical 30-export interface and is neither rebuilt nor adopted by
this task. No Swift/app consumer calls this function yet. This task establishes
bounded native report behavior, not session replacement, Apply, app or rendered
UI behavior, phone support, performance, whole-device fit or a new context range.

[Machine-readable evidence](../benchmarks/results/2026-10-30-native-session-transition-report-v1.json)
binds the exact ten source files, all completed receipts, retained failures,
raw reports and independent saved review.
