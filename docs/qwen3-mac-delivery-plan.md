# Qwen3 Mac delivery queue — 27 October 2026

The human requested Qwen3 on Mac first, completion of the existing source
experiment, Astra review of each code task, commits after checks pass, and an
updated improvement record. Astra reviewed this sequence on27 October.
This queue controls execution order over the earlier Gemma/architecture queues.
The broader requirements remain open; they are not prerequisites for this first
delivery. Near-term completion means **Q1–Q5**, followed by Q6 expansion.

## Selected delivery

Use `mlx-community/Qwen3-0.6B-8bit` at revision
`11de96878523501bcaa86104e3c186de07ff9068`, its pinned tokenizer and the named
plain Chat/raw Expert profiles on the Apple M1 Pro Mac. The exact rendering and
stop semantics must be frozen with checkpoint-specific quality inputs. Source
MLX execution is a reference experiment; Toklet remains the native Rust engine.

The model declares40960 shared positions. Rendered input/history plus requested
continuation must fit the usable window and any separate documented caps.
Smaller qualified or device-admitted windows must be visibly labelled partial.
Q1–Q5 do not establish the full model window or iPhone qualification.

Current app admission/context facts and lifecycle integration already exist
([L3 evidence](context-app-integration.md)). Current C512 cache/state schedules
pass their witnesses, but numerical qualification remains rejected. The
[experimental envelope](../tests/fixtures/qwen3-context-gates-v2/README.md) is
rejected for qualification and cannot authorize validation or larger app limits.

## Active sequence

| Task | Deliverable and decision gate | Current state |
| --- | --- | --- |
| Q1 — finish the source experiment | Two cold original-source captures, original bits/exact FP32 widening/full readbacks, then distributions, argmax/margins and diagnostic loss against both saved CPU profiles. | Complete. [Source experiment](qwen3-quantized-source.md):204201984 values per set, BF16 output,1344 compared rows/1338 diagnostic targets. |
| Q2 — precision decision | Choose one bounded implementation outcome and state qualification separately. Source parity cannot replace an unquantized quality baseline. | Complete: retain native Half unchanged **only as an experimental candidate**; reject current C512 qualification. No BF16 switch, adoption or gate expansion. |
| Q3 — diagnostic1025 and complete quality | Diagnostic-only1025 capacity/state checks, frozen checkpoint-specific inputs/candidate/policy, pinned original BF16 baseline, all10000 loss targets and100 tasks per profile, isolated grading and unchanged0.05-nat/2-percentage-point limits. App limits stay fixed. | Complete. [Full comparison](qwen3-full-quality.md): loss increase0.00400406; strict tasks3/100 versus5/100, exactly the two-case limit. Retain experimental Half only. Weak absolute quality and rejected C512 numerics remain explicit. |
| Q4 — qualify the retained precision/context profile | Preserve the uniform0.05 normalized-log-probability/exact-greedy contract and independent FP64/Half reference. A changed candidate must be frozen before screening; untouched validation remains conditional on passing. | [Old candidate remains rejected](qwen3-output-screen.md). [Astra's diagnosis](qwen3-attention-accuracy-experiment.md) identifies attention reduction rounding as a concrete local mechanism. An opt-in compensated candidate passes primitive/control checks and improves GPU Half mismatches16→3. Next: one complete logits-only original/future-tail512 screen under unchanged limits. No default/adoption, gate widening or automatic next-operator search. Eight validation inputs remain unopened. |
| Q5 — usable Qwen3 Mac milestone | Complete Chat, Expert and the supported pinned HF search/import/download/offline-reopen journey for this admitted profile. Verify long input/reply, shared budgets, cancellation/resume, model switching, truthful limits and whole-process memory. Reuse L3; retain a clearly stated partial window until Q6 qualifies more. | [Bounded experimental C512 functional milestone complete](qwen3-mac-functional.md): visible recovery/switching, pinned HF download/offline path, full raw Expert384+128 and normal-stop Chat-worker385+127, exact saved prefix/overflow/closure and memory checks pass. Qualified Q1–Q5 delivery remains incomplete because Q4 is rejected. |
| Q6 — expand the selected model window | Progress through named context/rotary/cache boundaries toward the declared40960 shared positions, including exact boundary and rejected maximum+1, separate long input/continuation, full quality and complete memory/text/UI delivery accounting. | Subsequent expansion; smaller windows remain partial. |

Gemma, additional architectures/optimizations, broader HF compatibility and
iPhone14Pro qualification remain in their existing plans, deferred from this
critical path. They are not removed or marked complete.

## Working rules and stopping conditions

- A coherent code task includes implementation, targeted checks, Astra review,
  improvement-record update and commit. Reuse existing controllers and raw
  evidence. Add another harness or replay only for uncovered behavior or a
  concrete unresolved risk.
- Review changed numerical contracts before measurements. Review completed code
  and results before commit; intermediate review is for material correctness
  changes, not every mechanical edit.
- Default to one final Astra review per coherent code task, with checks/results
  ready. Add pre-measurement review for changed numerical contracts, not routine
  tooling subdivisions. Reuse existing controls; no new general harness unless
  it covers a specific missing behavior.
- Q1's bounded endpoint is the source comparison and Q2 decision. Further
  environment work must be necessary to complete those existing captures.
- Q1/Q2 are closed. Astra's source review retains the Half implementation as an
  experimental candidate and rejects current C512 qualification. Q3/Q4 were
  narrowly reordered to obtain full quality evidence before selecting a new
  precision contract. Quality outcomes cannot tune that contract or its gates.
- Correctly rounded primitives close their investigations. Schedule-independent
  cache results close cache debugging unless contrary evidence appears.
- A failed envelope review triggers a profile decision, not another tolerance
  derivation. No rejected gate or quality threshold is loosened to obtain a pass.
- Track delivered capability: qualified usable context, completed full quality,
  and working Mac app journeys. Test/commit counts support that evidence.
- Status reports state the current usable profile, acceptance gap, next concrete
  deliverable and remaining uncertainty. Do not give a full-roadmap ETA from
  diagnostic throughput or count partial support as full-window completion.

The [model capacity contract](long-reply-engine-plan.md),
[broader engine inventory](engine-expansion-plan.md) and original milestone
acceptance policies remain applicable. Historical receipts keep their scope.

Q4's proposed uniform log-probability bound means every possible next-token NLL
differs by at most0.05, independently of the true target. It is a new stringent
output-fidelity requirement, not a consequence of Q3 passing mean held-out loss.
Bind exact archived reference/candidate identities and the method before computing
that screen. Report worst positions, normalized errors and greedy agreement.
Calibration inputs without existing native captures require no new execution.
If the screen fails, failure alone does not justify switching dtype or starting
another investigation chain. The original stage rejection remains separate.

The archived screen has now rejected this contract/candidate pairing. Astra
approved proceeding with independent Q5 functional work while preserving that
rejection, experimental/partial labels and all current limits. No numerical
validation is authorized by the relative-quality pass or matching greedy IDs.

## Current critical path and ETA

Q5's bounded experimental Mac functional scope is complete. Source/runtime
validation and repeated review/tooling subdivisions contributed substantially
to elapsed time. The remaining blocker is Q4 numerical acceptance, not Chat
integration. There is no defensible full-qualified-delivery date yet.

Astra's bounded same-input investigation now supports one accuracy experiment:
compensated Reference attention. This is not a demonstrated bug fix. CPU FP32
also fails the uniform FP64 trajectory rule, so that rule must not be confused
with generic FP32/Half conformance or the mean-loss quality policy.

The opt-in implementation and local GPU checks are complete. The next endpoint
is one frozen whole-model screen, then retain/reject that candidate under the
existing criterion. If it rejects, stop and resolve the explicit acceptance-policy
question; do not launch another trace, harness or envelope merely to keep activity
moving. Q6/full40960 and broader architecture/phone scope still have no reliable
completion ETA. No threshold/reference/default change is authorized by this result.
