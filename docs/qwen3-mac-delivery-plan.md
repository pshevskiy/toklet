# Qwen3 Mac delivery queue — 28 October 2026

**Historical queue:** on28 October the user replaced this single-model priority
with the [shared-block architecture and capacity plan](shared-block-model-plan.md).
Completed scoped results remain valid. The8192 experiment was interrupted before
deep quality/final audit and is [not qualified](qwen3-context8192-experiment.md).
Do not resume the full-window ladder as an active prerequisite for broader models.

The human requested Qwen3 on Mac first, completion of the existing source
experiment, Astra review of each code task, commits after checks pass, and an
updated improvement record. Astra reviewed this sequence on27 October.
This queue formerly controlled execution order over the Gemma/architecture queues.
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
pass their witnesses. On27 October the human approved the
[native arithmetic profile](qwen3-native-profile.md) as the qualification target;
current-build conformance and complete quality continuity now pass. Historical
CPU-trajectory fidelity remains rejected. The
[experimental envelope](../tests/fixtures/qwen3-context-gates-v2/README.md) is
rejected for qualification and cannot authorize validation or larger app limits.

## Historical sequence

| Task | Deliverable and decision gate | Current state |
| --- | --- | --- |
| Q1 — finish the source experiment | Two cold original-source captures, original bits/exact FP32 widening/full readbacks, then distributions, argmax/margins and diagnostic loss against both saved CPU profiles. | Complete. [Source experiment](qwen3-quantized-source.md):204201984 values per set, BF16 output,1344 compared rows/1338 diagnostic targets. |
| Q2 — precision decision | Choose one bounded implementation outcome and state qualification separately. Source parity cannot replace an unquantized quality baseline. | Historical source decision: retain native Half as an experimental candidate and reject the then-current C512 fidelity qualification. No BF16 switch or gate expansion. The separate native-profile Q4 result now qualifies C512 under its approved contract. |
| Q3 — diagnostic1025 and complete quality | Diagnostic-only1025 capacity/state checks, frozen checkpoint-specific inputs/candidate/policy, pinned original BF16 baseline, all10000 loss targets and100 tasks per profile, isolated grading and unchanged0.05-nat/2-percentage-point limits. App limits stay fixed. | Complete. [Full comparison](qwen3-full-quality.md): loss increase0.00400406; strict tasks3/100 versus5/100, exactly the two-case limit. Historical relative-quality decision retained experimental Half. Q4 now binds the current native profile to those exact full outputs; weak absolute quality and historical fidelity rejection remain explicit. |
| Q4 — qualify the retained native precision/context profile | Human-approved [native-profile v1](qwen3-native-profile.md): frozen Reference attention, unchanged independent operator/graph gates, exact state/schedule validation and complete current-native quality continuity under original0.05-nat/2-percentage-point limits. Strict trajectory fidelity remains a separate rejected result. | Complete at C512 on M1 Pro: unchanged independent gates and all eight reserved cases pass; full current-native10000-target/100-task replay is identical to the historical quality result. Astra accepts the scoped profile. No default/capacity increase or transfer to CompensatedReference. |
| Q5 — usable Qwen3 Mac milestone | Complete Chat, Expert and the supported pinned HF search/import/download/offline-reopen journey for this admitted profile. Verify long input/reply, shared budgets, cancellation/resume, model switching, truthful limits and whole-process memory. Reuse L3; retain a clearly stated partial window until Q6 qualifies more. | [Bounded experimental C512 functional milestone complete](qwen3-mac-functional.md): visible recovery/switching, pinned HF download/offline path, full raw Expert384+128 and normal-stop Chat-worker385+127, exact saved prefix/overflow/closure and memory checks pass. The selected native profile now passes Q4; C512 remains partial and larger-window delivery follows Q6. |
| Q6 — expand the selected model window | Progress through named context/rotary/cache boundaries toward the declared40960 shared positions, including exact boundary and rejected maximum+1, separate long input/continuation, full quality and complete memory/text/UI delivery accounting. | Scoped4096 diagnostic passes state, generation, deep relative quality and process fit. The explicit parallel Reference schedule is retained after exact complete replay/quality and repeated measured gains. Ordinary app512 stays partial. Next qualify8192 separately; full-window source memory, streaming/admission and final quality remain open. |

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
- Q1/Q2 are closed. Astra's historical source review retained Half as an
  experimental candidate and rejected the then-current C512 fidelity contract.
  The separately approved native-profile Q4 qualification now passes. Q3/Q4 were
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

The historical Q4 uniform log-probability bound means every possible next-token NLL
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
validation was authorized by the relative-quality pass or matching greedy IDs.
The human subsequently approved native-profile v1 as a different, explicitly
specified product target. That authorizes its frozen conformance/state validation
and complete current-quality bridge; it does not turn either fidelity rejection
into a pass or use the rejected calibration envelope.

## Current critical path and ETA

The native Reference profile passes current-build conformance/state and the
complete quality-output bridge. The bounded C512 native-profile milestone and
existing Mac functionality are complete. Historical fidelity remains rejected;
weak absolute quality and the partial model window remain explicit.

Follow the [Astra full-window design](qwen3-full-window-plan.md): first scoped
Qwen3 diagnostic admission and checked resource planning, then boundary/deep-quality
qualification and bounded real streaming/application delivery. The first
[diagnostic admission/resource mechanism](qwen3-context-admission.md) is checked,
including successful pinned-source40960 metadata and ordinary short regression.
The [finite4096 diagnostic](qwen3-context4096.md) now passes exact native
state/schedule/generation, deep controls, a1024-target BF16 relative-quality
supplement and whole-process fit. Native loss increase is0.00040249≤0.05.
The ordinary app remains512, and other-model limits stay unchanged.

Astra recommends one bit-exact Reference scheduling experiment before8192:
the ordinary4064-input chunk32 prefill alone records456.64 seconds of completed
model calls, followed by206.10 seconds for31 decode calls. These single
uncontrolled observations establish a delivery bottleneck, not a qualified
performance result. Keep Serial as oracle/default, qualify the separate explicit
schedule through exact F2 and complete10000/100 output bridges, and require
representative repeated/confirmed benefit before retaining it. The full-window
source's separate host-reader/workspace proof remains open.

Completed4096 call times are now retained. Source BF16 long-cache workspace and
full-process memory must still be admitted independently at later depths.
Full40960, other architectures and iPhone work have no reliable completion ETA yet.
