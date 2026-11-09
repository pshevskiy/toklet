# Explicit Qwen3.5 untied output head

Q2.C adds an explicit separate output projection to the existing hybrid numerical
core and a diagnostic binding for the [authenticated9B CPU store](qwen35-9b-storage.md).
Required checks, explicit source controls and matched release measurements pass.
Astra approves scoped retention with no actionable findings. This is a binding/correctness change;
full real-source9B numerical qualification and managed adoption remain open.

`Weights` and `Model::new` keep their public interfaces and tied-head behavior.
`Model::new_untied` takes an additional original Q4/G64/BF16 matrix of shape
`[vocab,hidden]`. Input lookup continues to use the embedding. A concrete optional
matrix selects the output once at the final operation boundary; numerical inner
loops, accumulation, masking, RoPE, epsilon and session semantics are unchanged.
The private model metadata and separate learned representation enter checked
memory accounting. The source factory replaces span counts with the actual store
capacities after validating complete representation coverage. Weights stay
borrowed and immutable, with no cloned or expanded full-model owner.

The4B factory remains tied and source-specific. The separate9B CPU-store factory
resolves its pinned embedding and distinct `language_model.lm_head.weight`; it
cannot substitute a4B store or qualify a9B GPU upload. Existing Metal binding
inventory can represent a separate head, as checked with the tiny fixture only.
The extra private optional matrix adds16bytes to the arm64 Metal model layout;
all other observed layouts remain unchanged. Old raw evidence is retained with
its original ABI, and new accounting uses an explicit frozen layout observation.

The new two-table fixture appends a different output head to the exact original
weights. An independent Python producer keeps all non-logit reference values
unchanged and changes104logit arrays. The original tied fixture and generator
are unchanged. Both original profiles (`ideal_fp64`, `staged_fp32`) and the
`1e-5 + 1e-4*abs(reference)` gate apply to all stages, recurrent/KV state and logits
for four cases and chunk partitions1/3/17. Tests reject head substitution,
shape/packing errors and an undersized memory budget. Late head execution,
observer and cancellation failures preserve the valid committed prefix; retry,
reset, Metal poison/repair and warmed zero Rust allocation checks remain required.

The actual9B CPU diagnostic passes complete store binding,512capacity/32chunk
session admission and one raw token. It records5,363,086,344admitted owner/policy
bytes and a5,309,649,072-byte native lifetime physical peak under explicit child
allocator control, below6GiB. All249projections and one token commit successfully.
The final projection matches direct execution of the same source head bitwise.
That checks routing and work, **not an independent whole-graph9B oracle**. The
separate full-source oracle, >4GiB reference layout, GPU execution, default
allocator, codec/application fit, request/overhead, quality and managed-route gates
remain open. The4B128/256-input [dual-profile rejection](qwen35-context256.md)
remains binding.

Required formatting and strict CPU/Metal all-target Clippy pass, with667CPU/
801Metal tests (45/324ignored). Three Python fixture controls pass. Explicit
release tiny CPU and Metal graph checks each pass3,675,072comparisons with a
maximum gate ratio0.153; state/partition, head substitution, failure/cancel/retry,
Metal inventory/poison/repair and warmed allocation controls pass. Native61 passes
three arm64 compile/export/link/package slices with46exports each and Mac CPU/
Metal C execution smokes. Physical iPhone execution remains deferred.

The matched performance scope uses an unchanged standalone probe against old
native60 and fresh native61 providers, identical opt3/unwind flags and source
weights. Two reversed4B pairs measure completed hybrid forward on17raw tokens at
offset37; two9B cohorts measure scalar one-token execution diagnostically. Each
process has3warmups and10measured repetitions, with reset, comparisons and logging
outside the timer. Full output bits, completed work, owner ledger, native lifetime
physical peak and a live conservative owned-group plus collector charge guard
are recorded. Loading/upload/binding are separate. The child allocator is
explicitly `MallocLargeCache=0`, with Nano/Aggressive overrides absent; application
defaults do not change. File-cache residency, paging absence, tails, sustained
performance, cross-engine/model/backend ratios and1% overhead are unqualified.

All six cohorts pass on an Apple M1 Pro MacBookPro18,3 with16GiB RAM, retaining
18warmups and60measured samples. The4B medians are3.074→3.078s (+0.13%) and
3.053→2.978s (−2.43%), with IQRs0.050/0.048s and0.025/0.023s respectively.
All four full17-row outputs match bitwise, and the owner ledger differs only by
16model metadata bytes. No repeated5% regression is observed. The9B medians are
15.480/15.546s, with IQRs0.086/0.173s; native peaks are5,309,222,968/
5,308,485,688bytes. Both outputs match the explicit9B diagnostic bitwise. These
are scalar raw-input baselines, with no cross-model/backend ratio or speedup claim.

New4B one/context17 captures match all1,051array descriptors and complete body
bits in each original accepted capture. This transfers those scoped independent
comparisons (32.135M/32.667M values), keeping the old gate and private ABI evidence
intact. New memory accounting explicitly adds16bytes. Native failure/recovery,
partition and zero-allocation reports match after removing timing values;16damaged
report controls reject. The regression reader initially assumed scalar reset
allocations; it now checks the original three zero counters. Raw captures were
unchanged. No deeper-context or broader-model qualification follows.

Initial compile errors (helper visibility and a Metal-only test accessor) and a
lexical JSON-number comparison failure were corrected without changing the
fixture or tolerance. Generated-cache cleanups preserve pinned native outputs and
all baseline link dependencies; missed initial headroom targets and a sandbox
process-inspection failure before mutation remain recorded. A later audited
cleanup frees1.36GB with490protected files unchanged; the2GiB guard holds.

[Contract](../artifacts/qwen35-9b-head-2026-11-09/acceptance-contract.json),
[required checks](../artifacts/qwen35-9b-head-2026-11-09/required-checks-1/checks.json),
[performance contract](../artifacts/qwen35-9b-head-2026-11-09/performance-contract-1.json),
[ABI contract](../artifacts/qwen35-9b-head-2026-11-09/four-b-abi-contract.json),
[measurements](../artifacts/qwen35-9b-head-2026-11-09/performance-summary-1.json),
[one-input regression](../artifacts/qwen35-9b-head-2026-11-09/four-b-regression-one.json),
[context17 regression](../artifacts/qwen35-9b-head-2026-11-09/four-b-regression-context17.json).

[Checks/review/commit](../artifacts/qwen35-9b-head-2026-11-09/publication-checks.json),
[Astra](../artifacts/qwen35-9b-head-2026-11-09/astra-review.json),
[next scope](../artifacts/qwen35-9b-head-2026-11-09/next-scope-analysis.json).
