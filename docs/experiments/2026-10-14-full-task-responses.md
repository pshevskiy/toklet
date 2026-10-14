# Full frozen Half and larger FP32 task responses

All three collections complete the original100 cases: small Half, larger Half and
the bounded larger FP32 reference. Original40 code/30 prose/30 continuation IDs,
prompt IDs,256/32/16 caps, lowest-ID greedy ties, hidden EOS, empty state per case
and explicit reset remain unchanged. No numerical code, source model, tokenizer,
contract, task payload, quality gate or default changes during collection.

Actual Half metadata is Metal/HalfCompensated/Original/FullLogits with FP16 weights/
KV, capacity2048/chunk128. The larger FP32 reference is CPU/Tiled/OrderedFP64/
FullLogits with FP32 weights/KV, capacity562/chunk8 and no live tokenizer during
the model phase. Its tokenizer is instantiated only after owned model exit/reaping.
The previously qualified immutable workers and collectors are reused.

| Collection | Cases | Visible tokens | Raw events | Complete logical admission | Maximum sampled model-phase worker-plus-driver native lifetime charge |
| --- | ---: | ---: | ---: | ---: | ---: |
|Small Half|100|6,390|6,591|1,443,266,720 bytes|1,273,792,696 bytes|
|Larger Half|100|5,519|5,720|3,577,266,480 bytes|3,469,020,880 bytes|
|Larger FP32|100|5,514|5,716|6,343,713,560 bytes|6,306,092,840 bytes|

All admissions and charges fit6GiB. Counter sums are conservative lifetime samples,
not a shared physical total. FP32 decoder-phase logical admission is180,663,760bytes
with model residency zero. All three owned workers naturally exit0, cleanup/endpoints
pass and fresh process/group absence is verified before grading. All18,027 events
replay through full backend-aware admission, with exact producer inventories,
qualification-pinned executables and inputs, full state/caps/memory and independent
post-exit finalized text. Partial probes remain excluded.

Larger Half and FP32 visible token IDs match exactly in99/100 cases. The differing
code case emits231 versus226 tokens. This is consistency evidence only; no answer
repair, partial scoring or quality acceptance follows. Every collection explicitly
keeps quality completion false. The next task executes the literal code programs
in the qualified Linux window and grades all100 cases per model. Small comparison
uses its existing qualified frozen FP32 task reference; larger now has a complete
matched reference. These response timings are not application or benchmark results.

Source parent `efc73dd`; model/collector/compiled identities and replay checks are
bound by the [receipt](../../benchmarks/results/2026-10-14-full-task-responses-qualification-v1.json).
[Small Half](../../benchmarks/results/2026-10-14-small-half-tasks-full-v1.json),
[small raw](../../benchmarks/results/2026-10-14-small-half-tasks-full-v1.raw.jsonl),
[larger Half](../../benchmarks/results/2026-10-14-larger-half-tasks-full-v1.json),
[larger raw](../../benchmarks/results/2026-10-14-larger-half-tasks-full-v1.raw.jsonl),
[larger FP32](../../benchmarks/results/2026-10-14-larger-cpu-task-reference-full-v1.json)
and [FP32 raw](../../benchmarks/results/2026-10-14-larger-cpu-task-reference-full-v1.raw.jsonl)
retain complete native responses and exact inputs. No Rust/source change or unrelated
engine tests accompany this data task; use the accepted worker/collector checks plus
full targeted native replay. Documentation links/content and whitespace are checked.

Matched literal scores, repeated controlled/application comparisons and sustained
release evidence remain outstanding. Both frozen Q4 full-loss rejections still block
the explicit Q4 exit. M5 and defaults remain unchanged.
