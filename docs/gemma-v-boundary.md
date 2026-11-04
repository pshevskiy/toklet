# Same-input Gemma4 V projection and normalization witness

The targeted witness passes both local boundaries and exact cache transfer.
The full-sequence [1,025-position numerical rejection](gemma-hybrid1025.md)
remains unchanged. This is local evidence, without whole-graph, app or phone
qualification.

A single 651-token pass captures the complete normalized projection input,
raw V projection and normalized V row at owner13/position433 and owners12/14
at650. It reuses the existing test-only producer failure boundary; no numerical
production code changes. Each failed attempt preserves the committed prefix.
Successful replay commits normally and its retained V bytes match the captured
normalized rows and the earlier wrap capture exactly.

The independent checker projects the original packed weights from the actual
native normalized input, then separately normalizes the actual native raw V row.
It preserves the frozen FP64 reductions, stored-FP32 epsilon, staged boundaries
and `1e-5 + 1e-4*abs(reference)` gate. Both operations pass all 2,048 values per
contract. Maximum mathematical ratio is 0.000525846; staged numerical differences
are zero. The observation does not identify the earlier full-sequence cause.

Twelve records occupy 30,720 bytes. Native peak RSS is 3,033,071,616 bytes and
physical footprint 3,027,110,720 bytes; replay peaks at 149,241,856 bytes. Source,
runtime, binary and input guards, exact prior-row agreement and clean owned
exits pass under the 6 GiB policy. Three cold controls check the frozen norm,
known projection, coherent numerical/cache corruption and incomplete framing.
Formatting, strict default/Metal Clippy, 550 default/658 Metal tests and release
build pass. Astra review is tracked in the evidence.

The next task is the [four-token original-BF16 source experiment](gemma-original-quality-next.md).
These results support leaving the sampled local operations and cache transfer
unchanged. The prior mathematical-logit and staged-KV rejections remain open;
there is no tolerance change or numerical promotion.
[Evidence](../benchmarks/results/2026-11-04-gemma-v-boundary-v1.json).
