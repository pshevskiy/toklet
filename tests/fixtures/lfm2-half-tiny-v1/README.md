# Independent tiny LFM2 Half graph reference

Four cases,63 complete groups per profile,504 descriptors and2,212,480 raw bytes
for H256/I256/V64/Q4/KV2/D64/C12, Conv→Attention→Conv. The unchanged original
Q4/G64/BF16 container/config live in the sibling lfm2-scalar-v1 fixture. Their
identity is normative; absolute provenance paths in JSON are not runtime paths.

MAIN/WIDE use uniform exact integer-dyadic affine sums and direct Half/FP32
rounding on their own Half inputs. Qualified direct-gain RMS and raw Conv leaf
oracles are reused; all other graph stores and attention masks are explicit.
Twelve controls, exact four-file export/replay and independent complete saved
array/state/weight/gate/component reconstruction pass before native outcomes.

Source snapshots describe mathematical tokenwise history/KV state. Compare only
actually observed native chunk boundaries and active KV prefixes. Source inactive
+0 does not demand native tail clearing. Raw future scores are retained but only
visible score columns participate in earlier-prefix causal invariance.

This is a source reference, not native graph, real-model, app, context, memory-fit,
performance or mobile qualification. See the
[reference report](../../../docs/lfm2-half-reference.md).
