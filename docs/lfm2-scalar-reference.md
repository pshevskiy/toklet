# Independent tiny LFM2 scalar reference

B4c1 supplies expected values for the next Rust hybrid composition task.
It does not execute or admit a Toklet LFM2 decoder. The synthetic original
container has18 Q4/group64/BF16 matrices, nine direct BF16 norms and two
`[256,3,1]` convolution owners:29 logical/65 physical entries. One packed table
serves embedding and output. Payload starts at byte6670, legally unaligned for
U32. No real checkpoint is downloaded.

The selected tiny graph is Conv→Attention→Conv, H256/I256/V64/Q4/KV2/D64,
capacity12, epsilonFP32(1e-5), ordinary theta1e6 and attention scale1/8. It uses
actual head64 and distinct KV heads and convolution histories. It is separate
from the strict16-layer real metadata adapter.

The independent stdlib source reuses generic packing/serialization helpers,
without calling another forward, normalization, attention or rotary graph.
Original affine values reconstruct to FP32; projection, norm and attention
reductions use FP64 mathematical sums and FP32 stores. Rotary, stable SiLU,
gating and residuals have explicit FP32 boundaries. Direct BF16 gains receive
no+1, embedding receives no scale, convolution has no SiLU and residuals have
no Gemma clamp. This is an independent mathematical oracle under a fixed gate,
not bit emulation of target reduction order, MLX trajectory or NativeHalf.

Four cases have lengths1,3,12,12 with the last case changing only the final ID.
They retain57 mathematical and six state groups each:252 groups/548,944 finite
FP32 values,2,195,776 array bytes. Source history tables record before/after raw
u suffixes for each row; KV snapshots identify each head and valid prefix.
Inactive source slots are+0. A target may retain stale inactive KV tails after
reset; comparisons must inspect active positions and exact within-engine
rollback separately. Raw scores retain future columns, while masked
probabilities are exact+0. Future chunk consumers map observed absolute key
columns rather than fabricating unavailable values.

The [frozen protocol](../tests/fixtures/lfm2-scalar-v1/protocol.json) predates target
outcomes and retains `1e-5 + 1e-4*abs(reference)` for every compared numeric stage.
Eight portable synthetic controls pass, including original ownership and
corruption, direct gain/store behavior, asymmetric taps/raw history, GQA masks,
rotary pairing/offset, causal corruption, projection and bounds. Two finite
source-only exports reproduce all six files byte for byte. Closed jobs and source
pins are recorded in the [result](../benchmarks/results/2026-10-29-lfm2-scalar-reference-v1.json).
The source scripts are Python-only; the already passing B4b Rust checks remain
current and need no repeat.

Astra reviews source and complete saved arrays before fixture retention. Rust
stages, causal partitions, physical state, compound publication/failure/reset
and warmed allocations remain the B4c2 task. Real payload, Half/GPU, full
context, app/text, iPhone and original adoption quality remain unqualified.
