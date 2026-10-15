# M6 conditional I8 KV component experiment

The frozen I8 component passes arithmetic, format, ownership and memory checks.
Its completed-command median reduction repeats at 5.08%/5.56% for the small
shape and 3.20%/2.67% for the larger shape. The larger shape misses the frozen
5% component usefulness threshold in both collections. The small result is a
positive component result, without engine adoption qualification.

The component decision is to keep FP16 KV and decline production integration
of this mixed-benefit candidate. The measured existing FP16 maximum profiles fit
6 GiB; complete selected-context/application qualification is still pending.
No I8 cache-write, full-model loss/task, actual long-generation or end-to-end
performance gate has been run or waived. Those would be required before any I8
adoption. The isolated experiment is retained for reproducibility, with no cache
format/API/default change in the engine. Final KV retention remains part of the
complete M6 review.

Astra accepted code and evidence after byte-identical independent replay of all
116 events/6,579 bindings, 29 receipt source/contract/evidence pins and 20 exact
published copies, including cross-collection inputs, timing statistics, memory,
power endpoints and protected CPU binaries. Declining an optional candidate
before adoption qualification is valid; the positive small component result is
preserved. This is not a demonstrated end-to-end failure or a new frozen
all-shapes rejection rule. Final FP16 retention and M6 closure remain pending.

## Frozen representation and execution

The [pre-collection contract](../../benchmarks/m6-kv-component-v1/manifest.json)
was committed as `5f2940e`. Version
`toklet-i8-kv-symmetric-half-scale-rne-g32-component-v1` uses separate K/V in
logical `[kv_heads, capacity, head_dim]` order. Adjacent signed two's-complement
codes occupy the low/high bytes of a `u16`; scales are positive Half values for
each 32-feature group. Scale is `maxabs/127`, rounded once to nearest-even Half.
Codes divide the exact original Half value by that stored scale in FP32, round
to nearest-even integers and clamp to [-127,127]. All-zero groups use positive
zero scales/codes; a nonzero underflowed scale becomes Half bits 1. Reserved
-128, negative/nonfinite scales and nonzero codes with zero scale are invalid.

Reconstruction multiplies FP32 code by the exact stored Half scale converted to
FP32, with no extra Half cast. With scale 1, independently packed words
`0100 02ff 7ffe 0081` reconstruct `[0,1,-1,2,-2,127,-127,0]`. Exact GPU bit
witnesses include canonical zero, minimum Half subnormal and maximum finite
Half scales with signed extremes. D64/D128 rows have complete groups/pairs;
partial groups and byte tails are excluded by the frozen component geometries.

All layers have distinct resident allocations and deterministic Half Q/K/V
contents. The small shape is L24/Q14/KV2/D64; larger is L28/Q12/KV2/D128, both at
32,768 positions. Original Half inputs feed both formats. FP16 uses the unchanged
production Online decode kernel. I8 copies its recurrence, changing only K/V
loads, with the same queried execution width and Safe/Precise Metal 2.3 policy.
The timed trusted specialization consumes CPU-validated immutable buffers; a
separate checked diagnostic specialization tests malformed inputs. Four
pipelines cover trusted/checked attention and reconstruction witnesses.

Each sample encodes one serial compute encoder per layer into one command.
Every layer overwrites the same tracked output allocation, establishing a write
dependency in both formats. Resource owners outlive one completed wait, and CPU
mapped access occurs only afterwards. Status atomics are not the serialization
mechanism. No model weights, intervening engine kernels or cross-layer
activation propagation are represented. No full CPU cache clone is retained.

## Checks and ownership

Eleven portable controller/build checks pass, along with isolated formatting,
strict all-target Metal Clippy and four packing tests. Astra's build preflight
independently passes eleven tests and 6,565 source/build/contract/binary bindings.
The exact release build record is
`845b65158b5b3cbe4f89a2e0e2fc7a4a4a766bbdca7bd4bf6c9495f4dbcad107`;
the copied executable is
`94f9e7e175bc33f89dacc263223489032c55e7b9e3d64be9f4ba0af1c758c80d`.
The fresh isolated snapshot preserves the reviewed baseline source and injects
only the probe module/assets. Root Cargo/src and protected CPU binaries remain
unchanged. Build commands, compiler/environment and complete source closures
are saved in the [qualification receipt](../../benchmarks/results/2026-10-15-m6-kv-component-qualification-v1.json).

Each native worker checks 256 exact FP32 reconstruction words, two attention
fixtures in all three paths and seven malformed-input/status cases. Prepared
input/guard hashes are checked before restoration. Independent FP64 attention
checks every layer for FP16, trusted I8 and checked I8 against the unchanged
`1e-4 + 1e-3*abs(expected)` envelope. Small checks 64,512 outputs per worker;
larger checks 129,024. Maximum arithmetic errors are 0.0000164857/0.0000297646.

Quantization error is separate from that arithmetic gate. Maximum K/V
reconstruction error is 0.0078582764 for both shapes; pooled RMS is
0.0043425015/0.0043427088, with zero clamped values. Against original-input FP64
attention, maximum output differences are 0.0006291938/0.0004785066 and RMS
0.0000975126/0.0000960276. These are synthetic observations, not model quality.

The parent owns a fresh process group, bounded stdout/stderr/events/timeouts,
and records both worker and driver native lifetime charges before acknowledging
each event. Final acknowledgement keeps allocations alive for the final memory
read. All four workers finish naturally with zero status, 29 events each,
unchanged inputs/source/binary and no cleanup/endpoint errors. Saved-data replay
covers all 116 events and 6,579 bindings. No sample is discarded.

## Measurements and limits

Apple M1 Pro, MacBookPro18,3, 16 GiB, macOS 26.6.2 build 25G83; optimized Rust
1.90.0. Each separate worker uses three alternating warmup pairs followed by ten
measured pairs, alternating FP16-first/I8-first. Medians, MAD and full ranges
below are seconds for the completed GPU command. Host completion and encoding
statistics and every raw sample are also preserved.

| Shape/run | FP16 median / MAD / range | I8 median / MAD / range | Median reduction |
| --- | --- | --- | ---: |
| Small primary | 1.145936 / .005818 / [1.136533,1.157709] | 1.087768 / .004735 / [1.071800,1.093599] | 5.0760% |
| Small confirmation | 1.152666 / .000539 / [1.144120,1.158659] | 1.088558 / .001553 / [1.084299,1.093156] | 5.5617% |
| Larger primary | 1.901361 / .009354 / [1.862348,1.917730] | 1.840526 / .004717 / [1.798356,1.852345] | 3.1995% |
| Larger confirmation | 1.894915 / .006454 / [1.865205,1.910197] | 1.844345 / .004057 / [1.823866,1.849375] | 2.6688% |

Observed FP16/I8 ranges are disjoint in every collection. This does not make the
larger reductions meet 5%, establish tail behavior, or turn component gains into
end-to-end gains. Power endpoints report AC attached, AC low-power mode off,
unchanged policy, and no recorded thermal/performance warning. Endpoint reports
are not continuous thermal, energy or system-pressure measurements.

| Shape | Simultaneous component admission bytes | Maximum joint native lifetime bytes |
| --- | ---: | ---: |
| Small | 1,221,914,718 | 660,883,232 |
| Larger | 2,044,053,222 | 1,482,541,288 |

Admission counts simultaneous FP16/I8/scales/queries/guards across all layers,
shared output/status, bounded host oracle/packing storage and framework/driver
allowances. Native charges can omit clean/shared pages. I8 payload is 53.125% of
FP16: 204 MiB versus 384 MiB small; 476 MiB versus 896 MiB larger. Simultaneous
comparison KV is 588/1,372 MiB. No model weights are loaded here, so these are
not full-engine/application memory totals. Root generation continues to use
FP16 KV; full M6 remains open.
