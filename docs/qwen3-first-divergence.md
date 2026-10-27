# Qwen3 first projection rounding divergence

The [layer27 diagnosis](qwen3-projection-localization.md) showed inherited input
differences rather than extra local native dot error. This final bounded check
uses existing layer0 captures, before any upstream divergence in the selected
row. It changes no kernel or precision policy.

A frozen plan selects the first two native/canonical-FP32 differing features
and two bit-identical controls in each Q/K/V projection at position0. It retains
authenticated full embedding and input-normalization rows. Independent readback
verifies all1024 values in both rows against each CPU reference before dots.
The selected original weight rows are identical under the exporter's separate
FP64 multiply/add then FP32 cast and the native fused FP32 reconstruction.

At all12 coordinates, the exact rational dot is identical across the three
input profiles. Native and CPU-FP64 saved outputs match its correctly rounded
Half result. At the six selected differing coordinates, CPU-FP32 lands on an
adjacent Half value instead. All six controls agree, and all original local
canonical gates pass. Exact signed distances to adjacent Half midpoints are
retained. This establishes an initial reduction/store-profile divergence with
identical inputs. It does not prove these six differences alone cause every
later C512 discrepancy or that all unobserved native operations are exact.

The stopping rule is met: keep the kernels unchanged and stop per-layer
localization. The historical [C512 numerical rejection](qwen3-context512-native.md)
remains rejected. Next is a separately versioned C512 precision-validation
proposal with independent calibration and untouched validation inputs. Native
errors cannot determine new tolerances. Source parity, unquantized quality and
the complete held-out workload remain separate requirements before adoption;
no workload may be clipped to fit512.

All33 cold tests pass, including11 projection controls. The previous14-coordinate
diagnosis is replayed unchanged. The new reader physical/RSS peaks are
287,343,528/268,206,080 bytes; the legacy replay peaks are
285,049,792/308,035,584 bytes, within512MiB. All three diagnostic groups and the
publication group close. Independent publication uses a second, bisection-based
rounder to check all36 new dots and their exact midpoint distances.
No new GPU execution or phone qualification is claimed.

See [complete public evidence](../benchmarks/results/2026-10-27-qwen3-first-divergence-v1.json).
