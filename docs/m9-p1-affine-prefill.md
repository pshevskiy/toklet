# M9 P1 affine prefill comparison contract

Frozen 5 November 2026 before candidate implementation. The Mac P0
counter results favor reuse in affine projection. At the 1,024-token
workload, the median 128-token chunk has 738.912 ms of layer affine
projection intervals and 13.066 ms of normalization intervals.
The medians pool eight prefill offsets across three profile trials.
[Baseline](m9-mac-baseline.md).

Use four activation rows per SIMD output-row group for affine8, group64,
F16 scale/bias metadata and Half destinations. Preserve each token's
column order, compensated FP32 additions, FMA residuals, SIMD reduction
width, Half rounding, strides and status. Mask tail rows. Keep the existing
kernel for decode, the final FP32 head and unsupported layouts or widths.
Select the candidate explicitly and account for its compiled pipeline.

The primary metric is uninstrumented completed `prefill_seconds` at
1,024 prompt tokens, chunk 128 and capacity 1,152, with the pinned
Qwen2.5 MLX8 checkpoint and token arrays. The ten measured P0 samples have
median 6.775373938 s, minimum 6.774008625 s, maximum 7.135271417 s and
median absolute deviation 0.000936667 s. All raw trials remain in the
baseline. Compare a fresh matched baseline and candidate in release on
the M1 Pro, with the same build settings, AC power, recorded thermal
observations and separate uninstrumented processes. Run three warmups
and ten measured requests per workload and mode.

For samples \(x_i\), dispersion is \(\mathrm{median}(|x_i-\mathrm{median}(x)|)\)
in the metric's units. Require the candidate median to be at most 95% of
the matched baseline median and the absolute improvement to exceed the
larger of the two median absolute deviations. Retain all raw samples and
ranges. Track 128- and 512-token prefill, decode time, total request time
and process peaks at all depths. Investigate control regressions exceeding
the larger median absolute deviation before P2.

Require bitwise agreement of complete baseline/candidate distributions and
state across all three workloads. Test independent affine fixtures, batches
1/3/4/5/7/8/9/128, strides, offsets, tails, bias, cancellation, midpoints,
overflow and guards. The screen and independent P2 reproduction retain the
existing quality, state, memory and scoped adoption gates.
