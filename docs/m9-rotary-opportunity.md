# M9 R1 rotary reuse decision

Defer the phase table for the pinned Qwen2.5 MLX8 Mac workload. Retabulating
all 75 P0 traces by absolute position and trial gives a maximum descriptive
rotary ratio of 0.123% of 128-token prefill and 1.721% of decode. The
denominators are average chunk/token time derived from uninstrumented depth-level median rates from
separate workers. These ratios identify the observed cost scale.

The 1,152-position, width-64 layout would need 294,912 payload bytes for
one FP32 sine and cosine per pair and position, shared across query and
KV heads. The observed work does not justify adding this retained table
for the current workload. Other configurations need their own evidence.

The four sampled 1,024-depth decode positions give attention medians
37.14, 38.33, 39.49 and 41.65 ms, compared with roughly 0.48 ms of rotary.
Those position-specific observations support a separate split-KV
investigation. Its numerical and performance gates remain open.
[Baseline](m9-mac-baseline.md).
[Position-level evidence](../benchmarks/results/2026-11-05-m9-rotary-v1.json).
