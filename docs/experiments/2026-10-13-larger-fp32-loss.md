# Larger-model FP32 held-out reference

The preselected Qwen2.5-1.5B checkpoint completed all 10,000 frozen held-out
targets. Mean negative log likelihood is **2.485958031019153**, total
24,859.580310191533. This is the matched FP32 baseline for later larger-model
quality evaluation. It does not qualify GPU execution or the performance release.

The accepted CPU row scorer used one resident FP32 model, existing tiled CPU
arithmetic, chunks of eight, and caller-owned all-row output. Ten independent
blocks contain at most 1,025 tokens with one-token overlap; all targets are
scored once. The final token is committed before each block completes. Four
short single/batch probe workers and the full worker exited successfully, with
no cleanup or endpoint identity errors. Short batch/single losses agree exactly.
The smaller checkpoint's probe losses are retained as diagnostics, not as gates
for the larger checkpoint.

The complete collection used a frozen regular-file source snapshot at revision
58dc39a, after independent larger CPU numerical qualification. Only immutable
model and old probe artifact trees were linked. The collector's build field
describes the original isolated release binary, copied unchanged into the
snapshot; no build took place inside the snapshot. This allowed separate Metal
source work without changing identities used by the running reference. No GPU
qualification or performance measurement overlapped the full reference worker.

Allocation admission is 6,370,269,388 bytes. Maximum sampled worker-plus-driver
lifetime physical charge is 6,290,069,216 bytes, below 6,442,450,944. Native
charge can omit clean/shared pages, so it complements allocation admission.
This is reference collection memory evidence, not a performance measurement.

The [binding receipt](../../benchmarks/results/2026-10-13-larger-fp32-loss-qualification-v1.json)
hashes the summary, raw events, stderr, source snapshot and independent numerical
qualification. Astra independently replayed all 11,306 raw events, target labels
and positions, 1,260 full-run chunks and ten completed blocks, and recomputed
the mean with `fsum`. All five worker exits and unchanged identities passed.

Matched larger GPU loss and 100-task quality, end-to-end comparisons and sustained
qualification remain pending. The frozen 0.05-nat loss and two-percentage-point
task gates remain unchanged. The earlier 0.5B Q4 quality rejection is unchanged.
