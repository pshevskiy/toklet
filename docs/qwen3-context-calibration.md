# Experimental Qwen3 CPU calibration capture

The [separate input freeze](../benchmarks/qwen3-context-candidate-v2/README.md)
defines an experimental precision proposal after the
[first-layer rounding diagnosis](qwen3-first-divergence.md). This task captures
CPU calibration references only. It derives no gates, opens no validation token
array and executes no native graph. Historical C512 failures remain rejected.

Six calibration inputs contain1344 tokens: two authored texts with nested
prefixes32/128/512. Each runs through the unchanged independent CPU graph with
FP32 and separate FP64 arithmetic, original FP32 reconstructed affine weights,
Half activations/KV and FP32 logits. These are arithmetic witnesses for the same
quantized checkpoint, not an unquantized quality baseline or upstream MLX parity.

Each run returns12 complete graphs, with63 states per graph:20 in each selected
layer0/1/27 plus embedding, final norm and logits. Unlike the native57-state
interface, CPU capture includes attention scores and probabilities. Their
presence does not add them to the proposal's57-state gate-pooling map.

One model owner and one returned graph are live at a time. Largest graph storage
is508,297,216 bytes; complete capture storage is2,569,469,952 bytes per run.
The established conservative memory ledger admits5,214,106,290 bytes within6GiB,
including source/tied-table ownership, graph, BLAS temporaries, frameworks and
comparison mappings. Two complete runs plus a2GiB reserve are admitted against
free disk space before execution. Large raw binaries stay outside Git.

Active array descriptors precede writes and track intended/actual bytes and
prefix hashes. Returned nonfinite arrays are retained before rejection, with
bounded stripe checks. Per-attempt journals and final partial manifests preserve
earlier captures, failures and owner closure. A failure inside the unchanged
forward cannot expose states it never returned; no such capture is claimed.

Independent cold readback streams every saved value and authenticates each
array/full-file hash, exact schema/shape/extent, complete case/attempt coverage
and descriptor/path endpoints. Two cold runs must have identical complete raw
record sets. Repetition establishes reproducibility and extraction integrity,
not cross-precision numerical acceptance, source parity or product quality.

Next derive experimental gates only from these calibration arrays and the
frozen formula, then freeze/review the gate artifact before numerically opening
validation. Envelope-ceiling/adoption review, exact native state checks, separate
source parity and complete unquantized/held-out quality remain required. No app,
capacity, precision, model-window or phone qualification changes.

## Capture result, 27 October 2026

Both cold capture sets have identical12 raw record sets. Each contains756 arrays,
768,983,040 scalar values and2,569,469,952 bytes. Both complete cold readbacks
pass schema, hashes, finite-value and source-endpoint checks. A separate
standard-library publication reader independently verifies every raw array and
full-file hash against frozen geometry. All13 cold tests pass, including seven
calibration capture/reader controls. Five execution groups and the publication
group close.

External physical/RSS peaks on the same M1 Pro/16GiB Mac are
1,275,463,936/2,188,410,880 and1,245,940,064/2,190,180,352 bytes, within6GiB.
Reader peaks292,684,760/274,661,376 and292,930,472/269,058,048 bytes fit512MiB.
These measurements qualify extraction/readback resource use only. No numerical
cross-precision envelope, quality or source-parity result is inferred.

See [frozen metadata](../tests/fixtures/qwen3-context-calibration-v2/README.md)
and [public evidence](../benchmarks/results/2026-10-27-qwen3-context-calibration-v2.json).
Experimental gate derivation/review remains next; validation stays unopened.
