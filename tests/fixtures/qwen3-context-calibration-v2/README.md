# Experimental CPU calibration capture metadata

[manifest.json](manifest.json) freezes the first complete CPU-only capture of
the [separate C512 calibration inputs](../../../benchmarks/qwen3-context-candidate-v2/README.md).
The second cold capture has identical raw arrays/records; run-specific memory
observations need not match. This is extraction integrity, not adopted numerical
qualification. Historical C512 failures remain rejected.

Raw files are outside Git in
`artifacts/m8-qwen3-context-calibration-checks-v1/capture-a` and `capture-b`.
Each set has12 graphs/756 arrays/768,983,040 values/2,569,469,952 bytes. Shapes,
dtypes, offsets and full/array hashes are in the manifest. Returned nonfinite
arrays and partial writes have explicit retention controls. The unchanged graph
cannot expose states it rejects internally before returning.

The capture covers only the six calibration cases. No validation token array is
read, no native graph runs, and no gate is derived. Selected layers0/1/27 do not
qualify unobserved layers. Full source parity, original quality workloads and
larger model-context/device requirements remain open. See
[the report](../../../docs/qwen3-context-calibration.md) and
[public evidence](../../../benchmarks/results/2026-10-27-qwen3-context-calibration-v2.json).
