# Scalar attention expectations for the FP64 matrix-dot reference candidate

Expected values were prepared before the new fast-reference files existed.
The compact extraction preserves the accepted raw source identities and offsets;
it does not compute expected values with the candidate. See the
[experiment record](../../../docs/gemma-reference-attention.md).

`manifest.json`:34549 bytes, SHA256
`36365f416c5f9572bfad7d6ff42c51760e43085025857648905350f86924edd8`.
`arrays.bin`:5597808 bytes, SHA256
`f7a13f39536c816d566a5caf167b6f439f11b59b34ae206561b93d5ac28a1b7f`.
There are48 complete output arrays: scalar scores, masked probabilities and
contexts for eight cases under mathematical-FP64 and staged-FP32 contracts.
The48 input descriptors authenticate deterministic rational/dyadic recipes;
portable tests reconstruct the exact original input bytes.

The full pre-target capture has96 arrays/38356944 bytes, SHA256
`e11448fa8c911f0360bb8d9eeaa91fc3665477f96e241ed2532d360cc496bcb0`.
Its34725-byte manifest SHA256 is
`063f23894be36dc435ca1c68f11b95470471b665888437723c402a1564ee1771`.
Two cold preparations match. Independent sequential-dot and input-recipe replay
checks932968 values with maximum gate fraction8.216144307792186e-10; seven
coherent controls pass. The pre-target receipt is
`artifacts/m8-gemma-reference-attention-oracle-v1/checks.json`,137901 bytes,
SHA256 `3312198392db67774e844e01bd6fb7acc37d29050c24ac34d96ae73e7283519e`.

Cases include D256/D512, full/localW512, key origin509/query origin1020,
an8+1 query tail, W4, translated origins, single keys, mixed-sign cancellation
and near-tied scores. Diagnostic scores include masked columns; probabilities
are zero outside visible intervals. This is single-KV-head reference evidence,
with no production kernel, cached execution, model quality or device-fit claim.
