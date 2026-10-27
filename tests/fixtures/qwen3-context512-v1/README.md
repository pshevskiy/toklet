# Qwen3 C512 independent reference preparation

`manifest.json` and `comparison.json` are exact saved records from the first
cold CPU capture. The ignored raw files live in
`artifacts/m8-qwen3-long-reference-checks-v1/capture-a/`; an independently
launched second capture has identical complete binary/array identities.

**The numerical qualification gate fails:**7282 FP32-versus-FP64 scalar
comparisons exceed the unchanged short-context bounds, including3250 logits
per case. The full captures remain useful reference data. Causal prefixes,
masked zeros, old17-position comparisons and all raw argmax IDs pass; those
results do not waive numerical, held-out quality or device requirements.

See [the report](../../../docs/qwen3-context512-reference.md). Do not treat
`common_greedy_oracle_ready` as full-context numerical or quality qualification.
