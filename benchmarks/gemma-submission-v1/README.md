# Gemma submission workload

Public Rust prefix-sum review:384 rendered input tokens and128 expected output
IDs,capacity512/chunk32. This is the unchanged frozen O1 workload; no model or
private input is included. Raw SHA256:
`de4ebba10f7bcf6a721c973da65ca8c0a246d04cf7471e159a18e460b4bcc4fd`.

The ordinary benchmark authenticates this descriptor and the actual codec IDs,
then exports `input_ids.u32le` before timing. Expected IDs/state come from the
previously qualified official BF16/G2 workload,not this optimization. Source
provenance is recorded in `docs/gemma-workload-identity.md`.
