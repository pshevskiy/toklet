# Caller-owned CPU all-row logits

The CPU session exposes an explicit diagnostic output of shape
`[next_chunk_rows, vocabulary]`. The caller supplies and separately admits this
buffer. The ordinary API still computes only the final distribution. Both use
the same immutable FP32 weights, session KV, fixed scratch, and existing
scalar/tiled FP64 projection arithmetic with FP32 storage.

The entire output shape is checked before cache mutation. A successful all-row
projection publishes only its final row to the session and then commits the
chunk. A failure can dirty the caller's output and uncommitted cache tail but
preserves the valid committed prefix. Zero pending tokens require empty output.
This enables a bounded larger-model reference scorer without a full diagnostic
trace or another live model copy; it is not quality or performance qualification.

The ordinary session test compares 84 complete distributions against independent
full-prefix execution across scalar/tiled, offsets 0/5, chunks 1/3/8 and tails.
It also covers malformed output before mutation, empty work, switching back to
ordinary execution, unchanged memory and zero Rust allocations in each scored
chunk. Both layer and final-projection failures preserve committed state.

Explicit pinned Qwen checks compare 36 all-row distributions per CPU projection
and retain 47 public distribution comparisons per projection. Equality is exact
numerical equality; this does not certify signed-zero bit identity. Native memory
and commands are retained in the separate scalar/tiled reports. Capacity 2048 is
admitted, while real-model execution remains short; full capacity execution is
covered by the small model.

The [task binding](../../benchmarks/results/2026-10-13-cpu-all-row-logits-v1.json)
records source, isolated binaries, checks, native reports and protected executable
identities. Formatting, strict Clippy and ordinary tests pass with both default
and Metal features. Astra accepted code and evidence before commit. Larger-model
actual admission, matched quality and all M5 release gates remain pending.
