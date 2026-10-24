# Resource planning for the native factory

Runtime C options now accept attention value2, meaning the existing native model
factory's choice. Values0/1 still select explicit Reference/Online. The report
contains the resolved mode; normalization and readback remain explicit.
`inspect_mlx_with_native_attention` exposes the same choice to Rust callers.

The resolver runs after source/config/geometry admission. Both inspection and the
actual factory call one exhaustive helper: Qwen2 and legacy non-MLX models use
Online; the currently admitted Llama, Mistral and Qwen3 graphs use Reference.
Adding a graph enum requires an explicit decision in that helper. Unknown graphs
and unqualified explicit mode combinations retain their existing rejection rules.
Swift and other clients need no duplicate architecture allowlist for this choice.
No normalization, rotary, cache, numerical kernel or memory formula changes.

`MODEL_SESSION_BRIDGE_BYTES` and `TOKLET_MODEL_SESSION_BRIDGE_BYTES` expose the
existing20,480-byte logical native model/session-owner allowance. Ordinary
`TokletOptions.memory_limit_bytes` includes this allowance; the factory subtracts
it with checked arithmetic. Runtime inspection continues to take exact numerical
budgets without deduction. A client planning an ordinary native load/session can
check that its whole budget exceeds the shared allowance, then subtract it for
both runtime budgets. Do not subtract it again from the actual factory options.

This allowance is separate from bounded JSON/text/application ownership. It is
not a measured peak or enforced reservation. Budgets at or below the allowance
remain invalid; allowance+1 normalizes to one numerical byte and cannot admit a
model merely because the arithmetic is valid. A captured report never skips
actual loading and session checks.

## Evidence and scope

Portable checks exercise allowance−1, allowance and allowance+1, output clearing,
unknown policy values and all current exhaustive factory choices. The explicit
native test compares four families with actual C model/session owners, resident
weight counts, frozen parent allocation reports and exact/minus-one load/session
budgets. The C caller also compares runtime admission plus the header allowance
with actual factory session state, using the resolved mode. Existing explicit
runtime consumer and frozen42-case resource regressions remain separate checks.

The [policy receipt](../benchmarks/results/2026-10-24-native-resource-policy-v1.json)
records exact results and source/package identities. Formatting, strict Clippy,
185 CPU/207 Metal-feature tests, five explicit release GPU tests and all three
Apple library slices pass. C callers link all SDKs and run four families on Mac;
strict Swift importer/metadata regressions and both real Qwen3 CLI reports pass. Native policy selection is
an app-preflight prerequisite; the Swift/app runtime consumer remains open.
No broader graph, text, task-quality, performance or physical-iPhone support is
implied by this change.
