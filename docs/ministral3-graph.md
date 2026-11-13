# Ministral packed graph and raw CLI diagnostic

This milestone connects the selected modified mixed-affine text child to the
resident Metal graph and a token-ID CLI profile. The current full-model route
is numerically rejected: 78 of 244,809,728 profile coordinates fail the unchanged
gates. Required code checks and targeted localization pass. Astra approves
retention of the explicit diagnostic tooling, preserving this rejection. The ordinary inspector/loader remains at
`unknown_architecture`. Stock index rejection is unchanged.

The [frozen graph contract](../artifacts/ministral3-graph-2026-11-13/prospective-graph-contract-1.json)
and [source/conversion qualification](../artifacts/ministral3-conversion-2026-11-13/publication-checks.json)
identify the exact candidate and acceptance policy. Neither a successful load
nor a diagnostic request qualifies codec, full-context numerics, task quality,
all-workload memory fit, request performance, O1 overhead or managed adoption.

## Execution and ownership

The explicit source opener validates the nested selected configuration, exact
mixed affine4/8 group64 recipe, 602 physical tensors and 236 logical owners.
Physical `language_model.model.*` names remain intact; checked logical aliases
are resolved once during loading. FP32 model expansion is rejected.

Hidden width is 3,072, query width 4,096 and KV width 1,024. Each of the 26 layers
uses query rotary on Q, key rotary on K and unchanged V, followed by Reference
GQA and an output projection back to hidden width. The packed embedding is also
the output head. No second full head or model copy is retained.

One immutable positional owner is shared by the model's layers and sessions.
Load planning counts temporary CPU tables, guarded resident GPU tables and
pipeline allowance; session and capacity-transition plans count the resident
owner once. The route requires MlxAffine projection, Compensated normalization,
Reference attention and at most 512 positions. Experimental tile4 and RMS-store
variants remain unavailable on this profile. Engine, setup/audit, compiler and
disk-floor policies remain unchanged.

## CLI and measurements

Use `--model-profile ministral3-diagnostic --backend metal
--metal-projection mlx-affine` with `--tokens` or `--token-file` and `--json`, or
use persistent `--interactive` token-ID requests. Default capacity is 128;
capacity 513 rejects before model access. Text, messages, tools and raw-text
formatting are rejected, including interactive requests. Tokenizers are never
loaded by this profile. EOS is token 2 unless `--no-eos` is selected; thinking
delimiters are not additional stop tokens.

Ready/done metadata names the diagnostic profile and explicitly labels Reference
attention. Existing load, conversion, input preparation, prefill/decode,
sampling, publication, request and logical-memory observations remain available.
The selected profile reports mixed affine4/8 storage, FP16 norms and group64.
Metal `compilation_seconds` exposes the existing completed setup clock on the
first completed request, with zero attributed to subsequent requests.
Compilation and conversion are components of loading and must not be added to
the load total. These observations do not establish observer overhead or a
cross-engine speedup.

## Validation scope

The independent tiny fixture uses hidden 64, query 256, KV 128, head dimension
128, mixed affine4/8 and a tied head. Its main/wide arrays precede native output.
Every stage must satisfy both original profile gates; tolerances are unchanged.
A separate high-position composition control detects query/key gain confusion
that admitted positions 0–511 cannot expose. This control widens no context
admission. All five release GPU controls pass, including zero Rust allocations
across eight warmed tiny decode steps. The 119-allocation debug-build failure
is retained; dependency inspection identifies debug-only method-encoding checks.
[Initial controls and retained debug failure](../artifacts/ministral3-graph-2026-11-13/tiny-controls-closure-1.json),
[release run](../artifacts/ministral3-graph-2026-11-13/tiny-gpu-release-1/result.json).

The actual full-model protocol covers five short-prefix cases, two chunk sizes,
reset replay and changed-prefix suffixes. It captures every coordinate of all
419 stages, with independently generated main/wide references, exact schedule
and causal-prefix checks, source/runtime identities and lossless raw arrays.
The boundary-offset case checks absolute-position wiring with a short KV prefix;
it is not a 512-token-prefix numerical qualification. All 22 runs, 9,218 saved
arrays, 7,123 exact schedule/reset checks and 419 causal-prefix checks complete.
The [complete numerical decision](../artifacts/ministral3-graph-2026-11-13/graph-comparison-1/manifest.json)
rejects the route. Capture clocks remain diagnostic. Performance cohorts are
blocked by this failure; full-prefix, codec, original 10,000-target/100-task
quality, fit, request and O1-overhead gates remain mandatory work.

The console functional run passes 19 requests across state/error, cancellation
and custom-stop controls, plus a cold bad-tokenizer/missing-weight check. Its
first prompt ID matches the independent references. Precision/mode/group,
positional-inclusive memory and phase clocks are validated; this ID agreement
does not override the numerical rejection. An earlier private manifest reader
stopped at a nested `arrays` field before starting a child. The corrected helper
passes synthetic and actual-prefix controls; both versions and the failure stay
retained. Natural EOS-2 termination remains unqualified.
[Console evidence](../artifacts/ministral3-graph-2026-11-13/console-functional-2/result.json).

Targeted native-input replay localizes the earliest failing boundary-offset
attention projection to differences already present in its input: all 3,072
output words of the selected row match the wide profile exactly. Sampled RMS matches its
main arithmetic profile; the rotary witness matches source-form FP32 table
arithmetic, with identical projected inputs in both references. These samples
show mixed rounding trajectories and identify no intrinsic defect in the
selected operations. Other failures remain unlocalized. Substituting native
inputs is diagnostic only and never replaces the independent full-graph gate.
All 78 failures remain rejected.
[Localization](../artifacts/ministral3-graph-2026-11-13/native-input-localization-output-1/manifest.json).

[Combined checks, Astra review and publication](../artifacts/ministral3-graph-2026-11-13/publication-checks.json).
