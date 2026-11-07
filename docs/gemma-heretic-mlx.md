# Selected uncensored Gemma MLX format

G1.A adds an explicit cold recipe for
[`deadbydawn101/gemma-4-E2B-Heretic-Uncensored-mlx-4bit`](https://huggingface.co/deadbydawn101/gemma-4-E2B-Heretic-Uncensored-mlx-4bit/tree/cde01efcb11f8a12e88a953b752fc16062aa1486),
revision `cde01efcb11f8a12e88a953b752fc16062aa1486`. Required checks, local metadata
probes and Astra review pass. This is the format
step toward native execution, not execution or managed-model qualification.

The new `mlx::gemma4::uncensored::inspect` requires exact config/header byte
identities, the fixed container extent and explicit text-only policy. The
checkpoint has one additional null alias for its disabled MoE branch, redundant
consumer K/V tensors, and an unquantized PLE projection. Existing affine4/QAT/
official-BF16 inspectors remain separate and strict. No tensors are requantized.

| Partition | Physical tensors | Source payload bytes |
| --- | ---: | ---: |
| Selected text | 1,094 | 2,623,765,062 |
| Unused shared-KV projections/norms | 140 | 10,629,120 |
| Excluded audio/vision | 1,415 | 946,370,944 |

The plan names 277 affine4/group64/BF16 matrices and one original BF16 matrix
`per_layer_model_projection.weight`, shaped `[8960,1536]`. Its remaining text
owners are 262 BF16 vectors. These are source representation bytes, not allocated
GPU/session owners or measured process footprint. The typed recipe keeps the
mixed matrix explicit rather than describing the entire checkpoint as uniform Q4.

The API consumes bounded config/header slices; it reads no files or tensor
payload and grants no execution capability. The recipe names a single fixed
`model.safetensors`; no shard index is consumed. A later loader must authenticate
its actual consumed bytes against full-file SHA256
`d4a2995ce3fdafbbd4a3376bc8524a8d26bcbf0f51197701b4e21ab27faa37e1`,
validate coefficient values and admit all owners/load transients below 6 GiB.

The complete local source was independently rehashed before freezing fixtures.
Only the public config/header and a Python/struct inventory are bundled in
[metadata fixtures](../tests/fixtures/gemma4-heretic-metadata-v1/README.md).
Portable controls cover exact mixed layouts, original shared-KV geometry,
separate old-route rejection, altered source bytes/extents/modality policy,
graph-option changes, missing/unknown tensors and malformed spans/coverage.
Fresh610 CPU tests (33ignored),735 Metal-feature tests (289ignored), strict
all-targets Clippy in both configurations and formatting pass. The explicit
local-file probe reads only the length/header; its warm timings
measure read/hash/parse/plan and establish no inference or cold-storage gain.
Both independent release cohorts (three warmups plus ten samples each) pass,
with medians6.168/6.225ms on M1 Pro16GiB, AC power and nominal thermal state.
The metadata-only children report RSS peaks10.39/10.55MB and lifetime physical
peaks8.34/8.50MB through macOS `time -l`; these do not establish model fit.
[Instrument source](https://github.com/apple-oss-distributions/shell_cmds/blob/main/time/time.c).
The first hardware-query preflight was sandbox-denied before the test started;
the authorized retry and raw logs are retained. The final offline report also
extracts the existing physical-peak field missed by the initial capture parser.

Next: resident mixed-format loading and session metrics, then independent
numerical, state, memory, held-out quality and performance gates. Preserve the
modified high-precision checkpoint as a separate identity from both official
Gemma and its quantized derivative. Publisher refusal/KL measurements do not
replace this repository's frozen 10,000-target NLL and 100-case task gates.
[Source lineage](https://huggingface.co/p-e-w/gemma-4-E2B-it-heretic-ara),
[binding model gates](model-observability-plan.md).

[Checks/review/commit](../artifacts/gemma-heretic-mlx-2026-11-08/publication-checks.json),
[Predeclared contract](../artifacts/gemma-heretic-mlx-2026-11-08/acceptance-contract.json),
[independent inventory](../artifacts/gemma-heretic-mlx-2026-11-08/source-inventory.json),
[required checks](../artifacts/gemma-heretic-mlx-2026-11-08/required-checks-1/checks.json),
[cold samples/instruments](../artifacts/gemma-heretic-mlx-2026-11-08/cold-report.json).
