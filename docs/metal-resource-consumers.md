# Runtime resource-plan consumers

`toklet plan --model PATH` explicitly initializes the Metal/MPS runtime and its
precise pipelines, then prints the shared resource plan. This is separate from
`toklet inspect`, which keeps its CPU-only metadata contract. No fallback to
scalar execution or metadata-only reporting occurs when runtime Metal is absent.

The command accepts session capacity/chunk/offset, an optional load-context
ceiling, independent exact load/session budgets, normalization, readback and
attention mode. Defaults are capacity512, chunk=min(128,capacity), offset0,
load context=offset+capacity,6GiB budgets, Compensated/FullLogits/Reference.
Projection is fixed to `MlxAffine`. Every selected value is in the runtime report.
Run `toklet plan --help` for flags. A completed plan exits0 even when fit flags
are false; malformed arguments exit2 and engine/build/output failures exit1.
A closed output pipe exits0, matching metadata inspection.

The additive ABI-v1 `toklet_mlx_plan_create` takes
`TokletRuntimePlanningOptions`:56 bytes with exact load/session budgets at
C offsets40/48. Every version, size, mode and reserved field is checked before
runtime initialization. Context, capacity, chunk and offset must satisfy the
engine bounds. Load budget is positive; a zero session budget yields fit=false.
Budgets are passed unchanged, without the ordinary20KiB model/session bridge
deduction. Normalization values0/1 select Original/Compensated, readback0/1
FullLogits/GreedyToken and attention0/1 Reference/Online. Unsupported family/mode
combinations reject explicitly.

The function returns the existing immutable `TokletCompatibility` owner with the
separate runtime schema v1. It has `session_memory` and budget fit flags; it does
not contain metadata `status`/`reason`. Read and release with the existing
`toklet_mlx_inspect_json`/`toklet_mlx_inspect_release`. The64KiB bound, query and
transactional short-copy behavior,131,328-byte logical JSON allowance and
creating-thread rule remain unchanged. A valid output handle is cleared on
failure. Budget-fit failures return `TOKLET_OK` with a report; invalid arguments,
unavailable modes, rejected artifacts and runtime failures have distinct errors.

Both consumers preserve the [planner's limits](metal-resource-planning.md):
driver/pipeline setup allocates, numerical owners are not created, overlapping
admission totals must not be added together, and fit flags reserve no memory.
Actual loading revalidates payloads and budgets. These historical reports do not
qualify numerical quality, Chat, performance or a physical device.

## Verification

Portable tests cover malformed arguments, unavailable builds, the C record
layout and offsets,17 invalid option cases, pointer/path errors and handle
clearing. Explicit Mac tests compare native reports with the direct Rust planner
for four graph families, both normalization/readback modes and Qwen2 Online.
They also cover fit=false, wrong-thread copy/release, immutable snapshots after
artifact replacement and unsupported modes. CLI checks compare complete JSON
with the direct engine and cover invalid engine requests and closed pipes.

The [consumer receipt](../benchmarks/results/2026-10-24-metal-resource-consumers-v1.json)
records exact source/package identities and captures. Formatting, strict Clippy,
184 CPU/205 Metal-feature tests,20 native/4 CLI direct-report comparisons,
three Apple library slices with24 symbols, C links and strict Swift imports
pass. Both readbacks of the pinned real Qwen3 CLI report equal the preceding
resource receipt. Initial test-literal Clippy and CLI type-inference failures
are retained with their corrections. The Swift/app runtime
consumer and receipt-based qualification remain open; ordinary app loading still
uses its metadata preflight and actual constructors' admission checks.
