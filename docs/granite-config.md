# Granite configuration admission

Status: required checks and Astra review pass; no actionable findings. This is source configuration support for the selected Granite 4.2 3B
track. Payload authentication, conversion, real-model execution and adoption
remain open.

The dense adapter accepts explicit `rope_parameters` with exactly `rope_type: "default"`
and numeric `rope_theta`. A concurrent legacy theta must match the
arbitrary-precision JSON declaration exactly before conversion to f32. Unknown
fields, other rotary modes and non-null scaling remain rejected. This reuses
the existing strict default-RoPE parser for Llama and Granite; other families'
admission does not change.

Granite exports may declare both `dtype` and `torch_dtype` only when both are the
same supported string (`float32`, `float16` or `bfloat16`). Source files are
read without modification. Declaration normalization does not convert tensors
or establish their actual storage precision.

The four explicit Granite scalars, untied head, SiLU/full causal attention,
split-half rotary layout, GQA mapping, affine plan/index validation and the
32,768 native context ceiling remain binding. No numerical loop, allocation
policy, resident weight format, session state or ABI changes. Metadata admission
alone does not establish memory fit: the original and modified BF16 bodies are
7,319,475,200 bytes, above the 6 GiB ceiling before other owners. The original
index's understated total remains rejected.

The [three source configurations](../tests/fixtures/granite-config-v1/manifest.json)
are byte-pinned metadata-only fixtures: official BF16, preferred Heretic BF16
source, and official affine4 reference. The raw Q4 config retains both identical
quantization declarations; ordinary source parsing owns that existing check.
The shared parser is exercised at the public source boundary with the small
frozen Granite graph in legacy, canonical-only and paired dialects, including
independent diagnostic arrays, nonzero offsets, cached chunks and reset/replay.
Negative cases cover unsupported/missing/ambiguous RoPE, pre-rounding numeric
conflicts, dtype conflicts/types and syntactic duplicate keys.

No performance gain or request-summary overhead qualification is claimed. These
cold metadata changes precede loading; full-model baseline, quality, memory,
state, request and overhead campaigns follow separately qualified bounded
conversion. The installed MLX mixed4/8 recipe must be explicit and independently
checked; the official global-Q4 artifact does not establish that recipe.

Local execution and full logs:
[preparation](../artifacts/granite-config-2026-11-12/preparation-1.json),
[bounded check plan](../artifacts/granite-config-2026-11-12/checks-plan-1.json).

The default-debuginfo Metal test build was stopped at the unchanged 2 GiB
compiler budget. Required feature checks are repeated with debug information
removed (`CARGO_PROFILE_DEV_DEBUG=0`, `CARGO_PROFILE_TEST_DEBUG=0`); assertions,
overflow checks and optimization level are unchanged. Earlier compiler-startup
counter-monitor failures remain failed and retained. The accounting rules have
no new tolerances; direct `cargo-clippy clippy` uses the installed checker with
the same original flags. These are check-build settings, not performance evidence.

A controlled 32 MiB owned child then executes `/bin/sleep`: its PID and birth
stamp stay unchanged, its native Mach-O UUID changes, and the reported task
lifetime footprint falls. The SDK declares that UUID in the first 16 bytes of
the same rusage-v4 response; the observed value matches the native Mach-O file.
The private check adapter permits a lower reported counter only with a changed,
nonzero executable UUID. It retains the larger per-birth RSS/physical maxima and
all historical concurrency charges. Same-image or unwitnessed resets, invalid
UUIDs and read/identity failures remain fail-closed. This sampling monitor still
cannot establish a continuous allocator cap or capture every short-lived peak.
Eight portable controls in normal/optimized Python and an owned live control
pass. [Proof](../artifacts/granite-config-2026-11-12/exec-counter-source-proof-1.json).

The 2 GiB compiler limit came from the local harness. After conservative
compiler/linker peaks exceeded it, the compiler-only verification cap is 3 GiB,
within the plan's 6 GiB Mac ceiling. The engine ceiling and 512 MiB audit/setup
budget are unchanged, with no full model loaded alongside compilation. All
failed attempts remain failed; successful checks record their actual cap and
build flags. [Compiler admission](../artifacts/granite-config-2026-11-12/compiler-admission-plan-3.json).

Required fmt and strict CPU/Metal Clippy pass; full suites pass 737 CPU and 883
Metal-enabled tests (52/340 explicit opt-in tests remain ignored). The dedicated
Granite GPU fixture executes separately and passes all three config dialects.
Across the eight frozen source cases, the CPU test checks 936 diagnostic arrays,
nonzero offsets and cached/reset schedules. Forty-four guarded CLI cases confirm
identical admitted reports for legacy, canonical-only and paired declarations,
both backends, and the original budget/context/conflict rejection behavior.
Eight private monitor controls pass in normal and optimized Python; SDK/Mach-O
and owned live exec checks pass. The final compiler check's peak conservative
charge is 2,759,737,344 bytes under the 3 GiB compiler-only cap. All six failed
attempts remain failed. Full logs and raw observations stay in artifacts.
[Portable checks](../benchmarks/results/2026-11-12-granite-config-v1.json),
[complete qualification](../artifacts/granite-config-2026-11-12/qualification-1.json).

Astra approves strict config/tiny-fixture retention with no actionable findings;
conversion, full-checkpoint numerics/quality/fit, request/overhead and managed
adoption remain unqualified. [Review/commit](../artifacts/granite-config-2026-11-12/publication-checks.json).
