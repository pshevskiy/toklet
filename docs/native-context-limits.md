# Ordinary native MLX context limits

`mlx::native_limits::inspect` reports the ordinary native loader's declared and
enforced position bounds before a caller chooses the upper for capacity search.
The C entrypoint is `toklet_mlx_context_limits_create`, using a new 16-byte
`TokletNativeLimitsOptions` and the existing immutable, creating-thread JSON
copy/release handle. Existing options, live-context records and reports remain
unchanged.

The source opens at a private seed context of one; that seed is not reported as
a loaded position limit. Config, tensor/header/index metadata and ordinary
geometry/layout checks must pass before `factory_position_upper` is present.
The source and actual loader use the same cold position policy. The upper is
the minimum of the validated declaration, global ceiling 32,768 and the
existing graph cap, plus Phi3's existing declared window bound when present.

| Admitted ordinary geometry | Existing additional graph cap |
| --- | ---: |
| Qwen2 | None; declaration/global bounds still apply |
| Tiny Qwen3 and Mistral | 32 |
| Selected real Qwen3 and Mistral | 512 |
| Admitted Llama and Phi3 | 512 |

This table does not admit every configuration in those families. Existing
shape, frequency, storage and profile checks remain. Only Phi3's declared
`sliding_window` additionally bounds its implemented full-attention prefix;
this does not implement a ring cache or apply a universal context/window
minimum. Smaller declarations and Phi3 windows lower the upper. Unsupported
geometry, invalid/missing metadata and unknown ordinary architectures produce
a null upper rather than a guessed global limit. Specialized Gemma/LFM and
explicit Qwen3 1,025/40,960 routes do not become ordinary support through this
report. Broader scalar metadata compatibility is also separate.

Load profile 0 is the ordinary MLX entrypoint; profile 1 is the existing audited
Smol entrypoint. The latter authenticates its frozen metadata, while complete
consumed-payload authentication remains required at actual loading. Unknown
profiles/reserved fields reject. Neither profile publishes a current payload
identity. A redundant tied head may be streamed under the existing source
rules; no expanded model, GPU or session is constructed.

The report separately records backend availability in the build, tensor metadata
validation, required payload/runtime/session revalidation, and unevaluated
budget, qualification, text and whole-app/device fit. Backend build availability
does not prove a usable GPU. The metadata reader's 6-GiB admission ceiling is
not a caller budget or a padded resident-weight fit result.

Callers select an enforced upper, then use the
[native numerical capacity report](native-capacity-report.md), honor its separate
loading/session fit results, and revalidate at load. This snapshot is historical;
it reserves no memory and cannot authenticate a different profile or stale
artifact. Typed Swift consumers and worker/request-identity handling follow as
separate tasks. No app setting or factory admission changes in this task.

Four new download-independent integration tests exercise the ordinary families,
declaration/global/graph boundaries, Phi3 windows 7/511/512/2047, scalar-versus-
native geometry, unknown/specialized families, invalid/missing metadata and
profile rejection. Each accepted upper and adjacent rejection are checked
against the ordinary metadata loader. A native ABI test checks the 16-byte
record, fields/profile, existing handle ownership, wrong-thread behavior and
invalid options/pointers. Existing diagnostic routines remain unchanged.

Formatting, strict default/Metal Clippy, default 462 tests (20 ignored), Metal
569 tests (184 ignored), release tests/static library, exact 30-header-export
matching, C compilation and strict Swift 6 compilation pass. Five explicit Mac
jobs pass: three tiny planning/factory regressions, the new C metadata caller,
and the existing Swift capacity caller. All were fresh for the changed Rust
source; the earlier capacity task's failed Swift-dispatch run was not reused.

Astra independently authenticates current source, retained products, runtime,
aliases, raw outputs and process closure.
[Exact evidence](../benchmarks/results/2026-10-30-native-context-limits-v1.json)
records the tested scope. Metadata/range assertions are compiled test evidence;
there is no additional independent saved-byte decoder of every full report.
No real-model forward, enlarged-context, phone, whole-app, performance or
broader family qualification is claimed.
