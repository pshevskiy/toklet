# Ministral 3 selected metadata and Metal rotary diagnostic

Status: required checks and three release GPU diagnostics pass. Astra approves
scoped retention with no findings; review/publication evidence is linked below.

This milestone supplies two prerequisites for the selected Ministral route:
strict configuration semantics and a borrowed Metal positional operator. It does
not authenticate payloads, admit an ordinary model, construct its graph or qualify
quality, codec, complete working sets, request performance or observer overhead.
The [CPU primitive](ministral3-rotary.md), original downstream gates and ordinary
512-position Metal limit remain unchanged.

`mlx::ministral3::validate(&Value)` borrows the complete selected multimodal
configuration and returns a fixed text-profile marker. Both pinned original and
Heretic configs are tracked as [fixtures](../tests/fixtures/ministral3-config-v1/manifest.json).
The validator preserves hidden/query widths 3,072/4,096, 32 query/eight KV heads,
128-dimensional heads, 26 layers, tied embedding/output and exact YaRN/query-scale
parameters. Parent/text BF16 declarations must agree. Supported aliases and
optional canonical token IDs are checked explicitly; unknown semantic fields,
sliding variants, changed geometry and conflicting precision are rejected.
Decimal comparisons precede binary rounding. Vision fields are selected metadata
only. Callers must use bounded, duplicate-rejecting JSON parsing before creating
`Value`; this API cannot recover duplicate keys already discarded by a parser.

The Metal component uses the approved immutable CPU table's FP32
`[capacity,64,{cos,sin}]` words and Half `[capacity]` gains. Upload happens once;
heads and layers can borrow those resident allocations. Tight Half operands have
layout `[tokens,heads,128]`; pair `i` joins `i` and `i+64`. Each allocation has
explicit guards. A nonzero absolute base avoids allocating an unused prefix.
At most 512 cached rows, 512 operand tokens and 32 heads are accepted. Higher
absolute positions are component diagnostics within the source declaration;
they grant no larger ordinary model context.

Separate key/query entrypoints select the gain path outside pair arithmetic.
Safe, precise Metal compilation and disabled contraction preserve separate FP32
products and sums. Rotation rounds to Half; queries then multiply that already
rounded value by an already Half gain and round again. Keys omit the gain.
Checked plans, byte extents, device identity, pairwise aliasing and retained,
uncommitted command state validate before encoder creation. Encode uploads no
operands, allocates no owned buffers, waits for no completion and resets no
status. Device nonfinite/overflow ORs a status bit; callers check completion and
status and discard failed output. No session or KV state belongs to this operator.

The borrowed module currently has test-only integration, following existing
standalone Metal qualification modules. Its real Rust/MSL implementation executes
in release diagnostics. A production graph owner must follow source/payload and
full-model qualification; no temporary public GPU/buffer hierarchy or dead-code
allowance is introduced to expose it prematurely. The metadata validator is
available independently, while the ordinary loader still rejects this architecture.

The [prospective contract](../artifacts/ministral3-metal-2026-11-13/prospective-protocol-1.json)
reuses all 11 unchanged dual-profile cases and their original per-array gates.
GPU diagnostics cover uploaded words, replay/chunk/rebase bytes, guards, preflight
and numeric-failure reuse. Release thread-local Rust allocation and two-run 3+10
completed operator clocks are separate from setup and full-request performance.
The 64 MiB pipeline/context allowance is policy, not measured residency. CPU
cache/fixture/upload owners, guarded operands and GPU tables count separately;
sampled process observations do not establish a continuous allocation cap.
Genuine foreign-device rejection requires a second physical Metal device and is
reported unavailable when the host supplies only one.

The stale selected Q4 index remains rejected. Future bounded conversion must
qualify 9,216-column matrices separately from the existing 8,192-column helper,
then pass original graph, 10,000-target/100-task quality, codec, fit, request and
overhead gates. iPhone and interactive capture remain deferred.

Required fmt/strict CPU+Metal Clippy, 755 CPU tests and 907 Metal-feature tests
pass; 52/343 explicitly ignored cases remain separate. Eleven selected-config
tests and two portable Metal tests pass. Eight guarded commands complete below
the 3 GiB compiler cap (largest observed charge 3,068,362,752 bytes). The initial
post-build collector wrongly expected Cargo target kind `lib`; actual declared
crate kinds are `rlib`/`staticlib`. Its error is retained; fresh binding of the
successful build and unchanged sources reuses all eight genuine check results.

The explicit GPU suite passes all 11 cases/55 arrays: 124,928 output/profile,
8,256 uploaded coefficient/gain-profile and 1,408 CPU period-profile coordinates.
All 22 replays, 64 chunk calls, 64 rebased calls, 11 same-device preflight
rejections and eight numeric-status/reuse cases pass. The single-device host
cannot execute a genuine foreign-device negative. Three completed tests stay
within the 6 GiB guard at a 109,035,520-byte sampled/terminal observed charge,
including helper reserve. Owned cleanup is empty; no model or compiler overlaps.

Release measurement observes zero Rust allocations on its measured thread for
48 admitted and 16 rejected encodes; driver allocations are unmeasured. Four
3+10 cohorts retain all 52 warmup/measured samples on AC, nominal thermal state
and Low Power Mode off. Each command contains one key and one query operation:

| Case/run | Host encode µs | Host encode-to-completion µs | Completed command GPU µs |
| --- | ---: | ---: | ---: |
| Eight rows at zero, run 1 | 9.021 | 305.792 | 28.688 |
| Eight rows at zero, run 2 | 9.876 | 303.479 | 28.687 |
| Two rows at 16,383, run 1 | 4.583 | 246.979 | 27.250 |
| Two rows at 16,383, run 2 | 4.000 | 249.562 | 27.375 |

These are median operator diagnostics with raw samples, MAD and range retained,
not request TTFT, reliable tail estimates or a performance retain decision.
Measured setup is separate and may use a warm system compilation cache; neither
energy nor whole-system pressure was measured. No O1 observer-overhead bound
follows. [Full checks](../artifacts/ministral3-metal-2026-11-13/check-queue-result-1.json),
[GPU evidence](../artifacts/ministral3-metal-2026-11-13/gpu-scalar-closure-1.json),
[scalar report](../benchmarks/results/2026-11-13-ministral3-metal-v1.json),
[review/publication](../artifacts/ministral3-metal-2026-11-13/publication-checks.json).
