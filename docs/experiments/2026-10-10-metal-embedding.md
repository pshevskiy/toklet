# Exact Half Metal embedding lookup, 10 October 2026

The standalone [Rust primitive](../../tools/experiments/rust_metal_embedding.rs)
and [shader](../../tools/experiments/rust_metal_embedding.metal) gather embedding
rows by token ID, preserving every stored Half bit. Native qualification
passes on Apple M1 Pro. The [driver](../../tools/experiments/rust_metal_embedding.py) uses five
existing frozen tiny embedding cases and one independent manual bit-copy case.
No numerical gate, model artifact or engine execution path changes.

## Layout and ownership

The immutable table is contiguous `[vocab,hidden]`; UInt32 token IDs are `[rows]`
and output is `[rows,hidden]`. One GPU lane owns each flattened output scalar.
It finds the token row, then copies one `ushort` from that table row and feature
to output. This performs no floating conversion, preserving signed zero and
Half subnormals. Adjacent lanes traverse adjacent output features and table
features within a selected row; rows can select arbitrary or repeated IDs.
Each lane logically reads one four-byte ID and two-byte Half and writes two
bytes, or 256 bytes per width-32 row before ID reuse/cache effects. No latency
or tuning conclusion follows from this correctness reference.

The loader bounds input to one MiB, table vocabulary 1–64, hidden width 32/896,
table elements at most 65,536, 1–32 unique bounded case names, 1–32 token rows
per case and 8,192 cumulative outputs. Every token must be an unsigned UInt32
within the vocabulary. Every table Half must be finite. These checks and budget
admission finish before device creation; file/JSON temporaries are released.
Only width 32 and the tiny vocabulary are qualified by this task.

Slots 0/1/2 carry the Half-bit table, UInt32 IDs and Half-bit output at offset
zero. Slot 3 copies three UInt32 fields: output count, hidden width and vocabulary.
Rust `repr(C)` size/alignment are asserted as 12/4 bytes and match Metal.
Bounded products and rounded grid indices fit UInt32. The shader checks count
and token range before reads. Seven sentinels follow each allocation.

The private shared owner adds `u32` to its closed scalar inventory of `f32`
and `u16`; all bit patterns and zero initialization are valid for these types.
Native length/storage/hazard mode, pointer bounds and scalar alignment are
checked. There is one table allocation/upload per process, outside the case
loop. Token and output allocations persist across each case's three warmups
and ten samples. Queue and pipeline are cached. Retained command resources
stay alive through one completion wait; host reads and output resets happen
only outside execution. Every repetition checks table, IDs and their sentinels
against immutable host inputs and checks every output sentinel. No raw handle
or borrowed slice escapes, and no model/session state is committed.

Admission counts the shared table, host table and table readback once each;
the shared IDs, host IDs and ID readback once each; and shared output plus its
readback. The fixed 64 MiB native/runtime/diagnostic allowance covers bounded
fixture and retained diagnostic storage. The largest tiny case uses 6,328
shared bytes and 10,594 host-upload/readback bytes, admitting 67,125,786 bytes.
The table is counted once across cases, and unified memory is counted once per
physical shared allocation. The engine's 6 GiB ceiling is unchanged.

## Independent correctness checks

Pinned [Half](../../tests/fixtures/fp16-v1/manifest.json) and original
[FP32](../../tests/fixtures/tiny-v1/manifest.json) manifests, table and complete
embedding payload hashes are checked before GPU work and rechecked afterward.
The five original cases supply 1,408 active outputs. The driver first confirms
that the frozen Half output exactly equals the independent integer table gather.
Native outputs must match those bits and also pass their unchanged original
FP32-source gates. Exact gather does not replace the source comparison.

The manual table has four distinguishable rows, including positive/negative
zero, minimum positive/negative Half subnormals and maximum positive/negative
finite Half. Seventeen IDs exercise repeated and final table rows, producing
544 outputs. Integer row indexing defines its exact oracle; finite Half values
also have exact FP32 representations, so no new numerical tolerance is needed.
Total embedding inventory is six cases, 1,952 values and 78 dispatches, with
25,376 active outputs and 546 output sentinels over all repetitions.
At a queried width of 32, `17 * 32` outputs fill 17 complete threadgroups;
that manual case does not exercise inactive rounded tail lanes. The existing
FP32 add/tail regression separately covers the shared rounded-launch helper.

Full native diagnostics are validated before compaction to hashes/counts and
source-error/gate proofs. Raw JSONL remains local and hash-bound. The shared-owner,
mode-parser and build-copy changes also repeat all 39 existing FP32 add/tail
dispatches, structured shader/admission failures and all 364 tiny RMS dispatches.
The isolated utility remains pinned and adds no engine dependency.

Three standalone Rust tests cover invalid IDs/nonfinite table before device,
duplicate/oversized inventories, guarded layout and bounded admission. Five
offline Python tests cover frozen/manual exact bits, source gates, signed-zero/
subnormal/last-row/guard corruption, immutable/completion/accounting/launch
checks, the table-to-oracle invariant and evidence preflight. The isolated build
passes formatting, Clippy with warnings denied and eight standalone Rust tests.
Engine formatting/Clippy, all 75 ordinary Rust tests and 71 offline Python checks
pass. Astra accepted the completed task without actionable findings after
independently verifying code safety, every exact output and source gate, identities,
compaction, memory accounting, final reports/copies and idle-window isolation
using light reads and small gather calculations.

The [native record](../../benchmarks/results/2026-10-10-metal-embedding-v1.json)
passes all 78 embedding dispatches and every active value and output sentinel.
Each frozen output passes its unchanged FP32-source gate; maximum gate fraction
is `0.44209390550671257` and maximum absolute source error is
`0.00024265050888061523`. The manual output is exact with zero source error.
Both children exit zero without a watchdog firing. Frozen/manual physical peaks
are 7,455,320 / 5,767,744 bytes, below their respective conservative admissions
of 67,125,786 / 67,112,166 bytes. Existing add and RMS checks also pass, including
structured shader-compilation and admission failures. The RMS regression peak
is 17,039,960 bytes; no preparation-process or system physical peak is inferred
from these child measurements.

The record binds copied sources/build inputs, input files, pinned manifests,
drivers, executable and full raw JSONL. Key SHA-256 identities are:

| Artifact | SHA-256 |
|---|---|
| Embedding driver | `c139d7490d9c8026f9664e1aac1f4473729f85bc56d9605fdea19f0ece36a69d` |
| Executable | `9b5cfee05bee611900e9ccf94d1ba1fa183a23957e3fe687185aba0dd2624e9c` |
| Full native JSONL | `536d5fceba577bea87e5898e2da18d796d5913a486194df2694c26c0c4be35fe` |

The host is Apple M1 Pro, macOS 26.6.2, SDK 26.5 and Rust 1.90. Metal compile
policy remains 2.3, safe math, precise functions and deployment target macOS 15.
Power is AC with AC low-power mode disabled; no thermal/performance warning is
recorded by `pmset`. These functional checks make no latency or energy claim.
The [stderr](../../benchmarks/results/2026-10-10-metal-embedding-v1.json.stderr.txt)
preserves build/check outcomes.

The [hold](../../benchmarks/results/2026-10-10-cpu-embedding-check-window.json),
[idle](../../benchmarks/results/2026-10-10-cpu-embedding-check-window-idle.json)
and [completion](../../benchmarks/results/2026-10-10-cpu-embedding-check-window-completed.json)
records preserve benchmark isolation. Only orchestration parent 34871 is held;
its tiled numerical worker 42524 finishes naturally before heavy checks start.
Idle state is verified at 03:10:00.658156 UTC; all checks and identity validation
finish before the parent resumes at 03:10:59.443475 UTC. The original scalar
worker remains idle with unchanged CPU time. Both frozen scalar executables,
the qualified candidate receipt and its numerical-source hashes remain exact.

Qwen's complete table, wider lookup, composed decoder/cache, quality,
performance and GPU traces remain pending. M4 remains open.

```sh
.venv/bin/python tools/experiments/rust_metal_embedding.py --output benchmarks/raw/metal-embedding.json --build-dir artifacts/metal-embedding --input-dir artifacts/metal-embedding-inputs
```
