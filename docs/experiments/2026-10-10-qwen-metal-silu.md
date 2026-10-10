# Selected Qwen Metal SiLU, 10 October 2026

The [qualification driver](../../tools/experiments/rust_metal_qwen_silu.py)
prepares all twelve saved SiLU stages at layers 0, 1 and 23 across four frozen
short Qwen cases. All native checks pass in the verified CPU idle window.
Astra accepted the completed code and evidence without findings.

The pinned config uses a 4,864-wide MLP activation. Complete saved Half
gate projections and SiLU outputs and original FP32 source arrays are hash
checked before splitting them into 36 complete rows, totaling 175,104 values.
Each row fits the native 8,192-value inventory without increasing its bound.
Native parsing adds only width 4,864 to the existing width-64 admission; at this
width two rows or two cases fail the unchanged inventory bound before a device
is created. Shape, dtype, bytes, finite values, full hashes and row hashes are
explicit. Original FP32 gates and all four frozen manifest identities stay fixed.
Exact independent Half output bits are a diagnostic alongside the required
source gate. No checkpoint payload or checkpoint hash is read by this driver.

The shader, stable FP32 formula, Half store and ownership/completion behavior
are unchanged from the accepted [tiny SiLU task](2026-10-10-metal-silu.md).
Contiguous row-major arrays have one Half read and write per element; no weights
or reductions occur in the kernel. A guarded row occupies 9,742 bytes. Two
distinct shared buffers total 19,484 bytes; one host baseline and two readbacks
total 29,226 bytes. Conservative admission is 67,157,574 bytes including the
64 MiB diagnostic/runtime allowance. Resources persist across all thirteen
submissions for each row, with immutable input and seven output guards checked
after Completed/no-error status. Failure discards the result; there is no
model/session state to advance.

Fresh disjoint input/build/evidence paths protect source artifacts. The same
isolated binary runs all tiny SiLU/manual regressions before selected Qwen rows,
plus add/tail and structured shader/budget failures. Full native output is
retained locally and hash-bound before compaction. Each row records its original
stage/index and full/row payload hashes. Final revalidation binds all inputs,
drivers, config and manifests. Compiler/tests/Metal work waits for the active
scalar medium numerical worker to finish naturally, with only its orchestration
parent held ahead of the next workload.

This prepares a primitive. Other layers, gating, MLP/decoder composition,
sessions, long context, quality, performance, GPU traces and M4 acceptance remain
pending. Per-row diagnostic submissions do not measure a production execution
schedule.

The [compact record](../../benchmarks/results/2026-10-10-qwen-metal-silu-v1.json)
retains every qualified command and its source/output hashes. All 468 Qwen
commands agree exactly with the independent Half oracle: 2,276,352 active
outputs and 3,276 tail guards pass. The largest original FP32-source gate
fraction is 0.495263925437; maximum absolute error is 0.3919973373413086. That
error belongs to the independently frozen Half reconstruction, not a newly
chosen tolerance. All 143 same-binary tiny SiLU/manual commands and 39 add/tail
commands also pass, totaling 650 successful completed commands. Required
invalid-shader and insufficient-budget failures exit cleanly with their
expected stages. Maximum own-child physical charge is 13,091,392 bytes, below
the 67,157,574-byte admission and 6-GiB ceiling.

Isolated formatting, Clippy with warnings denied, fourteen Rust tests and
locked/offline release compilation pass. All 58 offline experiment Python
tests pass, including six new coverage, corruption, source-gate and ownership
checks. The numerical worker exited naturally before checks began; only its
orchestration parent was held ahead of the next workload.

The isolated binary SHA-256 is
`c3ac6698cab6823dca4bf97fa8268a69dec8d4477f139c3f2ee489abeb3d0783`.
Complete original native output remains local at
`benchmarks/raw/2026-10-10-qwen-metal-silu-v1.json.native.jsonl` (14,333,870
bytes), SHA-256
`263d625856929b634727f78a645d68b99e04a52bb7d2b021c70d6834c93f7547`.
The compact record SHA-256 is
`45928bf4d4d4fec908d390f9e941a720a4b3fd7f2ddf404ffb39ff87051d2f0d`;
its [stderr](../../benchmarks/results/2026-10-10-qwen-metal-silu-v1.json.stderr.txt)
has SHA-256
`4058f0a542f213436d210b0ea9a9aafd9bc6f409c450b1b8b9297498a0b98135`.
Hashes bind full native values before their diagnostic compaction; source
manifests, config, all inputs and all six drivers were rechecked at completion.

```sh
.venv/bin/python -m unittest discover -s tools/experiments -p 'test_*.py'
.venv/bin/python tools/experiments/rust_metal_qwen_silu.py --output benchmarks/raw/qwen-metal-silu-fresh.json --build-dir artifacts/rust-metal-qwen-silu-fresh --input-dir artifacts/metal-qwen-silu-inputs-fresh
```
