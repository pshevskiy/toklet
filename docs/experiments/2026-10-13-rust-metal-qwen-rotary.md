# Qwen dimension64 rotary, 13 October 2026

Eight saved layer0 Qwen Q/K rotary stages pass all unchanged source FP32 gates.
The existing split-half shader is reused without arithmetic changes. A separate
native profile permits rows1–4,14 query heads or two KV heads, dimension64,
theta1e6 and checked `offset+rows<=2048`. Tiny dimension8 bounds remain
unchanged. Total input inventory stays at8192 Half values, and shape checks
precede allocation. Explicit dimension travels through pair indexing, launch
bounds, output layout and admission. No in-place input/output alias is allowed.

The [host](../../tools/experiments/rust_metal_rotary.rs) dispatches one lane per
split-half pair, using FP32 inverse frequency, angle, precise cosine/sine and
uncontracted products/sums before Half stores. Shared initialized owners and
tracked retained command resources live through one completion wait. Mapped
storage covers rounded lanes and guards. Ordinary offsets0 and5 are numerically
qualified; the entire2048 parser range is not. The [adapter](../../tools/experiments/rust_metal_qwen_rotary.py)
checks the pinned configuration, saved inputs, full frozen source/oracle hashes
and original gates. Position0 with nonzero dyadic inputs independently checks
exact identity; nonzero-position frozen cases cover split-half layout. All
inputs/guards stay intact.

| Stage | First-frame values | Equal frozen Half bits | Maximum original source gate fraction |
| --- | ---: | ---: | ---: |
| Q | 10,752 | 10,749 | 0.453450805196 |
| K | 1,536 | 1,536 | 0.446445228829 |
| Manual position0 | 128 | 128 | Exact identity |

All thirteen repetitions match exactly, covering161,408 active Half values
and819 raw guard words across117 Qwen/manual commands. No tolerance changes
are made for the three Q discrepancies. Same-binary regressions include273
tiny/manual rotary commands, four composed normalization/QKV graphs and nine
complete tiny decoders. Their active frames match the preceding qualification.
Together with add/tails there are598 successful commands; invalid shader,
insufficient budget and completed rotary overflow fail as required.

Maximum rotary admission is67,144,774bytes, counting shared buffers, baseline,
readbacks and the64MiB runtime allowance. Largest measured own-child physical
charge across all jobs is17,334,896bytes, below6GiB. Preparation/system memory
is not inferred from child peaks, and timings are diagnostic. Pinned converted
front weights are copied from prior hash-bound qualification inputs; no new
checkpoint conversion is needed. Post-completion proof checks protected CPU,
checkpoint and tokenizer identities and confirms prior workers absent/VM off.

One successful native attempt passes18 build checks,42 standalone Rust tests,
formatting/strict Clippy and release compilation. Offline115 experiment Python,
193 benchmark Python and76 ordinary engine Rust tests, root formatting and
Clippy pass. Root release outputs are preserved. Final binary SHA256 is
`65bc2c075470488b6196e452963c1509cffda3be3052bb60c3c254f776ee39cd`.

The [native record](../../benchmarks/results/2026-10-13-rust-metal-qwen-rotary-v1.json)
and [transcript](../../benchmarks/results/2026-10-13-rust-metal-qwen-rotary-v1.json.stderr.txt)
retain source/build/driver/input identities and completed-work evidence.
[Archive producer](../../benchmarks/results/2026-10-13-rust-metal-qwen-rotary-archive-v1.py.txt),
[idle proof](../../benchmarks/results/2026-10-13-rust-metal-qwen-rotary-idle-v1.json),
[analysis producer](../../benchmarks/results/2026-10-13-rust-metal-qwen-rotary-analysis-v1.py.txt)
and [analysis](../../benchmarks/results/2026-10-13-rust-metal-qwen-rotary-analysis-v1.json)
verify byte-identical raw-to-compact replay, guards, repetition equality and
prior graph regressions. Raw JSONL is retained locally at the recorded path.

Astra accepted the task after independent byte-identical replay, two targeted
tests and a report correction distinguishing identity from pairing coverage.

Inputs are saved Q/K projections; actual normalization/QKV→rotary propagation,
Qwen cache/GQA/full-layer/decoder and root Metal session integration remain
pending. M4 stays open. This task qualifies the primitive needed for the
remaining resident backend work.
