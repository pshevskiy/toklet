# Loaded Gemma4 content identity

The [Gemma payload store](../src/mlx/gemma4/payload.rs) now retains an immutable
`ContentIdentity` for the exact config/index buffers and complete container it
consumed. The [named short model](../src/model/gemma_packed/e2b.rs) carries a copy
from that store. This is a prerequisite for later engine-owned capabilities;
it does not bind an HF repository/revision or tokenizer to a runnable model.

The cold loader hashes the same borrowed config/index buffers passed to config
inspection. It counts successful header/payload consumption with checked u64
arithmetic, including excluded modality bytes, and hashes the same chunks used
for retained storage. Only after EOF, exact count/digest agreement, finite values
and capacity admission succeed does it construct the private-field identity.
It contains config SHA-256, index SHA-256, complete-container SHA-256 and actual
container byte count. A late failure returns no successful store or identity.
No later file open, path lookup or per-token hash participates.

The model copies the store's identity only after the named E2B adapter and shared
producer validation succeed. Ordinary tiny/manual constructors report None,
including manually supplied storage-accounting reports. Callers can copy content
evidence but cannot attach it to an arbitrary model through a public constructor.
This value is neither a live memory reservation nor a source-provenance,
tokenizer-pairing, session-reuse or application capability token. Future native
capabilities must still bind trusted artifact provenance, the actual codec,
request limits and current resource admission.

## Ownership and accounting

The 104-byte store identity and cold metadata digest state fit the existing
separate 64 MiB storage metadata reservation. On the measured arm64 build, the
model's optional copy adds 112 bytes, counted by the existing size-of-model
accounting before session admission. Weights remain borrowed, immutable and
counted once; no second checkpoint or matrix copy is retained. Numerical order,
32 executable positions, KV layout and all public C record layouts are unchanged.
Prior generation receipts retain their historical memory totals; this task does
not rerun or upgrade those generation measurements.

## Verification

Formatting, default/Metal strict checks, 312 CPU/334 Metal-feature tests and both
iOS targets pass. Seven release payload methods and 15 release packed-model
methods pass. New synthetic storage checks distinguish exact metadata encodings
with identical JSON meaning, verify getters against the supplied buffers and
completed container, preserve copied evidence after store drop, and reject count
off-by-one/overflow. These private synthetic admissions do not claim to be the
public E2B config path. Existing digest, excluded-byte mutation, truncation,
trailing-byte, read, finite-value and exact-budget failures continue to return
no store. Tiny/manual model checks retain absent identity.

The explicit [cold probe](../examples/gemma4_payload.rs) loads the actual pinned
E2B container, resolves all 540 cold views, constructs the named model without
forward execution, and compares its retained identity with the store, same
metadata buffers and independently fixed container pin. A reviewed controller
checks the raw captured result against the accepted checkpoint inventory; eleven
corrupted identity/state/count packets fail the actual verifier. Synthetic
packets are verifier evidence only. Their owned wrapper proves successful
source/log/cleanup handling, without an invocation-failure retention claim.

On MacBookPro18,3 / Apple M1 Pro, 16 GiB, the actual model retains 2,671,706,676
bytes including its store. Cold-load admission remains 2,671,730,828 bytes.
Observed completed resident/physical process peaks are 1,602,158,592 /
2,616,641,984 bytes, below the existing 6 GiB policy. The 14.71-second combined
load/view/model-construction observation has a warm/uncontrolled filesystem
cache and is not a representative latency benchmark. No numerical forward,
tokenization or generation is executed in this task. See the
[receipt](../benchmarks/results/2026-10-25-gemma4-content-identity-v1.json).

Broader quality/full G3, earlier precision/source/GPU/rotary failures, resident
Metal/performance, actual codec pairing, native ABI/app integration, independent
phone admission/execution and 131072 output tokens remain open. No profile,
catalog, ABI, output/context limit or eligibility change follows.
