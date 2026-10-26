# Immutable packed weights and reusable projection sessions

`gemma_affine::Projection` owns mutable scratch for a fixed matrix geometry and
borrows independently owned `Weights` for each request. One immutable weight
owner can serve both projection and lookup sessions after its construction
Context drops. This is a resident-model ownership prerequisite; the
[Gemma plan](gemma4-implementation-plan.md) remains open. The
[evidence record](../benchmarks/results/2026-10-26-gemma-projection-session-v1.json)
binds the exact source, original references, executed binaries and captures.

The workspace fixes total rows/columns, maximum child rows and batch capacity.
It owns three guarded GPU buffers (input, one child output, status), complete
transactional CPU output staging, one reusable CPU child-output array and a
pending command handle. It borrows Context and owns no weights. The concrete
owner is 240 bytes on the qualified target. Requested owned bytes include actual
CPU capacities and all GPU guards; weights, Context, caller, source loading and
diagnostic overlap have separate policies. Checked arithmetic, geometry and
device buffer limits precede GPU allocation. Bounds remain 262144 total rows,
16384 maximum child rows and at most 64 children under the existing fixed format.

`Projection::apply` validates poison/pending state, complete input/output shapes,
finite inputs, retained device identity, exact total layout and every child
layout/row bound before mutation. The cached maximum reconstruction bound over
all weight chunks controls the same monotone maximum/L1 admission as the legacy
operator. It includes unused reconstruction codes. No numerical threshold or
error priority is relaxed. A compatible weight owner can have different child
boundaries; larger children, mismatched shapes and inadmissible inputs reject
without changing warmed scratch, weights or the caller.

Each child uses the extracted existing statically dispatched affine encode,
completion, status/guard and readback path. The numerical shader and legacy
Affine geometry, ABI and public behavior remain unchanged. Output is assembled
as row-major `[tokens, total_rows]` and published only after every child succeeds.
A failure after staging begins poisons that workspace while preserving the
entire caller destination. Fresh scratch can reuse the immutable weights.
Commands retain their resources through completion; incomplete work is not read
back. No routing allocation is added after admission; Metal command and encoder
creation can allocate. Per-child waits/readbacks remain, so this does not claim
an allocation-free or fully scheduled resident graph.

Formatting, strict default/Metal Clippy, CPU 356/16 ignored, Metal 400/125 ignored,
iOS device/simulator compilation and optimized 2 ownership, 1 real-table plan,
5 projection and 4 gather-fixture checks pass. Fourteen source commands and
24 controlled actual commands complete; all owned processes/groups are absent.
Five tiny and four full-path head reader controls precede GPU execution. Their
synthetic captures test readers only. Existing affine, L1, partition, gather and
real-head numerical/capture regressions pass separately, including 10 L1,
22 partition, 29 gather and 24 legacy-head reader tests.

Four tiny epochs cover 19×128 and 11×64 matrices, original/doubled weights,
different 7/8-row child boundaries, sparse L1-only inputs, dense rejection and
shape/size/nonfinite failures. Each construction Context drops before fresh
same-device execution. All 88 steps, 1500 returned values, 24 immutable owners,
60 published chunks/callbacks and 204 GPU allocation identities across disjoint
epochs pass independent replay. Failure after two completed children leaves the
caller unchanged and poison sticky; a fresh workspace recovers with the same
weights while the poisoned workspace stays alive. All cold/active/inactive/stale
scratch, guards, signed zeros and odd metadata padding pass 39 actual-capture/
mutation tests. Tiny workspace policies are 2284 and 1420 bytes; conservative
surrounding workload policy is 324009984 bytes.

The real diagnostic loads the original Gemma4 tied table once, 262144×1536,
as sixteen 16384-row chunks with bounded three-buffer source staging. All
48 original callback chunk digests match independently authenticated source.
The construction Context and its autorelease pool drop before a new same-device
Context creates projection and lookup workspaces. The same weights serve two
four-ID lookups around seven projection calls (repeats, single-row partitions
and resumed full batch). All 4194304 projection and 12288 lookup returned values
match the accepted frozen staged-FP32 arrays bitwise. This new reuse witness
does not add an independent scalar evaluation. The separate unchanged legacy
head witness preserves its 1048576 ordered/scalar and 8388608 GPU/reference
comparisons, all with zero difference against rounded FP32 arrays.

All 49 raw arrays/54772416 bytes, complete parent/child CPU staging, cold and
final shared GPU scratch, immutable guarded-buffer digests and 51 observed GPU
allocation identities pass replay. Overwritten intermediate child device
outputs and private lookup scratch are explicitly not captured by this new
witness. Thirty-eight actual-capture/mutation tests pass, including coordinated
changes to numerical views. Safe buffer descriptors and original outcomes
precede fallible raw writes; a pure writer-failure control retains the original
outcome and attempt with no completed-array credit.

The real policies are 226495664 bytes immutable weights, 4743496 projection
workspace, 16851208 lookup workspace and 14155848 maximum source staging with
descriptors. Conservative aggregate workload policy is 439229020 bytes, including
two 64 MiB Context allowances even though they do not overlap, references,
parser/caller/capture state and other bounded diagnostic allowances. These are
requested bytes and conservative accounting, not exact physical residency or
whole-model/mobile fit.

On MacBookPro18,3, Apple M1 Pro, 16 GiB, macOS 26.6.2 (25G83), new real native
RSS/physical peaks are 274694144/265503464 bytes, below 1 GiB. Independent replay
peaks at 95125504/88507112 and reader regressions at 171622400/138904320, below
512 MiB each. Tiny native peaks are 20054016/9126464. These are separate process
peaks; diagnostic loading, hashing and capture times do not measure generation
latency. No comparative physical-memory or performance gain is claimed.

Retain the reusable projection/session boundary. Foreign-device and driver or
in-flight failure injection remain unqualified. Complete graph/KV/attention and
pointwise scheduling, historical source/precision/long-rotary discrepancies,
quality/performance, app/iPhone and 131072-output-plus-prompt gates remain open.
No model capability, download eligibility, numerical tolerance or context/output
limit changes.
