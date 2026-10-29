# Borrowed multi-head Half attention and staged KV

B4d3 extends the existing serial Reference attention and transactional KV set
to Q4/KV2/D64 and Q16/KV8/D64. The original Q4/KV1/D32-or256 profiles remain.
This is a bounded reusable primitive and cache task; it does not admit an LFM2
model, ordinary Chat, the declared128,000-token context or an iPhone profile.

Query/context storage is `[32,Q,D]`, scores/probabilities `[Q,32,capacity]`,
immutable old K/V `[KV,retained,D]` and separate new K/V `[KV,32,D]`. Active
chunks contain1..32 tokens; capacity is bounded by544. A query head selects
`head/(Q/KV)`. Fixed physical head strides remain independent of active token
count. No expanded KV heads or old/new concatenation is required.

The fourteen-word shader ABI appends the query-head count, KV-head count and
old retained stride. The existing serial FP32 dot/scale, stable softmax,
probability and context arithmetic is preserved, followed by one Half context
store. The source keeps MAIN fatal and WIDE diagnostic at its prospective
per-case-stage envelope; target results cannot change the gates.

Cache accounting uses physical KV heads and unique layer owners. Six attention
layers with eight KV heads are six owners. Begin fills each head's active new
rows; finish checks every active head through the unchanged Half validator and
actual buffer offsets. Publication uses the shared at-most-two copy runs per
head, with source stride32 and destination stride retention. All intervals are
checked before encoding. Existing Ready, completion, poisoning and zero-prefix
reset/replay rules remain in force. This cache transaction alone does not
publish convolution history, logits or a model token prefix.

The independent source fixes11 cases/143 arrays/22,405,366 raw bytes before
native outcomes, including one/partial/full chunks, rings,512/513 and all32
queries after committed512. Eight portable controls and exact three-file replay
pass. Astra reconstructs all8,358,427 physical Half words, both arithmetic
profiles, all36,232 masked positive zeros and the per-head fill/copy intervals.

The native package retains complete active attention values plus actual whole
physical buffer hashes. Reconstructing the known inactive canaries and checking
those hashes verifies inactive storage; it does not turn unlogged words into
individually captured observations. Cache source cases retain every physical
after-K/V word. The512→513 copy test uses one actual Set without reseeding or
replacing old storage between its two transactions. Other controls cover the
last head of the last of six owners, omitted/partial producers, invalid values,
metadata rejection, sticky status, partial-copy poisoning and reset/replay.

All four new release GPU jobs pass. Astra independently gates1,978,360 MAIN
values across98 submissions/119 raw records:753,432 source,1,154,176 partition
and70,752 recovery comparisons. The maximum MAIN gate fraction is0.009427;
WIDE has no diagnostic exceedance. All21,780 observed masked probabilities are
positive zero. Full physical hashes authenticate padding and inactive output
storage; all3,916,954 source after-cache words match exactly. Fifty-three new
release Rust allocation boundaries report zero; driver/capture allocations are
outside that counter.

Seven required legacy jobs also pass independent saved-raw review:377,464
attention source/recovery values,67 cache states and428,832 tiny Gemma graph
values across2,400 stages. Existing arithmetic, fixtures and gates remain.
Across all eleven jobs,100 observed release Rust allocation boundaries are zero.
The current host cache metadata residual grows from9,976 to9,984 bytes through
the explicit KV-head view field; physical buffers are counted separately. This
is a checked host ledger, not an independent measurement of Rust type layout.

Six full validation/build commands pass formatting, strict default/Metal
all-target Clippy,450 default tests,539 Metal-feature tests and optimized test
compilation. New logs retain39,103,956 bytes under16MiB per-stream caps. Existing
legacy captures need a separate prospective64MiB cap and retain96,364,008 bytes.
Both owners preserve all output and close every child group. The exact tested
executable is retained with its original executed-path identity.

The independent reader's first attempt incorrectly treated Set identities as
globally unique across separate test processes. The preserved correction resets
that namespace per process and retains all within-process checks. No target,
source or numerical gate changes. The six-owner fault workload uses capacity64;
single-owner source cases cover544 and the actual512→513 chain. The injected
partial-copy failure follows a real first K copy; it is not a hardware-error or
rollback claim. Sparse memory observations establish no process peak/model fit.

[Machine evidence](../benchmarks/results/2026-10-29-staged-gqa-half-v1.json)
records the complete source, reviews, raw audit and executable identity. The
[LFM2 plan](lfm2-hybrid-plan.md) retains the resident mixed graph, real-model/text/
memory qualification, app/adoption quality, performance and phone requirements.
