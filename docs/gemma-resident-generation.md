# Resident original BF16 task reference

The reference keeps4,559,518,278bytes of CPU BF16 weights resident, aliases the
tied head to the token embedding, and leaves the4,697,620,480-byte PLE table on
the authenticated descriptor. Compact PLE lookup is single-caller scratch;
request sessions own stock DynamicCache and committed IDs. This is reference
infrastructure, not a Toklet backend or app admission.

A567-position bound covers the frozen task contract. The physical producer KV
bound is9,775,104bytes, counting local retention≤512 and global retention≤567.
Weights,512MiB runtime,256MiB scratch and loading tiles produce a5,384,036,934-byte
admission estimate. Measured peak RSS is5,011,701,760bytes and physical footprint
5,022,585,128bytes, below6GiB. Phase load/replay counters are retained separately;
loading takes10.27s. Earlier source collection overlapped part of this run, so
latencies are diagnostic observations, not retained performance claims.

Three cold controls pass tiny resident/streamed bits, cached/fresh sharedKV,
reset/admission/input rejection, retained partial-load cleanup and a mid-forward
failure after KV mutation. Failure marks the cache as requiring reset/replay;
committed IDs advance only after a complete successful step.

The prospective real contract requires exact lowest-ID greedy choices at511,
512,513,566,567 and reset32. All six pass; position568 rejects before any source
read/cache/history mutation. Every output is finite, shape-correct and independently
replayed from twelve saved BF16 rows/6MiB. The complete10,870-read/4,579,875,398-byte
source chain, runtime/input/source endpoints and clean owned exit pass. Required
Rust inputs are unchanged from the full-loss checks; Astra approves code and evidence.

**Incremental logits are not bit-identical to fresh scoring.** The511/reset32
cases match exactly; at512/513/566/567 there are262,101/262,142/259,311/262,144
changed BF16 words. These comparisons were prospectively descriptive. Only the
six stated greedy decisions are qualified. Single-token cached observations
are0.585/0.882/0.213s, versus78–87s for complete fresh prefixes; these different
workloads are not a benchmark speedup.

Next run a controller pilot, then the unchanged100-task suite with one CPU thread.
Verify every sampled ID, including a hidden terminal stop, through a stateless
teacher-forced replay before grading. A divergent trajectory withholds grading.
The100-task gate, earlier fidelity failures and ordinary model/app/phone adoption
remain open. [Evidence](../benchmarks/results/2026-11-04-gemma-resident-generation-v1.json).
