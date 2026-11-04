# Real upstream BF16 four-token comparison

The official Gemma4 E2B checkpoint runs through pinned Transformers5.7.0 and
PyTorch2.9.1 CPU eager BF16 with bounded streamed storage. Both original
`[2,100,257,100]` and changed-final-token `[2,100,257,101]` cases pass the
prospectively frozen execution gates. This is a descriptive comparison across
arithmetic profiles, not numerical equivalence or full quality qualification.

All540 selected tensors/60 exclusions are admitted. Actual logical requests
match the sealed plan; the independently prepared19,482-read/9,119,183,500-byte
chain matches exactly. Full checkpoint authentication, input/runtime endpoints,
finite outputs/shapes, three-token causal prefix, source closure and clean owned
process exit pass. Twelve saved arrays contain2,162,688BF16 values/4,325,376bytes.
A separate stdlib readback verifies every finite value and reproduces greedy
IDs, top-two margins and the three next-token NLLs.

On Apple M1 Pro16GiB, peak RSS is687,980,544bytes and physical footprint
468,042,904bytes, below6GiB. Authentication takes5.805s. The two case intervals
are10.531s/10.607s and include reader/model binding, inference and capture.
These diagnostic intervals are not decode throughput or performance retention.

| Original-case observation | Upstream BF16 | Independent mathematical reference |
| --- | --- | --- |
| Greedy IDs | 236761,106,1018,106 | 236761,106,1018,106 |
| Three next-token NLLs | 9.661236,28.531245,7.146544 | 9.687131,28.578942,7.186636 |
| Maximum absolute full-logit difference | 0.370245 | Compared with upstream |
| RMS full-logit difference | 0.115179 | Compared with upstream |

The changed-final-token case also has identical greedy IDs but maximum absolute
logit difference0.660714. The staged-FP32 comparisons are retained separately.
Raw embedding values match exactly; later BF16 arithmetic differs from the
mathematical/staged profiles. No equivalence tolerance was fitted to these data.

Astra approves the experiment and recommends the pinned official upstream BF16
profile as the prospective original-quality baseline. Independent mathematical
and staged-FP32 graphs remain fidelity references; historical11/27 and long
context4/8 failures remain rejected. Full loss/task gates and app eligibility
remain open. Next compare the real full-width stock tied head against saved
tiled-head logits under a separately frozen bit-equality gate, then qualify a
representative loss block and its memory.
[Evidence](../benchmarks/results/2026-11-04-gemma-upstream-real-v1.json).
