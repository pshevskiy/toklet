# M5.3 scoped prefill implementation closure

M5.3 is functionally complete for the explicitly qualified compensated Q4 path.
The accepted components provide required-position vocabulary projection,
separate eight-row prompt execution with weight reuse, and checked explicit
chunks through 1,024. Default chunk remains 128. This closes the named prefill
mechanisms; release acceptance remains M5.6 and both fixed Q4 loss rejections
remain unchanged.

| Requirement | Accepted implementation and evidence |
| --- | --- |
| Separate tiled prompt execution | `Q4CompensatedTiled` reuses each reconstructed weight across eight independent prompt rows. Compensated reduction, bias addition and Half storage retain the single-row arithmetic. Below eight rows it selects the existing single-row path. [Accurate tile report](2026-10-13-accurate-q4-prefill.md), commit `aef86c9`. |
| Independent numerical reference and tails | The accurate tile passes all 4,550 expanded tiny independent stage comparisons. A separate 4,550-array refinement preserves every compensated stage bit. Primitive checks cover irregular rows, 65-column tails, padded/poisoned strides, inactive output guards, midpoint rounding and overflow. The same report retains all frozen gates and exact fixture identities. |
| Chunk sweep and bounded admission | Explicit 128/256/512/1024 choices fit checked position and memory admission. Eight exact-zero capacity/budget cases pass. The fixed larger compensated model at offset 17 preserves every final-logit bit across reset1024, append7 and edit1000; history and reused prefixes are exact. [Chunk report](2026-10-13-metal-chunks.md), `9b165e0`. |
| Required positions only | Ordinary prefill computes only the final causal vocabulary row using a precreated checked view; diagnostics keep all required rows. Seventy-six full distributions cover tiny/Qwen Half/Q4, offsets and irregular chunks. [Required-position report](2026-10-13-required-position-logits.md), `d93264e`. |
| Retention decision | The accurate tile improves larger 512-input/one-output completed engine TTFT 25.4128% and 25.6841% in two runs, preserving all 151,936 final-logit bits. Larger chunks improve at most 4.0506% in the named sweep, below 5%; retain default 128. Simple materialized attention is unchanged. |

The qualified experimental numerical configuration is compensated projection
with accurate eight-row prefill and compensated RMS. Its quality remains
rejected: the larger fixed Q4 mean NLL increase is 0.10268756036635285 against
the maximum 0.05, and the smaller fixed Q4 increase is approximately 0.173441.
No configuration search, source change or gate relaxation follows those results.
CPU/scalar, original projection/normalization, and chunk 128 remain defaults.

This closure does not qualify Original/Tiled's retained expanded independent
oracle failure. Public chunk admission does not establish universal nonzero
intermediate-array qualification for every arithmetic mode or arbitrary prompt;
the larger chunk evidence covers the named final distributions and state cases.
It does not extend context past 2,048, change attention, remove Float scratch,
or establish 128-output/application/external/sustained release performance.
Those distinctions are recorded in the accepted component reports and their
qualification receipts.

No engine code changes accompany this synthesis. Link/content and whitespace
checks suffice; the component reports retain native checks, strict formatting/
Clippy, ordinary CPU tests, measured artifacts, frozen identities and Astra
reviews. Astra reviewed this closure against the exact M5.3 requirements and
accepted the scoped functional completion without extending release claims.
