# Original BF16 32-target scorer pilot

The first33 frozen held-out loss IDs produce32 scored rows exactly equal to
the stock causal-LM forward:8,388,608BF16 words, zero differences. This qualifies
a bounded upstream scoring mechanism, not the full10,000-target quality gate.

`LossReference` runs one stock CPU eager BF16 text forward over the complete
prefix, retaining causal attention and shared K/V. It gathers unique embedding
and PLE IDs in groups≤32, reconstructs token order through stock scaled
embeddings, and projects at most32positions at once through the full-width tied
head and original softcap. FP64 logsumexp scores precisely row0→ID1 through
row31→ID32. Ordinary four-token entry points and Rust kernels are unchanged.

The full805MB tied head is loaded once through bounded reads. A shape-specific
ledger includes runtime, layer/root weights, eager attention, PLE/gather copies,
observed layer outputs, sharedKV/scratch and logit/reduction temporaries.
Admission estimates1,952,615,552bytes for33positions. The implementation has a
reference-only1,025 ceiling; that shape is not qualified by this pilot.

Eight cold controls pass, including independent scalar NLL alignment, stock-row
bits, a32+1position tail,33unique IDs split32+1, input/memory rejection before
reads, and prior stream/reader controls. Relevant Rust inputs are unchanged.
The real run seals exact IDs, planned requests and an independently prepared
17,460-read/8,287,377,036-byte chain before outputs. Actual requests/readchain,
source/runtime/input endpoints and owned process exit pass. A stdlib raw replay
confirms every BF16 word, lowest-ID greedy and32stable NLLs; maximum NLL replay
difference is1.78e-15, mean NLL12.0167252021.

On M1 Pro16GiB peak RSS is1,538,588,672bytes and physical footprint1,295,058,768bytes,
below6GiB and the admission estimate. Authentication is6.18s, scoring15.69s and
stock comparison13.94s. Intervals include binding/read/head/capture work as
applicable; they do not establish a performance improvement.

Next qualify one complete frozen1,025-position/1,024-target upstream block with
the same source/runtime/resource guards and independent per-target NLL replay.
Compare the first32causal rows with this pilot descriptively: larger shapes may
select different kernels. Native quantized deltas, full10k/100 quality, earlier
fidelity failures and app eligibility remain separate open requirements.
[Evidence](../benchmarks/results/2026-11-04-gemma-upstream-loss-pilot-v1.json).
