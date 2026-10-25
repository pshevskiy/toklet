# Four-step Gemma4 E2B greedy mathematical reference

An independent reference now freezes a short raw continuation over the
[immutable E2B checkpoint](gemma4-checkpoint.md), before native generation is
tested. It reuses the [complete mathematical graph](gemma4-text-math.md) and
bounded packed-row reader. No Toklet engine, tokenizer or chat formatter runs.

## Workload and ownership

The initial IDs are `[2,100,257,100]`. Four complete forward passes evaluate
prefix lengths4,5,6,7. Each visits all713 graph states in both precision
contracts. Staged FP32 selects the first index of the maximum capped logit;
FP64 is teacher-forced on that selected prefix. It is a diagnostic comparison,
not an independent FP64 rollout. Both contracts use FP32-reconstructed packed
weights and the same independently defined reductions as the earlier reference.

For each step and contract, retain the last position's complete final norm
`[1536]`, uncapped head `[262144]` and capped logits `[262144]`. The24 arrays
occupy25,239,552 bytes. Descriptors are appended before writes; raw bytes are
flushed before finite-value rejection. A graph or write failure preserves an
incomplete step without publishing its selected ID. An endpoint failure retains
the observations already completed and rejects the overall capture. Temporary
graph intermediates are discarded after validation. Eight distributed values
from each retained vector form the1,152-byte committed
[probe fixture](../tests/fixtures/gemma4-greedy-math-v1/math.json).
Full arrays remain in ignored local artifacts, authenticated by the
[receipt](../benchmarks/results/2026-10-25-gemma4-greedy-math-v1.json).

## Observed results

Both contracts select `[107,100,45518,107]`, with unique maxima at every step.
The staged-FP32 top-two margins are approximately0.051638,0.973206,6.781302
and16.876516. Exact margins use subtraction after promotion to Python FP64;
the first-index rule and tie counts are independently replayed from raw bytes.
The resulting committed prefix has seven IDs
`[2,100,257,100,107,100,45518]`; the final sampled ID107 is pending.

All12 complete last-row cross-precision comparisons pass unchanged
`atol=1e-5, rtol=1e-4` over2,103,296 paired values. The maximum error-to-bound
fraction is0.5103206474158878. All six step0 vectors are bitwise identical to
the last rows of the earlier accepted four-token reference. The earlier full
prompt/intermediate failures remain unwaived: five arrays/57 values in that
reference and27 native FP64-ideal logit failures per schedule. Passing these
selected continuation rows does not close complete graph precision gates.

The owned reference process reports249,626,624 bytes maximum resident memory
and75,465,424 bytes peak physical footprint, under its separate512MiB allowance.
Its90.08-second combined load/eight-graph/output observation is not a latency
benchmark. The largest FP64 head-plus-logits pair is29,360,128 bytes at prefix7.
The allowance reserves128MiB each for runtime/BLAS, tile/output and head/logits,
plus64MiB each for capture overlap and graph/metadata. It is not an engine or
iPhone resource qualification.

## Verification and limits

Four new offline methods and the existing eight graph/eight affine methods
pass. They check selection, exact margins/ties, typed raw capture, nonfinite
values through the actual driver, and partial retention after write,
interruption and endpoint failures. An earlier portable attempt passed three
methods before the actual-driver nonfinite regression was added; both frozen
attempts are retained. The accepted cold run admits dependency/native-image
hashes before NumPy import, disables user-site imports and uses a fresh cache.
The collector independently replays every complete vector, selection, gate,
prefix and resource bound, then rechecks its captured buffers and input pins.

Rust engine code, tests and examples remain unchanged from `4fe8841`.
Native continuation, tokenizer/chat/stop metadata, original-BF16/source
qualification, resident Metal, quality, representative performance, phone fit,
app integration and131072 output tokens remain open. Earlier GPU intermediate
and long-rotary failures are unwaived. No catalog, ABI or position-limit change.
