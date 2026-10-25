# Fixed four-token Gemma4 E2B mathematical reference

The independent reference now computes the complete text graph for the actual
[immutable four-bit E2B checkpoint](gemma4-checkpoint.md). It uses the bounded
[tiled packed-row reference](affine4-tiled-reference.md), without retaining
expanded model matrices. This is a mathematical witness for the next native
adapter; it does not execute Toklet or qualify upstream original-BF16 behavior.

## Scope and arithmetic

The exact named geometry is H1536, vocabulary262144,35 layers,15 KV producers,
Q8/KV1, PLE256, local D256/R256 and global D512/R128. Layers0..14 use I6144;
layers15..34 use I12288. The local window is512 and the last shared owners are
local13/global14. The reference resolves278 packed matrices and262 direct
vectors once before execution. Immutable byte-backed weights remain separate
from activations and graph metadata.

The raw input IDs are `[2,100,257,100]` at positions0..3. No tokenizer or chat
formatter runs. Independent NumPy/libm/Decimal80 math captures713 intermediate
states per precision contract: FP64 activations and explicit staged-FP32
operator boundaries, both over FP32-reconstructed four-bit weights with BF16
coefficients. Reductions use FP64; the tile helper permits documented BLAS
reassociation. This differs from original-BF16 execution and from a bitwise
ordered scalar reduction. Normalization computes its denominator once per row;
softcapping bounds temporary Python/libm values to1024 elements.

A streaming sink retains each state before reuse. The two complete contracts
occupy103,403,520 bytes. Eight evenly distributed values per state occupy68,448
bytes in the committed [compact fixture](../tests/fixtures/gemma4-text-math-v1/math.json).
The complete raw binary remains in ignored local artifacts with hashes bound
by the [receipt](../benchmarks/results/2026-10-25-gemma4-text-math-v1.json).
The compact fixture samples are not a replacement for full-array validation.

Changing only the final input ID to101 produces two additional complete runs.
All1,426 comparisons of positions0..2 preserve their exact typed prefix bytes.
Exact prefixes retain matching hashes; any nonexact prefix would be streamed
before its numerical comparison into a separately bounded77,552,640-byte file.
The final collector independently replays every prefix and all713 cross-
precision comparisons from captured bytes using scalar Python arithmetic.

## Results and remaining numerical gaps

Under the existing `atol=1e-5, rtol=1e-4`, five cross-precision arrays reject a
combined57 values:27 each in the uncapped head and capped logits, plus one each
in layer14/15 post-feedforward norms and layer26 Q projection. The largest
error-to-bound fraction is1.54940274476271. All intermediate observations and
failed values remain recoverable from the complete binary. No tolerance or
quality gate is loosened. The reference capture passes while complete cross-
precision qualification remains false.

Both contracts produce the same four diagnostic argmax IDs
`[236761,107,236951,107]`. These are per-position next-token predictions for the
fixed input, without sampling, generated continuation or quality qualification.
The final-position top-two margins are approximately0.05164 in both contracts.

The owned cold process uses a separate512MiB reference allowance:128MiB each
for runtime/BLAS, tiles/output and head/logits, plus64MiB each for capture overlap
and graph/metadata. The completed process reports270,467,072 bytes maximum
resident memory and101,827,400 bytes peak physical footprint. This allowance
is not a Toklet working-set, phone fit or performance claim. Cold runtime
admission pins source/dependencies and native images before NumPy import,
disables user-site imports and uses a fresh child cache. Full model/source/
protected-binary guards and first-captured output buffers are rechecked before
publication.

## Verification and retained attempts

Eight offline graph methods pass, including all180 states across both frozen
[tiny packed graph](gemma-packed-graph-reference.md) contracts, prefix/chunk
checks, bounded softcap tails, exact real geometry and admission, streaming
failure/interruption/endpoint retention, and future-prefix raw/hash failures.
The established eight affine tests also pass. Rust numerical code, tests and
examples are unchanged from `cb2aeec`; this reference-only task does not rerun
unrelated engine builds or benchmarks.

The first portable attempt incorrectly asserted180 states per contract instead
of90; it remains failed. Another early attempt changed an admission helper
while its guard was active; it remains unaccepted despite passing test output.
Both were repeated against frozen inputs. The first real child completed its
math, but its controller reused the runtime-path variable for probe bytes and
failed its endpoint guard. That attempt remains failed. A regression executes
the actual verification block and final guard, reproduces the old failure and
checks the corrected path list before the fresh accepted run.

Public E2B model/session construction, native checkpoint numerical parity,
tokenizer/chat/generation/state, resident Metal, quality, representative latency,
iPhone resources and app eligibility remain open. The earlier15 packed GPU,
15 dense GPU and8 long-rotary failures remain unwaived. No capability, ABI,
catalog, position limit or131072-output-token support changes here.
