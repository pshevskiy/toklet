# Independent real Qwen3 numerical reference

24 October 2026. This reference uses the
[immutable Qwen3 eight-bit candidate](mlx-qwen3-checkpoint.md) at revision
`11de96878523501bcaa86104e3c186de07ff9068`. Its frozen manifest SHA256 is
`4594118378d2c583c80c8f0987e06d3bbca0fa77188d4869e670252070b23fc7`.
The [receipt](../benchmarks/results/2026-10-24-qwen3-real-oracle-v1.json)
records the producer, inputs, package identities, raw commands and retained
failed preparation attempt. No native target or upstream BF16 execution runs.

## Layout and numerical contract

The [selected reader](../tools/reference/mlx_qwen3_real.py) reconstructs four
unsigned octets per little-endian U32. BF16 scale/bias bits are shifted into the
FP32 representation, never treated as numeric integers. Each affine expression
is evaluated in FP64 and rounded once to FP32. A single original FP32 tied table
serves output products; embedding gathers round to Half. Floating norm vectors,
stored activations and KV values also round to Half. Scores, probabilities and
logits store FP32. Separate FP32/FP64 arithmetic paths quantify reduction and
rounding variation under this explicit contract. This reference does not claim
parity with an upstream BF16 storage graph or establish quantization quality.

Weights are immutable. The reader retains the tied table and decodes one other
matrix at a time in bounded stripes. Its mapping uses the same size- and
SHA-authenticated descriptor throughout; source paths and the descriptor are
rechecked before acceptance. Owned stripe copies prevent exception tracebacks
from retaining mapped views during cleanup. The actual malformed-reader test
keeps its original rejection traceback live while confirming both resources
close, addressing Astra's source-review finding.

The separate [Qwen3 graph](../tools/reference/qwen3_fp16.py) composes the existing
independent CPU numerical primitives. Hidden width 1024 differs from query
width 2048. Q/K project first, then each width 128 head receives its respective
learned RMSNorm vector, followed by split-half RoPE at the actual offset. V
remains unchanged. Causal GQA maps 16 query heads to eight KV heads. No runtime
architecture hierarchy or target-engine code is introduced.

## Frozen probes and checks

Six authored cases cover one/four/seventeen tokens, changed futures and rotary
offset 5. Layers 0, 1 and 27 retain raw Q/K/V projections, normalized Q/K, rotary values,
attention and MLP boundaries. Together with four greedy and three scoring
distributions, these freeze 385 arrays/41,115,648 bytes. Independent literal array
contracts reject missing/extra arrays, changed shapes, dtype, offsets or gates.

Teacher forcing always follows the independent FP64 path, declared before
export. Both paths select `[1096,374,264,3491]` on the raw prompt
`[9707,11,1879,0]`. Both sets of logit margins are retained. The original
pre-target policy uses `max(1e-4,2*pooled FP32/FP64 error,2*equivalent-prefix error)`
with relative tolerance 1e-3. Largest absolute stage allowance is 2.0 at the last
layer's gated activation; the logits allowance is 0.07747364044189453. These
numerical envelopes are separate from unchanged held-out quality gates.

One authored tokenizer witness confirms `Hello, world!` gives those four raw
IDs without special tokens using the pinned tokenizer library. BOS/EOS/pad and
thinking-control identities are checked. No template is evaluated; thinking,
stop handling and multi-turn chat remain unqualified.

All 50 portable methods pass, including 12 new ones and the existing acquisition,
tiny Qwen3 and SmolLM reference regressions. The vectorized Qwen3 graph agrees
with the independent ordered tiny implementation on unequal-width fixtures;
bypassed norms change its output. Actual export and an external-SHA stdlib-only
verification pass. Seven producer sources and 445 package source/native files
are bound. No Rust/Metal/Swift source or protected release binary changes; prior
engine checks remain inherited on identical sources.

Logical reference admission is 2,597,950,130 bytes, including source mapping,
622,329,856-byte tied table, current-matrix/FP64/BLAS buffers, traces and framework
allowances. Observed completed-process lifetime physical peak is 866,552,040
bytes, including verifier work, below 6 GiB. Charged memory can omit shared/clean
pages; this measurement is neither an OS allocation cap nor a native or phone
memory result.

The first wrapper stops before export because one existing test assumes the
real model cache is empty. The test now confines destination and source-pin
preflight to its temporary fixture. Production collision rejection and the
no-publication assertion remain intact. The failed 50-test attempt, original
test bytes and logs are retained; the successful fresh retry passes all 50.

Retain this oracle for [selected native Mac qualification](mlx-qwen3-real-metal.md),
which now passes against this unchanged reference. Quality, performance, app
Chat, thinking semantics and physical iPhone qualification remain open in the
[active plan](engine-expansion-plan.md).
