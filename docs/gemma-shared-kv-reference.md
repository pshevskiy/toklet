# Independent Gemma4 shared-KV routing witness

This G2 witness freezes producer/consumer control flow before native shared-KV
integration. It authenticates captured `gemma4_text.py` at MLX-LM revision
`0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d`, SHA256
`73ffa75e9069ce7b2f864ccd8f0267bb80c5b5d4e2b73102b8a2a93fe03f936d`.
The [expected routing fixture](../tests/fixtures/gemma-shared-kv-v1/expected.json)
uses independent named cases and exact integer/Boolean state.

For the E2B35-layer layout, the first15 layers own KV. The last20 reuse the last
preceding producer of the same attention type: local producer13 or global
producer14. Consumer layers receive that producer's current call's exact KV
objects and rotary offset. They receive `cache=None`, so they cannot append
those already-updated keys again. Every layer still computes its own Q and
query rotary operation. Repeated calls reset intermediate routing state.

The adapter executes unchanged selected bodies: `Attention.__call__`,
`DecoderLayer.__call__`, and `Gemma4TextModel.__init__`, `_make_masks`, `__call__`.
It therefore observes both forwarding boundaries. Deferred annotations are the
only modification to selected executable source. Attention/DecoderLayer
constructors and all numerical operators are explicit stand-ins; the model
constructor uses an observed layer factory. Strict shape/identity-only tensors,
projections, norms, rotary, attention, cache and containers reject unsupported
operations. PLE, MoE and learned layer scalars are disabled in this control-flow
probe. No numerical arrays, actual weight allocations or MLX runtime execute.

Four cases cover the actual35-layer producer map and512-window transition,
alternating types, no sharing and an irregular layout/window1. Fifteen repeated
calls observe226 layer forwards,108 producer appends and118 shared consumers.
Each consumer receives current-step producer identity and offset; it makes no
K/V projection/normalization/rotary/append calls. Shape checks include separate
local256/global512 head widths, without qualifying their numerical behavior.
The existing [KV state witness](gemma-kv-reference.md) and
[cache owner](window-cache-scalar.md) remain separate mechanisms.

Seven portable methods check literal maps, invalid layouts/missing producer
types, exact expected counts/positions, strict stand-ins, authenticated source
selection and the actual missing-shared guard, normal routing identities and
partial observations after a synthetic consumer failure. Four owned-process
regressions check preflight rejection, source substitution and retained failed
logs/endpoint guards, including observation replacement after admission.
Ordinary caught comparison failures retain partial routing rows; interruption
can prevent publishing that JSON, while the owner retains available logs.
Four owned commands pass. Independent replay checks all
expected routing and recorded observations. The
[receipt](../benchmarks/results/2026-10-24-gemma-shared-kv-reference-v1.json)
binds exact source, interpreter/imported files, tests, captures and replay.
Astra reviews final evidence before commit; engine/shader sources and protected
binaries remain unchanged.

This establishes selected-source routing/identity only. The later
[native routing/publication plan](shared-kv-scalar.md) now checks stale/missing
frames and complete cache batches. This reference alone does not establish
numerical attention or rotary parity,
full DecoderLayer/Gemma execution, native shared cache integration, model
rollback, quality, memory fit, performance or phone qualification. Those remain
open in the [Gemma4 plan](gemma4-implementation-plan.md). No capability, ABI,
app limit or installed package changes.
