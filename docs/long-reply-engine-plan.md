# Engine plan for one 128K-token reply

The user clarified on24 October2026 that128K means **131,072 output tokens in
one reply**, in Chat and Expert. Prompt/template/history consume additional
positions. This is a new priority within E2/E4/H1/H3/H4 of the
[active engine plan](engine-expansion-plan.md), whose remaining architecture and
Hugging Face work stays open. Every code task receives Astra review and a commit
after its checks pass.

Current execution remains256 Chat output,128 raw Expert output,512 app context
positions and64KiB conversation/decoded text. The general engine ceiling32768
and selected native MLX geometry ceiling512 are separate constraints. Raising a
UI preference or constant does not implement a longer model/context path.
The [mobile preparation](long-reply-mobile.md) preserves future131072 preferences
and bounds stored text; runnable settings still follow current native limits.

## Tasks and acceptance

| Task | Mechanism and evidence | Status |
| --- | --- | --- |
| L1 | Freeze independent full-prefix/final decoding witnesses for all three existing pinned tokenizers, including split/incomplete UTF-8, controls, whitespace, literal replacement characters and Mistral fallback-run rewrites. Include exact131072-ID synthetic text cases and compact decode-only fixtures; no model-generation claim. | [Witnesses frozen](chat-stream-oracle.md); target implementation remains L2 |
| L2 | Bounded incremental response decoding and event delivery with explicit token/byte/carry/batch limits, terminal flush, failure state and no full-prefix decode or output clone per token. Match L1, document any decoder/input restrictions, preserve current three-profile behavior. Qualify text ownership separately from numerical state. | [Rust decoder checked](chat-stream-decoder.md); native/app event ownership remains L3/L5 |
| L3 | Engine-owned runnable capabilities: distinguish raw/Chat output, total context and input/decoded-byte bounds; admit prompt plus requested output with checked arithmetic and exact source position semantics. Cursor/batch delivery must preserve sampled versus committed history, hidden stops, cancellation and replay. Keep unavailable choices explicit. | Open |
| L4 | Implement a concrete long-context graph one mechanism at a time: independent scalar oracles, checked config/tensor mapping, scalar execution, resident Metal/state, memory planning and independent real-checkpoint parity. Require more than131072 admitted positions for the requested output plus actual prompt/history. | Open |
| L5 | Integrate bounded text and generation into Chat and Expert; include numerical weights/KV/recurrent state/scratch, tokenizer, callbacks, UI/history and conversion peaks in admission. Test exact131072 output boundary, overflow, early stop, cancellation near boundaries, stale events and replay on Mac before exposing the capability. | Open |
| L6 | Qualify a real long-context checkpoint and full app on Mac and separately iPhone14Pro, including meaningful long generation, quality gates, completed timings, memory and lifecycle. Restore/offline chat, model switching and UI controls must pass. An unavailable or over-budget device stays explicitly unqualified. | Open |

The working future text targets are8MiB per conversation JSON and decoded reply,
with64MiB persisted UI history. These are proposed engine limits, not current
native constants or a qualified phone process budget. Long records must have
bounded copying/backpressure and retain exact text. Synthetic131072-ID decoding
witnesses do not qualify131072 numerical generation, quality or device fit.

## Candidate graph investigation

The immutable official [Qwen3.5-0.8B configuration](https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/eb706f593d2d43c90a10271199c10b07ced7569a/config.json)
declares262144 positions and a hybrid linear/full-attention text graph. Its
[upstream text implementation](https://github.com/ml-explore/mlx-lm/blob/0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d/mlx_lm/models/qwen3_5.py)
uses causal convolution and gated-delta state alongside attention. This makes it
a candidate for L4, not an admitted or device-qualified model. Its partial rotary,
attention gate, normalization/weight-conversion conventions and recurrent state
need their own explicit fixtures. Text-only support must neither imply vision
support nor inherit Qwen3/Qwen2 tokenizer or chat semantics. Inspect immutable
quantized candidates and actual resource requirements before selecting one.

No reliable completion time is assigned before these missing mechanisms and
hardware gates are resolved. A fit report reserves no memory and cannot grant
numerical, text or physical-device qualification.
