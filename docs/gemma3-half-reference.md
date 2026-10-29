# Independent Gemma3 Half graph reference

29 October 2026. B3c4b freezes a new explicit graph over the existing immutable
tiny Gemma3 artifact. It supplies the source contract for resident GPU composition;
it does not run Rust, Metal, a real checkpoint or an application.

The tiny graph has two layers, hidden64, query width128, Q4/KV1, head dimension32,
intermediate128, vocabulary64, local window4 and capacity12. Four token cases
include a single token, short prompt, full12 and a changed final token. Each case
stores all48 meaningful stages in operation order, including lookup before
embedding scaling, Q/K norms and rotary, attention scores/probabilities, both
postnorm branches, physical GELU/gating stores and FP32 tied-head logits.

Both profiles use the same once-reconstructed FP32 matrices from the original
BF16/nibble artifact. Raw BF16 gamma widens before FP32 addition of one. Main
projection uses ascending separately rounded FP32 multiply/add; wide uses FP64
reduction. Every profile follows its own preceding Half stage outputs. Native
compensated SIMD projection and midpoint-aware stores are not emulated by main.
The main-only gate is frozen per case/stage from source comparisons:
`atol=max(1e-4,2*max_abs(main-wide)), rtol=1e-3`. Wide remains diagnostic.
Target results must not change those gates.

Eight additional ordinary-RoPE components cover D32/256, theta10000/1000000,
query4/KV1 and positions0/511/512/513. Main uses FP32 inverse/angle/trigonometric
values and separate products; wide uses Decimal80 inverse frequencies and FP64
trigonometry. Unrounded mathematical outputs and the existing per-coordinate
rotary bounds are retained. The explicit source position domain is `[0,544)`;
the actual tiny graph remains at most12 tokens. A parser bound or selected
component positions do not qualify every context position on a device.

Seven portable controls pass: source/shape identities, projection/store rules,
exact normalization/subnormal/GELU witnesses, one-key/local visibility, rotary
coverage/domain, main-only gates/finiteness, and graph causality/immutability.
Two bounded isolated exports produce identical manifest, arrays and progress.
Astra's independent decoder/graph reader reconstructs every stored byte without
importing the new producer, reusing only previously frozen scalar leaves.

The promoted fixture contains384 graph arrays and40 rotary arrays:424 arrays,
275232 stored values and715648 bytes. There are192 graph stage groups with108816
values per profile, plus11520 rotary values per profile. All shapes, order,
offsets, hashes, finite values, gates, both causal prefixes and23040 rotary
mathematical-envelope comparisons pass. Owned process groups are closed and
source/runtime identities remain unchanged.

The next native consumer must compare the correct absolute query rows and key
columns from cached `[4,32,C]` scores, preserve the W−1 old-view rule, and check
exact native full/chunk1/3/7 logits, state publication, failure/replay, accounting
and release allocations. This source task makes no native graph, real-model,
context-admission, quality, app, iPhone or performance claim.

See [machine-readable evidence](../benchmarks/results/2026-10-29-gemma3-half-reference-v1.json),
[fixture](../tests/fixtures/gemma3-half-tiny-v1/manifest.json),
[producer](../tools/reference/gemma3_half_graph.py) and [TODO](../TODO.md).
