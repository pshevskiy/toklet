# Independent tiny FP32 reference

For the separate pretrained qualification, see the [Qwen forward report](../../benchmarks/results/2026-10-09-qwen-forward.md). Tiny-v1's gates and files remain unchanged.

`tiny.py` uses NumPy to export the complete two-layer learning model from
`PLAN.md`. Rust tests read the frozen arrays in
[`tests/fixtures/tiny-v1`](../../tests/fixtures/tiny-v1/manifest.json); they do not
run Python or recreate weights from an RNG seed. The manifest records the actual
array hashes, tool versions, shapes, layout, model settings, and numerical policy.

Create an environment with Python 3.12 and the pinned NumPy version:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r tools/reference/requirements.txt
.venv/bin/python tools/reference/tiny.py --check tests/fixtures/tiny-v1
```

Generate a candidate in a new directory when an intentional fixture revision is
needed. The exporter refuses to overwrite any existing directory:

```sh
.venv/bin/python tools/reference/tiny.py --generate artifacts/tiny-candidate
```

The committed fixture is immutable. Changes to weights, layout, tolerance, or
position behavior require a new fixture version and explicit review. Regenerated
results may vary slightly with NumPy's platform math libraries; verification
checks the saved file hashes exactly and the recomputed values with the frozen
per-operation tolerances. Ordinary Rust tests require only the locked test
dependencies; use `cargo fetch --locked` once, then `cargo test --locked --offline`.

Weights are contiguous row-major `[out_features, in_features]`, activations are
`[tokens, hidden]`, Q/K/V intermediates are `[tokens, heads, head_dim]`, and scores
and probabilities are `[query_heads, query_tokens, key_tokens]`. Files have no
headers or padding and store little-endian FP32 or unsigned 32-bit token IDs.
Rust decodes bytes explicitly; file storage supplies no vector alignment
guarantee. The fixture owns its arrays. A future session's KV layout remains
`[kv_heads, capacity, head_dim]`, independently of the oracle intermediates.

The model has vocabulary 64, two layers, hidden size 32, four query heads sharing
two KV heads, head dimension 8, MLP size 64, and capacity 32. Q/K/V projections
have biases; other projections do not. Output weights are tied to embeddings.
RMSNorm epsilon is `1e-6`; rotary theta is `1e6`, pairing the first and second
halves of each head with inverse frequencies `theta^(-2i/head_dim)`. These epsilon,
theta, and tied-weight conventions follow the published
[Qwen configuration](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json).
Absolute positions start at the case's offset; causal attention permits only
keys at or before each query position. Query heads map to KV heads `[0, 0, 1, 1]`.

The oracle saves both layers' normalization, Q/K/V, rotary, unmasked attention
scores, causal probabilities, context, projections, residuals, SiLU/gating, final
normalization, and logits. Cases include one token, a four-token prompt, a changed
future token, nonzero position offset, and all 32 capacity positions. It also
stores a four-step greedy trace with lowest-token-ID tie breaking and no EOS.
Stable probes cover zero and large RMSNorm inputs, large-offset softmax, and
SiLU inputs from `-1000` to `1000`. Invalid/non-finite inputs and empty or entirely
masked softmax rows are designated errors for the future Rust kernels.

Each operation freezes `abs(actual - expected) <= 1e-5 + 1e-4 * abs(expected)`.
The manifest records the measured FP32-versus-FP64 envelope on the same FP32
weights and inputs, including a relative-error floor of `1e-6`. FP64 is an
envelope check, not the saved reference dtype. This gate does not qualify real
models, FP16, quantization, or hardware performance.

The manually checked linear example uses:

```text
W = [[1, 2, -1], [0.5, -2, 3]]
b = [0.25, -0.5]
x = [2, -1, 4]
y[0] = 1*2 + 2*(-1) + (-1)*4 + 0.25 = -3.75
y[1] = 0.5*2 + (-2)*(-1) + 3*4 - 0.5 = 14.5
```

A second input `[-1, 0, 2]` gives `[-2.75, 5]`. Both rows and their actual arrays
are frozen and checked independently in Rust.

`qwen.py` explicitly loads the pinned local BF16 source into Transformers CPU
FP32, disables dropout/cache and uses eager attention. Module hooks capture
layers 0, 1 and 23; an independent functional all-FP64 pass over the same weight
values measures the reference envelope. Full arrays remain local; selected values
and every-position argmax are versioned. No code from the Rust decoder participates
in reference export.

```sh
.venv/bin/python tools/reference/qwen.py --output artifacts/qwen-forward-v2 --probes-output tests/fixtures/qwen-v2
python3 tools/reference/qwen.py --check tests/fixtures/qwen-v2
cargo test --locked --offline --release --test qwen_forward pinned_qwen_fp32_matches_full_transformers_forward -- --ignored --nocapture
```

Export directories must be new. In an existing checkout, regenerate into fresh
candidate directories and verify identities before replacing local full artifacts;
committed reference files stay immutable. Ordinary tests check the selected probes
without Transformers or source weights. V1 is retained as evidence of the initial
per-case policy that failed to cover equivalent-prefix reference shape variation;
v2 freezes operation/layer-specific limits across the four independent reference
cases. The linked report records the reason and measured differences. These short
probes do not establish held-out quality or long-context qualification.

`chat.py` freezes token IDs and rendering from the pinned Transformers tokenizer
and Qwen Jinja template, including Unicode, special tokens, tool schemas/calls,
consecutive tool replies and generation prefixes. Chat-v1 also contains 1,010
independent Python float-format probes. Its small tokenizer retains original IDs
and the BPE merge closure needed for the eight raw and seven chat cases. It is
only an offline fixture; production uses the complete local tokenizer. All file
identities, exporter identity and Python/Jinja/Transformers/tokenizers versions
are recorded in its manifest.

```sh
python3 tools/reference/chat.py --check tests/fixtures/chat-v1
.venv/bin/python tools/reference/chat.py --output artifacts/chat-candidate
cargo test --locked --offline --release --bin toklet full_pinned_tokenizer_matches_frozen_ids -- --ignored --nocapture
python3 tools/reference/cli.py --output artifacts/qwen-cli-check
```

Both exporters require new output directories. `cli.py` verifies local source
hashes, exercises the release CLI in one persistent child, checks short greedy
output, pending/cached state, raw append, chat reset, prefill interruption and
recovery, over-capacity rejection, and its own child's physical footprint on
macOS. It retains all events and binary/driver identities. A watchdog bounds a
hang; the child is always released. This is functional qualification, separate
from workload benchmarks and held-out quality.

`coreml_slice.py` prepares [coreml-slice-v1](../../tests/fixtures/coreml-slice-v1/manifest.json)
before any Core ML evaluation. Its 42 arrays (36,992 payload bytes) derive from
the immutable tiny-v1 first layer: biased query projection and already-rotated
causal GQA attention at one, four, offset, changed-future and 32-token capacity
cases. Persistent K/V expectations use separate `[2,32,8]` head-major FP16
buffers with exact zero invalid suffixes. Rotary execution and a complete
decoder are outside this slice.

The independent NumPy implementation uses ordered scalar FP64 products/sums
and stable softmax. FP32 projection keeps the original `1e-5 + 1e-4*abs(expected)`
gate. FP16 projection/attention gates use twice the maximum independently
reconstructed FP16 storage/operation-rounding discrepancy against the saved
FP32 expected values, plus an absolute `1e-5` floor and `1e-4` relative term.
Attention reconstruction covers dots/scale with and without FP16 intermediate
rounding, and rounds probabilities and context storage. Gates are fixed before
target outputs are inspected; they do not establish a full-model FP16 gate.

```sh
.venv/bin/python tools/reference/coreml_slice.py --check tests/fixtures/coreml-slice-v1
.venv/bin/python tools/reference/coreml_slice.py --build artifacts/coreml-slice-candidate
.venv/bin/python -m unittest discover -s tools/reference -p 'test_coreml_slice.py'
```

Export refuses existing directories. Verification reconstructs every array and
gate independently and compares exact bytes, so enlarging a manifest tolerance
cannot make a target pass. No Core ML import, conversion, runtime execution,
ANE placement, model download, quality score, or engine selection is involved.
