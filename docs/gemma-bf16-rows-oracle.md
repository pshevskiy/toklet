# Original Gemma BF16 row and scalar references

The [stdlib producer](../tools/reference/gemma4_bf16_rows_oracle.py) and
[independent replay](../tools/reference/check_gemma4_bf16_rows_oracle.py) now
freeze original BF16 row bits and two scalar projection references before
testing the [bounded target reader](bf16-source.md) on the official checkpoint.
This prepares the unquantized precision/quality reference; it does not execute
the model or qualify an app, device, context window or arithmetic profile.

## Inputs and frozen selection

Both source files are authenticated in full through retained regular-file
descriptors: the [official baseline](gemma-unquantized-checkpoint.md) is
10246621918 bytes, SHA `2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550`;
the original four-bit candidate is 3550670554 bytes, SHA
`038e39a37a7667373d2c3991375446b10c96ae1d717a68674870343db376b76e`.
The fixed official header and independently prepared
[storage inventory](gemma-resident-storage-reference.md) determine tensor names,
shapes and offsets. All 540 selected tensors and 60 unused shared-KV consumers
are checked. The latter remain excluded from the graph's selected weights.

For each of 278 matrices, freeze rows zero, one, middle and last. For each of
262 vectors, freeze its complete contents. One additional case freezes the
last 128 rows of the 12288-column layer-15 down projection. Repeated and
nonmonotonic gather schedules are fixed in the manifest for later target tests.
The 541 cases contain 4683043 FP32 values, independently expanded from BF16 by
integer word operations. Comparison includes exact shapes, dtypes and bits.
Sampled matrix rows do not separately prove every matrix value.

All 262 candidate vector excerpts are also saved and compared with their
previous component hashes and official BF16 bytes. All match exactly. This
establishes selected vector lineage; it does not establish quantized-matrix or
whole-graph numerical agreement.

Two complete projections use original layer-0 K weights `[256,1536]` and
layer-15 down weights `[1536,12288]`, each with two fixed dyadic input rows.
Scalar FP64 `math.fsum` and left-to-right FP64 accumulation produce independent
dot references. These are mathematical references over exact BF16 values,
not upstream BF16 execution. The existing FP32 `1e-5 + 1e-4*abs(reference)` gate
is retained; an additional FP64 `1e-13 + 1e-13*abs(reference)` gate is frozen
before target trials. It does not replace any existing graph or quality gate.

## Ownership and actual checks

No NumPy, Torch, target row adapter or engine is imported by the producer or
checker. Full authentication uses bounded reads; each excerpt checks the same
descriptor/path identity before and after reading. Output files are exclusive,
and source/cleanup failures retain the primary error. Consumed official and
candidate bytes are saved before finite/equality gates. Partial binary data,
structured attempts, diagnostics and failure status remain independently
captured. Each binary has a 64 MiB ceiling.

Eleven [offline tests](../tools/reference/test_gemma4_bf16_rows_oracle.py) cover
geometry, exact conversion, signed zero/subnormals, each infinity/NaN control,
bounds, source replacement, deadlines and cleanup. They need no downloaded model
or numerical runtime. Thirteen captured-data controls reject wrong tensor,
row order, shape, source offset/bytes, FP32 bits, missing/duplicate cases, source,
header or inventory identity, projection input and scalar result. Ten controls
admit coherently updated binary hashes before rejecting the intended semantic
error; three reject the deliberately wrong source identities during admission.

On MacBookPro18,3 / Apple M1 Pro / 16 GiB / macOS 26.6.2 (25G83), capture,
independent replay and all controls pass. Physical peaks are 92046128,
141067032 and 264324080 bytes; corresponding RSS peaks are 119586816,
158892032 and 425066496 bytes, each below the 512 MiB diagnostic policy.
These include reference instrumentation and are not model-fit or latency results.
All seven collection process groups close. Rust, shaders, examples and protected
release binaries are unchanged; no Rust/GPU/model suite was rerun for this
stdlib-only reference task.

The [committed manifest](../tests/fixtures/gemma-bf16-rows-v1/manifest.json) is
776091 bytes, SHA `a17dede82620277d77149134750aa7e51abc72cf8995d6481799291e140113cc`.
The ignored capture retains 18900108 array bytes and 48473228 source-excerpt
bytes under `artifacts/m8-gemma-bf16-oracle-checks-v1/oracle/`; their hashes are
in the manifest. See the [public evidence](../benchmarks/results/2026-10-26-gemma-bf16-rows-oracle-v1.json).

The subsequent [target checks](gemma-bf16-reader.md) now pass positive official
binding, grouped/split/repeated access, recovery and bounded projections, with
independent saved-output/read-chain replay. Next compose
the unquantized graph using the existing weight interface, separate numerical
context from chunk scratch, and run the complete quality workload. Historical
precision failures and full model-window/app/phone/architecture gates stay open.
