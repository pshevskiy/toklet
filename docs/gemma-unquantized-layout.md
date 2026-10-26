# Official Gemma E2B unquantized layout

The official conversion-lineage checkpoint now has an acquired, frozen tensor
header and independently checked text layout. This closes the header/geometry
question for the [quality baseline](gemma-quality-contract.md). It does not
verify the weight payload, admit a reference runtime, measure model fit or
execute numerical inference. G2.1/G5 and full model-window support remain open.
The subsequent [complete acquisition](gemma-unquantized-checkpoint.md) separately
verifies the advertised full payload; this report retains its header-only scope.

## Source and exact scope

The [official pinned checkpoint](https://huggingface.co/google/gemma-4-E2B-it/tree/70af34e20bd4b7a91f0de6b22675850c43922a03)
is revision `70af34e20bd4b7a91f0de6b22675850c43922a03`, matching the candidate's
conversion README. The authenticated API inventory advertises 10246621918 bytes
and full SHA-256 `2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550`.
Two bounded HTTPS range requests acquired bytes0–7 and the declared header, with
exact final Content-Range/length and identity encoding. Transfer diagnostics and
owned-process closure are retained. No repository code was executed remotely.

The 263952-byte header has SHA-256
`12740d6fe7a66b316040fa4d77471a8e1809498a71992b3364a6d5417d10662e`.
Range provenance and frozen header bytes are distinct from cryptographically
verifying the complete advertised file. The latter requires full payload
acquisition and hashing. Metadata is stored in the
[offline fixture](../tests/fixtures/gemma-unquantized-header-v1/README.md).

## Complete layout and the extra shared-KV tensors

All 2011 tensor descriptors are BF16. Every shape, scalar rank-zero extent,
byte offset, and the complete nonoverlapping coverage of the advertised payload
is checked. There are 600 text tensors and 1411 excluded media tensors. Media
descriptor validation grants no media execution support.

| Selection | Tensors | BF16 payload bytes |
| --- | ---: | ---: |
| Selected text matrices/vectors | 278 + 262 | 9257138758 |
| Unused shared-consumer K/V parameters | 60 | 37761024 |
| Excluded audio/vision/connector tensors | 1411 | 951458176 |

Official names use `model.language_model.*`; the candidate's equivalent names
use `language_model.model.*`. All 540 selected shapes agree with the previously
frozen packed-model inventory. No assumption about value equality follows.

The extra 60 names are exactly `k_proj.weight`, `v_proj.weight` and
`k_norm.weight` for layers15–34. The pinned Transformers5.7.0 implementation
does not construct these parameters for shared consumers, reuses the last
nonshared producer of the same attention type, and explicitly ignores these
unexpected parameters during loading. This selects producer13 for sliding
consumers and14 for global consumers. The full inspected modeling source has
SHA-256 `fd93159380556e46029260ae38e5ec53ffd750ecb9872d0e89f57b334a25ed4e`;
reviewed original line ranges and exact excerpts are retained in the evidence.
The inspector records every unused name and extent explicitly. It does not
silently discard unknown text options or infer aliases from equal shapes.

## Mechanism and checks

The [inspector](../tools/reference/gemma4_unquantized_layout.py) is a concrete
stdlib-only reference tool for this exact official header/config. It resolves
row-major names, shapes and absolute file offsets once. It imports no numerical
runtime and loads no tensor data. Duplicate/nonfinite JSON, invalid integer
types, unsupported dtypes/ranks, bad extents, gaps/overlaps and unknown text names
reject. The regular-file reader uses bounded reads and checks the same descriptor
and path at both endpoints, rejecting symlinks/FIFOs/directories.

The [independent checker](../tools/reference/check_gemma4_unquantized_layout.py)
does not import the producer. It derives selected geometry from the earlier
278-matrix/262-vector inventory and independently constructs the shared-consumer
exclusions. It compares all 2011 extents, all 540 selected tensors and all 60
unused entries, including exact output flags and byte totals.

Eleven [offline test methods](../tools/reference/test_gemma4_unquantized_layout.py)
pass, including eight coherent layout mutations whose external hashes are
updated before independent replay. The changed missing entry, offset, shape,
unused count, classification, payload-verification flag, advertised digest and
media accounting all reject. No dataset, model download or third-party package
is required by these tests.

The successful collection has producer physical peak14828168 bytes and maximum
RSS24592384, below its128MiB diagnostic policy, on Apple M1 Pro/16GiB/macOS26.6.2
(25G83). These are header-processing observations, not model memory measurements.
The first attempt's producer/layout and tests passed, but macOS denied the
timing utility's clock query. That failed receipt and streams remain; retry
with the required tool access reproduced identical layout bytes and obtained
the measurement. All10 acquisition/attempt/collection groups close. Core Rust,
existing tests/examples, shaders and protected release binaries are unchanged.

The [public evidence](../benchmarks/results/2026-10-26-gemma-unquantized-header-v1.json)
binds inputs, source snapshots, raw outputs, source-rule excerpts and process
receipts. Layout SHA-256 is
`2419f70e7232fd75a6d361e0ef8e8acaf37f87f52124e63e1e04e814d1040840`.

## Next reference and context work

The actual header confirms `[262144,8960]` PLE storage occupies4697620480 bytes,
and `[262144,1536]` tied embedding/head occupies805306368. Selected BF16 text
storage alone exceeds the6GiB Mac policy. Excluding the PLE table leaves
4559518278 bytes, which is not yet a whole-process fit proof. PLE lookup needs
17920 bytes per selected row; the output head needs all vocabulary rows for
every complete projection. A bounded tile/row route is therefore a concrete
candidate; its IO costs, conversions and arithmetic still require measurement.

The subsequent complete acquisition authenticates the official payload without
retaining a second full assembly copy. Next qualify bounded BF16 row reads and
same-consumption identity, and declare/verify the reference arithmetic profile. Compare short
operators/full graphs with an independent source and measure whole-process fit
before using it for scores. BF16 storage converted for FP32/FP64 operations is
not automatically an upstream BF16 execution match.

Separately decouple chunk scratch from context capacity, qualify current and
511/512/513 plus1024/1025 boundaries, and execute the entire frozen quality
suite. Retain all quality/precision gates. Progressive numerical/context quality,
resident scheduling, model-driven Chat/Expert and independent phone fit still
precede full shared-window qualification. This metadata task changes no app
eligibility, output cap or model context claim.
