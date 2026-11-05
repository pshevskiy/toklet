# Sparse-MoE BF16 import and storage planning

5 November 2026. Adds a separate `mlx::granite_moe` source for the tied-head,
bias-free, ordinary-RoPE Granite MoE graph. It reuses the generic safetensors
header validator and authenticated file-delivery machinery. It does not extend
the dense model factory or grant inference/catalog eligibility.

Strict configuration and exact tensor inventory require BF16 weights. Fused
expert gate/up `[E,2I,H]` and down `[E,H,I]` stay intact; checked slice descriptors
select each expert without splitting or copying tables. Duplicate output heads,
quantization, shared experts, unsupported rotary options, alternate dtypes and
sharded indexes are rejected. Opening reads bounded metadata and authenticates
the full expected file with a reusable 64KiB buffer; it allocates no weight array.

Caller expectations bind repository/revision, config bytes and complete model
SHA256. They are not publisher authentication supplied by this API. Config bytes
are checked before parsing; the tensor prefix used for planning must equal the
subsequently authenticated prefix. Every physical payload range is reverified
as it is delivered from a retained regular-file descriptor. Path replacement
cannot redirect an existing import. Same-inode mutation, truncation, conversion
failure or wrong destination extent prevents completion.

`import_next` writes directly into a caller's final tight Half slice, whose
exact length is checked before writes. BF16 decodes through a scalar FP32 value;
no full FP32 or second Half model is retained. Nonfinite values and finite
BF16-to-Half overflow are rejected. IEEE conversion preserves signed zero and
representable Half subnormals; values rounded to zero are counted explicitly.
Any failed import requires discarding all partial destinations. Completion
returns a private-constructor historical delivery/conversion receipt, including
the converted Half digest. It does not attest destination lifetime, allocator
capacity beyond the passed slice, a live model owner, codec or GPU resource.

Storage planning reports exact tight destination bytes, largest tensor extent,
and retained importer Rust/table/name/range/stream capacities. A separate 64MiB
reservation covers transient metadata parsing and other overhead; it is not
labelled an actual allocation. Actual audit capacities and each destination
slice must match the admitted schema. No Metal padding/guards or session scratch
are invented before their consumer exists. KV bytes are explicitly prospective,
without implemented or qualified context-window admission.

The pinned real 218-tensor metadata gives:

- Tight Half weights: **2,669,256,704 bytes**; largest tensor 100,669,440.
- Importer owner/stream requirement: **105,304 bytes** on this build.
- Metadata reservation: **67,108,864 bytes**.
- Admitted load requirement: **2,736,470,872 bytes**.
- Prospective 512-token Half KV: 25,165,824 bytes, separate from that load total.

These are planning requirements, not measured process peaks. The real weight
payload has not been acquired or numerically qualified by this task. The source
is `ibm-granite/granite-3.1-1b-a400m-instruct`, revision
`0da7a48b0276d500ce5922fd2b33944091fc6c09`; expected complete model SHA256
`ac02591061f1344027a7e7b11dbb4143f75f166c47dc09b742f5de3ab1dde1d1`.

An independent authored BF16/Half bit oracle covers signed zero, Half subnormals
and underflow. Source/config/header snapshots and the oracle generator are
hash-bound; regeneration is byte-identical. Controls cover fused extents, dtype,
tied-head duplication, unsupported semantics, exact-budget-minus-one, destination
capacity, retained descriptor replacement, late mutation/truncation and poisoned
completion. A larger synthetic tensor crosses multiple 64KiB stream chunks while
retained import storage stays unchanged. An initial test-only sum type-inference
failure is preserved and corrected; no policy or acceptance gate was loosened.

[Fixture](../tests/fixtures/granite-moe-source-v1/manifest.json),
[checks](../artifacts/moe-import-checks-2026-11-05/).
Next: resident Metal expert/routing consumers and the exact session layout they
allocate, followed by real-model source/native state and memory testing.

All seven check jobs pass: formatting, strict Clippy/default+Metal, 574 CPU and
683 Metal-feature tests, release library build and seven optimized source controls.
Astra approves the final scoped code/evidence with no outstanding findings.
[Result](../benchmarks/results/2026-11-05-moe-bf16-import-v1.json).
