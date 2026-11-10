# MiniCPM5 bounded matrix reproduction

The selected Heretic MLX8 artifact reproduces exactly from its pinned BF16
parent using the prospectively frozen MLX 0.32.3 GPU affine8 recipe. Both whole
parent objects authenticate in the same pass that reconstructs all **296
matrices/888 physical quantized arrays**. There are **zero mismatching bytes**;
all 85 copied BF16 norms also match. Checks and Astra review pass.

This closes the bounded conversion-reproduction gate. It does not
authenticate the publisher's exact binary version or qualify model quality,
deeper contexts, managed/default adoption, engine speed or summary overhead.

## Identity and recipe

The child is
`bumblebuttpow/MiniCPM5-2B-heretic-abliterated-MLX-8bit` at `50ab1b22`, SHA-256
`6fadca0fcdefe8b77fc144851ee2c2fd3c8949ccc0ec756b8ada27e3156c8910`.
The parent is `insraq/MiniCPM5-2B-heretic-abliterated` at
`8d83eaf42e30f4e4f164b8685be0ecdccd8ef0d8`. The earlier
[parent audit](minicpm5-parent-auth.md) pins both objects; this campaign freshly
authenticates their complete 5,033,557,128 bytes.

The [prospective contract](../artifacts/minicpm5-matrix-lineage-2026-11-10/prospective-contract-1.json)
freezes the recipe before any quantizer or parent-matrix outcomes: GPU, affine
8-bit, groups of 64, original BF16 input bits, at most 256 rows per tile and no
MLX allocation cache. There is no FP16 cast or tolerance. The publisher declares
MLX-LM 0.31.3, whose dependency admits multiple MLX versions; it does not identify
one exact publisher binary. Seven official source captures are retained in the
[research receipts](../artifacts/minicpm5-matrix-lineage-2026-11-10/upstream-source-receipts-1.json).
The relevant affine quantization function bodies are identical in MLX 0.31.2 and
0.32.3, as recorded in the
[source comparison](../artifacts/minicpm5-matrix-lineage-2026-11-10/source-recipe-comparison-1.json).

CPU and GPU rounding/constant-negative metadata differ. Six hand-derived literal
controls cover both devices, including ties, byte packing and BF16 values beyond
FP16's finite range. All 12 actual controls pass under an admitted numerical
runtime. Three frozen parent range tiles then reproduce all child fields, before
the full stream starts. Range hashes are prior authenticated stream witnesses;
they are not fresh whole-object authentication or Merkle proofs.
[Literal controls](../artifacts/minicpm5-matrix-lineage-2026-11-10/literal-probe-2/checks.json),
[range controls](../artifacts/minicpm5-matrix-lineage-2026-11-10/range-preflight-1/checks.json).

## Layout, ownership and clocks

Parent matrices are contiguous BF16 `[out, in]`. Child codes are little-endian
U32 `[out, in/4]`, with four 8-bit codes packed from low to high bits. Scales and
biases are BF16 `[out, in/64]`. Norms remain separate BF16 vectors. All logical
shapes, physical fields, offsets and complete file extents are checked before
conversion. Child reads use one regular, non-symlink descriptor; both that
consumed descriptor and its path hash are checked before and after the run.

The unchanged parent authenticator still consumes at most 4 MiB per read and
hashes every source byte, each tensor and each block. A coarse stream adapter
accumulates at most one 256-row tile, reconstructs three child buffers and
compares every byte. Tiles cannot advance past a failed conversion; replay is
required after an exception. Matrix mismatches remain failures and cannot be
rescued by changing the recipe. Both objects must close with matching whole
hashes and successful HTTPS receipts before their journals become usable.

Each tile records separate child-read, completed quantization/readback and
comparison/hash elapsed clocks. GPU work is evaluated and synchronized before
readback. The 4,128 tiles cover 2,516,582,400 source values; completed conversion
phase clocks total 22.068 seconds. These are diagnostic audit clocks, without
warmups or matched inference workloads, and support no engine speed claim.

Numerical imports occur only after an externally hashed manifest admits all
3,287 source/runtime/native files, exact NumPy 2.3.2/MLX 0.32.3 versions and a
fresh absolute bytecode-cache prefix. Initial and endpoint reports bind actual
module origins and non-system dyld images; endpoint file hashes are rechecked.
No remote model code or model inference executes.

## Result and resource boundary

| Object | Matrices | Tiles | Exact norms | Whole-object authentication |
| --- | ---: | ---: | ---: | --- |
| `model-00001-of-00002.safetensors` | 291 | 4,086 | 83 | 4,989,507,944 bytes; pinned SHA-256 matches |
| `model-00002-of-00002.safetensors` | 5 | 42 | 2 | 44,049,184 bytes; pinned SHA-256 matches |

The separately declared **512 MiB source-audit allowance** includes numerical
runtime and bounded CPU/GPU tiles. It does not revise the earlier identity-only
256 MiB audit, engine's 6 GiB, codec's 512 MiB or 2 GiB free-disk floor. No 5 GB
parent payload is stored. The owned process group closes with 4,035 observations:

- Final auditor peak RSS: 306,790,400 bytes. Adding both curl peaks gives a
  conservative 322,764,800 bytes, below 512 MiB.
- Maximum sampled aggregate RSS: 306,790,400 bytes; aggregate sampled physical
  and lifetime physical counters: 260,064,264 bytes. Final physical peak is
  unavailable.
- Minimum observed free disk: 28,663,189,504 bytes.

These observations describe the source audit under its recorded allocator;
they establish no inference fit or memory-saving claim.
[Compact result](../artifacts/minicpm5-matrix-lineage-2026-11-10/observed-result-1.json),
[owned checks](../artifacts/minicpm5-matrix-lineage-2026-11-10/auth-checks-1.json),
[full manifest](../artifacts/minicpm5-matrix-lineage-2026-11-10/lineage-1/manifest.json).

## Rechecking and remaining gates

The offline consumer validates source/transport/runtime identity, typed clocks,
row coverage, all child row hashes and complete matrix aggregates, exact copied
norms and raw process ownership/resource observations. It independently rehashes
the retained child. It seals all raw evidence and rejects incomplete journals.
Single-tile source hashes bind directly to authenticated tensor hashes.

**Discarded BF16 parent tiles cannot be rehashed or requantized offline.** For
multi-tile tensors, well-formed tile hashes remain producer witnesses, without
Merkle membership proofs. Changing such a hash can remain structurally valid
but changes the evidence seal. Source-reproduction trust therefore includes the
frozen, reviewed full-stream producer; a fresh authentication requires another
stream. [Current offline seal](../artifacts/minicpm5-matrix-lineage-2026-11-10/witness-seal-2.json).

All 35 portable tests pass, including independent short-read/tail/poisoned-stream
fixtures and retained-witness controls. Actual copied outputs pass one positive
and 34 rejections for incomplete coverage, altered child hashes, source bindings,
transport, qualification flags, clocks, runtime identity and memory ownership.
One well-formed changed multi-tile source hash is explicitly accepted
structurally and changes the seal; this is the second structural acceptance.
Fresh seals reproduce byte-for-byte. An added reader-only guard rejects
nonempty sampling errors; all raw producer/evidence hashes remain unchanged.
[Fresh tests/seal replay](../artifacts/minicpm5-matrix-lineage-2026-11-10/fresh-checks-1.json),
[actual-output controls](../artifacts/minicpm5-matrix-lineage-2026-11-10/witness-controls-2/checks.json).

Required Rust/native checks are reused only after every 4,031 Cargo/fixture and
421 native source input plus 10 packaged outputs match reviewed `d88a8ebd` /
native67. This change adds separate Python source-verification tools, without
changing Rust, kernels or session behavior. Existing required evidence covers
679 CPU/821 Metal tests, strict Clippy, three native slices and C smokes. Fresh
Python and actual-output controls cover the changed tools. Formatting also
passes freshly. [Exact reuse proof](../artifacts/minicpm5-matrix-lineage-2026-11-10/required-native-reuse-1.json).

Astra approves scoped retention and commit with no actionable findings, after
independently running the 35 portable tests, rehashing checked inputs and
reproducing the offline seal exactly. Only administrative documentation closure
follows; reviewed code, contracts and raw evidence remain frozen.
[Review](../artifacts/minicpm5-matrix-lineage-2026-11-10/astra-review-result-1.json),
[commit/integration](../artifacts/minicpm5-matrix-lineage-2026-11-10/publication-checks.json).

The original 10k-target/100-task quality, deeper numerical/context, default
allocator/adoption and 1% summary-overhead gates remain open. Screen capture and
physical iPhone work remain deferred. The full improvement goal continues.
