# Heretic Gemma source and independent fidelity checks

G1.Ca is complete in its selected-context scope. The
[native diagnostic](gemma-heretic-native.md) has independent numerical checks
and a reproducible selected-text source lineage. Checks through 512 positions
and combined Astra review pass with no findings. Full held-out quality and
managed adoption remain separate work.

The selected MLX revision is
[`deadbydawn101/gemma-4-E2B-Heretic-Uncensored-mlx-4bit`](https://huggingface.co/deadbydawn101/gemma-4-E2B-Heretic-Uncensored-mlx-4bit/tree/cde01efcb11f8a12e88a953b752fc16062aa1486),
`cde01efcb11f8a12e88a953b752fc16062aa1486`. The modified high-precision parent is
separately pinned to
[`p-e-w/gemma-4-E2B-it-heretic-ara`](https://huggingface.co/p-e-w/gemma-4-E2B-it-heretic-ara/tree/c9a1d4c031981f14d86eeb0c7d87de7fafd34513),
`c9a1d4c031981f14d86eeb0c7d87de7fafd34513`. Its complete stored BF16 file is
10,246,621,918 bytes, SHA-256
`15605c16e479a8dc53928dd4b521a4ead9fe66222dc14d1ce392384e999a544b`.
The official original, modified BF16 parent and quantized artifact keep distinct
identities throughout the checks.

The parent header is byte-identical to the official BF16 header. Its config
differs only by `text_config.moe_intermediate_size: null`, a disabled branch.
These metadata observations grant no execution capability. Existing official
and affine inspectors remain strict.

All 262 BF16 norm vectors and the BF16 PLE projection match the parent bytes
exactly. A predeclared weight-only probe tests the first, middle and final rows
of five fixed matrices. Both tested MLX recipes reproduce those rows. The first
recipe then reproduces every row of all 277 consumed affine matrices: original
BF16 values, MLX 0.32.3 affine quantization, group size 64, four bits, BF16 scale
and bias. Codes and metadata match byte-for-byte. Together the two checks cover
all 1,094 selected physical components and 2,623,765,062 candidate bytes.

This demonstrates a reproducible transformation from the pinned parent. It does
not identify the publisher's historical tooling or qualify omitted media and
unused-KV arrays. The verifier streams at most 128 rows per tile and leaves both
source files unchanged. Its measured process peak is 92.7 MB; the MLX allocator
peak is separately 4.03 MB. These are weight-verifier measurements, not inference
fit, memory-saving or engine performance claims.

The numerical reference imports no Rust engine code. It binds the pinned
277 affine matrices and one BF16 projection to existing bounded row readers,
then runs the unchanged independent Gemma graph. Separate contracts use FP64
activations/reductions and staged FP32 operator boundaries; both preserve the
existing Decimal80 rotary definition and gate:

`abs(actual - reference) <= 1e-5 + 1e-4 * abs(reference)`.

On the frozen 15-token public prompt, every value in the final 262,144-logit row
passes both contracts. Maximum absolute errors are 6.07e-6 against FP64 and
1.335e-5 against staged FP32; greedy token 818 agrees. The final collector and
current native capture produce identical comparison bytes to the initial run.
Reference memory peaks are about 85 MB, distinct from the native owner's fit.

The native 512-position raw-All capture exports full rows at positions
0, 15, 30, 31, 63, 255 and 511. It verifies exact committed history, stable
allocation and unchanged valid state after one-token-over-capacity rejection.
Its observed lifetime physical peak is 2.942 GB, below 6 GiB. The independent
32-, 64- and 512-position phases pass all 22 selected row/contract comparisons:
5,767,168 values, zero failed values and no greedy mismatch. All repeated
reference prefixes match exactly, including signed-zero bits in staged FP32.
The largest error uses 0.479 of the unchanged gate. The 512-position reference
process peaks at 596 MB; source and runtime endpoint checks pass. This qualifies
the recorded selected distributions, not every possible prompt or context.

The offline checker validates complete bodies, hashes, source/workload identity,
bounded filenames, finite values, typed memory, every selected logit and causal
prefix compatibility. Staged prefix agreement includes signed-zero bits;
mathematical prefix agreement retains the existing 1e-10 compatibility limit.
Near-tied greedy differences remain explicit investigation cases. Owned
execution/source proofs complement the checker; a green consistency result
alone does not establish execution or adoption.

Fresh formatting, strict CPU/Metal Clippy and 610 CPU/742 Metal-feature tests
pass (33/295 ignored). Thirteen CPU adapter/capture controls and five synthetic
checker controls pass. A checker validation-order failure was fixed and retained
in the logs. The only Rust changes add an ignored capture test; production,
shaders, ABI, Swift and dependencies are unchanged. Native50 package/C smoke
evidence is reused by source equivalence; the new frozen release binary runs
the raw512 and full public-prompt/fault-replay controls explicitly.

The frozen complete quality contract is retained at the documented artifact
path, with all input hashes verified: 10,000 scored targets, 100 tasks
(40 code, 30 prose, 30 continuation), and the original 0.05-nat/2-point gates.
Its loss workload needs 1,025 positions; the largest task needs 567. These
require separately qualified test-only probes. The public 512-position/chunk32
limit and 6 GiB ceiling remain unchanged. Generated code belongs in the existing
isolated grader. No full quality, codec/template, managed UI or iPhone adoption
is claimed here.

[Contract](../artifacts/gemma-heretic-quality-2026-11-08/acceptance-contract.json),
[short fidelity](../artifacts/gemma-heretic-quality-2026-11-08/short-fidelity-final.json),
[32/64 comparisons](../artifacts/gemma-heretic-quality-2026-11-08/context-fidelity-progress.json),
[complete context comparisons](../artifacts/gemma-heretic-quality-2026-11-08/context-fidelity-final.json),
[raw512 capture](../artifacts/gemma-heretic-quality-2026-11-08/native-context-1/checks.json),
[packed reproduction](../artifacts/gemma-heretic-quality-2026-11-08/packed-full-1/capture/result.json),
[selected-text lineage summary](../artifacts/gemma-heretic-quality-2026-11-08/selected-text-reproduction.json),
[reproduction memory/source receipt](../artifacts/gemma-heretic-quality-2026-11-08/packed-full-1/checks.json),
[unquantized components](../artifacts/gemma-heretic-quality-2026-11-08/unquantized-component-lineage/checks.json),
[quality input inventory](../artifacts/gemma-heretic-quality-2026-11-08/quality-input-inventory.json),
[required checks](../artifacts/gemma-heretic-quality-2026-11-08/required-checks-1/checks.json).

[Astra review](../artifacts/gemma-heretic-quality-2026-11-08/review.json),
[checks/review/commit](../artifacts/gemma-heretic-quality-2026-11-08/publication-checks.json).
