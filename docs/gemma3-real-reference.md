# Gemma3 real Half reference

29 October 2026. The independent CPU reference exports and exactly replays the
selected original MLX Gemma3 270M four-bit artifact at revision
`ff1143e3a10547c9f2129e94ca37059b096b23f4`. This establishes a frozen comparison
source for the bounded native Mac journey. It does not qualify native inference,
application Chat, original-unquantized quality or iPhone execution.

## Original bytes and arithmetic

The producer reads the authenticated original container,127 affine packed
matrices and109 BF16 normalization vectors. It reconstructs each matrix once in
FP32 from the original nibble/scale/bias values. Effective coefficients are
FP32 `1 + raw BF16 gamma`, formed once without Half coefficient rounding. The
embedding and head share the same original table. The original Google source
revision remains separately unknown; this is a same-artifact reference.

MAIN is the explicit `NativeHalf_clipped_residual_numpy23_blas_v2` profile:
NumPy2.3.2 FP32 BLAS projections/dots/context, declared FP32 softmax, sequential
column Neumaier FP32 RMS, FP32 rotary operations and explicit Half stores.
Embedding scale is25.25. GELU stores Half before gating. Both branch residuals
add finite Half operands in FP32, clip to±65,504, and store Half. WIDE widens the
same reconstructed FP32 weights per operation, uses FP64 reductions and
Decimal80 rotary preparation, and preserves the same visible stores and explicit
FP32 clipped residual operation. Neither profile imitates Metal compensation.

The first unclipped export failed cleanly at layer8; its original398 groups and
669,568 bytes remain intact. Both corrected jobs preserve that exact byte prefix
and all398 descriptors. The correction follows pinned MLX-LM0.26.3 FP16 behavior;
it changes no gamma interpretation or other gate. See the
[separately qualified residual operation](gemma3-half-residual.md).

## Fixed probes and outputs

Three frozen inputs exercise two short plain-text prompts and a544-token
sequence. All400 stage identities are captured at eight selected absolute rows:
0/5,0/10 and511/512/513/543. The18-layer graph preserves fifteen local512-window
layers and three global layers at5/11/17. Scores retain raw dot values for all
materialized keys; only excluded probabilities are zero. Native comparisons
must align actual absolute key columns rather than invent omitted score cells.

The export contains3,200 graph groups and32 teacher-history head groups:
6,464 arrays,13,473,568 values per profile and97,110,272 array bytes. Selected
head rows avoid a544×vocabulary allocation. Two16-step histories are chosen by
WIDE; their MAIN greedy observations are diagnostic. These histories freeze
native comparison inputs, not a cross-profile exact-text quality gate. Native
free generation, codec/stop checks and application rendering remain separate.

MAIN is the fatal numerical reference. Each frozen group uses
`atol=max(0.0001,2*observed_main_wide_envelope)` and `rtol=0.001`;
WIDE remains diagnostic. The policy was fixed prospectively. Independent
saved-byte checks validate shapes, exact spans/hashes, finite stores, envelopes
and19,752 masked probability zeros across both profiles. Tolerances are not
changed after seeing target output. Astra independently reconstructs all109
original norm conversions,368,640 residual stores including40 saturations,
589,824 gating stores and10,240 embedding-scale words from saved bytes.

## Runtime and limits

Twelve portable reader/arithmetic controls pass. Two isolated source groups
export and replay all output bytes identically. Framework Python3.12.8 launches
the ordinary pinned virtualenv with isolated mode and a fresh bytecode prefix;
NumPy2.3.2, tokenizers0.22.2, Jinja3.1.6 and MarkupSafe3.0.4 are authenticated.
No synthetic prefix mutation, GPU model or native numerical execution occurs in
these jobs. Original norm bytes and effective FP32 values have a separate proof.

The prospective source ledger admits4,931,096,570 bytes, including conversion,
one wide matrix, workspaces and explicit framework/BLAS allowances. Export and
replay take49.03/46.54 seconds including worker setup/guards. Their OS RSS peaks
are2,806,284,288 /2,735,538,176 bytes. Direct observations and process cleanup are
recorded; these are source-job resource results, not engine throughput or a
mobile memory budget. Both owned groups close before native loading begins.

See [machine-readable evidence](../benchmarks/results/2026-10-29-gemma3-real-reference-v1.json),
[verified acquisition](gemma3-artifact-acquisition.md) and [remaining tasks](../TODO.md).
