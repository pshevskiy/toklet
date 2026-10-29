# B4 candidate: a shared short-convolution state block

29 October2026. This is the next mechanism after B3's bounded Gemma3 checks,
not an executable-model admission. Astra's cached primary-source research
identifies [mlx-community/LFM2-350M-4bit](https://huggingface.co/mlx-community/LFM2-350M-4bit/tree/18dc72abf3b2337f9123cfd6eeeb58dfa7947066)
revision `18dc72abf3b2337f9123cfd6eeeb58dfa7947066` as a practical representative
for recurrent/hybrid state. The [complete original artifact](lfm2-artifact-acquisition.md)
is now verified locally; real numerical execution remains pending.

## Why this mechanism

The pinned configuration has 16 layers: full attention at2/5/8/10/12/14 and
ten length3 depthwise short-convolution layers. Hidden width1024, attention
heads16/KV8/head64, actual FF width4608 and tied vocabulary65536 fit a compact
model. Projection,
ordinary split-half RoPE, Q/K RMS, SwiGLU, residual and full KV mechanisms have
existing counterparts. The new mechanism is a bounded convolution history that
must publish atomically with all attention KV and token state.

The authenticated header inventory has93 Q4/group64/BF16 affine matrices,
45 direct-format norm tensors and ten BF16 convolution tensors `[1024,3,1]`:
334 physical descriptors/148 logical owners. “Direct-format” describes the
stored tensor, not a qualified mathematical gamma interpretation. The advertised
weight file is199,527,820 bytes, payload199,491,072 bytes, LFS SHA256
`558353ad098c1230a8e2ea508cd34ef311a1b97a2bc4f47498f4df8229dc5c0f`.
Full-file and complete bundle verification now pass in B4e1; numerical
execution remains separate.

The source planning ledger is4,519,922,060 bytes and native cold ledger
3,552,479,116 bytes, each below the existing6GiB ceiling. These include explicit
conversion/framework allowances and are estimates, not observed process fit.
The declared128000 positions do not imply an initial implemented/qualified
window. Propose a bounded512-position Mac vertical; freeze its exact inputs and
limits before execution.

## Small reusable block and ownership

Resolve the fused input projection into checked B/C/x row spans at loading.
For each channel, the source mechanism computes `u[t] = B[t] * x[t]`, then
`z[t] = w[0]*u[t-2] + w[1]*u[t-1] + w[2]*u[t]`, and projects `C[t] * z[t]`.
Initial missing history is zero. Preserve the primary source's exact coefficient
orientation, operation/store precision and mask semantics in independent
fixtures before adopting this expression as a numerical contract.

Keep immutable convolution weights with the model. Session-owned history stores
the two preceding u rows per channel: `[history_row, hidden]`, contiguous across
hidden channels. Use separate committed/candidate banks and reuse bounded chunk
scratch. The ten layers require40,960 Half bytes per bank before guards and
alignment; FP32 would require81,920. Choose and qualify precision explicitly.
Reserve this storage once; avoid per-token allocation or whole-context copies.

A successful full step publishes candidate convolution state, attention KV and
committed positions together. Any failure/cancellation must retain the previous
valid prefix or require explicit replay. A convolution-layer success alone must
not advance the session. Reset, cached-prefix reuse, pending sampled tokens and
model/settings changes obey the same session identity rules as KV.

Use concrete functions over slices and a small layer-mechanism enum selected
outside numerical loops. Keep scalar reference and resident backend leaves
separate. Extend existing projection/norm/attention blocks only where actual
shapes require it; do not add a second graph framework or model-name fallback.

## Ordered tasks and acceptance

| Task | Delivery | Required evidence |
| --- | --- | --- |
| B4a | Strict config/header/source plan and text inventory | Original row ownership, convolution orientation, actual FF width, direct gamma provenance, unsupported-field rejection, exact memory equations/overflow controls |
| B4b | Scalar short-convolution operation and transactional history | Independent initial/one-/two-/three-row and multi-chunk fixtures; coefficient orientation, mask/tail, exact prefix/reset, failure/retry/publication controls, steady-state allocations |
| B4c | Tiny hybrid graph using shared projections/norm/attention/MLP | Independent complete stages and physical state across mixed attention/convolution layers; whole-step rollback, causal and partition checks |
| B4d | Resident backend block and hybrid composition | Original resident weights, persistent history/scratch, independent leaf/tiny comparisons, guards/status/recovery, physical-owner accounting and completed-work measurements |
| B4e | Acquire and qualify the fitting real representative | Immutable full-file hashes, fixed source/native/text protocol, bounded512 generation/session comparisons and measured process fit, independent Astra review |

Each code task gets relevant ordinary checks, Astra review, progress/improvement
updates and a commit. App admission/context controls follow through B5; original
unquantized10k/100 adoption quality remains M8 closure work. No need to qualify
128000 positions or add MoE/MLA before delivering this concrete mechanism.

## Known source and text questions

The card declares mlx-lm0.26.0, but its tag
`9ee2b7358f5e258af7b31a8561acfbbe56ad5085` lacks `mlx_lm/models/lfm2.py`.
The exact conversion runtime therefore remains unknown. The inspected
[MLX-LM implementation](https://github.com/ml-explore/mlx-lm/blob/1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7/mlx_lm/models/lfm2.py)
is pinned separately; do not label it the conversion revision or infer norm
semantics from BF16 dtype. Choose an explicit current pinned-loader same-artifact
profile after authenticating its complete load/sanitize/quantization path,
`nn.RMSNorm` implementation and actual payload/header. Its direct-weight norm
semantics can be independently corroborated by the inspected Transformers
implementation. Historical converter identity may remain unknown; it does not
require recreating that converter to qualify the explicitly declared profile.
Resolve this interpretation before exporting a real numerical reference.

### B4a resolved loader and native RMS profile

Astra authenticates the complete MLX-LM loader/sanitizer/quantization/strict
parameter replacement path at the revision above. Its actual dependency requires
MLX≥0.32.3 on Darwin; choose the compatible source pairing MLXv0.32.3 commit
`64ea011cb65f14d9ce2737e60db9a4ae91ed7441`. This is a pinned semantic source
pairing, not an installed or executed runtime claim. Strict loading replaces
the stored norm arrays directly. The LFM2 sanitizer only conditionally reorients
convolution axes; it performs no norm fold or unit offset. The selected already
oriented `[1024,3,1]` tensors are unchanged. Independent Transformers source
corroborates direct gain and the causal B/C/x mechanism.

The prospective **native RMS** choice is explicit: Half input → FP32 Neumaier
squares/reduction/inverse with epsilonFP32(1e-5) → multiply by the exact FP32
widening of the stored BF16 gain → Half store. There is no plus-one gain
preparation. Reuse the existing single-store FP32-scale operation; preserve
Gemma's separate unit-offset preparation and arithmetic unchanged. This avoids a
second normalization kernel/rounding variant and retains the mathematical
direct-gain operation. It does not claim MLX trajectory identity: pinned MLX
casts the normalized value to its output dtype before multiplying the weight.
Original-unquantized adoption quality remains a separate mandatory check.

At B4a freeze, native RMS admitted only epsilon1e-6 and widths32/64/256/640.
B4d1 now qualifies epsilon1e-5 at **head64, tiny hidden256 and real hidden1024**,
including the required512-head-row shape; this is more than a host-bound exception. Likewise, Gemma's Q4/KV1/D32-or256 staged attention is not
LFM2's Q16/KV8/D64 full GQA. Deliberately qualify generic dense GQA reuse or a
bounded second-consumer extension; a family rename does not grant support.
Other native operator/store contracts and tiny fixtures must be frozen before
their numerical runs. LFM2 ordinary residuals inherit no Gemma clipping.

B4a remains a metadata-only code task: strict config/header/index inspection,
93 affine/45 direct-norm/10 depthwise owners,30 borrowed B/C/x views and16
concrete layer recipes with **six KV** and **ten history** owner IDs. It reports
declared128000 and no implemented/qualified capacity. No payload, kernel/session
construction, text execution or backend eligibility follows from that plan.

Immutable adviser package SHA256:
`36755a8a530a317631b2c4d8366078ac56166aef1cf5561f9008416fef204e1d`;
independent source/profile/design review SHA256:
`07ef42c0eafd26ac6b9d8bd58a0d3d59cd82134375b0cf1a66d10191e850bb9a`.
The review includes the epsilon qualification addendum; the original package
remains unchanged. Both live under `artifacts/new-goal-b4a-loader-profile-*`.

The tokenizer maps64,400 IDs while the head has65,536 rows; unmapped IDs must
have explicit behavior. BOS1 is present in both template and tokenizer
postprocessing, so prevent duplication. The declared stop union is7; token2
is not automatically a stop. Template rendering, decoding and chat support need
their own fixtures, outside the numerical block. No arbitrary template execution
or silent vocabulary clipping is authorized by successful graph loading.

Cached research lives under `artifacts/new-goal-b4-lfm2-research-v1`, including
primary source files, immutable metadata URLs, header prefix and
`metadata-audit.json`. Its full-file identity and physical fit fields explicitly
remain unqualified.

Metadata audit SHA256:
`bc1cba0789d34e5a73dead22f24c9dcd0532d4c0fe5c4973cb3b312a6861a4cc`;
inspected MLX-LM model source SHA256:
`558dcbec61442b1f7216c0122d08d44083ea71d1510be23f4f934e62b0f71376`.

## B4a implementation result

The [cold adapter](lfm2-cold-plan.md) is implemented with the exact inventory above,
strict invalid metadata controls and checked hypothetical state equations. Full
default/Metal checks pass; numerical, payload and backend admissions remain
absent. B4b starts by reusing the already implemented scalar causal-depthwise
step, then validating the concrete transactional history integration.

## B4b implementation result

The [standalone scalar history owner](short-conv3-history.md) reuses the existing
convolution leaf unchanged and passes independent fixture, transactional failure,
identity/reset and zero warmed Rust allocation controls. Complete default/Metal
checks pass. Only convolution state is published here; B4c must validate KV,
history and session publication together. The next tiny witness is planned
at H256/I256/V64, Q4/KV2/D64, capacity12 and Conv→Attention→Conv. This tests
multi-KV layout and actual head64 while retaining a small independent reference;
it does not qualify the real16-layer or any Half graph.

## B4c1 source result

The [independent tiny scalar reference](lfm2-scalar-reference.md) is frozen after
eight portable controls, exact six-file export/replay and Astra's complete
saved-array/state audit. It has four cases,252 groups and548,944 finite values.
The selected tiny geometry remains separate from real16-layer and native Half
admission. B4c2 compares actual pre-/post-chunk committed state at its real
boundaries; tokenwise execution covers every source prefix. Inactive source KV
zeros do not impose tail clearing on reset.

## B4c2 scalar composition result

The [tiny scalar graph](lfm2-scalar-hybrid.md) passes independent stage/state,
partition, failure/retry, compound publication/reset and warmed allocation checks.
All full default/Metal checks pass. Only this synthetic CPU profile closes B4c.

B4d starts with the existing FP32-scale Half RMS operation: prospective direct
gains/epsilon1e-5 at head64, tiny hidden256 and real hidden1024, including up to
512 rows (real16-query-head ×32-token chunks). Preserve Gemma epsilon1e-6 cases.
Then qualify a borrowed three-tap Half operation and true multi-KV staged GQA
(tiny4:2/D64 and real16:8/D64) before resident composition. Existing dense GQA
kernels alone do not provide old+staged transactional views. Independent finite
artifact acquisition may overlap this work; it does not qualify any decoder.

## B4e1 acquisition result

The [nine-file original bundle](lfm2-artifact-acquisition.md) is verified locally
with complete model LFS SHA, all metadata identities and334 tensor extents. This
finite transfer overlaps leaf preparation only; no model/tokenizer execution,
128000-context admission or app qualification follows. Real source and native
journeys remain B4e2/B4e3 after resident mechanism qualification.

## B4d1 native RMS result

The [direct-gain Half RMS leaf](lfm2-half-rms.md) passes selected Mac release
qualification, all source/exact witnesses and unchanged legacy Gemma checks.
Its1e-5 profile covers head64 up to512 rows and hidden256/1024 up to32 rows;
Plan encodes the actual admitted epsilon bits. This grants no model admission.

B4d2 next implements only raw three-tap convolution over already stored Halfu,
immutable old[2,H] and a separate candidate[2,H]. Preserve separate FP32 products
and oldest-to-current sums before a Half output store. B*x/C*z and model-wide
publication are later concrete composition work; do not prematurely fuse these
into a new model-specific kernel. Multi-KV staging remains B4d3.

## B4d2 raw Conv3 result

The [borrowed raw Half Conv3 leaf](causal-conv3-half.md) passes selected Mac
release qualification, exact candidates and all233 source chunk boundaries.
Only stored rawu, direct FP32-widened BF16 tap coefficients, output/candidate and
sticky status are involved; B*x/C*z and model-wide publication remain later.

B4d3's all-query source is fixed prospectively at11cases/143arrays/22,405,366raw
bytes, including all32 deep queries at Q16/KV8/D64. Preserve fixed32-row head
strides and qualify each head's active fill/validation/copy intervals. A small
sampling draft is superseded planning with no outcomes. The existing Reference
arithmetic/legacy behavior remains the target, not a new attention algorithm.

## B4d3 multi-KV attention/cache result

The [borrowed GQA/cache blocks](staged-gqa-half.md) now pass full source and
Mac release qualification at the selected Q4/KV2/D64 and Q16/KV8/D64 cases.
Astra checks1,978,360 MAIN values, all physical per-head state and the actual
same-Set512→513 chain, plus seven legacy jobs. Six-owner failure/replay uses
capacity64; single-owner source cases include544. This does not admit a model.

B4d4 first adds the small strided Half product needed for B*x and C*z views into
one physical BCx parent. Existing cache-layout, packed projection, rotary and
activation blocks are reused. Then freeze an independent Half tiny-graph source
before concrete resident mixed-state publication/recovery. The scalar graph's
outputs alone are not a Half reference. Avoid a generic graph VM or new fused
model-specific kernels for these concrete uses.

## B4d4a strided-product result

The [borrowed strided Half product](strided-half-product.md) passes independent
source, full cold checks and selected Mac release qualification for both BCx
uses. Shared readonly parents are counted once and every writable alias rejects.
Astra reconstructs all347,384 finite native outputs exactly, including342,784
fixed MAIN/WIDE comparisons, padding,96 partition boundaries and10 zero-allocation
encodes. The leaf stays test-only until the mixed graph owns it.

B4d4b's separately approved protocol/source now targets the same original tiny
container with exact integer-dyadic affine sums and direct Half/FP32 rounding.
Its twelve controls, bounded export/replay and independent saved-array audit
now pass as recorded below; neither scalar outputs nor native outcomes set its
center. B4d4c next composes the concrete mixed-state publication boundary.

## B4d4b independent Half source result

The [independent Half tiny reference](lfm2-half-reference.md) is qualified before
native graph outcomes:12 controls, exact four-file export/replay and Astra's
complete504-descriptor/1,097,888-value reconstruction pass. All252 stage gates,
original packed weights, raw-u histories, KV snapshots, masks and components are
verified. The first export's diagnostic builtin-sum assumption is corrected in
a reviewed successor without changing any graph math or gate; failed evidence
remains. The original container is shared, with no scalar golden-output import.

B4d4c will own the existing blocks in one concrete resident Model/Session. Its
approved design uses the KV Set's command/status for complete candidate history
stamping/validation, then KV copy and infallible history/logit/token publication.
Uncertain copies poison every family. No native graph qualification follows
until actual source/state/allocator/recovery checks pass.
