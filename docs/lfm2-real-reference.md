# Bounded real LFM2 numerical reference

B4e2c implements the independent CPU reference for the original
`mlx-community/LFM2-350M-4bit` revision
`18dc72abf3b2337f9123cfd6eeeb58dfa7947066`. The code, synthetic controls and
finite runner have Astra approval. All thirteen new synthetic controls, six
repository checks, exact four-file real export/replay and the independent
saved-input audit pass. See the [result record](../benchmarks/results/2026-10-29-lfm2-real-reference-v1.json)
and [frozen protocol/gate metadata](../tests/fixtures/lfm2-real-source-v1/README.md).
This closes the bounded source fixture, not native model or app qualification.

This reference follows the selected native arithmetic profile. It does not
claim the unknown historical converter runtime or official MLX intermediate-
cast trajectory. The [exact-affine helper](lfm2-exact-affine-reference.md) and
[pinned text fixture](lfm2-text-reference.md) are separately qualified
prerequisites. No Rust/Metal, MLX, tokenizer or previous whole-graph source is
imported by the numerical producer.

The authenticated original container is 199,527,820 bytes, including its
36,748-byte header/prefix and 199,491,072-byte payload. The reader verifies the
whole-file identity, exact config/index/header, 334 contiguous physical tensor
extents and 148 logical owners: 93 affine parents, 45 direct BF16 norms and ten
three-tap convolution tensors. One read-only full-file mapping is counted once.
Tied input/output weights share the original parent; only small coefficients
and owned decoded row stripes survive an operation.

All 16 layers execute: ten short-convolution layers and six full-attention
layers. The graph preserves direct gains, separate Half stores, raw-u history,
ordinary residuals, Q/K-only head norms, ordinary split-half rotary positions,
GQA16/8 with head64 and serial Reference attention. It introduces no embedding
norm, V norm, branch postnorm, Gemma clipping or fused activation stores.
Every affine uses the exact original-FP32/stored-Half dot and direct destination
rounding. MAIN and WIDE operate on their own inputs with the same affine center.
Binary64 exactness assumes ordinary full-binary64 backend operations; system
library identity observations are separate from content-hashed runtime files.
NumPy 2.3.2 reports Apple Accelerate with an unknown library version.
WIDE rotary retains the declared Decimal80/120 same-method stability limitation,
rather than claiming certified transcendental rounding.

The fixed source cases have 1, 8, 11 and 64 raw tokens. Ten selected rows retain
all 293 semantic stages; four fixed teacher histories add full head outputs.
The source computes tied-head distributions for selected rows only, at most
four per forward. The complete producer/state sidebands preserve all raw u,
before/after two-row histories, and head-major Krot/V rows.

| Capture per profile | Groups |
| --- | ---: |
| Selected mathematical stages | 2,930 |
| Fixed teacher distributions | 4 |
| Full producer/state sidebands | 168 |
| Total | 3,102 |

Both profiles require 6,204 descriptors and exactly 48,049,152 binary bytes.
Every descriptor has explicit dtype, shape, byte extent and hash. Four files
must replay exactly: arrays, journal, components and manifest. Saved-byte
invariants check causal prefixes, masked positive zeros, raw-u history equations,
KV/producer prefixes and the shared teacher history without another forward.

MAIN comparison coefficients are fixed before native outcomes: per-group
`atol=max(1e-4, 2*MAIN/WIDE envelope)`, `rtol=1e-3`. Scores, context, final logits and
teacher outputs retain strict coupled gates; other native stages require
universal actual-input semantics plus their frozen local guard. All coupled
intermediate drift must still be reported. Sideband envelopes are diagnostic;
state publication remains an exact native-operand requirement. The independent
source fixture does not itself implement or qualify the future native checker.

The numerical reserve is 128 MiB, including 32 MiB for affine stripes and
conversion scratch. The ledger includes Python scalar-leaf word lists,
activations, selected head outputs and serialization/comparison temporaries.
Runtime, metadata and OS costs are additional. Explicit bounds are not measured
process peaks. The owner permits two 900 s real jobs with a 6 GiB direct-worker
ceiling, 128 MiB array cap, 16 MiB metadata/log caps and bounded teardown; the
synthetic job has separate 600 s/512 MiB limits.

On Apple M1 Pro/macOS26.6.2, export/replay OS maxRSS is
464,846,848/469,549,056 bytes, with 6,372/6,555 direct samples. Resident and
physical-footprint observations remain separately labelled; they are not added
together or presented as native/app fit. Both profiles use the certified
binary64 path for all 28,151,808 affine outputs across 18,048 stripe calls.
The largest decoded FP32 stripe is 2,359,296 bytes. No int64/block/scalar fallback
is observed in this workload; the separate helper controls qualify those paths.

The independent standard-library reader imports neither producer nor leaf
helpers. It checks all 6,204 descriptors, 22,072,064 finite values, 3,102 gates,
raw histories, masks, causal prefixes and teacher-prefix equality. It reconstructs
all selected local nonprojection coordinates and 3,720 first/last projection
coordinates across 1,860 captured invocations, using exact integers and ordered-
word binary-search rounding. The reader completes in 17.37 s with 59,981,824-byte
OS maxRSS. Astra's saved attestation reauthenticates 6,912 current identities and
all ten closed process groups; no second full forward is performed.

Remaining projection coordinates rely on the qualified exact helper plus
complete authenticated finite/gate/replay checks. Nonselected u/Krot/V producer
rows have complete state/finite/causal coverage, without independently rebuilding
their producer formulas. Teacher1–3 have full output descriptor/gate/replay
coverage but no retained internal trajectory; teacher0 additionally matches the
identical captured prefix. The source and audit do not claim an independent
correctly rounded transcendental oracle. The two reader-owner versions preserve
identical audit code; the executed successor adds a fresh bytecode-cache prefix.

Formatting, strict default/Metal all-target Clippy, default 450/Metal 545 tests
and release Metal test compilation pass. Counts overlap, and 20/181 checks remain
explicitly ignored. Rust/Metal numerical code, earlier fixtures, all source/native
gates and original adoption-quality requirements are unchanged.

The 64-token numerical trajectory, future 512-capacity constructor/admission
and model-declared 128,000 positions are distinct. Native loader/session/text/
memory testing remains B4e3. App/Chat, user context controls, original unquantized
adoption quality, performance and separate iPhone qualification remain open.
