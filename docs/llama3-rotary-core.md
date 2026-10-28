# Llama3 scaled rotary core and selected artifact

M8 B1a, 28 October 2026. The scalar positional block and independent fixture
pass checks and Astra review. Model-loader propagation, Metal execution and
real-model qualification remain open. See [TODO.md](../TODO.md) and the
[source-bound evidence](../benchmarks/results/2026-10-28-llama3-rotary-core-v1.json).

## Implemented block

`cpu::Llama3Scaling` validates the model's factor, low/high frequency factors
and original context. It transforms inverse frequencies using the canonical
[Transformers Llama3 rule](https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/modeling_rope_utils.py),
whose downloaded bytes are hash-pinned in the evidence. High frequencies remain
unchanged, low frequencies divide by the factor, and the intervening band blends
continuously. The exact low-frequency wavelength boundary divides by the factor.
Factor 1 preserves ordinary frequencies exactly.

`cpu::llama3_rotary` applies this transform to borrowed contiguous
`[tokens, heads, head_dim]` slices with full split-half pairing and absolute
positions. Scaling applies even to short prompts. Frequencies, angles and trig
use FP64, then sine/cosine and pair arithmetic use FP32 as in the existing
scalar rotary reference. Shapes, exact lengths, overflow, positions and finite
inputs are validated before writes. The function allocates no buffers; discard
partial output after an arithmetic error. Existing ordinary rotary code is
unchanged. No new loader, model or public app capability is enabled by this task.

## Independent checks

The stdlib-only generator constructs Decimal80 frequencies using frequency-space
cutoffs and effective wavelength stretching, independently of Rust's wavelength
branch expression. It then uses FP64 trig/output. The frozen fixture contains
seven cases, 148 frequencies, 1224 output values and five region/edge controls.
Positions include zero, the original context boundary, the model-declared end
and a separate higher-position mathematical probe. These are primitive checks,
not context qualification. A fresh generator export is byte-identical.

Six targeted tests cover exact transitions and neighbors, all candidate frequency
bands, factor-1/high-frequency ordinary bit equality, absolute chunk equivalence,
short-input scaling, rejected input preservation and explicit arithmetic failure.
The existing FP32 gate `1e-5 + 1e-4 * abs(expected)` is unchanged. Formatting,
strict default/Metal all-target Clippy, 381 default tests and 260 Metal-library
tests pass; counts overlap. No model or GPU run was needed for this scalar-only
change. Astra independently authenticated source/output hashes and all six
recorded process groups closed with no findings.

Astra found a separate edge issue in the pinned
[MLX-LM implementation](https://github.com/ml-explore/mlx-lm/blob/f39c4aa2af5d61a5e85bd50cc3f4294fedaff116/mlx_lm/models/rope_utils.py):
its strict middle-band predicate leaves the exact low boundary unscaled.
The mathematical fixture uses the canonical continuous rule. Source-profile
comparison remains separate; algebraic equivalence does not imply identical
FP32 reciprocals, trigonometry or whole-model outputs.

## Pinned candidate for the next tasks

Astra selected the public, non-gated
[Llama-3.2-1B-Instruct-8bit](https://huggingface.co/mlx-community/Llama-3.2-1B-Instruct-8bit/tree/d48cdf0a4ea22d893b7c63a99d6a693e24822795),
revision `d48cdf0a4ea22d893b7c63a99d6a693e24822795`.
Only metadata, tokenizer assets and the safetensors header were downloaded.
All 13 recorded pins were verified; config/tokenizer/index Git blobs and tensor
header extents agree. Full weights and their advertised hash remain unverified.

The graph is 16 layers, hidden 2048, MLP 8192, 32 query/8 KV heads of width 64,
tied output, theta 500000 and Llama3 factor 32/low 1/high 4/original 8192.
The header describes 113 affine8/group64 matrices with U32 packed payload and
F16 coefficients, plus 33 F16 norm vectors: 1,313,116,160 payload bytes.
Its 32 mathematical frequencies occupy 15 high, three blended and 14 low slots;
none hits either exact transition. Its declared context is 131072; the first
planned model vertical is bounded to 512 and is not yet admitted.

Next propagate an explicit positional plan from source into immutable model
state, preserving existing ordinary constructors. Then implement the scaled
Metal operation and checked structural admission, including composite `uint`
index bounds, before downloading/loading the real checkpoint. Keep Llama on
Reference attention. A pinned bounded plain-text template/BOS/stop contract is
also required: the tokenizer's remote template has date/tool branches and is
not automatically executable Chat support. Numerical, session, whole-process
memory and unchanged quality gates remain prerequisites for a support claim.
