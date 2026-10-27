# Prospective Qwen3 4096 diagnostic inputs

These inputs were frozen before executing the new model experiment. They grant
no app-capacity, accuracy or performance qualification. See the
[native profile](../../docs/qwen3-native-profile.md) and
[full-window plan](../../docs/qwen3-full-window-plan.md).

Each `.u32le` file contains exactly 4096 raw token IDs without added special tokens.
`A` repeats the existing 512-position structural fixture eight times. `B` differs
only at its last token, incremented modulo 151936, for a causal control. Repetition
is a structural witness and does not supply the quality sample.

`Q` uses the pinned WikiText test corpus, nonempty original rows joined with
newlines, and the pinned Qwen3 tokenizer. It selects the complete token stream's
half-open span `[10001,14097)`, immediately after the previous quality segment.
No truncation, padding or repeated text fills this span. The manifest records
the source revision, file identities, corpus identity and reconstruction recipe;
the parent [quality inputs](../qwen3-quality-v1/manifest.json) retain their scope.

Teacher-force all 4096 inputs. Score exactly 1024 next-token targets `Q[3072:4096]`
from output rows `[3071,4095)`; the final input commits without another target.
The native mean NLL increase over the original BF16 source must be at most 0.05,
inclusive. This supplementary loss gate does not replace the complete prior
quality comparison or establish instruction-following usefulness.

[Preparation](../../tools/bench/prepare_qwen3_context4096.py) and its independent
batch-read/file-tokenizer replay reconstruct identical complete bytes and verify
the old 10001-token prefix before selecting the new span. Source license and
attribution remain in the original dataset and [source inventory](../sources.json).
