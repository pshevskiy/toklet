# Gemma E2B 6-bit memory and quality preflight

5 November 2026. The immutable MLX Community checkpoint uses affine 6-bit
weights with group size 64. Its text configuration and tokenizer digest match
the existing Q4 checkpoint. The model card attributes the conversion to source
revision `70af34e20bd4b7a91f0de6b22675850c43922a03`.
[Checkpoint and model card](https://huggingface.co/mlx-community/gemma-4-e2b-it-6bit/tree/ebd7756d4e55627e11ae043af9cad8ed6465a2e2).

The two HTTP 206 ranges contain the eight-byte length and the 318,346-byte
safetensors header. Metadata Git blobs, index ownership, and the tensor spans
match the 4,708,626,264-byte container. The 1,096 selected text tensors occupy
3,761,052,230 bytes. The remaining 1,415 tensors are outside the selected text
layout. The weight payload remains to be downloaded and authenticated.

Replacing the text term in the [Q8 memory ledger](gemma8-feasibility.md) gives
5,388,066,446 bytes for the existing Mac512/chunk32 layout, with 1,054,384,498
bytes below the 6 GiB policy. This is a planning estimate. Q6-specific guards,
conversion, temporary allocations, and overlap with any existing model need
separate admission checks.

Astra reviewed the header and ledger. After authenticated acquisition, first
run the four [frozen Q4 decision probes](gemma-full-tasks.md) on stock MLX
using identical agreed prefixes. Verify strict load, selected module
geometry, complete decision rows, margins, and memory peaks. Compare every
choice against the original BF16 and stock Q4 decisions. A change is evidence
for investigating the Q6 candidate. The existing full quality gates still
apply. Native Q6 implementation follows only after a justified stock result.
[Evidence](../benchmarks/results/2026-11-05-gemma6-feasibility-v1.json).
