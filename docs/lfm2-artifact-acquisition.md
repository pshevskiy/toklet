# Pinned LFM2 artifact acquisition

B4e1 verifies the complete original public
[LFM2-350M four-bit export](https://huggingface.co/mlx-community/LFM2-350M-4bit/tree/18dc72abf3b2337f9123cfd6eeeb58dfa7947066),
revision `18dc72abf3b2337f9123cfd6eeeb58dfa7947066`. The local bundle is
`models/LFM2-350M-4bit-18dc72a`; model data remains outside Git. This acquisition
can precede native composition because it executes no model or tokenizer.

Nine files total204,378,119 bytes. Only the199,527,820-byte weight file needed
network transfer; eight metadata/tokenizer/template files were already cached
and matched the immutable API's Git blob identities. Complete streamed weight
SHA256 is `558353ad098c1230a8e2ea508cd34ef311a1b97a2bc4f47498f4df8229dc5c0f`,
matching the advertised LFS identity.

The36,740-byte original header starts after the length word; payload begins at
36,748. All334 original descriptors have checked dtype/shape byte products and
contiguous nonoverlapping extents ending at199,491,072 payload bytes. Inventory
is93 affine triples,45 direct BF16 norms and ten `[1024,3,1]` coefficients.
Thirty B/C/x row views remain borrowed semantic ranges, not physical tensors.

The reviewed controller uses one owned HTTPS curl job with exact file-size/hash
checks, no retries,300-second outer deadline and30-second total cleanup bound.
A200MiB aggregate body/log cap and256MiB free-disk preflight bound transport.
No credentials or user curl configuration are used. Only after all nine files
and header/index verify does exclusive atomic directory publication expose the
bundle. Partial and published model paths share one inode, retaining evidence
without another model-sized physical copy. HTTP200 and clean closed group
PID14575 are independently verified.

Astra reviews the concrete controller before launch, then independently checks
all nine published identities, full model digest, header/index extents, original
cached Git blobs, actual runtime/source pins and endpoint cleanup. The inherited
cleanup accounting was corrected to15s TERM+15s KILL+immediate reap before the
controller was frozen. No failed network acquisition occurred.

[Machine evidence](../benchmarks/results/2026-10-29-lfm2-artifact-acquisition-v1.json)
records every file digest and the original bounded transport/review receipts.
This artifact-only task requires content/link/whitespace checks; it changes no
production Rust and does not repeat the passing scalar graph regressions.

Transport verifies original bytes, not normalization fidelity, real inference,
process memory fit, performance, context or app admission. B4a separately chose
the current pinned loader/direct-gain semantic profile; the historical converter
identity remains unknown. Declared128000 positions remain distinct from
implemented/qualified capacity. Tokenizer maps only64,400 IDs of65,536 head
rows, and later encoding must avoid duplicate BOS1 and use the specified stop7.
No tokenizer/template is executed here; unmapped-token handling, Chat and
original adoption quality remain in the [LFM2 plan](lfm2-hybrid-plan.md).
