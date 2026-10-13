# Independent larger-model numerical qualification

The preselected Qwen2.5-1.5B checkpoint passes the independent short CPU oracle.
Across four frozen cases, all 228 complete selected arrays at layers 0/1/27,
final normalization and every-position logits pass the unchanged pooled
FP32/FP64 envelope. Maximum tolerance fraction is 0.520006. Every greedy ID
agrees, and maximum short teacher-forced NLL difference is 0.000055770 nats,
below the unchanged 0.001 guard. Scalar/tiled execution agrees in all 2028
complete trace arrays, including signed-zero bits. Cached all-row output agrees
numerically with full execution for 36 distributions at chunks 1/3/8, tails
and absolute offset five.

The exporter keeps exact source BF16 weights and uses pinned Transformers 5.7.0
CPU eager operators with FP32 activations. For each embedding, decoder layer,
final norm and tied vocabulary operation, pre-hooks install temporary FP32 data;
always-call hooks and an outer finally restore the original BF16 tensor objects.
Parameter identity and tied storage are checked. Original FP32 rotary frequencies
are required. Inference mode prevents retained output graphs. The independent
functional FP64 implementation runs only after all FP32 scopes are restored.
This changes reference memory ownership, not the numerical gate formula.

Before larger export, the streamed small model reproduced all 228 original
complete FP32 arrays and every independent envelope byte-for-byte. The
[small parity receipt](../../benchmarks/results/2026-10-13-streamed-small-oracle-parity-v1.json)
binds both manifests. Four explicit PyTorch tests cover tied identity, restoration
after operator/native failure, nested scopes and invalid dtype/admission.

Larger reference admission counts 3,087,428,608 resident BF16 bytes, one largest
1,866,989,568-byte FP64 temporary and a fixed 768 MiB application/framework
allowance: 5,759,724,544 bytes. Only one temporary FP32 module scope is active,
and it finishes before FP64 work. Observed exporter native lifetime charge is
2,149,044,976 bytes; mapped clean/shared pages can be omitted, so this does not
replace numerical allocation admission. The Rust short test retains one FP32
weight model, bounded diagnostic traces, scratch and caller buffers. Own-process
native peak is 6,391,338,192 bytes, below 6,442,450,944. This is short numerical
qualification, not full-workload physical memory or performance qualification.

The full/probe manifests were created before the first Rust comparison. The
[task binding](../../benchmarks/results/2026-10-13-larger-oracle-qualification-v1.json)
retains their hashes, source/config identities, recorded creation chronology,
every array hash, native reports, checks and protected executable identities.
Formatting, strict Clippy in both configurations, 86/92 ordinary Rust tests,
20 optional reference Python tests and old/new fixture integrity checks pass.
The [CPU parity report](../../benchmarks/results/2026-10-13-larger-cpu-parity-v1.json)
contains complete per-array metrics.

Astra accepted code, checks, export and native evidence. This qualifies the independent larger CPU reference geometry and the streamed
exporter mechanism. Full 10000-target reference loss, matched GPU loss/task
quality, GPU geometry, performance and sustained evidence remain pending. No
larger quality results were used to select the checkpoint or tune Q4 v1.
