# Same-checkpoint stock MLX decision witness

4 November 2026. **Four selected decisions agree with Toklet.** This is a
bounded diagnostic, not whole-graph equivalence or task-quality qualification.
The [original-relative Q4 gate](gemma-full-tasks.md) remains rejected52→43/100,
with a9-point loss against the unchanged2-point limit.

Before execution, freeze the first original-only and native-only win in each
affected category, using admitted100-case paired journals and resolved grading.
For each case, choose the first sampled-ID disagreement (including hidden stop
IDs); stock MLX receives the original prompt followed by agreed single tokens,
with a fresh cache, stopping before either disputed ID is fed.

| Case | Input tokens | Original ID | Native / stock ID | Stock top-two margin |
| --- | ---: | ---: | ---: | ---: |
| HumanEval/1 | 152 | 236751 | 19243 | 5.6875 |
| HumanEval/24 | 86 | 236770 | 236778 | 0.25 |
| truthfulqa/row-0 | 67 | 236776 | 236780 | 10.625 |
| truthfulqa/row-3 | 94 | 236796 | 236799 | 0.25 |

Stock `mlx-lm0.32.0` / `mlx0.32.3` strictly loads the exact Q4 checkpoint
`238767527555cb75a05732a84dff5d6ba0dd6809`, without Toklet arithmetic. The loader
consumes1096 exact-shape/dtype tensors:278 affine4/group64 matrices with scales
and biases, plus262 BF16 vectors. Original loaded-array identities, tied head,
15 fresh KV owners and all cache offsets are checked. Selected persistent
parameter storage is2,603,981,382bytes. Rows are BF16, saved by exact FP32
widening; independent full262144-value replay verifies hashes, finite values,
lowest-ID argmax, candidate logits and margins.

Cold runtime/input hashes are checked before numerical imports and after work;
actual imported modules and non-system native images must be accounted for.
The first run's final audit failed on Torch's generated temporary helper;
its four observations remain unqualified. The repeated run checks that exact
helper against its pinned template/instantiator and saves its bytes. The first
failure, source snapshot and receipts remain intact. A second launch attempt
stopped before creating a child because exclusive logs already existed; fresh
versioned logs correct that ownership error.

The owned repeated worker exits0, its process group is absent, and external
`time -l` corroborates peak RSS2,530,099,200bytes. Observed lifetime physical
peak3,239,480,320bytes and MLX allocation peak2,989,794,532bytes are below6GiB.
MLX's memory limit is a guideline; these are measured peaks, not a hard-cap claim.
Four cold controls and independent saved-row/runtime replay pass. Astra approves code and evidence. No Rust inputs changed; existing550 default/658 Metal checks apply.

These decisions also occur outside Toklet. They do not establish that
quantization alone caused the complete task loss, or close historical numerical
failures, Mac app admission or iPhone qualification. Next assess an immutable8-bit artifact’s lineage, text/tokenizer identity and
actual header-based combined-owner ledger. Only if it fits6GiB, run a stock
load/short generation on a predeclared non-held-out prompt. Defer native8-bit
kernels and the full conformance graph until that feasibility decision.

[Machine evidence](../benchmarks/results/2026-11-04-gemma-mlx-decisions-v1.json).
[Full local artifacts](../artifacts/gemma-mlx-decisions-2026-11-04).
