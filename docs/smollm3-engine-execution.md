# SmolLM3-3B MLX4: bounded Mac engine execution

5 November 2026. The pinned SmolLM3 checkpoint passes **Mac512 engine execution**
using the shared dense blocks and reviewed [per-layer rotary selection](smollm3-rotary-schedule.md).
This adds an executed architecture; it does not qualify managed Chat, held-out
quality, full numerical fidelity, the declared65536 window or iPhone.

Artifact: `mlx-community/SmolLM3-3B-4bit`, revision
`d3a7e0594d6642dbcfb7d149bed8b0bdf49f95ce`,1,730,051,993bytes, SHA256
`d501e6193d7eb84e10d3c377007d1fc4b4b2fa08799dd10fa4f6a75033334a70`.
Complete header/index spans cover832tensors:253U32 and579BF16. Affine4/group64
retains the original metadata; one tied table is counted once. All36 layers use
full causal attention; indices3/7/…/35 bypass rotary, exactly as the stock
adapter's actual instantiated modules verify.

The frozen37-token prompt uses the authenticated local template with thinking
disabled and fixed system text. No BOS is prepended. Up to8greedy slots stop at
128012/128001, with lowest-ID tie breaking. Stock MLX and native execution choose
`791,4320,374,220,19,13,128012`: visible “The answer is 4.” plus hidden stop.
Compare all seven complete128256-value rows at identical prefixes before feeding
the next ID. Source margins include0.25 for both first decisions; native margins
are0.2064724/0.3333969 and choose the same IDs.

The largest absolute row difference is0.8785772; these differences are
**descriptive**, not a passed fidelity envelope. Stock uses the stored BF16
profile; native uses Half activations/KV, Compensated normalization, Reference
attention and FullLogits. Original-weight quality is not evaluated here.

Native chunk1 and32 produce bit-identical prompt/final-generation and complete
512-position boundary distributions. A frozen synthetic repeated-text prefix
uses only the pinned tokenizer's valid IDs. Both schedules reach512; token513
rejects without changing history, committed positions or published logits.
Reset reproduces the prompt. Sessions replace one another; weights stay shared
once. The new shared probe also reruns Qwen1.5 through512 and reproduces both
previous native decision rows bit-for-bit.

| Memory | Bytes |
| --- | ---: |
| Resident native weights |1,729,958,910 |
| Native chunk32 admitted model/session |1,858,324,290 |
| Native process-lifetime RSS peak |1,820,540,928 |
| Greatest observed native physical high-water |1,810,810,728 |
| Stock process-lifetime RSS peak |2,047,393,792 |
| Greatest observed stock physical high-water |2,174,044,560 |

Native numerical admission is4GiB; the separate process trial policy is6GiB.
RSS peaks are Darwin `wait4`; physical values are the greatest external lifetime
high-water readings obtained before exit. All workers exit0 with absent groups.
Stock's source/model pin endpoints and independent full-row replay pass.

Formatting, strict all-target Clippy,554 default/662 Metal tests, release library
and both probe builds pass. The probe's common helper serves the two actual
models; it changes no numerical kernels. The preserved initial metadata fetch
rejected a tokenizer larger than16MiB; a bounded64MiB transfer authenticates the
actual file. No numerical gate changed. Astra approves the code and bounded execution evidence.

[Machine evidence](../benchmarks/results/2026-11-05-smollm3-engine-execution-v1.json).
Next candidate: Granite3.3-2B MLX8, pending header/memory and scaling-block work.
