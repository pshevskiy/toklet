# Qwen3.5-2B real-source diagnostic

The selected uncensored source now has authenticated parent comparison, bounded
Q4 conversion, an explicit hybrid token-ID CLI and a passing initial short graph
diagnostic. This remains a diagnostic route. Deeper context, codec, quality,
representative fit, performance, observer overhead and adoption are unqualified.
Astra approves scoped diagnostic retention; context17 fails numerical qualification.
Ordinary inspection and automatic eligibility
remain unchanged. The earlier [typed-source scope](qwen35-2b-typed-source.md) is
still separate from real-model qualification.

| Artifact | Exact identity |
| --- | --- |
| Selected BF16 | `TheCluster/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-MLX-bf16`, revision `78efacde810bbe7d1b80410fc5ac60db4f24fd8a`; `model.safetensors`, 4,426,559,145 B; SHA256 `f04765ef799190d41131132d43d69624e9ed6eac7b7cdd2a65a2eb3e266878ab` |
| Declared parent | `HauhauCS/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive`, revision `cafbddcb9940f38618e2b2ba84149584949379f9`; `Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-BF16.gguf`, 3,775,708,640 B; SHA256 `d98cff5f907e2186ac8243ac5676a47dc24e1822cd6ed674c9c92c7503915a1b` |
| Derived child | `models/qwen35-2b-thecluster-affine4-78efacde/model.safetensors`, 1,059,405,152 B; SHA256 `a09743503b034d79d24609cf07804c955e29bc19a37267f0b25778702dae0682` |

The complete parent and selected source bytes authenticate. Frozen prospective
rules compare 187 matrices exactly, round 115 small owners to the selected BF16
bytes, and check the 288 original F32 `A_log` values through the parent's decay
rate versus negative exponential under both independent profiles. This does not
prove the publisher's historical converter or reconstruct `A_log` bitwise from
its transformed parent representation. No parent payload is retained.

Conversion produces 187 affine Q4 matrices with group 64/BF16 scales and biases,
97 BF16 vectors, 18 BF16 convolutions and 18 unchanged F32 `A_log[16]` vectors.
Thus 115 BF16 small owners and 18 F32 owners survive exactly; 297 vision owners
are excluded. The child has 320 logical / 694 physical tensors and a
1,059,315,904-byte body. All 1,881,276,416 matrix values reconstruct exactly
against upstream F32 during production. The saved-byte audit verifies actual
source, child and component bytes without rerunning numerical conversion.
[Parent comparison](../artifacts/qwen35-2b-payload-2026-11-13/parent-lineage-result-1.json),
[conversion](../artifacts/qwen35-2b-payload-2026-11-13/conversion-result-1.json),
[saved-byte audit](../artifacts/qwen35-2b-payload-2026-11-13/saved-byte-audit-1/manifest.json).

The public cold `two_b::artifact::pins` parser requires a caller-selected manifest
SHA256, at most 64 KiB, exact fields, known selected-source identities, storage
counts and diagnostic-only flags. It rejects duplicate keys and invalid bounds.
The external hash binds those selected bytes; this parser does not independently
authenticate the referenced lineage certificate or replay its proof. The loader
then authenticates the complete child/config/header and EOF before publishing a
store. Consuming GPU upload drops packed CPU matrix owners. The child config is
2,570 B, SHA256 `1a569230a00594346d86a00b0b172f398f11b9c13d8b637317f6370fee35d658`;
the header is 89,240 B, SHA256 `ea3c847a30459cfd565f0abd8c8474b3d3818c2fd612a4bd0f220acec4965469`.

GPU projections/gather use packed Q4/G64/BF16 metadata and F32 inputs/outputs.
CPU F32 recurrent, convolution-history and attention-KV state stay distinct.
Small BF16 parameters widen exactly; original F32 `A_log` never narrows through
BF16. CPU normalization uses FP64 reductions/scaling and F32 stores. Weights are
immutable; session state, scratch and pending sampled history have separate
owners. The 6 GiB engine, 512 MiB setup/audit, 32 MiB helper, 3 GiB compiler and
2 GiB disk-floor policies remain unchanged. Observed command peaks do not qualify
all representative model/session/application workloads.

An exact local raw-ID invocation is:

```sh
ROOT=/Users/coloner/work/rust/toklet
MODEL="$ROOT/models/qwen35-2b-thecluster-affine4-78efacde"
"$ROOT/artifacts/qwen35-2b-payload-2026-11-13/native-bin-1/toklet" \
  --model "$MODEL" --model-profile qwen35-2b-diagnostic \
  --backend metal --metal-projection mlx-affine \
  --artifact-manifest "$MODEL/child-manifest.json" \
  --artifact-manifest-sha256 87133427e3b185d365f235a560f3212038ca4f6851de77cb4338998ddbdb2856 \
  --tokens 1,7,13 --max-new-tokens 2 \
  --context-capacity 128 --prefill-chunk 32 --json
```

Token files and persistent JSON-lines requests are also supported. Text, chat,
tools, other backends and unsupported modes reject. Default capacity/chunk is
128/32; maximum is 512/32. The raw-ID route uses selected nested-config EOS 248044
plus explicit user stops; `--no-eos` disables configured EOS. Parent GGUF EOS 248046
and the 4B codec's stop union are not inherited. Natural EOS termination and an
independent user-stop golden remain separate, unwitnessed gates.

Statistics expose completed generation phase clocks, load/authentication,
context initialization, upload, binding and session initialization. Compilation
time is explicitly unavailable: context initialization includes pipeline creation.
Offline conversion is not charged as runtime conversion. Detailed
`GenerationMemory` separates recurrent/history/KV owners; `GenerationState`
separates committed and pending tokens. `last_numerical_work` describes only the
last numerical call, not cumulative request work. Repair status is explicit:
cancel/error preserves a valid prefix, while reset does not repair poisoned GPU
execution. The API supports admitted explicit repair; this CLI requires restart
when poisoned and performs no implicit repair/retry.

All four explicit release GPU controls pass, including actual-source transaction,
late failure, cancellation, repair-overlap and cache-state checks. All 12 console
process cases pass: nine rejections plus CSV, token-file and one persistent
interactive cohort. Their owner arithmetic, phase/state/work fields, rejection
preservation and replay controls pass; native replay is not an independent golden.
[GPU/transaction controls](../artifacts/qwen35-2b-payload-2026-11-13/explicit-gpu-controls-1.json),
[console controls](../artifacts/qwen35-2b-payload-2026-11-13/cli-controls-1/result.json).

The initial short graph checks 159,187,008 coordinates against both independent
profiles at unchanged `atol=1e-5`, `rtol=1e-4`: zero failures. All 12 schedule/reset
controls and 6,021 causal-prefix arrays pass. This covers four short cases and
does not qualify deeper state trajectories. Both streamed context17 profiles
complete 680,841,024 coordinate comparisons with identical native traces:
ideal FP64 rejects three coordinates; staged FP32 rejects none. Each profile's
nine schedule/reset controls and 10,704 causal-prefix arrays pass. Both native
and comparison journals audit, and all retained failing arrays replay. Failures
are at token 12/layer 15/attention-context flat coordinate 1,591. The unchanged
paired numerical gate rejects this route; conditional fixed255 is not launched.
fixed255 teacher forcing is distinct from a 128-input/128-generated-output request.
Check reuse is bound to unchanged Rust, fixtures and compiled protocol; the
reader fix has refreshed runtime identities and 70 portable controls per Python
mode. [Short comparison](../artifacts/qwen35-2b-payload-2026-11-13/initial-short-check-1.json),
[check reuse](../artifacts/qwen35-2b-payload-2026-11-13/check-reuse-1.json).
[Paired deeper decision](../artifacts/qwen35-2b-payload-2026-11-13/context17-paired-decision-1.json),
[conditional gates](../artifacts/qwen35-2b-payload-2026-11-13/context17-conditional-gates-1.json),
[Astra approval](../artifacts/qwen35-2b-payload-2026-11-13/astra-final-review-1.json).

Failed attempts remain retained: source-download monitoring failed while closing
logs under Python 3.9; a fresh offline audit recovered source evidence without
claiming that monitor passed. The reference reader fix required fresh controls.
A reference run exceeded 512 MiB; the explicit child allocator profile then passed
without changing arithmetic or gates. Combining cold-import suites in one
interpreter also failed preflight; separate interpreter groups now pass.
[Monitor failure/recovery](../artifacts/qwen35-2b-payload-2026-11-13/source-acquisition-guard-failure-1.json),
[allocator rejection](../artifacts/qwen35-2b-payload-2026-11-13/initial-short-reference-command-2/result.json),
[allocator retry](../artifacts/qwen35-2b-payload-2026-11-13/initial-short-reference-command-3/result.json) ([profile](../artifacts/qwen35-2b-payload-2026-11-13/initial-short-reference-request-3.json)),
[CI-preflight rejection](../artifacts/qwen35-2b-payload-2026-11-13/combined-portable-opt0-1/result.json),
[current normal](../artifacts/qwen35-2b-payload-2026-11-13/combined-portable-opt0-3/result.json)/[optimized](../artifacts/qwen35-2b-payload-2026-11-13/combined-portable-opt1-3/result.json) controls.

These are correctness and diagnostic results. Full codec/template semantics,
original 10,000-target/100-task quality, representative fit, controlled request
performance, observer overhead and managed adoption remain open. No performance
improvement or model-adoption claim follows.
