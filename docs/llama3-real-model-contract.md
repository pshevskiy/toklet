# Bounded real Llama3 qualification contract

Next after [B1c2b](llama3-metal-composition.md): one explicit Mac vertical for
`mlx-community/Llama-3.2-1B-Instruct-8bit` at
`d48cdf0a4ea22d893b7c63a99d6a693e24822795`. The artifact's declared 131,072
positions remain separate from the initial 512-position implemented scope.
This contract does not add the model to app discovery or Chat admission.

## Artifact and arithmetic

The full 1,313,157,436-byte safetensors payload has now been downloaded and
verified as SHA256 `cd3582a78766654e3e2107e3b4d51d9cb18e6754b659bfb44a34328ef9d7fa52`.
All five metadata/tokenizer assets match their pinned SHA256 and Git blob IDs;
the exact header, 372 tensor names and contiguous spans match the prior
[metadata inventory](llama3-rotary-core.md). The first transfer reached its
600-second deadline with 20,136,981 bytes remaining. A separate exact-range
206 response completed the file; prefix, response and final hashes are retained.
This verifies bytes, not execution or quality.

Native target: MlxAffine projection, Compensated normalization, **Reference**
attention, serial execution, Half activation/KV storage, requested chunks 32/128.
A three-row diagnostic partition additionally exercises real cached prefill;
the offset-509 case explicitly caps each requested chunk at remaining capacity3.
FullLogits and GreedyToken must select identical lowest-ID maxima on the same
native prefix. No ordinary-kernel or unrelated-family arithmetic changes.

Independent profiles reuse the bounded NumPy/BLAS FP32 and FP64 primitives with
identical Half storage boundaries. They are not scalar-ordered reductions or
exact Metal emulation. Canonical scaled frequencies are FP64-to-FP32 in main
and independent Decimal80-to-FP64 in wide; packed reconstruction remains FP32,
including tied output weights, while embedding lookup rounds to Half.

Freeze six raw-ID probes, layers 0/1/15, four wide-selected teacher-forced
prefixes and three scoring prefixes before native outcomes. The 349 arrays in
each independent profile occupy 30,656,512 bytes. Stage tolerances remain
`atol=max(1e-4,2*pooled_main_wide_max_abs)`, `rtol=1e-3`; source equivalent-prefix
variation is diagnostic and cannot enlarge this gate. Source-profile greedy
agreement is not assumed. A separate four-step native raw-ID journey checks
actual self-selected history and FullLogits/GreedyToken consistency.

The explicit [capture runner](../examples/llama3_checkpoint.rs) streams bounded
raw arrays and flushed started/completed descriptors. A downstream checker must
validate reference policy/model pins, exact stage/row coverage and complete
source comparisons. It must stitch chunk rows and require exact numeric native
schedule and causal-prefix equality. `capture_completed` is not qualification.
The actual 512-token state check covers partial discard/resume, reset and
rejected token513; offset509 starts with only three local keys and carries no
512-key numerical-quality claim by itself.

## Text and quality boundaries

A separate metadata-only reference compares the pinned original Jinja to an
independent renderer: fixed date `26 Jul 2024`, plain alternating user/assistant
roles ending in user, no tools, exact original system/header text, renderer-owned
single BOS and tokenization with `add_special_tokens=false`. Stops are
`[128001,128008,128009]`, hidden from visible text. Decode is raw tokenizers
ByteLevel; Transformers cleanup parity is not assumed. Reserved literals and
malformed roles reject. Python/Jinja trim includes U+001C–U+001F, so a future
Rust renderer must implement that rule or explicitly reject and test those
controls. The text fixture alone does not enable application Chat.

After short numerical/state checks, a separate prospective consistency screen
uses four frozen 257-token blocks, reset per block, scoring 256 next-token
targets each (1,024 total). Compare same-artifact canonical FP64/Half mean NLL
with an increase at most 0.05, stable FP64 logsumexp and retained per-target
losses. Corpus spans and tokenizer IDs must be frozen before those outcomes.
This screen does not replace the original unquantized 10,000-target / 100-task
adoption quality gates. Full quality and app admission remain separate work.

The source admission ledger is 4,377,096,508 bytes: one FP32 tied table, the
whole source mapping, one current matrix, bounded widening/dequant/BLAS
workspace and explicit retained-trace/framework reserves. Actual source/native
processes must be observed against the 6 GiB Mac limit and bounded deadlines;
no CPU/GPU full-model coexistence. Phone tests remain deferred.

See [progress](../TODO.md) and [improvement record](improvements.md).

## Completed short Mac vertical

The fixed CPU export and saved-bit verification pass for all 349 arrays in each
profile; original Jinja/manual renderings, five text cases, BOS/stop checks and
text replay pass. The [source manifest](../tests/fixtures/mlx-llama3-real-v1/manifest.json)
and [text fixture](../tests/fixtures/llama3-plain-text-v1/fixture.json) are versioned;
large raw arrays remain bound in the local evidence. The source process observed
2,875,883,520 bytes peak RSS and 1,210,075,872 bytes lifetime physical peak.

The native Apple M1 Pro run passes **28,791,552 independent value comparisons**
across 1,282 source-compared capture slices, with zero failures. Maximum absolute
error is 0.125; maximum fraction of its frozen per-value gate is
0.5198856251624643. These are the unchanged source-only gates. The 1,288-slice
capture also contains six finite/state-only slices: four self-generated raw
steps and two final 512-token distributions. No independent full512 quality
claim follows from those state observations.

Exact numeric **logit** comparisons pass for requested chunks32/128/3, the
unchanged short/future causal prefix, 512-token schedules and the first native
versus teacher-forced distribution. All intermediate captured stages pass their
frozen numerical gates; they are not asserted bit-identical across schedules.
Both readbacks select the same native history. The four self-selected raw IDs
are `[662,662,883,1]`; these numerical probes are not a Chat or quality result.

| Mac observation | Bytes |
| --- | ---: |
| Resident learned buffers | 1,313,116,622 |
| Shared positional owner | 196 |
| Maximum diagnostic admission including 32 MiB application allowance | 1,782,219,068 |
| Native observed peak resident size | 1,557,823,488 |
| Native observed lifetime physical peak | 1,587,398,264 |

One diagnostic load took 4.105752292 seconds, including 3.789743542 seconds
conversion and 0.313554333 seconds compilation. This is an observation from the
checked workload, not a performance comparison or throughput benchmark.
Formatting, strict default/Metal Clippy, 384 default / 265 Metal-library tests
(overlap), three runner, seven reference, two renderer and three checker controls
pass. The checks authenticate the retained sources and frozen executable.

Retain this bounded diagnostic path. The separate 1,024-target loss screen,
production plain-text integration, full adoption quality and app/phone admission
remain open. [Compact evidence](../benchmarks/results/2026-10-28-llama3-real-short-vertical-v1.json)
binds all inputs, outputs, failed transfer/recovery and prospective reviews.
