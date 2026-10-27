# Experimental Qwen3 C512 input freeze

This is a separately named precision-validation proposal, not an adopted gate
or a new engine capability. Historical C512 captures remain rejected under their
original contract. No kernel, execution precision, position limit or app setting
changes. See [the first-divergence diagnosis](../../docs/qwen3-first-divergence.md)
and [active context plan](../../docs/long-reply-engine-plan.md).

[recipe.json](recipe.json) freezes two calibration source texts and three
validation source texts. The two calibration texts produce six nested-prefix
cases at32/128/512 tokens; the three validation texts produce eight cases at
127/128/129,255/256/257 and511/512. These are fourteen cases, not fourteen
independent texts. Each authored text plus two newlines is explicitly repeated
16 times, tokenized without added special tokens using the pinned original
Qwen3 tokenizer, and the declared token prefix is selected. This is a numerical
witness recipe, not a held-out quality workload. No implicit padding occurs.

Preparation rejects duplicate arrays, source-content aliases between splits,
cross-split token-prefix reuse, out-of-vocabulary/noninteger IDs, insufficient
text and replacement of an existing capture. Generated artifacts retain complete
token arrays, identities and a manifest, including partial success on failures.
The fixed source/profile/policy fields are validated independently of the recipe
hash. Two cold preparations must have identical manifest and array bytes.

The profile is the same original eight-bit affine artifact, FP32 reconstructed
weights, Half activations/KV and FP32 logits. Independent CPU FP32/FP64 graphs
remain separate arithmetic witnesses. Native MlxAffine/Compensated/Reference/
FullLogits stays unchanged. The manifest lists57 captured-state pooling keys,
each exact layer/stage name pooling only over calibration cases. Six explicit
same-source calibration prefix pairs compare token axis0; attention score and
probability matrices are outside that captured-state map. Unobserved layers
gain no qualification.

The proposal retains the existing formula, using only independent calibration:

`atol = max(1e-4, 2 * FP32_FP64_max_abs, 2 * equivalent_prefix_max_abs)`

`rtol = 1e-3`

No numeric gate is derived in this task. An absolute envelope ceiling is pending
independent review; no universal ceiling or multiplier may be invented from the
already observed native failures. Freeze and review the calibration gate file
before inspecting new validation results. Native errors never set its limits.
Any validation failure rejects that version, without retuning on the validation
split. Both splits exclude the historical C512 fixtures and all quality data.

Native scalars and argmax IDs compare against canonical CPU FP32. Near ties
retain full rows/top-two IDs/margins and receive no automatic mismatch exception.
CPU-FP32 versus FP64 values have a separately stated calibration-coverage audit;
CPU argmax disagreements are reported independently. This adds no second native
target oracle. Exact causal, masking, cache/state and rejected-capacity checks
remain mandatory.

Same-quantized-source parity requires a separately pinned source execution
profile. It cannot replace an unquantized quality baseline. Product adoption
still requires checkpoint-specific retokenization and the original complete
10,000-target loss/100-task demand, with unchanged0.05-nat/2-percentage-point
limits and at least1025 positions. Clipping that workload to512 is forbidden.
Those source, capacity and quality dependencies remain open. Full model-window
and independent phone qualification also remain open.

## Preparation result, 27 October 2026

Both fresh offline preparations have identical manifest/array bytes. All14
arrays,1344 calibration tokens and2175 validation tokens are committed beside
the [manifest](manifest.json). Six portable tests pass, including coherent
policy/profile mutations and partial-write retention. A separate standard-library
publication replay checks every array, split, declared length and prefix pair.
All three preparation groups and the publication group close.

On the M1 Pro/16GiB Mac, external physical/RSS peaks are
264,963,392/459,177,984 bytes and268,600,712/469,549,056 bytes, within512MiB.
These are tokenizer/preparation measurements, without inference, source parity,
quality or phone qualification. See [public evidence](../results/2026-10-27-qwen3-context-inputs-v2.json).
Next execute CPU calibration only, derive/freeze/review its gates, then open the
untouched validation split. No new native target evaluation precedes that freeze.
