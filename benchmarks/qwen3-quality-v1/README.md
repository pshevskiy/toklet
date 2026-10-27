# Frozen complete Qwen3 quality inputs

[manifest.json](manifest.json), SHA-256
`908e037de5b399d9e9a8dd09aa9fcc568f67abb714b06508672deac6bb8ac529`,
freezes the checkpoint, numerical source bytes, tokenizer/rendering, baseline,
10000 loss targets,100 tasks and original acceptance policies before scoring.
[held_out_loss.u32le](held_out_loss.u32le) contains10001 IDs/40004 bytes,
SHA-256 `f53c5b0c446959164c4d2265166040123f95be5b6d9eec784895fb45a8215beb`.

Full raw task text, execution requests, corpus verification prefix and controls
remain in `artifacts/m8-qwen3-quality-inputs-v1`. Task text and generated code
are never executed during preparation. Respect original dataset licenses when
redistributing derived text. The public manifest contains identities/counts;
reconstruct full inputs from the exact pinned sources and preparation tools.

The independently pinned official baseline's [API metadata](baseline-api.json)
and [config](baseline-config.json) are retained. Its actual1.503GB BF16 payload
was downloaded and verified at input freeze. The subsequent
[full comparison](../../docs/qwen3-full-quality.md) now executes that baseline
and the frozen native candidate on the complete workload.
The converter records its source repository without a source revision. The
baseline uses the official revision contemporary with conversion; its weight
digest matches the later official revision as well.

All inputs replay independently in Python and through Rust's tokenizer/plain
Qwen3 rendering, including70 app Chat-codec cases. Twelve coherent corruption
controls and two Rust negative controls reject. The explicit short worker probe
is non-quality. Source/model/runtime endpoints and process cleanup pass.
Current numerical qualification remains rejected. The subsequent full comparison
passes the relative limits with weak strict scores3/100 versus5/100, exactly
the two-case loss limit. App-window and full-model-window qualification stay open.
[reference-plan.json](reference-plan.json) and
[reference-actual-profile.json](reference-actual-profile.json) preserve the
reference profile frozen before held-out execution. Input-manifest bytes and
historical preparation flags remain unchanged. See
[the contract report](../../docs/qwen3-quality-contract.md) and
[public evidence](../results/2026-10-27-qwen3-quality-inputs-v1.json).
