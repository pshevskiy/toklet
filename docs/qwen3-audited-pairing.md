# Audited Qwen3 model and codec pairing

Two additive native factories bind the pinned Qwen3 model and codec:
`toklet_model_load_mlx_qwen3_audited` and `toklet_chat_create_qwen3_audited`.
[Verification](../benchmarks/results/2026-10-30-qwen3-audited-pairing-v1.json)
covers their Rust/C boundary and one cached Mac ownership probe.

The model attaches the [consumed-source receipt](qwen3-consumed-source.md) only
after successful resident Metal assembly. The chat factory requires that live
audited model, authenticates the actual tokenizer buffer passed to the parser,
and retains that exact model `Rc` only after codec construction succeeds.
Wrong owners/profiles, threads, unsupported builds, missing assets and insufficient
allowances reject without publishing a partial chat owner. Ordinary and existing
Smol factories retain their previous behavior.

Historical evidence distinguishes these roles:

| Asset | Role |
| --- | --- |
| config/index/container | Authenticated original source consumption, attached after assembly |
| tokenizer | Authenticated buffer actually parsed by the codec |
| tokenizer config | Verified against fixed Rust text policy; arbitrary Jinja is not executed |
| special-token map | Verified distribution asset; not an inference input |

The canonical six-file identity remains
`5e9aafa305e0e48182a856cafe3ae2b597599033e61b16664f42b09a4af5ff51`.
It describes historical matching content, not a simultaneous directory snapshot
or a hash of resident GPU buffers. Standalone codec evidence cannot establish
native model ownership; the native factory checks and retains the actual owner.
Raw token APIs can still pass IDs to another session.

All twelve cold commands pass: formatting, strict Clippy, 494 default tests,
602 Metal-feature tests, exact 34 library exports, C compile/link/invalid-input
checks and a release test build. Their respective 21 and 188 ignored tests remain
separate. Eight new portable controls run in both configurations. Owner-local
copies preserve the three compiled products before the actual test is launched.

The single reviewed cached test assembles one model with loaded ceiling 512,
using the existing 1.5 GiB numerical budget and a separate 256 MiB text allowance.
It creates no session or KV allocation and runs no generation. Its five raw
records show a null result/no retained owner on missing codec assets, exact
model owner count 1→2, then 1 after external model-handle release. The retained
chat still encodes/decodes the synthetic `hello` request and returns stop 151645.
Chat release destroys the final owner. All 706 source ranges/reads and all six
asset hashes match the pinned catalog; an opposite-profile rejection also passes
inside the test. Astra independently audits the saved results and closed groups.

The direct test has 92 sparse process-usage samples. Largest sampled values are
838,483,968 bytes RSS, 772,719,744 physical footprint and 812,532,864 for the OS
lifetime-maximum field. These are observed native-probe metrics, not a complete
peak measurement, app/device fit or performance qualification.

No numerical kernel or context policy changes. The selected Apple package still
has 32 exports; the new 34-export library is verified and retained separately.
Swift app adoption, an accepted-load identity report, preferences, generation,
rendered UI and iPhone checks remain subsequent work.
