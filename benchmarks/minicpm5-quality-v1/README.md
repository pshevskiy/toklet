# Complete MiniCPM5 quality inputs

[manifest.json](manifest.json), SHA-256
`fe954ea5afa82f3fc33981c139727bef4fb3cd10f1b6485b7f0001f3b43875a7`,
freezes all 10,000 loss targets and 100 original tasks before model scoring.
[held_out_loss.u32le](held_out_loss.u32le) contains 10,001 IDs / 40,004 bytes,
SHA-256 `297451dd509fd620d91a997dac58ad40f514f4fa2f63efcc92253a314c7476cc`.

The full local directory is
`artifacts/minicpm5-quality-inputs-2026-11-10/inputs-1`. It also retains
`tasks.jsonl` (211,893 bytes) and `loss_verification_prefix.txt` (65,536 bytes).
These source text files are intentionally absent from Git; their exact hashes
are in the manifest. Preserve the dataset licenses and attributions in
[sources.json](../sources.json) when handling derived text.

The suite retains the original first 40 HumanEval, 30 TruthfulQA and 30 LAMBADA
cases, full 256/32/16 output caps and ten contiguous loss blocks. Loss/task
thresholds remain 0.05 nats/token and two percentage points. Quality is **not
scored**. Generated code and numerical models have not executed in preparation.

Three distinct identities remain explicit: the original
[stock checkpoint](https://huggingface.co/openbmb/MiniCPM5-2B/tree/42433e90dad95123ab87729fd7011e7110629556),
the modified BF16 parent at `8d83eaf4`, and the selected MLX8 child at `50ab1b22`.
Their two quality comparisons remain separate. Stock weight payloads are not
downloaded or authenticated by this input step. All corpus IDs and all 100
task renders/IDs match the original stock metadata under explicit single-case
settings; differing wrappers, padding and template tool branches are retained.

Use the fresh external runtime manifests and owned controllers linked in the
[scope report](../../docs/minicpm5-quality-inputs.md) to reproduce preparation.
The producer and replayer require `-s -B`, a fresh absolute `-X pycache_prefix`,
the explicit shared environment root, exact five candidate text assets, pinned
stock metadata, and external contract/runtime digests. The replayer additionally
requires the manifest digest above. It reconstructs the complete recipe and
compares every payload byte; it is not an independent numerical oracle or a
new native codec qualification.

The full suite needs **1,025 positions**; the largest task, HumanEval/32, needs
559 including its full output allowance. Current engine diagnostics stay at
512, so they cannot run this complete contract yet. No prompt, cap, loss block
or original acceptance gate is shortened to fit that limit.

Astra approves this input-preparation scope after a fresh complete source
replay. [Checks/review/commit](../../artifacts/minicpm5-quality-inputs-2026-11-10/publication-checks.json).
Full numerical/context and both quality comparisons remain open.
