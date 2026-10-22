# Immutable eight-bit MLX candidate acquisition

The next explicit local candidate is
[mlx-community/Qwen2.5-0.5B-Instruct-8bit](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-8bit/tree/6474d4c1a05bd7bd634a9c993560e4e7272eda3d),
revision `6474d4c1a05bd7bd634a9c993560e4e7272eda3d`. It is an off-the-shelf
higher-precision alternative to the [rejected four-bit candidate](mlx-full-loss.md).
No quantizer calibration, held-out tuning, threshold change or model inference
occurs in this acquisition task. Independent numerical reference gates must be
frozen before any target trial.

The data-only downloader retains the accepted HTTPS, bounded size/time,
per-file Git/LFS hash and exclusive publication mechanism. Three portable
methods pass, including real exclusive directory/dangling-link/file publication
checks. Eleven owned download commands pass. Ten whitelisted files total
536,580,527 bytes; the weight file is 525,045,902 bytes. It is published locally
under `models/MLX-Qwen2.5-0.5B-Instruct-8bit-6474d4c` without replacing any prior
model or receipt. No repository Python or model-card commands execute.

| Selected input | SHA-256 |
| --- | --- |
| config.json | `4dae7c9622e41e50752ba422149df0694044266c06cd851aff630799701c83c6` |
| model.safetensors | `3dd0b6c2983ac5fe35f60ba260b1c7c35e4c38f17f3d5139d0bb477924e7aef4` |
| model.safetensors.index.json | `e48752ffebac15ee35d1d86e96c9b87f2efd7783b19ebf82a26955dbd43e0476` |

Bounded data inspection finds Qwen2/tied weights and eight-bit/group64 settings,
628 tensors (169 U32 packed matrices and 459 F16 tensors), consistent packed
matrix/metadata shapes, contiguous payload offsets and exact index mapping.
This is metadata inspection, not numerical payload or native loader qualification.
The BPE vocabulary, merge order and added-token ID/content pairs match the frozen
original source. Tokenizer files and chat template differ; the existing strict
CLI template policy remains unchanged and this template is unqualified.

The [acquisition receipt](../benchmarks/results/2026-10-22-native-mlx8-checkpoint-v1.json)
and [raw inventory](../benchmarks/results/2026-10-22-native-mlx8-checkpoint-raw-v1.json)
bind downloaded files, consumed metadata, source copies, checks and protected
historical CPU binaries through publication. Rust, shader and Swift sources are
unchanged. There is no inference, device-fit, quality, performance, broader
architecture or product download/search claim from downloading this candidate.
The four-bit rejection and all frozen numerical/quality policies remain intact.

Astra accepted data acquisition only after verifying 99 pins, all ten Git/LFS
file identities, 84 raw files, 51 source copies, exact metadata/tokenizer checks,
all download/portable outcomes and 50 unchanged numerical/app sources. Privacy,
all 223 local links and whitespace checks pass. Numerical and device qualification
must still precede adoption.
