# LFM3 external-engine configuration preflight

The installed Transformers 5.20.2 rejects the original pinned LFM3 candidate's
configuration: `lfm3` is absent from its built-in AutoConfig mapping, and the
candidate omits an AutoConfig declaration. This blocks a same-checkpoint vLLM
comparison before model loading. The installed versions are vLLM 0.33.0+cpu,
vLLM-Metal 0.33.0, MLX 0.32.4 and MLX-VLM 0.7.11.
[Original admission probe](../artifacts/lfm3-engine-comparison-2026-11-11/config-probe-result.json).

`tools/bench/lfm3_external_config.py` verifies the complete 3,457,635,448-byte
candidate and its six configuration/text/code assets against the existing
authenticated source contract. It copies metadata into a new directory and adds
only `auto_map.AutoConfig`, pointing at the unchanged pinned parent configuration
class. The MLX-LM variant preserves the publisher's custom module. The MLX-VLM
variant additionally removes `model_file`, explicitly selecting the installed
built-in LFM3 implementation. Every numerical configuration field stays unchanged.
The original checkpoint is never written. The derived directory contains no
weights, weight links or model instance and cannot load a model.
[MLX-LM receipt](../artifacts/lfm3-engine-comparison-2026-11-11/mlx-lm-config-1/receipt.json),
[MLX-VLM receipt](../artifacts/lfm3-engine-comparison-2026-11-11/mlx-vlm-config-1/receipt.json).

The offline installed-runtime probe admits both local AutoConfig variants.
MLX-VLM's actual module/config selection rejects the publisher's MLX-LM module
because it has `ModelArgs` rather than `ModelConfig`. The built-in variant passes
that selection. No model constructor is called; an audit hook prohibits model
weight-file opens. The guarded process and supervisor's conservative lifetime
RSS sum is 536,625,152 bytes, below the separate 512 MiB reference budget.
This result qualifies configuration preparation only: vLLM engine startup,
complete loading, numerical correspondence and memory/performance qualification
are still required.
[Installed probe](../artifacts/lfm3-engine-comparison-2026-11-11/installed-config-probe-result-1.json),
[resource guard](../artifacts/lfm3-engine-comparison-2026-11-11/installed-config-probe-1-guard.json).

Eight portable controls cover source preservation, declared dispatch-only changes,
the absence of weights/links, changed weights/tokenizers/configuration code/
contracts, and existing or in-checkpoint destinations. An initial synthetic
fixture used an incorrect byte count; the failed log is retained and the fixture
now derives its actual size. An additional growth-after-stat control verifies
the actual metadata read stays bounded. Astra approves with no findings.
Check results and review are recorded in the
[publication receipt](../artifacts/lfm3-engine-comparison-2026-11-11/publication-checks.json).
Cargo/native inputs are unchanged; prior required checks are reused by exact
input comparison rather than rerunning unrelated numerical suites.

Next, run the exact BF16 weights through external startup and matched workloads
after the resident quality jobs finish, under the unchanged 6 GiB ceiling. The
built-in MLX-VLM graph is a different arithmetic implementation; configuration
admission does not establish its equivalence to the publisher or Toklet graph.
Retain token/output differences and each timing boundary. Ollama's private-store
import must independently authenticate every imported tensor before inference.
Its versioned [create/blob API](https://github.com/ollama/ollama/blob/v0.40.15/docs/api.md#create-a-model)
allows uploading source files directly to an owned server without using the CLI's
default model cache. No import or performance result is claimed here.

The subsequent [request observation report](lfm3-engine-comparison.md) now
records complete private-import authentication, four fresh external cohorts,
their guards and retained failures. That separate scope preserves missing
Ollama prompt/output IDs, observed cross-engine output differences and distinct
CPU/API/server clocks. It does not qualify numerical equivalence, quality,
speedup ratios or managed adoption.
