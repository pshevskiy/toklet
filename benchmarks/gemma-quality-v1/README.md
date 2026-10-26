# Gemma quality inputs v1

This directory versions the full Gemma-specific quality manifest and 10001 raw
little-endian u32 loss IDs. It freezes inputs before scoring; it contains no
quality result. See the [contract and actual checks](../../docs/gemma-quality-contract.md)
and [public evidence](../results/2026-10-26-gemma-quality-contract-v1.json).

| Versioned file | Bytes | SHA-256 |
| --- | ---: | --- |
| `manifest.json` | 28428 | `c4e6fc5cce765a30c43c9ed2d3ee4bfda58a4b23bbfc3284e2160431e6fb6506` |
| `held_out_loss.u32le` | 40004 | `877cbe18574b47e4b6a8d1b58fb79621c37e0531bd1f9504e50ce63d186f8c46` |

The manifest also references `tasks.jsonl` (188152 bytes) and
`loss_verification_prefix.txt` (65536 bytes). These held-out text files are
intentionally absent here. The complete accepted directory is locally retained
at `artifacts/m8-gemma-quality-contract-checks-v2/contract`. Its exact hashes
appear in the manifest/public evidence. Do not run the full checker against
this reduced Git directory or replace missing text with invented payloads.

Reconstruction requires the four original datasets at paths in
[sources.json](../sources.json), the pinned candidate metadata/tokenizer at
`models/MLX-Gemma4-E2B-4bit-2387675`, official source inventory at
`artifacts/m8-gemma-quality-source-metadata-v2/inventory.json`, and the pinned
official config at
`artifacts/mobile-gemma4-testing-2026-10-24-v1/official-E2B/config.json`.
The preparer checks exact hashes and refuses mismatches. The public receipt
records the official inventory acquisition and conversion-source lineage.
There is no need to download or load model weights for this codec-only recipe.

Use the pinned Python environment from the public receipt: Python 3.12.8,
NumPy 2.3.2, PyArrow 20.0.0, tokenizers 0.22.2, Jinja2 3.1.6 and
MarkupSafe 3.0.4. For a fresh local reproduction, create separate byte inventories
before importing parser/tokenizer extensions, using a new parent directory:

```sh
.venv/bin/python -s -B - <<'PY'
from pathlib import Path
import hashlib, importlib, json, sys
sys.path.insert(0, str(Path('tools/bench').resolve()))
out = Path('artifacts/gemma-quality-replay-new')
out.mkdir()
for role, name in [('prepare', 'prepare_gemma_quality'), ('replay', 'check_gemma_quality')]:
    module = importlib.import_module(name)
    files = {}
    for path in module.runtime.paths(module.sources()):
        with path.open('rb') as file:
            files[str(path)] = dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(file, 'sha256').hexdigest())
    raw = (json.dumps(dict(format='toklet-text-reference-runtime-v1', versions=module.runtime.VERSIONS, files=files), indent=2) + '\n').encode()
    with (out / (role + '.json')).open('xb') as file:
        file.write(raw)
    print(role, hashlib.sha256(raw).hexdigest())
PY
```

Run each CLI below in a separate cold `.venv/bin/python -s -B` process with a
fresh `-X pycache_prefix=ABSOLUTE_NEW_DIRECTORY`. Set `PYTHONHASHSEED=0`,
`TOKENIZERS_PARALLELISM=false`, and `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
`ARROW_NUM_THREADS` to `1`. Substitute the printed byte-inventory digests:

```text
tools/bench/prepare_gemma_quality.py
  --output artifacts/gemma-quality-replay-new/contract
  --official artifacts/m8-gemma-quality-source-metadata-v2/inventory.json
  --runtime-manifest artifacts/gemma-quality-replay-new/prepare.json
  --runtime-sha PREPARE_DIGEST

tools/bench/check_gemma_quality.py
  --contract artifacts/gemma-quality-replay-new/contract
  --manifest-sha c4e6fc5cce765a30c43c9ed2d3ee4bfda58a4b23bbfc3284e2160431e6fb6506
  --official artifacts/m8-gemma-quality-source-metadata-v2/inventory.json
  --runtime-manifest artifacts/gemma-quality-replay-new/replay.json
  --runtime-sha REPLAY_DIGEST
```

Preparation must reproduce both versioned files byte-for-byte. The checker
independently reconstructs all source cases, answers and tokenization. Use
`--exercise-controls` in another fresh checker process for the six coherent
source corruptions. `tools/bench/test_gemma_quality.py --recipe` needs only the
standard library; `--capture COMPLETE_CONTRACT_DIRECTORY` runs eight structural
controls. The explicit Rust `gemma4_quality_contract` example takes the pinned
model directory, complete contract directory and manifest SHA as arguments.
Build it in a separate Cargo target to preserve qualified release binaries.

These commands reproduce preparation, not the owner/process/memory qualification
recorded by the reviewed collection controller in the public evidence. They do
not run numerical inference, grade generated code or verify official weights.
Keep full task caps and all ten loss blocks: the suite needs 1025 positions,
while its largest complete prompt/output task needs 567. A 32-position run
cannot substitute for it. The model's full shared window remains a separate
open requirement. Preserve the dataset licenses/attributions in sources.json
when redistributing any derived text.
