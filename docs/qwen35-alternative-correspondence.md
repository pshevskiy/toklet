# Qwen3.5 alternative source correspondence

The finite 4B source audit establishes a complete constructive conversion
witness for `TheWirelessPhoenix/Qwen3.5-4B-abliterated_4bit` at
`916d9f7c2e27ed41e2b9ed39f3a2badccb2bbc7c`, using the BF16 Huihui parent at
`5581467dfd52bf338c782006a6cdce05c42594be`. This is a separate candidate from
the previously rejected 4B source. Original numerical gates remain unchanged.

All three complete files authenticate against their pinned SHA256 values,
totalling 11,445,857,581 bytes. Each is hashed again before and after the value
audit, with stable file descriptors and path endpoints. Headers are read from
the actual files; the checker independently reconstructs the text mapping.

The two arithmetic witnesses were fixed from MLX-LM 0.31.3 and MLX 0.31.2
source before reading tensor values. Both use staged FP32 arithmetic and BF16
round-to-nearest-even stores. CPU uses ties-to-even code rounding; Metal uses
half-away rounding and its source's zero-initialized group maximum. Code
generation uses the unrounded FP32 scale and bias. Eight low-to-high nibbles
form each little-endian U32 word. No per-tensor profile mixture is permitted.

The complete Metal witness matches **every retained bit**. CPU has 6,863,958
differing code/metadata words. Coverage is 426 parent arrays, 249 matrices,
177 unquantized arrays, 924 child descriptors, 65,699,840 groups, 10,820 row
blocks and 961,536 unquantized elements. Typed norm shifts, singleton convolution
axis movement and copied arrays all match. The 297 vision arrays are excluded
from the text mapping; their complete containers are authenticated.

The value audit reads 8,411,502,592 parent text bytes and 2,367,117,312 child
payload bytes in bounded windows. It completed in 119.47 seconds under the
512 MiB phase guard, with 167,985,152 bytes maximum observed charge including
the 32 MiB helper reserve. These are setup observations, not inference latency
or model fit measurements. The guard samples owned processes; it is not a
continuous allocator cap. Full parent text weights alone exceed 6 GiB, so this
streaming check does not admit a resident reference model.

The result proves correspondence under the defined Metal witness. Historical
converter invocation, installed backend/compiler, dependency versions and
floating-point environment remain unavailable. Native source/codec admission,
the original idealFP64/stagedFP32 128/256 comparisons, state, quality, physical
fit, request metrics and performance gates remain open. Existing source
rejections are preserved. The separately pinned 9B candidate follows the 4B
disposition.

## Portable checks

The 24 independent controls pass in normal and optimized Python. They include
literal CPU/Metal goldens, all 65,280 finite BF16 norm inputs against an exact
integer reference, software nonfinite policies, packing and resource bounds,
readers, complete coverage and admission rejection. They use no model downloads
or GPU. CI runs both modes with CPython 3.12 and hash-pinned NumPy 2.3.2 wheels.

```sh
python3 -m pip install --only-binary=:all: --no-deps --require-hashes \
  -r tools/experiments/qwen35-correspondence-requirements.txt
python3 -B -m unittest discover -s tools/experiments \
  -p test_qwen35_parent_correspondence.py
python3 -O -B -m unittest discover -s tools/experiments \
  -p test_qwen35_parent_correspondence.py
```

The explicit real-file checker is
[`qwen35_parent_correspondence.py`](../tools/experiments/qwen35_parent_correspondence.py).
It accepts only the fixed authenticated case, requires an explicit receipt SHA
and a fresh output file, and runs under a separately admitted process guard.
An early rejection is recorded as incomplete coverage; it cannot establish full
correspondence. Numerical inference and adoption fields remain false.

[Authentication](../artifacts/qwen35-alternative-admission-2026-11-15/source-content-authentication-2.json),
[frozen audit plan](../artifacts/qwen35-alternative-admission-2026-11-15/correspondence-plan-1.json),
[complete result](../artifacts/qwen35-alternative-admission-2026-11-15/parent-correspondence-1.json),
[resource guard](../artifacts/qwen35-alternative-admission-2026-11-15/parent-correspondence-guard-1/result.json),
[unchanged Rust check inputs](../artifacts/qwen35-alternative-admission-2026-11-15/required-check-reuse-2.json).
[Astra approves](../artifacts/qwen35-alternative-admission-2026-11-15/astra-correspondence-review-1/verdict.json)
scoped functional retention with no findings.
[Commit, push and exact-head CI](../artifacts/qwen35-alternative-admission-2026-11-15/correspondence-publication-1.json).
