# Independent eight-bit Qwen graph reference

The [checkpoint-specific CPU exporter](../tools/reference/mlx_qwen_real8.py)
freezes the [immutable eight-bit candidate](mlx8-checkpoint.md) before any native
target trial. The previously frozen four-bit exporter, arrays and rejection are
unchanged. Both references share the independent numerical graph functions;
their artifact-specific readers and producer identities remain explicit.

Each little-endian U32 packs four unsigned bytes, least-significant byte first.
A row with `in_features` columns stores `in_features / 4` words and separate
Half scale/bias arrays with `in_features / 64` groups. Independent FP64
`code * scale + bias` reconstruction rounds once to FP32. Literal division/modulo
tests include 0, 1, 127 and 255, signed/zero metadata and the FP32 rounding boundary.
They do not call the Rust decoder or GPU kernel.

The graph retains the same Half gather/exclusion/activation/KV semantics and
original reconstructed FP32 tied-output weights. One tied table and one current
matrix/stripe are retained; it does not require a second fully expanded model.
Six authored prompt/offset cases cover lengths 1/3/4/17, offsets zero/five,
selected layers 0/1/23 and four full greedy distributions. FP32 and FP64 paths
produce the same four teacher-forced IDs: `[4460,311,1855,264]`.

All **346 arrays / 38,461,888 bytes** pass exact layout, finite-value, slice and
whole-file checks. The numerical policy remains
`atol = max(1e-4, 2*same-stage FP32/FP64 error, 2*equivalent-prefix error)` and
`rtol = 1e-3`, frozen before target evaluation. The largest absolute allowance is
1.375; the largest logits allowance is 0.2790870666503906. These independent
numerical envelopes do not replace the separate held-out loss/task gates.

| Frozen output | SHA-256 |
| --- | --- |
| manifest | `1f98482bceba6c5627767d50cd2c4bda461d97174ee6e29346c5223215270531` |
| arrays | `73122c407578baa5a70f41de232b127f02e5775c8cd8b4c6ccc081cdf50ba084` |

The [committed manifest](../tests/fixtures/mlx-qwen-real8-v1/manifest.json) binds
all input/producer identities, 19 selected NumPy native binaries and the exact
policy. Verification requires an external manifest SHA and consumes each array
through the same descriptor. Logical reference admission is 2,009,108,622 bytes;
observed lifetime peak physical charge is 629,425,784 bytes under the independent
6 GiB Mac reference cap. File-backed pages and RSS use different accounting.

Seven new portable methods and seven existing four-bit methods pass. All four
owned commands pass: both suites, CPU export and stdlib-only external-SHA
verification. Output paths/logs reject existing files or dangling links before
starting any child. Sources, output endpoints and protected historical CPU
binaries remain unchanged during collection and publication.

Receipts: [reference qualification](../benchmarks/results/2026-10-22-native-mlx8-real-oracle-v1.json)
and [raw inventory](../benchmarks/results/2026-10-22-native-mlx8-real-oracle-raw-v1.json).
Retain this frozen CPU oracle for native target checks. No eight-bit target GPU,
native app, device, full quality, performance or adoption result is claimed here.

Astra accepted the frozen CPU reference after verifying 98 pins, 72 raw files,
59 source copies and every array's values/layout/hash/gate, plus all four owned
commands, 14 portable methods and exact greedy IDs. The four-bit oracle and
policies remain unchanged. Privacy, 227 links and whitespace checks pass.
