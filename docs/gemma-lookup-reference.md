# Complete real embedding and per-layer lookup references

This fixture freezes two complete arrays from the already accepted
[independent Gemma text-math capture](gemma4-text-math.md). It prepares real
original-table lookup comparisons after the [small resident GPU gather](gemma-table-gather-gpu.md).
It is an extraction integrity result; no engine, GPU, model generation or new
numerical evaluation runs in this task.

The source is the pinned staged-FP32 contract, with reconstructed-FP32 packed
weights and token IDs `[2, 100, 257, 100]`. The repeated token supplies identical
rows at positions 1 and 3. These are raw lookup outputs before embedding scaling
or later PLE processing.

| Role | Original full shape | Complete F32 values | Bytes |
| --- | --- | ---: | ---: |
| Embedding | `[1, 4, 1536]` | 6144 | 24576 |
| Per-layer lookup | `[1, 4, 8960]` | 35840 | 143360 |
| Total frozen binary | Two arrays | 41984 | 167936 |

The [fixture](../tests/fixtures/gemma-lookup-reference-v1/observation.json)
records source offsets, complete hashes and shapes. Producer and checker select
explicit spans independently. They reuse the earlier head reference's bounded IO,
source authentication and strict record helpers at the artifact boundary. The
checker imports no new producer and compares every copied byte and finite value
against authenticated full and compact original trace metadata. The original
source recipe and complete 103403520-byte trace are authenticated before and
after extraction.

Two captures and their independent replays are byte identical. Seventeen actual
output mutations/reader checks reject contract/name/offset/shape/array changes,
coherently changed values/hashes, incomplete/trailing data, capabilities,
duplicate JSON and nonfinite JSON. Three retention checks preserve the completed
first array after a late descriptor failure and the original cause through
report/manifest publication failures. The frozen binary cap is 256 KiB; reports
are bounded to 64 KiB and each owned process to 512 MiB/600 seconds.

The [public evidence record](../benchmarks/results/2026-10-26-gemma-lookup-reference-v1.json)
binds exact scripts, copied helper modules, interpreter, source identities,
complete outputs, process resource measurements and cleanup. Maximum observed
resident set is 69533696 bytes and physical footprint 31933136 bytes, both below
the separate process allowance. This is not model residency or a timing result.
Protected release executables and every Rust/shader/test source stay unchanged;
no new Cargo suite is required for this reference-only task.

The three fixture files are copied exclusively from verified captures. Their
complete identities are:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `observation.json` | 2163 | `b1818331b5d341444fbc4ea22e018fc99241bd2d4fe5edbca7df9222739546a1` |
| `arrays.bin` | 167936 | `aea89e9e83d266ae5e648e1c4bebb01509370ef29ef777dea7018b5822cbf469` |
| `manifest.json` | 386 | `6651c175ea8d90e9759ea40b8746e6c816a8c867a969b9075e5d83e50c860a95` |

This does not qualify native original-table gather or loading, bounded upload
overlap, resident whole graph, source/BF16 precision, app/phone, quality or
performance. Historical cross-precision failures (five arrays/57 values), native
ideal-FP64 logit failures, source GPU intermediates and long rotary discrepancies
remain open. The original PLE representation alone is 1321205760 bytes and needs
separate whole-model admission. No capability or context/output limit changes;
131072 output tokens plus a nonempty prompt remain unqualified.
