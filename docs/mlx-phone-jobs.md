# Native MLX app checks on iPhone 14 Pro

The [physical transport](../tools/experiments/m8_physical_mlx.py) runs sealed
version-two MLX measurement, quality and task jobs in the existing signed app.
It admits the exact native package, signed installation, unexpired profile,
device identity, model inventory and job bytes before copying or launching.
The [shared stager](mlx-mac-jobs.md) keeps tokenization and downloads outside
the engine. This task changes host transport and validators; Rust, shaders,
Swift app sources, numerical gates and the signed app remain unchanged.

The selected [public Qwen checkpoint](mlx-real-oracle.md) executes on the wired
iPhone 14 Pro (iPhone15,2, iOS27.0.1). Ten stages pass: one tiny transfer-only
preparation, three fresh tiny observations, three public-checkpoint transfers
and three no-copy public-checkpoint launches. Each transfer requires its own
successful same-app/device tiny observation, dual-clock and host-boot freshness,
a 300-second maximum age and a 2 GiB free-space reserve. Freshness is checked
immediately before every copy. Auxiliary payloads and canonical job files have
explicit device-side inventory checks. A saved ready receipt enables fast
launch without another model transfer or signing operation.

Both preflight and final cleanup verify the exact owned app path and selected
device; failed or wrong-device queries cannot establish absence. Cleanup remains
available after a model/source endpoint failure. Cleanup records 19 natural
absence endpoints and one verified initial preflight reset of the exact installed
app using SIGTERM, before transfers or measurements. All 20 endpoints finish
absent without errors. All 104 transport commands (94 devicectl commands and ten
host sysctl boot queries), ten outer child commands and seven portable commands pass.

Three tiny jobs and the public measurement each complete 3 warmups and 10 reset
requests, with four exact IDs per request. Public IDs are
`[264,5458,315,17407]`, matching the frozen independent reference. The public
quality check scores all three short targets, preserving positions and stable
aggregation. The public task check collects the first original task with its
eight-token short cap and validates every chunk, callback, count, hidden stop,
reset and complete response. These are functional checks, not full quality gates.

Observed native lifetime peak physical charges are:

| Public-checkpoint app job | Peak physical bytes |
| --- | ---: |
| Short measurement | 307,971,936 |
| Three-target scoring | 306,284,384 |
| First task, short cap | 397,641,664 |

The three tiny peaks range from 21.37 to 21.51 MB. Physical footprint includes
the process lifetime peak and differs from RSS and clean/shared page accounting.
These observations stay within the explicit 2.5 GiB trial process cap; numerical
admission remains separately bounded at 1.5 GiB for the public model. They do not
establish fit at longer contexts or under sustained/OS lifecycle workloads.

The shared validators now accept the explicit `metal-mlx` backend only for an
exact integer version-two job with true Metal/MLX flags and no separate Q4
folder. Legacy version-one schemas remain intact. Ninety-six portable methods
pass, including actual mocked transfer/ready reuse, consumed-byte replacement,
auxiliary listing failure, strict schema types, storage-proof substitution,
cached signing-proof substitution and cleanup after source mutation. Private
device/signing logs remain local and digest-bound; published receipts redact
identifiers. Sources and protected historical CPU binaries match their endpoints.

Receipts: [qualification](../benchmarks/results/2026-10-22-native-mlx-phone-jobs-v1.json)
and [raw inventory](../benchmarks/results/2026-10-22-native-mlx-phone-jobs-raw-v1.json).
Retain the guarded transport and selected physical execution. Full 10,000-target
quality, 100 task cases, representative/sustained performance, physical state and
OS lifecycle, chat-template compatibility, other architectures and in-app
discovery/downloads remain open. No overall M8 or physical-device qualification
is claimed by these short jobs.

Astra accepted the source and completed evidence after checking 484 provenance
pins, all raw files/source copies, six actual console replays, original/reused
storage decisions, command outcomes and privacy. The corrected publication
preserves the initial accounting versions and distinguishes host queries and the
owned preflight reset. All 126 final raw inventory entries and document links pass.
