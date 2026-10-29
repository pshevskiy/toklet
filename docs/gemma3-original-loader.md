# Gemma3 original-byte loader

29 October 2026. The selected Gemma3 artifact can now load into one immutable
packed CPU owner and upload into the resident Metal model without expanding a
second FP32 weight model. Actual file/storage/upload tests and independent
original-range reconstruction pass. Native forward/text qualification remains
separate.

## Loading and planning

`mlx::gemma3::payload::load` consumes a reader plus expected config/index/header/
container identities and a memory ceiling. It validates the strict inspected
plan, byte counts, layout, hashes, finite BF16 norms and finite affine endpoints
before publishing the store. Short/trailing/corrupt/budget-rejected inputs fail.
Streaming hash scratch is bounded; matrix access borrows original byte slices.
Three shared cold I/O/validation helpers also serve the existing Gemma4 loader;
its API and numerical interpretation remain unchanged.

The real payload owner is150,885,632 bytes. Admission reserves64MiB for metadata
and64KiB streaming scratch:218,060,032 bytes before additional caller owners.
Weights remain original Q4/group64/BF16 data. Small typed BF16 gamma vectors and
layer metadata exist only during explicit cold preparation/upload. Caller-owned
container buffers, tokenizer/application state and other live owners must be
reported through the external-residency argument.

`prepare_raw_gamma` resolves names once and constructs the explicit raw-gamma
input. Requested capacity is1..544; declared context32,768 and local window512
remain separately reported metadata. For smaller capacities, physical local
retention is `min(512,capacity)`, equivalent within the admitted prefix.
`metal::gemma3::Model::from_payload_raw_gamma` uploads synchronously into original
packed owners, forms FP32 `1+raw BF16 gamma` and retains no CPU-store lifetime.
Cold source/upload overlap and actual metadata capacities enter checked memory
accounting before allocation. Storage admission is broader than the separately
checked native kernel domain; a successful parse alone does not enable Chat.

## Checks and evidence

Formatting, strict default/Metal Clippy,431 default tests,327 Metal library tests
and26 overlapping targeted Gemma3/Gemma4 controls in debug/release pass. Seven
new portable loader controls exercise tiny synthetic storage and failure cases;
ordinary tests require neither model download nor GPU.

Two explicit release tests use the actual authenticated container. Each emits
236 source logical records:127 packed matrices and109 raw norms. Capacity
mapping checks1/12/511/512/513/544;0/545/usizeMAX reject. Resident upload emits237
logical records covering491 unique physical buffers, before and after dropping
the CPU store. A separate original-file reader reconstructs all component
hashes and55,936 FP32 effective coefficient words, including guards and zero
bias. Both resident passes match exactly, including addresses.

The actual resident model report admits218,657,930 bytes and a conservative cold
peak bound436,781,146 bytes, with no session allocated. Direct observed lifetime
physical maxima are156,123,688 /325,321,496 bytes for the two test processes;
sparse observations are not continuous peaks. No Session, forward call, explicit
GPU command submission, speed or numerical-model qualification occurs here.

A controller-placement error was rejected before launching any child; its
receipt remains. The corrected approval changes only inventory placement. An
independent reader's initial capacity assumptions were also corrected from the
actual literal controls; target code/output/gates were unchanged.

See [evidence](../benchmarks/results/2026-10-29-gemma3-original-loader-v1.json),
[acquisition](gemma3-artifact-acquisition.md),
[frozen real reference](gemma3-real-reference.md) and [TODO](../TODO.md).
