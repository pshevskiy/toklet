# Native shared-KV routing and staged publication

`model::shared_kv` is a concrete immutable layer-to-producer plan around the
[global/local cache owner](window-cache-scalar.md). The first M layers own cache
buffers; shared layers reuse the last preceding producer of the same attention
kind. Construction resolves routes once, validates positive geometry and all
size arithmetic, rejects missing producer kinds or mismatched consumer geometry,
and keeps the layer lookup outside numerical loops. Only M caches are created,
with no private cache or K/V copies for consumers.

A private frame binds its cache owner, committed offset and generation. The
plan additionally checks complete storage geometry: heads, head width, mode,
window, reserved capacity and maximum positions. All committed owners must have
the same offset and generation. Geometry-compatible owner arrays can be used
with the plan; model/session/token-history provenance remains explicitly the
caller's responsibility. Rejecting a foreign frame relative to a supplied
cache does not establish cross-model/session identity.

Sequential execution prepares producer0 and publishes its frame before later
producers have computed K/V. `published_view` takes an ordered prefix of frames,
checks all committed owners, validates every published frame and common query
end, and requires the requested producer to be present. Empty, oversized,
missing, reordered, duplicate, foreign or stale publication rejects. An
unpublished owner's proposed token count is unknown; it is checked when that
frame is published. Bad committed metadata in any unpublished owner already
rejects early views. Views borrow actual producer old/new spans with immutable
input lifetimes; no cache is appended or copied by a consumer.

The complete-batch `view` requires all M proposals. Final `commit` also requires
and revalidates all M proposals before invoking the cache owner's checked batch
commit. All mutation follows successful validation; discard or rejected
publication/commit leaves the committed prefix intact. Staging K/V and frame
container capacity are caller-owned and must be preallocated/accounted for
steady-state execution. This is a routing/publication/cache-commit primitive,
not yet a Gemma graph or model-wide token-history transaction.

Plan memory reporting includes route Vec capacity and metadata. Cache reporting
includes the owner Vec header/excess capacity and actual per-cache capacities,
without counting occupied cache objects twice. Planned sizes are checked before
allocation; the caller still admits these plus weights, staging, scratch and
application memory against the device budget. These formulas do not establish
Mac or phone process-memory fit.

Eight methods authenticate the prior
[routing witness](gemma-shared-kv-reference.md), compare226 layer views and108
producer updates across all four cases, verify exact retained data and producer
span/new-input pointers, and exercise every geometry field, absent/reordered/
duplicate/foreign/stale frames, last-owner offset/generation/token mismatch,
full discard/retry, moved owners, global limits and allocation accounting.
The sequential test obtains producer0/consumer views before producer1 exists,
rejects partial commit and late invalid publication, then completes one valid
batch. Frame capacity is reused; no performance claim follows.

Formatting, strict default/Metal Clippy,237 CPU/259 Metal-feature tests and both
iOS cross-checks pass. Hardware suites remain explicitly ignored here. The
[receipt](../benchmarks/results/2026-10-24-shared-kv-scalar-v1.json) binds source,
fixtures, checks and unchanged earlier numerical/shader/protected binary hashes.
Astra reviews exact final code/evidence before commit.

Ring-aware numerical attention, full layer/Gemma execution, tokenizer/stops,
model/session replay, Metal residency, rotary parity, quality/performance and
real-model/device memory qualification remain open in the
[Gemma4 plan](gemma4-implementation-plan.md). No native ABI, engine capabilities,
app limits or installed phone package changes follow from this task.
