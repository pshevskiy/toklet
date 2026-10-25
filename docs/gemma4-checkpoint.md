# Immutable Gemma4 checkpoint and Mac storage load

The [receipt](../benchmarks/results/2026-10-25-gemma4-checkpoint-v1.json) binds
data acquisition and a data-only Mac load through the
[cold payload store](gemma4-payload-storage.md). This does not execute the E2B
numerical graph, qualify original-BF16 inference, run the tokenizer/template,
add app eligibility or qualify any iPhone memory allowance.

## Candidate and transport

The candidate is `mlx-community/gemma-4-e2b-it-4bit` at immutable revision
`238767527555cb75a05732a84dff5d6ba0dd6809`. Its single safetensors container is
3,550,670,554 bytes with SHA256
`038e39a37a7667373d2c3991375446b10c96ae1d717a68674870343db376b76e`.
Nine selected data files total 3,583,087,091 bytes; metadata totals 32,416,537.
They include config, index, tokenizer/config, generation/processor config,
chat-template text and README. No remote repository code or Jinja executes.

A fixed data-only downloader uses at most four owned children with 16MiB ranges.
Every response admits status, exact Content-Range, unencoded body, size and
captured diagnostics. Independent chunk and whole-file hashes precede atomic
non-overwriting publication. Git/LFS identities check every metadata file.
The container requires 212 ranges; inventory and eight metadata transfers bring
the accepted record set to 221 transfers. One initial response ended early;
that run remains failed, with four rejected/canceled attempts retained. Recovery
reuses 195 independently verified successful ranges and makes 26 new HTTP calls
for the remaining 17 ranges, fresh inventory and metadata. Across both runs
there are 226 actual HTTP attempts, including the obsolete first inventory.
Temporary chunks and all failed raw evidence remain available locally.

The aggregate cap is 4GiB, with 64MiB reserved for selected metadata and separate
storage headroom before transfer/assembly. HTTPS redirects use a bounded 4KiB
transport envelope for tiny final files; final size remains its original exact
API limit. HTTP/1.1 avoids the earlier observed HTTP/2 framing problem. Deadline
checks precede child creation, follow completion and precede publication. Worker
failure/cancellation joins and reaps all owned processes before returning.

## Native storage probe

The explicit `gemma4_payload` example pins config/index and the container digest
independently, uses the existing 6GiB desktop logical allowance, and resolves all
540 text-weight views after integrity/value admission. It retains 278 packed
matrices and 262 converted FP32 vectors, without expanding matrices or retaining
excluded audio/vision payload. Actual numerical execution requires the next
separately reviewed E2B geometry/model task.

The controller creates exclusive outputs, checks all inputs before/after every
owned command, retains raw success/failure streams, and restores signal handlers
when publication fails. The probe runs in a fresh native process. All files are
hashed beforehand, so filesystem cache is uncontrolled/warm. The raw process
resource report is a single storage-load observation; it is not a qualified
latency, generation throughput, end-to-end memory fit or thermal result.

The expected tensor-capacity requirement is 2,604,553,356 bytes and the conservative
load reservation is 2,671,730,828 bytes. The actual probe matches both requirements and resolves all540 views. On
MacBookPro18,3 / Apple M1 Pro /16GiB, macOS26.6.2 build25G83, the fresh debug
process reports maximum resident memory of1,520,435,200 bytes and peak physical
footprint of2,616,527,296 bytes. These are separate Darwin process metrics;
neither replaces the conservative logical reservation. The single load takes
183.83 seconds with prior hashing and uncontrolled/warm filesystem cache. This
is a debug storage-validation observation, not an optimized inference benchmark.
Sessions, tokenizer, application state and future GPU allocations need their own
admission; no mobile allowance is inferred from this Mac result.

## Verification and remaining work

Eleven portable tests cover exact identity/inventory limits, response/range
checks, chunk assembly, admitted inventory buffers, close and diagnostic failure,
existing-output refusal, actual child cancellation/reaping, small redirect
responses and expired deadlines. Four recovery regressions execute the actual
scheduling function with bind/result/submission faults and confirm every
submitted future is drained while the first error remains intact. The initial nine-test source remains retained;
its redirect/deadline fixes precede any actual download. The pre-correction
cold controller is likewise retained as unexecuted preparation. An unsupported
formatting invocation ran no formatter; the corrected required check passes.

Formatting, strict default/Metal Clippy, 291 CPU tests, 313 Metal-feature tests,
both iOS compile targets and the example build pass as nine required/build
commands. Numerical tests keep their previous fixtures and tolerances. Astra
reviews the source and final raw evidence before the task is committed.

Next are real E2B geometry and 15-producer/shared-KV execution, independently
frozen real numerical/tokenizer/chat/generation witnesses, resident Metal,
quality and representative Mac performance, independent phone admission and app
integration. The existing 15 packed-GPU, 15 dense-GPU and 8 long-rotary failures
remain unwaived. Gemma4 Chat and 131072 output tokens are not granted by a download
or storage load; the latter still needs prompt/history capacity in addition.
