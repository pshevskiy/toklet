# Frozen benchmark contract v1

The [source manifest](sources.json) pins Qwen2.5-0.5B-Instruct and its tokenizer to
`7ae557604adf67be50417f59c2c2f167def9a775`, including the BF16 safetensors SHA-256.
It also pins MLX-LM, MLX, Transformers, PyTorch, tokenizer versions, and a
llama.cpp commit. The [Mac Python lock](../tools/bench/requirements.lock) records
the complete installed dependency set. These are comparison identities, not a
claim that every baseline has run.

[Contract v1](contract-v1/manifest.json) freezes raw little-endian u32 token
arrays for 128, 512, 1,024, and 1,920 input positions with 128 teacher-forced
output steps. Ignore EOS in controlled throughput runs. The 1,921 + 128 case
must fail admission before execution. Prefix cases cover a 64-token exact append,
an entirely evaluated prompt, a sampled but unevaluated token, and a single edit
at token 128. M2 resets and replays changed histories; later prefix reuse is a
separate optimization. Cold loading uses a new process, warm requests reset the
session with the model loaded, and sustained runs last at least 600 seconds.
The authored multi-turn transcript is a separate application workload.

Timing uses completed work, three warmups, ten raw measured repetitions, median
and min/max dispersion. TTFT starts at request submission and ends at the first
token event. For N emitted tokens, decode rate is `(N - 1) / (last - first)`.
Record first-visible-text latency separately. Identify whether tokenization,
sampling and detokenization are included; controlled token workloads exclude
them. Run a promising optimization again and retain it only beyond measured
variation, normally at least 5% on its declared primary metric. Report other
workloads, memory and quality alongside speed. Controlled and application tracks
have separate labels and must not be combined into one ranking.

The [result schema](result.schema.json) requires device/OS/build/power identity,
source and token hashes, weight/KV representation, cache semantics, timing scope,
raw samples and memory accounting. Unavailable memory, energy, thermal, or
visible-text measurements remain `null`, with their limitations stated. The
initial admitted engine working-set ceiling is 6 GiB; include transient loading,
weights, KV capacity, scratch and retained application buffers. Baseline logical
allocations and process/GPU footprints are different instruments and are
reported separately. The schema permits native BF16 baseline weights and KV;
engine support and numerical qualification remain governed by the plan.

The quality gate is frozen before any lossy evaluation:

- Mean teacher-forced NLL increase at most 0.05 nats per scored token against the
  unquantized checkpoint, using the first 10,001 tokens of the pinned WikiText-2
  raw test corpus. Score all 10,000 targets once in independent blocks of up to
  1,025 tokens, overlapping only the previous token between blocks. Reset state
  at block boundaries; do not average block means with unequal target counts.
- A loss of at most two percentage points across 100 equally weighted binary
  task cases: first 40 HumanEval tasks, first 30 TruthfulQA rows, first 30 English
  LAMBADA test rows. Use greedy selection and the lowest token ID on a tie.
  HumanEval appends the emitted completion to the source prefix and passes only
  if all source tests pass. TruthfulQA rotates up to four frozen options and
  passes only when trimmed output equals the correct uppercase letter. LAMBADA
  predicts the last whitespace-delimited word and compares the first emitted
  word exactly, including case and punctuation. Code/prose use the pinned chat
  template; continuation uses the raw natural-language prefix. Caps and EOS
  rules are stored per case. There is no model-based case selection.

These are project variants, not claims to reproduce the full published task
scores. Task payloads and original corpora stay in ignored `artifacts/`; their
identities, selectors, preparation recipe and prepared-payload hash are in Git.
The 10,001 small loss token IDs are versioned. All held-out sources and selected
cases are excluded from calibration/tuning. Use separately identified data for
quantizer tuning, and do not change gates to accept a failing optimization.
Future execution of generated HumanEval code belongs in an explicit isolated
quality suite; preparing this contract does not execute dataset code.

Dataset provenance is the pinned [Salesforce WikiText repository](https://huggingface.co/datasets/Salesforce/wikitext),
[OpenAI HumanEval](https://github.com/openai/human-eval),
[TruthfulQA](https://github.com/sylinrl/TruthfulQA), and
[EleutherAI LAMBADA OpenAI](https://huggingface.co/datasets/EleutherAI/lambada_openai).
Source URLs and byte hashes are preserved in the manifest. Retain each source's
license and attribution when redistributing derived text.

On the Mac, prepare an ignored environment and explicit sources:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r tools/bench/requirements.lock
.venv/bin/python tools/bench/prepare.py --fetch
.venv/bin/python tools/bench/prepare.py --build artifacts/contract-candidate
.venv/bin/python tools/bench/prepare.py --check-quality
.venv/bin/python tools/bench/test_contract.py
```

The exporter refuses an existing contract directory and changed quality payloads.
The default verifier uses only committed files and the Python standard library:

```sh
python3 tools/bench/prepare.py
cargo test --locked --offline --test bench_contract
```

Fetch the approximately 988 MB source weights separately when running a baseline
or real-model suite:

```sh
.venv/bin/python tools/bench/prepare.py --fetch-weights
```

The scalar CPU [worker](../examples/cpu_bench.rs) and [adapter](../tools/bench/cpu.py)
run fixed teacher-forced token IDs, excluding tokenization, sampling,
detokenization and JSON transport. Each request completes its final distribution
before its token timestamp and leaves the last fixed output token pending. Cache
preparation is outside the measured request. Admission and bounded timestamp
storage are also outside timers. TTFT/total include request state changes;
prefill time covers only evaluation, decode time covers the 127 subsequent
distribution intervals, and total includes staging the final pending ID.

```sh
cargo build --locked --offline --release --example cpu_bench
.venv/bin/python tools/bench/cpu.py --workload short --output benchmarks/raw/cpu-short.json --profile benchmarks/raw/cpu-short.sample.txt
.venv/bin/python tools/bench/cpu.py --workload medium --output benchmarks/raw/cpu-medium.json
.venv/bin/python tools/bench/cpu.py --workload long --output benchmarks/raw/cpu-long.json
.venv/bin/python tools/bench/cpu.py --workload boundary --output benchmarks/raw/cpu-boundary.json
.venv/bin/python tools/bench/cpu.py --workload append --output benchmarks/raw/cpu-append.json
.venv/bin/python tools/bench/cpu.py --workload cached --output benchmarks/raw/cpu-cached.json
.venv/bin/python tools/bench/cpu.py --workload pending --output benchmarks/raw/cpu-pending.json
.venv/bin/python tools/bench/cpu.py --workload changed --output benchmarks/raw/cpu-changed.json
```

Run sequentially with other performance jobs stopped. The adapter always uses
three warmups and ten samples, checks 1,921 + 128 admission rejection before
warmup, and retains each measured raw row immediately in `.json.samples.jsonl`.
The final schema-validated record is written only after clean worker exit.
Existing record/profile/raw paths are rejected before model loading. An optional
macOS `sample` trace runs during the first warmup; its time is separate from
measured requests. The adapter terminates/reaps only its own worker on errors or
Ctrl-C, and a 24-hour watchdog bounds unexpected hangs. Interrupted raw rows
remain evidence, but are not complete benchmark records and cannot be silently
resumed or overwritten.

`prepare` in the worker evaluates a complete base history; `stage` appends a
pending ID without computation. Exact append measures 64 new inputs after 512
cached positions; fully cached input measures no prefill tokens; pending output
measures one prefill token; changed input replays all 512. The pending record
identifies the 512-token base plus the contract's pending ID 2077 (513 history
positions). Pre-load source hashing warms the file cache. Conversion is a nested
part of total load: byte decoding, finite checks and storage, excluding reads,
allocation and final model validation. It must not be added to total load again.

[The short worker probe](results/2026-10-10-cpu-profiler-probe.json) is functional
evidence for this harness, with eleven worker records and a 2,033,928,208-byte
own-child peak. It is not a workload benchmark. The full matrix, controlled
external comparison, sustained/application records and qualification are still
pending. Native BF16 MLX records remain on the application track until compatible
FP32 measurements are available.
