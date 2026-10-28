# Llama3 bounded consistency screen and B1 closure

B1's selected Mac engine-adapter scope now passes the final 1,024-target
same-artifact consistency screen, after its [short numerical/state vertical](llama3-real-model-contract.md).
The next architecture task is B2 Phi. This closes the bounded engine work;
production Chat and full adoption quality remain explicit B5/M8 requirements.

## Frozen workload and policy

Artifact: `mlx-community/Llama-3.2-1B-Instruct-8bit` at
`d48cdf0a4ea22d893b7c63a99d6a693e24822795`, complete payload SHA verified.
Use the existing pinned WikiText-2 raw test parquet from
[the original source manifest](../benchmarks/sources.json), preserving row order,
truthy-row newline joining and all whitespace. The 1,290,546-byte raw text hash
is `430983e5a711fbc4c6576edffecacf47ff23d58f62f3923d2386d798e0a794c9`.
The actual Llama tokenizer produces 288,936 IDs with special processing disabled;
this is a fresh encoding, not Qwen token IDs.

Freeze the first 1,025 IDs as [versioned input bytes](../benchmarks/llama3-loss1024-v1/tokens.u32le)
and [manifest](../benchmarks/llama3-loss1024-v1/manifest.json), SHA
`e4086f7621d9d781f5487db05f17df63e9693d0d2788d43a69c8108fc28aa493`.
Four 257-token blocks start at 0/256/512/768, reset position/KV per block and
score local rows 0–255 against the next ID. Global targets 1–1,024 are covered
once; row256 is evaluated and hashed but unscored. EOS never truncates this
teacher-forced workload. Independent full-table and 257-row-batch input replay
matches the bytes and complete-corpus token count.

Native profile remains MlxAffine/Compensated/Reference/Half, serial, capacity512.
Source main FP32 and wide FP64 use the same packed artifact and Half storage.
One full-block FP32 logits owner lives at a time; main/wide blocks are released
sequentially. The same existing Rust maximum/Kahan/exponential/NLL calculation
now has a narrow shared statistics helper; legacy quality worker constants and
numerical order remain unchanged.

The primary rule, frozen before model outcomes, is
`fsum(native_nll[i] - wide_nll[i])/1024 <= 0.05`. No statistics-coherence allowance
is added. Source Python uses `math.fsum`; native retains the existing compensated
sum. The reader checks finite target bits, max/mass/NLL formula, complete target
identity and reported means. An eight-ULP coherence check verifies reported
statistics; it is separate from the exact primary gate.

## Results on Apple M1 Pro

| Profile | Mean NLL | Paired native minus wide | Gate |
| --- | ---: | ---: | --- |
| Source main FP32/Half | 2.994686052266415 | diagnostic only | none |
| Source wide FP64/Half | 2.9947481041626225 | reference | none |
| Native chunk32 | 2.9947039894934804 | −0.00004411466914202308 | pass ≤0.05 |
| Native chunk128 | 2.9947039894934804 | −0.00004411466914202308 | pass ≤0.05 |

All 1,028 finite native FP32 rows per schedule match numerically, expressed by
row hashes that canonicalize signed zero. Raw hashes remain available separately;
no source/native row-hash or greedy-ID equality is required. The journal retains
2,048 scored native records and eight final unscored records. Source journals
retain 2,048 records across main/wide. These are authenticated statistics and row
hashes: full logits were not retained and are not independently rescored here.

Observed native peak resident/physical memory is 1,597,259,776 / 1,587,906,168
bytes; source peak resident size is 3,133,161,472 bytes and its lifetime physical
peak is 1,247,431,464 bytes. The native maximum diagnostic admission including
64 MiB application allowance is 1,694,265,532 bytes. All remain below 6 GiB.
This is verification memory, not a throughput measurement or iPhone budget.

Formatting, strict default/Metal Clippy, 384 default /265 Metal-library tests
(overlap), 15 targeted example tests and three each input/source/reader controls
pass. Initial Clippy found an unused reexport in an existing test include;
a type alias preserves its interface without changing math. Astra caught a
hash-then-second-read gap before outcomes; the exact consumed token buffer is
now bounded and hashed. Failing schedule rows are journaled before rejection.

## Completion boundary

B1 is complete for the selected native Mac512 engine-adapter scope: positional
fixtures, ordinary regressions, structural composition/accounting, short complete
source/native/cache checks, raw generation, measured memory, independent text
protocol and this consistency screen. The declared 131,072 model window remains
separate. Longest scored block here is257; prior512 state checks are not a full512
independent quality campaign.

Original **unquantized 10,000-target /100-task adoption gates** stay open in M8.
The pinned production renderer/codec, hidden-stop cancellation, app admission,
whole-app/offline journey, iPhone and performance are separate unfinished work.
Proceed to B2 architecture breadth as the active plan requires; no quality gate
is waived. [Compact evidence](../benchmarks/results/2026-10-28-llama3-loss1024-screen-v1.json),
[TODO](../TODO.md) and [improvements](improvements.md) record this boundary.
