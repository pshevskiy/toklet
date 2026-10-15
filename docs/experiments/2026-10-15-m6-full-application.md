# M6 full maximum-context applications

Both selected Online configurations complete an actual tokenizer-owning CLI
generation with 32,640 admitted prompt IDs and 128 sampled/published outputs.
Both collectors pass, exit naturally with zero status and preserve consumed
identities. This closes the two full application runs, while long numerical
quality, lower-depth profiles, boundary timing and final M6 retention remain
pending.

The [application collector](../../tools/experiments/m6_application.py) is unchanged
from its accepted implementation/allocation probes. These two runs omit
`--check`. They use the reviewed CLI, explicit Online attention, FP16 KV,
FullLogits, chunk 128 and a capacity of 32,768. Small selects compensated Half;
larger selects the frozen group8 Q4 v2 compensated tiled configuration. Original
CPU/GPU/default attention and default 2K admission remain unchanged.

The request uses the exact frozen cyclic prompt IDs, with ordinary `generate`
and no teacher-forced continuation. It intentionally ignores EOS and publishes
all sampled IDs until the output limit. Input formatting/tokenization times are
zero because IDs are supplied directly. The CLI still loads and owns the real
tokenizer and performs incremental detokenization/publication; an independent
batch decoder checks finalized output after the owned model process exits.

| Model | Prompt / outputs | Final history / committed / pending | Raw events |
| --- | --- | --- | ---: |
| Small | 32,640 / 128 | 32,768 / 32,767 / 1 | 131 |
| Larger | 32,640 / 128 | 32,768 / 32,767 / 1 | 131 |

Both reach `output_limit`, with zero reused prompt tokens. Each raw stream
contains ready, 128 ordered token events, done and final reset. Every published
ID/text fragment, completed clock, count, stop reason and reset is checked.
History hashes are derived from the frozen prompt and published IDs; the CLI
exposes counts, not an independent hash of its internal history. No stronger
history-hash claim is made.

Both models produce the same 128 IDs and 546 UTF-8 bytes under this repeated
input, matching its frozen continuation; 89 distinct IDs occur. This observation
is not independent long-document quality, retrieval ability, or a task-score
comparison. The cyclic prompt already repeats the source passage. Exact output
IDs, finalized text and hashes are retained in the saved evidence.

| Model | Complete application admission bytes | Maximum joint native lifetime bytes | Independent post-exit decoder bytes |
| --- | ---: | ---: | ---: |
| Small | 1,791,516,840 | 1,615,907,072 | 78,037,616 |
| Larger | 3,122,841,472 | 2,960,706,752 | 77,202,008 |

Admissions include selected weights, allocated full-capacity KV, scratch,
history, logits, metadata and the 247,772,624-byte tokenizer/application
allowance. The KV allocations are 402,653,856/939,524,880 bytes including guards.
Model-phase native values conservatively sum owned worker/driver lifetime
charges at ready, generation completion and reset. The independent decoder is
a separate post-model phase, with its own admitted allowance and native charge.
Each admission/phase fits 6,442,450,944 bytes. Native counters may omit shared or
clean pages; complete numerical accounting is separate. Serial worker peaks are
not added together as if they were simultaneous allocations.

Power endpoints report AC attached, AC low-power mode off, unchanged policy,
and no recorded thermal/performance warning. These are endpoint observations,
not continuous thermal, energy or memory-pressure measurements. One application
per model establishes state/memory/output completion, without a latency ranking
or tail claim. Full prefill, decode, sampling and user-facing times remain in
the raw statistics for reproducibility.

The [qualification receipt](../../benchmarks/results/2026-10-15-m6-full-application-qualification-v1.json)
binds both full records, raw streams, stderr, the bounded saved-data replay and
exact published copies. Main replay covers all 262 events and 322 bindings,
including the exact reviewed executable/source/input inventory, requests,
derived response, memory sums, natural exits/cleanup and finalized text. No
runtime, collector, numerical gate or default changes for this evidence task.

Astra accepted source and evidence after byte-identical independent replay,
all 322 consumed bindings and eight published pins/copies, with exact commands,
output IDs/text, derived history, state, memory phases, cleanup and power wording
verified. All review execution and hashing finished before the next GPU work.
