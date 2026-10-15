# M6 Online attention full task quality

Both selected Online configurations pass the unchanged100-task precision-loss
gate. Scores match their frozen FP32 references exactly, with zero percentage
point loss against the original two-point limit.

| Selected configuration | Online | Matched FP32 | Code | Prose | Continuation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small compensated Half | 2/100 | 2/100 | 0/40 | 1/30 | 1/30 |
| Larger frozen Q4 v2 | 11/100 | 11/100 | 0/40 | 10/30 | 1/30 |

Absolute scores remain low. All80 literal code programs fail with SyntaxError;
the original program construction and responses receive no repair. This
qualifies precision retention on the frozen suite. The independent
[10000-target loss gate](2026-10-15-m6-online-full-loss.md) also passes for both
selected representations.

Two serial model workers produce12221 raw events and200 original-cap responses
at capacity2048/chunk128, with actual Online/FullLogits/FP16 KV and the selected
projection/normalization modes. Full raw/state/stop/reset replay,101ordered
native-memory observations per model, natural exit and independent post-exit
tokenizer text pass. Complete application admissions are1,413,906,600 bytes
small and2,241,914,752 larger. Sampled worker/driver joint native lifetime
charges are1,245,890,768 and2,076,068,552 bytes, within6GiB. Native charge counters
may omit clean/shared pages; complete allocation admission remains separate.

The existing qualified arm64 Linux window grades only after fresh owned-model
process/group absence checks. All200 outcomes resolve, with no infrastructure
unresolved cases. Eighty exact literal program/completion bindings and
classifiers, plus120 original text outcomes, replay independently. Each window
removes its owned containers, confirms an empty running-container inventory,
restores the initially stopped VM and completes helper pipes. A fresh final
inventory confirms stopped/not-starting state. The retained14October filename
prefix is required by the qualified controller; these fresh runs' UTC records
are15October.

The read-only replay archives current grading/helper sources, complete
collection/control/runtime identities, matched baseline receipt pins and
byte-identical publication copies. Its first attempt assumed an absolute saved
invocation path and stopped; the corrected replay resolves relative paths
against the repository root. The failed attempt remains excluded, with native
evidence and gates preserved.

[Qualification and bindings](../../benchmarks/results/2026-10-15-m6-online-task-quality-qualification-v1.json)
cover full task quality only. Real long-context quality, profiles, repeated
latency, maximum-context application generation and the measured KV decision
remain before M6 completion. Production defaults stay unchanged.

Astra independently accepted all411 bindings, exact publication/archive copies,
matched baselines,12221 response events, all200 outcomes,80 literal-code and120
text replays, source endpoints and both stopped-VM restorations, with no findings.
