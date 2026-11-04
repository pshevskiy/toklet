# Exact source-trajectory recovery

HumanEval/17's original cached trajectory remains rejected. At sample205,
absolute row411, cached BF16 scoring ties IDs1788 and58089 at21.75 and chooses
the lower ID1788. Fresh scoring gives1788=21.625 and58089=21.75. Both a412-token
prefix and the saved458-token trajectory produce the same fresh row hash.
Direct stock forward with a fresh DynamicCache reproduces all206cached rows
through that point exactly, excluding the controller wrapper as the cause.

Astra's deterministic recovery preserves the original stateless gate. At the
first mismatch, keep only the preceding verified IDs, take the exact fresh
choice, discard the entire remaining suffix and regenerate from the corrected
prefix. Never retain the old suffix after changing a token. Recheck the whole
proposal from the original prompt, including its terminal stop. A repeated
proposal or eight exhausted repairs leaves the case unverified and ungraded.
The rule uses no expected answers or task scores.

The real case needs one repair. It discards47samples, regenerates a new tail,
and all247resulting decisions—including stop106—pass the unchanged verifier.
The221.53s run peaks at5,023,367,168bytes RSS and4,977,266,960bytes physical
footprint, below6GiB. Independent source extent/read-chain replay, runtime/
input/model endpoints and clean owned exit pass. Three cold controls cover
complete suffix regeneration, stop/cap/pending state and cycle/repair-bound
rejection. Relevant Rust inputs are unchanged; Astra approves code and evidence.

Restart full source collection with this uniform rule, reusing the three
approved pilot cases and this recovered receipt. Recollect the other partial
records: the rejected worker exited before its final collection guards.
Native100-case collection is independently complete but remains ungraded.
No quality tolerance, task cap or numerical kernel changed; earlier fidelity
failures and the100-task quality gate remain open.
[Evidence](../benchmarks/results/2026-11-04-gemma-task-recovery-v1.json).
