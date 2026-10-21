# M7 preserve frozen literal task identifiers

The full frozen100-task inventory uses literal IDs containing `/`, including
HumanEval/0. The preceding generic Swift task runner admitted only alphanumeric,
dot, underscore and hyphen labels, so it would reject this input before model
loading. This compatibility fix adds ASCII slash to the bounded literal-ID
whitelist. IDs remain1..80 ASCII bytes, unique and unmodified. They are used only
for uniqueness and output labels, never filesystem paths; artifact path rules
stay unchanged. The frozen benchmark labels are preserved without renaming,
encoding or reordering them.

The actual tiny runner now uses HumanEval/0 as its first case on CPU and Metal.
Admission, native response streaming, host parsing and reset/endpoint/lifetime
checks retain that exact string. The duplicate-ID rejection uses the same label;
space-containing IDs remain rejected. Prompts, output caps, all10 visible tiny
IDs, numerical behavior and quality thresholds are unchanged.

Twelve portable parser methods pass. The31-case native program and all27 owned
commands pass, including existing quality, measurement/256MiB hashing and both
complete iOS SDK app regressions, with clean exits/endpoints/cleanup. Four tiny
cases complete across CPU/Metal. A separate source-bound offline audit reads the
original SHA-admitted100-case task payload and frozen manifest, validates its
original policy and confirms every literal identifier fits the new bounds; all
100 contain slash. This is an identifier audit, not model-quality collection.
Ordinary parser tests need no downloaded task/model artifact.

The [qualification receipt](../../benchmarks/results/2026-10-21-m7-task-id-compat-qualification-v1.json)
binds180 paths,71 exact public copies, four changed code files, four exact parent
source mappings and six current runner source pins. The baseline runner proof
remains intact with the earlier source correspondence. No phone is launched or
signed by this task. The pending physical collector will require the reviewed
compatible runner proof before using frozen100-case inputs.

Astra accepted source and completed component evidence with no findings. Retain this literal
label compatibility fix. Full physical loss/tasks, grading, lifecycle/sustained/
supported-limit and base iPhone14 gates remain; M7.3 is open.
