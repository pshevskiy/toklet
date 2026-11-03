# Frozen Llama 100-task contract

The shared manifest binds identical raw prompt IDs, reserves, stop IDs and
immutable model identities for the cached original and native MLX8 workers.
All 100 source cases remain in order: 40 HumanEval, 30 TruthfulQA and 30 LAMBADA.
The original grading payload and its answers/tests remain separately pinned;
execution inputs contain only ID, category, prompt IDs and output reserve.

The 70 instructed prompts replace the canonical source wrapper with the pinned
Llama plain renderer. Independent offline Jinja replay matches rendered bytes
and IDs. The 30 continuation prompts remain raw. Automatic special processing is
disabled; instructed prompts contain one BOS. Raw ByteLevel decoding skips
special tokens and applies no Transformers cleanup. Stops are 128001, 128008 and
128009; sampling chooses the lowest token ID on ties.

Reserves remain 256, 32 and 16 tokens respectively. Every case fits 566 positions;
HumanEval/32 requires exactly 310 prompt tokens plus 256 output tokens. Category
and integer reserve are checked per case ID, including swapped-case controls.
Three controls, preparation, independent replay and Astra review pass.
[Inputs](../benchmarks/llama3-quality-tasks-v1/manifest.json),
[evidence](../benchmarks/results/2026-11-03-llama3-task-inputs-v1.json).

No task answers were generated during preparation. Before grading, the original
worker must reproduce every visible greedy prediction and any terminal hidden
stop in a stateless teacher-forced replay. Both collections preserve emitted IDs
and literal decoded bytes. HumanEval code runs only in the qualified isolated
executor; unresolved infrastructure cannot count as a model failure. The frozen
relative pass-rate gate permits at most two fewer native passes across 100 cases.
