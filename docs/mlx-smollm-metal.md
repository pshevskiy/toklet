# Real SmolLM2 Llama-family native execution

The immutable [SmolLM2-135M eight-bit snapshot](mlx-smollm-checkpoint.md)
executes through Toklet's resident Rust/Metal decoder on Apple M1 Pro. This is a
real LlamaForCausalLM-family checkpoint, not a Meta-branded Llama release.
Configuration admission requires the inspected bias-free SiLU/full-causal graph,
nine query heads mapped to three KV heads, hidden576/intermediate1536/head64,
30 layers, vocabulary49152, rotary theta100000 and RMS epsilon1e-5. Context is
bounded to512; the artifact's8192-position declaration is not general support.

Known export switches admit only pretraining_tp1, rope_interleaved false and
is_llama_config true. The exact inspected transformers.js_config object is
inert export metadata; unknown variants reject. Original Qwen admission and
Llama Reference-only routing remain unchanged. Explicit Llama Online rejects.
Resident original affine matrices and metadata avoid full-model expansion;
142,955,990 allocated weight bytes include Half normalization guards and one
tied embedding/output allocation. No shader or numerical tolerance changes.

The [independent CPU oracle](mlx-smollm-oracle.md) was committed before target
edits. All816 selected-stage snapshots/4,988,928 values pass its frozen gates
across six cases, chunk sizes1/3/full, positions0/5 and layers0/1/29. The maximum
absolute stage error is16, within the predeclared maximum allowance32; maximum
observed decode/scoring logit error is0.0589752197265625. Four full greedy
distributions and three scoring-prefix distributions pass; full-logit and
small-token readbacks select the eight expected IDs exactly. Pending rejection,
changed-prefix replay, reset, cached reuse and budget rejection pass.

A separate actual512-capacity/chunk128 witness commits128 positions, discards
and restores the384-token input tail, completes512 positions, rejects append
and replacement overflow while preserving exact history/logits, and matches a
fresh same-recipe full distribution exactly. This is committed partial-prefill
discard/resume, not an asynchronously interrupted GPU cancellation witness or
an independent long-context numerical oracle. Maximum logical session admission
is254,860,338 bytes; short diagnostics admit up to225,474,548 bytes. These are
numerical admissions, not whole-process peaks or phone memory limits.

Rust1.90 formatting, strict all-target Metal Clippy,142 CPU passes/eight ignored
and158 Metal-feature passes/80 ignored pass. All14 explicit release GPU cases
pass, including the three real-model cases, tiny Llama, rotary, Qwen MLX,
ordinary/compensated Half, CLI and Rust C-ABI callers. The first Clippy test-only
allocation failure and sandbox no-Metal-device trial remain retained. The same
GPU sources pass with hardware access. Protected CPU binaries remain unchanged.

The [receipt](../benchmarks/results/2026-10-22-smollm-real-metal-v1.json) binds
3,425 files and77 current source copies, plus exact failed-trial correspondence.
Retain selected real-model Reference execution. App/package and physical iPhone
execution are subsequent tasks. Online attention, held-out quality, chat
formatting, full-context numerical qualification and broader architectures
remain open; this does not close M8.3–M8.5.
