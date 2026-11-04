# Native existing-generation-path task controller

The native controller reuses one existing Gemma generation session and the
standard `generation::generate` path at capacity567/chunk32. No sampler or
numerical kernel is duplicated. It authenticates the bounded frozen task payload
before model loading and validates every case ID/category/prompt/reserve/stop list.

Each run checks contiguous visible emission indices, hidden-stop exclusion,
complete stop-versus-cap accounting, exact sampled history and
`history == evaluated + pending`, with one pending sample. Every reset clears
history/evaluated/pending counts and hides logits. Weight owners, allocation
admission, model and input identities remain checked throughout.

The real pilot covers cases0/40/70. Visible counts51/1/3 respectively end with
hidden stop106; all state/accounting checks and independent record replay pass.
On M1 Pro16GiB, peak RSS is2,782,937,088bytes and physical footprint2,897,070,816bytes,
below6GiB. Model and replay workers exit cleanly; source/input/binary endpoint
seals pass. These outputs are not task grades.

fmt, strict default/Metal Clippy,550default/658Metal tests and release build pass.
Two cold validator controls cover pilot/remaining coverage, hidden stops, complete
caps, coherent oversampling, duplicate cases, bad pending state and nonfinite
timings. The initial test-only keyword-spacing error is retained. Additional
external Python cap/admission guards were checked after Cargo completed;
Rust/dependency inputs are unchanged and their checks are reused explicitly.
Astra approves code/results and reuse of the three pilot receipts.

Next collect the other97cases and join all100 in exact frozen order before any
grading. Original BF16 sampled trajectories independently require stateless
verification. The task-quality gate, prior fidelity failures and model/app/phone
adoption remain open. [Evidence](../benchmarks/results/2026-11-04-gemma-native-task-controller-v1.json).
