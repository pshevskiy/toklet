# Direct-gain Half RMS on Mac

B4d1 qualifies the existing borrowed Half RMS operation for the selected LFM
profile on Apple M1 Pro. Input and output are Half; stored BF16 gain widens
exactly to FP32. There is no+1 or intermediate gain narrowing. Squares,
Neumaier reduction, inverse and gain multiplication retain the existing FP32
shader arithmetic, followed by one Half output store.

The new profile admits exact epsilonFP32(1e-5), width64 with1..512 rows and
widths256/1024 with1..32 rows. These cover head64 including16 query heads×32
chunk rows, tiny hidden256 and real hidden1024. Other shape/epsilon combinations
remain rejected. The host stores the admitted epsilon bits in Plan and sends
those bits through existing shape[4]; widening admission alone would have kept
the old hardcoded1e-6 argument. Shader arithmetic and the old Gemma1e-6
width32/64/256/640,1..128-row profile remain unchanged.

The independent source was frozen before native outcomes. Seven cases contain
75,008 finite Half outputs in both MAIN serial-FP32 and WIDE FP64 profiles.
Six portable controls, exact four-file replay and Astra's complete saved-operand
reconstruction pass. MAIN remains fatal with the preregistered gate
`atol=max(1e-4,2*maxabs(main-wide)),rtol=1e-3`; WIDE is diagnostic. Exact semantic
witnesses supplement that gate: direct zero gain stays+0, gain2^-25 yields
Halfword0x0004, and signed minimum-subnormal inputs yield0x013c/0x813c at
1e-5 instead of the old1e-6 words0x03e8/0x83e8. These catch premature gain
narrowing and an incorrect epsilon ABI even when a numeric tolerance could pass.

The release GPU suite passes150,016 normal MAIN comparisons across strided and
tight encodes, plus the same WIDE diagnostics. Astra independently verifies every valid
Half word exactly; maximum absolute error is0. The layouts agree exactly, including active row511. Five invalid sites
exercise15 invalid/sticky/reset phases;640 valid recovery comparisons pass.
Invalid NaN/overflow output bits are retained as evidence, not an oracle. Raw
physical outputs/status and command completion/error precede assertions. Short
operands and aliases reject; padding/guards, input/scales and sticky status are
checked. Fourteen measured encodes report zero Rust allocations in release;
driver allocations are outside that counter.

All three unchanged legacy Gemma RMS GPU tests also pass, including108,544
normal source comparisons, status/recovery and exact subnormal controls. Cold
checks pass formatting, strict default/Metal all-target Clippy,450 default tests
and534 Metal-feature tests, plus the frozen optimized release test build. Counts
overlap; ignored hardware tests are run explicitly in the five-job GPU suite.

The first GPU attempt stops with SIGXFSZ before a raw RMS result under a
process-wide16MiB file-size cap. Its logs are116B/0B, and the exact incidental
write/startup phase is unknown. Both Astra investigations preserve that failed
receipt. A separately reviewed successor caps only stdout/stderr pipes, retaining
bounded prefixes and enforcing an absolute deadline through drain and process
exit. Four real owned CPU pipe controls pass before the same binary, fixtures,
gates and five tests are retried. No numerical policy changes accompany retry.

The successful five jobs close cleanly under300-second individual deadlines,
30-second cleanup and a6GiB observed-process guard. Native usage samples are
sparse—several jobs finish before a second sample—and do not establish complete
process peaks or whole-model memory fit. The unchanged release binary is bound
by SHA256, optimization level3 and disabled debug assertions. Astra's saved-raw
audit and [machine evidence](../benchmarks/results/2026-10-29-lfm2-half-rms-v1.json)
record the exact source, executable, controller and results.

This closes a standalone native leaf. It does not qualify MLX's intermediate
normalized-Half trajectory, resident convolution/GQA, a complete LFM model,
context range, app/text, quality, speed or iPhone. Those remain in the
[LFM2 plan](lfm2-hybrid-plan.md).
