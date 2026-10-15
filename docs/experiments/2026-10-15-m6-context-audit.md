# Independent replay of M6 context collections

The saved-data auditor reissues each frozen collection plan to a reader of saved
phase events. It launches no numerical worker. Every expected request must match
exactly, every phase must be consumed, ordinary phases have exactly one event,
and comparison/profile phases retain their strict complete-event validators.
The raw JSON stream must equal the initial admission plus every phase event.

The audit recomputes quality rows, workload samples/summaries and profile
fractions through the qualified validators. It verifies exact completed state,
allocated resources, natural-exit metadata, native lifetime charges and the
selected compiled command. Complete consumed-file inventories and fixed pins
are checked independently, then consumed paths are deduplicated for the final endpoint pass.
The compiled admission and final compiled check also hash their fixed subset.

Summary and raw bindings use the SHA of the exact bounded buffer parsed for
replay, rather than a later read. A final endpoint check rejects replacements.
Astra found and the implementation fixes that separate-read race and the
possibility of an extra ordinary event appearing in both summary and raw data.
Mutation checks cover both boundaries and incomplete, altered or oversized data.

All seven new auditor checks and 28 existing context/helper/Transport checks
pass. Exact saved replay covers eight records, 568 events and 327 bindings;
Astra independently reproduced the audit byte for byte and accepted the final
code, checks and evidence without findings. This audit's pass flag
only covers its listed collections; it does not assert complete M6 coverage,
performance retention or context support.
