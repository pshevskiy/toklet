# Session capacity source witness

The manifest and binary were frozen before the native context/cache/history
changes, using [the source producer](../../../tools/reference/gemma_session_capacity_reference.py)
and an [independent stdlib reconstruction](../../../tools/reference/check_gemma_session_capacity_reference.py).
The pre-target receipt is
`artifacts/m8-gemma-session-capacity-oracle-v1/checks.json` (SHA256
`fe04c16f7e6289947aa6672fdc8b1983f9a1394a701ee128a71c1daf54ad2b49`).
Its archived Rust sources describe the baseline, independently of later edits.

`manifest.json` is 11582 bytes, SHA256
`96f1aba1335e3efd7b72f4b36fa05b7925970ef480d5e4a373b8e90787e25dfc`.
`kv.bin` is 16896 bytes, SHA256
`3450e059297fc605d68e7bd957b2f0abd8676ec68ff85fa198beffc1099710b4`.
Values are exact dyadic FP32 data, with distinct producer, head, position and
feature identities. There are 35 consumer layers, 15 producers, two KV heads,
local/full head dimensions 2/3, original local storage512, global32 and maximum
positions512. Shared consumers resolve to producer13 or14 by attention kind.

The 19 schedules use capacities1/2/3/4/5/17/32 and unique chunk bounds
`1`, `min(3,C)`, `min(8,C)`, `C`. They contain98 steps and6860 wide/narrow
consumer views. This is source data and a schedule witness, not a saved native
cache result, numerical attention oracle, real checkpoint or larger-context
qualification. Native logical heads, segmented attention views, visible ranges
and ownership must be checked separately against these source bytes. The tiny
four-layer model uses its existing independent graph fixture for numerical rows.
