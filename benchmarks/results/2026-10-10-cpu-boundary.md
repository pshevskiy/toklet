# Scalar Qwen capacity workload, 10 October 2026

The frozen scalar engine completes the real 1,920-input/128-output workload
within its 2,048-token capacity. All ten measured requests finish with 2,047
evaluated positions and one pending fixed output token. After repetitions, the
original driver evaluates that final pending token, checks 2,048 committed
positions and zero pending tokens, and rejects a further token before mutation.
This satisfies the real-model capacity criterion; M2.5 remains open for the
remaining scalar workload matrix and application measurements. Astra accepted
the saved-evidence task without findings after independently verifying all
states, summaries, identities, historical assertions and idle-window records.

The [result](2026-10-10-cpu-boundary.json) and
[complete measured states](2026-10-10-cpu-boundary.json.samples.jsonl) retain
three warmups and ten measured repetitions. The
[analysis](2026-10-10-cpu-boundary-analysis-v1.json) validates the schema,
reconstructs all summaries and binds exact input hashes. Its
[executed source](2026-10-10-cpu-boundary-analysis-v1.py.txt) and the
[original driver](2026-10-10-cpu-boundary-source-driver-v1.py.txt) are retained.
The final capacity response JSON was not retained separately; the completed
driver's checked assertions and capacity note are the evidence for that final
step. Every measured state's complete JSON is available.

| Metric | Median | Minimum | Maximum |
|---|---:|---:|---:|
| Warm TTFT, seconds | 653.578164 | 652.909296 | 656.742609 |
| Completed request, seconds | 749.021123 | 748.219608 | 752.132535 |

Median decode rate is 1.331183 steps/s over the 127 intervals after the first
token event; median prefill rate is 2.937675 tokens/s. FP32 weights and KV,
empty cache, chunk 128 and all prompt/output IDs are fixed. This is a controlled
teacher-forced workload: tokenization, sampling, detokenization and JSON
transport are excluded. Request time covers synchronous session evaluation
through the final projection and staging of the final pending ID. No formal
confidence interval, tail percentile, held-out quality or speed ranking is
claimed.

Logical weights are 1,976,131,072 bytes, allocated KV is 50,331,648 bytes and
scratch is 10,436,608 bytes. Conservative process admission is 2,172,348,800
bytes; maximum measured own-child physical charge is 2,043,774,992 bytes,
below the 6 GiB ceiling. The driver queries child peak after the final capacity
commit/rejection as well as after samples, so the retained process peak includes
the final step. Shared allocations are not counted twice; system memory
pressure and energy remain unavailable.

Hardware is Apple M1 Pro, MacBookPro18,3, 16 GiB RAM, macOS 26.6.2 (25G83),
Rust 1.90.0 and ordinary optimized release settings. The AC source observation
precedes repetitions; AC low-power configuration is queried at the end, when
it reports zero. This record does not establish a start-time low-power setting
or retain end battery/energy observations. Start/end thermal reports contain
no recorded thermal or performance warning.

The frozen executable SHA-256 is
`e3debabbae7db9e5801bc9bfba177275444ebe1d9849eb3ad0f2dff487ac8dea`.
Numerical sources were built at `996f8e7` and are unchanged through the dispatch
revision `337515e7780943b9f0f869f053808b543e9fbf26`. The original Python driver
SHA-256 is `6e9c30aeeefb28fb9f4c1403c537452926dcfcbf47aba394171245fb66fdbbad`;
the shared summary/footprint helper is
`911806d7254e905979217145b6e173ed1def09576ba3685b2c3c105176984b34`.
The model/tokenizer revision and full hashes remain pinned in the result.

Transport/orchestration holds outside active numerical requests permitted the
separate CPU comparison and checked primitive tasks. Active numerical workers
were never stopped. Those pauses mean this is not a contiguous sustained run.
The numerical boundary worker finished normally before the
[idle observation](2026-10-10-cpu-cli-check-window-idle.json) at 04:38:36 UTC;
CLI/application/SiLU checks then ran while the next-workload parent remained
held. The [completion record](2026-10-10-cpu-cli-check-window-completed.json)
verifies frozen binaries, current qualified sources and full local model assets
before resuming only orchestration at
`2026-10-10T04:55:10.704789+00:00`. All heavy checks preceded resumption.

Result, raw states and analysis SHA-256 are respectively
`2e2e88f18e9b1c2cbcc3339894e48ba37fa797bd0ec435e18ba1e6008dcc5f13`,
`991d9e76bdb5fd5a0af66c1138e6f23f80acb61e7fae30faffac0d3d8b1f31b1`
and `cf4b1047973584c645801b23c5041c68e068008445bf20db20255b393077b25e`.

Documentation links and whitespace checks apply to this saved-evidence task;
no engine code changes or new numerical execution are needed.
