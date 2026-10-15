# Bounded wait for M6 long collections

The small-model paired quality collection passed at 4K and 8K. The 8K workflow
lasted about 20 minutes, so the remaining higher depths need a larger bounded IPC
allowance than the original collector's two-hour event-read wait. These elapsed
workflow observations are scheduling evidence, not completed GPU performance.
The first 16K attempt was intentionally interrupted through its owned caller;
it remains explicitly incomplete, with clean native cleanup, and will be retried.

A separate adapter preserves the original collector and all completed source
identities. It reuses the frozen worker, input admission, numerical validators,
phase collection and owned cleanup. The only execution change is an explicit
2/6/12-hour choice per blocking JSON event read, under the unchanged 24-hour
worker lifetime, output and event bounds. The default is six hours. Both adapter
and original collector bytes are captured and reasserted before launch and exit.
This does not provide a six-hour bound for an entire multi-event request.

Four new and ten existing collector tests pass; seven existing transport cleanup
and resource-bound tests pass. Two serial native selected-model probes pass six
raw events with natural zero exits and unchanged identities. The small probe
uses 21,600 seconds and the larger 43,200 seconds. Their four-input/two-output scope
checks configuration and cleanup, not long-context performance or support.

[Qualification and evidence](../../benchmarks/results/2026-10-15-m6-long-budget-qualification-v1.json)
bind the final adapter, tests, logs and both native captures. Long quality,
profiling, repeated latency, max-context generation and the KV decision remain
before M6 completion. Runtime numerical code and production defaults are unchanged.
