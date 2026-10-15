# M6 maximum-context application collector

The collector runs the actual interactive CLI with a resident tokenizer and
explicit Online attention at capacity32768/chunk128. Full collection uses the
frozen32640-token prompt and128 greedy outputs with configured EOS ignored.
Every published ID, original token/text/done/reset ordering, complete committed
and pending counts, completed timing and full application admission are checked.
The exact capacity-sized KV, history and logit allocations are asserted. The
history hash is derived from prompt/published IDs and honestly labelled; the CLI
does not expose an independent history hash.

Qualified binary/current compiled source, model, packed payloads, tokenizer and
contract arrays are reasserted before launch and after owned natural exit. Text
is independently decoded only after exit; the post-exit decoder's separate
native charge is also checked. Bounded transport and the reviewed context
worker's primary-error-preserving cleanup path are reused. One boundary run
qualifies memory/state/output semantics, without repeated application latency
or natural long-document quality claims.

Four new portable checks and two serial actual selected-model probes/ten raw
events pass. Each probe allocates the full32K capacity with tokenizer, while
using only four input IDs/two outputs. Admissions are1,791,516,840 bytes small
and3,122,841,472 larger. Sampled model/driver native joint lifetime charges are
1,617,987,816 and2,960,116,952 bytes. Natural exit, independent text, full source
endpoints and cleanup pass.

[Qualification and evidence bindings](../../benchmarks/results/2026-10-15-m6-application-collector-qualification-v1.json)
cover collector/probe behavior only. Actual32640+128 generation, long-context
quality/performance and M6 closure remain pending; production defaults stay.
