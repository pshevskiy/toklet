# Original BF16 short graph preparation

`plan.json` is the exact pre-execution plan: 707199 bytes, SHA
`04be1a0f1134441aabacb14de6b688fe4c81e97a01cf5d0919e2d0cd37397dd3`.
It fixes both token sequences/contracts, all 2852 state descriptors, the 512 MiB
lifetime ledger and complete original-file read audit before graph execution.

`capture.json` is the exact first successful cold capture manifest: 1056305
bytes, SHA `66668be28232b7073ee8ed202b7330463b92e42331bb400cdf3a7705f36679cb`.
The complete arrays remain in ignored
`artifacts/m8-gemma-bf16-graph-checks-v1/graph-a/`:

- `original.bin`: 103403520 bytes, SHA
  `0e175fccc78fd76e02234a35fe72d0afafd5ad61a05a231047ed5e663ec67793`.
- `future.bin`: 103403520 bytes, SHA
  `8e2e2b33918c6a9e78f291f2db5b957b36601b7d743304d4808f17bca43a4b88`.

The independent cold repeat in `graph-b/` matches both raw files exactly.
The first three causal positions pass all 1426 comparisons; 24 selected scalar
boundaries pass. Each replay still records eleven failed cross-precision values
across seven sequence/state comparisons. This fixture prepares references; it
does not establish independent full-graph agreement, upstream BF16 arithmetic,
quality, whole-model fit, app eligibility or the model's full input/output window.
Full-array replay requires the ignored binaries and official source bytes.

See the [method and results](../../../docs/gemma-bf16-text-reference.md) and
[public record](../../../benchmarks/results/2026-10-26-gemma-bf16-text-reference-v1.json).
