# Exact W512 structural state witness

`expected.json` is4147990 bytes, SHA256
`ffeb7e3c375f061de09cc4f47b232dbadf98bcb168e37176a5107f411eaa8f29`.
Its six cases cover C511/512/513/1024/1025 with two heads,15 producers and35
routes. Boundary-cut schedules and two multirow wrap-straddling calls are both
included. The signed integer recipe is exact in FP32. This is structural state
and layout evidence, with no real-model numerical or device qualification.

The first five cases were frozen before production edits. Their original
receipt/source/compact fixture remain under
`artifacts/m8-gemma-long-capacity-oracle-v1`. Astra requested actual straddling
appends during review; the sixth case was added after draft constructor/test
edits and before any native build/test, with unchanged numerical/cache kernels.
Two v2 preparations are byte-identical:11323582 bytes, SHA256
`dace4070ddb1efe4f3ff75d07e0cbbf310c5c1bd86af16920ecad532606ef04d`.
The compact JSON preserves the same parsed values. See the
[task record](../../../docs/gemma-scalar-long-context.md).
