# Independent raw Half Conv3 source

Twelve prospective cases cover H5/256/1024, T1..32, nonzero old history,
explicit FP32 products and ordered adds then one Half store. Direct original
BF16 coefficients widen to FP32. MAIN is fatal with the unchanged per-array
`max(1e-4,2*maxabs(main-wide))+1e-3*abs(main)` gate; WIDE is diagnostic.
Source has46,571 output words per profile and7,238 exact raw-u candidate words.
Candidate is not publication; old history remains immutable. Exact tap/store/
signedzero witnesses and35 partition schedules precede native outcomes.
The [protocol](protocol.json) freezes arithmetic, failure domain and layout.
This fixture alone establishes no native/kernel/model or transaction support.
