# LFM direct-gain Half RMS source

Independent stdlib MAIN serialFP32 Neumaier and WIDE FP64 profiles, one Half
store, direct original BF16 gain without+1, epsilonFP32(1e-5). Seven cases retain
75,008 finite outputs per profile, including512 head64 rows and hidden256/1024.
The [protocol](protocol.json) fixes MAIN fatal/WIDE diagnostic and the unchanged
`atol=max(1e-4,2*maxabs(main-wide)),rtol=1e-3` rule before target outcomes.
Raw operands, exact direct-gain/epsilon/subnormal witnesses and invalid controls
are authenticated by manifest and source audit. This fixture alone qualifies
no native operation, complete graph, real model, device or app.
