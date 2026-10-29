# Original LFM2 payload preparation

B4e3a adds strict original-byte storage and direct-weight preparation for the
selected LFM2-350M layout. Seven synthetic controls and all six repository
checks and the separately reviewed actual-file CPU probe pass. Astra's saved
audit independently verifies original bytes, observations, budgets and ten closed
process groups; exact evidence is in the
[machine-readable result](../benchmarks/results/2026-10-29-lfm2-original-loader-v1.json).
This module does not construct a backend, model/session, tokenizer or app profile.

The loader authenticates caller-supplied config/index identities, the exact
header and complete container stream. It requires the existing strict plan,
exact byte count, EOF and whole-file digest before returning a store. All 93
original Q4/group64/BF16 matrix parents borrow one retained payload allocation;
tied embedding/head storage is retained once. Finite coefficient/reconstruction
checks preserve the [selected source profile](lfm2-real-reference.md), including
rejection of an overflowing FP32 nibble product even when bias could cancel it.

Forty-five direct BF16 gains and ten `[channels,3,1]` convolution tensors are
decoded into small typed vectors with explicit little-endian reads. No alignment
beyond `Vec<u8>` is assumed. Preparation resolves all sixteen routes, six KV
owners and ten history owners once; name/map lookups remain outside numerical
loops. Raw direct gains have no unit offset or intermediate activation cast.

Load admission counts actual payload/header capacities, a 64 MiB metadata
allowance, stream reserve and explicit caller overlap. The header is discarded
before return. Preparation counts actual typed-vector and descriptor capacities
alongside the retained payload; an in-memory reader/container remains an explicit
caller owner. The complete consumed file, retained payload, typed copies and
constructor allowance are distinct quantities, rather than one model-size guess.

Prepared Config capacities 1–512 are a data contract. The metadata plan still
reports declared 128000 with no implemented or qualified backend capacity, and
the existing tiny scalar/resident constructors remain unchanged. Native real
composition, session/recovery, numerical conformance and app admission remain
subsequent tasks.

The actual 199,527,820-byte file passes two sequential successful File loads.
Independent source bytes verify 279 matrix observations/558 endpoint values and
all 65,280 prepared BF16 words at capacities 1 and 512 (130,560 comparisons).
Sixteen routes, six attention owners and ten histories resolve correctly.
Matrix values are endpoint samples; whole-file authentication is separate,
rather than an all-weight numerical reconstruction claim.

Observed payload/header capacities are 199,491,072/36,740 bytes. With 274,489 bytes
of explicit caller residency, load admission is 266,976,701 bytes and preparation
is 267,016,841 bytes. Typed BF16 storage is 130,560 bytes; descriptor storage is
11,856 bytes, at capacities 16/6/10. Exact-budget admission passes and one-byte-
under preparation/load reject; load-under stops after only 36,748 bytes, before
reading payload. No duplicate full payload or expanded FP32 model is retained.

The owned CPU probe samples 21 process observations, with maximum observed
resident 203,522,048 and lifetime physical footprint 202,818,040 bytes. These are
sparse observations, not complete process or aggregate compiler peaks. All four
jobs use bounded 600 s/6 GiB/1 MiB-per-stream supervision and clean teardown. The
library, dependency archives and 852,160-byte executed probe are pinned. The
original root CLI/benchmark binaries remain unchanged. Seven synthetic controls,
formatting, both strict Clippy configurations, default 457/Metal 552 tests and
release test compilation pass; 20/181 tests remain ignored and counts overlap.
