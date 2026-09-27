# `core/transforms/trig/` — DCT-I..IV, DST-I..III, DHT

Real in, real out, on a real-FFT inner. **1D and SPLIT only** — a 2D or
interleaved trig request is refused loudly, and both refusals are pinned as
golden cells (`REFUSE.dct2.2d`, `REFUSE.dct2.interleaved`).

## 🔴 How a trig plan is keyed

A trig INNER c2c is keyed by **(owning transform, OUTER N, K)** — never as an
ordinary c2c of the inner size. Keying it by the inner size would collide with a
genuine c2c request for that size and hand one family the other's verdict. The
inner size derives from `vw2_stride_trig_inner_n`.

## 🔴 This family has no banked wisdom

The store holds 539 cells and **zero** are trig. Consequences: the fingerprint
replay has almost nothing to replay here, and several trig configs RACE at
create — every rigor level measures (`VFFT_ESTIMATE` is unimplemented), so with
nothing to replay the clock picks the plan.

Output coverage is therefore `build_tuned/trig_digest_probe.c` +
`trig_capture.py`: digests over 14 cells against a warmed scratch fixture.
That is a REGRESSION check — it proves the output did not change, not that it
is correct. The naive O(N²) reference that would prove correctness is still
absent, deliberately: the plane-role contract is not stated plainly enough in
`include/vfft.h` to encode without guessing, and a wrong expectation baked into
a baseline is worse than a missing one.

| file | role |
|---|---|
| `dct.h` | DCT-II / DCT-III (+ the inner r2c) |
| `dct1.h` | DCT-I / DST-I (boundary r2c) |
| `dct4.h` | DCT-IV (inner c2c of N/2) |
| `dst.h` | DST-II / DST-III (wrap DCT-II) |
| `dht.h` | the discrete Hartley transform |
| `trig_create.h` | the trig CREATE tier and its builder cluster (`_build_trig` and the three it calls) |
