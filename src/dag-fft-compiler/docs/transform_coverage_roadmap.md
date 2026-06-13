# Transform coverage: what we have, what's missing, what it costs

Snapshot at notebook section 54. Answer to "do we support all
DSP/trig/FFT transforms?": nearly all KINDS; the remaining gaps are
three small kinds, a handful of high-demand PRODUCT layers over
existing machinery, and two strategic AXES.

## Have (gated, in-tree)

- **Complex DFT (c2c)**: in-place split, OOP, strided; mixed radix
  (2..64 incl. 5,7,11,13,17,19,25); prime N via Rader (N-1 smooth)
  and Bluestein (any N), wisdom-tuned (M,B); forward/backward,
  DIT/DIF, log3/t1s/flat variants; wisdom + multi-tier planner;
  threads. Beats FFTW PATIENT 1.2-3.1x at primes ON THEIR HOME
  layout; parity at composite home; ~3x on batched split.
- **Real transforms**: r2c + c2r (even N), rdft (real-input DFT
  codelets), hc2hc/hc2c cascade primitives.
- **Trig family (r2r)**: DCT-II/III/IV, DST-II/III, DHT — production
  plan shells (core/dct.h, dst.h, dct4.h, dht.h, Makhoul/Lee/R2C
  reductions, MT) AND generated fused codelets N{8,16,32,64} x
  {avx2, avx512}, lean 3-arg ABI, constant hoisting, coverage
  citizens (codelets/trig, 48 files), all numerics/property gated.
  Beats FFTW r2r 2.9-19x (their r2r doesn't batch-vectorize at all).
- **Any K >= 1**: width cascade (8 -> 4 -> scalar), bit-exact
  (section 53); scalar ISA restored to the generator.

In FFTW r2r-kind terms: REDFT10, REDFT01, REDFT11, RODFT10, RODFT01,
DHT covered.

## Missing kinds (tier K) — COMPLETE (sections 55-58)

All four items closed: DST-IV (s55), DCT-I/DST-I codelets (s56),
odd-N r2c/c2r (s57), DCT-I/DST-I arbitrary-N shells + nine-kind
bench sweep (s58). Wins vs FFTW PATIENT in every measured cell;
ledger #17-#19 record the honest misses along the way. Original
plan retained below for the record.

## Original execution plan (section 55)

- **DST-IV (RODFT11)** — DONE (section 55). Exact reduction
  DST-IV[x][k] = DCT-IV[(-1)^n x][N-1-k]; algsimp folds the signs,
  generated codelet costs exactly a DCT-IV. Gates: formula 4.6e-14
  (N=8) / 1.4e-13 (N=32), involution 3.0e-14 / 5.0e-14. Seventh
  coverage kind; 56 trig codelets. Pending: RODFT11 row in
  bench_trig_vs_fftw.

- **DCT-I (REDFT00) / DST-I (RODFT00)** — Phase 2 codelets DONE
  (section 56): direct symmetric-extension expanders, gated at
  N=5/9/17/33 and 3/7/15/31, all ISAs. Cost ~0.64-0.73 of the
  embedding rdft (ledger #17). Phase 1 production shells (arbitrary
  N via pad-embedding) still pending. Original plan below.
  Phase 1 (shells, ~1 day): pad-embedding through existing r2c.
  DCT-I_N: even-extend into scratch of M = 2(N-1) (always even),
  r2c, Y[k] = Re(Z[k]); the identically-zero Im is a free internal
  check. DST-I_N: odd-extend into M = 2(N+1) with forced zeros at 0
  and N+1, Y[k] = -Im(Z[k+1]). Shell pattern = dst.h's. Cost 2x
  optimal but wisdom-tuned inner; primes via Bluestein give full
  generality. Gates: vs FFTW REDFT00/RODFT00 elementwise; DCT-I
  involution scale 2(N-1); DST-I scale 2(N+1).
  Phase 2 (fused codelets, 2-3 days, research-flavored): math-layer
  expanders build the symmetric-extended rdft DIRECTLY and let
  CSE/algsimp exploit the symmetry (each x[n] appears twice).
  PRE-REGISTERED: algsimp recovers >= 70% of the 2x symmetry saving.
  Natural sizes N = 2^k + 1 (M = 2^(k+1)) align with Chebyshev
  grids, exactly this kind's user base.

- **Odd-N r2c/c2r** — Phase 1 DONE (section 57): embedding shell in
  core/r2c.h, gated at N=8/9/15/21/105/17/31/47 incl. Rader and
  Bluestein primes. Deficit prediction falsified (#18): 1.56x FASTER
  than FFTW PATIENT at N=105 K=256. Phase 2 demoted. Original plan:
  Phase 1 (unblock, ~half day): shell route via the c2c/rdft path
  with im = 0 and Hermitian-half output write. Correct, ~2x cost,
  API parity with FFTW (prime real N falls out free via the
  existing prime c2c machinery).
  Phase 2 (optimal odd real-split algorithms): parked, gated on a
  user appearing.

SEQUENCE: DST-IV done -> integrate Tugbars's 2D FFT when it arrives
(row-column over batched 1D + the membrane transposes; bench it)
-> DCT-I/DST-I phase 1 -> odd-N phase 1 -> DCT-I/DST-I phase 2 with
the symmetry-CSE measurement. All of it behind the standing global
priority: i9 stride-wisdom regeneration (+50%).

## Missing product layers (tier P — thin layers over existing
machinery, outsized demand per effort)

- **MDCT / IMDCT** (50%-lapped, windowed): pure DCT-IV layer. Days.
  The audio-codec workhorse (AAC/Opus/Vorbis class); pairs with the
  19x dct4 result.
- **FFT convolution / correlation** (overlap-save/add, batched): the
  Bluestein engine already does internal circular convolution;
  exposing a user-facing API is packaging. HFT kernels, imaging,
  matched filters.
- **Hilbert transform / analytic signal**: one spectral mask over
  r2c/c2c. Hours. Envelope/phase extraction; EEMD/HHT pipelines.
- **Chirp-Z / zoom FFT (CZT)**: Bluestein generalized to arbitrary
  spiral contours. Days. Radar zoom, narrowband inspection; a
  differentiator (MKL lacks it; scipy only recently added it).
- **Goertzel / sliding DFT**: O(1)-per-tick single-bin streaming
  estimators. Hours. Completes the K=1 streaming story in
  docs/layout_market_and_membrane.md — the honest answer for the
  hot-path single-stream user.
- **Polyphase filterbank channelizer (PFB)**: FIR + batched FFT.
  Radio astronomy / SDR staple; our shape is its native shape.

## Missing axes (tier A — strategic, not kinds)

- **2D / 3D plan shells**: row-column passes ARE batched 1D (our
  engine's native food) + the membrane transposes between axes. This
  is the lithography / weather / PME product, and it is plan-level
  work, not kernel work. STATUS: a 2D FFT exists in Tugbars's
  codebase; integration pending delivery.
- **float32**: doubles the lanes (16/zmm), the ML/SDR currency.
  Generator-wide ISA axis (Isa records, cnum precision, twiddle
  tables); the largest single roadmap item.
- **Fused membrane codelets** (section 53b/c): the layout-crossing
  product; gated on the v1 tile measurement.

## Suggested order

1. Tier-P quick wins in one sweep: DST-IV + Hilbert + Goertzel/sliding
   + MDCT (about a week total, four product announcements).
2. CZT + convolution API (Bluestein packaging).
3. DCT-I/DST-I (kind-completeness vs FFTW).
4. 2D plan shell (opens the largest markets).
5. float32 (the long pole; schedule deliberately).

- D2 natural-split terminator: SHIPPED (section 69). Native rfft P2 functionally complete; packed and natural executors both gated.
