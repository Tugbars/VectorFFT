# FFTW 3.3.10's real transforms: the mechanisms, as read

The design input for the IL real engine. Facts from the FFTW 3.3.10 source (the
tree at `~/fftw-3.3.10`, GPL: structure and math are described, no code is
reproduced) and from the plans FFTW's own planner printed on the i9-14900KF
(`gauntlet/fftw_plan_probe.c`, vcpkg's avx2 build, FFTW_MEASURE, out of place).
Section 7 states what follows for our engine.

## 1. The real problem and its layouts

- A real array is modelled as two interleaved arrays, the even samples and the
  odd samples, with stride 2 (`rdft/rdft.h`, `rdft/problem2.c`: r1 = r0 + is,
  is doubled). The split costs nothing; it is what lets a complex DFT run on
  the real array reinterpreted as pairs.
- The complex side is the unpacked CCE half-spectrum, n/2+1 interleaved pairs.
  In place needs the real row padded to 2·(n/2+1) reals and a batch distance
  covering the padded block (`rdft/rdft2-inplace-strides.c`).
- The problem hash keys on in-placeness and the alignment (mod 16) of all four
  pointers: wisdom is alignment-specific.
- c2r out of place destroys its input by default (the API sets DESTROY_INPUT);
  preserve-input c2r exists in 1D only through a buffered copy and is
  unplannable for rank >= 2.

## 2. The SIMD path for even N: a complex child plus one real stage

`rdft2-ct-dit/r` (`rdft/ct-hc2c.c`, `rdft/ct-hc2c-direct.c`), r even. Every
SIMD real codelet FFTW ships is of this family and only this family:
`hc2cfdftv_r` / `hc2cbdftv_r` at r in {2, 4, 6, 8, 10, 12, 16, 20, 32}
(`rdft/simd/codlist.mk`); the leaves `r2cf`, `r2cfII` and the halfcomplex
stages `hf`/`hb`/`hc2cf` are scalar in every build.

- Child: a complex DFT of length m = n/r over r/2 vectors, z_j[k] =
  x[kr + 2j] + i·x[kr + 2j + 1], input stride r reals, vector stride 2. At r = 2
  the child is the plain n/2-point DFT of the array read as pairs; at r = 4 the
  two sub-sequences are one 32-byte load (two lanes).
- The real stage, one in-place pass over the child's output: for each pair of
  bins (m', m − m'), the two-real-spectra untangle E_j = Z_j[m'] + conj Z_j[m − m'],
  O_j = i·(conj Z_j[m − m'] − Z_j[m']), the twiddles w_n^{l·m'} (the ×i folded
  into the complex multiply), a radix-r DFT over the E/O inputs, ×½, and the
  stores X[jm + m'] to the ascending side and conj(·) to the descending side.
  The radix-r butterfly is a genuine stage of the complex transform of n/2
  (radix r/2 of it) with the untangle folded in; at r = 2 it is the untangle
  alone.
- Cost of that stage: t1fv_r + 2r vector ops (r pair sums as FMAs, r halvings),
  the same memory traffic as the complex twiddle codelet of radix r.
- Vectorization runs along m' with VL = 2 complex (AVX double), the mirror side
  loaded at a negative stride so every lane holds an exact (m', m − m') pair.
  A count not divisible by VL is closed by one aliased 2-lane call at stride 0
  with the valid lane stored last (`apply_extra_iter`); this works only at
  VL = 2. Alignment: 16-byte pointers, even element strides, interleaved only.
- The self-mirror bins are separate tiny problems: bins 0, m, …, n/2 come from
  a scalar real DFT of size r (`r2cf_r`, the DC/Nyquist column), bin m/2 from
  the half-shifted real DFT of size r (`r2cfII_r`); the DC/Nyquist imaginary
  zeros are written by the solver afterwards.
- Twiddles: one vector per (lane block, l), no replication, computed once from
  an exact-trig generator and shared between plans.
- A buffered twin of every hc2c solver copies batches of round4(r) + 2 columns
  into a contiguous tile to satisfy alignment or bound the working set.
- r is a race: each codelet radix registers its own solver; the planner keeps
  the measured winner.

Passes, n = 4096 as planned here (r = 2): the child's leaf (32 × 64-point, out
of place, no twiddles), the child's twiddle stage (in place), the real stage
(in place). Three sweeps of the 2048-pair output, one read of the input. n = 256
(r = 4): one direct 2-wide 64-point child (out of place) and the real stage:
two sweeps.

## 3. What FFTW planned on this machine

| N | r2c | c2r |
|---|---|---|
| 256 | r = 4: `hc2cfdftv_4` over one direct 64-point 2-wide DFT | r = 2 over a radix-2 twiddle stage and 64-point leaves |
| 1024, 4096, 1000 | r = 2: `hc2cfdftv_2` over a twiddle stage (8 / 32 / 5·10) and 64-point leaves | the mirror |
| 65536 | r = 4 over a three-level child | r = 2 |
| 1215 (odd) | the scalar halfcomplex ladder `hf_3`, `hf_9`, `r2cf_15` | the mirror (`hb`, `r2cb`) |

At the sizes it chose r = 2, FFTW's r2c is a complex DFT of N/2 followed by a
standalone untangle pass: three passes for N = 4096, the same count as our zr2c
(leaf, mid, fold). Only where it chose r = 4 is the untangle fused with a
radix stage (two passes at N = 256).

## 4. The native halfcomplex ladder (odd N, and every non-SIMD size)

- Halfcomplex packing: re[k] at [k], im[k] at [n − k], DC at [0], Nyquist at
  [n/2] for even n. The CCE output is a separate scalar copy pass plus a
  malloc/free on every execute (`rdft/rdft2-rdft.c`).
- `rdft-ct-dit/r` with `hf_r`: n = r·m, the r2cf leaf runs out of place once,
  every `hf_r` stage in place once. A stage computes columns k = 1..(m−1)/2
  only and writes each output bin to its two halfcomplex slots; column 0 is a
  real r-point DFT, column m/2 (even m) the half-shifted one. `hf_r` is a full
  complex radix-r twiddle butterfly per column (the same flops as `t1_r`); the
  halving comes from the column range, not the codelet.
- N = 1215 = 3·3·9·15 as planned: 81 fifteen-point scalar leaves (9 per call
  at input stride 81), then in-place `hf_9`, `hf_3`, `hf_3` stages, then the
  CCE copy: five passes, about 21k flops of which 73% are the twiddle stages,
  all scalar.
- Primes without codelets: a real Rader through the Hartley transform (two
  real transforms of a padded {2,3,5}-smooth size >= 2n − 3 plus permutations
  and a malloc per execute), or the O(n²/2) direct real DFT below 173.

## 5. Rank >= 2, batches, c2r

- `rank-geq2-rdft2`: the r2c runs on the last (contiguous) dimension with every
  leading dimension and the batch folded into its vector tensor; a complex DFT
  over the leading dimensions then runs in place on the n/2+1 half-spectrum
  columns. No transpose, no inter-pass twiddle. c2r is the mirror: the complex
  stage first, in place on the input. 3D has two splits, only the first is
  tried below PATIENT.
- Batches are a vector dimension every leaf consumes as a scalar loop; the real
  stage loops over the batch in scalar too. Nothing in the real path vectorizes
  across a batch; the SIMD lanes are always along m'.
- `hc2cb` (c2r) reads bins 0..n/2 only and assumes conjugate symmetry.

## 5b. Rank 2 as planned and timed on this machine (2026-10-01)

The plans `plan_dft_r2c_2d` / `plan_dft_c2r_2d` print under FFTW_MEASURE out of
place, and each pass timed alone (`plan_many_dft_r2c` over the rows,
`plan_many_dft` over the columns in place: the two children sum to the 2D
time within 1%):

| Shape | Rows | Columns |
|---|---|---|
| N2 <= 32 (16x16 .. 4096x16, 32x32 .. 128x32) | `rdft2-r2hc-direct-N2-xN1`: ONE call of the scalar `r2cf_N2` codelet, the row loop inside it | N1 <= 128: `dft-direct-N1-x(hp1)`, ONE call of the straight-line `n1fv_N1` (n1fv_128 included) over every column, two columns per vector; N1 >= 512: `dft-buffered`, a few columns at a time copied into a contiguous buffer, a full 1D plan there, copied back |
| N2 >= 256 | `rdft2-vrank>=1` over the rows of `rdft2-ct-dit/r` (`hc2cfdftv_r` over the N2/2 child, r = 4 at 256, r = 2 at 512 and 1024) | the same column rule (`n1fv_128` x513 at 128x1024; `vrank>=1` of a ct plan at 512x256) |
| c2r | the mirror: `r2cb_N2` / `hc2cbdftv_r` | first, in place on the input (destroyed) |

The odd column count (hp1 = N2/2 + 1 is odd for every even N2) costs FFTW
nothing: `apply_extra_iter` (`dft/direct.c`) runs the codelet on hp1 - 1
columns and the last one as a two-lane call at vector stride 0. And no access
of FFTW's ever splits a cache line: the double AVX `LD` is a 128-bit load plus
`insertf128` of the next column's 128-bit load, `ST` the two halves stored
separately (`simd-support/simd-avx.h`), so every complex element is one
16-byte-aligned access whatever the row pitch.

Against ours (the IL 2D real tier, natural order, the default request; one
thread, same machine, the same day), r2c in ns, FFTW / ours:

| Shape | rows | columns | total |
|---|---|---|---|
| 128x16 | 769 / 1276 | 625 / 851 | 1404 / 2127 |
| 4096x16 | 31340 / 41220 | 73740 / 90780 | 105080 / 132000 |
| 128x32 | 2063 / 2312 | 1132 / 1944 | 3195 / 4256 |
| 16x1024 | 8147 / 9388 | 2706 / 5426 | 10853 / 14814 |
| 128x128 | 7502 / 7614 | 4726 / 7798 | 12228 / 15412 |
| 128x1024 | 69820 / 79267 | 40947 / 80173 | 110767 / 159440 |
| 512x256 | 62107 / 73760 | 83280 / 78647 | 145387 / 152407 |

The pass structure is the same (rows, then columns over hp1 in place, no
transpose). The differences are mechanical:

1. THE ODD PITCH SPLITS OUR LINES. Our column kinds load and store a column
   pair as one 32-byte `vmovupd`; at an odd pitch every other row starts 16
   bytes off a 32-byte boundary and a quarter of the accesses cross a 64-byte
   line. Inside L1 that is cheap (the radix-16 leaf runs at 0.80-0.83 of
   n1fv_16's time up to 65 columns); past L1 it is not: the radix-16 leaf over
   513 columns takes 6288 ns at pitch 513 and 2015 at pitch 514 (the same
   kernel, the same column count; n1fv_16: 2702). Radix 32: 7214 -> 6257 (FFTW
   6449); radix 64: 15879 -> 14019 (15663). Every 2D real plane has an odd
   pitch, so every column pass past L1 pays it: the 16xN2 column passes run at
   2x FFTW's.
2. THE ROW KERNEL AT SMALL N2. FFTW's scalar `r2cf_16` is 6.0 ns a row; our
   rn1 called directly is 8.7 (the complex n1 body on (x, 0) lanes at count 1,
   one complex per 128-bit vector), 10.0 through the transform-contiguous
   wrapper (one public execute per row). N2 = 32: 15.5-16.7 against 18 (the
   pair) / 25-29 (rn1); N2 = 64: 41 against 60. FFTW vectorizes nothing across
   rows; it wins on the real codelet's halved arithmetic.
3. N1 = 128 COLUMNS. One `n1fv_128` pass against our two-stage chain (8.16 /
   16.8) through the natural column pass and its scratch plane: 851 / 625 at
   128x16 (L1-resident, so not item 1), 1944 / 1132 at 128x32.
4. TALL COLUMNS. FFTW buffers a few columns into contiguous scratch at
   N1 >= 512; our natural request runs the multi-stage chain unbanded (the
   create races the row route and the band only for the scrambled class and
   single-stage chains).
5. WIDE ROWS. Per row 8-19% behind in the batch (512x256: 144 against 121 ns),
   level with FFTW at 128x128; the hc2c form with the fold in a radix-4 stage
   is ZTT-r's and the real pair's ground.

## 6. The planner

- MEASURE times each sub-problem in isolation on zeroed, cache-warm data
  (min of 8 repeats, doubling until 100 ticks, 2 s cap); ESTIMATE counts
  add + mul + fma + "other" (copies) and prunes at the first direct codelet.
- Solvers register in a fixed order; the halfcomplex route is registered before
  the hc2c codelet solvers but is marked UGLY out of place, so out-of-place r2c
  effectively never takes it.
- Non-PATIENT planning disables rank and vector-rank splits and the large-N
  fixed-radix path; the FMA-scheduled codelet bodies are compiled only under
  `--enable-fma` (the default x86 build uses the non-FMA bodies with FMA
  intrinsics inside the vector macros).

## 7. What follows for the IL real engine

1. Structurally, our zr2c equals FFTW's r = 2 plan: complex child of N/2 plus a
   standalone fold pass. Any loss to FFTW at those sizes is in the child or in
   the fold kernel, not in the pass count.
2. FFTW's only structural edge is the r >= 4 form, where the untangle rides in a
   radix stage. An engine that fuses the fold into its last stage at every size
   (form A: `t2h` on the pair routes, `tlfh` on ZTT) runs one pass fewer than
   FFTW at r = 2 sizes and equals it at r = 4 sizes; a real leaf (form B) reaches
   the same count with stock mids.
3. The fused stage's math is the `sym1` untangle followed by the twiddled
   radix-r butterfly and the mirror store; its cost over a complex twiddle
   stage of the same radix is 2r vector ops and no extra traffic.
4. Odd N: FFTW is scalar end to end, five passes at 1215, twiddle-dominated. A
   vectorized real odd leaf (the odd blocked emission with real inputs) on our
   mids competes against scalar code there.
5. Batches: FFTW gains nothing from K > 1 in the real stage; K separate K = 1
   transforms (our transform-contiguous route) is the same work.
6. Rank >= 2: our 2D real tier already has FFTW's shape (rows real, columns
   complex over hp1, no transpose); the natural-order restriction is ours.
7. Things to take: the untangle-in-the-butterfly math, the negative-stride
   mirror walk per lane, the ×i folded into the twiddle multiply, the
   self-mirror bins as tiny separate real DFTs, per-lane twiddle tables.
8. Things not to take: scalar leaves, the halfcomplex ladder and its copy pass,
   a malloc per execute, alignment-keyed wisdom, buffered copies as the answer
   to alignment.
