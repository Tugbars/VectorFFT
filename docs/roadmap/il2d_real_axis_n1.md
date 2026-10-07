# The real axis on N1: a 2D real plan form for an even N1 and an odd N2

**Status (2026-10-07): BUILT, r2c, the serial form** (`il/rank2/il2d_real_axis.h`;
the decisions of §5 taken by the owner the same day). The c2r twin and the
threaded form are their own pieces.

## What it is

Today's 2D real walk puts the real transform on the rows (N2 real points
each) and a complex chain on the columns. At an odd N2 that row transform is
the odd door's engine per row (the real mono, the flat DIT, or the Bluestein),
N1 of them. At a prime N2 each of them costs nearly a full c2c(N2) (0.94 of it
at 1021), so the row pass pays N1 near-complex transforms for a real plane.

This form puts the real axis on N1 instead. Rows 2m and 2m+1 are packed into
one complex row, the column transform of length M = N1/2 runs on that half
plane, a fold across columns recovers the N1-point real column spectra, and the
complex c2c(N2) runs on M + 1 rows only. The row work halves; the fold is
elementwise across a row (SIMD over columns, no in-row mirror). FFTW has no
such form: its rank>=2 rdft2 always takes the real transform on the last
dimension.

Measured 2026-10-06/07 as a probe-side prototype (docs/research/il2d_real_levers,
arm W_n1real): at prime N2 with N1 <= 256 it ran 1.47-1.79x the promote route
and 1.23-1.53x the odd engines per row (the row plan built 2026-10-07); at
N2 = 15 it was 1.01-1.12x the row plan; at the other composite odd N2 it lost
(0.63-0.97x); at every even N2 it lost (0.50-0.98x). The prototype's column
transform was one kernel (n1c at M <= 64, the b448 leaf at 128) and its row
c2c one batched K = M + 1 call; a real engine runs the column chain at M.

## The r2c walk

For x[N1][N2] real, M = N1/2, w = e^{-2 pi i / N1}, hp1 = (N2 + 1) / 2:

1. **Pack.** W[m][n] = x[2m][n] + i x[2m+1][n], m < M: a complex M x N2
   plane at pitch P (P = N2, or the skewed N2 + 8 of the c2c tier's
   `VFFT_IL2D_CSK_PITCH` where 16 N2 mod 4096 is small -- a plan input).
2. **Columns.** The c2c column chain of length M down every column of W, in
   place: the chain the 2D c2c tier would serve at (M, N2), raced in this
   plan's role (a child store on the real row, never the c2c cell's row). The
   pass may run in the SCRAMBLED class: the fold below reads row pairs by
   index, so it un-scrambles for free through the chain's permutation, and
   the natural pass's second plane is not needed.
3. **Fold.** For each column n and k = 1..M/2, with Z = the column spectrum of
   W: E = (Z[k] + conj Z[M-k]) / 2, O = -i (Z[k] - conj Z[M-k]) / 2,
   A[k] = E + w^k O, A[M-k] = conj(E - w^k O); A[0] = Re Z[0] + Im Z[0],
   A[M] = Re Z[0] - Im Z[0]. A is the (M + 1) x N2 plane of the real column
   spectra's rows 0..M (the rest are their conjugate mirrors). One read of W,
   one write of A, SIMD across the row.
4. **Rows.** c2c(N2) on the M + 1 rows of A: a transform-contiguous batch at
   N2 x (M + 1), raced in this plan's role (its own child store on the real
   row, as `rp_` is for the standard walk's row child).
5. **Out.** CCE row k (k <= M) = bins 0..hp1-1 of the transformed A[k]; CCE
   row N1 - k (0 < k < M) = conj of bins (N2 - f) mod N2 of the same row. The
   first form writes the batch into a scratch and copies / mirrors; a store
   edge that writes the half rows and their mirrors directly is a later
   kernel.

The c2r twin reverses it (gather the full rows from the CCE half rows and
their mirrors, the inverse batch, the inverse fold, the inverse column chain,
unpack), its own piece after r2c (the owner's rule: r2c and c2r are separate).

## Where it lives

A whole-plan FORM of the 2D real IL plan (fft2d_create_il.h's real branch),
admitted at an even N1 and an odd N2 (the parities the walk needs; the race
decides the rest), raced against the standard walk on the whole transform at
create and banked on the real row with its own token. The arm's own children
(the column chain at M, the row batch at N2 x (M + 1), the pitch) ride on the
real row under their own prefixes, raced in role. Threading: the batch
threads itself (TC clones), the column pass through the column-MT machinery
at (M, N2), the pack / fold / out passes by row ranges -- after the serial
form is measured.

Memory: W (N1 N2 doubles), A and the row scratch ((M + 1) 2 N2 each): about
three real planes beside the caller's two.

## Gates

The owner's rule: the form is a candidate in the race, never a default. It
serves only where it wins its cell's race against the standard walk (with
the odd-N2 row plan in it), at the race's 3% hysteresis toward the standard
walk. Correctness: the arm's plane against the standard walk's at 1e-10, as
every row-plan arm is gated; replay bitwise with no race; the c2r twin
against pts * x.

## The decisions (owner, 2026-10-07; §5)

1. **Name.** Derived from what it does: the form is "the real axis on N1";
   the real row's token is `raxis=n1` (the form serves) / `raxis=n2` (the
   standard walk, the real axis on N2); the code is `il2d_rax_*`; the
   children ride on the real row as `rax_*` (the column chain) and `raxr_*`
   (the row batch). `VFFT_IL2D_RAXIS=n1|n2` pins.
2. **Admission.** Every cell the walk can build at: r2c, an even N1 >= 4, an
   odd N2, one thread. The race decides per cell (never a heuristic): the
   cost is one more arm at create.
3. **The column class.** Not tested in the research (the prototype's column
   transform was one natural kernel); built as the scrambled pass in place,
   the fold reading the row pairs through the chain's permutation. The
   natural pass with its second plane is strictly more traffic; it can be
   raced as a second form if ever wanted.
4. **The out pass.** Scratch + copy / mirror, as measured. The direct
   half-row store edge is a later kernel.
5. **c2r** as its own piece after r2c.

Related: `docs/roadmap/fft2d_real_il_design.md`, `docs/research/il2d_real_levers/REPORT.md`,
`il/rank2/il2d_real_plan.h` (the row plan the standard walk uses at an odd N2).
