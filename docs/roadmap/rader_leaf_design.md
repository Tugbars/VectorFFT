# The Rader leaf — a prime factor above 47 inside the flat chain (design of record)

*Declaration of the stage-level Rader engine of the interleaved K=1 route: a prime radix
p above the flat DIT's kernel pool, served at stage 0 of the flat chain as a vectorized
Rader convolution over the stage's columns. DESIGN 2026-10-13, the order of work agreed
with the owner the same day: this complex leaf first, then the real odd-prime method
(§8), then the real leaf (§8). Roadmap item 2b of
[`1d_c2c_and_real_open_work.md`](1d_c2c_and_real_open_work.md). Not built.*

## 1. The gap and the contract

A length with a prime factor above 47 has no chain: the flat DIT's pool stops at 47, so
the K=1 planner serves such a length as a prime cell, Bluestein on the whole N at the
next power of two past 2N. That is 2,157 of the 4,095 lengths to 4,096 (every composite
with a factor of 53 or more) at 1.12x FFTW with cells at 0.41x (2101 = 11 x 191), 0.46x
(2118 = 2 x 3 x 353), 0.5x (1149 = 3 x 383). A prime N whose N minus 1 carries such a
factor loses Rader for the same reason (4099: 4098 = 2 x 3 x 683) and runs Bluestein at
1.1x MKL, the weakest class of the 1D table. The radix-29 to 47 kernels exist because
this engine did not; they stay until the race says otherwise.

The contract is the flat chain's: natural input, natural output, both directions, in
place legal, the same twiddle stages and the same tail kinds after the leaf. Nothing a
caller sees changes; the chain `p . R1 . R2 ...` becomes buildable for a prime p that no
kernel carries.

## 2. The form: the prime takes the leaf

The flat chain's stage 0 runs R0-point DFTs over N over R0 columns at stride N over R0,
no twiddles, the un-turned store: output digit l of column c keeps the slot l times D0
plus c. Read as a plane, the input is p rows of D0 complex (row l = x[l D0 .. l D0 + D0)),
and the output plane has the same shape with row l = X_l of every column. That plane is
exactly the geometry of the 2D tier's column pass (`il2d_cols.h`: a plane of N1 rows and
rn columns, the chain applied to every column, lanes across the columns), so a p-point
Rader over the columns is a vector operation with no turn and no per-column call:

    DC row:        X_0[c]       = sum_l x_l[c]
    gather:        a_q[c]       = x_{g^q}[c]                      q = 0 .. p-2
    convolve:      b_q[c]       = sum_m a_m[c] . w^{g^{-(q-m)}}    cyclic, length p-1
    scatter:       X_{g^{-q}}[c] = x_0[c] + b_q[c]

with g a primitive root mod p and w = exp(-2 pi i / p). The convolution is a forward
column chain of length p minus 1 over the columns, a row-wise multiply by the kernel
omega_q (the forward transform of w^{g^{-q}}, scaled by 1 over p minus 1), and the
backward column chain. The permutations, the kernel and its conjugate twin are the ones
`il/rank1/il_prime.h` builds for the whole-N Rader (`gpow`, `ginvpow`, `omf`, `omb`);
they are reused, not re-derived.

## 3. The execute: one pass over the data

The leaf runs over column windows [c0, c1) the way the column Bluestein does
(`_il2d_blu_cols_range`): every step is column-local, the window's scratch is p minus 1
rows of wc complex, and the whole convolution stays in cache:

1. the window's DC row and `x_0` row from the input plane; the gather copies rows g^q
   of the window into scratch rows q (a strided row copy, wc complex per row);
2. `_il2d_col_pass_range(scr, scr, p-1, wc, 0, wc, nst, Rs, Ls, ff, tf, 0)`: the forward
   chain, in place in the scratch;
3. the row multiply by omega (`_il2d_row_cmul` per row, the kernel broadcast over the
   window);
4. the backward chain, `fb`, `tb`, reverse;
5. the scatter: plane row g^{-q} of the window gets `x_0` plus scratch row q; plane row
   0 gets the DC row.

The input is read once and the output plane written once; the scratch never leaves L1
or L2 at the widths the 2D tier races. So the Rader leaf costs the same memory passes
as a direct leaf, and its extra cost is in-cache arithmetic: about 2 (p-1) log(p-1)
complex operations per column against the direct kernel's p squared over 2. At p = 191
that is roughly 2,000 against 18,000. Against today's whole-N Bluestein at 2101 (two
transforms at 4096 plus the modulation, every one a multi-pass engine over twice the
data), the leaf plus one radix-11 stage is three to four times less traffic.

The window width is a form the race decides, as the 2D tier's column widths are; the
first build takes the Bluestein pipeline's width rule and the race follows when the
chain race admits the leaf (§6).

## 4. The inner chain at p minus 1

The column chain of length p minus 1 is the 2D tier's own: `_il2d_resolve` validates a
radix list against the column kinds, `_il2d_build_tables` builds its tables,
`_il2d_apply_forms` applies the forms, and the column pool closes a 2 times prime length
with the lone-2 rule (26, 22, 14, 12, 10, 6, 2 as the closing stage). So p minus 1 chains
whenever its own largest prime factor is in the pool: 190 = 19 . 5 . 2, 352 = 16 . 22,
382 = 191 . 2 does not (191 is not a column kind), 106 = 53 . 2 does not.

The chain is a child raced in role (owner, 2026-10-02): the flat plan's chain race
times the whole forward with each candidate inner chain as a heat, the way the 2D tier's
`_il2d_race_heat` races column chains over (N1, N2). The leaf's inner is keyed by
(p minus 1, D0): the column count matters to the width and the forms. The verdict banks
on the K=1 row beside the flat chain (§6); no 2D row is read or written by a 1D plan.

A p whose p minus 1 has its own prime factor above 47 is the recursion case (107, 167,
263, ...). The first build declines it: the chain is not buildable, the arm drops out of
the race, the prime cell keeps serving. A second level, the inner's own Rader leaf, is
the owner's call (§10).

## 5. Directions and order classes

- Forward: §3. Backward, the natural class's conjugate pipeline: the same leaf with the
  conjugate kernel `omb` and the permutations swapped (gather by g^{-q}, scatter by g^q),
  as `_ilprime_exec_rader` does for the whole-N case; the chain's later stages run their
  backward kernels and conjugate tables as today.
- The SCRAMBLED class runs the transposed pipeline with the leaf's transpose last. The
  first build does not provide a transposed Rader leaf: a chain with a Rader leaf has
  `scr_ok = 0`, the natural class alone carries it, and the scrambled class's race keeps
  its other arms. The transposed leaf is a later piece if the scrambled class ever wants
  one.
- In place: the leaf reads the input plane and writes the staging plane, as the direct
  leaf does; in place is unchanged.
- Threads (`il_flatdit_mt.h`): the leaf stage splits by columns already; the window is
  the unit, so a thread's column range is a run of windows with its own scratch. MT
  equals ST bitwise by construction (the same windows, the same kernels).

## 6. Admission, planning, wisdom

- `vfft_ilfd_create_chain` builds the Rader leaf when R[0] is a prime above the pool and
  p minus 1 resolves as a column chain; otherwise the chain fails to build as today. The
  chain descriptor `il_flat=p.R1.R2...` already carries the leaf radix as its first
  number, so the chain itself needs no new token: a first radix above 47 names the Rader
  leaf.
- The K=1 planner's flat heats (`dp_planner_il.h`, `_il_dp_flat_pool`) gain the leaf
  candidates: for an N with one prime factor p above 47, every chain of N over p from the
  pool, prefixed by p. They race against the prime cell (Bluestein) and every other arm
  as ordinary flat chains; the verdict is banked and replayed as today. No heuristic
  anywhere: a length with the leaf available and a faster Bluestein keeps Bluestein.
- Policy (`policy_il.h`): a length with exactly one prime factor above 47 whose p minus 1
  chains joins the FLAT family's pool beside PRIME; `vfft_policy_prime_cell` stops
  claiming it outright. The policy gate's mirrors follow.
- The inner chain's verdict (its radices, forms and width) banks on the K=1 row as new
  tokens. Their names are the owner's (§10).
- The prime N whose N minus 1 is a leaf length (4099, 2131, ...): the whole-N Rader's
  inner pool (`_ilprime_inner_make`) asks the flat chain for N minus 1, and the flat chain
  now answers with the leaf. Rader at those primes then races Bluestein as at every
  other prime. Nothing else changes there.

## 7. Verification

- The flat gate (`flatdit_gate.c`) gains cells with a prime factor above 47 (524 =
  4 x 131, 1149, 2101, 2118, 2276 = 4 x 569, 2575 = 25 x 103) and one prime whose
  N minus 1 is a leaf length (4099): forward against the long-double reference,
  backward roundtrip, in place, replay bit-identity, MT equal to ST.
- The probe for the engine alone before the planner admits it: the leaf against the
  reference at a few (p, D0), and its last-stage time against the whole-N Bluestein on
  the same cell, the canonical protocol.
- One sweep of the class against FFTW after the planner admits it (the 2 to 4,096
  gauntlet restricted to the class), per the racing budget: once, saved.

## 8. The real transforms, later

- An even real N already inherits the complex child's engines through the CCE
  reinterpretation (`zr2c`), so N = 2p gains the leaf's effect for free once the complex
  prime cell does.
- The real odd prime method: Rader on real input is a real convolution of length N minus
  1, an even length, so its inner is an r2c and c2r pair at N minus 1, itself the CCE
  reinterpretation over a complex child at half that. A second arm beside the real
  Bluestein in the odd builder's sweep (`odd_build.h`), raced the same way. Second.
- The real odd composite with a factor above 47 needs the real flat DIT's leaf (`r1c`)
  replaced by a real Rader leaf. Third.

## 9. Expected outcome

The class's floor moves from 0.41x to above parity where the leaf serves: the leaf plus
the smooth stages cost about one pass over N against the whole-N Bluestein's several
over 2N or more. The geomean of the class (1.12x) should move toward the smooth
composites' 2.15x only where the cofactor is large enough to carry several stages;
cofactors 2 and 3 are already fine because N is small. The exact numbers come from the
race, not from here.

## 10. Decisions

Agreed 2026-10-13: form B (the leaf in the flat chain) over the four-step's 2D view;
the order complex leaf, real odd-prime method, real leaf.

Open, the owner's:
1. Leaf only (this design), or also a mid-chain Rader stage with pre-twiddles for a
   length with two prime factors above 47. Below 2^20 such lengths are rare (p q with
   both above 47 and the cofactor smooth); the leaf covers every length with one.
2. The recursion when p minus 1 has its own factor above 47: decline (this design), or
   one level of the inner's own leaf.
3. The names of the K=1 row tokens that carry the inner chain's verdict.

## 11. Phases

1. The engine: the leaf's create (permutations, kernel, the inner chain resolved from a
   given radix list and forms, scratch) and execute (both directions, column windows,
   the MT column split), behind `vfft_ilfd_create_chain`; the probe of §7.
2. The planner: the leaf candidates in the flat heats, the inner raced in role and
   banked with the owner's tokens, the policy and the policy gate, the flat gate's cells.
3. The class sweep against FFTW, once; the results page.
4. The real odd-prime method (§8); then the real leaf.
