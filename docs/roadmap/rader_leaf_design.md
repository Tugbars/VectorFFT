# The Rader leaf — a prime factor above 47 inside the flat chain (design of record)

*Declaration of the stage-level Rader engine of the interleaved K=1 route: a prime radix
p above the flat DIT's kernel pool, served at stage 0 of the flat chain as a vectorized
Rader convolution over the stage's columns. DESIGN 2026-10-13, the order of work agreed
with the owner the same day: this complex leaf first, then the real odd-prime method
(§8), then the real leaf (§8). Roadmap item 2b of
[`1d_c2c_and_real_open_work.md`](1d_c2c_and_real_open_work.md). PHASE 1 BUILT
2026-10-13: the engine (`il/rank1/il_rader_stage.h`, position-general, the nested inner,
the threaded column split) behind the flat DIT's hooks and the tool door, verified by the
probe (§7, §9). PHASE 2 BUILT 2026-10-13: the planner admits leaf chains, races the
inner chain and width in role and banks them, the prime cell's inner pool offers the
flat chain with a Rader stage; the class sweep (phase 3) waits for a quiet machine.*

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
263, ...): that prime takes stage 0 of the inner chain as a nested leaf over the scratch
plane, one level (its own inner must chain from the pool). The inner column chain is a
DIF, so its stage 0 carries the span's twiddles on its outputs: the nested leaf multiplies
its output rows by w_(p-1)^(l d) on the forward pass and its gathered rows by the
conjugates on the reverse pass (the engine's second twiddle mode; the flat chain's own
mid stages use the first, the pre-twiddle per block). Raced per the owner (§9).

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

## 9. Phase 1 measured (the engine alone: the pool's greedy inner, a fixed width)

The probe (`rader_probe.c`) builds the chain through the tool door and races the whole
forward against the front door's plan at the same N (the prime cell, Bluestein on the
whole length), pinned, 15 alternating rounds; the chain's time over the door's:

| N | chain | leaf first | prime last | inner |
|---|---|---|---|---|
| 524 = 4 x 131 | 131.4 / 4.131 | 0.34 | 0.79 | 13.10 |
| 1233 = 9 x 137 | 137.9 / 9.137 | 0.43 | 0.90 | 8.17 |
| 2101 = 11 x 191 | 191.11 / 11.191 | 0.36 | 0.74 | 19.10 |
| 2575 = 25 x 103 | 103.25 / 25.103 | 0.41 | 1.00 | 17.6 |
| 2073 = 3 x 691 | 691.3 / 3.691 | 0.88 | 1.10 | 23.15.2, three columns |
| 2276 = 4 x 569 | 569.4 / 4.569 | 0.67 | 1.16 | 71 nested over 8; 7.10 |
| 1177 = 11 x 107 | 107.11 | 1.02 | | 53 nested over 2; 4.13 |
| 428 = 4 x 107 | 107.4 / 4.107 | 1.16 | 1.98 | 53 nested over 2; 4.13 |
| 1149 = 3 x 383 | 383.3 / 3.383 | 1.95 | 2.21 | 191 nested over 2; 19.10 |
| 3127 = 53 x 59 | 53.59 / 59.53 | 1.30 | 1.29 | two Rader stages |
| 2809 = 53 x 53 | 53.53 | 1.01 | | two Rader stages |

Every chain reads 1.8e-16 to 3.1e-16 against the long-double reference, roundtrips at
1.6e-15 to 3.2e-15, in place bitwise equal to out of place, and threaded (T = 4) bitwise
equal to serial, nested chains included.

The owner's two races, answered:
- Placement: the leaf wins at every cell, by 1.9 to 2.4x over the prime-last placement;
  with two primes above the pool, two Rader stages lose to the whole-N Bluestein (1.30 and
  1.01). Phase 2 admits leaf chains only; the engine stays position-general, no candidate
  names a mid or last Rader stage.
- Recursion: the nested inner wins at 2276 (0.67), ties at 1177 and loses at 428 and 1149,
  where the leaf has two to four columns to vectorize over. It stays a raced arm: the race
  protects a cell, and it pays where the leaf is wide.

The cofactor sets the leaf's width: at 2073 the three columns run two lanes wide. The
width and the inner chain are forms the phase-2 race decides.

## 10. Decisions

Agreed 2026-10-13: form B (the leaf in the flat chain) over the four-step's 2D view;
the order complex leaf, real odd-prime method, real leaf.

Answered by the owner 2026-10-13 and by the phase-1 race (§9):
1. The mid-chain placement was built and raced: it never beats the leaf, and two Rader
   stages lose to Bluestein. The planner admits the leaf only.
2. The one-level recursion was built and raced: a raced arm, kept.
3. The token names are the implementer's: the inner chain banks on the K=1 row as
   `il_rd=` (its radices, dotted, an `n` after a nested prime), `il_rdf=` (its forms) and
   `il_rdw=` (the window width).

## 11. Phases

1. BUILT 2026-10-13. The engine: the stage's create (permutations, kernel, the inner
   chain, the nested leaf, scratch per worker) and execute (both directions, column
   windows, the MT column split), behind `vfft_ilfd_create_chain` through the hooks
   `il_flatdit.h` declares and `il_rader_stage.h` defines; the tool door
   (`vfft_tool_ilfd_chain` / `_exec` / `_exec_mt` / `_destroy` / `_rd_desc`) and the
   probe `docs/research/field_trial/gapfill/work/rader_probe.c`.
2. BUILT 2026-10-13. The chain search takes a fixed leading prime (`leadp`, after the
   lone 2 where one leads: then the prime is the fixed second stage, a Rader stage with
   pre-twiddles, the only flat form of a 2 x odd x p length); the flat search sets it for
   a length with exactly one prime factor above the pool; the bench races the inner chain
   (every column chain of p-1 in the two orders, a nested prime fixed first, at most ten)
   and then the width around the winner's, every arm a live engine swapped into the stage;
   the verdict rides on the K=1 row as `il_rd=` and `il_rdw=` and replays through the
   commit; no tile axis on such a chain; the flat gate holds 524, 2101, 2276 and 2118.
   The policy needed no change: the flat family was admitted at every non-pow2 length
   already, it simply had no chain. The prime cell's inner pool gained the kind `fd`: the
   flat chains of N minus 1 with that length's own prime above the pool as a Rader
   stage, so 4099 (4098 = 2 . 683 . 3) and 2131 (2130 = 2 . 71 . 15) race Rader again;
   in the first (noisy) cold creates both lost to their Bluestein inners, the 2 x odd
   shape leaving the Rader stage two columns wide. Verified: cold creates through the
   front door at 2101 (`il_flat=191.11 il_rd=19.10 il_rdw=10`, 1.9e-16), 2118
   (`2.353.3`, inner `4.4.22`), 2276 (raced, the prime cell won that run), each replaying
   bitwise from its row.
3. The class sweep against FFTW, once; the results page.
4. The real odd-prime method (§8); then the real leaf.

## 12. Phase 2 as built (2026-10-13)

Phase 2 is built and verified. The flat gate passes with the four new cells through the
front door, which covers the forward against an independent DFT, the backward roundtrip,
in place, the banked verdict replaying bitwise, and the threaded form equal to serial at
T = 8. The policy gate passes its 182,859,624 checks after the change.

**What the planner does now.**

- The chain search takes a fixed leading prime, so a length with exactly one prime
  factor above 47 gets its leaf chain among the flat candidates; a lone factor 2 keeps
  the leaf and the prime becomes the fixed second stage, the only flat form a
  2 x odd x p length has.
- The bench races the inner chain of p minus 1 in role on the whole forward, every
  column chain in both orders with a nested prime fixed first, then the window width
  around the winner's. The verdict banks on the K=1 row as `il_rd=` and `il_rdw=` and
  replays through the commit. The tile axis is off for such chains.
- The prime cell's inner pool gained the flat chain with a Rader stage as an inner kind
  (`fd`), so a prime whose N minus 1 has a factor above 47 offers Rader again: 4099 raced
  one such inner, 2131 three.
- No policy change was needed: the flat family was already admitted at every
  non-power-of-two length, it simply had no chain.

**Cold creates through the front door**, in a fresh scratch store: 2101 banks
`il_flat=191.11 il_forms=n il_rd=19.10 il_rdw=10` at 1.9e-16 and replays in no time,
2118 banks `2.353.3` with its inner `4.4.22`, 2276 raced its nested chain and the prime
cell won that run, and 4099 and 2131 both kept Bluestein. Those last verdicts came from a
noisy machine and the 2 x odd shape leaves the Rader stage two columns wide, so they are
not final.

**Two bugs found and fixed on the way.** A division inside a dead ternary branch that the
compiler evaluated anyway, and the plan's call records still naming a freed engine after
the inner race installed its winner.

**The re-race on a quiet machine (the same day).** 4099 and 2131 keep Bluestein: their
Rader inners, the lone 2 then a Rader stage two columns wide, lose to the ZTURN-T inners
at 8192 and 4096. 2276 keeps the prime cell at 13.2 us against the nested leaf's 14.5 us.
The race protects those cells; the leaf serves where it wins.

**Phase 3, the class sweep against FFTW** (every composite to 4,096 with a prime factor
above 47, 2,157 lengths, `rader_class_fftw_20261013`, every cell re-raced cold: the
gauntlet seeds its store from the shipped wisdom, so without the re-race flag a cell
replays its old verdict and the leaf never races). The first 1,562 lengths, 106 to 3149,
against the September record on the same lengths: the class geomean 1.075x to 1.775x,
lengths below parity 663 to 24, below 0.8x 213 to 2; 1,020 lengths took the flat route
(their geomean 1.06x to 2.05x) and 542 stayed on the prime cell (1.22x). The ten worst
lengths before, 0.57x to 0.63x, now read 1.04x to 2.33x. Forty-four lengths lost more
than a tenth, most of them prime cells whose cold re-race picked a slower inner, a few
leaf chains at cofactor 3 (1893 = 3 x 631: 1.24x to 0.76x), where the Rader stage has
three columns. The full class, all 2,157 lengths against the September record: the class geomean 1.146x
to 1.790x, below parity 684 to 24, below 0.8x 213 to 2; 1,412 lengths on the leaf chain
(1.13x to 2.06x as a group), 745 on the prime cell (1.17x). On the page's rule (the
slower of the two engine orders per length) the class reads 1.64x at the median with 98
percent at or above parity, and the whole 2 to 4,096 sweep 1.63x median, 91 percent, from
1.24x and 73 percent. The record is `gauntlet/results/rader_class_fftw_20261013`
(`compare_vs_sept.txt`), the page `docs/performance/vs_fftw.md`. The accuracy sweep of
every length 2 to 4,096 after the admission (`gauntlet/results/accuracy_c2c_20261014`,
4,095 lengths against the long-double reference): VectorFFT's l2 error reads 4.7e-16 at the
median and 1.3e-15 at worst, MKL's 4.0e-16 and 6.1e-16. The 17 lengths above 1e-15 are all
leaf chains whose prime carries a nested inner leaf (107 with 53 inside 106, 263 with 131,
269 with 67, 317 with 79, 823 with 137): two Rader levels cost two to three times the median
error, a declared side effect of the speed race, which gates accuracy only at the
planner's tolerance. The build script gained a
`VFFT_DEBUG_SYMS=1` switch for line numbers under a debugger.

## 13. The spread: every interleaved feature (owner, 2026-10-14)

The leaf is one engine with one home, the 1D c2c flat chain, and the composites inherit
it wherever they plan a 1D c2c child in role. Where a prime factor above 47 stands today,
per interleaved feature:

| feature | a prime p > 47 in the length goes to | after the spread |
|---|---|---|
| 1D c2c, K = 1 and its batches (TC, plane queue: clones of the K=1 plan) | the leaf (§2-§12) | done |
| 1D r2c / c2r, even N | the zr2c child at N/2: the leaf through the child | done |
| 1D r2c / c2r, odd prime N | the real Bluestein `zrb` at M >= (3N-1)/2 | piece B |
| 1D r2c / c2r, odd composite N = p . m | `zrb` (the real flat DIT `zrf` has no leaf at p) | piece C |
| 2D / 3D c2c and real, the last axis | the row plan: a 1D cell (the odd-N2 row child is c2c; the even row engine's zr2c child) | done |
| 2D c2c and real, axis N1; 3D c2c and real, axes N1 and N2 | the column-axis Bluestein (`blu`, an M >= 2N-1 chain over the lanes) or the turned prime pass (`tpc`, a prime N through the 1D route per lane) | piece A |

**Piece A, the Rader stage in the column chain (2D and 3D, c2c and real at once).** The
engine is already the column pass's geometry: block b of span L, legs at stride D, the
columns contiguous, and its twiddle mode 2 is exactly a DIF column chain's stage 0 (the
nested inner leaf runs there today, §4). So the column pass admits a prime p above the
pool as stage 0 of its chain with the engine at D = D_s . rn, L = N1 . rn, nb = 1, the
stage-0 tables replaced by `_ilrd_dif_tw(span N1, Dp = N1/p, twrun = rn)`, and every walker
that runs stage 0 (`_il2d_col_pass_range`, `_il2d_col_pass`, the natural and strip and
tile walkers, the banded prefix) runs the engine's windows over its lane range instead of
the kernel: a lane range [k0, k1) of the plane is the engine's runs [d . rn + k0, d . rn +
k1) per digit d, which the engine's own run splitting already handles. One descriptor
field (`rd0` on `vfft_ilcol_t`) carries the engine; `_il2d_resolve` admits p at stage 0
when p - 1 chains from the column pool; the enumerator offers the leaf candidates (p
first, the chains of N1/p after it) for an N1 with exactly one prime factor above the
pool; the race takes them as ordinary chains against Bluestein and the turned pass. The
Rader stage's own forms are its inner chain and window width, raced in role on the column
pass the way the flat's `vfft_ilrd_race` races them on the whole forward, and banked on
the column row as new tokens (axis-suffixed at rank 3 like every column token: `rd=`
`rdw=` at axis 0, `rd1=` `rdw1=` at axis 1 are the proposed names; the owner names
tokens). Threads: the strips arm hands each worker a lane range, the block arm a digit
range, and both are runs of the engine's column space with the worker's scratch slot; no
arm needs a form it does not have. The 3D tiers build their axes through the same
`_il2d_col_build` and run them through the same walkers, so they inherit piece A with no
code of their own; the 3D real tier's axis 0 drops its "a Bluestein axis has no plain
chain" restriction where the Rader chain serves (the chain is plain: the natural
permutation derives as for any chain).

Phase A1, BUILT 2026-10-14, no wisdom change: the engine on the descriptor (`rd0`), the
seven walkers switching on a NULL stage-0 kernel and taking the engine from the stage-0
table slot (a direction tag, a block of the table set, so a race's own table set names the
engine the build attached through `_ilrd_col0_ctx` and frees its tags with its tables),
the resolver and the table builder, the digit-split trampoline, the pool's per-thread slot
(`thread_pool_self_slot`: the caller 0, worker i at i + 1) for the engine's scratch, and
the three whole-plan forms that read the stage-0 kernel directly declining such a chain.
Reached through the `VFFT_IL2D_CHAIN=p.R1...` pin. The probe
(`docs/research/field_trial/gapfill/work/rader_col_probe.c`): every cell passes the naive
DFT in natural order at 1e-14, the scrambled and the natural roundtrips at 2e-15, in place
bitwise against out of place, at T = 8 with the column MT engaged, the real pair, and the
3D cube; against the route the library serves today, raced on its own scratch store:

| cell | pinned chain | today's route | time today / Rader |
|---|---|---|---|
| c2c 524x64 | 131.4 | turned prime pass | 1.66 |
| c2c 262x64 | 131.2 | turned prime pass | 2.46 |
| c2c 131x256 | 131 | turned prime pass | 1.80 |
| c2c 1048x32 | 131.8 | turned prime pass | 1.44 |
| c2c 2101x16 | 191.11 | turned prime pass | 1.48 |
| r2c 524x64 | 131.4 | column Bluestein | 2.82 |
| r2c 131x256 | 131 | column Bluestein | 2.73 |
| c2c 524x8x8 | 131.4 (axis 0) | turned prime pass | 1.68 |

Two library defects came out on the way and are fixed with it: GCC on Windows ignores
`__declspec(thread)` (a warning), so the pool's serial-scope depth was one static shared by
every thread until this day (`threads.h` uses `__thread` except under MSVC now), and
`vfft_destroy` frees the 2D column tables by hand rather than through `_il2d_col_free`.

**The regression is `src/tools/gates/rader_gate.c`** (owner, 2026-10-14: a check of the
Rader stage's own touch points, run after every Rader change instead of the old gates;
`run_gates.py --only rader_gate`, a cold scratch dir, about two minutes): the leaf through
the tool door (forward, backward, in place, threaded bitwise, the nested inner, the inner
race's winner), the column stage through the pin for 2D c2c and real (natural, both
roundtrips, in place, both threaded arms engaged, the banded walk bitwise, the pair, the
in-place real plane, a prime alone, p minus 1 a power of two, nested inners, the half-store
leaf forms), the 3D tiers, a create / execute / destroy loop, and the planner's own pick
with its replay at one leaf cell. Its first runs settled three more rules: the flat
threaded form's unit for a Rader stage is a whole window (a column unit cut windows and
moved the bits by 1e-16), the real 3D tier's pay-once form declines a one-stage Rader chain
(it calls the leaf kernel directly), and a Rader radix is odd and above the pool (64 is a
kernel radix). One gap it reports and does not fill: a one-stage column at the c2c tier,
the prime alone, has no threaded column form there (its walks split by digit or need two
stages); the real tier's strips arm threads it.

**Against MKL (2026-10-14, the record `docs/performance/v1_0_results.md`, the Rader column
section; every cell raced cold on a fresh scratch store, the canonical bench's protocol).**
The column stage, pinned, against the route MKL runs: 2D c2c 3.2x to 6.7x at a wide prime
(131 or 191 over 4 to 8 digits), 1.2x at 53 alone; 3D c2c 2.9x to 6.2x, 1.3x at a nested 107;
2D r2c and c2r 2.9x to 6.0x, 1.3x to 1.5x at 53 alone; 3D r2c and c2r 2.2x to 6.6x. The 1D
real cells through the planner, the leaf in the zr2c child: 2.5x to 6.3x at a plain leaf,
1.7x at a nested one (4552, 2354). The pattern is the 1D class sweep's: the win is the width
the prime is vectorized over, and a nested inner or a two-column leaf gives half of it.

Phase A2, BUILT 2026-10-14, the planner's side. The law is `vfft_policy_il2d_rader_prime`
(`planning/policy_il.h`): a column length whose part above the chain radices is one prime p,
once, with p - 1 within the radices or one prime above them (once) whose predecessor is,
takes p as stage 0; the policy gate mirrors it over N = 2..2^23 by a Miller-Rabin spelling.
The chain enumerator (`il2d_cols.h`) offers the leaf candidates for such an N1 when the
engine builds p: p first, the pool's compositions of N1 / p after it, the chain [p] alone
at a prime N1 (no other chain exists there: these are its pool). The column build attaches
the engine for the race's prime before the chain race (every arm's table set names it),
then after the natural permutation and before the N-arm race builds the descriptor's
tables once and RACES THE ENGINE on the pass the cell serves (the N-arm harness, serial at
every thread count): the inner chains of p - 1 (`_ilrd_enum_inners`, a nested prime fixed
first, two orders) at the default width, then the widths around the winner's (half, double,
quadruple, the column space), every arm a live engine swapped into the stage's tags; the
winner stays attached and banks as `rd=` (the inner radices dotted) and `rdw=` (the width)
on the column row, the owner's token names (2026-10-14), axis-suffixed at rank 3 and `_p`
at door 2 like every column token; the attach replays them (never under recalibrate) and
takes the default when a banked recipe no longer builds. At a prime N1 the N-arm race
times the chain [p] against Bluestein as any odd chain, and when the chain wins the turned
pass is the third arm (timed as `_il2d_tpc_race` times it); a turned win banks `blu=M
tpc=1` and the cell is the turned cell it was. A pinned chain (`VFFT_IL2D_CHAIN`) races
the engine too but has no row at race time (the axis race creates it): its verdict lands
at the cell's next create. The inner enumerator's miss at p - 1 = q * 2 (a nested prime
then a lone closing radix: 107, 1177) is fixed with it: the flat leaf races there now too.
The gate's section A2 creates three cells cold with no pin (c2c 1048x16 and 131x32, r2c
1048x16): the pick is the Rader chain (or the turned pass at the prime), the rows carry
both tokens, the replay is bitwise. Against MKL with no pin (the results page, the same
cells and protocol as the pinned run): the planner picks the Rader chain at every cell
and lands on the pinned numbers or a few percent above them (2D c2c 6.9x, 6.7x, 6.3x,
3.2x, 3.3x, 1.1x; 3D c2c 6.2x to 1.3x; 2D real 6.3x to 1.2x; 3D real 5.5x to 2.3x).
The engine's raced verdicts: 131 takes the inner 13.10 at widths 32 to 120 by cell, 191
takes 19.10 at 44, 107 takes the nested 53.2, 353 takes 32.11. Phase A3: the class
sweep of 2D cells with such an N1 against FFTW, once.

The threaded forms (owner, 2026-10-14: "test MT of every new rader too"): the gate's
section M creates, cold at T = 8 with no pin, one cell per tier the stage reaches (1D
c2c 134464 = 191·11·64; 1D r2c and c2r 134144, the child at 67072 = 131·512; 2D c2c and
real 1048x256; 3D c2c and real 131x32x32), judges the forward against the reference (the
naive DFT at the 2D and 3D cells, the T = 1 plan's result at the 1D cells), requires the
library's engagement counters to move (a threaded result without that proof is vacuous),
runs the backward or the c2r pair, and replays at T = 8 bitwise. Its first run (338
checks): the 1D c2c flat chain, the 2D c2c, r2c and c2r column passes and the 3D c2c, r2c
and c2r tiers all engage a threaded pass at T = 8, agree with their serial plans to
1e-12 and replay bitwise. The 1D real cells run with no threaded pass: the even-length
real engine creates its complex child single-threaded (`_zr2c_kid_cfg`), so the leaf's
threaded form never runs inside a 1D real plan, and the fold's own threaded arm lost
the race at 134144. A gap, reported, not filled; the gate prints it and does not fail
on it.

**Piece B, the real odd prime (1D r2c / c2r at a prime N > 47).** Rader on real input:
with h = (N-1)/2 and g^h = -1 mod N, the kernel splits as omega_q = u_q + i v_q with u
h-periodic and v h-antiperiodic over q, so the length-(N-1) convolution of the real
gathered sequence a is two real convolutions of length h, a cyclic one of the folded
a_q + a_{q+h} with u and a negacyclic one of a_q - a_{q+h} with v, and X_k for k <= h
(the half spectrum the r2c owes) is their sum at q = 0..h-1. Each is a real transform
pair at h (r2c, the pointwise product, c2r; the negacyclic one through the odd-index
twist), served by the real door's own engines at h raced in role, the way the prime
cell's inner pool serves Bluestein. The plain form (one r2c and two c2r at N - 1 over the
unfolded a) is the first correct build and the folded form the raced target. A second
arm beside `zrb` in the odd builder's sweep (`odd_build.h`), gated against the same
reference, banked as a new engine on the real row (`eng=zrr`, the proposed name). The
c2r is the backward of the same stage (the conjugate kernel, the permutations swapped)
with its half-spectrum input unfolded by the Hermitian symmetry on the gather.

**Piece C, the real Rader leaf (1D r2c / c2r at an odd composite N = p . m, p > 47).**
The real flat DIT's level puts a real leaf `r1c` over D0 columns: digit 0 real (down a
level), digits 1..h complex blocks. A Rader leaf at radix p is piece B's structure over
columns: the same two real convolutions of length h, vectorized across the D0 lanes
through the real axis's pack trick (two real rows as one complex row, the complex column
chain of length h/2 or h, the fold), their DC row to the real plane and rows 1..h to the
complex blocks. The engine is piece A's with real rows and the fold; it enters `zrf` as
the leaf of a level whose radix is above the pool, raced in the flat DIT's own sweep
(`_zrf_try`) beside the other chains, banked on the real row with the inner tokens of
piece A. `_zrf_has_chain` admits p as a leaf radix when p - 1 (or h) chains.

**Order.** A, then B, then C: A is the engine's own geometry and unlocks four tiers at
once; B and C share the real-convolution core, and C's engine is A's with real rows. The
design's earlier order (B, C, then the columns) is superseded by this section.

**Decisions open (asked 2026-10-14).** (1) The token names: `rd=` and `rdw=` on the
column row (suffixed by axis at rank 3) for piece A, `eng=zrr` for piece B. (2) Whether
piece B's first build may ship the plain form (three real transforms at N - 1) with the
folded form as a later raced arm, or must land folded. (3) Whether the two-level leaf
stays admitted (the accuracy note of §12: the 17 lengths above 1e-15 are all nested) or
the planner admits one level only; the same rule would apply at the column pass.
