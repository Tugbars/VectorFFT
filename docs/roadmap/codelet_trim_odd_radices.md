# Radices the estimate no longer picks: candidates for the codelet trim

Note for the owner, 2026-10-26. The radices the estimate's chains stopped using, the reason, and the
measurements behind each. Input for the codelet trim; the trim itself is the owner's ruling. The
radix 32 and 64 question is [codelet_radix_sunset.md](codelet_radix_sunset.md).

## The rule

AVX2 has 16 YMM registers. A butterfly up to about radix 16 fits in them; above that it spills
(the owner, 2026-10-26). For the **odd composite radices above 16** (21, 25, 27) there are two kernel
forms, and the rule follows the form:

- the **direct (conjugate-pair) kernel**: one 21-, 25- or 27-point DFT in one pass. The estimate names
  it nowhere; such a step is split instead (21 -> 7.3, 25 -> 5.5, 27 -> 9.3).
- the **odd-composite Cooley-Tukey twin** (`_ct`, il_kv form 5): the same radix as internal stages
  inside one call (21 = 3 x 7, 25 = 5 x 5, 27 = 3 x 3 x 3). It spills less, at the price of more memory
  passes inside the call. It exists only in the 1D pair and chain3 kernels (`t2_ct`, `n1t_ct` and the
  backward twins), and there the estimate keeps 21, 25 and 27.

| radix | status in the estimate |
| --- | --- |
| 21, 25, 27 | the `_ct` twin in 1D pair and chain3 slots; split everywhere else |
| 15 | kept: below 16, measured winner (below) |
| 9, 11, 13 | kept: below 16 |
| 17, 19, 23, 29, 31, 37, 41, 43, 47 | kept: prime, no smaller split; the only alternative is a Rader or Bluestein stage |

One place is stricter: the inner chain of a 2D column Rader stage (p - 1 for a prime column p) also
drops 15, 64 and a leading 32 (below).

## Where the estimate applies it

| estimate chain | kernel form there | radices it no longer picks | state |
| --- | --- | --- | --- |
| 2D/3D column chains (`est__pool`, estimate_nd_il.h) | direct | 21, 25, 27 | committed f9f640e8 |
| 2D column Rader stage, inner chain of p - 1 (`est__rader`) | direct | a step above 32 (64), an odd step above 13 (15, 21, 25), a first step of 32 | committed 47d2ab0c |
| 1D four-step, column chain at N1 (`est_fs_col_chain`, estimate_il.h) | direct | 21, 25, 27 | committed d83f867f |
| 1D Rader stage, inner chain of p - 1 (`est_rader_window`) | direct | 21, 25, 27 | uncommitted |
| 1D solo kernel at N = 21, 25, 27 | direct | 21, 25, 27 | committed f9f640e8 |
| 1D convolution inner's pair (`est_inner_pair`, the prime route) | direct (no form token) | 21, 25, 27 | f9f640e8, kept uncommitted |
| 1D pair and chain3 slots (`est_pair_legal`, `est_c3_legal`) | `_ct` (`est_nib_fwd`) | none: 21, 25, 27 kept | uncommitted (reverts f9f640e8 there) |

The flat DIT route builds 21 or 25 only when a chain passes 10 stages (`est_merge10` merges 7.3 and
5.5); no length of the 82,283-length 1D scan reaches it. The ZTURN-T odd mids are 15, 9, 7, 5, 3.

Still naming a direct kernel: the 1D Rader stage at p = 4051 (16 lengths), whose p - 1 = 4050 has no
inner candidate without 25 or 27; and the backward of a chain3 with 21, 25 or 27 in a mid slot (the
backward mids have no form twin, il2p.h:1649-1652).

## The measurements

All one thread, natural order, out of place, 14900KF, 15 rounds with a control arm, the arms in one
process. Evidence (git-ignored): `docs/research/estimate_threaded/evidence/fourstep/` and
`.../evidence/nd/rader2d/`.

- **1D four-step column chains**: every chain of the planner's pool at every tested split, 8 lengths
  from 300,000 to 1,200,000. The best chain with a 25 or 27 lost 2-17 % to the best chain without
  one, at every split but one; at 1,000,000 a 25-led chain tied (0.968 against 0.971). At the
  estimate's own split, dropping the 25 alone took 480,000 from 1.39 to 1.05 against the planner,
  360,000 from 1.25 to 1.09, 500,000 from 1.35 to 1.03.
- **Radix 15 stays**: at 300,000 the 15-led chains won (15.5.8 0.99, 15.8.5 1.00 against the
  planner); the best chain without a 15 measured 1.08.
- **2D column Rader inner chains**: every inner chain against windows 8-256, 26 shapes (primes
  53-509). Against the planner: 4.15 1.13-1.86 (the best is 5.12), 21.6 1.37-1.40 (9.14), 4.25 1.42
  (4.5.5), 25.10 1.27 (5.5.10), 4.64 1.28-1.33 (8.32), 3.64 1.06-1.07 (16.12 or 4.4.12), 32.3
  1.14-1.15 (3.32).
- **3D, the flat tier on odd planes**: every odd plane whose chains carry a radix of 11 or more lost
  to the 2D plan per plane, 12 of 12 (1.04-1.23).
- **2D/3D column chains without 21, 25, 27** (the estimate, 96 shapes): 100^3 3.48 -> 2.98 ms,
  125^3 9.80 -> 8.60 ms, 21 x 21 726 -> 593 ns; 21^3 and 21 x 21 x 64 unchanged; the other 91 shapes
  kept their chains. The planner with the three out of its column pool, against its stored winners
  at the only 4 stored cells whose column chains hold one: 100^3 1.12-1.15 (the estimate's plan
  without 25: 1.06), 1000 x 1000 1.01-1.10, 21 x 21 x 64 0.84-1.00, 125^3 0.96-0.99 (two runs).
- **1D pair and chain3 keep them**: of the store's 490 pair and chain3 slots at 15, 21, 25, 27 the
  race took the `_ct` twin at 482; the direct kernel won 8. The planner with 21, 25, 27 out of every
  pool lost 1-3 % on average, up to 1.27 at 1470; the estimate without them lost 11 % over the
  chain3 lengths. Restoring them (same run, old estimate against new, 8 lengths): 861 0.62,
  399 0.87, 3570 0.87, 125 0.90; 1600 1.02 and 19683 0.99 within the control's spread.
- **1D Rader stage inner chains without 21, 25, 27** (same run, old estimate against new, 22
  lengths 404-64,576): every one faster, 0.64-0.94 (4.27 -> 9.12 at 436 0.64, 4.25 -> 4.5.5 at
  404 0.74, 21.6 -> 9.14 at 508 0.82, 16.21.3 -> 8.9.14 at 64,576 0.87).
- **Where the direct kernel still wins in the store** (single race results): 1D, 12 steps (21 at
  105 and six flat chains, 25 at 10125 and 225225, 27 at 6561 and 177147); 2D/3D, 4 column chains
  (21 x 21 x 64 one axis; 25 at 1000 x 1000 and the outer axis of 100^3 and 125^3).

## For the trim

- **Removal candidates: the direct 21, 25, 27 kernels** in every family (the column kinds
  `t2c`, `t2cp`, `t2cs`, `t2csg*`, `n1c*`, and the 1D `t2`, `n1t`, `n1`). No estimate plan names them
  except the two cases above. Before they go: the chain3 backward mids need a form for 21, 25, 27
  (or chain3 keeps them out of its mid slots), and the planner's pools drop them, with the re-race of
  the cells that bank them.
- **Keep**: the `_ct` twins (`t2_ct`, `n1t_ct`, `n1_ct_bwd`, `t2t_ct_bwd` at 15, 21, 25, 27), 15,
  the odd radices up to 13, and the primes 17-47.
- **64 and a leading 32**: see [codelet_radix_sunset.md](codelet_radix_sunset.md); the Rader inner
  measurements above add to it.
