# Radices the estimate no longer picks: candidates for the codelet trim

Note for the owner, 2026-10-26. The radices the estimate's chains stopped using, the reason, and the
measurements behind each. Input for the codelet trim; the trim itself is the owner's ruling. The
radix 32 and 64 question is [codelet_radix_sunset.md](codelet_radix_sunset.md).

## The rule

AVX2 has 16 YMM registers. A butterfly up to about radix 16 fits in them; above that it spills
(the owner, 2026-10-26). So the estimate builds no chain with a **composite odd radix above 16**:

| radix | split into | status in the estimate |
| --- | --- | --- |
| 21 | 3.7 | eliminated |
| 25 | 5.5 | eliminated |
| 27 | 9.3 | eliminated |
| 15 | kept | measured winner (below) |
| 9, 11, 13 | kept | below 16 |
| 17, 19, 23, 29, 31, 37, 41, 43, 47 | kept | prime: no smaller split exists; the only alternative is a Rader or Bluestein stage |

One place is stricter: the inner chain of a column Rader stage (p - 1 for a prime column p) also
drops 15, 64 and a leading 32 (below).

## Where the estimate applies it

| estimate chain | radices it no longer picks | state |
| --- | --- | --- |
| 2D/3D column Rader stage, inner chain of p - 1 (`est__rader`, estimate_nd_il.h) | a step above 32 (64), an odd step above 13 (15, 21, 25), a first step of 32 | committed 47d2ab0c |
| 1D four-step, column chain at N1 (`est_fs_col_chain`, estimate_il.h) | 21, 25, 27 | in progress |
| 1D three-stage route (chain3) and two-pass pairs | 21, 25, 27 | next, after a timing check |
| 2D/3D column chains | 21, 25, 27 | after that |

The flat DIT route never builds 21, 25 or 27 (it merges 3.3 into 9 and takes primes otherwise).

## The measurements

All one thread, natural order, out of place, 14900KF, 15 rounds with a control arm; the arms are the
estimate's own plan with only the chain changed, built with nothing measured. Evidence (git-ignored):
`docs/research/estimate_threaded/evidence/fourstep/` and `.../evidence/nd/rader2d/`.

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

## For the trim

- **25, 27, 21**: removal candidates. Once the rule covers every estimate chain, no estimate plan
  uses them. The planner's race can still pick them (1,000,000's 25.8.10 tied the best), so taking
  them out of the planner's pools, with the re-race of the cells that bank them, is the trim's step.
- **64 and a leading 32**: see [codelet_radix_sunset.md](codelet_radix_sunset.md); the Rader inner
  measurements above add to it.
- **Keep**: 15, the odd radices up to 13, and the primes 17-47.
- Not counted here: which codelet families (column, pair, row-loop) carry 21, 25 and 27, and how
  many shipped wisdom rows bank them.
