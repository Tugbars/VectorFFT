# 1D estimate: the sketch (interleaved library, FP64, AVX2 and AVX-512)

2026-10-22. A sketch for the owner, not a build: nothing was built, run or timed. The sketch is one
model, scored as one model. It lives in `synth/synth_estimate.py`, is scored by `synth/score_synth.py`
(`synth/results/score.txt`) and checked by `synth/legal_synth.py` + `synth/legal_wide.py`. All
paths are under this folder. Code citations are relative to `src/core/`. Row citations read
`store/shard:line`. Tags: **[V]** read in code or a row, **[M]** counted by a script here,
**[I]** interpretation.

**Revised after an independent check** (`check/`, laws in `check/laws.py`). The text now fixes the
backward forms `il_bkv`, the c2r mirror, the threaded four-step's child, chains past 10 stages, the
real bounds and the primes in (2^23, 2^24]. `synth/synth_estimate.py` was not updated: where the two
differ the text is the design, and the scores in section 8 are the model's from before the revision.

**How it was put together.** Three designs were judged on formulas first, legality, fit to the rows,
table size and simplicity. All three name the route equally well (91.5-91.6% i9, 97.4-97.6% Zen4).
- **Base: the MKL-faithful design.** It is MKL's own shape (an ordered list of cheap tests on N, one
  shape formula per route), with the smallest table and the widest legality sweep.
- **From evidence-fit:** the slot-order rules, the per-radix form table, the in-place and convolution
  ZTURN-T chains, the real even-N engine rules, the zrf chain, the prime inners and the threaded
  four-step split. Each one fits the rows better than the base.
- **From cache-model:** the L1-derived tile span in place of the literals 10/11, the Rader bound
  L1d/16, the scrambled chain for tiled sizes, the flat forms and the zrf tile budget.
- **Corrections found while merging** are in section 11.

---

## 1. What estimate is

Estimate is a planning method the user selects. It never measures at create: it returns a plan at
once, computed from N, the request and the machine's sizes. It is not the default and never a
fallback. The default stays the raced planner (patient), and a wisdom miss still races. Estimate
accepts compromises, as the owner ruled on 10-22 ("finance shops, very time critical solutions will
use the planner, basic user wants fast results").

The plan is mostly formulas. A table stays only where no formula reproduces the rows (section 5). The
banked wisdom is the evidence the formulas were fitted to and checked against. Estimate does not
read wisdom at create.

## 2. Inputs it reads from the machine

| input | used for | source |
|---|---|---|
| N, transform, order, placement | everything | the request |
| lane count VW (4 or 8) | the ZTURN-T floor (16 / 64), the 8-lane chain ends, no zrp / ZTT-r at 8 lanes | the build's ISA |
| L1d | the tile span s1 = floor(log2(L1d/16)), the serial bound 2·L1d, the flat and zrf tiles, the Rader bound L1d/16 | `l1d_used` (`common/support/cpu_cache.h:116`) |
| L2 | the four-step band width `fs_wl` only | `l2_used` |
| L3 | the threaded super-band `il_mtsb` only | `vfft_cpu_l3_bytes` (`cpu_cache.h:569`) |
| T (threads asked) | the threaded arm only; the route never depends on T [V] (`il/wisdom/wisdom2_oop_il.h:233-240`) | the request |
| core count | **not used**: no row relates a verdict to it (one CPU, T = 8 only) | — |

**Which core's caches (hybrid CPUs).** Estimate must read the P-core's sizes, the reading the
store's identity stamp carries [V]. The library reads L1d and L2 on a P-core by default
(`cpu_cache.h:89-97`, `_vfft_cpu_pcore_enter` at `:342`), and falls back to 32 KiB / 1 MiB when that
read fails (`:479-492`; s1 is still 11). MKL reads whichever core type runs first, so its thresholds
change between launches (MKL rule 30). [M] `synth/hybrid.py`: reading the i9's E-core sizes
(32 KiB / 4 MiB) would change 584 of 65,543 natural out-of-place plans at T = 1 (556 flat tiles, 26
Rader-or-Bluestein verdicts, 2 four-step band widths).

## 3. The decision procedure, complex 1D

Derived once per machine: s1 = floor(log2(L1d/16)), which is 11 for any L1d of 32-63 KiB;
W_lo = 2^(s1-1) and W_hi = 2^s1 (1024 / 2048 at s1 = 11, which is the planner's own ladder,
`il/planning/dp_planner_il.h:1455`; at other L1d sizes see Open decisions);
floor = 16 at 4 lanes, 64 at 8 lanes (`il/planning/policy_il.h:63-67`). The first test that holds
decides the route:

1. N < 2 or N > 2^24: **refuse** (the planner's ceiling).
2. N has a solo kernel and (no legal pair exists, or N ≤ 10): **mono**, `il_kv=0`.
3. N has a prime factor above 47 (a prime cell, `policy_il.h:169-182`): **prime cell** (4.6). A
   composite with 2N−1 > 2^24 is refused, as the code refuses it (`policy_il.h:298-302`).
4. N is a power of two:
   - a. N ≥ 2^19: **four-step** (4.5). ZTURN-T ends at 262144 (`il/rank1/ztt.h:121`).
   - b. N < floor, or the natural class with 32 ≤ N ≤ 512: **pair**. The mid R1 = 4 up to 128, else
     16; R2 = N/R1.
   - c. Otherwise: **ZTURN-T** (4.1).
5. N is in ZTURN-T's odd band: 2^a times a product of at most 5 factors from {15, 9, 7, 5, 3}, in
   2048..262144, with a ≥ 4 (a ≥ 6 at 8 lanes): **ZTURN-T odd chain** (4.2).
6. N > 262144:
   - a split exists: **four-step**, except the natural class in place, which takes the **flat DIT**
     (the four-step pays a plane copy there);
   - otherwise: the door's unraced Bluestein, refused when 2N−1 > 2^24. That leaves 3 lengths
     refused: 15759439, 16267463 and 16414841.
7. A legal pair exists: **pair** (4.3).
8. A legal chain3 exists: **chain3** (4.3).
9. Otherwise: **flat DIT** (4.4).

T > 1 keeps this route and adds only the threaded arm (section 7). Every order class and placement
uses this list, and every route it emits is in that class's pool (`policy_il.h:355-444`). The
scrambled pool is ZTURN-T alone in its band, the four-step alone at a power of two above it,
ZTT_ODD alone in the odd band, and the four-step + flat (+ prime) in the non-power-of-two
four-step band. Everywhere else it also admits the natural writers. The list sends a scrambled
request in each band only to that band's members.

**Scope:** one transform (howmany = 1). A contiguous batch on one thread is that transform's plan, run
once per transform (the owner, 2026-10-31); a batch on more than one thread and the lane-major batch are
not covered (see Open decisions).

## 4. Parameter formulas

### 4.1 ZTURN-T at a power of two (n = log2 N)

- **Fewest-stage chain** (the untiled shape): an ingest 8, then the forced 4s (0, 2 or 1 of them by
  (n−3) mod 3), then 8s. 16 is 4.4.
- **Natural, out of place.**
  - While n ≤ s1: the fewest-stage chain, untiled.
  - Otherwise, let s be the one value in {s1−2, s1−1, s1} with (n−s) divisible by 3. The chain is
    an ingest 8, then fill(s−3), then (n−s)/3 eights on the sweeps.
  - fill(b) is b/3 eights when 3 divides b. Otherwise it is radix-4 stages, with a single 8 in
    second place when b is odd.
  - `il_tw` = 2^s (W_lo when s = s1−2).
  - At s1 = 11 (the 10-22 chain rule with s1 in place of its literals): 4096 = 8.8.8.8 t1024;
    8192 = 8.4.8.4.8 t1024; 16384 = 8.4.4.4.4.8 t2048; 65536 = 8.4.8.4.8.8 t1024;
    131072 = 8.4.4.4.4.8.8 t2048; 262144 = 8^6 t1024.
- **Natural, in place.** The fewest-stage chain. `il_tw` is the longest chain prefix whose product
  is on {W_lo, W_hi} (else W_lo); untiled while n ≤ s1.
- **Scrambled (explicit request).**
  - While n ≤ s1: the forced 4s first, then 8s, untiled.
  - Above that: the natural out-of-place chain with its tiled part moved to the end, because the
    plain schedule tiles the suffix (`ztt.h:606-620`). At 8 lanes the tiled part is reversed so
    that the chain ends in 8.
  - `il_tw` = W_lo up to n = s1+2, else W_hi. If the scrambled tile law refuses that width, the
    other width is tried, else the plan is untiled.
- **The 8-lane law holds by construction** (`ztt.h:356-364`): every natural chain starts with 8 and
  every scrambled chain ends with 8.
- **Why** [V] (EVIDENCE §3.2): the ingest is 8 in 23/26 i9 rows, and the stage count is the minimum
  or one more in 26/26. MKL's plain radix-4 rule matches 0/26.

### 4.2 ZTURN-T in the odd band

- **Chain:** [4] + the odd part split greedily, largest first, over {15, 9, 7, 5, 3}
  (`dp_planner_il.h:1482-1535`), then the rest of the power of two as 8s and the forced 4s.
- **At 8 lanes:** the natural chain ingests 8 (`t0tp` exists at 8 only, `il_registry_avx512.h:575`).
  The scrambled chain ends in 8 (`tld`, `:1123`), with [4] as its ingest when a = 7.
- A 4-lane chain longer than 7 stages ingests 8 instead (only 200000 = 8.5.5.5.5.5.8). Untiled.

### 4.3 Pair and chain3

- **Pair factors:** the most balanced legal pair, min |log(R1/R2)|.
- **Pair order:** when exactly one factor is a prime ≥ 23, it is the mid R1; otherwise the larger
  factor is the leaf R2.
- **Chain3 factors:** the most balanced legal triple, min variance of log radix.
- **Chain3 order:** the leaf is the largest even factor (else the largest that is not a prime ≥ 23);
  the two mids follow in descending order. If the enumerator refuses that order
  (`dp_planner_il.h:1674-1776`), the two mids swap; failing that, the legal order with the
  largest leaf is taken (`design-evidence-fit/scripts/estimate.py:105-109`).
- **`il_kv`:** one nibble per slot from the forward form table (5.1), mid | leaf<<4 (chain3: A |
  B<<4 | leaf<<8).
- **`il_bkv`:** from the backward form table (5.1), not a copy of `il_kv`. The pair packs mid |
  leaf<<4; chain3 packs the leaf only, leaf<<8, because a nonzero mid nibble refuses
  (`il/rank1/il2p.h:1649-1652`). A nibble with no backward kernel makes
  `vfft_il2p_apply_kv_forms_bwd` / `vfft_il3p_apply_kv_forms_bwd` return −1 (`il2p.h:1084-1110`,
  `:1649-1659`), and the planner refuses the plan (`dp_planner_il.h:397,407`). This was the case
  for radix 32: no backward mid twin, and no backward variant 4 at any leaf (`il2p.h:892-947`).
- The power-of-two pair law (8 and 16 slots tangent only, no 64, `dp_planner_il.h:1598-1610`) holds
  by construction in both directions.

### 4.4 Flat DIT

- **Chain:** a lone 2 leads when N ≡ 2 (mod 4) (`il/planning/chain_search.h:239-243`); otherwise the
  largest radix leads. Then the 8s and the forced 4s, then the odd primes in descending order, with
  3·3 merged into 9. A Rader stage is never emitted, so the scrambled class always builds
  (`il/rank1/il_flatdit.h:313`).
- **At most 10 stages** (`VFFT_ILFD_MAX_K`, `il_flatdit.h:29`). While K > 10, merge two 5s into
  25; failing that, a 3 and a 5 into 15, else a 3 and a 7 into 21. All three are pool radices, and
  each merged radix takes its place in descending order. [M] This touches only 2 of the 111,510
  47-smooth non-power-of-two N in (262144, 2^24]: 11718750 becomes 2.25.5.5.5.5.5.5.5.3 and
  16406250 becomes 2.7.25.5.5.5.5.5.5.3 (`check/results/sweep_large.txt`).
- **`il_forms`:** `t` on every middle stage. The last stage is `t` at K = 2, otherwise `n`, and `o`
  in the natural class from K = 5. Every pool radix has the `t2csgn` tail (`il_flatdit.h:286-289`).
- **`il_tw`:** untiled while 16N ≤ 2·L1d. Otherwise it is the widest block span R_s·D_s of a
  non-tail stage (D_s > 4) with 16·span ≤ L1d (`il_flatdit.h:904-916`).

### 4.5 Four-step

- **Split, powers of two:** N1 = 4096 and N2 = N/4096 from 2^20; 2^19 is 512.1024, a table entry.
- **Split, other N:** the most balanced legal split with N1 ≥ N2. N1 needs a column chain and N2
  must not be a prime cell (`il/rank1/k1_fourstep.h:86-158`).
- **`fs_chain`:** 8s with one 4 or 16, then the odd part largest first, with a closing radix last;
  depth ≤ 4 (`il/rank2/il2d_cols.h:164-241`).
- **`fs_wl`:** the widest column stage span w < N1 with 16·w·N2 ≤ L2. That is the 2D tier's own L2
  gate (`il/rank2/il2d_tier.h:2000-2010`); 0 when no span qualifies.
- **The other switches:** `fs_tf` = (fs_wl > 0) (`il2d_tier.h:3936`); `fs_ro=0 fs_sw=0 fs_rbk=8
  fs_turn=0 fs_csk=0 il_kv=0`.
- **Row plan:** this procedure at N2 (c2c, natural, **in place**, T = 1). That is the row plan's
  child key, `ord=nat place=ip` (`il/rank1/k1_fourstep.h:226`); the earlier text said out of
  place. Its backward twin, `fs_row_bwd_` (key at `:229`), is the same plan carrying the row
  pair's or chain3's `il_bkv` (4.3). Other row routes have no backward twin.
- **One thread:** form 0 only (`il_kv=0`, no `il_sb`). The super-band form 1 (`k1_fourstep.h:955-958`)
  is not emitted at T = 1. The one T = 1 row where it was admitted (4194304, 16N > L3) banked form 0
  (`14900KF/wisdom2_oop.txt:209`).
- **In place:** the same split and child; the child's keys ignore the placement
  (`k1_fourstep.h:219-221`). Unfitted: the in-place rows differ (1048576 ip = 512.2048,
  `wisdom2_oop.txt:10261`) and carry no child, so the code treats them as stale.
- **The same formulas name every other four-step child, at its own length,** so that no create
  measures:
  - the prime cell's and zrb's four-step inner, at M (scrambled class, out of place, T = 1, form 0,
    `k1_commit.h:276-307`);
  - zfsr's child, at N/2, carried as `fs_*` tokens on the zfsr row (`il/wisdom/wisdom2_real_il.h:21-28`).

  Without these tokens the child races at create (`k1_commit.h:276-307`, `il/real/zfsr.h:91-92`).

### 4.6 Prime cell

- **Method:** Rader iff N is prime and one of these holds:
  - N−1 is a ZTURN-T length at VW;
  - lpf(N−1) ≤ 23 and N ≤ L1d/16, with an inner at N−1;
  - the Bluestein M would pass 2^24 (N > 2^23) and N−1 has a four-step split (4.5).

  Otherwise Bluestein.
  - lpf = the largest prime factor.
  - L1d/16 is 3072 on the i9 and 2048 on Zen4; both give 259/294 on Zen4 [M].
  - The third case: Rader has no size ceiling (`il/rank1/il_prime.h:299-311`), and its inner pool
    holds the four-step at M = N−1 in the four-step band (`k1_commit.h:470-481`). [M] From a
    sample of 4000 of the 513,708 primes in (2^23, 2^24], 38 qualify (about 0.9%), e.g. 15417851
    (`check/results/sweep_large.txt`).
- **Bluestein length:** the policy lists the power of two ≥ 2N−1, then the first three even
  2,3,5,7-smooth sizes ≥ 2N−1 (`vfft_policy_ilprime_blue_m`, `il/planning/policy_il.h:311-331`). M is
  the smallest of those smooth sizes that is at most 0.9 of the power of two and has a ZTURN-T inner;
  if none has one, the smallest smooth size when it is at most 0.9 of the power of two and has any
  inner; else the power of two.
  - Why the ZTURN-T test: a ZTURN-T length needs 16 to divide it, and the smallest smooth size often
    lacks it and runs a three-stage inner. At N = 1031 it is 2100 = 4.21.25, while the next size, 2160,
    runs ZTURN-T 4.15.9.4. At 4099 the smallest, 8232, has no inner at all and the power of two (16384)
    ran; 8400 runs ZTURN-T. The planner picks a ZTURN-T size in both cases.
  - Measured 2026-10-27, against the planner: 2D rows of 1031 (32, 64, 256 rows) went from 1.61-1.80
    to 1.04-1.05, 16 × 4099 from 1.41 to 0.99, and 28 primes from 1031 to 4099 from 1.147 to 0.794
    (geometric mean).
  - Refused above 2^24, which leaves the other primes in (2^23, 2^24] refused. The code serves some
    of them through the Rader-stage flat inner, which estimate does not emit (see Open decisions); it
    refuses those whose N−1 has neither a four-step split nor exactly one prime factor above the flat
    pool (`k1_commit.h:365-372`).
- **Bluestein inner:**
  - M = 128 (N = 53..61; no smaller M occurs): the pair 4.32 (`pow2_pair`, 5.3), which is the
    row at N = 59 (`14900KF/wisdom2_prime.txt:32`).
  - M ≤ 262144: ZTURN-T, scrambled, in its convolution form: 8s first, then the forced 4s at 4 lanes;
    the 4s first, then 8s at 8 lanes. Untiled while M ≤ 2^s1, else W_lo.
  - Above 262144: the four-step, with the split and child of 4.5 at M.
- **Rader inner at M = N−1:**
  - a pair if M ≤ 4096 has one;
  - otherwise ZTURN-T (the convolution chain, or the odd chain);
  - otherwise the code's own `vfft_il3p_default_chain` (`il/rank1/il2p.h:1523-1572`);
  - otherwise, in the four-step band, the four-step (4.5) at M.

  The prime inners carry no `il_kv` / `il_bkv` (`14900KF/wisdom2_prime.txt:30-33`).

## 5. What stays a table (31 entries)

Code facts the formulas read are not tuning and are not counted here: the kernel lists, the floor,
the ceilings, the bands and the tile ladders.

| # | entries | content | why no formula |
|---|---|---|---|
| 5.1 | 6 | slot form per radix. **Forward (`il_kv`):** {15,21,25,27} → 5; {8,16} → 3; 32 → 3 as mid, 4 as leaf; others 0 (the create's default, `il2p.h:747-750`). **Backward (`il_bkv`):** mid {8,16} → 3, leaf {8,16,32} → 3, {15,21,25,27} → 5 in either slot, others 0 (radix 32 has no backward mid twin and no backward variant 4, `il2p.h:892-947`; the CT radices are `VFFT_IL_*_CT_BWD_RADICES` in both registries) | a kernel property. Forward, against the rows: 95% (i9) / 92% (Zen4) where the ordered pair matches (153/161, 216/235); 84% / 87% where the factors match in either order. The default form alone gets 63% / 59% (`check/results/kv_default.txt`). Backward: reproduces the i9 dir=bwd rows 128 = 0x30 and 512 = 0x33 (`14900KF/wisdom2_oop.txt:155,161`) |
| 5.2 | 2 | power-of-two pair band, natural 32..512 | the handover is 512/1024 on the i9 and 1024/2048 on Zen4: one octave later on the smaller L1, the opposite of any L1 formula (EVIDENCE §3.1) |
| 5.3 | 1 | power-of-two pair mid: 4 up to 128, then 16 | i9 rows 5/5; Zen4 takes 16.8 at 128 |
| 5.4 | 1 | mono also at smooth N ≤ 10 | 6, 9, 10 are mono on all four row sets |
| 5.5 | 2 | 23: a "large prime" for slot order, and the Rader smoothness bound lpf(N−1) ≤ 23 | fitted on the i9 rows; cache-model's register-file formula also gives 23 at 4 lanes but 89 at 8, untested |
| 5.6 | 1 | flat `o` form from K = 5 | fitted (cache-model); forms match 41% / 53% given the banked chain |
| 5.7 | 11 | four-step: 2^19 = 512.1024; `fs_ro/sw/rbk/turn/csk` = 0/0/8/0/0; threaded split N/2048 up to 2^20; super-band chain 8.64.8 at N1 = 4096; at T > 1, `fs_cmt=1`, and `fs_mtarm=1 fs_msw=128` when fs_wl = 0, else 0 and 0 | the five switches are constant in every one-thread row. The T = 8 rows (`14900KF/wisdom2_oop.txt:8387-8390`) carry `fs_cmt=1` in 4/4 and `fs_mtarm/fs_msw` = 1/128 exactly where `fs_wl=0`, but also `fs_sw` = 8 and 256 at 524288 and 1048576. Estimate emits `fs_sw=0`, the code's default when the token is absent (`il/rank2/fft2d_create_il.h:686-688`). The 2^19 split has no cache explanation |
| 5.8 | 7 | real: zrm at even N ≤ 16 and odd N ≤ 17; zrp up to N = 160; zttr from N = 512 (pow2) and 2048 (other), `stk=3`; c2r child in place from N = 1024 | 160: every zrp-legal N up to 160 won zrp and every one from 192 lost; nothing cache-sized separates them |

**Fitted constants inside formulas:** the 2 in 2·L1d (serial bound, flat tile), L1d/16 (Rader
bound), and the zrf tile's 8N > L1d and 16b ≤ L1d/2. The L2 and L3 tests are the code's own gates.

**Not adopted:**
- Evidence-fit's Table A Zen4 column (7 power-of-two entries keyed by L1d < 40 KiB) would restore
  Zen4's 32, 64 and 1024, but it is a two-point fit that may encode the 10-02 code date.
- Its exception tables B-G (34 entries) repeat single races at the race's noise level.

## 6. Real 1D (r2c, c2r)

The door picks by parity (`il/real/real_create_il.h:30-34`). A real N plans like complex N/2.

- **Bounds:**
  - Refuse N < 2. N = 1 is refused because the odd door admits only N ≥ 3
    (`il/real/odd_build.h:50-54`).
  - Refuse even N with N/2 > 2^24, the ceiling of the c2c child and of zfsr.
  - Odd N has no ceiling beyond zrb's (below) and zrf's 10 stages.
- **Odd N:**
  - **zrm** when N is in the rn1 list and (N ≤ 17 or N is prime).
  - **zrb** when N is a prime cell. Its length m is the smallest 2^a·{1,3,5,7,9,15} ≥ N+(N−1)/2
    that an inner builds. Its inner is as in 4.6, with M < 2048 using a pair, then the il3p seed.
    Refused when (3N−1)/2 > 2^24 (`il/real/zrp_build.h:260-264`).
  - Otherwise **zrf**: the prime factors, 3·3 → 9, in ascending order. zrf needs 2 ≤ K ≤ 10
    (`il/real/zrf.h:228`), so above 10 stages the 4.4 merges apply (5·5 → 25, then 3·5 → 15,
    3·7 → 21), re-sorted ascending. Example: 29296875 = 3·5^10 becomes 3.5.5.5.5.5.5.5.5.25.
    The split body is off (row
    token `msz=0`, `il/wisdom/wisdom2_real_il.h:188,210`) when a middle stage is in {3,5,7,9,15}.
    The tile budget is 0 while 8N ≤ L1d, else the widest of {64..2048} with 16·b ≤ L1d/2.
- **Even N, the first that applies:**
  1. **zrm** at N ≤ 16 in the rn1 list.
  2. **zrp**, at 4 lanes only, for N ≤ 160 with an even × even pair: the most balanced pair,
     smaller factor first, form B (the r2z leaf).
  3. **zfsr** when N/2 is a power of two in 2^19..2^24 (`vfft_zfsr_band`, `il/real/zfsr.h:86-89`).
     It takes the split of 4.5 at N/2 and that split's child as `fs_*` tokens (4.5).
  4. **zttr**, at 4 lanes only, when N is a power of two in 512..524288, or N ≥ 2048 with N/2 in
     ZTT-r's grammar and 32 | N/2 (`il/real/zttr.h:113-143,180`). Chain:
     - power-of-two N/2, c2r: 4, then 8s, then 4s; r2c: 8, the 4s, then 8s, closing 4;
     - 2^a·odd N/2: the first power-of-two stage, the odd primes 7, 5, 3, the other power-of-two
       stages, closing 4;
     - tile: untiled below N/2 = 4096, else the first legal width of 2048, 3072, 1024, 512;
       `stk=3`.
  5. Otherwise **zr2c**: the child is this c2c procedure at N/2, natural. The route token is
     `child_nat_ip` in place and for c2r with N ≥ 1024, else `child_oop_il`. For c2r a pair
     child is mirrored, leaf and mid swapped, and then **both** `il_kv` and `il_bkv` are
     recomputed from 5.1 for the swapped slots. A c2r child runs only its backward forms
     (`wisdom2_real_il.h:382`), and `_il_dp_build` refuses a nibble with no kernel
     (`dp_planner_il.h:396-397`; `il/real/zr2c_build.h:36-37`). [M] Among the i9 c2r children
     whose pair route is hit, the mirrored recipe matches 129/175 (`synth/results/score.txt:229`).
- **At 8 lanes zrp and zttr do not exist** (`il_registry_avx512.h:477,1024,1029`; `zttr.h:164-172`),
  so even N goes to zrm, zr2c or zfsr. The r2c power-of-two column alternates with no size rule
  (child 7, zttr 5 of 256..524288); one zttr rule for both directions costs 1 row against the child.

## 7. Threads

The route is the one-thread route; T only adds the threaded arm. There is no thread-count model:
T is the caller's and a key of the plan (`nthreads=`). Every rule below was fitted at T = 8 only.
T = AUTO is not defined (see Open decisions). **Serial bound:** one transform stays serial
while its complex data fits two L1s, 16·N_c ≤ 2·L1d (N_c = N for c2c, N/2 for real): 6144 points on
the i9, 4096 on Zen4.

| engine | threaded arm above the bound |
|---|---|
| ZTURN-T, prime inner on ZTURN-T | `il_mt=2` (TILES) when tiled with ≥ 2 tiles, else 1 (BLOCKS) (`il/rank1/ztt_mt.h:13-17`) |
| flat DIT | TILES with `il_mt_tw = il_tw` when tiled, else BLOCKS |
| four-step | always threaded. `il_pair` stays the one-thread split; `il_mt` = the threaded split's N1_T: N/2048 up to 2^20, then 4096 (other N: the most balanced legal split, smaller column side). `il_mtsb=8.64.8` when natural, 16N > L3, N1_T = 4096 and N2_T % 16 = 0 (`k1_fourstep.h:955-958`); else 0. **The row's child is the threaded split's**, because it is seeded under (N1_T, N2_T, T) and replays only as "the verdict AND its child at T" (`k1_commit.h:1538-1552,1595-1620`); otherwise it races. So: `fs_chain` at N1_T, `fs_wl`/`fs_tf` by the L2 test at (N1_T, N2_T), the row plan at N2_T (4.5), the T = 1 switches, and the T > 1 tokens of 5.7 |
| zrf | `mt=2` (LEVELS) |
| zr2c | token `fold=mt` (the race arm `zr2c+foldmt`, `il/real/zrp_build.h:795-818`) when N ≥ 64 and above the bound, else `fold=st` (`wisdom2_real_il.h:7-8,434`). A non-prime child stays one-thread (`il/real/zr2c_build.h:38-41`). A prime child under `fold=mt` also takes its own threaded arm, `il_mt` on the row (`wisdom2_real_il.h:402-403`). That arm follows the prime-inner rule above at the child's M: 2 or 1 on a ZTURN-T inner, and absent (serial) otherwise |
| pair, chain3, mono, zrb, zrp, zttr | serial: the first three have no threaded form; zrb, zrp and zttr have no threaded row |

## 8. Expected quality: how often estimate names the banked winner

[M], `synth/results/score.txt`, each store scored with its own caches, stale rows excluded as in
EVIDENCE §0. "Route" is the engine; "whole" is every token.

**How far races agree (EVIDENCE §1, lines 95-115, 386-387).**
- **Sibling cells,** which are different cells: on the i9, natural out-of-place vs natural in-place
  races agree on the whole recipe 47% of the time (244/522), and natural vs scrambled out of place
  46% (637/1376). Pair order agrees 70% between out of place and in place. i9 vs Zen4: 32%.
- **The same cell:** the only true repeats are the Bluestein-inner races of one M, where chain plus
  tile agree 24-58%.
- [I] So 25-50% whole is about where races of nearby cells agree.

No row holds a losing arm's time, so what a miss costs is unknown (section 10).

**c2c, natural, out of place** (i9 / Zen4):

| N class | rows | route | shape (given the route) | whole |
|---|---|---|---|---|
| all | 3813 / 2047 | **91.6% / 97.4%** | — | 33.3% / 23.6% |
| primes ≤ 47 | 13 / 14 | 100% / 100% | — | 100% / 100% |
| powers of two | 22 / 11 | 22/22 / 8/11 | pairs 5/5 with `il_kv`; ZTURN-T 2048..262144 chain 6/8, tile 5/8; four-step split, `fs_chain`, `fs_wl` 4/4, switches 24/24, row child route 3/4 | 18/22 / 7/11 |
| odd band | 25 / – | 23/25 | chain 14/23, radix multiset 20/23 | 14/25 |
| smooth composites | 1343 / 850 | 76.2% / 94.1% | pair factors 97% / 97%, order 61% / 81%, `il_kv` 95% / 92%; chain3 triple 86% / 73%, order 42% / 22%; flat multiset 97% / 100%, chain 56% / 69%, forms 41% / 53% | 30.4% / 39.2% |
| composites with a factor > 47 | 1860 / 878 | 100% / 100% | inner kind at M ≥ 4096 100%, at M ≤ 2048 54% / 97%; inner chain at M ≥ 4096 59% / 9% | 32.6% / 4.3% |
| primes ≥ 53 | 550 / 294 | 100% / 100% | method 92.5% / 88.1% (documented verdicts 11/12, miss 1021); Rader inner kind 92% / 90% | 38.0% / 31.0% |

- **Other cells, route:** i9 scrambled out of place 89.2%, natural in place 91.3%, scrambled in
  place 83.8%; Zen4 scrambled out of place 97.3%. The scrambled power-of-two ZTURN-T chain at
  2048..262144 (i9) is exact in 4/8, its tile in 7/8.
- **Real (i9 only):** engine 3046/3067 r2c, 2055/2069 c2r (99.3% each), odd N 100%. The zr2c
  child's route 93% / 83%. zrf: radix multiset 95%, chain 53% / 59%, tile budget 84% / 86%. zrb
  length 65%. Whole recipe 37.0% / 27.5%.
- **Threads (i9, T = 8):** ZTURN-T verdict and `il_mt` 10/10; flat verdict 109/111; four-step
  `il_mt` and `il_mtsb` 4/4; zrf verdict 25/26 (r2c), 23/26 (c2r); zrb serial 14/14. The T row's
  child (new in this revision) was computed with the model's own functions at the threaded split:
  `fs_chain` 3/4, `fs_wl` 1/4, row route 3/4.
- **Legality: not established.**
  - [M] `synth/legal_synth.py` reported 0 illegal in 2,502,064 plans, but its laws had gaps:
    - it never checked `il_bkv` against the backward resolvers;
    - it never checked the chain3 `il_kv`/`il_bkv` nibbles;
    - it never checked that `il_mt` agrees with the four-step child at T > 1;
    - it skipped real N > 2^25 and N < 2, and sampled N > 300000 sparsely.
  - [M] An independent checker (`check/check_sketch.py`, `check/sweep_large.py`) found illegal
    plans in the earlier text:
    - 21/334 pair and 177/2457 chain3 `il_bkv`;
    - 117/315 c2r mirrored pair children;
    - the four-step at T > 1 wherever the threaded split differs from the one-thread split;
    - 2 flat chains and 1 zrf chain past 10 stages;
    - zfsr above N = 2^25, and real N = 1.

    This revision fixes each one in the text. A re-check against the updated model is still owed
    (section 10, step 6).

**Known weak spots:**
1. **The i9 flat wins over chain3 or a pair:** 291 cells. The share grows with the largest prime:
   11% at ≤ 13, 38% at 29..47. Zen4 has 39 such cells, and CPU and code date are mixed (EVIDENCE §2).
2. **Zen4:** ZTURN-T won 32 and 64 and the pair won 1024. Its convolution races ingest 4, so the
   inner chain hits 9%.
3. **Exact order:** chain3 order 22-42%; the ZTURN-T chain misses 2048 and 32768. The c2r zr2c
   route token is right in 65%, and the non-power-of-two four-step split in 0/7 documented cells.
4. **Evidence limits:** every 8-lane value is a legality mirror with no row behind it. The prime
   rules fit the September pool, before the smooth Bluestein lengths and the Rader-stage flat DIT.
   The thread rules rest on one CPU at T = 8.

## 9. MKL's rule against ours, decision by decision

The ratios are MKL time over the raced plan's (above 1 = ours faster), from `mkl/MKL_1D_RULES.md`,
which cites `docs/performance/v1_0_results.md`. They carry over to estimate only where estimate names
the raced plan.

| decision | MKL (rule) | estimate | measured |
|---|---|---|---|
| inputs | ISA tier, N, placement, strides; caches never shape the plan (2, 4) | the same, plus L1d, L2 and L3 read on the P-core | — |
| shape | ordered backend list, the first that accepts (5) | ordered tests on N | — |
| N ≤ 64 | per-N codelets, zsmallradix table (8, 24) | mono kernel list, then pair | ours median 1.52 |
| pow2 64..512 | `rowbatch` two-factor table at 128..2048 (6) | pair: mid 4, then 16 | level, 0.98-1.03 |
| pow2 1024..262144 | log4 N radix-4 stages plus one 8, no search: 5-9 stages (5 at 1024 and 2048, 6-9 from 4096; `mkl/MKL_1D_RULES.md:140-143`) (10) | ZTURN-T: radix 8 at the ingest and the sweeps, 4s inside the L1 tile: 4-7 stages (4 at 1024, 2048 and 4096) | 1.06-1.28; 0.97 at 32768 (an estimate miss) |
| tile | literal seeds (10) | `il_tw` from s1 = log2(L1d/16) | — |
| four-step | ≥ 64 Mi (pow2) / 16 Mi, or WORKSPACE=AVOID (12) | from 2^19 (pow2); other N above 262144; band width from L2 | 1.05-1.37 pow2; 1.99-3.97 other N |
| smooth non-pow2 | cascade, one stage per factor; trial division gives up past 88 (22) | pair > chain3 > flat over the 47-prime kernel set | 1.45-1.63 at largest prime 29..47; 0.92 at radix 13 |
| factor > 47 | a direct radix-p stage (22) | whole-N Bluestein | median 1.23 over 878 cells: parity near p = 53, 2x by p = 89 (`v1_0_results.md:773`) |
| primes | Bluestein past about 90 (23) | Rader where N−1 is a ZTURN-T length, or 23-smooth with N ≤ L1d/16, or N > 2^23 with a four-step split of N−1 | Rader cells 1.36-4.82 |
| one transform, T > 1 | serial unless the threaded via-2D kernel is installed; then round(√(B·ln B)/560)·2 threads (13-15) | serial while 16N ≤ 2·L1d; a threaded arm per engine; no thread-count model | 1.15-1.70 at 8192..262144 (0.98 at 32768) |
| hybrid cache read | whichever core runs first (30) | the P-core | — |
| real even | IPP real DFT ≤ 4096; complex N/2 plus a post-pass above (26-27) | zrp ≤ 160; zr2c (estimate at N/2 plus the fold) or zttr | pow2 r2c 1.11-1.51; MKL ahead for c2r at 2^20..2^22 (0.93-0.99) |
| real odd | IPP real DFT, never threaded (29) | zrf (threaded) or zrb | median 3.85 for odd smooth N; MKL ahead at a prime factor > 100 (0.62-0.68) |

## 10. How to validate it later (on a quiet machine; the owner runs it)

1. **The cost of a miss,** the one number missing (no row holds a losing arm's time).
   - **Method:** time estimate's plan against the banked winner in the same run, arms alternated:
     pin core 2, high priority, pace ≥ 200 ms, ≥ 15 rounds, report the spread and a control. Report
     the ratio per class. Threaded arms need their own protocol (the memory's MT parking trap).
   - **Cells:** ZTURN-T 2048, 4096 and 32768 (`14900KF/wisdom2_oop.txt:182,188,7362`), with 8192
     and 65536 (`:190,:7364`) as matched controls. A sample of the 291 i9 flat-won cells by largest
     prime (≤ 13, 17..23, 29..47). Pair and chain3 order misses. The 41 i9 Rader-or-Bluestein misses,
     including 1021 (`14900KF/wisdom2_prime.txt:719`). Bluestein inners at M = 4096 on Zen4 (mode
     4.8.4.4.8). Real r2c 1024, 2048, 4096, 8192, 32768 and 131072, where the child won.
2. **A second CPU, out of sample.** Race the classes the Zen4 store lacks: powers of two from 8192,
   N > 2047, real, and threads. Then score them with `synth/score_synth.py` without refitting. This
   is the test of the cache formulas: s1, the 2·L1d bound, L1d/16, the L2 band width and the L3
   super-band.
3. **AVX-512.** Rows from an 8-lane machine. Today every 8-lane plan is only legal, not fitted.
4. **Thread counts.** T = 2, 4 and 16 on the i9, against the 2·L1d bound (fitted at T = 8 only).
5. **Today's pool.** Re-race a sample of prime cells. The smooth Bluestein lengths and the
   Rader-stage flat DIT were never raced in c2c. The zrb rows pick a smooth length 65% of the time,
   a hint that M = next power of two is stale.
6. **After any store or pool change,** rerun `synth/score_synth.py` and `synth/legal_synth.py`.
   Both are read-only and take minutes. Before that, two things are owed:
   - bring `synth/synth_estimate.py` in line with this revision;
   - add the checker's laws (`check/laws.py`: backward resolvers, chain3 nibbles, `il_mt` against
     the four-step child, K ≤ 10, the real bounds, all smooth N above 262144) to the legality run.

## 11. Corrections to the three designs, and gaps noticed (reported, not filled)

- **cache-model:** its "zrf tile budget 94.6%" compares against a `tb` token no row carries (the
  721 zrf rows say `tile=`), so every estimate of 0 scored as a hit; the rule's real rate is
  84% / 86%. Its 8-lane refusal of 30 odd-band lengths comes from requiring both chain ends to be 8.
  Only the natural ingest and the scrambled last stage must be 8 (`ztt.h:361-364,398-410`; `tlf`
  exists at 4, `il_registry_avx512.h:1132`), and `policy_il.h:75-83` keeps only lengths that have a
  lawful chain in both classes.
- **evidence-fit:** it reads MKL rule 22 backwards (MKL loses 2x by p = 89, `v1_0_results.md:773`).
  Its Zen4 Rader count 251/294 is stale (three scripts give 259/294). Its model sends 200000 to the
  flat DIT, outside the odd band's pool (fixed here).
- **mkl-faithful:** the one-thread split as the threaded four-step split gives 2/4; the threaded
  rule gives 4/4.
- **Gaps:** no 8-lane row exists. For even real N at T > 1 only zttr and zfsr thread the transform,
  and estimate does not switch engines by T (no row). The documented threaded split at 2^19..2^21
  (2048, `v1_0_results.md:670-673`) disagrees with the store's rows (256 / 512 / 4096).

**Files** (this folder): `synth/synth_estimate.py` (the model); `synth/score_synth.py` +
`synth/extra_tallies.py` → `synth/results/score.txt`; `synth/legal_synth.py` and `synth/legal_wide.py`
→ `synth/results/legal.txt`, `legal_wide.txt`; `synth/hybrid.py`. `make_score.py` and
`make_legal.py` build the scorer and checker from evidence-fit's scripts. The independent check:
`check/` (`laws.py`, `check_sketch.py`, `sweep_large.py`, `results/`).

## 12. Open decisions (for the owner)

1. **Batches.** Answered in part (the owner, 2026-10-31): the contiguous batch on one thread is in
   scope, served as the plan of one transform, run once per transform (the door's transform-contiguous
   wrapper). Still open: the contiguous batch on more than one thread (the loop of one threaded plan
   against slabs of transforms, one per worker: `_tc_mt_decide`, vfft.c) and the lane-major 1D batch
   (`il/rank1/lm_batch_il.h`, `policy_il.h:232-258`), which is the real batch's DEFAULT geometry.
2. **Thread counts.** What should estimate do at T = AUTO, and at T other than 1 and 8? Every thread
   rule here was fitted at T = 8 on one CPU. The 2·L1d serial bound rests, for the flat DIT, on
   rows up to N = 2002 only (EVIDENCE.md:512).
3. **L1d outside 32-63 KiB.** There s1 moves the ZTURN-T tile to 512/1024 (s1 = 10) or 2048/4096
   (s1 = 12). Those widths are legal (`ztt.h:596-603`) but are off the raced ladder {1024, 2048}
   (`dp_planner_il.h:1455`) and no row tests them. On a 32 KiB L1d, W_hi = 2048 points is the
   whole L1d. Should estimate follow the formula there, or keep the ladder?
4. **The threaded four-step's child.** This text gives the T row the threaded split's own child,
   which is legal but rests on 4 rows (`fs_wl` 1/4). It emits `fs_sw=0` where the rows carry 8 and
   256. The other legal choice is the one-thread split at T, which scored 2/4 on `il_mt`. Which
   should estimate use?
5. **Prime-pool members estimate never emits:**
   - Bluestein's smooth lengths (`policy_il.h:304-331`);
   - the Rader-stage flat DIT (inner kind 5, `k1_commit.h:353-410`);
   - Rader with the four-step inner below 2^23.

   No c2c row has raced the first two. Is leaving them out an accepted compromise, or should
   estimate emit them? The same choice decides whether the remaining primes in (2^23, 2^24] stay
   refused.
6. **In-place four-step.** Estimate uses the out-of-place split. The four in-place rows chose other
   splits (`wisdom2_oop.txt:10260-10263`) but are stale, with no child. Keep this until those rows
   are re-raced?
7. **AVX-512.** Every 8-lane plan here is legal but fitted to no row. Ship the 8-lane estimate on
   legality alone, or wait for 8-lane rows (section 10, step 3)?
