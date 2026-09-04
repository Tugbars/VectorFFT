# Measurement arms: what races, per transform x layout x order

> **Read [`planning_model.md`](planning_model.md) first.** This file is the terse
> lookup catalogue; it assumes the vocabulary and the structure that document builds.
> Every term used here — stage, radix, chain, twiddle, FLAT/LOG3/T1S, DIT/DIF, leaf,
> mid, turn, tape, arm, verdict, bank, cell, key, hysteresis — is defined there, with
> diagrams. If a record below reads as jargon, the explanation is in the model doc, not
> missing.
>
> | you want to | read |
> |---|---|
> | understand how planning works | `planning_model.md` §1-§7 |
> | understand one engine family | `planning_model.md` §8-§14 |
> | look up one axis' arms / race site / bank key | **this file** |
> | go from a fingerprint field back to an axis | **this file**, §9 |

**What this is.** The catalogue of every decision the library makes by *measuring*,
which configurations reach each one, and where the verdict is stored. A plan is not
produced by one search; it is assembled from up to six independently-banked verdicts
across three shards, raced at different times under different validity scopes.

**Why it exists.** Reconstructing this from the code costs days and gets the same things
wrong every time. Both failure modes are recorded here so they are not repeated: claims
about *reach* ("which cells hit this axis") were wrong far more often than claims about
*mechanism* ("where the clock is"), and roughly one axis in six turns out not to be a
measurement at all.

**Status of the contents.** Mechanism fields are read from source. `WITNESS` and
`[measured]` marks come from `build_tuned/benches/harness_grid_probe.c` and its
dark-field siblings, one process per cell against a fresh seeded store. Where a prose
claim and a measurement disagree, **the measurement wins**.

---

## 0. How to read an axis record

```
### <axis name>
STATUS     RACED | RACED-NOT-BANKED | DERIVED | ENV | STRUCTURAL
WHEN       create | offline (calibrate_*) | both
ARMS       what competes against what
GATED BY   the axis whose outcome decides whether THIS axis exists
GATES      the axes whose existence THIS one decides
RACE       file:line of the clock
BANK       shard + key axes + payload token, or "none"
CALLER     the caller-visible config that reaches it
WITNESS    a config measured to exercise it, and the observed value
FP         fingerprint field, or "none" (= invisible to the safety harness)
```

### Status vocabulary

| status | meaning |
|---|---|
| **RACED** | a real clock decides between arms, and the verdict is banked |
| **RACED-NOT-BANKED** | measured every create, never persisted: re-decides per process |
| **DERIVED** | computed from another axis' verdict; no arms of its own |
| **ENV** | an A/B knob only; env pins never bank (the tcut law) |
| **STRUCTURAL** | fixed by construction or by a refusal; no measurement |

### Three banking classes

1. **Banked, key-matched** - replays deterministically.
2. **Banked with validity conditions** - carries the conditions it was raced under; a
   mismatch **re-races** rather than serving. `cmt` + `cmtt` is the worked example.
3. **Plan-local** - raced every create, every process.

Owner ruling on MT specifically:

* **one transform per core** (TC batches, plane queue) - nothing about the plan depends
  on T, so bank **T-free**, valid at any T.
* **cores sharing one transform** (2D column passes, the cascade walk) - T decides how
  the work is *cut* (band counts, worker clamps, legal row widths), and a cut that wins
  at T=2 can lose at T=8. Bank **per-T**; only this class re-races on a T mismatch.

Thermal robustness comes from the race SHAPE, not from refusing to bank: arms are
alternated within one run, so *which arm wins* is robust even though the absolute
nanoseconds are not. Banked ns are informational; the DECISION is what is banked.

### Two traps this document exists to prevent

**A store key is a PLAN key, not a caller config.** The store holds
`t=r2c n=64x64 q=1 ord=scr place=oop lay=il`, but a caller passing
`VFFT_ORDER_SCRAMBLED` for 2D r2c is **refused**. Never infer a caller config from a
banked key. `CALLER` and `BANK` are separate fields for exactly this reason.

**Absence of a `lay=` axis does not mean "not banked for that layout."** Only 27 of 539
shipped records carry `lay=`; the rest are pre-1.2 and serve as a fallback tier. For K=1
OOP the reader deliberately scans `lay=IL`, `lay=SPLIT` *and* lay-less cells and fills
both route axes regardless of the caller's layout - which is why a split caller and an IL
caller at the same N report identical `sp=` / `il=`.

---

## 1. Which tournament set am I in?

Layout and order do not modify a shared set of tournaments - they **select** the set. IL
scrambled and IL natural share almost none.

```
transform -+- C2C --+- 1D -+- SPLIT -+- in-place --- proto-stride nest        (S1)
           |        |      |         +- OOP -------- K=1 planner / classic    (S2)
           |        |      +- IL ----+- N <= 64 ---- mono                     (B0)
           |        |                +- 128..1024 -- Bailey 2-stage pair      (B1)
           |        |                +- N >= 2048 -- boundary-split cascade   (C1)
           |        +- 2D -+- SPLIT -------------- strided codelet family     (D1)
           |               +- IL ----------------- il2d tier                  (E1)
           +- REAL -+- 1D --- route race: K < 32 vs K >= 32                   (R1)
                    +- 2D -+- SPLIT -------------- calibrated vs fallback     (D2)
                           +- IL ----------------- il2d REAL tier             (E2)
```

**Placement is a different path, not a flag.** Out-of-place engages the K=1 planner
(`k1=1`, with `sp=` / `il=` set); in-place never does - every in-place cell reads
`k1=0 sp=0 il=0` regardless of layout. [measured, 91 configs]

**In-place real is accepted only for 1D interleaved.** Refused for 1D split and for all
2D, both layouts. [measured]

---

## 2. 1D C2C - SPLIT

### S1. The proto-stride nest (in-place; also the OOP fallback)

One nested tournament, **not four independent ones**. Two passes:

```
pass 1  shortlist chains       DP top-K multisets -> permutations,
                               coarse-benched at FIXED T1S/DIT
pass 2  for each surviving chain
          for orient in {DIT, DIF}
            for each per-stage variant tuple    <- iterator is orientation-specific
              bench REFINE_RUNS, keep min
        single argmin over (chain x orient x variants)
```

Consequence worth knowing: a chain that is mediocre under T1S/DIT but would win under
DIF or log3 is **pruned in pass 1 and never reaches the variant search**. The search is
exhaustive over variants *given* a chain shortlist chosen at one fixed variant.

#### S1.1 chain + stage ordering

```
STATUS     RACED
WHEN       create (VFFT_MEASURE) / offline
ARMS       every factorization multiset x every unique permutation, each built and
           timed whole-plan at the caller's K
GATES      S1.2, S1.3 (both search *within* a chosen chain)
RACE       src/core/planning/dp_planner.h:450 (_vfft_proto_dp_bench)
BANK       wisdom2_scr | t=c2c n=N q=K ord=scr place=ip | token chain=
CALLER     1D c2c split, any placement
WITNESS    1d.sp.ip.c2c.256 [measured: replays, races=0, no fp field set]
FP         none
```

#### S1.2 orientation DIT vs DIF

```
STATUS     RACED (a coordinate of S1's argmin, NOT a separate head-to-head)
ARMS       use_dif_forward = 0 vs 1, as an outer loop around the whole variant
           cartesian. DIF has no T1S (it de-dups to FLAT)
GATED BY   S1.1 (runs per surviving chain)
RACE       src/core/planning/measure.h:474 (the orient loop); winner via a single
           `if (ns < best_ns) best_use_dif = orient`
BANK       same record | token dif=
FP         none
```

There is **no orientation race site to preserve** - break the argmin and the orientation
silently pins.

#### S1.3 per-stage twiddle variant

```
STATUS     RACED
ARMS       cartesian of FLAT / LOG3 / T1S over the TWIDDLED stages only; the
           no-twiddle stage (stage 0 for DIT, nf-1 for DIF) is pinned FLAT
GATED BY   S1.1 and S1.2 (the iterator is orientation-specific)
RACE       src/core/planning/measure.h:253 (_vfft_proto_dp_variant_search)
BANK       same record | token vars=flat.t1s.t1s
FP         none
```

#### S1.4 pad_me - pad vs tail

```
STATUS     RACED (genuinely separate from S1.1-S1.3)
ARMS       TIGHT  = (N,K) executed at me=K, narrow SSE2/scalar tail
           PADDED = aligned (N,Kp) at me=Kp, Kp = roundup(K,8)
RACE       src/core/vfft.c:841 (_pad_burst); driver _calibrate_pad at :850
BANK       wisdom2_scr | token pad_me=
CALLER     K > 1
FP         padded=
```

#### S1.5 exhaustive sweep

```
STATUS     RACED, rigor=VFFT_EXHAUSTIVE only
ARMS       every coverable multiset, every unique permutation, a T1S pre-screen,
           then the full 3^(nf-1) variant cartesian per surviving ordering
```

### S2. SPLIT out-of-place

#### S2.1 sp_route x sp_pair (K=1)

```
STATUS     RACED
WHEN       OFFLINE ONLY - vfft.c never includes dp_planner_split_oop.h; the only
           caller is build_tuned/benches/calibrate_k1.c:70. At create it REPLAYS.
ARMS       10 routes x every legal pair (R1,R2), both %4==0, both in [4,128]:
             MONO family       mono, mono-alt   (the whole-four-step kernel)
             Bailey multipass  3p, 2pa, 2pb, twl, 3p-l3, 2pa-l3
                               differ in WHERE the transpose is paid: a separate
                               pass / in loads / in stores; _l3 = log3 twiddle
             CCOL              composed column pass, batch-engine
           Each candidate is correctness-gated BEFORE it is timed.
RACE       src/core/planning/dp_planner_split_oop.h:510
BANK       wisdom2_oop | t=c2c n=N q=1 ord=nat place=oop role=comp lay=split
           | eng=k1 sp_route= sp_pair=R1.R2 [chain= vars= when ccol]
CALLER     1D c2c OOP K=1, EITHER layout (see the lay= trap in section 0)
WITNESS    1d.il.oop.c2c.64  -> sp=4 (MONO)  [measured]
           1d.sp.oop.c2c.256 -> sp=2 (2PB)   [measured]
FP         sp=   (sp_pair, chain and vars are NOT in the fingerprint)
```

#### S2.2 CCOL inner column chain

```
STATUS     RACED (proposals only - S2.1 decides)
ARMS       proto DP at (N=R2, K=R1), DIT-only, cc-encodable, deduped, <= 3
BANK       no own cell; rides S2.1's record as chain= / vars=. The proposer's
           isolated timing is explicitly NEVER banked.
CALLER     R1 in {8,16,32,64} dividing N, with R2=N/R1 >= 16 and R2%4==0
NOTE       CCOL is the ONLY K=1 split route above N=16384 - no classic pair exists
           once R1 or R2 exceeds 128.
```

#### S2.3 OOP pair tuner (K>1)

```
STATUS     RACED
ARMS       up to 34: the direct LEAF codelet (N<=128) plus, for every divisor R2 of
           N from min(N,128) down to 2, the pair plan at BOTH t1p variants
           (flat = FMA-leaner, log3 = port-rebalanced)
RACE       src/core/oop/oop_auto.h:143 (__rdtsc, 15 rounds, per-candidate min)
BANK       wisdom2_oop | t=c2c n=N q=bK ord=nat place=oop
           | eng=classic route=bailey2 chain=R1.R2 t1p=flat|log3
CALLER     K % 8 == 0 (refused otherwise at oop_auto.h:52)
FP         none
```

#### S2.4 classic kind: native vs MODEB (K>1)

```
STATUS     RACED - but the ORDER overrides the clock
ARMS       native champion (S2.3's winner) vs the DP-planner MODEB champion, both
           built for (N,bK), __rdtsc min-of-9 on the same buffers
RACE       src/core/oop/oop_dp.h:136 and :146
BANK       both champions banked when bK % 8 == 0; a K%8!=0 champion is measured
           and NEVER banked
```

The measurement decides only under DEFAULT order (`vfft.c:8919-8945`):

```c
if      (ord == NATURAL)   { op = nat; }   /* clock ignored */
else if (ord == SCRAMBLED) { op = mb;  }   /* clock ignored */
else                       { op = (nns <= mns) ? nat : mb; }
```

### S3. SPLIT natural order

#### S3.1 reorder mode - @nat mode=pcyc|pswap|scr

```
STATUS     RACED - and UNBANKED at the probed cell
ARMS       PURE_CYCLE floor (the deployed chain + its cycle tape) vs up to 5
           injected palindrome chains (uniform T1S, DIT) each with a paired tape;
           plus SCR
RACE       vfft_natorder_race, src/core/transforms/natorder/natorder_calibrate.h
BANK       wisdom2_oop | ord=nat, via _bank_nat_1d / _bank_natoop_1d (rehomed 2026-09-02)
WITNESS    1d.sp.ip.c2c.256 order=NATURAL
           [measured: FLAPS - nat=5/natcyc=96 in 8 of 10 runs, nat=4/natcyc=34 in
            2. Both outputs CORRECT: 3.0e-16 and 3.1e-16 rel err vs a naive
            long-double DFT. Different radix chains, last-bit differences.]
FP         nat= natcyc=
```

---

## 3. 1D C2C - INTERLEAVED

### B0/B1. The three N bands

IL 1D is **three different engines**, not one with parameters:

| band | engine | contract |
|---|---|---|
| N <= 64 | MONO: the SOLO kernels (one call, natural order), raced against the pairs where both exist | any N in the solo radix set (2..64) |
| 128..1024 | pure IL Bailey 2-stage pair | interleaved everywhere, count % 2 == 0 |
| N >= 2048 | boundary-split cascade | z at the boundary, split planes inside, count % 4 == 0 |

The cascade converts layout **exactly twice** (ingest, terminator) and runs the interior
split, because a complex multiply on split data is 4 real multiplies with zero lane ops.
That only pays when there are many interior stages to amortize over - which is why the
tier starts at 2048 and the Bailey pair below it stays interleaved. Full-IL interior has
been refuted twice.

### B0.1. MONO tier - the SOLO kernels and the mono FORM axis

```
B0.1 mono form            RACED, BANKED (2026-09-04). A solo kernel is the whole
                          N-point transform in ONE call: the pure-IL n1 kind
                          (gen_radix R --cil-n1 [--cil-bwd]), natural order in and
                          out, twiddle-free, one leg (Ls = OLs = 1, count = 1: the
                          VEX-128 tail IS the transform). Emitted at every radix the
                          family has: 2 3 4 5 6 7 8 9 10 11 12 13 15 16 17 19 21 25
                          27 32 64, both directions.
     FORMS   0 = radixN_z_n1 (every N in the set); 1 = mono64_8x8_il (N = 64).
             Each form is its own planner candidate (route MONO, il_kv = form)
             beside the pairs and chain3; the measurement decides.
     PLACE   out-of-place serves the __restrict__ n1 kernels; IN-PLACE serves the
             alias-tolerant n1c twins (same math, no restrict, z -> z legal by
             construction; n1c at 2/6/10/12 was emitted for this door). Both
             forms map onto n1c in place.
     BANK    the cell's kind-3 row: il_route=mono il_kv=<form>; in place the mode
             row (@scrmode/@nat) says mode=ilp and the kind-3 row names mono.
     VERDICT on this box (cold, 2026-09-04): mono at 2..11, 13, 17, 19 and 9;
             the pair at 12, 15, 16, 21, 25, 27, 32, 64 (the solo body at 16
             measured 12.4 ns vs 4x4 8.4 ns). The primes 5/7 left Bluestein
             (43/76 ns -> 11/10 ns); 6/8/10 left their awkward-composite
             routes (75/34-78/133 ns -> 12/10/12 ns).
     REACH   1D c2c K=1 both placements; the zr2c child at N/2 (real N = 4..);
             the 2D real row door at N2 = 4/6/8 (il2d_real 16x8, 128x8, 4096x8
             were red until this landed); K>1 batches over the K=1 plan.
```

### B1. Bailey pair - route

```
STATUS     RACED — offline (calibrate_k1) AND at create since 2026-09-03: a kind-3
           MISS or recalibrate below 2048 runs the same planner from the in-place and
           out-of-place K=1 creates (_k1_il_plan_race in k1_commit.h) and banks
ARMS       route enum VFFT_K1_IL_*:
             3 MONO      the SOLO tier (B0.1): radixN_z_n1 at 2..64 + mono64 8x8 (form axis)
             5 2P_PURE   the Bailey pair                 <- this tier
             6 CHAIN3    3-stage chain (odd factors as kernel RADICES) —
                         RACED since 2026-09-02: every legal (R2, A, B)
                         enters the natural pool; banked as il_chain=R2.A.B.
                         Its per-slot kernel FORMS race too since 2026-09-03
                         (B1.2: the same pools, three nibbles A|B<<4|leaf<<8
                         in the row's il_kv; replayed by both create sites;
                         no backward cell yet — see B1.3)
             7 PRIME     ilprime (see B4)
             4 CASCADE   the >= 2048 tier (see C1)
BANK       wisdom2_oop | t=c2c n=N q=1 ord=nat place=oop role=comp lay=il
           | eng=k1 il_route= il_pair=R1.R2 [il_kv=]
FP         il=   (the PAIR has no fingerprint field)
```

#### B1.1 pair enumeration

```
ARMS       R2 from VFFT_IL_N1T_PAIR_RADICES, derived from the generated registry
           (widened 2026-08-23 from a hardcoded {4,8,16,32,64});
           R1 = N/R2, bounded 3 <= R1 <= 64.
           ORDERED: (R1,R2) and (R2,R1) are distinct plans, covered by
           construction - no permutation pass.
GATES      B1.2 - which form arms exist depends on the chosen R1 and R2
CAP        VFFT_IL_DP_MAX_CAND = 1024 per (N, ord); _il_dp_push REFUSES the cell
           outright past the cap rather than truncating - a truncated pool is a
           BIASED pool.
NOTE       There is NO pow2 test on R1. Removing it is what let non-pow2 cells
           enumerate candidates at all; the leaf/mid existence check is strictly
           tighter.
```

#### B1.2 il_kv - forward kernel form (mid nibble | leaf nibble << 4)

```
STATUS     RACED, offline
GATED BY   B1.1 - the arm sets are per-radix
```

| R1 (mid) | default | arms |
|---|---|---|
| 32 | 2 (`t2b48`, 4.8) | 2, 1 (`t2b` 2.16), 3 (tangent wing32), 4 (`t2bw32` M-128) |
| 64 | 2 (8.8) | 2, 1 (4.16) - no tangent forms exist at R64 |
| 16 | 0 (monolithic) | 0, 3 (tangent), 4 (`t2tan` M-128), 1 (`t2b16`) |
| 8 | 0 | 0, 3 (tangent) |
| odd | 0 | 0, 5 (`_ct`) - only if the resolver has the kernel |

| R2 (leaf) | default | arms |
|---|---|---|
| 32 | 2 | 2, 1, 3, 4 (`n1tbw32` T256 turn-edge) |
| 64 | 2 | 2, 1 (4.16) |
| 16 | **0 (monolithic)** | 0, 1 (4.4), 3 (tangent) |
| 8 | 0 | 0, 3 (tangent) |
| odd | 0 | 0, 5 (`_ct`) if the resolver has it |

Cross product **minus the (default, default) combo**, which is already the base
candidate. The per-radix pools are ONE source (il2p.h vfft_il2p_mid_arm_pool /
leaf_arm_pool since 2026-09-03), shared with the CHAIN3 route: there every legal
chain enumerates A x B x leaf from the same pools (full cross product up to 16
combos, else one slot at a time), banks the winner as a three-nibble il_kv on
its chain3 row, and the chain3 create applies the pair's blocked default to
every slot at R >= 32 (mids included since 2026-09-03).

```
NOTE       R2=64 MUST be raced: 4.16 won at count 8 (+19.2%) and 16 (+16.9%),
           8.8 won at count 32 (+61.3%). A structural rule would silently take
           4.16's cells.
           R=16 is deliberately NOT in the blocked-by-default rule - it fits the
           register file (8.6% spill), so a non-monolithic form must win per cell.
           Nibble 0xF = FORCE MONOLITHIC, so a platform where blocked loses stays
           expressible as a banked verdict.
BANK       rides B1's record | token il_kv=
FP         NONE - two plans differing only in leaf/mid form fingerprint
           IDENTICALLY. Invisible to the safety harness.
```

#### B1.3 il_bkv - backward kernel form

```
STATUS     RACED, offline; a separate cell at dir=bwd
ARMS       {0 = whatever create installed} U {1..5} minus each slot's default,
           capped at VFFT_IL_DP_BKV_MAX_ARMS = 24; roundtrip-gated BEFORE timing
NOTE       backward uses a different decomposition: t2t_b at radix R1, then
           n1_b_r2 at radix R2 - NOT R1. Using n1_b there measured 1.1e+00.
BANK       wisdom2_oop | ... dir=bwd ... | metric=bwd1
FP         none
NOTE       bkv == 0 ("the defaults won") banks as an explicit il_kv=0 row since
           2026-09-02; the reader's -1 means "no row" (the old KNOWN GAP).
CHAIN3     since 2026-09-03 the 3-stage chain has its own dir=bwd cell too:
           only its LEAF slot has backward twins (n1_bwd_v_fn), so the pass
           races leaf-only arms, packs A=B=0 | leaf<<8, and the row carries
           il_chain=R2.A.B beside il_pair for the replay's validation
           (vw2_oop_lookup_k1_bwd_chain). The chain3 create installs the
           blocked backward leaf at R2 >= 32 as the pair does. The mids
           (t2 bwd, t2tg) have no rival forms: nothing to race there.
```

#### B1.4 pair-order swap

```
STATUS     RACED, BANKED (2026-09-02): the winner IS the pair order, so it
           banks as the cell's kind-3 lay=il row (il_route=2p il_pair=R1.R2,
           measure-less) and the existing pair replay (ke->il_R1) serves it;
           the offline planner's measured row replaces it. Runs only for a
           cell with NO banked pair; VFFT_NO_T2B pins, never banks.
WHEN       create
ARMS       vfft_il2p_execute_fwd on the heuristic (R1,R2) vs on
           vfft_il2p_create(N, R2, R1). 5 bursts, min-of-5, 3% hysteresis
FP         none
```

### B2. IL in-place create - the IL plan race (no split baseline since 2026-09-03)

```
STATUS     RACED, BANKED, at create
LAW        owner 2026-09-03: "we DO NOT see split as a fallback of IL". An
           in-place interleaved caller never gets a split plan. The cell is
           served by an IL engine and the only question is WHICH IL plan wins.
           Before this the create built the classic split plan first and raced
           the IL engine against that split-behind-convert incumbent; whenever
           its one candidate builder came back empty the convert served — 177
           of the 255 sizes below 257 executed through the convert in place
           while the same kernels served 252 of them out of place.
ARMS       N < 2048: the IL PLAN RACE (_k1_il_plan_race -> vfft_il_dp_plan_and_bank,
           the planner calibrate_k1 runs offline): every legal pair x its kernel
           forms, every legal 3-stage chain x forms, the order swap, the backward
           forms; verdict = the kind-3 lay=il row (+ its dir=bwd row), replayed
           on every later create. Runs on a kind-3 MISS or recalibrate; logs on
           entry (a cold cell takes seconds). Prime N: B4's method race.
           N >= 2048: the cell's K=1 IL engine (pair/chain3 from its kind-3 row)
           vs the cascade (kind-4 recipe; natord under NATURAL) — aliased z->z
           arms, 5 rounds alternated, median-of-5, reps 200/80/32; one arm
           serves and banks; NO arm = the create REFUSES (nothing to fall back
           to, by design).
RACE       src/core/oop/c2c_ip_create.h (_c2c_ip_create_il) and
           src/core/oop/k1_commit.h (_k1_il_plan_race). K=1 ONLY: an explicit
           lane-major K>1 interleaved request is REFUSED (it lost to
           transform-contiguous at every measured cell and its only engine was
           the split K-lane plan behind a convert); DEFAULT geometry never
           arrives here (the transform-contiguous wrapper over a K=1 inner).
DIRECTION  forward executes; the verdict serves both directions (structural by
           measurement, see the 2026-09-03 flip probe: no cell flips).
BANK       @scrmode (ord=scr place=ip lay=il, DEFAULT/SCRAMBLED) or @nat
           (ord=nat place=ip lay=il, NATURAL) | mode=ilp | mode=zcasc — a
           mode=zcasc row signposts the recipe that served (ref=cell(...,
           place=oop[,role=comp])); mode=ilp rows are self-contained. mode=conv
           and the tape modes are NOT IL verdicts any more: a row carrying one
           re-races. The convert machinery (deinterleave/split/reinterleave,
           il_me pad A/B, the OOP convert executor, the il2il executors) is
           DELETED from the library.
CALLER     1D c2c IN-PLACE interleaved, any order.
WITNESS    cold 256: "[k1plan] N=256: ilp=134ns -> ILP (route 5, 16.16)" then
           "replay ILP"; cold 2048: "race: ilp=2400ns zcasc=2072ns -> ZCASC"
           then "replay ZCASC" [measured 2026-09-03].
FP         the VERDICT shows as have[] bits 5/6/7 (k1il2p / k1il3p / k1ilpr) or
           zroute=1; the ilme/ilrace fields are gone with the machinery.
```

### B3. IL natural order

```
B3.1 nat-ilp        il2p/il3p vs the finished natural-TAPE handle    RACED
B3.2 nat-zcasc      natord cascade vs the tape handle                RACED
B3.3 natoop-zcasc   natord cascade vs the K=1 OOP handle             RACED
B3.5 scroop-zcasc   SCRAMBLED cascade vs the K=1 OOP handle, order=  RACED, BANKED
                    DEFAULT, N >= 2048 (2026-09-03). DEFAULT is order-
                    agnostic, so the scrambled cascade is a legal arm --
                    it was never offered (built only for an explicit
                    SCRAMBLED request): DEFAULT OOP served the pair at
                    2048/4096 and the classic champion behind a convert
                    above (4.7x at 8192, 7x at 16384). The pair beats
                    the cascade at 2048 and loses at 4096, hence a race,
                    not a rule. Row: t=c2c n=N q=1 ord=scr place=oop
                    lay=il | mode=zcasc (ref= the kind-4 verdict) |
                    mode=free (the engine won). VFFT_NO_NAT_ZCASC pins.
B3.4 nat-tape opportunistic PSWAP                                    STRUCTURAL
     -> no clock at all; a deterministic short-circuit that still BANKS.
        Belongs on a "banks without measuring" list, not among measurement arms.
```

### B4. IL prime

```
STATUS     RACED, BANKED (2026-09-02): a component row in the PRIME shard,
           t=c2c n=N q=1 ord=scr place=ip role=comp lay=il | eng=rader|bluestein;
           every ilprime create goes through _ilprime_create_banked (replay on a
           hit, race + bank on a miss); VFFT_ILPR_METHOD pins, never banks.
ARMS       _ilprime_create_rader(N) vs _ilprime_create_bluestein(N), both fully
           constructed, warmed once each, then min-of-3 ALTERNATED forward executes
INNER      B4.1 (2026-09-02): the convolution's inner transform at length M
           (N-1 Rader / pow2 Bluestein) is SERVED by the K=1 IL pair machinery
           (_k1_il_candidate: kind-3 pair replay with kernel forms, else the
           pair race + bank) through the engine's inner-provider hook, and the
           prime row signposts it: ref=cell(t=c2c,n=M,...,role=comp,lay=il).
           The create races the PAIR only; the kernel FORMS (il_kv) and the
           backward cell come from calibrate_k1 at the inner length, as for
           every kind-3 row (owner ruling 2026-09-02: the IL convention, not a
           create-time form race). SWEEP ITEM: run calibrate_k1 at N-1 for
           every prime cell in the store and at every Bluestein M it uses.
           The cascade inner (pow2 M > 4096) replays the kind-4 recipe through
           _k1z_wisdom_replay and the prime row signposts that row (2026-09-02).
           A CHAIN3 inner replays the banked il_chain=R2.A.B (B1 route 6, raced
           by the planner since 2026-09-02). Nothing structural is left in the
           prime engine's inner except the engine's own fallbacks on a miss.
RACE       src/core/oop/il_prime.h:385-428
NOTE       An earlier claim that the prime METHOD is "never raced" was REFUTED by
           verification - this race is real.
DIRECTION  forward executes only; one plan serves both directions. STRUCTURAL
           BY MEASUREMENT (2026-09-03 flip probe at 127/131/251/4093, banked
           method vs the other pinned): the fwd and bwd ratios are IDENTICAL to
           two digits at every cell (the two methods do the same work in both
           directions - conjugated twiddles, same inner), so no per-direction
           method row can ever differ from the forward one.
           ⚠ the probe also read 4093: banked BLUESTEIN loses to a pinned Rader
           by 7% in both directions - a stale-magnitude concern for the
           pre-release sweep, not re-raced during development.
```

---

### B5. K>1 transform-contiguous batch - the THREADING verdict

```
B5.1 tcmt - serial loop vs slabs   RACED, BANKED (2026-09-04). The K>1
                                   interleaved tier is a wrapper: K K=1 transforms
                                   end to end over the store-served K=1 inner (the
                                   plan comes from the inner's own arms; lane-major
                                   is REFUSED, so geometry is not an axis). Its one
                                   arm is threading: the serial loop vs slabs of
                                   ceil(K/T) transforms over per-worker clones.
     RACE    src/core/vfft.c (_tc_mt_decide): min-of-R alternated, placement-
             honest buffers (in-place aliased + reseeded), the batch's own cell.
             No clones (no pool, inner not pool-free, K=1 workers) => no arm,
             serial by construction.
     BANK    the batch's OWN row: t=c2c|r2c|c2r n=N q=K ord=scr|nat place=ip|oop
             lay=il | eng=tcb tcmt=0|1 tcmtt=<T raced at>. T-FREE: one transform
             per core, nothing about the plan depends on T (planning_model 'The
             MT rule'); tcmtt is provenance. eng=tcb fails every natx family gate,
             so the row can never be served as a plan.
     ENV     VFFT_TCMT=0|1 pins (never replays, never banks); VFFT_NO_TCMT = no
             clones at all (kill switch); VFFT_TCMT_LOG / VFFT_TCMT_VERBOSE log.
     RETIRED the 2048-complex-point scalar floor (an offline 2026-08-22 table,
             never a verdict) and its VFFT_TCMT_FLOOR knob.
```

## 4. 1D C2C - the cascade (N >= 2048)

### C1. The tournament tree

The cascade is the clearest instance of **axes gating axes**:

```
route (zturn vs zsplit)
  |
  +- chain / factorisation over {4,8}
       |
       +- chain[nf-1] decides the TERMINATOR FAMILY
            |
            +- last==8  -> stf / stf2       -> t2q is RACEABLE
            +- last==4  -> radix4_z_stf_r4  -> t2q FORCED 0 (no stf2@r4 twin;
            |                                  _calibrate_zturn_t2q refuses it)
            +- natord   -> stfn / stfbn     -> t2q structurally ignored
                                               AND tfuse FORCED 0
```

`zsplit_create` builds only chains ending in 8; ZTURN builds `last==8` **and**
`last==4`, so ZTURN is a strict superset - which is why zsplit survives as the control
arm and must not be deleted.

```
C1.1 route zturn vs zsplit      RACED (offline). Runtime is ZTURN-only since the
                                2026-07-27 cutover; zsplit is the control arm.
C1.2 chain / factorisation      RACED. Ordered {4,8} chains, nf in [3, MAX_NF],
                                each validated by ITS OWN engine's create.
                                GATES the terminator family and C1.4.
                                ODD MIDS (N = 2^a * odd, 2026-09-02): the odd
                                part decomposed into msg radices {3,5,7,9,15}
                                at every interior position x ordered {4,8} for
                                the rest; same validation, same width and t2q
                                axes (one helper). Banked as the role=comp
                                recipe: the problem-verdict key at odd N stays
                                the OOP cell's own winner (an odd cascade races
                                the finished handle at the commit, never by fiat).
C1.3 terminator kind            DERIVED from (chain[nf-1], natord). Never an
                                independent arm.
C1.4 t2q single vs 2-quad       RACED **only if** last==8 AND not natord.
                                Bit-identical twins, memcmp-gated, so the delta is
                                pure code placement - measured per cell, never
                                hand-set.
C1.5 tcut width (zt_tw)         RACED (offline). UNTILED (kept so "tiled" stays
                                falsifiable) vs every legal width. Odd mids
                                included since 2026-09-02 (C1.2).
C1.6 tfuse                      DERIVED; FORCED 0 under natord (rho spans the whole
                                section, so a per-tile cut cannot exist).
C1.7 thonest                    ENV only (VFFT_TCUT_TW). Bit-identical pair, kept
                                reachable as F1's discriminator.
C1.8 natord on/off              Decided by the B5 natorder race, NOT by the env hook
                                (VFFT_ZT_NATORD is a probe/gate hook).
C1.9 zt_mt serial vs threaded   RACED, BANKED PER-T (2026-09-02): the verdict
                                rides the recipe row that served the cascade as
                                zt_mt_t=<T> zt_mt=<0|1>; a T match replays, a
                                mismatch re-races and re-banks; a re-raced
                                recipe drops it. VFFT_ZT_NO_MT (env) never
                                replays and never banks. BOTH exits ask: the
                                in-place exit joined 2026-09-02 with aliased
                                z->z arms and its own pair (zt_mt_ip_t /
                                zt_mt_ip), so the two placements' verdicts never
                                overwrite each other. natord cascades cannot
                                engage and bank the "no" implicitly.
```

WITNESS: `1d.il.ip.c2c.4096` -> `zroute=1`, and it RACES [measured].
`1d.il.ip.c2c.4096` order=NATURAL -> `zroute=1 nat=6`, and it REPLAYS [measured] -
different terminator family, different banked cell, different lookup outcome.

---

## 5. 1D REAL

### R1. The route race

```
STATUS     RACED
ARMS       r2c: rfft cascade           vs decoupled proto-stride
           c2r: NATURAL packed cascade vs SPLIT decoupled stride
CROSSOVER  _vfft_r2c_decouple_min_k = 32   (r2c_dispatch.h:97; c2r uses the SAME
           constant). The code is candid that 32 is a fossil: "the K=32 default is
           the N=256 crossover, but the true crossover shifts per N" - which is
           exactly why the race exists.
BANK       wisdom2_real | route=rfft|stride  /  route=natural|split
           metric=fwd1 for r2c, bwd1 for c2r; dir stays ABSENT on both
CALLER     the axis is K, not N
```

**Above the crossover there is no real-specific tournament.** The decoupled path builds
an inner **c2c at N/2** and inherits the entire c2c stack (S1.1-S1.4). This is why r2c
single-threaded "loses by design" - at high K the real path *is* the c2c path plus
fold/unfold.

```
R1.1 rfft chain x per-stage variant   RACED. Multisets <= 5 stages, radix <= 16, x
                                      the full FLAT/LOG3/T1S cartesian. Low-K side.
R1.2 inner c2c(N/2) chain             RACED. The high-K side; inherits S1.
R1.3 zr2c composite route (kind-5)    RACED. child_oop_il vs child_nat_ip.
R1.4 1D odd-real bridge               RACED, BANKED (2026-09-02): the K=1 OOP IL
                                      r2c race (rfft handle vs the c2c bridge)
                                      banks a component row in the real shard,
                                      t=r2c n=N q=1 ord=nat place=oop role=comp
                                      lay=il | eng=oddr route=rfft|bridge, and
                                      replays it; VFFT_ODDR_NORACE pins. c2r odd
                                      and prime r2c take the bridge by RULE (no
                                      rfft arm exists): structural, nothing to bank.
R1.5 smooth-odd r2c                   same race and same row as R1.4 (the site
                                      does not distinguish smooth from awkward).
R1.6 c2r packed/natural factorization STRUCTURAL. Raced nowhere, banked nowhere;
                                      wisdom-first off a static pointer.
```

WITNESS: `1d.il.ip.r2c.1024` -> `zr2c=1`, child `[zr2c]` [measured].
`1d.il.oop.r2c.1024` -> `zr2c=0` but still carries a `zr2c` child [measured].

---

## 6. 2D C2C

### D1. SPLIT - its own codelet family

2D split is **not 1D machinery applied twice**. It uses the generated `strided` family:
a *uniform 6-arg in-place 2D ABI, direction selects the slot*
(`emit_strided_registry.ml`, coverage `strided-avx2`).

```
D1.1 calibrated 2D plan vs the 1D-wisdom-inner fallback   RACED
     ARMS  top-K row candidates (N=N2 at K=B) x top-K col candidates
           (N=N1 at K=N2), each cross product built, roundtrip-gated and timed
           end-to-end, then the calibrated winner vs the fallback
D1.2 NATURAL: J_nat sweep                                 RACED
     ARMS  the natural pool (DP candidates + injected palindromes), each scored on
           the FULL natural cost (2D scrambled fwd + dim-1 whole-row reorder)
     FP    nat2d= natpairs= nat2dcyc=
     WITNESS  2d.sp.oop.c2c.256 order=NATURAL
              -> nat2d=1 natpairs=1 nat2dcyc=112 [measured]
D1.3 2D split threading                                   STRUCTURAL
     No race anywhere; threads unconditionally with structural floors. No
     fingerprint field exists.
```

### E1. IL - the il2d tier

```
E1.1 il2d chain (sets nst, R[], L[])  RACED. Every factorization of N1 over
     POOL {64,32,16,8,4, 27,25,21,19,17,15,13,11,9,7,5,3}, depth <= 4, capped at
     24 candidates with the drop LOGGED.
E1.2 wl - banded column walk width    RACED. {0 unbanded} + WPOOL
     {8,16,32,64,128,256} filtered by (w<=N1, N1%w==0, some stage with w%L[s]==0)
     + the L2-gated cascade width.
     GATES E1.3, E1.4, and whether E1.6 EXISTS AT ALL.
E1.3 cut                              DERIVED from wl (the tcut law: width is the
                                      INPUT, cut is the OUTPUT).
E1.4 tf (tfuse)                       DERIVED; slaved to wl (tfuse = w > 0).
E1.5 roop - row route                 RACED. in-place per-row K=1 child vs an OOP
                                      child + 2*N2 scratch + copy-back.
E1.6 cmt - column/band MT             RACED, banked WITH cmtt (the per-T class).
                                      Bluestein column axes included since
                                      2026-09-02 (the column-window pipeline,
                                      mode 3 of the c2c walk / the real strip);
                                      the banded walk (E1.2) stays structural
                                      for them (no chain walk to tile).
E1.7 N1-arm                      RACED, BANKED (2026-09-02): native odd chain vs
                                 COLUMN-AXIS Bluestein -> sets blu, rewrites
                                 nst/R/L. Verdict token blu= on the lay=il row
                                 (0 = chain, M = Bluestein length; the row's
                                 chain= is the chain that SERVES), direction-
                                 shared on the real row; replayed by both tiers
                                 on blu > 0 ALONE (a prime N1's banked chain is
                                 the pow2 M chain — no odd factor to gate on;
                                 the hasodd-gated replay ran the M chain as N1's,
                                 caught by the naive-DFT probe 2026-09-02),
                                 re-raced only unraced or under recalibrate. Env
                                 VFFT_IL2D_BLU pins and never banks. Before this
                                 the race ran on EVERY odd-N1 create.
E1.8 wc                               ENV only (VFFT_IL2D_WC).
E1.9 NATURAL n1 (M4-lite)             STRUCTURAL. Closed-form leaf redirection, or
                                      the create REFUSES. No race, no bank.
E1.10 2D NATURAL per-axis reorder tapes  STRUCTURAL, deterministic.
E1.11 column-stage KERNEL FORM          RACED per cell per stage (2026-09-02, parity
                                      with the 1D il_kv axis). Stages at r32
                                      (b48 | b84) and r64 (b88 | b416) race their
                                      rival BLOCKED forms at create - coordinate
                                      descent, the whole column pass timed, the
                                      construction-table form is the incumbent
                                      (3% margin) - and bank BY NAME on the 2D
                                      chain row: forms=b48.-.b84 (one name per
                                      stage, "-" = single form). Replayed by
                                      installing the named kernels over the
                                      resolved defaults; both tiers and the
                                      Bluestein inner (M, N2) row. Monolithic is
                                      never served at r32/r64 (standing rule);
                                      r4/8/16 and the odd radices have one form.
                                      Env VFFT_IL2D_FORMS pins, never banks.
```

**`wl` gates the existence of the MT axis.** `nb = N1/wl`; `nb < 2` means no bands, so
no column-MT race can run. Verified three independent ways:

| cell | wl | nb | outcome | measured |
|---|---|---|---|---|
| 256x256 | 256 (=N1) | 1 | no race; cmt=0 banked with cmtt | `wl=256`, no cmt |
| 64x64 | 8 | 8 | race runs | `wl=8 cmt=1` |
| 1024x1024 | 16 | 64 | race runs | `wl=16 cut=3 cmt=1` |

---

## 7. 2D REAL

### E2. IL - the REAL tier

```
E2.1 rw - the ROW ROUTE       RACED, PER DIRECTION. The per-row TC door (one
                              vfft_execute on the transform-contiguous batch
                              handle at (N2, K=N1)) vs ROWSPLIT at width W.
                              r2c and c2r race their own rw/wl/cmt and bank them
                              as separate token sets (c2r: rw_c2r wl_c2r cmt_c2r
                              cmtt_c2r) on the ONE direction-shared real IL row
                              (the chain is shared by the pair law). Until
                              2026-09-02 both directions wrote the same tokens and
                              c2r replayed r2c's verdicts.
E2.2 wl - banded column walk  RACED. unbanded vs WPOOL, filtered by
                              _il2d_real_wl_cut >= 0 and wl < N1.
E2.3 cmt / cmtt               RACED, per-T. serial _il2d_real_cols vs threaded
                              _il2d_real_cols_mt at T = h->nthreads. A "no" is
                              banked exactly like a "yes"; if the threaded arm
                              cannot engage (T<2 or units<T) the early bank writes
                              cmt=0.
E2.4 zr2c route (row child)   RACED, kind-5.
E2.5 rfft factorization for a ROWSPLIT arm  RACED - and banked EVEN WHEN THE ARM
                              LOSES.
E2.6 inner c2c (N2/2, W) for a ROWSPLIT arm  RACED.
E2.7 r2c route for a ROWSPLIT arm            RACED.
E2.8 c2r route for a ROWSPLIT arm            RACED.
E2.9 column chain             STRUCTURAL for the real tier: precedence is
                              env > banked lay=il real row > greedy-longest.
                              (Asymmetry with E1.1, which IS raced for c2c.)
E2.10 oddn2                   STRUCTURAL. Odd N2 real rows ride a K=1 c2c child.
E2.11 norowz                  ENV only (VFFT_IL2D_NO_ROWZ).
E2.12 wc / roop               STRUCTURALLY UNREACHABLE for real.
E2.13 column-axis Bluestein M STRUCTURAL. M = 16, then while (M < 2*N1-1) M <<= 1.
                              Its INNER chain at M: served from the (M, N2) 2D
                              chain row — replayed when banked, else the E1.1
                              chain race at (M, N2) and banked there (both
                              tiers, 2026-09-02). Was the greedy chain, never
                              measured.
```

**The oddn2 / column-MT guard asymmetry is DELIBERATE.** The row-route race at
`vfft.c:6914-6919` carries `!il2d_oddn2`; the column-MT guard at `:6923-6924` does not.
Odd N2 has no ROWSPLIT arm to race, but column threading stays valid. Measured
consistent: 128x127 at T=8 engages `cmt` and is BIT-IDENTICAL to T=1 (0 of 16448 doubles
differ).

WITNESS: `2d.il.oop.r2c.256` -> `nst=2 wl=8 cut=1`, children `[il2drow, tcb, zr2c]`
[measured]. `2d.il.oop.r2c.128x127` -> `oddn2=1` [measured].
`2d.il.oop.r2c.4096x16` -> `rw=64` [measured].

---

## 8. Cross-cutting

```
X1 pqmt - plane queue loop vs queue  RACED, BANKED (2026-09-02). dims=2, howmany>1.
                                     pq=<0|1> pqn=<P> pqt=<T> on the primary
                                     plane's own row (IL c2c / IL real / split);
                                     replays on a (P, T) match, re-races on a
                                     mismatch; VFFT_PQ_NO_MT pins, never banks.
X2 pqw / pqn                         STRUCTURAL. pq_n = K unconditionally; pq_wn is
                                     the clone count after clamping.
X3 tcbw                              STRUCTURAL. The number of clones that built AND
                                     passed _tc_clone_equiv. Whether they RUN is
                                     the raced verdict tcmt (B5.1).
X4 tcbsn / tcbdn                     DERIVED arithmetic from (transform, placement).
X5 mtunsafe                          STRUCTURAL - a CORRECTNESS self-check, not a
                                     timing race. Whole-batch reference vs a
                                     sequential replay of every slab size.
```

---

## 9. Reverse index: fingerprint field -> axis

| field | axis | lit by [measured] |
|---|---|---|
| `k1` `sp` `il` | S2.1 | any 1D c2c **out-of-place**, either layout |
| `ilme` | RETIRED 2026-09-03 with the convert machinery (B2 is the IL plan race now) | - |
| `zroute` | C1.1 | IL cascade, N >= 2048 (also 3072) |
| `zr2c` | R1.3 | IL 1D real: ip.r2c, ip.c2r, oop.c2r (**not** oop.r2c) |
| `nat` `natcyc` | S3.1 | 1D natural order |
| `nat2d` `natpairs` `nat2dcyc` | D1.2 | 2D **split** natural |
| `il2d.nat` | E1.9 | 2D **IL** natural |
| `il2d.nst/wl/cut/tf` | E1.1-E1.4 | any 2D IL |
| `il2d.cmt` | E1.6 / E2.3 | 2D IL + nthreads > 1 |
| `il2d.oddn2` | E2.10 | 2D IL real with **odd N2** (128x127) |
| `il2d.blu` | E1.7 | 2D IL with **prime N1** (127x100 -> blu=256) |
| `il2d.rw` | E2.1 | 2D IL r2c asymmetric (4096x16 -> rw=64) |
| `il2d.roop` | E1.5 | 2D IL c2c at large N1 (16384x64) |
| `tcbsn` `tcbdn` | X4 | K=8 + BATCH_TRANSFORM_CONTIGUOUS |
| `tcbw` | X3 | as above **plus** MT |
| `tcmt` | B5.1 | as above **plus** MT: the raced serial-vs-slabs verdict (0 = serial) |
| `pqn` | X2 | 2D howmany > 1 |
| `pqw` `pqmt` | X1 / X2 | as above **plus** MT |
| `ztmt` `ilrace` `mtunsafe` `il2d.wc` `il2d.norowz` | C1.9 / - / X5 / E1.8 / E2.11 | **never observed non-zero** in 91 probed configs |

**Axes with NO fingerprint field** (invisible to the safety harness): `il_kv`, `il_bkv`,
`sp_pair`, every `chain=` / `vars=` token, S1.1-S1.4, S2.3, S2.4.

---

## 10. Known gaps

* **MT verdicts** - closed 2026-09-02 for the cascade (`zt_mt`/`zt_mt_t`, both
  placements), the 2D column pass (`cmt`/`cmtt`, both real directions) and the plane
  queue (`pq`/`pqn`/`pqt`); the shipped store carries them for the seeded cells (T=8),
  other cells and thread counts race once on first use and bank themselves.
* **RACED-NOT-BANKED** - none left as of 2026-09-02 (B1.4, B4, R1.4/R1.5 bank now).
  Cells and thread counts the shipped store has not seen race once on first use and
  bank themselves.
* **The create-race counter is blind to MT.** None of `_zt_mt_race`, `_pq_mt_race`,
  `_il2d_real_colmt_race`, `_il2d_c2c_mt_race` increments it, so `races=0` on a threaded
  plan is a FALSE ZERO.
* **Layout collision.** Only 27 of 539 records carry `lay=`, so IL and split verdicts at
  the same (t, n, q, ord, place) can collide on one key.
* **`il_kv` / `il_bkv` are invisible to the fingerprint**, so kernel-form changes cannot
  be detected by the migration harness.
* **`il_bkv == 0` ambiguity** - closed 2026-09-02: the candidate carries
  `il_bkv_raced`, a raced "defaults won" banks as an explicit `il_kv=0` backward
  row, the reader returns -1 for "no row" (B1.3's KNOWN GAP is gone).
* **UNVERIFIED**: the odd/prime block (B4, E1.7, E2.13, the Bluestein sweeps) lost its
  adversarial verification three times to rate limits. Treat as unchecked.
* **Backward direction** (audited 2026-09-03): only kernel FORMS have their own
  backward cells (B1.3, the pair and since 2026-09-03 the chain3 leaf slot). The
  in-place attach (B2), the prime method (B4) and the cascade recipe (C1: t2q has no
  backward twin, tw is shared) serve the forward verdict backward. B2 and B4 were
  measured in both directions (the dirflip probe) and no winner flips: declared
  structural by measurement, not by ruling. The cascade backward tile width is the one
  raceable direction axis left unmeasured (no backward-only width exists to pin).
