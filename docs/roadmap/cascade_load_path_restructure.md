# K=1 z-cascade load path — restructure plan (ZTURN)

**Date:** 2026-07-26
**Scope:** `src/core/oop/zsplit.h` (K=1 SCRAMBLED OOP cascade, N ≥ 2048), emitter
`src/dag-fft-compiler/generator/lib/codelet_zsplit.ml`.
**Status:** design frozen, nothing built. Phase 0 is a kill test.

Evidence tags used throughout, per the project's MEASURE-never-ESTIMATE rule:

| tag | meaning |
|---|---|
| **[M]** | MEASURED — a number from a run, with its harness quality stated |
| **[D]** | DERIVED — algebra or static op counting, and where noted, verified by an executable model |
| **[A]** | ASSUMED — no measurement, no proof. Every [A] is a risk, and they are ranked in §7 |

---

## 1. VERDICT

### 1.1 The decision, stated plainly

**Build the turn-at-the-ingest. Do NOT build the ping-pong.**

The brief's premise was *"adopt MKL's shape: a PING-PONG interior (a second N-complex
plane) with the corner-turn absorbed into an out-of-place INGEST's stores."* Two halves,
and the evidence now separates them cleanly:

* **The turn-at-ingest half is right, is new, and needs ZERO extra planes.** The ingest
  (`radix{4,8}_z_s0s_fwd_avx2`) is *already* out-of-place — `zsplit.h:195` runs
  user `zin` → `p->sp`, two distinct buffers — and its stores are *already*
  an index-identity map (`radix4_z_s0s_avx2.c:73-80` stores to the exact indices it
  loaded). Changing its store address map costs no plane, no traffic, and no in-place
  hazard. **[D]**
* **The ping-pong half is refuted, by measurement taken during this design review.**
  A second interior plane is free for a *terminator-shaped* pass and expensive for a
  *mid*. Both numbers are now measured, and the asymmetry is the missing fact in the
  whole prior record.

### 1.2 The evidence that kills the ping-pong

| what was raced | result | harness quality |
|---|---|---|
| **[M1]** 1 plane + TR4 · vs · 2 planes − TR4 · vs · 1 plane − TR4, terminator-shaped pass, N=2048/4096/8192/16384 (`build_tuned/benches/spike_pingpong_vs_tr4.c`) | turn costs **356 / 692 / 1367 / 2715 ns**; second plane costs **+3.1 / +1.4 / −8.1 / −1.5 ns ≈ 0** | **HIGH** — control 1.0034, 64 MB cachebust, reps forced to a multiple of arm count, one arena + 64 B skew |
| **[M3]** `msg` in-place · vs · the *same kernel*, zero shuffles added, merely made out-of-place | **1.49 – 2.26× SLOWER** at 2048/4096/8192/16384. Two-cursor wrapper pointed back at one plane reads 0.99–1.09 ⇒ 100 % of the penalty is the second plane. Offset swept over {0,64,128,320,576,1088,2112,4160}: flat ⇒ **not** 4 KB aliasing | good — control passing, skew sweep included |
| **[M5]** whole cascade, turn-in-last-mid + 2nd plane vs incumbent, one binary, control arm | **1.03 – 1.13× SLOWER**; 10 of 12 cell-runs slower | good — two compilers, three runs, control passing |
| **[M6]** isolated last-mid + terminator region, same change | **+7.4 / +21.4 / +8.3 / +14.3 %** | good |

**[M6] reproduces `z_cascade_plan.md:911-915`'s "+5…+22 % for out-of-place into a 2nd
plane" — the row that had no committed code.** That number was right. It is now
committed.

Mechanism **[D]**: an in-place RMW pass writes lines it already owns; an out-of-place
pass must fill A *and* RFO B. "Same bytes at the algorithm level" is not the same traffic
at the cache level. The terminator has slack for that (it is ~2.4× port-5 oversubscribed);
the mid does not — it runs at ~98 % of its two-FP-port bound (80 FP ops per 32-complex
iteration in 40.8 measured cycles at N=16384) **[M]**.

### 1.3 What the turn-at-ingest is worth — honest numbers only

Two measured inputs, one derived composition. **No estimate is used where a measurement
exists, and no measurement is extrapolated past what it covered.**

**[M2]** In-cascade terminator, incumbent `sterm` vs a turn-free terminator that reads
8 contiguous 64-byte granules (`term_ledger.c`):

| N | sterm (ns) | turn-free (ns) | ratio |
|---|--:|--:|--:|
| 2048 | 1177.8 | 811.1 | **0.689** |
| 4096 | 2375.0 | 2274.2 | **0.958** ← anomalous, see §7 |
| 8192 | 4228.4 | 3140.7 | **0.743** |
| 16384 | 8075.6 | 6306.7 | **0.781** |

Harness quality: **MARGINAL.** Control read 0.99–1.09. A 1.09 control is 9 % off; this
project's standard is ~1.00 (**[M1]** managed 1.0034). These numbers are directionally
solid and quantitatively soft. They must be re-run with a passing control (Phase 0b).

**[M7]** In-situ terminator time share, measured the same session: **50 % @2048 · 44 %
@4096 · 36 % @8192 · 32 % @16384.**

> **Correction to the record.** The brief's "41–46 %" and `z_cascade_plan.md:485`'s 44 %
> come from VTune's **retiring** column, a pipeline-slot metric. The **ticks/pass**
> column in the same table gives 17.3e9 / 49.9e9 = **34.7 %** at N=16384, and
> `z_cascade_plan.md:687` states 35 % directly. Any arithmetic that mixes the 44 % with
> the 17 %/49 % leaf/mid shares from the ticks table produces shares summing to 1.10.
> Two of the three candidate designs did exactly that; it inflated their headline by
> ~1.3×.

**[D]** Composition — terminator saving as a fraction of the whole transform:

| N | share [M7] | terminator ratio [M2] | whole-transform saving |
|---|--:|--:|--:|
| 2048 | 0.50 | 0.689 | **−15.6 %** |
| 4096 | 0.44 | 0.958 | **−1.8 %** (anomaly) |
| 8192 | 0.36 | 0.743 | **−9.3 %** |
| 16384 | 0.32 | 0.781 | **−7.0 %** |

**[D]** Ingest charge, from static op counting: the new ingest costs **+2 port-5 ops per
16 complex** (18 vs today's 16) = N/8 extra shuffles = **359 ns at N=16384** if fully
exposed at 1 shuffle/cycle @5.7 GHz = **+1.4 %** of the transform. **This term is
UNMEASURED and is the subject of the Phase-0 kill test.**

**Net, DERIVED from measured inputs: −6 % to −14 % forward**, best at small N, worst
exactly where the MKL gap is largest. Against today's 0.78–0.89× vs MKL forward
(2048 0.89 · 4096 0.76 · 8192 0.81 · 16384 0.82, `z_cascade_plan.md:426-431`), that lands
approximately **0.88 – 1.03×**. It beats MKL at 2048 and **does not reach parity at
8192/16384**.

### 1.4 So is it worth building?

**Yes — with the expectation corrected downward and one structural caveat.**

Reasons to build:
1. The terminator half is **already measured** in situ [M2] and needs no new idea.
2. It costs **zero extra memory**, unlike the premise it replaces.
3. The route is a **per-cell searched axis** (§4), so a cell where it loses simply keeps
   the legacy route. There is no all-or-nothing ship risk.
4. The move is genuinely untried, and its failure mode is bounded and cheap to detect.

Reasons for restraint, stated up front so nobody is surprised later:
1. **The honest projection is 6–14 %, not 13–16 %.** `z_cascade_plan.md:741` independently
   projected **−5…−11 %** for exactly this shuffle deletion. The two agree.
2. **[D] The new layout makes the 4 KB-aliasing problem worse, not better.** New mid leg
   stride is `16·R0·D[s] = 64·D[s]` bytes vs today's `16·D[s]`. Working through
   `zsplit.h:140-143`'s `D[]` for the four calibrated chains:

   | N | chain | old mid leg strides (B) | new (B) | mids on exact-4096 |
   |---|---|---|---|---|
   | 2048 | 4.8.8.8 | 1024 / 128 | 4096 / 512 | 0 → **1** |
   | 4096 | 4.4.4.8.8 | 4096 / 1024 / 128 | 16384 / 4096 / 512 | 1 → **2** |
   | 8192 | 4.4.8.8.8 | 8192 / 1024 / 128 | 32768 / 4096 / 512 | 1 → **2** |
   | 16384 | 4.8.8.8.8 | 8192 / 1024 / 128 | 32768 / 4096 / 512 | 1 → **2** |

   At every cell, one **more** mid stage puts all 8 leg streams into a single L1d set.
   The mids are 49 % of the transform and are today the clean part. Lever 2's clean-v2
   methodology measured **−4.5…−18.9 % on a single pass from stride padding alone** [M10]
   — this class of effect is not noise. **The padded-plane-pitch lever therefore stops
   being optional and becomes part of this work** (Phase 6), and a mids-only A/B with its
   own control is mandatory (Phase 4b).
3. The 4096 cell's 0.958 [M2] is unexplained and must be re-measured before any claim
   covers it.

### 1.5 This REOPENS "last-15 % is architectural, hunt CLOSED"

`z_cascade_plan.md:926-935` concluded: *"This closes the 'last ~15 %' investigation …
The remaining gap to MKL is the **architectural** in-place-vs-ping-pong choice plus MT —
not a missing kernel optimization."*

That conclusion **did not cover this move**, for four independent reasons:

1. **It measured a different move.** LEVER 1 relocated the turn to the **LAST MID**
   (`msgt`). **[D]** Under any digit-weight permutation of the plane's address map,
   exactly one stage is "bad" — the one whose butterfly digit is the innermost (lane)
   digit. Moving "bad" onto a mid moves it onto a stage that had *zero* mandatory
   shuffles, so the cost transfers 1:1. **Lever 1 is a wash by construction**, and that
   is exactly what §4.9996 measured. The inference "therefore the architecture is the
   problem" does not follow from it.
2. **Its stated premise is factually wrong.** `z_cascade_plan.md:922-924` says *"MKL
   escapes this only because it **ping-pongs planes for every stage**."* The
   disassembly says otherwise [M9]: fn `0x1825c3800` has **ingest 37 shuffles, interior 0
   shuffles across ~1900 instructions, finisher 80**, all on **one** scratch plane `r9`;
   `r8` is an int32 offset table, not plane B. MKL's middles are **in-place on a single
   plane**. Every downstream sentence built on that premise ("capturing it would mean
   re-architecting to ping-pong planes, adds traffic to *every* stage") is void.
3. **Its harness violated two project rules.** `zil_orient_spike.c` has **no control
   arm**, **per-arm output buffers** (`:315-318` vs `:327-330` — the pattern the project
   records as having once fabricated a bogus 1.51×), and **five separate power-of-two
   `_aligned_malloc` calls**. Its measured spread was −2…+8 %, i.e. entirely inside the
   ±5 % code-placement band §4.9993 declared uninterpretable, from a single binary with
   new symbols.
4. **The ingest destination was never tried.** Grep-verified: no `s0st`/`s0t`/ingest-turn
   kind exists in any bench, in `zsplit.h`, or in `codelet_zil.ml`/`codelet_zsplit.ml`.

**What is now settled that was not before:** the second plane is free on a
terminator-shaped pass [M1] and costs 50–126 % on a mid [M3]. That asymmetry is the real
reason Lever 1 died. It appears nowhere in the record and should be written into
`z_cascade_plan.md` §4.9996 as an erratum.

### 1.6 What was fatal, and what it kills

**The "TURN-AT-THE-SEAM" design (turn in the last mid's stores + one extra plane for one
pass) is REFUTED, not merely doubted.** It was built and run:

* it failed **its own** Gate-1 kill criterion (`msgt/msg > 1.50` ⇒ dead) at **all four
  cells**: 2.35 / 1.89 / 1.88 / 1.96 **[M4]**;
* end-to-end it measured **1.03–1.13× slower** **[M5]**;
* its load-bearing claim — *"traffic-neutral, the second plane is free"* — is false
  **[M3]**;
* its mandated *"+64 B skew decorrelates the streams"* mitigation is inert **[M3, swept]**.

Its **index map is correct** (bit-identical output, maxabs 0.0, all four cells) and its
kernels compile — so this is a clean, committed negative, not a botched attempt.

**What it kills, permanently:**
* any variant of "relocate the corner-turn into a mid" (Lever 1 in all its clothes);
* "give a mid a second plane";
* the general form "the turn can be hidden under a mid's spare port-5 capacity" — port 5
  *is* free in the mid, and it buys nothing, because the mid's constraint is FP ports,
  load/store ports and the register file, not port 5.

**One thing it must NOT be read as killing:** the turn-at-ingest. The ingest is not a mid.
It is already out-of-place, already crosses the user↔scratch boundary, and already runs a
mandatory shuffle network (the z→split deinterleave) that the turn can share.

---

## 2. THE DESIGN — ZTURN

**One-sentence statement:** change the interior plane's digit order so that the
*ingest's own output digit* `a_0` occupies the vector-lane axis; the turn is then absorbed
into a store network the ingest already has to run, every mid stays shuffle-free and
in-place on **one** plane, and the terminator's four `TR4`s become sixteen plain
contiguous loads.

### 2.1 Buffers

| buffer | size | change |
|---|---|---|
| caller `zin` | N complex, interleaved, natural | none |
| `p->sp` — **the one** interior plane | N complex, 64-byte `[re×4][im×4]` **block-split** | **size, alignment and granularity all unchanged**; only the *address map* changes |
| caller `zout` | N complex, interleaved, scrambled | permutation changes (§2.6) |

**No second plane. No ping-pong. Mids stay in-place.** This is the shape the MKL
disassembly actually shows [M9], and it is the shape the early-kill spike's arm C
measured [M1].

**Hard constraint 1 is honoured:** the interior remains 64-byte block-split. `ρ` permutes
which *complex index* sits in which lane; it does not touch the `[re×4][im×4]` granule
contract, so split `cmul` still needs no `cflip` and the 29 %-at-16384 granularity result
stands. Nothing anywhere in this design converts an interior to interleaved.

**Hard constraint 2 is honoured:** no new IL-boundary/split-interior codelet is created
for the small-N / two-pass IL tiers. The two kernels that touch interleaved data here
(`s0t`, `stf`) are the cascade's own z↔split *boundary* stages at N ≥ 2048 — the tier
where z-in / split-interior / z-out is the correct arrangement. Their signature is the
frozen 11-arg z ABI (`zin`, `zout`, no `in_re`+`in_im` pair), which is the stated test.

### 2.2 Forward dataflow

Chain `r_0 … r_{nf-1}`, `R0 = r_0` (**tranche 1: R0 = 4**; all four calibrated chains have
`chain[0]=4`), `r_{nf-1} = 8`.

```
[0]        s0t          zin (user, natural z)  ->  sp        OUT-OF-PLACE   <-- THE TURN LIVES HERE
           radix R0, twiddle-free
           loads : 8 plain vmovupd, zin[2*(l*Ls+k)] and +4, Ls = D[0]      ZERO shuffles
           body  : DFT-R0 on INTERLEAVED 2-complex vectors
           stores: ONE 4x4 network that does DEINT and TURN together,
                   4 contiguous 32-byte stores per half at zout[8*p + {0,4}]
           p5    : 18 per 16 complex  = 1.125/complex   (today: 16 = 1.000)

[1..nf-2]  msg          sp -> sp               IN-PLACE       ZERO shuffles
           *** THE KERNEL OBJECTS ARE UNCHANGED. NOT REGENERATED. NOT RECOMPILED. ***
           new argument values only:  Ls = count = R0*D[s]   (was D[s])
                                      Gs        = G[s]/R0    (was G[s])
           new twiddle TABLE CONTENTS only (4-distinct-lane records, 4x smaller)

[nf-1]     stf / stf2   sp -> zout (user)      OUT-OF-PLACE
           loads : 16 PLAIN vmovupd at zin[2*R0*radix*h + 2*R0*l + {0,VW}]
                   = zin[64*h + 8*l + {0,4}] at R0=4          ZERO shuffles
           body  : IDENTICAL to today -- TP_PowW1 packed w^1, squaring tree, SPLIT_BFLY8
           stores: IDENTICAL to today -- E_z REINT, zout[2*(l*OLs + R0*h) + {0,4}]
           p5    : 32 per 32 complex = 1.000/complex   (today: 64 = 2.000)
```

`zin == zout` stays safe by the same argument as today: forward, `zout` is written only by
`stf`, which reads only `sp`.

### 2.3 Backward dataflow (full mirror — same win, same tranche, not deferrable)

```
[nf-1]     stfb         zin (user comb) -> sp   OUT-OF-PLACE     runs FIRST
           loads : E_z DEINT over the comb -- UNCHANGED shape, 4 p5/leg/4 complex.
                   The R0 lanes are 4 CONSECUTIVE complex, so this is two ymm + the
                   standard unpack/permute4x64 pair, exactly as stermb does today.
                   Its 8 distant LOAD streams are untouched -- the reason the bwd
                   2-quad was refuted (+29..36%, §4.9993) still holds, so NO stfb2.
           body  : IDFT-8 then POST-twiddle from the conjugated packed w^1 (table_conj)
           stores: PLAIN block stores zout[64*h + 8*l + {0,4}]
                   *** stermb's STORE-side TR4 is DELETED -- 32 p5 per 32 complex gone ***

[nf-2..1]  msg_bwd      sp -> sp   IN-PLACE   unchanged objects, same arg rescale,
                                   twspb[] pre-conjugated as today (b[4+j] = -sin)

[0]        s0tb         sp -> zout (user, natural z)   OUT-OF-PLACE    runs LAST
           loads : the exact inverse of s0t's store network, 8 p5 per 8 complex
           body  : inverse twiddle-free interleaved DFT-R0 (+i rotate)
           stores: PLAIN interleaved, zout[2*(l*Ls+k)] and +4     ZERO shuffles
```

Backward ledger: today `s0s_bwd 1.00 + mids 0 + sterm_bwd 2.00 = 3.00` p5/complex; new
`s0tb 1.125 + mids 0 + stfb 1.00 = 2.125`. **[D]** Identical saving to forward. Backward
is 0.73–0.89× vs MKL today, so it gains proportionally as much.

`zin == zout` stays safe backward: `zin` is read only by `stfb`, the first stage.

> **Note on the load/store framing.** Forward's turn is deleted from a *load* edge;
> backward's from a *store* edge. Both cost exactly 32 port-5 ops per 32 complex in the
> shipped kernels (verified by census). Any framing of this work as "load-side turns are
> expensive, store-side turns are free" is **not** supported by the code. What differs is
> *which kernel* absorbs it — one that already runs a mandatory shuffle network, or one
> that does not.

### 2.4 The index map — DERIVED

All of the following is **[D]**, and steps 4–8 have been **independently verified twice by
executable lane-explicit models** during adversarial review: 6 chains at
`max |new − legacy∘σ| = 0.000e+00`, and 8 chains (incl. `4.4.8`, `4.8.8`, `4.4.4.8`,
`4.8.4.8`, `4.4.8.8`, `4.8.8.8`, `8.4.8`, `8.8.8`) at `0.00e+00`, plus an address-level
emulation of `leg_addr`/`blk_addr` and the `msg` wrapper bump showing the plane is covered
exactly once, no overlap, no out-of-range, at all four production cells.

**Setup** (from `zsplit.h:140-143`, read):
`D[nf-1] = 1`, `D[i] = D[i+1]·r_{i+1}` ⇒ `D[s] = ∏_{j>s} r_j`;
`G[0] = 1`, `G[i] = G[i-1]·r_{i-1}` ⇒ `G[s] = ∏_{j<s} r_j`.
Digit vector `a = (a_0 … a_{nf-1})`, `a_i ∈ [0, r_i)`.

**Step 1 — what a plane slot means today.**
`msg`'s body addresses leg `l` at `2*(l*Ls + k)` with `Ls = D[s]`, and its wrapper bumps
`bp` by `2*R*Ls` doubles per group (`radix8_z_msg_avx2.c:159-177`, read). So the plane
index at stage `s` is `g·r_s·D[s] + a_s·D[s] + k = g·D[s-1] + a_s·D[s] + k`. Unrolling
over `g` and `k`:

```
    λ(a) = Σ_i a_i · D[i]                                                       (1)
```

a mixed-radix bijection onto `[0,N)` with `a_0` most significant. (This also proves
`p->gb[s][g] == g·D[s-1]`, i.e. the `gb[]` table filled at `zsplit.h:152-155` is exactly
the linear bump the kernel already performs. It is **never read** by either execute path —
dead memory and dead code, to be dropped.)

**Step 2 — why the TR4 exists, as an equation.**
Lane index = `λ mod VW = λ mod 4`. Since `D[s] ≥ r_{nf-1} = 8` for every `s ≤ nf-2`,

```
    λ(a) mod 4 = a_{nf-1} mod 4
```

**The lane axis IS the terminator's butterfly digit.** Every other stage butterflies a
digit of weight `D[s] ≥ 8`, hence lane-invariant — which is precisely why `msg` measures
`unpck=0, perm2f128=0, perm4x64=0`. The TR4 is not data movement; it is the resolution of
`butterfly axis == lane axis`.

**Step 3 — the relocation lemma.**
A vectorized butterfly over digit `a_s` requires `a_s` to be lane-invariant. Under **any**
permutation of digit weights, exactly **one** digit is innermost, hence exactly **one**
stage is "bad". *A turn cannot be deleted, only relocated.* It is **free** only if
relocated onto a stage that already runs a shuffle network it cannot avoid — i.e. one of
the two z↔split boundary stages (`s0s` deinterleaves; `sterm` re-interleaves). Relocating
onto a mid gives a stage with 0 mandatory shuffles a 1.00/complex bill: **wash by
construction**, which is what §4.9996 measured and [M4]/[M5] have now re-measured.

**Step 4 — choose `a_0` innermost.**

```
    ρ(a) = R0 · ( Σ_{i≥1} a_i · D[i] ) + a_0 ,      R0 = r_0                    (2)
```

equivalently `ρ = R0·(λ mod D[0]) + (λ div D[0])`: the transpose of the `R0 × (N/R0)`
array.

*Bijective, and this is the load-bearing algebraic fact:* for `s ≥ 1`,
`D[s] = ∏_{j>s} r_j` **never contains the factor `r_0`**. So `{D[s]}_{s≥1}` is already a
valid mixed-radix weight set for the digit tuple `(a_1 … a_{nf-1})`, with range
`∏_{s≥1} r_s = N/r_0 = D[0]`. Exact, no rescaling, no remainder. **This is why the mid
kernels survive untouched — they reuse literally the same integers.**

**Step 5 — per-stage consequences.** For `s ∈ [1, nf-1]`, varying `a_s` changes `ρ` by
`R0·D[s]`, a multiple of `VW=4` ⇒ whole 64-byte granules; the 4 contiguous lanes vary
`a_0 ≠ a_s` ⇒ **plain**. Loop shape:

```
    ρ = h·(R0·D[s-1]) + a_s·(R0·D[s]) + inner,   inner ∈ [0, R0·D[s])
    h  = MSD-first index of (a_1 … a_{s-1}),     h ∈ [0, G[s]/R0)

    ⇒  Ls = count = R0·D[s]        Gs = G[s]/R0        group span = 2·r_s·Ls doubles
```

`Ls`, `Gs` and `count` are runtime arguments; the group bump `2*R*Ls` and twiddle bump
`(R-1)*2*VW` are already the correct formulas. **The mid kernels do not change.**

**Terminator** `s = nf-1`, `D[nf-1] = 1` ⇒ leg stride `R0·1 = R0` complex = **exactly one
64-byte granule**. Legs are 8 consecutive granules. Loads are plain. **The TR4 is gone.**

**Stage 0** — its butterfly digit `a_0` *is* the innermost digit. Bad, by design, and
that is the whole point.

**Step 6 — the ingest's store permutation.** Stage 0 reads user position `k ∈ [0, D[0])`,
leg `l = a_0`. Source complex (natural z) `= l·D[0] + k`. Destination:

```
    ρ = R0·k + l                                                                (3)
```

A straight `R0 × (N/R0)` transpose: the `R0` butterfly outputs of **one** position become
contiguous. At `R0 = 4` each position owns exactly one 64-byte granule, so in doubles the
re block of position `p` is at `zout[8p .. 8p+3]` and the im block at `zout[8p+4 .. 8p+7]`
— which is **exactly the existing emitter helper `blk_addr` with the kernel's own radix**
(`2*radix = 8`, `codelet_zsplit.ml:339-343`). No new address helper is needed on the
ingest side.

**Step 7 — how the turn becomes nearly free.** Load leg `l` as two **raw interleaved** ymm
(no deinterleave):

```
    A_l = [re(k),   im(k),   re(k+1), im(k+1)]
    B_l = [re(k+2), im(k+2), re(k+3), im(k+3)]
```

Run the twiddle-free DFT-4 elementwise on `{A_l}` and on `{B_l}`. In interleaved form the
only non-elementwise op is the single `−i` rotation:

```
    t0=x0+x2; t1=x0-x2; t2=x1+x3; t3=x1-x3;
    r = -i*t3   ==   vpermilpd(t3, 0x5) then vxorpd with [0,-0.0,0,-0.0]     (1 p5 + 1 p015)
    Y0=t0+t2; Y2=t0-t2; Y1=t1+r; Y3=t1-r
```

Store network per half (positions `q`, `q+1`):

```
    u0 = unpacklo(Y0,Y1)   u1 = unpackhi(Y0,Y1)
    u2 = unpacklo(Y2,Y3)   u3 = unpackhi(Y2,Y3)
    perm2f128(u0,u2,0x20) = [Y0re(q),  Y1re(q),  Y2re(q),  Y3re(q)]   -> zout[8q + 0]
    perm2f128(u1,u3,0x20) = [Y0im(q),  Y1im(q),  Y2im(q),  Y3im(q)]   -> zout[8q + 4]
    perm2f128(u0,u2,0x31) = [.. q+1 ..]                               -> zout[8(q+1) + 0]
    perm2f128(u1,u3,0x31)                                             -> zout[8(q+1) + 4]
```

**8 port-5 ops per 8 complex, and the same 8 ops do BOTH the z→split deinterleave AND
the 4×4 turn.** Per 16-complex iteration: 16 layout + 2 rotate = **18**, versus today's 16
(deinterleave only).

*Corroboration, and it is not theory:* this is instruction-for-instruction MKL's high-N
radix-4 ingest — `mkl_highN_cascade_suite.asm:1530-1553`, **8 `vshufpd` + 8 `vperm2f128` +
2 `vxorpd` per 16 complex** [M9]. An independent derivation from our own index algebra
lands on MKL's own instruction mix. (We do not need MKL's ~N-byte load-side offset table,
because we ship SCRAMBLED rather than natural.)

> **ERRATUM (2026-07-26, Phase-0 verifier, recounted from the asm bytes):** the line
> above undercounts. The actual `0x38c5..0x39f0` body has **10 `vshufpd`**, not 8 — the
> two extras are same-source `0x5` rotate swaps (`vshufpd ymm4,ymm4,ymm4,0x5` etc., the
> swap-fusion trick). MKL's true budget is **18 shuffle-class + 2 xor per 16 complex**,
> matching the anatomy census row (shuf+xor = 20) and making our 18 a **MATCH**, not a
> pass-with-diff. Our 2 `vpermilpd` occupy the same slots as MKL's 2 rotate-`vshufpd`.

> **The trap to avoid.** Keeping the current *split* butterfly and merely bolting
> `out_edge = E_blocks` onto `s0s` costs `16 (DEINT) + 16 (TR4 store) = 32` p5 per 16
> complex — **exactly the 16 the terminator saves. Net zero. That is Lever 1 in a
> different hat.** Dropping the load-side `permute4x64` and absorbing its lane order into
> the store's column relabel recovers to 24 (net −0.50/complex). Only the **interleaved**
> butterfly reaches 18 (net −0.875/complex). See §5 Phase 2 for why the 24-op variant is
> still built — as scaffolding, not as a shippable win.

**Step 8 — twiddles: a repack, not a recomputation.** Stage `s`'s exponent is
`l · brev_s(g) mod M_s`, `M_s = r_s·G[s] = N/D[s]` (`zsplit.h:156-158`), with
`brev_s(g) = Σ_{i<s} a_i ∏_{j<i} r_j` (derived by reading `_vfft_zs_brev`,
`zsplit.h:88-95`). Splitting off `i = 0`:

```
    brev_s(g) = a_0 + R0 · brev'(h),    brev'(h) = _vfft_zs_brev(h, s-1, chain+1)     (4)
    g         = a_0 · (G[s]/R0) + h                                                    (5)
```

Within one 64-byte mid record `(h, l)` is fixed and `a_0` varies across the 4 lanes, so the
record holds **four distinct twiddles**:

```
    twsp[s][(h·(r_s-1) + l-1)·8 + j]     = cos(θ(l, a_0=j, h))     j = 0..3
    twsp[s][(h·(r_s-1) + l-1)·8 + 4 + j] = sin(θ(...))
    twspb[s][... + 4 + j]                = -sin(...)      (conjugation stays plan-side)
```

Table size `(G[s]/R0)·(r_s-1)·64 B` = **one quarter** of today's, carrying the same
`G[s]·(r_s-1)` distinct values (today they are splatted 4× at `zsplit.h:161-164`).
Kernel-side addressing `tw_re[(l-1)*8]` / `+4` and cursor bump `(r_s-1)*8` are
**unchanged** — and I verified against `radix4_z_msg_avx2.c:41,43` / `radix8_z_msg_avx2.c:44,46`
that the kernels load a full ymm of cos and a full ymm of sin. They never assumed the four
lanes were equal. This again matches MKL exactly [M9]: *"each 64-byte record is `[re×4][im×4]`
holding four different twiddles, one per lane."*

Terminator (`s = nf-1`, `G[nf-1] = N/8`, one record per group `h ∈ [0, N/(8·R0))`):

```
    twq[8h + j]     = cos(-2π·( (j·(N/(8·R0)) + h) reversed ) / N)      j = a_0 = 0..3
    twq[8h + 4 + j] = sin(same);   twqb = the negated-sin twin
```

Record shape and total size are unchanged (`(N/32)·64 B = 2N B`); only the index formula
changes. The in-register squaring tree that derives `w²…w⁷` is untouched.

**Mandatory plan-time self-check** (this is the anti-guessing gate; it is free and it is
the artifact that would have prevented the `il2p.h` session):

```c
assert( twsp_new[s][(h*(r-1)+l-1)*8 + j]  ==  twsp_old[s][((j*Hs + h)*(r-1)+l-1)*8 + 0] );
assert( twq_new[8*h + j]                  ==  twq_old[2*((j*H+h) & ~3L) + ((j*H+h) & 3L)] );
```

Both were run during review across 8 chains: **0 violations.**

### 2.5 Where the turn lives, in one line

**Forward:** the ingest's store network. **Backward:** the leaf's load network (`s0tb`).
In both directions it is fused into a shuffle network the stage was already obliged to run
(z↔split conversion), and the marginal cost is the `∓i` rotate — 2 port-5 ops per 16
complex — that the split form gets for free.

### 2.6 The one real cost: the SCRAMBLED output permutation changes

**[D, verified exactly]** Under `ρ` the terminator's lanes carry `a_0` and its group index
is `h`; by (5), `g_old = a_0·(N/32) + h`. A contiguous-lane store therefore lands at

```
    out_new[ l·(N/8) + R0·h + a_0 ]   ==   out_old[ l·(N/8) + a_0·(N/32) + h ]
```

i.e. **each of the 8 leg sections is transposed** from a `4 × (N/32)` row-major array to
column-major. Nothing else changes. Verified at `0.00e+00`, all chains, both reviewers.

**Permissibility — checked, not assumed:**
* `include/vfft.h:108-118` defines `VFFT_ORDER_SCRAMBLED` as an explicit *"I am
  order-agnostic"* knob, MKL's `DFTI_BACKWARD_SCRAMBLED` intent, and tells
  roundtrip/convolution consumers to keep DEFAULT because *"order is irrelevant there"*.
* `vfft.c:2339` reaches the cascade only when `cfg->order == VFFT_ORDER_SCRAMBLED`;
  `vfft.c:3493-3502` is the sole consumer and requires only that fwd/bwd be **matched**.
* The permutation is **already chain-dependent today** — a chain change reshuffles it —
  so this is not a new hazard class.
* The shipped order gate (`build_tuned/test/natorder_oop_order_test.c:85-99`) asserts
  "not natural" plus roundtrip; it does not pin the permutation.
* **The order CLASS does not change.** Output remains a permutation of X, still
  SCRAMBLED, never natural.

**And it is unavoidable:** preserving `out_old` bit-for-bit would force the terminator's
lanes to be `a_{nf-2}`, hence `a_{nf-2}` innermost, hence stage `nf-2` is the bad stage —
**that is Lever 1**, which is refuted [M4, M5]. Worth writing down: it explains why
§4.9996 could not find this move.

**Mitigation:** `lane0`/`route` is carried in the plan and asserted in **both** execute
paths so a legacy-fwd + zturn-bwd pairing is impossible to express. Both routes stay
linked; the legacy route is the permanent fallback and the permanent A/B control.

---

## 3. WHAT'S NEW vs WHAT'S REUSED — file by file

### 3.1 Reused, byte for byte — no regeneration

| artifact | why it survives |
|---|---|
| `radix{4,8}_z_msg_avx2.c`, `radix{4,8}_z_msg_bwd_avx2.c` | **[D, verified]** the wrapper consumes only `(Ls, Gs, count)` + a 64-B-record cursor; `{D[s]}_{s≥1}` is a valid weight set for `(a_1…a_{nf-1})` (Step 4), so the same integers work. Twiddle *contents* change; the objects do not. |
| `emit_tr4` (`codelet_zsplit.ml:309-337`) | reused **unmodified** by the scaffold ingest (Phase 2) |
| `E_z` DEINT/REINT edges (`:398-429`, `:523-555`) | `stf` keeps the REINT store verbatim; `stfb` keeps the DEINT load verbatim |
| `E_blocks` + `blk_addr` (`:339-343`, `:556-590`) | the scaffold ingest's store edge, and the address form the interleaved ingest also lands on (`2*radix = 8` at `R0=4`) |
| `leg_addr` (`:346-359`) | ingest loads, with `plus = 4*h` |
| the 11-arg z ABI (`:615-632`) | unchanged. `zin`/`zout` are already distinct `__restrict__` pointers |
| `Dft`, `Algsimp`, `Pipeline`, `Schedule`, `Emit_c`, `Isa`, `expr.ml` | **no IR change.** Output slots are bare ints (`expr.ml:42-45`); every address is string-formatted in the edge emitter. Any claim that "the IR can't express this" is wrong |
| `radix{4,8}_z_s0s*`, `radix8_z_sterm*` | left in tree, still emitted, still linked — the legacy route is the fallback and the control |

### 3.2 New — emitter (`src/dag-fft-compiler/generator/lib/codelet_zsplit.ml`)

**(a) `emit_codelet` gains `~(r0 : int)`.** Plan INPUT, never chosen by the emitter — the
same contract `codelet_cil.ml:1030-1039` already states for `~chain`.

**🔴 (b) `fname` MUST carry `r0`.** Currently (verified by reading):

```ocaml
let fname = Printf.sprintf "radix%d_z_%s_%s_%s" radix k.base dir_s isa.Isa.name in
```

`r0` is baked into `stf`'s addresses (`64*h + 8*l`) and into its column expression
(`R0*h`), so `stf@r0=4` and `stf@r0=8` are two mathematically different kernels that would
share the symbol `radix8_z_stf_fwd_avx2`. **That silently ships a wrong FFT the moment
tranche 2 lands.** Change to `radix%d_z_%s%d_%s_%s` (or append `_r%d` only for the
r0-dependent kinds). One line; it is the only finding in this review with wrong-answer
consequences.

**(c) New edge `E_gblk of int` — group-flat contiguous block edge (~15 lines).**

```ocaml
let gblk_addr buf leg off =
  Printf.sprintf "%s[%d*(size_t)h + %d*%d + %d]" buf (2*radix*r0) (2*r0) leg off
(* radix 8, r0 4  ->  zin[64*(size_t)h + 8*l + 0] / + 4 *)
```

In the `in_edge` match it emits one plain `loadu` pair per leg into `lane_re_<l>` /
`lane_im_<l>` — structurally the existing `E_planes` arm with a different address string.
In the `out_edge` match it emits plain `storeu` (the `stfb` side). `uses` returns `false`
(no runtime stride), like `E_blocks`, so the computed `(void)` list stays correct.

**(d) `leg_addr` gains an optional `~col_expr:string` (default `"k"`).** `stf`'s comb store
passes `"4*(size_t)h"` (= `R0*h`). The `E_z` REINT code is otherwise untouched.

**(e) New edge `E_zu` — unpack-only DEINT (~10 lines), scaffold ingest only.** Drops the
`permute4x64 0xD8` from the `E_z` load edge; the resulting lane order `(0,2,1,3)` is
absorbed by a compile-time column relabel in the paired `E_blocks` store (`blk_addr`'s `c`
is an emit-time int). Cost 8 p5 instead of 16 on the load side.

**(f) New emitter-local template `emit_ileaf ~radix ~bwd ~isa ~uarch` (~120 lines OCaml →
~60 lines C).** The interleaved twiddle-free leaf. It **cannot** go through
`emit_col_loop`: that path consumes `re_tag.(sl)`/`im_tag.(sl)` built from
`Expr.Output(l, is_real)` (`:494-503`), and the IR models re/im as **separate** slots,
which an interleaved 2-complex register does not have. Emitted as a closed-form template,
exactly as `emit_tr4` already is. Reuses `emit_signature`, `leg_addr`, `blk_addr`,
`Isa.intr`, `Isa.const_decl`, and the existing pinned `xor_pd` (−0.0 contract).

> **Honest caveat, tagged [A]:** the usual justification — *"the leaf is twiddle-free, so
> Algsimp/fma_lift has nothing to do"* — addresses **Algsimp**. It says nothing about
> `Schedule.su_schedule`, whose job is ordering 8 loads / 16 add-sub / 18 shuffles / 8
> stores under register pressure, and this project's own record attributes spill and move
> garbage to losing the SU+GH order. **Static spill/regmov census against
> `radix4_z_s0s_avx2.c` (16 loadu/storeu, no spills) is mandatory before any timing.**

**(g) Kind table, 5 new entries (`:100-129`):**

```ocaml
| "s0t"   -> { mid with base = "s0t"; twiddled = false }                       (* ileaf template *)
| "s0tb"  -> { mid with base = "s0t"; twiddled = false; bwd = true }
| "stf"   -> { mid with base = "stf";  policy = TP_PowW1; tw_off = "8*(size_t)h"
             ; in_edge  = E_gblk r0; out_edge = E_z "OLs" }
| "stf2"  -> { … as stf, base = "stf2"; uj2 = true }
| "stfb"  -> { mid with base = "stf"; bwd = true; policy = TP_PowW1
             ; tw_off = "8*(size_t)h"
             ; in_edge  = E_z "OLs"; out_edge = E_gblk r0 }
```

Note `base` is **`"stf"`, not `"sterm"`** — deliberately. Both executors must be linkable
simultaneously (for the A/B, for the per-cell wisdom fallback, and for rollback). Since
`fname` derives from `k.base`, reusing `"sterm"` would **overwrite the incumbent files**.
Consequence to handle: the radix-8 guard at `codelet_zsplit.ml:139` tests
`k.base = "sterm" || k.base = "sterm2"` and the header-comment match at `:241-261`
likewise — both must learn the new stems, or `stf` silently escapes the radix gate and
lands on the generic `"ms"` header arm.

**(h) Loop shell.** `stf` runs a flat `for (size_t h = 0; h < count; h++)`; `stf2` runs
`h += 2` with instance B at `+1` group = `+64` doubles and `+1` twiddle record. The
existing `~open_line` parameter carries this; `emit_col_loop`'s body is reused as-is.

**⚠ `stf2` gate:** the uj2 instance-B column offset is `colo = (sl/radix)*vw = 4`, which
equals "one group" **only because `vw == R0 == 4`**. Tranche 1 must assert `r0 == vw`;
tranche 2 (`R0 = 8`) needs an 8-column unroll and this coincidence breaks.

**(i) `uses` / `plain_voids` (`:594-613`)** — add `E_gblk` ⇒ `false` (voids `Ls`), `E_zu`
⇒ `true` on its stride name.

### 3.3 New — `gen_main.ml`

Flags added to the `--zp-*` family block (`:418-435`) and dispatch (`:1647-1653`):

```
--zp-s0t  --zp-s0tb  --zp-stf  --zp-stf2  --zp-stfb        --zp-r0 <N>   (default 4)
```

Family mutual-exclusion (`:1550-1574`) needs the new names registered. Generation, WSL
absolute path only — **never a bare `dune build`**:

```sh
DUNE_CACHE=disabled /home/tugbars/.opam/5.2.0/bin/dune build \
    --root /mnt/c/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/generator
./gen_radix.exe 4 --zp-s0t  --zp-r0 4 --isa avx2 --uarch raptor_lake_avx2 --emit-c
./gen_radix.exe 4 --zp-s0tb --zp-r0 4 --isa avx2 --uarch raptor_lake_avx2 --emit-c
./gen_radix.exe 8 --zp-stf  --zp-r0 4 --isa avx2 --uarch raptor_lake_avx2 --emit-c
./gen_radix.exe 8 --zp-stf2 --zp-r0 4 --isa avx2 --uarch raptor_lake_avx2 --emit-c
./gen_radix.exe 8 --zp-stfb --zp-r0 4 --isa avx2 --uarch raptor_lake_avx2 --emit-c
```

One family flag per invocation (the families emit byte-identical symbol names and dispatch
is fixed-priority). **5 new files** in `src/dag-fft-compiler/codelets/zil/avx2/`.

### 3.4 New — runtime: `src/core/oop/zturn.h` (a new header, NOT a mutation of `zsplit.h`)

Precedent: `il2p.h` alongside `oop_plan.h`. Keeping `zsplit.h` untouched means the shipped
path cannot regress and rollback is a one-line dispatch change.

```c
typedef struct {
    int N, nf, r0, t2q, tcut;
    int  chain[VFFT_ZSPLIT_MAX_NF];
    long D[VFFT_ZSPLIT_MAX_NF], G[VFFT_ZSPLIT_MAX_NF];
    double *twsp[…], *twspb[…], *twq, *twqb;
    double *sp;              /* ONE plane, N complex, 64-B aligned — as today */
} vfft_zturn_plan_t;
```

`vfft_zturn_create(N, chain, nf, r0)` guards:

```
chain[0] == r0 && r0 == 4 (tranche 1, and r0 == VW)     chain[nf-1] == 8
prod(chain) == N                                        nf ∈ [3, MAX_NF]
(N/(8*r0)) % 2 == 0  if t2q                             D[0] % 4 == 0
```

Plus, at create time and not optional given the `il2p.h` history:
1. the Step-8 twiddle cross-check assertions;
2. a one-shot roundtrip `bwd(fwd(x)) == N·x`;
3. a one-shot comparison against the legacy route **under the leg-section transpose**,
   required to be **bit-identical** (`memcmp`, not a tolerance).

`gb[]` is **not** carried over (dead — filled at `zsplit.h:152-155`, never read).

`vfft.c` gains a symmetric zturn block beside the existing `zsplit` block at `:2338-2392`
and `:3491-3502`, plus a `route` assert in both execute paths.

**`zsplit.h` itself changes only in one place:** nothing. It stays the legacy route,
unmodified, forever, until/unless zturn wins every cell and someone decides to retire it.

---

## 4. PLANNER + WISDOM

### 4.1 A prerequisite bug that must be fixed first

**Verified by reading `src/core/planning/dp_planner_il.h`:**

* `:459-490` — the SCRAMBLED branch enumerates **every ordered `{4,8}` chain** for
  `nf ∈ [3, MAX_NF]`, validated by `vfft_zsplit_create` ("the validator is the law"), and
  pushes 2 candidates per chain (`t2q ∈ {0,1}`).
* `:537-549` — a **single** `have_ref` flag spans all candidates of one `ord` class:
  the first candidate that runs becomes the reference; every later candidate is compared
  against it at `VFFT_IL_DP_GATE_TOL = 1e-12` and dropped on mismatch.
* `:315-318` — the comment asserts *"Candidates are compared WITHIN a class against the
  first one that ran — same order contract, so they must agree to rounding."*

**That assumption is false.** Different chains produce different digit-reversal
permutations (`drev` depends on the chain radices). So every SCRAMBLED chain except the
first legal one **fails the gate and is silently dropped**, and the cascade chain search is
effectively benching one candidate.

Scope note: this is **not** a blocker for racing the routes — `vfft_il_dp_plan` has no
call sites outside `dp_planner_il.h`, and the shipped create path uses
`vfft_zsplit_default_chain` + `_calibrate_zsplit_t2q`. It **is** a blocker for the chain
re-search in §5 Phase 5, and it is worth fixing on its own merits.

**Fix (permutation-agnostic, ~30 lines):** replace the class gate with
(a) roundtrip `bwd(fwd(x)) == N·x` to 1e-13 relative, plus
(b) a permutation-aware spectral gate — the plan knows its own chain and `r0`, so build the
`out_index → natural bin` map from `_vfft_zs_brev` at gate time and compare against ONE
natural-order reference. Both gates are chain- and route-agnostic, which is what makes
cross-chain **and** cross-route ranking legal.

### 4.2 The search space

Extend `vfft_il_cand_t` with `int route` (`LEGACY_ZSPLIT | ZTURN`). Enumerate, for
`ord = SCRAMBLED`:

```
route ∈ {LEGACY, ZTURN}
  × chain: ordered {4,8}^nf, nf ∈ [3, MAX_NF], prod == N,
           validated by THAT ROUTE'S OWN create()   (ZTURN additionally: chain[0] == 4)
  × t2q ∈ {0,1}
```

Candidate count at N=16384: ~6 legal chains × 2 routes × 2 t2q ≈ 24 — well inside
`VFFT_IL_DP_MAX_CAND = 64` (checked). No bump needed until tiling adds a `tcut` axis.

**Nothing is composed.** `_il_dp_run_once` already *builds* each candidate and
`_il_dp_bench` *measures* `vfft_*_execute_fwd` end to end. DP prunes the search; it never
composes costs. `r0` is **not** a free knob — it is `chain[0]`, and the chain comes from the
search.

**The chain must be re-searched, not transplanted.** The new layout changes the mids'
argument shape (`Ls` ×4, `Gs` ÷4, inner trip count ×4, group span ×4, leg stride ×4) and
shrinks the twiddle tables 4×. All of that moves the per-stage cost, so a chain that lost
under the legacy layout can win under ZTURN. Shipping ZTURN with the old
`vfft_zsplit_default_chain` winners would violate the measured-search rule in spirit.
Re-run with `build_tuned/benches/zil_chain_dp.c` (or `vfft_il_dp_plan_and_bank` per cell)
at 2048/4096/8192/16384 with **both** routes available.

**Terminal radix stays LOCKED at 8.** The radix-16 terminator is census-REFUTED (three
independent confirmations), and MKL's own radix-8 finisher downgrades its turn to
`vextractf128`-to-memory — 48 p5 + 32 store uops per 32 complex vs radix-4's 32 + 16 —
*"a cost, not a saving"*. Nothing here reopens that axis.

### 4.3 The wisdom key — extend the integer, not the grammar

Verified: `oop_wisdom.h:62` grammar is `N 1 4 zs_t2q cc_chain ns`; `:124` reads the token
with `atoi`; `:288` writes it with `%d`. So **widen `zs_t2q` into a bitfield with zero
reader/writer change:**

```
bit 0      : t2q          (0 = sterm/stf single-quad, 1 = sterm2/stf2 two-quad)
bit 1      : route        (0 = LEGACY zsplit, 1 = ZTURN)
bits 2..4  : tcut         (RESERVED for the tiling lever; 0 today)
bits 5+    : reserved
```

* **Existing banked wisdom is NOT invalidated by the format.** Every shipped line carries
  0 or 1 ⇒ bit1 = 0 ⇒ legacy route with its banked `t2q`. Backward compatible in both
  directions; an old binary reading a new line gets legacy + `t2q = 1`, i.e. a
  suboptimal schedule, never a wrong answer.
* **Existing banked wisdom IS invalidated by placement.** Any binary containing the five
  new symbols is a full relayout, and §4.9993 established that `sterm`-vs-`sterm2` is
  code-placement luck (*"the wis-gate binary banked sterm2 at all 4 cells; the api-gate
  binary banked 0/1/1/0"*). **Every `t2q` verdict must be re-raced on the installed
  binary after this lands. Never carry a `t2q` line across this rebuild.**
* Reader change is exactly one line at `vfft.c:2373`:
  `zs_pending->t2q = ze->zs_t2q ? 1 : 0;` →
  `t2q = ze->zs_t2q & 1; route = (ze->zs_t2q >> 1) & 1;`
* Writer: `dp_planner_il.h`'s `vfft_il_dp_emit_wisdom` encodes `t2q | (route << 1)`.
* A stale line asking for ZTURN with `chain[0] == 8` must make `vfft_zturn_create` return
  NULL; the existing *"stale/invalid wisdom chain: fall back"* path at `vfft.c:2364-2371`
  already handles that.
* **No new wisdom file.** Reuse `oop_wisdom.txt` / `zsplit_wisdom.txt` per the
  wisdom-inventory rule.

### 4.4 Create-time race

Rename `_calibrate_zsplit_t2q` (`vfft.c:577-660`) → `_calibrate_zsplit_route`, extending
from 2 arms to up to 4 (legacy+sterm, legacy+sterm2, zturn+stf, zturn+stf2), keeping its
existing discipline: **alternating arm order** (`:630`), median of 9 (MEASURE) / 21
(PATIENT), **3 % hysteresis toward the compiled default**. Budget ~10 ms → ~20 ms.

**Its `memcmp` sanity check (`vfft.c:597-609`) must be restructured, not deleted.** It
compares the two arms' outputs, and legacy vs zturn produce different permutations, so a
cross-route `memcmp` will fail. Correct shape:
* `memcmp` **within** a route (sterm vs sterm2 are bit-identical by construction — keep);
* **plus** the create-time transpose-comparison and roundtrip self-checks of §3.4.

> Gap to close: `vfft.c:2373`'s **wisdom-HIT** path installs a route with **no**
> validation. The create-time self-checks of §3.4 must run on the hit path too, or a stale
> line silently installs an unvalidated map.

---

## 5. PHASING

Each step ends in something measurable. Step 0 is the cheapest thing that could kill the
whole design.

### Phase 0 — THE KILL TEST — ✅ MATCH, 18/18, 2026-07-26

**Result: 18 port-5 ops per 16 complex on BOTH gcc 13.2 and gcc 15.2 — equal to MKL's
shipping ingest. First transcription attempt, zero iterations needed.**
Spike: `build_tuned/benches/spike_s0t_ingest.c`.

* Census (both compilers, identical vector classes): 4 `unpcklpd` + 4 `unpckhpd` +
  4 `perm2f128` + 4 `insertf128` + 2 `permilpd` = **18 p5**; 2 `xorpd`; 16 add/sub;
  8 loads + 8 stores; **zero spills**; mask hoisted pre-loop (register-only in-loop).
* Correctness: bit-identical to the builder's scalar map reference at 2048/4096/16384;
  independently confirmed **non-circularly** by a verifier-written naive-DFT reference
  (different association order, maxabs 1.8e-15 / 8.9e-16, incl. N=8192 which the builder
  never ran), plus impulse traces that pin the eq-(3) address map AND the forward `−i`
  rotate direction.
* vs MKL: same totals (18 + 2), different mix (MKL: 10 `shufpd` + 8 `perm2f128` — see the
  §2.4 erratum). Total insns/iter **52/56 vs MKL's 59** — we win the scalar bookkeeping
  (two linear cursors vs MKL's index-table cursor) because SCRAMBLED needs no offset table.
* ⚠ For any later **timing**, use the **gcc 15.2** object: gcc 13.2 folds 4 leg loads
  into add/sub memory operands and double-reads 4 addresses (20 accesses vs the faithful
  8+8). Same p5, same correctness, different load stream.
* Residual risk, stated honestly: builder and verifier both derived from §2.4, so a
  misreading shared by the DOC ITSELF would not be caught — the Phase-2
  `memcmp`-vs-legacy-plane gate is the check against the incumbent and remains mandatory.

*(Original phase spec follows, retained for the record.)*

**Static op census of the interleaved ingest body.** Hand-write it as a ~40-line C
function, compile `-O3 -mavx2 -mfma`, count port-5-class ops in the objdump per
16-complex iteration:

```
required : vunpcklpd 4 · vunpckhpd 4 · vperm2f128 8 · vpermilpd 2 · vxorpd 2
           vmovupd 8 loads + 8 stores
port-5 total = 18 per 16 complex = 1.125/complex
```

| reading | verdict |
|---|---|
| 18 | **MATCH** — MKL's own count (census `mkl_highN_cascade_anatomy.md:171,190`: r4 ingest, shuf+xor 20 per 16 complex). Proceed |
| 19–20 | **PASS** — within reach of the reference; proceed, record the diff |
| ≥ 21 | **OUR LATTICE IS WRONG — iterate, do not conclude.** MKL achieves 18 on this machine (fn `0x3800`, ingest `0x38c5..0x39f0`, disassembly on file). A worse count from our hand-written body cannot falsify a method a shipping binary demonstrates; it is a defect report on the body. Diff op-class-by-op-class against the census row and the required sequence above, fix, recount |

> **Directive (2026-07-26, Tugbars):** this test has NO kill outcome. MKL is the
> existence proof for the fused lattice; the only thing a bad reading can indict
> is our transcription of it. This project's record backs the asymmetry: every
> measurement error to date has flattered the incumbent structure. The original
> "≥ 25 KILL" row was an agent-invented band and is retired.

Also record `vmovupd` count vs `radix4_z_s0s_avx2.c`'s 16 (a jump ⇒ spills) and check the
sign mask is hoisted, not rematerialized per iteration.

**Why this is the right first test:** the early-kill spike explicitly left exactly one gate
open — *"the spike does not measure the ingest… if making `s0s` corner-turning costs ≥1
shuffle/complex on its store side, the move merely relocates the cost and the design
dies."* That gate is an **instruction count**, not a timing question, and this project's
standing warning is that timing at this effect size is where harness artifacts live.
Answer it with instructions.

**Fallback if it fails:** the ping-pong route is *measured to work on a
terminator-shaped pass* [M1, control 1.0034]. It should not be re-proposed as the primary,
but it should not be discarded as a named fallback either.

### Phase 0b — ✅ MEASURED 2026-07-26 — terminator CONFIRMED, ingest gate FAILS at ≥4096 with a named suspect

Harness `build_tuned/benches/bench_phase0b.c` (real `zsplit.h` plan/twiddles, per-cell
smoke gates incl. cascade bitcmp, full discipline; logs in scratchpad `p0b/`).
**Statistic: MEDIAN of 12–54 rotated reps** — best-of minimums chronically tripped the
control band on large-N `sterm` (thermal-tail min-sampling); medians pass 0.983–1.016 on
EVERY run. One 8192 run excluded on ABSOLUTE level (med 7858 ns vs 4430 elsewhere =
degraded clock; its control passed — a passing control validates internal consistency,
not machine state).

| pair | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|
| **B terminator** `stermt/sterm` | 0.80–0.84 | **0.705** | **0.690/0.710** | **0.688/0.690/0.691** (3 runs) |
| **A ingest** `s0t/s0s` | **0.890/0.893/0.898** (3 runs) | 1.355 | 1.337 | 1.353 |

* **[M2] re-confirmed under passing controls; the 4096 anomaly (0.958) VANISHED**
  (now 0.705–0.722) — it was the marginal control, as §1.3 suspected.
* **Gate A (≤1.20) FAILS at ≥4096** — but ~1.35 cannot be the shuffles: the +2 p5 bounds
  the port-5 charge at 1.125 even if the ingest were p5-bound, and it is not. The charge
  is MEMORY-side, it FLIPS SIGN at 2048 (ingest an 11% WIN there, ×3 reproduced), and it
  sits exactly on our one known divergence from MKL: **ρ makes the ingest's stores one
  LINEAR cursor; MKL's observed ingest SCATTERS one record per plane section.** Per the
  Phase-0 presumption order this is under diagnosis (store-pattern attribution matrix +
  MKL store-addressing re-read), NOT a method verdict.
* Fallback economics if 1.35 stood as-is: terminator −10…−13% of transform vs ingest
  charge ≈ +6% (17% leaf share) — still net-negative, and 2048 wins on both stages.

#### Phase 0b addendum — ✅ DIAGNOSED same day: the charge was ρ's LINEAR STORE CURSOR, not the turn. **Amendment ZTURN-S adopted.**

Attribution matrix (`build_tuned/benches/bench_ingest_diag.c`, 2×2 lattice×stores, all
arms bit-exact vs own scalar refs, censuses c≡b / d≡a on every vector class; medians,
logs `diag_*.log`):

| N (hot) | s0t = turn+LINEAR | **s0t_sect = turn+MKL geometry** | s0s_lin = no-turn+LINEAR | ctl |
|---|---|---|---|---|
| 2048 | 0.883 | 0.949 | 1.255 | ✅ |
| 4096 | 1.338 | **1.003** | **2.146** | ✅ |
| 8192 | 1.353 | **0.978** | **2.150** | ✅ |
| 16384 | 0.929 | **0.992** | 1.141 | ✅ |

* **Store column guilty, lattice innocent**: no-turn+linear alone reads 2.15; turn+sectioned
  reads ≈1.00. Cold (one-shot, >2×L3 ring): linear arms 1.25–1.41, sectioned 1.04–1.07.
* **Mechanism (predicted, then fingerprinted)**: s0t's single cursor advances 256 B/iter =
  4× the leg-load rate ⇒ its mod-4K offset SWEEPS the loads ⇒ periodic 4K false aliasing
  whose cost = store DRAIN latency (L1 at 2048 ≈ free; L2-RFO at ≥4096 ≈ 1.35×). Skew
  probes match: skew=0 craters the rate-matched arms (constant overlay), skew=2048 changes
  nothing for the sweeping arm. No placement fixes a sweeping cursor.
* **MKL asm (OBSERVED, `mkl_il_512_disasm/mkl_highN_cascade_suite.asm`)**: output digit IS
  innermost in the 64-B record (ρ right there); sections at {0,4N,8N,12N} B take
  record→section **bitrev2(p mod 4)** with a SHARED +0x40/iter cursor — four rate-matched
  streams. MKL permutes the LOADS (index table; we don't need one — SCRAMBLED), keeps
  store rates steady. Their finisher taps the 4 sections N/4 apart, contiguous within each
  tap ⇒ **the turn-free terminator survives under this geometry.** Also OBSERVED: with OOP
  + 32B-aligned output, MKL's plane IS the user output buffer (scratch-free cascade) — a
  separate future lever.
* **ZTURN-S**: keep record interior (a_0 lanes, split re/im); replace the linear plane
  order with MKL's sectioned geometry. §2.4's map, the mids' arg rescale, and stf's load
  addressing must be RE-DERIVED against this layout in Phase 2 (the memcmp gate catches
  any slip). Arm `c` of the diag harness is the proven ingest reference body.
* **Honesty items**: the 2048 hot "win" (0.89) is REGIME-BOUND — cold reads 1.25 linear /
  1.06 sectioned; expect ≈parity in production, not a win. Hot-16384 s0t read 0.93 here vs
  1.35 in Phase 0b (arena steady-state differs at the L2 boundary) — does not affect the
  verdict (4096/8192 agree across harnesses; sect ≈1.0 in both). Cold controls at
  4096/16384 read 1.06/1.19 (VOID) — cold conclusions rest on the passing 2048/8192 cells.
* **Updated economics**: ingest ≈ FREE (1.00 hot / ~1.05 cold) instead of +1.35 ⇒ net
  forward projection back to **−10…−15%**.

*(Original phase spec follows.)*

Two paced isolated A/Bs, `zil_sterm_pipe.c`-shaped harness (§6), **both with control arms**:

* **arm pair A** — ingest: `s0s` (today, 16 p5) vs `s0t_interleaved` (18 p5), production
  shape `Ls = count = D[0]`. **This is the term nothing has measured.**
* **arm pair B** — terminator: `sterm` vs `stf`, production shape. This **re-runs [M2]
  with a passing control** and firms up the 0.689/0.958/0.743/0.781 numbers, and resolves
  the 4096 anomaly.

**[D] `stf` at `r0 = 4` is byte-identical in addressing to the already-measured `stermt`**:
`zin[16k + 8l]` with `k += 4` ≡ `zin[64h + 8l]` with `h = k/4`. So [M2] transfers — but its
control was 0.99–1.09 and this project's standard is ~1.00.

**Gate:** `s0t/s0s ≤ 1.20` at every cell. Above that, the ingest charge eats the
terminator win at N ≥ 8192 and the design must be restated as a small-N-only lever.

> **Presumption order if this gate fails while the Phase-0 census reads 18:**
> harness first, our kernel body second, the method last. MKL pays these same 18
> ops in-situ, profitably, in the faster transform. Every wrong number in this
> project's record has favoured the incumbent; check the control arm, the SMT
> sibling, and the spill census before restating the design.

### Phase 1 — fix the DP gate bug — ✅ DONE 2026-07-26

Fixed in `dp_planner_il.h` (+258/−25), stronger than specified: independent scalar
reference per cell with the candidate's own permutation applied, explicit non-finite
bail (the old gate passed an all-NaN buffer), tolerance held at 1e-12 (measured accept
band 8.6e-17…7.7e-16). Benched candidates went 2 → 8/10/14/18 at the four cells;
fault-injection rejects 12 defect classes + 266/266 cross-chain wrong-permutation
cases. Winners are now 4-heavy like the calibrated defaults (frozen cand-0 chains were
8-heavy). t2q twins proved bit-identical ⇒ the frozen axis was the CHAIN, and the
t2q=0/1 wisdom disagreement is the create-path `_calibrate_zsplit_t2q` ±5% placement
axis, as originally documented. No banked wisdom originated from the broken search;
nothing voided. `dp_planner.h` (split) has no comparison gate at all — correctness
defers to the roundtrip gate at `calibrate.c:55-71` — so the defect cannot exist there.
⚠ `ns` from this planner is not comparable across the fix (per-cell bench count
changed ⇒ different thermal regime under the `% 4` pacing).

### Phase 2 — emitter tranche A + the scaffold route (correctness at production scale)

Build `E_gblk`, `~col_expr`, the `stf`/`stf2`/`stfb` kinds, the `fname` r0 fix, the
radix-8 guard/header-comment stems, and `zturn.h` **with the 24-op scaffold ingest**
(`E_zu` + existing `E_blocks`, 8 + 16 p5).

**Why a scaffold ingest:** it exercises the entire index map, the entire twiddle repack and
the entire runtime at production scale with **zero new emission path**, so any map error
surfaces before the hand-written interleaved leaf exists. **[D]** It is roughly break-even
on speed (+0.50 at the ingest, −1.00 at the terminator) and **must not be shipped on its
own**.

**Gates:**
* create-time twiddle assertions: 0 violations;
* zturn output == legacy output under the leg-section transpose: **`memcmp` == 0**, both
  directions, all four cells (identical DAG, identical twiddles, identical FP order — only
  addresses differ, so exact is the right bar, not 1e-15);
* roundtrip at the shipped 1.1–1.4e-15;
* static census: `stf` loses exactly 16 `unpck` + 16 `perm2f128` vs `sterm` (64 → 32 p5);
  `regmov`/spill within ±2 of the incumbent.

**Also measurable here:** per-pass timing inside the scaffold route gives the **in-situ**
leaf and terminator costs under the new layout, which tells you exactly what the
interleaved ingest has to deliver.

### Phase 3 — emitter tranche B: the interleaved ingest

`emit_ileaf` for `s0t`/`s0tb`, swapped in for the scaffold.

**Gates:**
* the emitted plane must be **bit-identical** to the scaffold ingest's plane (`memcmp`);
* **[A → to verify]** bit-identity with the split leaf rests on IEEE add commutativity at
  one particular association order and `a − (−b) == b + a`. It holds at `r0 = 4`; it will
  **not** survive `r0 = 8` (ω₈ rotations, `1/√2`, Algsimp's `fma_lift` on the split side).
  **State the gate as `memcmp` for `r0 = 4` and 1e-15 for `r0 = 8`** — otherwise tranche 2
  fires a false failure. Transcribe `radix4_z_s0s_avx2.c:48-71` operand-for-operand and
  diff the emitted arithmetic before believing any timing;
* static spill/regmov census vs `radix4_z_s0s_avx2.c`.

### Phase 4 — THE DECISIVE MEASUREMENT (§6)

### Phase 4b — mids-only aliasing A/B (mandatory, separate, own control)

**[D §1.4]** the new layout adds one exact-4096-byte-stride mid at every cell. Race
`msg` at legacy args vs `msg` at zturn args — **the same object**, only `(Ls, Gs, count)`
and the table differ — so the only independent variable is the address map. Predicted
regression is unknown; the mids are 49 % of the transform.

### Phase 5 — planner + wisdom + chain re-search

Route axis into `_il_dp_enumerate`; bitfield into the kind-4 token; 4-arm create-time race;
re-run the chain search at all four cells with both routes; update
`vfft_zsplit_default_chain`'s zturn twin from the banked lines. **Do not hand-pick a chain
to suit the turn shape.**

### Phase 6 — padded plane pitch (no longer optional), then tiling (speculative)

* **Pitch:** Lever 2 measured −4.5…−18.9 % on the leaf, −0…−6 % on the terminator, ~2–4 %
  overall, clean v2 methodology, **never built** [M10]. Given Phase 4b's aliasing shift it
  becomes a repair, not a bonus. Add `p->pitch`; derive per stage (the leg strides differ
  under `ρ`), never guess. **Measure as a separate arm** so it cannot contaminate the route
  verdict.
* **Tiling** (`tcut`): because `ρ` puts `(a_1…a_tcut)` in the HIGH bits, a contiguous chunk
  of `T = R0·D[tcut]` complex is **closed** under every stage `s > tcut`, so stages
  `tcut+1 … nf-1` run inside an L1-resident tile with a ~15-line driver and **no new
  kernels**. Per-tile twiddle slices are contiguous base offsets, never rebuilds.
  Correctness verified during review (`0.00e+00` at every legal `tcut`).
  **[A] The payoff is entirely unmeasured.** It removes **zero** shuffles, **zero** FLOPs
  and **zero** compulsory bytes; it only converts plane↔L2 traffic
  (`16N·(2nf−2)` → `16N·(2·tcut+3)`, 2 MB → 1.25 MB at N=16384/nf=5/tcut=1). Its own
  falsifier: **exactly 0 at N=2048**, where the 32 KB plane is already L1-resident. Exclude
  `tcut = nf-2` (T = 32 complex ⇒ 512 one-group calls at N=16384: pure overhead). Measure
  with its own control arm; its predicted effect sits inside the ±5 % placement band.

---

## 6. THE DECISIVE MEASUREMENT

### 6.1 What is raced

**Whole-transform, front-door, both directions, one binary containing both routes.**

| arm | content |
|---|---|
| **L** | `vfft_zsplit_execute_fwd` — legacy: `s0s` + `msg`×(nf−2) + `sterm`/`sterm2` |
| **Z** | `vfft_zturn_execute_fwd` — new: `s0t` + `msg`×(nf−2) + `stf`/`stf2` |
| **C** | **CONTROL** — arm L again, in the other rotation slot. **Must read 1.00 ± 0.01** |
| **M** | MKL `DftiComputeForward`, same cell, for the absolute position |

Repeat for `_bwd` (arms `Lb`, `Zb`, `Cb`, `Mb`). Region-level per-pass timing is collected
as a **diagnostic only**; whole-transform time is the ground truth.

### 6.2 Harness discipline — non-negotiable

Model on `build_tuned/benches/bench_1d_vs_mkl.c` and `zil_sterm_pipe.c` (the only campaign
bench with proper controls).

1. **Pin logical core 2** (affinity mask `4`), `HIGH_PRIORITY_CLASS`.
2. **64 MB cachebust** between arms. **Not 32 MB** — [M1] found 32 MB does not sweep this
   chip's 36 MB L3 and planes survived it.
3. **Rotate arm order**, and **force reps to a multiple of the arm count** so every arm
   visits every rotation slot equally. This is the fix that took [M1]'s control from 1.13
   to 1.0034; without it, best-of-N silently re-admits intra-rep thermal drift.
4. **`Sleep()` pacing** between arms (50 ms) and rounds (150 ms). Thermal drift has flipped
   winners in this project.
5. **Best-of-9** (21 for the final PATIENT run).
6. **ONE shared output buffer** for all arms. Per-arm output buffers once fabricated a
   bogus 1.51×.
7. **ONE arena with a 64 B skew** for every plane and buffer. Two page-aligned allocations
   cause 4 KB aliasing and bimodal timings — this trap has been hit twice, and §4.9995(c)
   measured 27 % from allocation alignment alone.
8. **CONTROL ARM mandatory.** If C ≠ 1.00 ± 0.01, the harness is lying and the numbers are
   void. Report the control reading with every result.
9. **Isolated cell-per-process**, JIT/plan built at plan time.
10. **Two compilers** (MinGW gcc 15.2 = `build.py`'s dev compiler, and gcc 13.2 as
    cross-check; gcc-11 is not present on this host — verified). **Three runs.** Any verdict
    that does not reproduce across compiler and run is not a verdict.
11. Check for **bimodality** in the raw sample distribution before quoting any ratio.

### 6.3 Cells

`N = 2048, 4096, 8192, 16384` — the whole shipped tier — with **each cell's calibrated
chain** for the legacy arm and the **re-searched** chain for the zturn arm once Phase 5
lands (before Phase 5, run both arms on the same chain to keep the comparison clean).

### 6.4 The number that decides — ✅ **MEASURED 2026-07-27: GO**

`bench_zturn_final.c` (smoke-EXACT before every timed sample; nm-verified shared msg
objects; deterministic mod-4K arena; medians of 15–21 rotated reps). Two independent
15-rep runs + patient 21-rep runs at the contested cells. Only ctl=PASS readings quoted;
ranges span all passing readings.

| N | fwd hot `r` | fwd cold | bwd hot | bwd cold | joint hot |
|---|---|---|---|---|---|
| 2048 | **0.865–0.893** | 0.94–1.00 | **1.036–1.042 (LOSS, real ×3)** | 1.039 | 0.966 |
| 4096 | **0.880–0.897** | 0.957 | 0.865–0.948 | 0.932–0.969 | 0.87–0.92 |
| 8192 | **0.883–0.963** | 0.93–0.99 | 0.962 | 0.951 | 0.91–0.96 |
| 16384 | **0.881–0.893** | 0.95–0.98 | 0.939–0.962 | 0.978 | 0.92 |

* **GO under the amended bar** (`r ≤ 0.95` at ≥3/4): 2048/4096/16384 clear it in every
  passing reading; 8192 wins in every reading (0.88–0.96) with magnitude varying by
  machine state. Forward beat the 0.99 forecast at 4096 outright.
* **2048 backward loses ~1.04 consistently** (hot and cold) — the predicted stfb-spill /
  s0tb-turn signature. Joint fwd+bwd still wins the cell hot (0.966); cold joint is a
  wash (~1.00–1.02). Phase-3's SU re-schedule of stfb is now aimed at a measured number.
* Per-cell ship decision goes to the planner's create-time race (Phase 5), as always —
  measured, never hand-set. Cutover atomicity respected: fwd/bwd flip together per cell.
* Machine-state note: absolute levels swung up to 47% between runs while ratios under
  passing controls reproduced within 1–2%; ratios are the only currency here.
* Projected vs MKL forward (composing with the 0.78–0.89 standing): **~0.86–1.03** —
  parity at 2048, 8–14% residual at the upper cells → Phase 6 levers (pitch, tiling,
  MKL's 16 KB blocked mids, scratch-free OOP).

*(Original decision table follows.)*

Let `r_N = Z/L` (forward) and `r'_N = Zb/Lb` (backward), lower is better.

> **Bar reset (2026-07-26, Tugbars): "even a 5% win is okay from our perspective."**
> The GO threshold below is amended from `r ≤ 0.94` to **`r ≤ 0.95`**; the PARTIAL band
> becomes `0.95 < r ≤ 0.98`. A clear, control-passing ~5% win productionizes. Note the
> resolution context: Phase-0b median ratios reproduced within ~0.5–3% across runs under
> passing controls, so a 5% effect is above the established noise floor — but cells
> landing 0.95–0.98 stay PARTIAL (per-cell ship via the searched route axis) because
> there they are within reach of the ±5% placement band.

| outcome | decision |
|---|--:|
| `r ≤ 0.95` at **≥3 of 4 cells**, control in band, reproducing on both compilers | **GO** — productionize, bank per-cell wisdom |
| `0.95 < r ≤ 0.98` | **PARTIAL** — ship per-cell (the route is a searched axis; losing cells keep legacy) but do **not** invest further until Phase 4b and Phase 6 (pitch) are measured; the aliasing shift is the prime suspect |
| `r > 0.98` anywhere with a passing control | that cell keeps legacy. Investigate the object, **not** more timings |
| `r > 1.00` at ≥3 of 4 cells | **NO-GO** — bank as a committed negative alongside [M4]/[M5]/[M6] |

**Predicted, DERIVED from [M2]×[M7] minus the derived ingest charge:**
`r ≈ 0.85 / 0.99 / 0.92 / 0.94` at 2048/4096/8192/16384. **Note 4096 and 16384 sit inside
or near the "partial" band, and 4096 is inside the ±5 % placement band §4.9993 declared
uninterpretable — which is exactly why the control arm and the two-compiler/three-run rule
are load-bearing here, and why Phase 0b's re-measurement of [M2] matters.**

The 2048 and 8192 cells carry the verdict.

---

## 7. OPEN QUESTIONS AND RISKS — ranked

| # | risk | tag | why it is ranked here | mitigation |
|---|---|---|---|---|
| 1 | **The interleaved ingest may not emit 18 p5.** Break-even is 32; a badly materialized sign mask, a spilled constant, or a serializing `vpermilpd` chain inflates it | **[A]** | the entire design rests on this one number, and nothing has measured it | **Phase 0**, an afternoon, before anything else. Fallback: encode the rotation as `permute_pd + addsub_pd`, moving one op off port 5 |
| 2 | **The new layout puts one MORE mid on an exact-4096-byte leg stride at every cell** (2048: 0→1; 4096/8192/16384: 1→2) | **[D]**, consequence **[A]** | the mids are 49 % of the transform and are today the clean part; [M10] measured this class of effect at −4.5…−18.9 % on a single pass | **Phase 4b** mids-only A/B with its own control, using the *same object* at both arg sets. **Phase 6** padded pitch, derived per stage, never guessed |
| 3 | **The 4096 cell's terminator delta is 0.958** where the other three read 0.689–0.781 | **[M2]**, unexplained | it drops that cell's projected win to +1.8 %, and 4096 is our worst cell vs MKL (0.76×) | **Phase 0b** re-measures with a passing control. Suspect: `t2q=1` default at 4096, or code placement |
| 4 | **[M2]'s control read 0.99–1.09**, against a project standard of ~1.00 | **[M]** quality | the terminator saving is the *whole* benefit and its magnitude is soft | **Phase 0b** re-runs it in the `zil_sterm_pipe.c` two-control shape |
| 5 | **`fname` does not carry `r0`** while `r0` is baked into `stf`'s addresses | **[D]**, verified by reading `codelet_zsplit.ml:150` | the **only** finding with wrong-answer consequences: `stf@r0=4` and `stf@r0=8` collide on one symbol at tranche 2 | one-line fix, in Phase 2, before any generation |
| 6 | **`emit_ileaf` bypasses `Schedule.su_schedule`.** The "twiddle-free ⇒ nothing for Algsimp" argument does not cover the SU scheduler | **[A]** | this project attributes spill and move garbage to losing the SU+GH order | static spill/regmov census vs `radix4_z_s0s_avx2.c` (16 loadu/storeu, 0 spills) as a Phase-3 gate. If it spills, halve the unroll — the lattice is per-half anyway |
| 7 | **The whole backward path is unmeasured.** Every backward claim is a mirror argument | **[A]** | it is half the deliverable and the platform rule is "both placements, both directions" | full gates in Phase 2/3/4; the bwd terminator's win is a **deletion** of the same 32 p5, so the fwd result should transfer, but it must be shown |
| 8 | **Bit-identity of the interleaved leaf is `r0=4`-only**, resting on add commutativity | **[D]** | a gate stated unconditionally will fire a false failure at tranche 2 | state the gate as `memcmp` at `r0=4`, 1e-15 at `r0=8` |
| 9 | **`chain[0] == 4` is a hard structural limit** and `zsplit.h:121` admits `chain[0] == 8` today | **[D]** | with `chain[0]=8` the zturn arithmetic still divides cleanly (`G[s] % 4 == 0`), so the failure mode is a **silently wrong FFT**, not a crash | the `create()` guard is load-bearing, not defensive. A `chain[0]=8` winner keeps the legacy route (the route is a searched axis, so the win is not guaranteed uniform across the tier) |
| 10 | **The terminator's 8 output streams are unchanged** — `2N` bytes apart, all 4096-multiples, 9 streams against 12 L1d ways | **[D]** | once the terminator stops being port-5 bound, this may become the new visible wall and the measured win could come in at the bottom of the band | Phase 6 pitch. This is the most likely reason for a disappointing first number |
| 11 | **The SCRAMBLED output permutation changes.** Any out-of-tree consumer that reverse-engineered the comb breaks silently | **[D]** | in-tree it is provably safe (§2.6); out-of-tree is unknowable | `route` in the plan, asserted in both execute paths; document `out_new` explicitly in `zturn.h`; both routes stay linked |
| 12 | **Placement luck.** Five new symbols guarantee a full binary relayout; every banked `t2q` line is void | **[M11]** | the predicted effect (6–14 %) is outside the ±5 % band, but the 4096 cell is not | re-race `t2q` per binary (the Phase-5 create-time race does this automatically). Never carry a `t2q` line across this rebuild |
| 13 | **`stf2`'s uj2 instance-B offset works only because `vw == R0 == 4`** | **[D]** | breaks silently at tranche 2 | assert `r0 == vw` in tranche 1 |
| 14 | **The wisdom-HIT path installs a route with no validation** (`vfft.c:2373`) | **[D]**, verified | a stale line silently installs an unvalidated map | run the §3.4 create-time self-checks on the hit path, not only the calibration path |
| 15 | **Tiling's payoff is entirely unmeasured**, and its predicted effect sits inside the placement band | **[A]** | it is the "tiled" angle's whole contribution and it could be zero | measure separately, with its own control, **after** the route verdict is banked. Its own falsifier: exactly 0 at N=2048 |
| 16 | **MT is untouched.** The plan holds one `sp` and is not thread-shareable — unchanged, but also not fixed | **[D]** | §4.9996's residual gap was "architecture **plus MT**" | do **not** claim the gap is closed. MT remains open |
| 17 | **Second-plane freedom is not being spent.** If Phase 0 kills the interleaved ingest, the [M1]-measured ping-pong terminator remains a real, measured asset | **[M1]** | discarding a measured asset because a derivation looked better is how sessions get lost | keep it named as the Phase-0 fallback |

---

## Appendix A — constraint compliance checklist

| constraint | status |
|---|---|
| Interior stays BLOCK-SPLIT | ✅ `ρ` permutes complex indices, not the `[re×4][im×4]` granule. No interleaved interior is proposed anywhere. Split `cmul` still needs no `cflip` |
| No new hybrid IL-boundary/split-interior codelets for the small-N / two-pass IL tiers | ✅ nothing in this plan touches those tiers. `s0t`/`stf` are the cascade's own z↔split boundary stages at N ≥ 2048, on the frozen 11-arg z ABI (`zin`/`zout`; **no `in_re`+`in_im` pair**, which is the stated test) |
| Plans/factorizations/radix chains come from MEASURED whole-plan search | ✅ §4. Route and `t2q` are searched axes; the chain is re-searched with both routes; `r0 = chain[0]` is not a knob; emitters take `~r0` and `~chain` as INPUT. The DP-gate bug that currently prevents an honest chain search is fixed in Phase 1 |
| Never a bare `dune build` | ✅ §3.3 uses `DUNE_CACHE=disabled /home/tugbars/.opam/5.2.0/bin/dune build --root <abs>` |
| No `git commit` / `git push` | ✅ nothing in this plan commits |
| Gate new kernels on SPEED too, not just accuracy | ✅ static object censuses (p5 class, `regmov`, spill) are gates in Phases 0, 2 and 3, ahead of every timing |
| Canonical vs-MKL bench discipline | ✅ §6.2 |

## Appendix B — errata to write back into the record

1. `z_cascade_plan.md:920-936` — *"MKL ping-pongs planes for every stage"* is **refuted**
   by the disassembly [M9]: interior 0 shuffles across ~1900 instructions, in-place mids,
   **one** scratch plane `r9`; `r8` is an int32 offset table.
2. `z_cascade_plan.md:911-915` — the *"+5…+22 % for out-of-place into a 2nd plane"* row is
   now **reproduced with committed code** [M6]: +7.4 / +21.4 / +8.3 / +14.3 %.
3. **New fact, recorded nowhere:** a second plane is **free** for a terminator-shaped pass
   [M1] and costs **50–126 %** for a mid [M3]. That asymmetry is the actual reason Lever 1
   died.
4. `z_cascade_plan.md:485` — the 44 % terminator figure is the VTune **retiring** column,
   not a time share. Ticks give **34.7 %** at N=16384; `:687` already says 35 %. The
   in-situ measured shares are **50 / 44 / 36 / 32 %** at 2048/4096/8192/16384 [M7].
5. `dp_planner_il.h:315-318` — the comment *"same order contract, so they must agree to
   rounding"* is **false across chains**; the SCRAMBLED chain search is currently ranking
   one candidate.
6. `zsplit.h:61,152-155` — `gb[]` is **dead**: filled at create, never read by either
   execute path.
7. §4.9996's *"micro-levers are exhausted"* is over-broad — the padded plane pitch
   (measured 2–4 % overall, clean v2 methodology) was never built.
