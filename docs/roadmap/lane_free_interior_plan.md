# Lane-free interior for high-radix IL codelets

> ## 🔴 CANCELLED 2026-08-12 — but ONLY the LAYOUT idea. The goal is very much alive.
>
> **The live successor is [`newsplit_for_cx_plan.md`](newsplit_for_cx_plan.md).**
> Read that first. What was wrong here was the *diagnosis* (layout), not the
> *target*. The reference kernel's advantage is **algebraic**, not a data
> arrangement: it runs a scaled conjugate-pair split radix, which our tree already
> implements on the real side as `Split_radix` NEWSPLIT (Johnson-Frigo / Van
> Buskirk). Our cx/IL emitter dispatches pow2 to a plain radix-2 DIT and therefore
> never uses it.
>
> Tugbars, correctly, on the cancellation below: *"either your conclusion was wrong
> or how you applied it. because MKL's code is there existing and working fine. no
> need to refute or lose hope… we have a working example, of course we won't copy
> 1:1 but the recipe is there."* The conclusion was wrong. The recipe is real.
>
> The rest of this box remains valid **for the layout approach only**:
>
> **Do not run Phase 1. Do not re-propose the LAYOUT conversion.** The experiment
> exists, with a
> same-radix control, in `never_build_hybrid_il_split_codelets.md`: untwiddled leaf,
> hybrid (`z` in → split interior) vs full IL (`z`→`z`), ratio = IL/hybrid, **<1 =
> full IL faster**:
>
> | R | mono/hyb | blocked/hyb |
> |---|--:|--:|
> | 16 | 0.750 | 0.657 |
> | **32** | **0.787** | **0.658** |
>
> Our shipped `n1tb48` is the **blocked leaf** — precisely the arm tested. **Full IL
> is 34% faster than the lane-free alternative at R32.** The form proposed below is
> strictly worse than the one measured: that hybrid paid **one** conversion
> (interleaved in, split out); this plan pays **two** (ingest and egress).
>
> **Why the lane-op census does not carry the day** — the tax only outruns the
> conversion cost when it is amortized over many passes. From the same record:
> *"mids run ZERO shuffles/xors; conversions live only in S0 (paid once)… a pure-IL
> one pays a cflip every complex multiply, and a cascade runs log-many passes so it
> accumulates. **At TWO passes there is nothing to accumulate** — which is exactly
> why pure IL wins below 1024 and loses above it."* N=512 and N=1024 are two-pass
> `il2p` cells. Both conversions get paid; nothing accumulates to repay them.
>
> **What survives:** the census in §1–§3 and in
> [`lane_tax_in_il_codelets.md`](../performance/lane_tax_in_il_codelets.md) is
> correct and still worth having — it supplies the *mechanism and the numbers* for
> the tiering rule we already ship (packed IL ≤2048, split ≥4096). It explains an
> existing decision; it does not license a new codelet family.
>
> **What is still open** (unchanged by this): a *tiled* pure-IL cascade, per the
> same record — a different question, not this one.
>
> Everything below is kept as the reasoning record. Its §1 findings about the
> reference kernel stand; its *proposal* does not.

**Status:** ~~proposed, not started~~ **CANCELLED — refuted by prior measurement.**
**Target:** the two K=1 cells still behind MKL — 512 (0.87–0.88×) and 1024 (0.83–0.90×).
**Date:** 2026-08-12

## 1. The finding this rests on

MKL's 32-point column kernel for N=512
(`docs/research/mkl512_gap_campaign/asm/mkl512__col32_fwd_loop.asm`) was censused
against our radix-32 leaf `n1tb48`. Same work unit, verified: both do 32 ymm loads →
32 ymm stores (= 64 complex = 2 columns of 32), both hoist twiddles as constants,
neither streams a twiddle table.

| | MKL col32 | ours `n1tb48` |
|---|--:|--:|
| instructions | **460** | 563 |
| fp-arith | **186** | 208 |
| fma / bare mul | **68 / 0** | 36 / 20 |
| naked add+sub | 118 (**63%**) | 152 (73%) |
| shuffle + xor | **54 + 15 = 69** | 82 + 29 = **111** |
| stack ops | **78 (0.42/fp-op)** | 48 (0.23/fp-op) |

Two things follow, and the second is the actionable one.

**MKL spills 1.6× more per arithmetic op than we do and still wins.** So "reduce
spills" is not the lever against MKL. It treats stack traffic as cheap.

**MKL already uses our tangent construction — but on lane-separated operands.**
The idiom, verbatim from the loop body:

```asm
vmovapd      y10,y14        ; copy the constant
vfnmadd213pd y8,y13,y14     ; y14 = y8 − c·y13
vfmadd213pd  y13,y10,y8     ; y8  = c·y8 + y13
```

That is exactly the tangent-scaled butterfly pair — two FMAs, no multiply. What is
*absent* is the point: **no shuffle, no xor.** The two operands live in separate
registers, so the rotation never crosses lanes.

The positional evidence is unambiguous. Shuffle/xor density across each loop, in
ten buckets:

```
MKL col32     0  0  0  0  6  6 10 15  7 18     <- ZERO for the first 40%
ours n1tb48   7  7  8  8 11 17 13 11 16 12     <- nonzero from the first bucket
```

MKL pays lane cost **once, near the output**. We pay it **per rotation, throughout**.

## 2. Why this bites at R32 and not at R16

Rotation count grows with radix; the ingest/egress lane cost is fixed. At R16 the
per-rotation tax is affordable and our kernel reaches parity with a hand-built one.
At R32 it dominates — which is precisely why the two cells still behind MKL are the
two R32-bound ones (512 = 2⁹ cannot factor without an R32 or R64 pass; 1024 = 32×32
is R32 in *both* slots).

It also explains the one result that did not fit the tangent story: our R32 tangent
kernel achieves a **better** port mix than our R16 tangent kernel (39% vs 47% naked)
yet returns −3.2% against R16's −25%. We applied the right construction inside the
wrong encoding — each tangent rotation still paid its shuffle+xor, and the added
constants pushed register pressure up. The construction was never the problem.

## 3. What is proposed

A **lane-free interior** for high-radix IL codelets: de-interleave once at ingest,
run the interior (including tangent rotations) on lane-separated registers where a
rotation costs 2 FMAs and nothing else, re-interleave once at egress.

### This is NOT a hybrid IL/split codelet

Flagging this explicitly, because it will look like one at a glance and the standing
rule is to kill those on sight.

The rule's own test is on the **signature**: a codelet taking `in_re` + `in_im` is
hybrid. This codelet's signature is unchanged — interleaved complex in, interleaved
complex out, byte-identical contract to the kernel it replaces. What changes is the
*working form inside the function body*, which no caller can observe.

The project already ships this shape and it is the winning form there: the cascade
tier keeps its boundary conversion, and the "full-IL interior" alternative was
refuted twice. This is the same trade at codelet scope.

## 4. Plan

Proxy-first, because static census has failed to predict time on this machine
repeatedly (the remat lever, the Belady spill planner, and three static
stack-traffic proposals all measured the wrong way round). **No emitter work until
a hand-edited proxy wins on hardware.**

### Phase 0 — pin the mechanism — ✅ RUN 2026-08-12, **MECHANISM REFUTED**

Phase 0 did its job: it killed the mechanism this plan originally proposed.

**Established (robust):**

- The first **184 instructions — 40% of the loop — contain zero shuffles and zero
  xors.** Positional density across ten buckets: `0 0 0 0 6 6 10 15 7 18`. Lane work
  is absent early and concentrated toward the output.
- The rotation idiom is confirmed at multiple sites: `vmovapd` a constant, then
  `vfnmadd213pd` / `vfmadd213pd` — two FMAs, no multiply, **no lane op**.
- Constants are **uniform scalar broadcasts** (7 × `vbroadcastsd`). Ours are
  pre-signed alternating vectors (`{s,−s,s,−s}`), the interleaved-complex signature.

**Refuted:** the "de-interleave at ingest" model. It predicts ~32 shuffles in the
*first* bucket; there are **zero**. The kernel does not construct a lane-separated
form on entry.

**Also withdrawn — my follow-up guess that it is "handed a lane-separated layout
from upstream."** Tugbars asked the obvious question that breaks it: *how can it be
lane-free while taking interleaved data?* It cannot, and it isn't. Per-bucket FMA
against lane ops:

```text
bucket    0   1   2   3   4   5   6   7   8   9
fma       4   6   6   6   6  12  11   3  14   0
lane      0   0   0   0   6   6   7  10   5  12
```

There **are** FMAs in the lane-free region, using broadcast *scalars* — and an FMA
by a real scalar is elementwise, hence lane-free on interleaved data too. So the
first 40% is simply **the part of the decomposition whose coefficients are real**
(±1 butterflies, real shears such as the √½ fold). Every `i`-multiply and general
rotation — the operations that must cross lanes — sits in the back 60%, exactly
where the lane ops are. The kernel takes interleaved data and pays lane work where
rotations need it. Its lane work is **deferred and concentrated, not eliminated.**
No upstream layout is involved; do not go looking for one.

**Still unexplained:** the *total* — 69 lane ops against our 111. Real, but not a
layout difference. More likely how many general rotations its decomposition needs
at all, plus our blocked form's replay pass contributing its own lane work.
Moot for this arc, which is cancelled on other grounds.

### Consequence for this plan

The plan **survives, but its justification changes.** It is no longer "do what the
reference implementation does" — that mechanism is refuted at codelet scope. It is
now grounded solely in **our own census**
([`lane_tax_in_il_codelets.md`](../performance/lane_tax_in_il_codelets.md)): our
interior encoding costs **3.44 lane ops per input vector at R32** where a boundary
conversion costs a flat **~2**. That number is ours, measured from shipped sources,
and stands independently.

Two consequences for how Phase 1 is read:

- A proxy **win** is real and bankable — it is our own kernel getting faster.
- A proxy **loss does not refute the layout hypothesis**, because the upstream-layout
  mechanism was never testable at codelet scope. Do not record a loss as "lane work
  doesn't matter"; record it as "boundary conversion at codelet scope doesn't pay".

### Phase 1 — proxy race (hand-edited C, no emitter)

- Take the emitted `n1tb48` C, hand-edit to: de-interleave at ingest, run tangent
  rotations on separated registers, re-interleave at egress.
- Gate for correctness (tolerance, not bit-identity — association changes).
- Race vs shipped `n1tb48` under the banked paired protocol: pinned core 2, HIGH
  priority, ≥15 rounds, alternating order, one arena +64 B skew, control twin,
  report spread.
- **Kill criterion: no win outside the control spread ⇒ STOP.** Do not touch the
  emitter. Record the refutation in `tangent/README.md` under "Killed by measurement".

### Phase 2 — emitter (only if Phase 1 wins)

- The change lives at the rotation-rendering seam (`cx_render.ml`) plus ingest/egress
  in `codelet_cil.ml`; the constant set and scheduling are unchanged.
- New knob, **default-off**, per the established pattern (`VFFT_CX_*`).
- **Gate:** the 183-case matrix must stay byte-identical with the knob off.
- Gate new kernels on speed as well as correctness — bit-identical ≠ same instructions.

### Phase 3 — race and promote

- `calibrate_k1.exe <COPY> 1 512 1024`. Rebuild the calibrator first (stale
  wisdom-writers strip fields). Diff every other row to prove a clean writeback,
  then hand-promote.
- Bench vs MKL through `bench_1d_vs_mkl.c --k1noop` with explicit N (the default
  wisdom walk skips K=1 kind-3 cells).
- Front-door gate before promotion.

## 5. Scope, and what this does not do

- **Blast radius is wider than R32.** The encoding is shared, so every IL radix is
  affected. Expect the win to scale with rotation count: best at R32, smaller at R16,
  plausibly *negative* at R8 where the fixed ingest cost cannot be amortized. Race
  per radix; do not promote globally.
- 128 and 256 already lead MKL (1.04×, 1.01×). This is about the two cells that do
  not, and about margin on the ones that do.
- **This does not touch spills, and should not.** If the proxy comes back with more
  stack traffic and a better time, that is the expected shape, not a regression.

## 6. Risks

| risk | handling |
|---|---|
| Static counts mispredict time (8-for-9 on this machine) | Phase 1 proxy race is the gate; emitter work is downstream of a hardware win |
| Lane-separated interior widens the live set → RA blowup | MKL accepts exactly this trade; measure, do not assume. Watch reg-movs, not just spills |
| A new stack plane hits 4 KB aliasing (trap hit twice before) | Skew the plane; check pitch explicitly |
| Machine is thermally noisy | Minima not means; control twin mandatory; a delta inside control spread is not a result |
| Reads as a forbidden hybrid codelet | §3 — signature is unchanged; the rule tests signatures |
