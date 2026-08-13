# NEWSPLIT for the cx (IL) family — conjugate-pair split radix at pow2

> ## 🔴 PORT REFUTED 2026-08-12 by a one-command gate — do NOT transliterate this
>
> Real-side DAG stats, same radix and emitter, `VFFT_NEWSPLIT` off vs on:
>
> | R | | vector insns | mul | **fma** |
> |---|---|--:|--:|--:|
> | 16 | DIT | 144 | 0 | **40** |
> | 16 | NEWSPLIT | **167** (+16%) | 23 | **3** |
> | 32 | DIT | 386 | 6 | **140** |
> | 32 | NEWSPLIT | **452** (+17%) | 80 | **12** |
> | 64 | DIT | 978 | 50 | **336** |
> | 64 | NEWSPLIT | **1143** (+17%) | 227 | **35** |
>
> **+17% instructions and the FMA structure collapses** (140 → 12 at R32) while bare
> multiplies explode (6 → 80). That is the *opposite* of the target profile. The
> scalings emit as standalone `Mul`s that the FMA lifter never fuses — the scaled
> construction without the deferred normalization that makes scaling pay (which is
> precisely what our tangent butterfly *does* do). Fixing that is a research task on
> shelved code, not a transliteration. **Park it.**
>
> ### And the "just use the split family's profile" idea is dead too
>
> Our split R32 has a *better* arithmetic profile than MKL (36% vs 37% fma; 3.02
> non-mem instructions per complex point vs MKL's ~4.5). It is unusable here: that
> codelet loops `b += 4` — **its efficiency comes from batching 4 transforms across
> the lanes**, the documented "real codelets vectorize along K ⇒ K=1 runs scalar"
> property. At K=1 there is nothing to batch. MKL's `col32` does **not** batch either
> — 32 loads for 64 complex = 2 complex per vector, the same IL packing we use. It
> beats us at our own packing: **4.5 vs our 4.97 non-mem instructions per complex.**
>
> **Net: the R32 instruction gap is real and now has two fewer explanations.**
> Eliminated by measurement: data layout (see the superseded plan), and NEWSPLIT.
> Not yet explained: where MKL's remaining ~10% per-point advantage comes from at
> identical packing. Next probe should be the DIT twiddle-class selection in cx —
> our R32 emits 20 bare muls where the real-side DIT DAG at the same radix emits 6.

**Status:** ~~proposed~~ **PARKED — port refuted by the Phase-0 gate above.**
**Target:** the two K=1 cells still behind — 512 (0.87–0.88×) and 1024 (0.83–0.90×),
both R32-bound.
**Supersedes:** [`lane_free_interior_plan.md`](lane_free_interior_plan.md), whose
diagnosis (data layout) was wrong. The target was right.

## 1. What the working example actually does

A like-for-like census of a reference 32-point kernel against our `n1tb48`
(same work unit: 32 ymm loads → 32 ymm stores, twiddles hoisted as constants):

| | reference | ours `n1tb48` |
|---|--:|--:|
| instructions | **460** | 563 |
| fma / bare mul | **68 / 0** | 36 / 20 |
| naked add+sub | 118 (**63%**) | 152 (73%) |
| within-lane swap | **15** | **49** |
| lane crossing | **16** | 32 |
| xor | **15** | 29 |

Its rotation idiom, verbatim and at many sites:

```asm
vmovapd      y10,y14        ; a real constant
vfnmadd213pd y8,y13,y14     ; y14 = y8 − c·y13
vfmadd213pd  y13,y10,y8     ; y8  = c·y8 + y13
```

Two **data** vectors, one **real** broadcast constant, two FMAs. This is a real 2×2
rotation applied to a **pair** — it needs no `cflip`, no sign xor, and no multiply.
Our form rotates a **single** vector by a **complex** twiddle, which costs a swap
every time. That one difference accounts for the whole census: bare muls vanish
(magnitude is the FMA constant), xors halve (sign rides in `fmadd` vs `fnmadd`), and
within-lane swaps drop by two thirds.

**It is not a layout advantage.** That was the previous plan's error. The data is
interleaved on both sides; see the withdrawn analysis in the superseded document.

## 2. We have this algorithm written down — as a shelved experiment

> ⚠ **Status of `split_radix.ml`, per Tugbars: "we are not using split radix
> anywhere, it was just an experiment."** Verified: gated behind `VFFT_NEWSPLIT`,
> **never raced** (no results in `docs/`), **never emitted** into the codelet tree.
> `dft_select.ml` says why — opt-in so correctness could be validated "without
> disturbing the existing R=16/32/64 path", with adaptation deferred to a PR that
> never happened.
>
> **Treat it as a reference for the math, not a dependency.** Transliterate the
> construction; do not build on the module or assume it is maintained.
>
> **This is neutral evidence, not negative — and there is an asymmetry that matters:**
> the real/split side has **zero lane ops** (measured: `radix32_n1_oop` = 0 permute,
> 0 perm2f128, 0 xor). So on the side NEWSPLIT was written for, it can *only* reduce
> multiply count; it is structurally incapable of showing the `cflip` reduction.
> In cx/IL it does both. **Its payoff should be strictly larger in cx than in the
> family it was built for** — which is a plausible reason it looked unexciting as a
> real-side experiment and got shelved. A shelved split-side experiment does not
> predict the cx result.

### Where the pieces are

[`split_radix.ml:305`](../../src/dag-fft-compiler/generator/lib/split_radix.ml#L305):

> **NEWSPLIT — Johnson-Frigo / Van Buskirk scaled (tangent) conjugate-pair**

Entry points: `Split_radix.dft_newsplit` and `dft_newsplit_blocked`, with all scale
factors compile-time constants. That is the canonical multiplication-minimal pow2
FFT, and it is exactly the construction the reference kernel is running.

Two facts make this a small gap rather than a research project:

**We built the scaled-tangent half this week.** The tangent family
([`tangent_scaled_butterflies.md`](../performance/tangent_scaled_butterflies.md)) is
the "scaled (tangent)" in NEWSPLIT's own name.

**We already emit conjugate-pair structure in cx — but only for odd radices.**
[`cx_math.ml:509`](../../src/dag-fft-compiler/generator/lib/cx_math.ml#L509) is the
dispatcher: *pow2 → radix-2 DIT, odd → conjugate pair*.
[`codelet_cil.ml:50`](../../src/dag-fft-compiler/generator/lib/codelet_cil.ml#L50)
describes what the odd path does: *"sign rides in the opcode, magnitude becomes the
FMA constant, 0/±1 weights collapse"* — the reference idiom, in our emitter, today.

So pow2 radices are the only ones that never see it. The same file already names
this as **"the open math-layer question."**

## 3. What this reframes

The tangent arc at R32 is **half-applied, not exhausted.** We shipped the scaling
without the structure it was designed to scale. That resolves the result which never
fit the story: tangent returns −25% at R16 and only −3.2% at R32. At R16 the DIT
twiddle count is small enough that scaling alone pays; at R32 the *structure* is what
carries the win, and we did not port it.

It also predicts the shape of the gain: the R32 census should move toward the
reference column on all four axes at once — bare muls to zero, xors roughly halved,
within-lane swaps down by ~2/3, FMA count up — because they are one mechanism, not
four independent ones.

## 4. Plan

Proxy-first, unchanged: static census has predicted the wrong direction repeatedly on
this machine, and this plan's own predecessor was wrong. **No emitter work until a
hand-built kernel wins on hardware.**

### Phase 0 — read `Split_radix.newsplit_core` and price the port

- What does NEWSPLIT need that cx lacks? The real side works on `Expr` over separate
  re/im expression trees; cx works on complex atoms. The scale factors
  (`s, sinv, sdiv2, sdiv4, t`) are compile-time — check they survive the cx constant
  model, which already carries tangent constants.
- Decide the seam: a `dft_cx_newsplit` alongside `dft_cx`, selected in the pow2 arm
  of the `dft_small` dispatcher — **not** a replacement.
- **Exit:** a written port shape with an arm count, or a specific blocker.

### Phase 1 — proxy at R32 only

- Hand-build (or machine-translate) one R32 leaf in NEWSPLIT form, pure IL in/out.
- Gate for correctness against a direct DFT (tolerance — association changes).
- Census it first: if bare muls, xors and within-lane swaps do **not** move toward
  the reference column, the port is wrong before it is slow.
- Then race vs shipped `n1tb48` under the banked paired protocol: pinned core 2,
  ≥15 rounds, alternating order, control twin, report spread.
- **Kill criterion:** no win outside the control spread ⇒ stop and record.

### Phase 2 — emitter (only if Phase 1 wins)

- `dft_cx_newsplit` in `cx_math.ml`, selected by a knob, **default-off**.
- **Gate:** the 183-case matrix stays byte-identical with the knob off.
- Compose with the existing tangent arm rather than duplicating it — NEWSPLIT's
  scaling *is* the tangent scaling.

### Phase 3 — race and promote

- Race per radix (R8/R16/R32/R64) — do not promote globally.
- `calibrate_k1.exe <COPY> 1 512 1024`, rebuild the calibrator first, diff every
  other row, hand-promote. Bench via `--k1noop` with explicit N. Front-door gate.

## 5. Risks

| risk | handling |
|---|---|
| Static census mispredicts time (8-for-9 here) | Phase 1 race is the gate |
| NEWSPLIT's numerics differ (scaled outputs) | Tolerance gate, not bit-identity; check the scale factors unwind at the boundary |
| The real-side port is large | Phase 0 prices it before any code; the odd-IL conjugate-pair path is the cheaper precedent to copy from |
| It only pays at high radix | Race per radix; R8/R16 may not move |
| Wrong diagnosis again | This one is grounded in a named algorithm we already implement, not an inference from a histogram |
