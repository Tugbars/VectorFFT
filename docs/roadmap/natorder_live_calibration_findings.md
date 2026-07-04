# Natural-order live-calibration findings (2026-07-05)

**What this is.** The first run of the *real* natural-order calibrator race (not the T6–T11 idealized
bakeoff, not forced wisdom) through the public API, after FREE / PURE-cycle / PSWAP / SCR were all wired
into `vfft.c`. Probe: `build_tuned/test/natorder_calibrate_probe.c` — creates each cell with
`order=VFFT_ORDER_NATURAL, rigor=MEASURE` on a fresh wisdom dir, which calibrates the c2c chain
(`dp_best`) then runs `vfft_natorder_race` (PURE vs injected-palindrome PSWAP vs SCR) and stamps the
verdict. Every cell was validated: natural forward vs naïve DFT `1e-13…1e-14`, roundtrip `1e-14…1e-15`.

> ⚠️ Calibrating a grid is expensive (each cell = full `dp_best` + the natural race). Probe 1–2 cells
> at a time. See [[natorder-calibration-cost]].

## Results — winner + per-stage plan per cell

Variant codes: `0=FLAT 1=LOG3 2=T1S`. The **untwiddled** stage (stage 0 for DIT, the *last* stage for
DIF) runs the `n1` codelet regardless of its stored code.

| N | K | chain | orient. | per-stage variants | natural mode |
|---|---|---|---|---|---|
| 64 | 4 | 64 | DIF | [n1] | **FREE** (nf=1) |
| 128 | 4 | 8·16 | **DIF** | [LOG3, n1] | PURE-cycle |
| 256 | 4 | 16·16 | **DIF** | [LOG3, n1] | PURE-cycle |
| 1024 | 4 | 64·16 | **DIF** | [LOG3, n1] | **PSWAP** → injected **4·64·4** (DIT, all-T1S) |
| 4096 | 4 | 4·64·16 | **DIF** | [FLAT, LOG3, n1] | PURE-cycle |
| 1024 | 32 | 4·4·8·8 | DIT | [n1, T1S, T1S, T1S] | PURE-cycle |
| 4096 | 32 | 4·4·4·8·8 | DIT | [n1, T1S, T1S, T1S, T1S] | PURE-cycle |
| 256 | 256 | 4·4·4·4 | DIT | [n1, T1S, T1S, T1S] | PURE-cycle |
| 128 | 64 | 4·32 | DIT | [n1, T1S] | PURE-cycle |

Raw stamped wisdom (v7 format `… exec_me nat_mode nat_ns [nat_nf nat_factors… nat_prof]`):

```
64 4 1 64 117.61 0 0 0 1 0 0 0 0.00
128 4 2 8 16 250.78 0 0 0 1 1 0 0 4 625.00
256 4 2 16 16 636.35 0 0 0 1 1 0 0 4 1308.33
1024 4 2 64 16 3751.86 0 0 0 1 1 0 0 5 7500.00 3 4 64 4 2
4096 4 3 4 64 16 20560.16 0 0 0 1 0 1 0 0 4 45900.00
1024 32 4 4 4 8 8 25332.81 0 0 0 0 0 2 2 2 0 4 43950.00
4096 32 5 4 4 4 8 8 139043.75 0 0 0 0 0 2 2 2 2 0 4 198941.66
256 256 4 4 4 4 4 45598.44 0 0 0 0 0 2 2 2 0 4 65175.00
128 64 2 4 32 4275.39 0 0 0 0 0 2 0 4 6525.00
```

## Interpretation

### 1. The machinery works end-to-end
FREE, PURE-cycle, and PSWAP were all selected and all correct through the public API — the first time
the natural-order feature has been exercised as a live calibrated pipeline (chain calibration → mode
race → wisdom stamp → execute → validate). No correctness failures.

### 2. PSWAP self-discovered a chain the normal planner would never keep
At 1024/4 the scrambled-optimal plan is `64·16` DIF, but the natural-order winner is the **injected
`4·64·4` DIT, all-T1S** — reached only because the race *injects* palindromic candidates
(`vfft_natorder_palindromes`) that `dp_best`'s beam prunes for being slower under scrambled/T1S
scoring. This is the chain-injection thesis validated in a live run: the natural-order winner can differ
from the scrambled winner in **both chain and orientation**, and the calibrator finds it.

### 3. The headline finding — **SCR was gated out, not out-timed**
SCR won **zero** cells. The reason is not speed. Read the orientation column: **every K=4 cell calibrates
as DIF.** SCR's applicability gate (`natorder_scr_build`) rejects DIF outright — MODEB's stage-0
out-of-place redirect requires an *untwiddled* stage 0, which only DIT plans have. So across the entire
K=4 band — exactly where the bakeoff projected SCR would win (128/4 at +19%) — **SCR never entered the
race**: `natorder_scr_build` returned 0, `have_scr=0`, and the race ran PURE-vs-PSWAP only.

This corrects a mid-session misread ("real SCR is slower than PURE"). The T6–T11 bakeoffs hand-fed SCR
*DIT* chains and it looked strong; the *real* calibrated K=4 chains are **DIF**, so SCR's DIT-only gate
excludes it before any timing happens. The scatter kernel was never the reason it lost here.

### 4. A clean twiddle-method split falls out of the data
- **K=4 → DIF + LOG3/FLAT.** At tiny K there are few lanes to amortize twiddle work, and the DIF
  orientation puts the untwiddled stage last; LOG3/FLAT win.
- **K≥32 → DIT + all-T1S (stage 0 `n1`).** Once K is large, T1S's tiny scalar tables dominate — the
  memory-bound thesis: minimize table traffic, pay a little extra math. This is the "3 twiddle methods
  = a real contribution" point [[three-twiddle-methods-contribution]] showing up per-cell.

## Implications & next move

- **Natural order ships now** on FREE / PURE-cycle / PSWAP, MT end-to-end, correct across the grid.
  PURE carries most cells; PSWAP wins where an injected palindrome beats it; FREE covers single-stage.
- **SCR needs orientation injection to matter.** To let SCR compete at the K=4 band, the race must
  **inject a DIT chain for the SCR candidate** (analogous to PSWAP's palindrome injection) instead of
  only trying SCR on the calibrated (DIF) plan. A DIT re-plan of the same N/K is cheap; if the fused
  scatter then beats PURE by the 5% margin, SCR ships there — otherwise the honorable PURE fallback
  stands. This is a concrete, well-motivated change, unlike "optimize the SCR kernel" (which the data
  now shows was never the bottleneck).
- **Do not conclude SCR is dead.** It is untested against PURE on a fair (DIT) footing. The bakeoff's
  +16–27% projection may or may not survive the real pre-twiddle + generic-MODEB path — but that is a
  *measurement we have not yet run*, because the DIF gate pre-empted it.

## Reproduce
`cd build_tuned && python build.py --src test/natorder_calibrate_probe.c --vfft` (edit the cell list to
1–2 cells to keep it fast), then dump `natorder_cal_probe/spike_wisdom.txt`. See
[natural_order_inplace_design.md](natural_order_inplace_design.md) for the mode designs and the T6–T11
bakeoff history this run reconciles against.
