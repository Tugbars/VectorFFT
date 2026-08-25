# TRUE IL 2D R2C/C2R — the native interleaved real tier (design of record)

**STATUS: DESIGN OPEN (2026-08-26). Owner directive: the shipped 2D real
tier (a fused-z veneer over split pads, §6a30) violates the standing
no-wrapper instruction — this tier replaces it for interleaved callers.**

Laws respected:
- 🔴 **NO CROSS-LAYOUT SERVING** (owner, 2026-08-25): an IL caller gets a
  native IL plan (cold = race + bank + serve) or a LOUD refusal — never a
  veneer over split machinery. The split doors keep serving split callers
  untouched.
- 🔴 **PURE IL END-TO-END**: no split pads anywhere in the plan; z at
  every seam. A codelet signature mixing `zin` with `in_re`/`in_im` is
  the forbidden hybrid shape (`never_build_hybrid_il_split_codelets`).
- 🔴 **EVERYTHING MEASURED**: every axis raced per cell, verdicts banked
  per layout (`lay=il` real cells), single writer per key. No structural
  defaults where a race can decide.
- 🔴 Docs are declarations: this file records the design, not a journal.

## 1. The construction (both proven components exist)

Forward (r2c): per-row **1D IL r2c** — the shipped zr2c machinery
(recursive child, interleaved-native Hermitian fold, beats MKL 1.01–1.50×
in 1D) — writes the CCE half-spectrum plane N1 × hp1 (hp1 = N2/2+1)
interleaved. Then the **il2d column machinery** (n1c/t2c chain, banded
walk, L2-derived widths — the tier that beats MKL CCE 10/10 in 2D c2c)
runs the column pass with Ls = hp1. Backward (c2r) is the Hermitian
mirror: reversed column chain (conj tables), then per-row 1D IL c2r.

No transpose (adjacent columns contiguous in IL), no inter-pass twiddle
between the row and column passes (2D), column twiddles column-invariant
(broadcast records — the 6c edge).

## 2. The known hard points (each gets a gated milestone)

1. **hp1 is usually ODD** (N2/2+1; odd whenever 4 | N2). The IL column
   kernels carry a `count % 2 == 0` contract. Two candidate answers,
   RACED not chosen: ghost-pad hp1 to even (one dead column, the K_pad
   precedent) vs the odd-count tail machinery (`cil` odd-tail precedent).
2. **DC / Nyquist columns** (k2 = 0 and N2/2) are real-symmetric along
   N1. The generic column pass treats them as complex — correct,
   symmetry unexploited. Optimization lever, not a correctness item;
   raced only after the tier is green.
3. **Output convention**: the column chain leaves N1 digit-reversed
   (the scrambled contract, as in 2D c2c). Natural-order multi-stage N1
   waits on the same rho-tape feature as c2c — LOUD refusal until then.
4. **Row children as plans**: rows reuse the 1D real IL front door as
   recursive children (the zr2c precedent) — their banked 1D verdicts
   serve for free. Any fused row-kind (r2c row with column-friendly
   turned stores) is NEW OCAML EMITTER LOGIC and enters only when a
   measured gap demands it — the c2c tier's construction law.

## 3. Baseline to beat (the M0 door race, 2026-08-25/26)

The veneer's like-for-like numbers vs MKL real CCE 2D (first-ever honest
per-door measurement; noisy-host caveats recorded in memory):
squares r2c ≈ 0.97–1.58, c2r weaker ≈ 0.88–1.01; long-column 4096×64
r2c 0.90 (the un-banded split column pass is the known hole). Full
square + aspect grid: `bench_1d_vs_mkl --2dreal` (r2c + c2r rows per
cell, MKL bwd validated). The native tier must beat the veneer AND MKL
per cell, or the cell keeps the veneer — measured serving, no faith.

## 4. Milestones (gated; construction before optimization)

- **M0 — DONE**: the door race (`--2dreal`); per-layout wisdom + z-door
  planner timing shipped (2026-08-25).
- **M1 — correctness spike**: driver-only native tier (row children +
  n1c/t2c columns, ghost-pad arm first), elementwise vs naive separable
  DFT both directions + roundtrip; gate before any racing.
- **M2 — the odd-hp1 race**: ghost-pad vs odd-tail, per cell class.
- **M3 — serve + bank**: native-first create for IL real callers, lay=il
  real cells (chain/wl/tf/ro + the real-specific axes), refusal for
  inexpressible cells; the veneer arm retires per cell as coverage lands
  (the c2c wrapper-deletion pattern).
- **M4 — levers**: DC/Nyquist symmetry, fused row-kinds (new emitter
  logic, only measurement-driven), MT over bands (shared arc).
