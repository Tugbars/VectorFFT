# TRUE IL 2D R2C/C2R — the native interleaved real tier (design of record)

**STATUS: DESIGN OPEN (2026-08-26). Owner directive: the shipped 2D real
tier (a fused-z veneer over split pads, §6a30) violates the standing
no-wrapper instruction — this tier replaces it for interleaved callers.**

**SHAPE VERIFIED vs FFTW 3.3.10 (2026-08-26, 4-agent source pass):
`rdft/rank-geq2-rdft2.c` — FFTW's ONLY rank≥2 real solver — is this
exact decomposition: r2c along the contiguous dim, then a full complex
DFT in place over the hp1 half-spectrum columns, NO transpose, no
inter-pass twiddle; c2r is the mirror (columns first, re/im-swap
inverse, then hc2r rows). Its SIMD gate demands interleaved vstride-2
data — independent evidence for the IL-native tier. FFTW has ZERO SIMD
r2c leaf codelets (scalar rows; its only small-row edge is call
amortization). Three amendments folded in below: batched rows (§2.4),
row/column fusion illegality (§2.5), the OOP c2r intermediate contract
(§2.6).**

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

Forward (r2c): the row pass is **ONE batched TC K=N1 zr2c execute** —
the banked transform-contiguous K-batch door (wins 30/30 vs lane-major;
`_zr2c_fold_fwd/bwd` already take K/zs/xs, zs = the plane row pitch) —
writing the CCE half-spectrum plane N1 × hp1 (hp1 = N2/2+1) interleaved.
NOT N1 per-row child dispatches: the 1D in-place IL service floor is
~6× the mono math at tiny N, and FFTW's small-row edge is exactly this
call amortization (its `direct2.c` folds ALL rows into one codelet call
via vl/ivs/ovs). A per-row-child arm exists only as a raced alternative
where it wins a cell. Then the **il2d column machinery** (n1c/t2c chain,
banded walk, L2-derived widths — the tier that beats MKL CCE 10/10 in
2D c2c) runs the column pass with Ls = hp1. Backward (c2r) is the
Hermitian mirror: reversed column chain (conj tables), then the batched
K=N1 c2r row execute.

**Two-phase law**: the fwd row pass COMPLETES before column stage 0;
the bwd row pass runs AFTER the last column stage. The Hermitian fold
conjugates (R-linear, not C-linear) and does not commute with the
complex column stages — unlike c2c, where row/column commutation is
what legalized tfuse (§2.5). FFTW's apply_r2hc/apply_hc2r run the same
two disjoint phases, never interleaved.

No transpose (adjacent columns contiguous in IL), no inter-pass twiddle
between the row and column passes (2D), column twiddles column-invariant
(broadcast records — the 6c edge).

## 2. The known hard points (each gets a gated milestone)

1. **hp1 parity flips with N2** (hp1 = N2/2+1: ODD iff 4 | N2, EVEN at
   N2 ≡ 2 mod 4). VERIFIED (2026-08-26, emitter + emitted kernels): the
   n1c/t2c count contract is **ANY >= 1** — the inline VEX-128
   odd-count tail is emitted unconditionally for every col-kind form
   (il_odd_count_tail.md §3), and pitch is a separate runtime arg
   already run at pitch != count in production (the §10b staged path).
   Odd hp1 is LEGAL natively (FFTW likewise takes n/2+1 as a plain
   vector length — no pitch assumption); a padded-pitch arm may still
   be RACED for performance (4KB aliasing), never for legality. M1's
   gate exercises BOTH parities (no existing gate runs n1c/t2c at odd
   counts). **Odd N2 is a LOUD refusal** — zr2c is even-N-only; FFTW's
   scalar fallback covers odd/prime rows, we declare the gap instead
   (the no-cross-layout law: gaps stay gaps until their feature lands).
2. **DC / Nyquist columns** (k2 = 0 and N2/2) are real-symmetric along
   N1. The generic column pass treats them as complex — correct,
   symmetry unexploited. Optimization lever, not a correctness item;
   raced only after the tier is green. (FFTW also leaves this
   unexploited — parking it concedes nothing.)
3. **Output convention**: the column chain leaves N1 digit-reversed
   (the scrambled contract, as in 2D c2c). Natural-order multi-stage N1
   waits on the same rho-tape feature as c2c — LOUD refusal until then.
   FFTW proves natural order needs no transpose in this shape (its
   column children handle ordering internally), so this is OUR
   restriction, not structural: benches vs MKL CCE / FFTW must stay
   like-for-like on ordering.
4. **Rows = the batched TC K-batch door** (§1): one K=N1 zr2c execute
   per direction, banked 1D K-batch verdicts serving for free; the
   per-row recursive-child arm is a raced alternative only. Any fused
   row-kind (r2c row with column-friendly turned stores) is NEW OCAML
   EMITTER LOGIC and enters only when a measured gap demands it — the
   c2c tier's construction law.
5. **Row/column fusion is ILLEGAL here — tfuse structurally OFF.** The
   c2c banded walk fused rows into bands because rows commute with
   every column stage (both C-linear, disjoint axes). The Hermitian
   fold conjugates, so the real row pass does NOT commute — and cut>=1
   always (stage 0 spans N1, never divides wl<N1), so in NEITHER
   direction is a band containing both column stages and rows legal.
   Consequences: (a) the c2c cells' banked wl/tf verdicts do NOT port —
   real lay=il cells re-race wl with rows OUTSIDE the walk and tf
   structurally 0; (b) the executor must not inherit the c2c banded
   branch's tfuse path (silent-wrong-answer risk — double-executed or
   misordered rows); (c) the banded column walk itself (suffix
   depth-first per band, L2 widths) survives intact — it is pure column
   loop interchange. The M4-class salvage is the DEFERRED CROSS-ROW
   FOLD (fold after the column pass, mixing rows k1 and N1−k1): the
   row's c2c(N2/2) part then commutes and fuses AND the column pass
   shrinks to N2/2 even-count columns — new math, enters only if knee
   cells price it.
6. **OOP c2r intermediate contract — decide BEFORE M1.** The
   column-inverse intermediate is N1 × hp1 complex = N1·(N2+2) doubles,
   which exceeds the N1·N2 real dst — it cannot live in dst. Arms:
   destroy-src (column pass in place on the caller's spectrum — FFTW's
   choice: rank-geq2-rdft2 hard-refuses input-preserving OOP hc2r and
   the API sets FFTW_DESTROY_INPUT) vs scratch plane (input-preserving,
   one extra plane + sweep — the veneer's split pads played this role
   and are part of why c2r is the weak door). Race the honest arms;
   whichever serves, the contract is declared per door. In-place c2r
   has no such issue (the padded plane IS the buffer).
7. **In-place fwd r2c pitch is LEGALITY, not a race axis**: X == z
   requires the caller's real rows at padded pitch N2+2 doubles (the
   1D zr2c in-place contract; same convention as FFTW/MKL in-place
   real). The M2 pitch race is the OOP door's bare-vs-padded
   PERFORMANCE axis only; the batch door's zs/xs strides carry the
   padded pitch for free.

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
- **M1 — DONE (2026-08-26, first emission)**: driver-only native tier
  (env-gated VFFT_IL2D_REAL=1; batched TC K=N1 zr2c row door + the
  n1c/t2c chain over hp1 columns on the bare plane, rows OUTSIDE the
  banded walk per §2.5; create-time PURITY gate enforces the zr2c row
  route — a split-interior fall-through can never serve under the
  native flag). Gate `benches/il2d_real_gate.c` ALL PASS 26/26 (13
  cells x r2c-fwd-vs-naive mapped + c2r pair contract), rel
  5.5e-16..6.7e-15, BOTH hp1 parities (odd 5/9/17/33, even 6/10 — the
  first odd-count n1c/t2c coverage), engagement self-proven at nst>1 +
  26/26 [il2d-real] log lines, §2.6 input preservation held. OOP c2r
  contract DECIDED: input-preserving via the il2d_rscr plane (the
  reversed chain's first executed stage does the z->scratch move — no
  extra sweep; FFTW must destroy input here, we don't).
- **M2 — the pitch race**: OOP door only — bare hp1 plane vs padded
  pitch (a PERFORMANCE axis — aliasing — legality needs neither; the
  in-place door's padded pitch is contract, §2.7), per cell class.
- **M3 — serve + bank**: native-first create for IL real callers, lay=il
  real cells (chain/wl/ro + the real-specific axes; tf structurally 0,
  §2.5 — wl re-raced fresh, c2c verdicts do not port), refusal for
  inexpressible cells; the veneer arm retires per cell as coverage lands
  (the c2c wrapper-deletion pattern).
- **M4 — levers**: DC/Nyquist symmetry, the deferred cross-row fold
  (§2.5 — restores row fusion AND makes the column count even),
  store-side-only band staging (FFTW `dft/buffered.c` precedent:
  strided loads tolerated, only stores staged contiguous then copied
  out — a half-staged variant of the §10b route), fused row-kinds (new
  emitter logic, only measurement-driven), MT over bands (shared arc).
