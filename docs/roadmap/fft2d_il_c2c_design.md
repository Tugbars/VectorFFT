# fft2d IL c2c — native tier via 2D IL strided codelets

Status: M0 + M1 COMPLETE (2026-08-25) — M0a wrap tax MEASURED 1.33-1.50x
(the 24-39% inference confirmed) and an IL-2D gap established (O-inter
0.58-0.98 xMKLcce, noisy-run signs); M0b simulator + gate ALL PASS; M1
native tier SHIPPED env-gated (VFFT_IL2D_NATIVE=1, native-FIRST create —
an engaged create never builds the split tplan) and its gate is 36/36
(9 cells x 2 dirs x 2 placements, elementwise vs naive, natural x natural,
prime and non-pow2 N2 included). NEXT: M1 race verdict (--2dil O-NATIVE
arm), then M2 t2c. Predecessors: `il_native_design.md`
(the cil family + construction laws), `interleaved_design.md` (pitfall #6:
"1D K=1 as degenerate 2D" — run backward here), `../research/mkl_2d_campaign/
IMPLICATIONS_for_our_2d.md` (G1 = the convert-around; phase-0 measurement law;
candidate C parked 2026-08-09 on premises that no longer hold: the native IL
kernel family has since shipped and won its K=1 races).

Laws respected: pure IL end-to-end (no `in_re`/`in_im` signature anywhere —
`never_build_hybrid_il_split_codelets`); a NEW driver, the split `fft2d.h`
untouched (`dont_retrofit_il_onto_split_machinery`); every axis below is a
raced wisdom field in the `lay=il` 2D cells (v1.2 store, shipped 2026-08-25),
never a constant; measurement-first (owner directive on 2D IL) — M0 funds
everything else.

## 1. Thesis

A 2D c2c on an interleaved plane is the K=1 four-step WITHOUT the inter-pass
twiddle: pass 1 = N1-point FFTs down the columns, pass 2 = N2-point FFTs
along the rows, no `W_N^{ij}` between, no transpose anywhere — pass-1 outputs
land back in their columns and pass 2 reads rows that are contiguous by
layout. Both passes eat the user's plane directly:

- **Column pass** — the split strided family's own taxonomy (`c2c_split.ml`)
  names this the UnitGroup case: adjacent transforms contiguous, "strided
  loads populate the lane registers — no transpose". In IL it is cheaper
  still: two adjacent columns are 16 contiguous bytes, so the batch-axis
  load is one plain `vmovupd`. The shipped cil ABI already expresses it —
  `zin[2*(l*Ls + k)]`, strides in complex units, `count ANY >= 1` with the
  inline VEX-128 odd tail. `Ls = N2`, k runs over a column tile.
- **Row pass** — each row is a contiguous K=1 IL transform: a
  transform-contiguous loop over the production K=1 engine (mono / bailey /
  zcasc by N2), inheriting every banked per-N verdict (il_kv, chains, t2q,
  wings, natural tiers) with zero new code.

What is genuinely new: ONE kernel kind (the broadcast-twiddle column stage,
§3), a 2D driver composing existing kernels, and the wisdom/race wiring.

## 2. The twiddle-geometry flip (the one strategy that MUTATES)

Bailey stage-2 twiddles are `W_N^{jk}` — they vary with the column index k,
the vector axis, so the 1D family streams per-lane VTW2 records (cursor
+120 doubles per group at R=16). A pure 2D column stage has NO j-dependent
twiddle: stage twiddles `W_{N1}^{...}` are functions of the position digits
WITHIN the column transform (the butterfly group) only. Both lanes of every
vector share one twiddle.

That is verbatim the admissibility condition of `il_native_design.md` §6c —
**z-T1S broadcast, "admissible ONLY where both lanes share W (batch
contexts; NOT K=1 columns)"** — specified in the family design and never
built for lack of a consumer. The 2D column stage is the consumer:

- per group, per leg 1..R-1: ONE `[c×4][-s,+s,-s,+s]` record, loaded
  BEFORE the k-loop and reused across the whole column tile. The apply is
  the unchanged BYTW2 (`fmadd(c, x, mul(s, cflip x))`, one data-side
  shuffle). Table traffic collapses from per-column-pair streaming to
  per-group constants.
- LOG3 derivation chains and the compact form remain admissible and are
  raced per cell as always; per-lane VTW2 has no role in a pure 2D stage
  (nothing varies across the lanes to pay for).
- The parked `[c,tan]` record lever (roadmap_post_tangent_levers #1) is the
  tangent form of this record — see §4.

## 3. The kernel set

| kind | role | status |
|---|---|---|
| `n1c` (SHIPPED 2026-08-25) | single-layer column pass when N1 <= codelet radix (4..64): one call, `Ls = OLs = N2`, in place | fwd+bwd pairs emitted for {4,8,16,32,64}; corpus cells + registry lists (`VFFT_IL_N1C_*`) |
| `t2c` (NEW; name provisional) | multi-stage column mid: same-slot in-place DIF stage along the column axis, legs at `D_s*N2`, per-group broadcast twiddle records hoisted out of the k-loop, in-kernel group loop | the one new emission left |
| row pass | K=1 IL engine per row | SHIPPED — composition only |

The column leaf is its OWN kind, not a reuse of plain `n1` (owner directive:
distinct names): plain n1 had NO forward population — only `n1_bwd`, which
belongs to the 1D F-DIAG chain, and reusing it would have manufactured a
fake fwd/bwd "pair" spanning two subsystems. `n1c` is n1 math with the 2D
family's contract stated in the name and in the ABI: alias-tolerant (no
`__restrict__`) in BOTH directions, because the 2D column pass runs
`zin == zout` — plain N1 grants that backward only. Emitter delta: kind
`N1C` in `c2c_il.ml` (+ both edge-dispatch matches), `--cil-n1c` in
gen_main/codelet scanner+selector+argv, `Cil_n1c` in codelet.ml(i).

`t2c` emission slots into `c2c_il.ml` beside n1/n1t/t2: same cx math layer,
same schedulers, new twiddle-record edge + group loop. No new backend, no
new IR.

## 4. Transfer table — every 1D IL optimization, disposition here

| strategy (source) | transfers? | 2D form |
|---|---|---|
| class-aware butterflies (`W^{R/4}` free rotation, `W^{R/8}` FMA fold, conjugate-pair odd) | YES, unchanged | math layer is edge-independent |
| blocked construction law (spill-free radix-8 pieces; r16/r32 blocked, r64 blocked2; corner-turn free in zspill indexing) | YES, unchanged | DAG-body property; same per-radix static defaults |
| odd-count tail (`count ANY >= 1`) | YES, unchanged | arbitrary column-tile widths free |
| SR scheduler + cx_cpl race, port-5 pressure | YES, unchanged | raced per slot as always |
| VTW2 per-lane stream | NO ROLE | no lane-varying twiddle exists in a 2D stage (§2) |
| z-T1S broadcast (6c, never built) | YES — **finally gets its consumer** | per-group hoisted records (§2) |
| LOG3 / compact tables | YES | raced per cell |
| tangent / wing (R16 leaf -17%, 256 -20..-24%, 512 -14..-17%) | YES with a twist | emit-time tangent carries for n1 constants; for runtime per-group records the cos normalization cannot fold into emit-time constants → the `[c,tan]` RECORD form is the tangent path here. Race, don't assume; scoped <=512-class radices as in 1D |
| turned stores (N1T corner-turn, TURN128, t2tg) | **NO — deliberately** | the axis exists to serve the four-step's inter-pass transpose; 2D has no corner turn. Plain leg-major stores. (TURN128's half-store idiom may be raced as a store-idiom variant later; it has no structural role) |
| pre-twiddle position / t2t (bwd = table-side conj, kernel bit-identical) | YES | bwd `t2c` = conjugated records + placement moved, never text derivation |
| msg in-kernel group loop | YES — required | groups × stages here multiply the per-call tax that sank fine tiles at 2048 |
| store-sinking (B1) | YES | our stores are singleton contiguous per leg — the sinkable shape |
| sterm2 / uj2 unroll-and-jam + per-cell t2q pick | YES as a raced body variant | braid two column-tiles; never hand-set (placement luck ±5%) |
| TP_PowW1 squaring tree | NO ROLE | it derives per-COLUMN w^1 powers; 2D stages have no per-column twiddle |
| ZTURN-S sectioned geometry | AS A LESSON | see §5 — the aliasing cure shape, applied to the staged route's scratch |
| natorder load-side law (P0c: loads free 0.96-1.12x, stores +29-50%) | YES | §6 |
| in-place (il2p) + shadow-plane natural | YES | same-slot DIF stages are in-place safe by construction; the il2p scar law binds (§7) |
| per-cell wisdom / kv encoding | YES | new il-2D kv codes in the `lay=il` 2D cells; race through the IL door |

## 5. The aliasing hazard and the route race (C1, priced honestly)

Column-pass leg stride = `16*N2` bytes. At pow2 N2 >= 256 that is a 4 KB
multiple: all R legs of a butterfly land in one L1d set group (Raptor Lake
48 KB / 12-way), so R >= ~12 thrashes and R = 8 is marginal. **[I] — code-
derived, unmeasured (the split side's C1 was never measured either; Q4).**
Two routes, RACED per cell, never chosen by rule:

- **direct-strided** — kernels on the user plane as-is; radix choice per
  cell lets wisdom prefer radix-8 stages at aliasing-hot N2.
- **staged** — copy a column tile (Wc columns × N1) through contiguous
  scratch with a raced pitch skew, FFT at tiny strides, copy back. This is
  the cascade's "every pass streams contiguously" law, and the skewed
  scratch is the lever MKL structurally cannot pull (it gathers from the
  user's pitch; IMPLICATIONS §Phase-2). ZTURN-S is the precedent that a
  cursor/geometry change of exactly this class was worth 1.35×.

Tile width Wc is a raced wisdom axis in both routes (the campaign's own
recommendation A: the tile of a 2D pass is a first-class raced parameter;
borrow the axis, never MKL's constant 4).

## 6. Ordering contract

Same-slot DIF stages leave each column scrambled along i; the row pass
produces its own per-N2 order. Canonical tier contract = `ord=scr` (matches
split 2D). Natural order rides the P0c law: the i-descramble is ρ⁻¹ block
tables addressing the ROW pass's LOADS (the `stfn` mechanism — table via the
unused `tw_im` arg, stores stay ascending); the j-order comes from the row
engine's own natural tier. Scattered stores to user memory are condemned
(the dtso fence). Gates are FORWARD, per direction, elementwise vs a
separable naive DFT — roundtrip can never gate a permuted transform.

## 7. Milestones (oracle-first; each gated before the next)

- **M0 — measure + derive.** (a) The IMPLICATIONS phase-0a three-arm cell
  (O-split / O-inter / M-inter-CCE, one run, canonical bench `--mode`) —
  pins the target and prices the convert-around for real (today [I]-only,
  24–39%). (b) Scalar simulator for the 2D stage index maps, gated
  elementwise vs naive at N1×N2 ∈ {small, pow2, odd} — the il2p scar law:
  eight guessed stride maps once fell in one session; maps are proven by a
  running simulator before any SIMD uses them.
- **M1 — small-N1 native tier. [DONE 2026-08-25]** n1c column pass
  (`Ls=N2`, N1 <= 64) + per-row K=1 IL NATURAL children; native-first
  create (an engaged create skips `_build_2d` — gate runtime 15min -> 22s);
  gate 36/36 elementwise vs naive, both directions and placements,
  engagement self-proven (natural output; the scrambled wrapper cannot
  pass). Serving is opt-in (`VFFT_IL2D_NATIVE=1`) until the route race +
  `lay=il` banking (M3/M4). Gate: forward elementwise per
  direction; race vs convert-wrapper, split 2D, MKL CCE at the same cells.
  This alone serves every cell with N1 <= 64 natively.
- **M2 — `t2c` + the multi-stage column chain.** Broadcast-record kind,
  in-place DIF stages, in-kernel group loop; N1 to 4096-class. Bit-gate vs
  the M0 simulator; speed-gate per cell.
- **M3 — variant races.** Blocked bodies per the construction table;
  tangent/[c,tan] records; direct vs staged route + Wc + pitch skew; uj2;
  store-sinking. Bank winners as il-2D kv codes in `lay=il` cells.
- **M4 — productionize.** Natural-order (ρ tables on the row pass),
  in-place + OOP parity, wire as the IL 2D door in create (serving replaces
  the convert wrapper ONLY where the race says so — the wrapper stays as
  fallback), 2D gate arms, wisdom persistence.

## 8. Open questions (closed by measurement, not argument)

- Q-A: does the broadcast-record stage beat a plain-VLIT specialization at
  small stage counts? (Emit-time constants are free; records cost loads —
  the n1-vs-t2c boundary may sit above N1=64.)
- Q-B: where is the direct/staged crossover, and does the pitch skew move
  it? (M3; falsifier: if raced Wc × skew moves nothing beyond control
  spread at >= 6 of 9 aspect cells, the staged route is dead here.)
- Q-C: does the row-pass TC loop want a fused row-tile (several rows per
  engine call) at small N2? (The K>1 TC machinery already exists; race.)
- Q-D: MT — the 2D K-split corruption (G10) is still open on the split
  side; the IL tier must not inherit it. MT for this tier is a separate,
  engagement-proven campaign (mt_results_need_engagement_proof).
