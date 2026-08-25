# fft2d IL c2c — the native tier (design of record, living)

Status: AS-BUILT + IN-PROGRESS (2026-08-25). M0–M2 complete; M3 partially
shipped (chain race + banking, banded walk); current work: kernel-body
optimization (blocked t2c, tangent interiors) per the owner directives
below. This document is the design of record and is UPDATED AS THE TIER
EVOLVES; measured campaign detail lives in the session memory, not here.

Predecessors: `il_native_design.md` (the cil family + construction laws),
`interleaved_design.md` (pitfall #6 run backward), `../research/
mkl_2d_campaign/IMPLICATIONS_for_our_2d.md` (G1; the measurement-first
law; candidate C parked on premises that dissolved when the cil family
shipped).

Laws respected: pure IL end-to-end (`never_build_hybrid_il_split_codelets`);
a NEW driver, the split `fft2d.h` untouched
(`dont_retrofit_il_onto_split_machinery`); every tunable below is a raced
wisdom axis in the `lay=il` 2D cells, never a shipped constant
(`never_heuristic_always_wisdom`); index maps are simulator-proven before
SIMD (the il2p scar law); forward gates per direction — roundtrip alone
never gates a permuted transform.

## 1. Thesis

A 2D c2c on an interleaved plane is the K=1 four-step WITHOUT the
inter-pass twiddle: pass 1 = N1-point FFTs down the columns, pass 2 =
N2-point FFTs along the rows, no `W_N^{ij}`, no transpose anywhere.
Both passes eat the user's plane directly:

- **Column pass** — the UnitGroup case of the split strided taxonomy:
  adjacent columns are 16 contiguous bytes in IL, so the batch-axis load
  is one plain `vmovupd`; the strided part is only the LEG axis, which the
  cil ABI already parameterizes (`zin[2*(l*Ls + k)]`, complex-unit
  strides, `count ANY >= 1` with the VEX-128 odd tail).
- **Row pass** — each row is a contiguous K=1 IL transform served by the
  production K=1 engine (order NATURAL child; inherits every banked 1D
  verdict).

MEASURED FOUNDATION (M0): the convert-wrapper tax on interleaved callers
is 1.33–1.50x across the cell map (twice reproduced); our split engine is
at rough MKL-CCE parity; the interleaved caller lost that parity entirely
to the wrapper. The native tier exists to delete that tax.

## 2. The kernel set

| kind | role | construction |
| --- | --- | --- |
| `n1c` | column-stage LEAF (twiddle-free; the last stage of every chain; the WHOLE chain when N1 <= codelet radix) | fwd+bwd pairs, radices {4,8,16,32,64} |
| `t2c` | column-stage MID: same-slot DIF stage along the column axis | fwd = post-twiddle DIF; bwd = the HERMITIAN-TRANSPOSE stage (§4); radices {4,8,16,32,64} |
| row pass | K=1 IL engine per row | composition only, no new kernels |

Both kinds are their OWN family names (owner directive: never squat on the
1D kinds' names — plain n1's bwd is the F-DIAG role and its fwd was never
populated), alias-tolerant in BOTH directions (the column pass runs
`zin == zout`), emitted by `c2c_il.ml` kinds N1C/T2C through the shared cx
pipeline. Registry lists `VFFT_IL_N1C_*` / `VFFT_IL_T2C_*` derive from the
corpus.

**t2c ABI** (the frozen 11-arg z ABI, all four stride slots meaningful):
`Ls = D*N2` leg stride, `Gs = N2` row pitch (the in-kernel d-advance),
`OGs = D` digit count, `count` = column tile; one call = one stage over
ONE block (the driver loops blocks). `tw_re` = the stage table, d-major:
per digit d, per leg 1..R-1, one `[c x4][sign-folded s x4]` record,
DRIVER-built at plan time.

**The twiddle-geometry flip (why t2c exists):** stage twiddles
`W_L^{d*r}` depend on the position WITHIN the column transform only —
never on the column index, the vector axis. Both lanes share every
twiddle, which is verbatim the admissibility condition of
`il_native_design.md` §6c (z-T1S broadcast, "batch contexts only") —
specified there, never built for lack of a consumer. t2c hoists the
records out of the column loop (`emit_group_prologue` / `ctx.tw_group`
sourcing for CTwL); table traffic collapses versus Bailey's per-lane VTW2
stream (which has NO role in a pure 2D stage).

### 2a. Construction directives (owner, 2026-08-25)

- 🔴 **r32/r64 t2c are NEVER monolithic.** The 1D spill census (r32
  ~14–15%, r64 ~17% of instructions) and the blocked-twin race (−5…−33%)
  transfer; emit and adopt the BLOCKED constructions (r32 = 4x8/8x4
  pieces, r64 = 8x8 blocked2 — the spill-free-radix-8 law). r16 monolithic
  is provisionally acceptable (16-ymm fit) — CONFIRM BY RACE, not by the
  1D verdict (t2c bodies differ from 1D t2: the hoisted broadcast records
  add R-1 register-resident constant pairs the streamed 1D form never
  held live).
- 🔴 **Tangent interiors are a MUST-HAVE** — battle-tested ILP lever (1D
  wings: −14…−24% in exactly the <=512-class radices the column stages
  use). Scope v1: tangent CLASS-CONSTANT interiors (the emit-time
  butterfly constants — the same scope the 1D wing-t2 tangents, keeping
  BYTW2 for the runtime records); the full `[c,tan]` RECORD form (runtime
  normalization folding) remains the parked follow-up lever
  (roadmap_post_tangent_levers #1).
- Sequencing: **optimize the kernel bodies FIRST; only then wire the
  whole optimization search space into planning and the wisdom
  machinery** (the axes below stay env-raced falsifiers until the bodies
  are settled; premature banking would freeze verdicts over unoptimized
  bodies).

## 3. The driver (vfft.c)

**Create (native-first):** for a dims==2 C2C INTERLEAVED create with
`VFFT_IL2D_NATIVE=1` (opt-in until the route verdict is banked), the tier
is attempted BEFORE `_build_2d` — an engaged create never builds (or
calibrates) the split tplan its execute would never touch. Any miss (env
off, chain not expressible, row-child create failure) falls through to
the convert wrapper unchanged.

**Chain selection precedence:** `VFFT_IL2D_CHAIN` env (loud warn on
invalid, greedy fallback) > banked `lay=il` verdict (`chain=` token;
product-validated; ANY-fallback rows without the token are refused) >
**create-time RACE of the full ordered-composition pool** (depth <= 4,
cap 24, the cap LOGGED when it bites) > greedy. The race times ONLY the
column pass — the component the axis changes — through
`_il2d_col_pass`, the SAME function execute serves with (race ==
serving path); the winner banks via `vw2_2d_il_chain_bank` +
`_vw2_persist`.

**Execute:** flat walk = column pass (stage-major, block-looped) then the
per-row K=1 children; banded walk per §5. OOP is free: the first executed
column stage performs the src→dst move (alias-tolerant kernels).

**Stage tables:** `_il2d_build_tables` — per t2c stage the d-major records
for `w = e^{-2πi(d·r)/L}` as `[c x4][-s,+s,-s,+s]` (svec = [-s_math,
+s_math], verified against the emitted class-constant convention); the
bwd table is the CONJUGATE (table-side, never text derivation).

## 4. The bwd contract

The tier obeys the matched-roundtrip law (the cascade's): fwd natural →
comb; **bwd consumes the SAME comb** → N·natural. bwd is the HERMITIAN
TRANSPOSE of the forward chain: stages in REVERSE order, conjugated
tables applied PRE-butterfly (`(S_m…S_1)^H = S_1^H…S_m^H`). The t2c bwd
kernels are emitted in that shape.

🔴 Recorded failure modes, do not repeat: (a) same-order conjugate bwd is
per-direction correct but breaks roundtrip for EVERY multi-stage chain
(`P·cF·P·F ≠ N·I`); (b) a palindromic-chain restriction does NOT fix it —
an involutive permutation still does not commute with the DFT (retracted
same day, measured).

Rows commute with every column stage (disjoint axes), so both directions
keep rows LAST in a band — which also makes the band's first op the
OOP-capable kernel.

## 5. The banded walk (the cascade's tcut, 2D form — SHIPPED)

Tiling is the 1D cascade's mechanism (zturn.h §TILING AXIS), NOT staging:
a **loop interchange over provably independent kernel calls on the
plane** — only loop order and base pointers change. Stage-s blocks are
`L_s` consecutive rows = CONTIGUOUS memory windows in a decreasing
divisibility chain, so:

- **width is the INPUT** (`VFFT_IL2D_WL`, in rows), **the cut is
  DERIVED**: the suffix of stages with `L_s | WL` runs depth-first per
  band; the wide prefix runs first (the mirror of MKL, whose spans grow).
- **tfuse** (`VFFT_IL2D_TFUSE`, default on) folds the ROW PASS per band —
  the terminator analog — deleting its separate full-plane sweep.
- **F0 law verbatim**: banded output must be memcmp-IDENTICAL to
  unbanded, both directions (verified at every falsifier cell).
- bwd banded = per band [REVERSED suffix, rows-bwd], then the reversed
  wide prefix (legal by the rows-commute fact).

MEASURED (chains pinned, same-run): +15% (256²), +21% (512², tfuse ~10
pts of it), +17–18% (4096x64, STACKING on the 1.30x chain win), and
LOSES 8% at 1024² — a per-cell wisdom axis (`wl=`/`tfuse=` fields,
banking queued behind the kernel-body work per §2a), never a default.

🔴 Column-axis tiling (`VFFT_IL2D_WC`) is DEAD standalone — it tiles the
comb axis (Wc columns of every row at full pitch compacts nothing);
mechanism retained only as substrate.

## 6. Ordering contract

Single-stage chains (N1 <= codelet radix): natural x natural (the column
map is the identity — simulator-proven bitwise). Multi-stage: i is
digit-reversed by the chain (the scrambled contract, matching the
library's 1D scrambled tiers), j natural from the row children; the pair
contract (§4) holds regardless. Natural order along i for multi-stage =
the P0c load-side ρ-table mechanism on the row pass — an M4 item, never
scattered stores to user memory.

## 7. Verification (all standing green)

- **Simulator** `src/core/oop/il2d_proto.h`: the column-pass DIF algebra
  (legs `x[d+tD]`, twiddle `W_L^{dr}`, same-slot stores, first-digit-most-
  significant output map), gated vs naive at 17 cells x 2 dirs (chain
  variants, aspect extremes, odd counts). ⚠ its bwd MODEL is the old
  same-order form (per-direction only) — the fwd algebra is the
  authority; update when next touched.
- **Gate** `benches/il2d_m1_gate.c`: 15 cells (N1 4..4096, prime and
  non-pow2 N2) x 2 dirs x 2 placements — fwd ELEMENTWISE vs naive through
  the (env-pinned) chain map; bwd via the PAIR contract; the race/replay
  probe (race → bank → serve BITWISE + roundtrip ~5e-16 on raced chains).
- **Falsifiers** `benches/il2d_axes_race.c` (chain/Wc),
  `benches/il2d_band_race.c` (WL/tfuse + the F0 memcmp check): same-run
  rotated arms, median + spread, deltas under spread = NOT A RESULT.

## 8. Measured position (2026-08-25, thermally-noisy caveats apply)

Native vs the convert wrapper (the interleaved caller's uplift): 1.14x –
1.65x at every cell measured EXCEPT 64x64 (0.53 — the 1D in-place IL
small-N row-service floor, ~180 ns/row vs ~30 ns mono math, probed).
Native vs MKL CCE: parity band (0.85–1.20, tilde-marked). Chain + band
verdicts add 15–50% where alive (per cell).

## 9. Work queue (owner-ordered)

1. **Blocked t2c** — r32/r64 blocked constructions emitted + adopted
   statically (§2a); r16 monolithic-vs-blocked RACED.
2. **Tangent t2c/n1c interiors** — must-have; race the twins, adopt per
   the 1D wing scoping; `[c,tan]` records = the follow-up lever.
3. Small-N2 row lever (OOP-natural-mono child + copy-back, raceable) —
   flips the one losing cell.
4. **Then** wire the whole search space (chain, wl, tfuse, body variants,
   row route) into planning + the wisdom machinery (`lay=il` cells;
   il2dkv-style variant codes; route default-on where the race says so;
   `VFFT_IL2D_NATIVE` demotes to force-override).
5. Later: 1D ILP-attach gap (lifts every scrambled in-place IL caller);
   MT over bands (the banded walk IS the parallel decomposition;
   engagement-proof discipline); `[c,tan]` records; natural-order-i via
   ρ tables (M4).

## 10. Final optimization research angle (owner, 2026-08-25)

Two items close the tier's measured story once the bodies are settled:

- **Bank the per-cell chain + band verdicts.** Already measured worth
  14–30% where alive (chain: 1.30x at 4096x64, 1.18x at 1024²; band:
  +15/+21/+17% at 256²/512²/4096x64, −8% at 1024²) — but the band axis is
  ENV-ONLY today. Adding `wl=`/`tfuse=` beside `chain=` in the `lay=il`
  cells (full-execute race arm; the chain race stays column-only) turns
  those falsifier results into served verdicts and finishes the
  per-cell-tuning story.
- **Price the wide-prefix aliasing, then decide the staged/skew route.**
  The banded walk confines the 4 KB leg-stride exposure (legs `D*N2*16`
  bytes apart, one L1 set group at pow2 N2) to the one or two WIDE PREFIX
  stages of each plan — but that residue is UNMEASURED (Q4 was never run
  for the IL side). The experiment: same-run race of a prefix stage at
  pow2 N2 vs an odd-pitch control (or a padded-plane arm); if the delta
  clears spread, the staged/skew-scratch route (the lever MKL structurally
  cannot pull — it reads the user's pitch, we would own the scratch) gets
  built for the PREFIX stages only; if not, it stays parked permanently
  with the number recorded.
