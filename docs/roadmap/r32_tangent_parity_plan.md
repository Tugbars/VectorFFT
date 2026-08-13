# R32 tangent parity plan — the 512 frontier

**Status:** A-0 EXECUTED 2026-08-13 — hand wins, **A-1 opens, scoped to the LEAF
first** (see the A-0 result box below). Plan below otherwise as proposed.

> ## ✅ A-0 RESULT (2026-08-13, 6 clean runs, controls ≤0.10%)
>
> Harness `build_tuned/benches/tangent_hand/fft512_a0.c` (7 lanes, paired
> same-round, core 2 HIGH). **Toolchain matters enormously — see the trap box
> below; all verdict numbers are under the production recipe**
> (`C:\mingw152\mingw64\bin\gcc.exe -O3 -mavx2 -mfma -march=native`).
>
> Floors (ns), stable across runs: **F hand(32,16) 305.3–306.5** · E hand(16,32)
> 310–321 (bimodal, see below) · H1 emL+handM 318.5–319.4 · H2 handL+emM
> 314–325 · **G EMITTED(16,32) 323.8–325.1** · A classic 380.7–382.1.
> G reproduces the production ~327–333 plateau in this harness ⇒ the banked
> plateau and the hand-route ~306 are now SAME-harness comparable.
>
> - **G−F = +4.4…+5.0% every run** (G-wins 2–9/35) ⇒ the branch criterion
>   ("hand wins ≥3%") fires decisively. **A-1 opens.**
> - **Slot attribution:** LEAF is at parity (H1−E = +1.6/−0.2/−0.6% ≈ noise —
>   emitted `n1ttan` ≈ hand `w16tgL`). MID is behind by **+1.0…+2.0%** (H2−E;
>   also G−H1 = +4.6…+6.5 ns — emitted `t2btan216` vs hand `w32tg`). The
>   remaining ~3% of G−F is the **SHAPE**: (32,16) ≫ (16,32) for pure tangent
>   (re-confirms the 08-11 finding), and the emitted pool **cannot build
>   (32,16) pure-tangent at all — it has no R32 tangent leaf**.
> - ⇒ **A-1 deliverable #1 = the wing32 LEAF** (unlocks the winning shape;
>   biggest single piece). Deliverable #2 = the wing32 mid (~1–2%).
> - ⚠ Observed: arms carrying the hand `w16tgL` leaf (E, H2) are **bimodal**
>   (~310 vs ~320 modes across runs) while every emitted-leaf arm is stable to
>   ±0.4%. Does not affect the verdict (F is stable and is the decision arm).
>
> ### 🔴🔴 TOOLCHAIN TRAP (cost half the session — pin it forever)
>
> The SAME sources under cygwin gcc 12.4 `-O2 -mavx2 -mfma` gave **F 347 /
> E 330 / G 327** — the hand champion 13% slower, the E/F ORDER FLIPPED, and
> G−F showed −5.7% (emitted "winning"). Under mingw 15.2 `-O3 -mavx2 -mfma
> -march=native` (= build.py's recipe): F 306, verdict reversed. The
> 08-11 binary re-run today confirmed F≈306.8 ⇒ the hand bodies' speed rides
> gcc RA luck (the banked RA-luck arc, now demonstrated at **13% route scale**),
> while the SR-scheduled emitted bodies are compiler-robust (G: 327↔324 across
> compilers). **Rule: every race in this campaign builds with the production
> mingw-15.2 recipe; a race that changes toolchain or flags is a different
> experiment.** Corollary: the emitter pipeline buys compiler robustness the
> hand artifacts do not have — when wing32 is emitted through cx, expect ±few %
> vs the hand artifact from RA alone; race, never assume.
**Target:** the 512 K=1 natural-OOP cell, currently **0.87–0.88×** vs MKL
(331–335 ns vs 291–293 on the canonical bench, `--k1noop`).
**Out of scope, on purpose:** 1024 (the tangent route inverts +16% at L1 exit —
shape problem, not arithmetic; do not promise it from this plan), R=64, and
anything split-radix/NEWSPLIT-shaped (refuted + parked, see the banner in
[`newsplit_for_cx_plan.md`](newsplit_for_cx_plan.md); SR-shaped codelets measure
far below the CT/IL forms).

## Why the R32 pass, and what is actually undecided

Three banked facts frame this:

1. **A cell crosses MKL iff both slots of its winning pair have a tangent form.**
   128 (8×16) and 256 (16×16) crossed; 512 = 2⁹ cannot form an all-R16-class
   pair — every route carries an R32 or R64 pass. Production 512 plateaus at
   ~330 ns across two structurally different routes ⇒ the residual is in the
   shared component, the R32 pass.
2. **The hand route beat that plateau — in a different harness.** `fft512_full.c`
   route F (hand `w32tgL` + `w16tg`) hit **~306 ns**, while the production
   pure-tangent route (emitted `t2btan216` + `n1ttan`, kv 51) sat at **327–333**
   in the calibrator/bench harness. Cross-harness numbers are not comparable on
   this machine. **Whether the emitted R32 construction is really ~7% behind the
   hand one is the single most decision-relevant unknown, and it is cheap to
   decide.** (Precedent: at R16 the emitted-vs-hand gap was real, and closing it
   — wing translation → CPL → RA analysis — is what produced the shipped parity
   kernel.)
3. **The constant ladder is objectively wrong at R32** (census 2026-08-13): the
   emitted tangent R32 carries **13 magnitudes in 23 vector literals**, including
   the un-reduced >π/4 complement ladder (tan 5π/16, tan 3π/8, tan 7π/16 and
   their small cosines) *and* ulp-twin duplicates — mirror sites compute tan/cos
   at the raw angle, so the same magnitude lands twice differing in the last bit
   (`_ZW1_s` = …309503 vs `_ZW12_s` = …30952), in separate un-shareable 32-byte
   vectors. MKL's col32 runs **7** broadcast constants total. Scope honestly: the
   ladder is an *internal* economy lever (MKL wins with 1.6× MORE stack traffic —
   its cross-library edge is instruction count + FMA density + lane tax); what
   the reduced ladder buys is making the 2-FMA rotation form affordable at R32
   grain.

## Track A — the R32 interior

### A-0 · DECISIVE: same-harness route race (½ session) — do this first

The hand artifacts survive in the 2026-08-11 session scratchpad
(`…/dfcd8b2a…/scratchpad/`: `w32tg_kernel.c`, `w32tgL_kernel.c`, `w32tg_gen.py`,
`fft512_full.c`, `w16tg*`). Temp dirs are not durable:

1. **Preserve first**: copy the set into `build_tuned/benches/tangent_hand/`
   (house convention: preserved race arms live under `build_tuned/benches/`).
2. Extend `fft512_full.c` with an arm built from the **shipped emitted pool**:
   `radix32_z_t2btan216` + `radix16_z_n1ttan` in the kv-51 route's constructible
   shape. Note the arms may differ in shape as well as construction (the emitted
   pool has no R32 tangent leaf — deleted after losing) — that is fine: the
   contrast we need is *best hand route vs best constructible emitted route*,
   which is exactly the production question.
3. Same harness, same tables (`w32tg` consumes t2b48's VTW2 stream verbatim),
   ONE page-aligned arena with +64 B skews, full paired protocol.

**Branch on the result:**
- Hand wins ≥3% consistently (paired, controls clean) → the emitted R32
  construction is the deficit → **A-1 is the main line**.
- Wash → the emitted construction is exonerated, the shared-layer thesis holds
  at 512 → skip A-1, keep A-2, shift weight to Track B.

### A-1 · wing32: translate the hand w32tg dataflow into cx_math (1–2 sessions)

The method is proven at R16: `w16_to_cxml.py` machine-translated the validated
dataflow into `dft_cx16_wing` (numeric self-gate 1.9e-14 *during* generation,
zero hand algebra), and that construction — not scheduling — was the −25%.
Build the R32 analog from `w32tg_gen.py`'s dataflow: two tangent DIT-16 halves
plus the 16-butterfly W32 combining stage with signs folded into constants.

Banked traps to carry in:
- Fwd kernels are **CRotNI-only** — translate the origin's +i rotations as
  `crot` with consumer sign absorption (cadd↔csub, cfma↔cfnma), never `crotp`.
- Monolithic R32 = 73/73 spills — the **blocked/halves S-plane shape is
  mandatory**; the wing32 must land as passes, not one body.
- Gates in order: OFF byte-identity (knob default-off) → `tangent_gate` →
  `il2p_tangent_gate` (the wired path is the one that catches table-layout
  bugs) → paired race vs shipped `t2btan216`.
- **The leaf twin is in scope.** The +32.4% R32-leaf kill verdict was for the
  *old blocked-2.16 construction*; the hand `w32tgL` leaf *won* inside route F.
  A wing32 leaf is a new construction — the README's "do not regenerate"
  applies to re-deriving the same thing, not this.

### A-2 · constant plumbing (independent, ~½ session — do regardless of A-0)

Two small, OFF-safe changes that fix census defect #3:

1. `cx_math` butterfly_pair general arm: **canonicalize the angle** — compute
   t, c at min(θ, π−θ) and swap opcodes for the mirror — so ulp-twins collapse
   and the table drops 23 → 13 literals with zero DAG change.
2. `cx_render` c=1 CTwC peephole: absorb the shear **sign into the opcode**
   (fmadd vs fnmadd on a shared |s| vector) so sign-mirrored sites share one
   constant.

Tolerance re-gate (these are tolerance-gated by design), regenerate via
`emit_ship.sh`, race vs shipped. Plausible mechanism: rip-mem constants sitting
on the critical path (+6c each, banked idiom-3 finding). But statics are
~7-for-8 at predicting time here — **ships only on a measured win**, else
document and drop.

### A-3 · quarter-turn composition, 13 → 7 literals (conditional, proxy-first)

Rewrite the six >π/4 shear sites to compose a reduced-angle shear with a free
∓i rotation. ⚠ This **adds a dependent rotation into the shear→butterfly
chain — the exact +depth mechanism that killed t2bpd (+2.1–2.5%)**. Therefore:
hand-edit the emitted C at those sites and race the proxy FIRST (`cil_ab`
style, paired resolution). Emitter work only if the proxy wins. If A-1 lands a
wing32 whose combining stage already has the reduced ladder by construction,
A-3 is moot — check before spending.

### A-4 · wire + bank (only for kernels that won a race)

- `il2p.h` registry: replace the variant-3 R32 symbol if strictly superseded
  (pool-sunset policy: banner + delete the loser per slot), else take the next
  nibble. Patch **both** slot enumerations (`msv` AND `lsv` — the banked
  one-slot-list bug) and verify by decoding banked nibbles, not by reading ns.
- Rebuild every wisdom-writing binary first (stale writers strip fields), run
  `calibrate_k1` v2 against a **COPY** of the wisdir, diff all non-target rows,
  hand-promote only the raced row.
- Verdict on the canonical bench `--k1noop` A/B (never a new harness).
- Plans come from dp machinery — never hand-set a route.

## Track B — the stage-count axis (independent; stacks with Track A)

**3-stage chains beat every 2-stage pair at 512 by +7.6–8.8% — measured twice,
never banked.** This is route-level and orthogonal to the R32 interior. Work
item: get the 3-stage shape into the K=1 IL plan search (planner enumeration,
not a hand row), re-race the cell, bank what wins. Run it after A-0 regardless
of which way A-0 branches — if A-0 is a wash this becomes the primary lever.

## Protocol box (non-negotiable, applies to every arm above)

- Pin core 2 (mask 0x4), HIGH priority, ≥200 ms/arm, ≥15 rounds alternating,
  ONE arena +64 B skew, paired same-round deltas + twin control; delta < control
  spread ⇒ NOT a result. `leaf16_race.c` (post-fix) is the harness template —
  the 4 KB-alias lottery has been hit four times.
- Every generator knob default-off; classic emissions byte-identical, verified.
- WSL build: dune via `/home/tugbars/.opam/5.2.0/bin`, scripts via
  `MSYS_NO_PATHCONV=1 wsl sh <script>`; NEVER bare `dune build`.
- Regenerated codelets: re-add fact-sheet headers by hand (see `emit_ship.sh`)
  or diff bodies against committed files.

## Success / exit criteria

- **Success:** 512 ≥ 0.95× on the canonical bench, banked through the dp
  re-race. Stretch: parity (the hand route's ~306 vs MKL's ~292 says ~0.95 is
  the demonstrated kernel-level ceiling of the current constructions).
- **Exit:** A-0 wash AND A-2 flat AND Track B banked ⇒ declare the 512 interior
  closed at its plateau; the residual belongs to the parked different-math
  research arc, and per the project thesis the library's wins live in batched
  MT, ≥2048 cascade, and r2c transfer — not in grinding this cell further.

**References:** `codelets/zil/avx2/pure_il/tangent/README.md` ·
`docs/performance/tangent_scaled_butterflies.md` ·
`docs/performance/v1_0_results.md` §1 ·
`docs/performance/lane_tax_in_il_codelets.md` ·
`docs/roadmap/newsplit_for_cx_plan.md` (status banner).
