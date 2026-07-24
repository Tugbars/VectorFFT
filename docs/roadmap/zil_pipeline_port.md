# zil → DAG-pipeline port: design + regression plan

Status: DESIGN (2026-07-24). No code changes yet. Follows the §4.9996 closure of the
last-15% hunt in `z_cascade_plan.md`: the zil single-thread optimization is done;
this workstream re-hosts the codelet family on the production generator pipeline
for REACH (AVX-512 / EPYC) and maintainability — explicitly NOT for i9 speed
(arithmetic parity with the pipeline is already proven; the wins were structural).

---

## 0. Why

`codelet_zil.ml` (1798 lines) is a self-contained C-string emitter: its own mini-IR
for some kinds, raw string templates for the production split family, 486 literal
`_mm256_` intrinsics, a hard `vec_width <> 4 → failwith` gate. It bypasses
~9.2K lines of shared machinery (`algsimp` cascade, `schedule.ml` SU scheduler,
`regalloc`, `emit_render`/`emit_c` devices, `Isa` parameterization). Consequences:

- **No AVX-512 / EPYC path.** The pipeline gets width-8 nearly free (all shuffle
  devices already have `_mm512` twins; `Isa.t` is consumed only at emission).
- **Family growth cost.** Every new kind (the sterm2 campaign, the msg lever, the
  bwd twins) is another hand template; the split family is now 10 production
  kernels plus ~50 spike kinds, all frozen to one ISA.
- **Pass improvements don't propagate.** FMA-lift/collect improvements, SU/GH
  scheduler wisdom, regalloc widening — none of it reaches zil.

Per the sequencing decision in §4.9995(e0): finish zil optimization first (done),
then port. This document is the port design.

---

## 1. The two trees today (established by census 2026-07-24)

### 1a. Production pipeline (the model to follow)

```
math layer            Expr.expr trees over elem_ref = Input(i,re/im) | Output | Twiddle
  dft.ml              dft_expand (n1) / dft_expand_twiddled (t1, DIT|DIF, Fwd|Bwd,
                      TP_Flat|TP_Log3) / _spill / _n1_blocked / twidsq / il2
        ↓
prepare               Pipeline.prepare_codelet  (pipeline.ml:124)
  algsimp.ml          hash-cons (tags = CSE) → dedup_sub_pairs → [aggressive passes]
  fma_passes.ml       → fma_lift → 8-remap cascade → spill_info
        ↓             returns { assigns : (elem_ref * Algsimp.t) list; spill_info }
schedule              Schedule.su_schedule_subset (schedule.ml:1134) — SU list
                      scheduler; Emit_c.classify_passes for PASS1/PASS2 membership
        ↓
emission              per-caller. codelet_oop.emit_codelet (codelet_oop.ml:1963):
                      config { radix; isa; direction; load_pat; store_pat;
                               buffer; twiddles; name }
                      edge emitters bind Input(j)/Output(j) ↔ lane_re_j/lane_im_j
                      registers; render_node_def renders each DAG node via Isa.t.
```

Facts that shape the port:

- **The DAG is layout-free and ISA-free.** `elem_ref` is element-space; memory
  binding happens in the *edge emitters*; `Isa.t` (vec_width 4/8/2/1) is consumed
  only at rendering. Tails re-run the same edges at narrower ISAs.
- **The DAG has no shuffle node.** All cross-lane movement lives in hand-written
  edge templates: UnitLeg 4×4 AOS→SOA transpose (codelet_oop.ml:462), its inverse
  store (:727), the IL de/interleave devices (:586, :920), the k1_mono T4
  transpose (:2454). AVX-512 8×8 twins exist for all of them (emit_c.ml:971+).
- **Twiddle vocabulary** (`twiddle_kind`): NoTwiddles | PerGroupTwiddles
  (`tw[(j-1)*me+b]`, per-lane vector load) | BroadcastTwiddles (set1 scalar) |
  PerPositionTwiddles. Physical indexing decided in `render_load`
  (emit_render.ml:230), not the math layer. `TP_Log3` derives twiddles as in-DAG
  cmul trees — the substitution mechanism we will reuse.
- **Vectorization model**: outer loop over the batch/column axis in vec_width
  steps; one scalar-real DAG lowered elementwise to vectors. `codelet_oop` is
  already the K=1 single-transform generator (Bailey rows, k1_mono).
- **GOTCHA (pipeline.ml:82)**: `gen_main.run` still carries an *inline copy* of
  the prepare cascade for the Emit_c families. The new zil module MUST call
  `Pipeline.prepare_codelet` (like codelet_oop), not add a third copy.

### 1b. codelet_zil (the bypass)

Two families behind one **11-arg ABI** (identical to the oop signature shape;
`zin_unused`/`zout_unused`/`tw_im` dead):

- **zx-IR kernel family** (`emit_z_kernel`): n1/t2 kinds over packed interleaved
  complex (2 complex/ymm), `zx = In|Add|Sub|RotNI|Fmadd|Fnmadd|CTw`, hand r8_body,
  blocked/blocked2 spill devices, VTW2/BYTW2 streamed twiddles, squaring tree.
  **Bench/spike only — no production reference.**
- **Split family** (`emit_z_split`, raw templates): the production cascade.
  `src/core` references exactly these 10 entry points (zsplit.h:46-52):
  `radix{4,8}_z_s0s_{fwd,bwd}`, `radix{4,8}_z_msg_{fwd,bwd}`,
  `radix8_z_sterm_fwd`, `radix8_z_sterm2_fwd`, `radix8_z_sterm_bwd`.

Split-family semantics (what must be reproduced):

| kernel | in layout | body | tw source | out layout |
|---|---|---|---|---|
| s0s | natural z, leg stride Ls (DEINT on load) | radix-R DIT, **no tw** | — | block-split planes `2*(l*Ls+k)` re, `+4` im |
| msg | block-split, in-place | radix-R DIT, tw **pre**-butterfly legs 1..R-1 | splat-pair `[c×4][s×4]` per (group,leg); cursor bumps in-kernel | block-split, in-place |
| sterm | block-split col-blocks `16*k+{0,4,8,12}`, **TR4** on load | radix-8 DIT, tw pre-butterfly | packed per-col w¹ `[c(k..k+3)][s(k..k+3)]` + **squaring tree** w²..w⁷ | **REINT** → digit-reversed z comb `2*(l*OLs+k)` |
| s0sb | block-split planes | radix-R **IDFT**, no tw | — | natural z (REINT) |
| msgb | block-split, in-place | IDFT, tw **post**-butterfly | splat-pair, **conj is table-side** (twspb has −sin) | in-place |
| stermb | drev z comb (DEINT) | IDFT, tw **post**-butterfly | packed w¹ conj table-side (twqb) + squaring tree | block-split col-blocks (TR4 back) |

Structural devices: in-kernel group loop (`bp += 2*R*Ls; twg += (R-1)*8`) with
`always_inline _body` + thin wrapper (msg); 2-quad unroll-and-jam schedule
(sterm2, bit-identical to sterm — scheduling only); 4-cols-per-iter loops
(`count % 4 == 0`); no bwd scaling (roundtrip = N·x); conj table-side.

Runtime contract (zsplit.h — FROZEN by this port): plan struct
(`chain/D/G/twsp/twspb/twq/twqb/sp/t2q`), table builders (splat-pair sets;
packed w¹ with `_vfft_zs_brev`-baked exponents), stage-call formulas, and the
kind-4 oop_wisdom / `_calibrate_zsplit_t2q` machinery in vfft.c.

---

## 2. Scope

**In scope:** the 10 production split-family kernels, re-derived through
math layer → `Pipeline.prepare_codelet` → SU schedule → shared edge/Isa emission.
Emitted function names, the 11-arg ABI, table layouts, and zsplit.h stay
byte-compatible → drop-in `.c` files in `codelets/zil/avx2/`.

**Out of scope:** the zx-IR kernel family (n1/t2 spike kinds). They stay in
`codelet_zil.ml` untouched for bench reproducibility; retirement is a separate
later decision. No runtime changes (zsplit.h/vfft.c) except the eventual
AVX-512 plan-geometry parameterization (§5, deferred).

**Non-goal:** i9 performance improvement. Acceptance is parity within the
±3% placement-luck band (§4.9993), not a win.

---

## 3. The mapping (zil device → pipeline counterpart)

| zil device | pipeline counterpart | status |
|---|---|---|
| radix-R DIT butterfly (SPLIT_BFLY4/8) | `Dft.dft_expand` / `dft_expand_twiddled ~direction:DIT` | **exists** |
| IDFT bodies (_INV) | `~sign:` \`Bwd (+θ kernel) | **exists** |
| tw pre/post butterfly | `direction×sign` pre/post logic in dft.ml:217 | **exists** |
| splat-pair mid twiddles | new `twiddle_kind` = `SplatPairTwiddles`: `Twiddle(j)` → `loadu(tw + (cursor + j)*8)` re, `+4` im (pre-splatted; no set1) | **new, small** (render_load case) |
| packed w¹ + squaring tree | new `twiddle_policy` = `TP_PowW1`: `Twiddle(0)` = vector load of w¹; W^l for l≥2 derived in-DAG via `cmul_pattern` with the squaring-tree index arrays `[0;0;1;2;2;4;4;4]/[0;0;1;1;2;1;2;3]` (crit path 3) | **new, ~15 lines** in `twiddle_expr` (TP_Log3 is the precedent) |
| table-side conj (bwd) | new `~table_conj:true` on `dft_expand_twiddled`: POST structure with `~conj:false` cmul (tables already carry −sin; avoid double conj) | **new, small** — see trap §6.1 |
| block-split plane load/store | new edge pattern `ZBlockSplit`: two plain vector ops, re `2*(l*Ls+k)`, im `+vec_width` | **new, trivial** |
| DEINT (z→planes on load) | existing IL load device (unpack + permute4x64 0xD8) — same shuffle sequence | **reuse** |
| REINT (planes→z on store) | existing IL store device | **reuse** |
| TR4 (block→leg-major on load) | existing 4×4 transpose template (UnitLeg/il_in family) with block-split addressing; AVX-512 8×8 twin exists | **reuse + new addressing** |
| digit-reversed comb store | leg-major store addressing (`2*(l*OLs+k)`) + REINT; scramble itself is plan-side (brev in tables). DAG Output(l) slots stay natural | **edge param only** |
| in-kernel group loop + always_inline body/wrapper | new structural device in the signature/loop-shell emitter: `group_loop: bool` config → emit `_body` + wrapper with `bp/twg` bumps | **new, ~40 lines** |
| 4-cols/iter loop shell | `emit_loop_open`-style column loop in vec_width steps | **exists (shape)** |
| sterm2 2-quad unroll-and-jam | il2 precedent (`dft_expand_twiddled_il2`): concatenate TWO shifted column-quad instances in one DAG; SU scheduler braids by readiness | **exists (mechanism)** — schedule quality measured, §6.2 |
| hand A/B phase interleave | `Schedule.su_schedule_subset` | **exists** (this is the point of the port) |

Everything in the "new" rows is edge/config vocabulary — no new DAG node kinds,
no scheduler changes, no new pass. The math is entirely expressible today.

---

## 4. Module design

New file `generator/lib/codelet_zsplit.ml` (name avoids clobbering the legacy
module during transition), modeled on `codelet_oop.ml`'s skeleton:

```ocaml
type zs_kind = S0S | MSG | STERM | STERM2 | S0SB | MSGB | STERMB
type config = { radix : int; isa : Isa.t; kind : zs_kind; name : string }

emit_codelet c:
  1. math      : per-kind Dft expansion (table in §3); Algsimp.reset first
  2. prepare   : Pipeline.prepare_codelet ~aggressive:false ~algorithm:(pick radix)
  3. schedule  : topo_sort_reachable → compute_inline_set → SU (same calls as
                 codelet_oop.prepare_butterfly / emit path)
  4. emission  : zs edge emitters (ZBlockSplit / DEINT / REINT / TR4 / comb) +
                 SplatPair / PackedW1 twiddle rendering + group-loop wrapper,
                 all through Isa.t helpers — zero literal intrinsics outside the
                 shared shuffle templates
```

`gen_main.ml` wiring: the existing `--z-s0s/-msg/-sterm/-sterm2/…b` flags route to
`Codelet_zsplit.emit_codelet`; a `--z-legacy` escape hatch keeps
`Codelet_zil.emit_z_split` callable for A/B during transition. zx-kind flags
(`--z-t2*`, `--z-n1*`) keep routing to the legacy module.

Build discipline (unchanged): WSL, `DUNE_CACHE=disabled`,
`/home/tugbars/.opam/5.2.0/bin/dune build bin/gen_radix.exe` with absolute
`--root`; emit via `gen_radix.exe <R> --z-<kind> --emit-c`.

ABI freeze: signature emitter reproduces the 11-arg shape verbatim, including the
dead `zin_unused/zout_unused/tw_im` slots and `(void)` silencers.

---

## 5. ISA parameterization (the payoff, and its one plan-side string)

The emitted code becomes width-parameterized for free EXCEPT the block-split
geometry: the interior block is `[re×vw][im×vw]` — 64 B at vw=4, 128 B at vw=8.
That geometry is baked into `vfft_zsplit_create`'s table/scratch layouts and the
stage-call address math. So AVX-512 support needs a `vw` parameter in zsplit.h's
plan builder (mechanical; deferred until there is AVX-512 hardware to run it —
the i9-14900KF has none, so avx512 output is **compile-gated only** for now).
The generator side carries no such coupling: `Isa.vec_width` flows through
addressing and the 8×8 shuffle twins.

---

## 6. Known traps (write-downs from the census)

1. **Double-conjugation.** zsplit.h passes *conjugated* tables to bwd kernels
   (twspb/twqb carry −sin), but `dft_expand_twiddled ~sign:`\`Bwd applies
   `~conj:true` in `cmul_pattern` → conjugating twice. The `~table_conj:true`
   variant (POST structure, plain cmul) fixes this. Gate R2 catches it
   instantly (bwd output garbage), but it should never get that far.
2. **sterm2 schedule quality.** The hand template's phase order ([all loads] →
   [TR4 A] → [TR4 B] → [alternated tw chains] → [BFLY+stores A] → [B]) is a
   *measured* winner with ±5% placement sensitivity. The SU braid over an
   il2-style 2-instance DAG will produce a *different* interleave. Do not
   argue about it: the per-cell `t2q` race (§4.9994) already picks
   sterm-vs-sterm2 by measurement, so a weaker pipeline-sterm2 degrades to
   "race picks sterm" — the cascade cannot regress past the single-quad
   baseline. If pipeline-sterm2 loses everywhere, keep the legacy sterm2
   template as the one grandfathered hand kernel and note it.
3. **Shuffle-template reuse is addressing-sensitive.** DEINT/REINT/TR4 reuse
   must bind the zil address formulas (`16*k+{0,4,8,12}`, `2*(l*Ls+k)`, `+4`)
   exactly; off-by-one in the im-half offset produces plausible-looking
   near-correct output at some sizes. Gate per-kernel, not just end-to-end.
4. **Algsimp could "improve" the arithmetic.** The cascade may contract
   differently than the hand FMA shapes (e.g. WPROD's fnmadd/fmadd pairing).
   That changes rounding, not correctness — hence R2's 1e-15 gate is vs a
   reference, not bit-vs-old. But watch R1 op counts: a *count* change
   (not just shape) means a pass misfired.
5. **Group-loop cursor is kernel state, not DAG state.** `twg` advances
   `(R-1)*8` per group — the twiddle cursor lives in the wrapper, and the
   body's `Twiddle(j)` rendering must index off the *loop-carried* pointer,
   not a `g`-indexed absolute expression (that's what keeps the addressing
   strength-reduced; MKL does the same).
6. **Determinism.** SU scheduling + hash-cons tags must be run-to-run
   deterministic (they are — no randomness in the cascade) so regenerated
   files diff clean. R0 checks this before anything else.

---

## 7. Regression plan

Ordering principle: gate each phase before the next; never compare through more
layers than necessary (per the §4.9993 measurement lessons: vary ONE thing).

- **R0 — baseline determinism.** Regen the current 10 kernels with legacy
  `codelet_zil` and diff against the committed tree byte-for-byte. Guards
  against untracked drift before the port starts. Also freeze a reference
  copy of the legacy `.c` files for the A/B harness.
- **R1 — static op-count census (per kernel).** Count
  vadd/vsub/vmul/vfmadd/vfnmadd/shuffle-class/load/store in legacy vs
  pipeline output. Arithmetic parity was proven in §4.9995(f); the pipeline
  path must reproduce it. A count delta = a pass misfire → fix before
  running anything.
- **R2 — numeric gates (per kernel, then cascade).**
  - New harness `build_tuned/benches/zil_pipe_gate.c` (modeled on
    zil_sterm_pipe.c's emit/copy controls): both versions of each kernel
    linked side-by-side (legacy names suffixed `_ref` via sed at copy time),
    identical inputs + identical zsplit.h-built tables → max-abs-delta.
    Gate: ≤1e-15 relative vs each other AND vs long-double scalar DFT
    reference. Bit-identity NOT required (SU reorders sums) but log it
    where it happens.
  - Full cascade: fwd∘bwd = N·id at 2048/4096/8192/16384, both chain shapes
    per cell (wisdom chain + default chain).
  - API level: existing gates re-run unchanged with swapped `.c` files —
    zsplit gate, `zsplit_wis_gate.c` (calibrate→wisdom→create round trip),
    API 1e-15 gates 2048–16384 fwd+bwd. PATH discipline: mingw152 bin + MKL
    bin, MKL_THREADING_LAYER=SEQUENTIAL.
- **R3 — performance race (canonical discipline).** Pinned logical core 2
  (mask 4), HIGH_PRIORITY_CLASS, 32 MB cachebust, arm rotation, Sleep
  pacing, best-of-9, cell-per-process where it matters. Two levels:
  per-kernel (legacy vs pipeline, same allocation) and full front-door
  (vfft.h) vs the §4.9994 reference numbers (fwd 0.78–0.89×, bwd 2048 ≥
  1.0×). Acceptance: within ±3% per cell (placement-luck band). sterm2
  regression is absorbed by the t2q race (§6.2); any OTHER kernel losing
  >3% → diff the asm, expect a schedule/addressing cause, fix or bank a
  finding — do not ship a regression.
- **R4 — cross-ISA.** Emit `--isa avx512` for all 10 kinds: compile gate
  (gcc -c -mavx512f) + review only. Runtime validation deferred to EPYC
  hardware (no AVX-512 on the i9). This is the reach deliverable, proven
  compilable, not benched.

**Cutover:** only after R0–R3 green: point the regen flow at
`codelet_zsplit`, regenerate `codelets/zil/avx2/` for the 10 production
kernels, re-run R2/R3 once more on the regenerated tree, and leave legacy
`emit_z_split` behind `--z-legacy` for one release cycle. User reviews and
commits every step; nothing is committed by the assistant.

---

## 8. Implementation phasing (each phase = R1+R2 gated before the next)

1. **P0** `dft.ml`: `TP_PowW1` policy + `~table_conj` variant. Standalone
   check via `bin/dbg_eval` (DAG evaluates to reference DFT numerically).
2. **P1** `codelet_zsplit.ml` skeleton + `ZBlockSplit` edges + `SplatPair`
   twiddles → the **ms body** (simplest kind: shuffle-free, in-place).
   Gate vs legacy `ms` (the per-group kind still emittable via `--z-ms`).
3. **P2** group-loop wrapper device → **msg** (+ msgb wiring shape).
4. **P3** DEINT/REINT edge reuse → **s0s / s0sb**.
5. **P4** TR4 edge + `TP_PowW1` + comb store → **sterm / stermb**.
6. **P5** il2-style 2-instance concat → **sterm2**; per-cell race verdict.
7. **P6** bwd msg twins (`~table_conj` post-tw path) → **msgb**; full-cascade
   R2/R3; cutover per §7.

Estimated new OCaml: ~600–900 lines (codelet_zsplit.ml) + ~60 in dft.ml +
~30 in gen_main.ml + render_load cases; zero changes to
algsimp/schedule/regalloc/pipeline.
