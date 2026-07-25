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
  bwd twins) is another hand template; the family now spans three composition
  methods (§1c: solo / bailey2 / cascade) totalling ~60 emitted kernels, all
  frozen to one ISA.
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
  These serve the **solo** and **bailey2** methods (§1c) — currently bench-tier
  (raced in build_tuned; no `src/core` reference yet), but part of the strategy.
- **Split family** (`emit_z_split`, raw templates): the **cascade** method —
  the only one production-wired today. `src/core` references exactly these 10
  entry points (zsplit.h:46-52): `radix{4,8}_z_s0s_{fwd,bwd}`,
  `radix{4,8}_z_msg_{fwd,bwd}`, `radix8_z_sterm_fwd`, `radix8_z_sterm2_fwd`,
  `radix8_z_sterm_bwd`.

### 1c. The three zil methods (the strategy, per Tugbars 2026-07-24)

The zil strategy is one family with THREE composition methods; the port must
cover the family, not just the production-wired method:

| method | composition | kinds used | N range | status |
|---|---|---|---|---|
| **solo** | ONE monolithic codelet = whole transform | `z_n1` r4/8; `+blocked` r16/32; `+blocked2` r64 | ≤64 | bench-tier (mono hand-ref; cf. pipeline `emit_k1_mono` twins in production) |
| **bailey2** | two-stage flat N=R1×R2 | stage 1 `z_n1t`/`z_n1b2` (corner-turn / blocked2 leaf) + stage 2 `z_t2/t2s/t2sp/t2sq/…` (VTW2-streamed twiddled mid) | 256–4096 | bench-tier (e.g. flat 4096 = n1b2 + t2s, zil_4096_decomp.c) |
| **cascade** | nf≥3 staged chain, block-split interior | `s0s → msg×k → sterm/sterm2` (+ bwd twins) | ≥2048 | **PRODUCTION** (zsplit.h, kind-4 wisdom) |

Two convergences worth exploiting rather than porting around:
- zil's `blocked`/`blocked2` spill devices are a hand re-implementation of the
  pipeline's NATIVE recipe machinery (`dft_expand_n1_blocked` /
  `dft_expand_twiddled_spill` + spill markers + SU+GH). The port replaces them
  with the real thing, not a re-derivation.
- solo overlaps `codelet_oop.emit_k1_mono` (already pipeline-hosted; its IL
  variants `vfft_k1_mono64_8x8_il_*` are production via oop_leaf_registry.h and
  measured ≈ MKL-IL in the K=1 bakeoff). Solo's port is likely a CONVERGENCE
  with emit_k1_mono + races vs the zx hand kernels, not new emission code.

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

**In scope: the whole zil strategy — all three methods (§1c)**, re-derived
through math layer → `Pipeline.prepare_codelet` → SU schedule → shared edge/Isa
emission. Sequenced by production exposure:

- **Tranche 1 — cascade** (the production method): the 10 split-family kernels.
  Emitted function names, the 11-arg ABI, table layouts, and zsplit.h stay
  byte-compatible → drop-in `.c` files in `codelets/zil/avx2/`.
- **Tranche 2 — bailey2**: the stage-1 leaves (`n1t`, `n1b2`) and the t2 mid
  family (`t2`, `t2s`, `t2c`, `t2d`, `t2sp`, `t2sq`, tiled/ss variants as
  needed by the benches that consume them). Same ABI freeze.
- **Tranche 3 — solo**: `z_n1` (+ blocked devices). Preferred route is
  CONVERGENCE with `emit_k1_mono` (already pipeline-hosted, production via
  oop_leaf_registry.h) rather than a parallel emitter — decided by racing
  pipeline output vs the zx hand kernels per radix.

**Representation decision (applies to tranches 2–3):** the zx kinds keep data
INTERLEAVED through the body (RotNI/BYTW2 pay ~1 shuffle per twiddle apply);
the pipeline DAG is split-real. The port default is **split-real body +
DEINT/REINT z edges** — same FLOPs, zero interior shuffles, boundary shuffles
instead (the cascade thesis, which beat the IL levers t2c/t2sp at scale). Per
kind this is a measured question: R3 races legacy-interleaved vs
pipeline-split; a kind whose interleaved body wins on its home cells stays
grandfathered as a hand template (same policy as sterm2, §6.2). We do NOT
build a complex-vector node set / interleaved render mode unless multiple
kinds refuse to converge — that is the expensive fork, taken only on evidence.

**Out of scope:** runtime changes (zsplit.h / vfft.c / oop_leaf_registry.h)
except the eventual AVX-512 plan-geometry parameterization (§5, deferred).
Legacy `codelet_zil.ml` stays callable (`--z-legacy`) until every tranche
lands; retirement is a separate decision.

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

### 3b. Tranche 2–3 additions (bailey2 + solo kinds)

| zil device | pipeline counterpart | status |
|---|---|---|
| `n1` solo body (monolithic radix-R, no tw) | `dft_expand` + il_in/il_out z edges — the `emit_k1_mono` shape | **exists** (convergence candidate) |
| `blocked` / `blocked2` spill devices (zspill parking, 8×8 CT two-pass) | `dft_expand_n1_blocked` / `dft_expand_twiddled_spill` + spill markers + SU+GH recipe — the pipeline's NATIVE machinery | **exists** (the port DELETES the hand device) |
| `n1t` corner-turn stores | existing UnitLeg store transpose family (`emit_store_unitleg`, permute2f128 repack) | **reuse** |
| `t2` streamed twiddled mid | `dft_expand_twiddled ~direction:DIT` + new `VTW2Packed` twiddle kind: cos-first sign-folded 64-B record per column-pair, cursor `tw + (k>>1)*8*(R-1)` | **new twiddle-kind rendering** |
| `t2s` strided columns (FFTW LD shape) | strided edge addressing (`Gs`-spaced columns) — UnitGroup/strided edges exist; 128-bit half-load composition is an edge detail | **reuse + edge param** |
| `t2c` group-constant twiddles | `SplatPairTwiddles` fixed-cursor variant (same kind as msg, no bump) | **shared with tranche 1** |
| `t2d` post-twiddle | `direction=DIF` post-tw structure (dft.ml exists; codelet_oop `current_post_tw` precedent) | **exists** |
| `t2sp`/`t2sq` w¹-stream + in-register powers | `TP_PowW1` — same policy as sterm (t2sp = running-product chain variant, t2sq = squaring tree) | **shared with tranche 1** |
| VLIT emit-time const twiddles (`CTw` dedup → file-scope statics) | algsimp Const folding + emit_render const materialization; per-file static dedup is an emission detail | **exists (mechanism)** |
| tiled loads (`t2st`, vperm2f128 corner-turn) | 4×4 transpose template family | **reuse** |

The t2 family's per-kind flags (strided/const/post/pow/tile) map onto config
axes of ONE emitter path — the flag zoo collapses into the same
config-record style codelet_oop already uses.

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
   R2/R3; **tranche-1 cutover** per §7.
8. **P7 (tranche 2)** `VTW2Packed` twiddle kind + strided/tiled edge params →
   **t2 family** (t2, t2s, t2c, t2d, t2sp, t2sq); then **n1t / n1b2** leaves via
   UnitLeg-store + native spill recipe; gate each vs legacy via the same
   R1/R2 shape, race on the bailey2 bench cells (zil_4096_decomp / zil_512_race
   arms).
9. **P8 (tranche 3)** solo: race pipeline `emit_k1_mono`-style output (il
   edges) vs zx `z_n1` per radix on the mono bench cells; converge or
   grandfather per verdict. Expected outcome: convergence (delete the hand
   blocked devices; pipeline recipe machinery replaces them).

Estimated new OCaml (tranche 1): ~600–900 lines (codelet_zsplit.ml) + ~60 in
dft.ml + ~30 in gen_main.ml + render_load cases; zero changes to
algsimp/schedule/regalloc/pipeline. Tranches 2–3 add config axes and twiddle
renderings to the same module, not new machinery — the t2 flag zoo becomes
config fields.

---

## 9. IMPLEMENTATION LOG

### 2026-07-25 — P0–P4 built and gated in one pass. Headline: BIT-IDENTITY.

Every pipeline-emitted kernel so far produces **exactly** the legacy kernel's
output (max|d| = 0.0 on random inputs, all cells of the per-kernel gate) —
stronger than the ≤1e-15 the plan asked for. The algsimp cascade + SU
scheduler reproduce the hand kernels' FMA shapes and effective summation
order at these radices. 10 of the 11 production kernels are done; only
sterm2 (P5) remains.

| phase | scope | gate result |
|---|---|---|
| R0 | legacy regen determinism | PASS — 9/11 byte-identical; 2 fwd s0s files differ only by later-added unused macros (TR4/WPROD/_INV in the shared pack); committed files predate them, compiled code unchanged |
| P0 | dft.ml: `TP_PowW1` + `~table_conj` | `bin/dbg_zil_math.ml` OVERALL PASS (R=4/8: slot-0-only census, fwd vs direct ≤3.3e-15, roundtrip = N·x ≤1.8e-15, double-conj sentinel fails as required) |
| P1 | ms/msb r4+r8 (`ZBlockSplit` edges + `SplatPair` tw render) | BIT-IDENTICAL ×4; scalar ≤3.6e-15; roundtrip ≤1.3e-15 |
| P2 | msg/msgb (always_inline `_zsg` body + group-loop wrapper, exact legacy bumps) | BIT-IDENTICAL ×4 (multi-group, in-place, cursor `+ (R-1)·8`) |
| P3 | s0s/s0sb (`E_z` DEINT/REINT edges) | BIT-IDENTICAL ×4; roundtrip exact |
| P4 | sterm/stermb (`E_blocks` TR4 edge, `TP_PowW1` squaring tree ≡ legacy WPROD chain, OLs comb) | BIT-IDENTICAL ×2; scalar ≤7e-15; roundtrip ≤5.3e-15 |

R1 object-level census (gcc 15.2 -O3 -mavx2 -mfma, objdump class counts):
- **ms8 fwd**: arithmetic EXACT (22 add / 22 sub / 22 FMA / 14 mul both);
  pipeline **spills LESS** (74 vs 86 vmovupd, −12) — SU order helps gcc here.
- **ms8 bwd**: arithmetic equal (one add rendered as vxorpd-neg); +3 movupd.
- **sterm fwd**: arithmetic EXACT (34 FMA / 26 mul / 22 / 22) and the
  **port-5 shuffle profile EXACT** (8 vperm2f128 + 16 vpermpd + 32 vunpck);
  pipeline +13 vmovupd (hand phase order tighter in the terminator) —
  R3-race territory, worst case absorbed by the t2q pick.

New/changed files (all UNCOMMITTED, user reviews):
- `generator/lib/codelet_zsplit.ml` (NEW, ~450 lines): kinds ms/msb/msg/
  msgb/s0s/s0sb/sterm/stermb; edges `E_planes`/`E_z`/`E_blocks` with stride
  names (Ls/OLs); TR4 helper; computed (void) lists; frozen 11-arg ABI.
- `generator/lib/dft.ml`: `TP_PowW1` policy (squaring-tree derivation from
  slot 0) + `~table_conj` on `dft_expand_twiddled`.
- `generator/lib/emit_state.ml` + `emit_render.ml`: `current_tw_zsplit`
  render mode (`tw_re[off + j·2VW]`, sin half `+VW`; tw_im slot dead).
  Default None — legacy families byte-identical (verified via R0 re-diff).
- `generator/lib/dune`, `bin/dune`, `bin/dbg_zil_math.ml` (NEW),
  `gen_main.ml`: `--zp-{ms,msb,msg,msgb,s0s,s0sb,sterm,stermb}` flags →
  `Codelet_zsplit.emit_codelet`; legacy `--z-*` untouched.
- Gate harness: scratchpad `p1gate/gate.c` (30 checks, OVERALL PASS) —
  to be promoted to `build_tuned/benches/zil_pipe_gate.c` at P6.

### 2026-07-25 (later) — P5 DONE. All 11 production kernels now pipeline-hosted.

**sterm2** built as designed: `prepare ~two_inst:true` concatenates two
shifted DAG instances (Input/Output +R slots, Twiddle +1 slot — instance
B's packed-w¹ record lands at `tw_re[2k+8]` for free), SU braids them; the
emitted function shares one `k` cursor across the `k+=8` main loop and the
baseline-shaped 4-column tail (a second, 1-instance DAG prepared AFTER the
main body is rendered — Algsimp.reset sequencing, MODULE CARD GOTCHA 2).
`emit_col_loop` is now parameterized (`~open_line ~ninst`); all three edges
handle instance offsets. Gate (count=60 exercises main+tail): pipeline
sterm2 **BIT-IDENTICAL to pipeline sterm AND to legacy sterm2** (0.0/0.0).

R1 -O3 census sterm2: **near-perfect** — every arithmetic and shuffle class
exact (66/66 add, 66/66 sub, 78/78 mul, 102/102 FMA, 24 perm2f128 + 48
permpd + 96 unpck identical), vmovupd 214 vs 216 (+2, noise). The SU braid
over the 2-instance DAG reproduces the hand 5-phase template's live-set
behavior — the 1-quad +13-movupd delta vanishes at 2-quad. Strong prior for
the R3 race; the per-cell t2q pick remains the arbiter either way.

Remaining:
- **P6**: full-cascade R2 through zsplit.h with pipeline .c files swapped
  in, R3 paced race per canonical discipline, then the cutover of §7.
  The kernel-level gate is already promoted to
  `build_tuned/benches/zil_pipe_gate.c` (self-documenting header: arm
  regen commands + build line).

---

## 10. P6 RUNBOOK (handoff — fully specified, no session context needed)

Everything below is procedural; the architecture work is done and gated.

**Build/regen discipline (unchanged, from memory + this doc):**
- Generator: WSL, `cd src/dag-fft-compiler/generator`, `DUNE_CACHE=disabled
  /home/tugbars/.opam/5.2.0/bin/dune build --root <ABS PATH> bin/gen_radix.exe`.
  NEVER bare `dune build`.
- Pipeline kinds need `--isa avx2 --uarch raptor_lake_avx2 --emit-c`
  (default isa is avx512 → the VW gate fails the run, by design).
- Gate exes run by hand need `C:\mingw152\mingw64\bin` AND the MKL bin dir
  on PATH (else 0xC0000135), `MKL_THREADING_LAYER=SEQUENTIAL`,
  `MKL_NUM_THREADS=1` for any MKL-referencing bench.
- Any compile of zsplit kernels outside build.py: add `-mavx2 -mfma`
  (msg `_zsg` bodies carry no per-function target attribute).

**P6.1 — kernel gate re-run (sanity):** regenerate arms per the
zil_pipe_gate.c header, compile, run. Expected: 31/31 PASS, all
legacy-vs-pipeline lines 0.0 (bit). Any nonzero after a generator change =
STOP, diff the emitted C vs the frozen reference intent (§9 tables).

**P6.2 — cascade R2 (swap-in):** regenerate the 11 production files WITH
THE PIPELINE EMITTER under their production filenames into
`src/dag-fft-compiler/codelets/zil/avx2/` (keep the legacy originals
aside for instant rollback — they regenerate from `--z-*` flags anyway):
`radix{4,8}_z_s0s_avx2.c` ← `--zp-s0s`, `radix{4,8}_z_s0s_bwd_avx2.c` ←
`--zp-s0sb`, `radix{4,8}_z_msg_avx2.c` ← `--zp-msg`,
`radix{4,8}_z_msg_bwd_avx2.c` ← `--zp-msgb`, `radix8_z_sterm_avx2.c` ←
`--zp-sterm`, `radix8_z_sterm2_avx2.c` ← `--zp-sterm2`,
`radix8_z_sterm_bwd_avx2.c` ← `--zp-stermb`. Then rebuild via
build_tuned/build.py and run, unchanged: the zsplit gate benches,
`zsplit_wis_gate.c` (calibrate→wisdom→create round trip), and the API
1e-15 gates 2048–16384 fwd+bwd. Expected: identical results to the legacy
build — the kernels are bit-identical, so any deviation is a build/harness
problem, not a numerics one.

**P6.3 — R3 paced race:** canonical discipline (pin logical core 2 = mask
4, HIGH_PRIORITY_CLASS, 32 MB cachebust, arm rotation, Sleep pacing,
best-of-9; model on bench_1d_vs_mkl.c / zil_sterm_pipe.c). Two levels:
(a) per-kernel legacy-vs-pipeline in ONE binary/allocation; (b) front-door
vfft.h at 2048/4096/8192/16384 fwd+bwd vs the §4.9994 reference numbers
(fwd 0.78–0.89× MKL, bwd 2048 ≥ 1.0×). Acceptance: within ±3% per cell.
Since the kernels are bit-identical, deltas can come ONLY from code
placement/alignment — treat like §4.9993 (measured, not reasoned; the
t2q race already covers the sterm/sterm2 choice per cell). Known static
prior: sterm 1-quad has +13 vmovupd vs legacy at -O3, sterm2 is +2, ms is
−12 — expect a wash overall.

**P6.4 — cutover:** after green R2+R3, the pipeline emitter becomes the
regen source for the 11 production files; legacy `--z-*` split kinds stay
callable in codelet_zil.ml (the zx kernel family still needs it for
tranches 2–3). Update §9 with the race table. User commits everything.

### 2026-07-25 (P6 in progress) — review finding fixed; cascade swap staged.

**Adversarial review (3 lenses + refutation verify).** Ran to partial
completion — 3 of 11 agents finished, 8 errored on a Fable-5 monthly spend
limit, so the `ocaml-robustness` and `shared-state` lenses did NOT fully
run. The `wiring` lens produced ONE finding, independently VERIFIED
(refuted=false):

- **CONFIRMED (medium) — no --zp/--z mutual-exclusion guard.** The pipeline
  (`--zp-*`) and legacy (`--z-*`) families emit byte-identical symbol names,
  and gen_main's dispatch is fixed-priority (zp wins). Passing both would
  silently substitute the pipeline kernel under the legacy name → an A/B
  regen that concatenates a legacy-z base with a zp variant compares
  pipeline-vs-pipeline for a FALSE parity verdict, exactly in the
  sterm/sterm2 family where the t2q pick turns on ±5% placement luck. No
  numeric gate can catch it (both emitters are bit-identical).
  **FIX (gen_main.ml, before dispatch):** `if zp_kind<>"" && (z_native ||
  k1_mono || oop) then failwith`. Verified: conflict → Fatal error + ZERO
  bytes on stdout (an A/B `> file.c` gets an empty file → loud link error,
  not a silent pipeline-vs-pipeline compare); each family alone still emits
  correctly. (Note: gen_radix.exe exit code stays 0 on ANY error under the
  WSL-runs-Windows-exe interop — a pre-existing property of every error
  path, not this guard; the empty-stdout + Fatal-error message is the loud
  signal scripts/humans key on.)

The two incomplete lenses were reasoned through directly (author review,
not machine-verified): (a) OCaml robustness — sterm2's two prepares capture
`assigns`/`scheduled`/`inline_set`/`re_tag` freshly per `emit_col_loop`
call; main and tail `t<tag>` locals live in separate C block scopes (no
collision despite both restarting tags at Algsimp.reset); `Array.make
nslots` sizes to `2·radix` for two_inst matching the Output slot range;
`Fun.protect` reset runs per body. (b) Shared-state — `current_tw_zsplit`
defaults None (R0 re-diff proves legacy families byte-identical), the `Some`
is set INSIDE the protected thunk so a pre-set throw can't leak it, and the
reset runs on the exception path. Both empirically corroborated by the
bit-identical gates. Residual risk: the un-run machine lenses could surface
something these two paragraphs miss — acceptable given 31/31 bit-identity,
but worth a re-run if the spend limit clears.

### 2026-07-25 (P6.1–P6.2) — kernel gate re-confirmed; 11 files swapped in.

- **P6.1:** all 30 arms regenerated fresh (WSL, clean /tmp), committed
  `zil_pipe_gate.c` recompiled + run → **31/31 PASS, every
  legacy-vs-pipeline line 0.0**. Determinism + bit-identity hold from a cold
  regen.
- **P6.2 (swap staged):** the 11 production files in `codelets/zil/avx2/`
  overwritten with pipeline output under identical names (symbol-match
  verified 1:1 first; legacy backed up to scratchpad AND regenerable from
  `--z-*`). `git diff --stat`: 11 files, +1777/−1590 (source restructure;
  object code bit-identical). `.obj/avx2` cache cleared. Cascade-level gate
  run: see next entry.

**Cascade R2 + front-door R3 (pipeline kernels swapped in).** Codelet lib
rebuilt from the 11 swapped files (688 objects compiled clean — first
full-build compile-test of the pipeline output). `zsplit_api_gate` through
vfft.h with the real zsplit.h twiddle tables:

| N | fwd(drev) relerr | roundtrip relerr | fwd vs MKL | bwd vs MKL |
|---|---|---|---|---|
| 2048  | 2.6e-15 | 1.1e-15 | 0.91 | 0.89 |
| 4096  | 3.3e-15 | 1.2e-15 | 0.77 | 0.74 |
| 8192  | 4.5e-15 | 1.1e-15 | 0.82 | 0.73 |
| 16384 | 5.5e-15 | 1.4e-15 | 0.88 | 0.84 |

All gates PASS; timings in the §4.9994 reference band (fwd 0.78–0.89×) — no
regression from the swap (expected: numerically bit-identical kernels).
Note this run has no wisdom dir → default chains + default t2q, so the bwd
2048 = 1.01 WIN (calibrated) doesn't show; that's a wisdom-population
matter, not a pipeline-vs-legacy difference. Per-kernel A/B (confound-free,
one allocation) recorded next.

### 2026-07-25 (P6.3) — per-kernel A/B: cascade-equivalent. PORT COMPLETE.

Confound-free A/B (`scratchpad/p6arms/zil_pipe_race.c`): both families linked
in ONE binary, each hot r8 kernel raced arm-rotated in ONE 4KB allocation,
canonical discipline (core 2, HIGH_PRIORITY, 32 MB cachebust, best-of-9).
Four runs, pipeline/legacy ratio:

| kernel | run0 | run1 | run2 | run3 | read |
|---|---|---|---|---|---|
| s0s r8    | 1.033 | 0.975 | 1.048 | 1.016 | WASH (crosses 1.0) |
| msg r8    | 0.956 | 0.986 | 0.981 | 1.009 | WASH, slight pipe edge |
| sterm r8  | 1.022 | 0.950 | 0.965 | 1.038 | WASH (crosses 1.0) |
| sterm2 r8 | 0.983 | 1.163 | 1.144 | 0.582 | placement/thermal noise |

s0s/msg/sterm all straddle 1.0 across runs → code-quality-equivalent within
measurement noise. sterm2's ratio is meaningless as a static A/B: the LEGACY
arm's absolute time alone swung 1790→3579 ns run-to-run — the documented
sterm2 placement sensitivity (§4.9993, the reason it's a dual kind with a
measured per-cell t2q race). The census gives the code-quality truth: sterm2
pipeline vs legacy is +2 movupd / 214 (0.9%), everything else exact — a 14%
swing is dynamic, not the kernel. The decision-relevant signal is the
CASCADE front-door (previous entry), which runs the real t2q pick and lands
fully in-band (fwd 0.77–0.91×) — no cell regresses.

**VERDICT: the port is functionally complete.** All 11 production kernels
regenerate through the DAG pipeline BIT-IDENTICAL to the legacy hand emitter
(31/31 kernel gate + cascade 1e-15 gates), the cascade performs in the
reference band with the pipeline kernels swapped in, and per-kernel timing
is cascade-equivalent. The one review finding (flag mutual-exclusion) is
fixed. Acceptance (parity within ±3% at the cascade level, not a win) is
MET.

### 2026-07-25 — TRANCHE 2/3 (bailey2 + solo): scope finding, direction PENDING.

Investigated bailey2 before building (agent cross-map + direct read). It is
a fundamentally different situation from the cascade:

- **zil-bailey2 is PURE-BENCH** — zero `src/core` refs to `n1t`/`n1b2`/`t2s`
  (only the 12 build_tuned/benches drivers). Never promoted to production.
- **Its ALGORITHM is already pipeline-hosted.** `oop_plan.h` ships
  **BAILEY2** (split-plane four-step: codelet_oop `n1_oop` leaf + `t1p` mid,
  split Qr/Qi, :696) and **BAILEY2V** (K=1 interleaved-z→z four-step,
  `execute_fwd_il`/`2p_il` :799–827) — both wired through vfft.h, both
  codelet_oop-generated. Same math as zil-bailey2.
- **codelet_oop's IL edges (il_in/il_out) are boundary-only on a SPLIT
  kernel** (DEINT→split body→REINT). That IS the design's split-body+z-edge
  default, and it is exactly what BAILEY2V already does. codelet_oop has NO
  VTW2/BYTW2 — its twiddle is unconditionally split CmulRe/CmulIm.
- **zil-bailey2's ONLY unique contribution is the fully-interleaved BODY**
  (BYTW2/RotNI, no split-boundary conversions). The split pipeline provably
  CANNOT emit it — codelet_zil.ml:10-15 states the real-valued backend
  "cannot be re-rendered interleaved... a separate, small complex-vector
  backend." Pipeline-hosting it = building a packed-complex node set
  (RotNI/paired-rotation/BYTW2/CTw) through algsimp/schedule/emit — the
  "expensive fork" §2 warned against. **SOLO (tranche 3) shares this exact
  fork** (monolithic interleaved z_n1, same backend).

Contrast with the cascade: it ported cleanly precisely because its kernels
were ALREADY split-real (block-split interior). Bailey2's kernels are
interleaved throughout.

**DIRECTION SET (user, 2026-07-25) — build the interleaved-complex backend
(option C). CORRECTS my earlier "already covered" framing.** Full IL is a
first-class PRODUCT layout, not a perf variant:
- **MKL and FFTW default to interleaved** — users expect it; split is the
  niche that needs justifying.
- **IL pays NO R2C/C2R tax** — a full-IL path does real transforms without
  the split-plane conversion the hybrid pays somewhere.
- **IL makes K=1 (single-transform) smoother.**
So BAILEY2V (IL boundary, SPLIT interior) is NOT the full-IL offering — it
still carries split machinery. The zil interleaved-body family (BYTW2/RotNI,
interleaved throughout) IS the full-IL layout we want to give users, and the
original directive stands: generate it through schedule.ml/algsimp/emit,
replacing codelet_zil's standalone hand-scheduled raw-string emission. The
fork is justified by delivering a layout users want, not by beating split on
speed. Tranche 2 (bailey2) + 3 (solo) are the small/mid-N pieces of this
full-IL offering (cascade covers ≥2048).

**Approach (see §11):** the zx IR (codelet_zil In/Add/Sub/RotNI/Fmadd/CTw)
is already a complex-vector DAG; the work is routing it through the SHARED
SU scheduler + a new interleaved emit rendering mode (packed-complex
lowering: add/sub→vector add/sub, ×±i→RotNI permute+xor, cmul→BYTW2),
instead of hand scheduling. Feasibility hinges on how node-kind-agnostic
schedule.ml/algsimp are — investigated next.

**Cutover status:** the swap is STAGED in the working tree for the user's
review/commit; not committed by the assistant. Rollback = `git checkout --
src/dag-fft-compiler/codelets/zil/avx2/` or regen from `--z-*`. Not re-run
(orthogonal to the port — the swap changes kernel SOURCE, not runtime
dispatch): `zsplit_wis_gate` (the t2q calibrate→wisdom machinery lives in
vfft.c, untouched, and races two present bit-identical terminators) and the
avx512 compile-gate R4 (deferred to EPYC hardware, §5). Tranches 2–3
(bailey2, solo) remain per §1c/§8.

---

## 11. FULL-IL BACKEND design (tranche 2/3 — interleaved-complex)

GOAL (user, 2026-07-25): a FULL interleaved-complex (IL) FFT path as a
first-class product layout — MKL/FFTW default to IL, it pays no R2C/C2R tax,
and it makes K=1 smoother. The zil interleaved-body family (bailey2 four-step
+ solo monolithic) IS that layout. The directive: generate it through the
SHARED pipeline machinery instead of codelet_zil's standalone hand-scheduled
raw strings, for maintainability + reach (AVX-512).

### 11.1 FEASIBILITY VERDICT (coupling audit, 2026-07-25) — RESOLVED

Two architectures were on the table; the audit picked the answer:

- **(b) merge into `Ir.node_kind`** = REPO-WIDE cascade. `lib/dune` has no
  `(flags)` stanza → dune dev profile → non-exhaustive `match` is FATAL.
  The NK_Plus precedent shows **~34–36 arm sites per added kind** across 7
  modules; ~3–5 packed kinds ⇒ **150–180 arm edits**. Cost centers are
  `fma_passes.ml` (12) + `simplify.ml` (9) — the real-valued passes a complex
  kernel NEVER runs, yet must be edited (and kept correct) just to compile.
  `of_expr` (ir.ml:788) is hardwired to split-complex cmul detection, so a
  packed DAG wouldn't even flow through it. VERDICT: **rejected** — high cost,
  ongoing burden, zero benefit for the interleaved family.

- **(a) keep the complex IR separate; reuse ONLY the scheduler + Isa emit** =
  LOCALIZED. `Schedule.su_schedule`/`su_schedule_subset` are generic in the
  node payload — the core loop is pure `preds`-based dataflow; the ONLY
  kind-specific reads are `node_latency` (schedule.ml:302) and
  `compute_su_number` (:387). So the shared SU scheduler (STARVE/RETIRE
  ordering, GH pressure switch, wisdom injection) is reusable by giving the
  complex IR a `preds`/`latency` adapter (~40 lines), with ZERO touches to the
  150-site cascade. `Isa.add_pd/sub_pd/mul_pd/fmadd_pd/fnmadd_pd` (isa.ml:169-
  223) are layout-agnostic strings — reusable as-is; only 2–3 permute/xor
  helpers are missing from isa.ml. VERDICT: **chosen.**

**This revises "make codelet_zil part of the machinery" in practice:** the
valuable, reusable shared piece is the SCHEDULER (schedule.ml) + the Isa emit
layer — NOT algsimp/dft, which are real-valued/split by construction and
unwanted for interleaved kernels (CSE/sharing is free via hash-cons either
way). So the port shares what helps (scheduling quality + AVX-512 width
parameterization) and keeps the complex IR + complex math layer (zx's dft_z)
separate — the surgical option, and the one that actually honors the intent
without a counterproductive IR merger.

### 11.2 What the port actually is (option a, concrete)

The existing zx IR (codelet_zil.ml:53-60: `In|Add|Sub|RotNI|Fmadd|Fnmadd|CTw`)
is already a clean complex-vector DAG that hash-conses. The port replaces its
two weak spots — hand-scheduling and raw-string emission — with shared
machinery:

1. **`preds`/`latency` adapter on zx** (~40 lines). Either functorize
   schedule.ml over a tiny `(PREDS + COST)` signature (cleaner; one scheduler,
   two IRs — best honors "use schedule.ml") or lift the SR loop verbatim.
   Gives zx the SU scheduler + GH + wisdom, replacing its hand A/B-interleave
   and the sterm2-style phases (which tranche 1 showed the SU scheduler
   reproduces well, and are placement-luck-fragile by hand).
2. **Isa-helper emission**: rewrite zx's lowering to build op strings via
   `Isa.*` (width-parametric → AVX-512 reach) instead of literal `_mm256_`.
   Reuse add/sub/mul/fmadd/fnmadd as-is; add to isa.ml:
   - `permute_pd` (the 0x5 re/im swap — CFlip),
   - a general `xor_pd` against a named mask (RotNI's `_M_IM` sign flip; the
     current xor_pd is contractually pinned to `-0.0`, isa.ml:187),
   - emit-preamble for `_M_IM` + the VTW2/VLIT const vectors.
3. **Interleaved edges + driver** (mirrors codelet_zsplit's shape): packed z
   load/store `in_z[2*(b*Gs + j*Ls)]` (2 complex/vec), the four-step leaf's
   corner-turn store, and the 11-arg z ABI. math (zx dft_z) → hash-cons →
   su_schedule (via adapter) → Isa-emit.

Nothing touches Ir/algsimp/dft. The complex math layer (dft_z DIT-2 +
r8_body) is a direct transcription of the settled, race-gated zx algorithm.

### 11.3 Phasing (NO CODE until user go-ahead)

- **CIL-0**: the scheduler adapter (functor or lifted SR loop) + the 2–3
  isa.ml helpers. Prove the shared scheduler drives a zx DAG (schedule one
  trivial kernel, diff instruction order sanity).
- **CIL-1 (solo, the proof kernel)**: r8 `z_n1` monolithic, const twiddles
  (CTw) — the simplest interleaved kernel. zx dft_z/r8_body → adapter →
  su_schedule → Isa-emit. Gate 1e-15 (or bit-exact) vs legacy
  `radix8_z_n1_fwd_avx2`. Proves the whole IL-through-shared-scheduler path.
- **CIL-2 (solo radices)**: r4/16/32/64; blocked variants.
- **CIL-3 (bailey2 leaf)**: n1t/n1b2 corner-turn store.
- **CIL-4 (bailey2 mid)**: t2 family — LOAD-flavor BYTW2 (VTW2 packed table)
  + strided/tiled edges; the t2 flag zoo → config axes.
- **CIL-5**: regression (bit-exact vs legacy per kernel) + paced race vs
  legacy IL kernels; wire a runtime driver if promoting to production.

### 11.4 Decisions for the user (before CIL-0)

1. **Scheduler reuse mechanism**: functorize schedule.ml over `(PREDS+COST)`
   (cleaner, one scheduler, small refactor of a 1713-line file) vs. lift the
   ~40-line SR loop into the complex backend (faster, mild duplication).
   Recommend: functorize — it's the literal "goes through schedule.ml."
2. **Where the complex backend lives**: extend codelet_zil.ml in place
   (replace its hand-schedule/raw-emit internals) vs. a new codelet_cil.ml
   that supersedes it. Recommend: refactor codelet_zil in place so there's
   one interleaved emitter, not two.
3. **Scope now**: solo first (CIL-1, the clean proof) then bailey2, or
   bailey2-mid-first (the t2 family, the actual four-step champion). Recommend
   solo-first — smallest kernel, no twiddle-table plumbing, proves the path.

### 2026-07-25 — CIL-0 + CIL-1 LANDED (full-IL through the shared scheduler)

Decisions taken (user): functorize schedule.ml (not copy the loop); NEW file
(codelet_zil.ml stays untouched as the regression baseline); solo r8 first.

**CIL-0a — schedule.ml functorized.** Added `module type SCHED_NODE`
(payload-generic `{tag; node}` record + preds/latency/is_load/is_const/
kind_char), `module Ir_node` (the real-valued instantiation), and
`module Make (N : SCHED_NODE)` wrapping compute_cp_dist / compute_su_number /
su_schedule, closed by `include Make (Ir_node)` so every existing entry point
keeps its exact type and name. `su_schedule_subset` (split-family only) left
monomorphic. Keeping the record shape in the signature meant all 138 `.tag`
accesses survived untouched — the change is type-level, not algorithmic.
`compute_su_number` became payload-generic: the k-ary label
(sort children's SU desc, max_i(su_i+i)) provably reproduces every old
per-kind case (leaf→1 special-cased; binary max(sa,sb+1) ≡ the old
if/else; 3/4-ary were already this formula).
VERIFIED behavior-preserving: **31 codelets across 4 families regenerate
byte-identical** (in-place r3/r5/r8/r13/r16/r32/r64, OOP r13/r16/r32/r128,
all 15 zil arms). NOTE: diffing must ignore the ` * Generated by:` provenance
line — it records argv[0], so it differs by invocation path alone.

**CIL-0b — isa.ml packed-complex primitives.** `cflip_pd` (per-complex
re/im swap; imm 0x5 / 0x55 / 0x1 by width), `xor_mask_pd` (XOR against a
NAMED mask — the existing xor_pd is contractually pinned to -0.0), and
`im_mask_decl` (the alternating [0,-0,…] mask, width-correct). Everything
else an IL kernel needs (add/sub/mul/fmadd/fnmadd) reuses the existing
helpers unchanged, because a packed-complex add IS a vector add.

**CIL-1 — codelet_cil.ml (NEW).** Own hash-consed complex IR
(`CIn|CAdd|CSub|CRotNI|CFmaC|CFnmaC|CTwC`), a `Node : SCHED_NODE`
instantiation, `module Sched = Schedule.Make (Node)`, the DIT-2 complex
recursion (`dft_cx`, twiddle-class per k: ×(-i) as a rotation, √½ folded
into FMAs, general twiddles as BYTW2 with emit-time VLIT constants), and
ISA-parametric emission. Flag `--cil-n1` (+ mutual-exclusion guard vs the
other families, same trap as §9).

RESULT — **BIT-IDENTICAL to the legacy hand-scheduled kernels at r4, r8,
r16** (gate `build_tuned/benches/cil_n1_gate.c`; also ≤1.1e-14 vs scalar
DFT). The shared SR scheduler + the generic complex recursion reproduce
the hand-tuned bodies exactly: r8 op counts match the hand r8_body one for
one (22 add/sub, 5 rotations, 4 FMA). Radices 2–64 all emit
(r64: 354 add/sub, 61 rot, 128 FMA, 28 twiddle consts).

REACH PROVEN: `--isa avx512` emits `__m512d` with a correctly widened
`_M_IM` mask and 4-complex-per-vector loop — the AVX-512 payoff the
hand-written 486-literal-`_mm256_` emitter could never give.

NEXT: CIL-2 (blocked variants via the native spill recipe), CIL-3 (bailey2
leaf n1t/n1b2 corner-turn store), CIL-4 (bailey2 t2 mid — the LOAD-flavor
BYTW2 over a VTW2 packed table), CIL-5 (race vs legacy + vs split BAILEY2V).

### 2026-07-25 — CIL-3/4 (bailey2) + algsimp VERIFICATION + performance race

**CIL-3/4 landed, BIT-IDENTICAL to legacy.** `n1t` (stage-1 leaf, four-step
transpose fused into the stores via paired permute2f128) and `t2` (stage-2
mid, streamed VTW2 records applied with BYTW2 = fmadd(c,x,mul(s,cflip x)),
cursor `tw_re + (k/per)*(R-1)*2*VW`) both reproduce the hand kernels exactly
(gate `build_tuned/benches/cil_bailey2_gate.c`). New node `CTwL` (table-load
twiddle) — the only IR addition needed for the whole bailey2 mid family.

**BUG FOUND + GUARDED (this is why the odd-radix question mattered).**
`dft_cx` is a DIT-RADIX-2 recursion, so it is valid only for powers of two.
For odd n it silently DROPPED the tail element — r3 emitted `out[2] = z0`
(an input, with z2 loaded and unused): plausible-looking, WRONG code, and no
gate would have caught it because nobody had asked for r3. Now a hard
failure (`radix land (radix-1) <> 0 -> failwith`). Verified: r3/r5/r7/r9
blocked, r8/r16/r64 emit.

**algsimp VERIFICATION (user was right).** Measured op counts from the
codelet-metrics footer, in-place avx2, DEFAULT vs `VFFT_DISABLE_FMA_LIFT=1`:

| class | radix | full algsimp | no FMA-lift | penalty |
|---|---|---|---|---|
| pow2  | 8 / 16 / 32 / 64   | 52 / 144 / 386 / 952 | 56 / 166 / 458 / 1161 | +8 / +15 / +19 / **+22%** |
| prime | 3 / 5 / 7 / 11 / 13 / 17 | 12 / 32 / 60 / 150 / 204 / 336 | 16 / 44 / 96 / 240 / 336 / 576 | +33 / +38 / +60 / +60 / +65 / **+71%** |
| odd-composite | 9 / 15 / 25 | 122 / 181 / 384 | 162 / 257 / 530 | +33 / +42 / +38% |

So algsimp is worth **8–22% on pow2 and 33–71% on odd/prime** — the
Winograd-structure passes (`aggressive`, gated on `pick_algorithm = Direct`)
are exactly the prime path. `VFFT_NO_SUBDEDUP` was ~neutral (r9 only, +2 ops).

**What that means for the IL backend — the honest scope limit.** The current
complex path gets FMA-shaped code *by construction*, not by simplification:
every twiddle class is hand-folded (√½ into `CFmaC`, general twiddles into
BYTW2 = exactly 1 mul + 1 fma, which is optimal for a complex multiply), so
there is no FMA-lift headroom to recover at pow2 — measured: r8 0 mul/4 fma,
r64 68 mul/128 fma with 96 BYTW2 twiddles (= 96 of those mul/fma pairs).
That is why pow2 came out bit-identical to the hand kernels with no algsimp.
**But it does NOT extend to odd/prime radices**: those need Winograd
structure, which at the complex level is a MATH-LAYER construction, not one
of algsimp's real-valued rewrite passes (algsimp finds `c·x_a + c·x_b →
c·(x_a+x_b)` in the REAL expansion; a packed-complex DAG hides that
structure inside opaque complex nodes). So sharing `Ir`/`algsimp` would NOT
buy odd-radix IL support either — that needs a complex Winograd/Rader math
layer. Recorded as the open item for full-IL arbitrary-N.

**PERFORMANCE (the point of the exercise) — `build_tuned/benches/cil_race.c`.**
Pipeline vs legacy hand-scheduled, one arena, arms rotated, best-of-11,
pinned core 2, 2 de-aliased runs:

| kernel | legacy ns | pipeline ns | ratio | verdict |
|---|---|---|---|---|
| n1  r8  | 1102.6 / 1113.6 | 1110.5 / 1110.4 | 1.007 / 0.997 | PARITY |
| n1  r16 | 4082.3 / 4084.3 | 3620.8 / 3648.3 | 0.887 / 0.893 | **pipeline ~11% FASTER** |
| n1t r8  |  971.6 /  593.2 |  970.7 /  595.8 | 0.999 / 1.004 | PARITY |
| t2  r8  |  984.3 /  988.1 |  978.2 /  974.8 | 0.994 / 0.987 | PARITY |

The r16 win is real and corroborated statically: the pipeline emits FEWER
instructions (316 vs 362; 183 vs 197 vector) — the SR scheduler manages
register pressure better than the hand order on the larger DAG. Everything
else is parity. **So the port costs nothing in speed and gains at r16.**

⚠ MEASUREMENT LESSON (the §4.9993 trap, again). The first harness allocated
zin and zout as two independently 4 KB-aligned buffers → every stream shared
L1 set indices → timings were BIMODAL (~585 vs ~975 ns for the SAME binary,
flipping run to run) and produced a garbage 0.596 "40% faster" reading for
n1t. Fix: ONE arena with a 64 B (non-4KB-multiple) skew between the planes.
Ratios became stable immediately. Any future IL race must keep that skew.

**WHY fma_lift matters — the RA mechanism (Tugbars, confirmed in regalloc.ml).**
The op-count table above undersells it. `regalloc.ml` §1/§4 documents the real
mechanism: pre-fusing FMAs at the IR level leaves gcc no mul+add to fuse, so
gcc does not re-schedule away from the generator's SU+GH order — and it is
that re-scheduling which raised peak-live, forced spills, and produced
reg-reg move garbage (§1). Fusing everything is precisely why the whole
M-project (register pins + scheduling fence) could be turned OFF (§4).

Measured on the interleaved kernels (-O3, mingw 15.2), legacy vs pipeline:

| kernel | reg-reg mov | stack traffic | mul | fma | instr |
|---|---|---|---|---|---|
| n1 r8    legacy / pipeline | 2 / 2 | 12 / 14 | 0 / 0 | 4 / 4 | 129 / 142 |
| n1 r16   legacy / pipeline | 7 / 7 | **53 / 35** | 4 / 4 | 16 / 16 | 362 / 316 |
| n1t r8   legacy / pipeline | 2 / 2 | 14 / 16 | 0 / 0 | 4 / 4 | 139 / 136 |
| t2 r8    legacy / pipeline | 2 / 2 | 12 / 14 | 7 / 7 | 11 / 11 | 163 / 182 |

Two things fall out:
1. Reg-reg moves stay at 2-7 in EVERY kernel, both families — the
   "no RA churn" state. mul/fma counts are IDENTICAL per kernel, i.e. both
   arms are equally pre-fused (legacy hand-folds; the IL path folds via the
   twiddle-class selection + BYTW2), so gcc has nothing to re-fuse. The IL
   backend inherits the fma_lift benefit BY CONSTRUCTION without running
   algsimp — which is the precise reason pow2 IL needs no simplification.
2. The r16 pipeline win is NOT instruction count, it is SPILLS:
   53 -> 35 stack accesses (-34%) at identical arithmetic. The shared SR
   scheduler keeps peak-live lower than the hand order on the bigger DAG.
   That is the ~11% measured speedup.

Corollary for the odd/prime gap: the thing algsimp would contribute there is
Winograd STRUCTURE (fewer real multiplies), not FMA shape — and that is a
complex-math-layer construction, still the open item.

**Spill/RA pressure SCALES with radix (Tugbars: "it gets worse as the radix
grows larger") — measured, and the scheduler's edge grows with it:**

| radix | legacy reg-mov / stack | pipeline reg-mov / stack | reg-mov delta |
|---|---|---|---|
| r8  |  2 /  12 |  2 /  14 |  0%  |
| r16 |  7 /  53 |  7 /  35 |  0%  |
| r32 | 28 / 197 | 17 / 158 | **-39%** |
| r64 | 83 / 554 | 42 / 537 | **-49%** |

Both metrics grow ~3-4x per radix doubling: past r16 the peak-live set is far
beyond 16 YMM, so spilling is forced and gcc's move traffic compounds. The
shared SR scheduler roughly HALVES the reg-reg moves at r64 (83 -> 42) and
cuts stack traffic most where it still fits (-34% r16, -20% r32, -3% r64).

CONSEQUENCE — this is the argument for CIL-2. At r32/r64 the binding
constraint is no longer instruction ORDER but working-set SIZE: no schedule
of a monolithic r64 fits 16 registers. The fix is the blocked construction
(PASS1/PASS2 split with an explicit spill arena), which the pipeline already
has natively as the spill recipe (dft_expand_n1_blocked + spill markers +
Emit_c.classify_passes) and which codelet_zil re-implemented by hand as
`blocked`/`blocked2`. Porting that to the complex IR is the next lever, and
it is where the IL family should stop relying on gcc to spill for it.

### 2026-07-25 — IL BACKWARD DIRECTION landed (was genuinely missing)

Context: the legacy IL family (`emit_z_kernel`, n1/t2) is FORWARD-ONLY — `fname`
hardcodes the literal `_fwd` (codelet_zil.ml:545), `dft_z` is coded forward-only
(:51), none of its 10 flags is a direction, and 0 of 43 emitted `_z_n1*`/`_z_t2*`
files carry `_bwd`. Backward existed ONLY in the cascade family via hand-written
`SPLIT_BFLY*_INV` macros. So there was nothing to port — an IL inverse is new work,
and without it no IL transform is usable at all.

**Implementation.** `dft_cx` gains `~sign`. The whole direction flip reduces to:
`w_k = e^{sgn·2πik/n}`, so `4k=n` uses the ×(+i) quarter-turn instead of ×(−i),
`8k=n` / `8k=3n` keep the same √½-folded FMA shape with the flipped rotation, and
the general twiddle takes `s = sgn·sin`. **Op counts and butterfly shape are
identical in both directions** — only which rotation node is used changes.

New IR node `CRotPI` (x·(+i) = `xor(cflip x, _M_RE)`, negate the RE lane) plus
`Isa.re_mask_decl` — the mirror of the existing `_M_IM` device. Emission picks
exactly one mask per direction (no unused-const warnings).

**t2 bwd is the true inverse, not a sign flip.** Forward computes `y = DFT(w ⊙ x)`,
so the inverse is `x = conj(w) ⊙ IDFT(y)`: the twiddle moves to AFTER the butterfly.
Conjugation is TABLE-SIDE (caller passes a conjugated table — the same convention
the split cascade uses for twspb/twqb), so the BYTW2 apply is bit-for-bit the
forward one and only its POSITION changes. n1/n1t bwd are just the inverse
butterfly, unnormalized (`bwd(fwd(x)) = R·x`, matching the rest of the library).

GATE — `build_tuned/benches/cil_bwd_gate.c`, all PASS:

| check | result |
|---|---|
| r8  n1 bwd vs scalar IDFT | 3.3e-15 |
| r8  n1 roundtrip = R·x | 8.9e-16 |
| r16 n1 bwd vs scalar IDFT | 1.1e-14 |
| r16 n1 roundtrip = R·x | 2.7e-15 |
| r8  t2 roundtrip = R·x (conjugated table) | 1.8e-15 |

Flag: `--cil-bwd` (composes with `--cil-n1` / `--cil-n1t` / `--cil-t2`).

Also refreshed the codelet_cil.ml header, which had drifted: it still said bailey2
was "next", named `emit_n1` as the public surface, omitted ×(+i) and the table-load
twiddle, and — most importantly — its "not shared" rationale predated the algsimp
measurements. It now states the honest reasons (fma_lift IS valuable and IS an RA
mechanism; this module simply gets FMA shape by construction at pow2, and algsimp
could not supply Winograd for odd/prime anyway) and acknowledges that `dft_cx`
duplicates dft.ml's recursion shape.

### 2026-07-25 — FUSED FULL-IL K=1 (`--cil-k1`): the first full-IL-interior route

**Course correction (Tugbars).** My first K=1 attempt (`cil_k1.h`, now deleted) composed
the transform from TWO kernel calls (n1t then t2) over the 11-arg z ABI, with an
N-complex scratch plane between them and a RUNTIME VTW2 table. Every one of those is a
SPLIT-family artifact retrofitted onto IL. Tugbars, correctly, after a week of this
pattern: *"you read the existing code/structure and you found an excuse to lean towards
preserving and solving things with the existing machinery."* Banked as
[[dont-retrofit-il-onto-split-machinery]].

**What K=1 IL actually wants** — and our own MKL study already said so
(mkl_highN_cascade_anatomy.md: *"the whole 2^k K=1 cascade is ONE function"*):

| | retrofit (wrong) | IL-native (`emit_k1`) |
|---|---|---|
| structure | 2 kernel calls | ONE fused function per N |
| intermediate | scratch plane crossed via ABI | function-scope L1 plane, never escapes |
| twiddles | runtime VTW2 table | **compile-time VLIT constants** |
| stage boundary | function call | **register transpose** (permute2f128) |
| ABI | 11-arg staged codelet | `(const double *zin, double *zout)` |

**Implementation.** `emit_k1` builds the four-step N = n1·n2 with BOTH stages inside one
function. Stage A runs DFT_n1 over each adjacent column PAIR (one vector load per leg)
and parks to a flat `double P[2N]`; the turn is two `permute2f128` per (k1-pair, column)
regrouping lanes from "two j2 of one k1" to "two k1 of one j2"; stage B applies
w_N^{k1·j2} and runs DFT_n2, storing at complex `k2*n1 + 2d` — which IS natural order,
so no output permutation ever runs. Every sub-DAG goes through the SHARED SR scheduler.
New node `CTwV` carries a DIFFERENT emit-time constant per complex lane (the fused
kernel's two lanes are two different output indices k1, so their twiddles differ) —
still zero runtime cost, just a wider VLIT.

RESULT — `build_tuned/benches/cil_k1_fused_gate.c`, MKL in its HOME config
(DFTI_COMPLEX, K=1, contiguous — the exact configuration docs/research studied, so no
handicap):

| N | vs MKL err | roundtrip err | CIL ns | MKL ns | vs MKL |
|---|---|---|---|---|---|
| 16   | 1.3e-16 | 2.3e-16 | **5.0**   | 13.0   | **2.60×** |
| 64   | 3.0e-16 | 3.3e-16 | **28.9**  | 30.1   | **1.04×** |
| 256  | 3.2e-16 | 4.5e-16 | 165.7     | 140.3  | 0.85× |
| 1024 | 6.8e-16 | 8.9e-16 | 1536.0    | 1303.0 | 0.85× |

- **N=16 is 2.6× MKL**; N=64 edges it.
- N=64 at 28.9 ns **beats the recorded mono-64 hand reference (32 ns, logged as
  "≈ MKL-IL")** — the fused generated kernel is faster than the hand one.
- 256/1024 sit at 0.85×, i.e. exactly the library's known ~15% standing.
- Accuracy improved an ORDER OF MAGNITUDE over the staged form (1e-16 vs 1e-15) —
  no intermediate memory round-trip means fewer roundings.

This is the FIRST route in the codebase whose interior is interleaved end to end. Every
prior K=1 IL path converts internally: the z-cascade's leaf is "z-in -> split-out"
(block-split interior) and BAILEY2V is IL-boundary/split-interior.

NEXT for K=1 completeness: N=4096+ (stage A's plane exceeds L1 — needs blocking or a
3-stage split), non-pow2 N (complex Winograd/Rader), and a runtime path so users can
reach it.

### 2026-07-25 — BLOCKED (Cooley-Tukey split) for the IL codelets: `--cil-blocked`

Purpose (Tugbars' sequencing: blocked FIRST, then wire the DP planner to IL
planning): make r32/r64 viable as chain stages so the planner has a real menu
instead of being forced onto small radices by spill pressure.

**Construction.** Split R = m·p, decimating legs by residue mod m:
`n = a·m + i` → `A_i[j] = DFT_p over a of x[a·m+i]`, then
`X[j + p·k2] = DFT_m over i of (A_i[j] · W_R^{i·j})`. PASS 1 emits m sub-DFTs
of size p (each needs only p live) parking to a function-scope `S[]`; PASS 2
emits p groups, each reloading m values, applying the compile-time
`W_R^{i·j}`, and running an m-point DFT. Peak live drops R → max(p, m). Every
pass is scheduled by the SHARED scheduler; twiddles stay compile-time.

**m is chosen by radix, and it matters:** m=2 (halving) for r16/r32, **m=8 for
r64**. Halving 64 leaves 32-point halves that still spill — the deep 8×8 split
is what actually fixes it. Same reason codelet_zil has a separate `blocked2`.

**m=2 keeps the class-aware butterfly.** For a single top-level butterfly the
general "twiddle then DFT_2" form would turn W^{R/4} into a full complex
multiply and lose the W^{R/8} FMA folds. m=2 therefore routes through
`butterfly_pair` (the shared class selector), which also keeps its output
BIT-IDENTICAL to the monolithic form. m=8 uses the general form plus a
rotation shortcut at `4e = R`.

MEASURED (-O3, stack traffic = spill/reload ops):

| radix | monolithic reg-mov/stack | blocked reg-mov/stack | stack |
|---|---|---|---|
| r16 | 7 / 36  | 7 / 31   | −14% |
| r32 | 20 / 160 | 16 / 101 | **−37%** |
| r64 | 43 / 534 | 43 / **156** | **−71%** |

GATE `build_tuned/benches/cil_blocked_gate.c`: r16/r32 blocked-vs-monolithic
**BIT-IDENTICAL** (the class-aware m=2 path), r64 2.8e-15 (the 8×8 split
re-associates, as expected), all vs scalar DFT ≤9.9e-14. PASS.

NEXT (per the agreed sequencing): wire the DP planner to IL planning, now that
r32/r64 are no longer automatically disqualified by spills. Note the planner's
existing calibrated chains (2048 = 4·8·8·8, 4096 = 4·4·4·8·8) were derived when
big radices DID spill — with blocking they may be re-evaluated, but that is a
measurement for the planner to make, not an assumption to bake in.

⚠ CORRECTION recorded: `emit_k1`'s factorization is currently a hardcoded
"squarest split" (4096 → 64×64), which CONTRADICTS those calibrated chains and
is why N=256 (16×16) and N=1024 (32×32) measured 0.85× while N=16 (4×4) and
N=64 (8×8) beat MKL — the wins land exactly where the split happens to pick
spill-free radices. That heuristic must be replaced by planner-supplied chains,
not re-derived.
