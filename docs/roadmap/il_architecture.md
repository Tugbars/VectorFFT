# Interleaved-complex support — architecture & design (as built)

*docs/roadmap/il_architecture.md — 2026-07-14. Companion to interleaved_design.md
(the pre-implementation design study with the 10-pitfall table). This document
describes what P1a actually shipped, the contracts it froze, every interaction with
the rest of the engine, the measured economics, and the exact mechanics of the
removal path. All numbers: container host, same-process ratios.*

---

## 1. The two orthogonal problems

"Interleaved support" conflates two independent problems; separating them was the
design's first act and everything else follows from it.

**Component layout** — (re,im) adjacent pairs `z[2f], z[2f+1]` vs the engine's split
planes. This is a *local, position-preserving* permutation: element f maps to pair
(2f, 2f+1) and nothing moves anywhere else. Cheap, SIMD-friendly (a fixed 4-shuffle
lattice), and fusible into any pass that already touches the bytes.

**Batch geometry** — transform-major batched 1D (`z[k·2N + 2i]`, the FFTW/MKL idist
convention) vs our lane-major batching. This is a *global corner turn*: a transpose
across the batch dimension, cache-hostile, its own algorithm (blocked Bailey-style).

P1 addresses only the first. The second is P3, and the API **refuses rather than
fudges** it: the wrapper's header states the geometries it serves (multi-dim, 1D K=1,
lane-major-interleaved batches). Silently accepting transform-major input and
producing garbage was pitfall #1 in the design table; the defense is documentation
plus the absence of any code path that pretends.

## 2. Why IL is a boundary property, never an internal one

The lane-batched split-complex engine is load-bearing in three ways:

1. **The accuracy result depends on it.** FMA-as-IR twiddle math operates on
   separate re/im operands; the 1.8–5.3 eps MKL-parity numbers are a property of
   that arithmetic shape.
2. **The register economics assume it.** Zero-spill CT blocking, the AVX-512 ILP
   pairing, the strided in-register transposes — all count registers as one plane
   each. Interleaved arithmetic (addsub/movddup complex kernels) halves the
   effective register file and forfeits the codelet tree.
3. **The planner/wisdom corpus measures it.** Every calibrated verdict is a verdict
   about split-plane kernels.

Therefore: **interleaved exists only at the user-memory frontier; everything inside
is split.** The entire architecture question reduces to *where is the frontier
crossed, and can the crossing ride memory movement that already exists?*

## 3. The P1a component stack (shipped, gated)

Three layers, deliberately thin:

### 3a. Conversion kernels — the vocabulary

`vfft_il2sp(z, re, im, n)` and `vfft_sp2il(re, im, z, n)` in
`src/core/transforms/conv/il_layout.h`. Four complex per iteration:

```
il2sp:  2 loads (r0 i0 r1 i1 | r2 i2 r3 i3)
        unpacklo → (r0 r2 r1 r3), unpackhi → (i0 i2 i1 i3)
        permute4x64 0xD8 fixes the cross-lane order → (r0 r1 r2 r3), (i0 i1 i2 i3)
        2 stores
sp2il:  the exact mirror.
```

Properties that matter: **exact** (pure moves, zero arithmetic — every bit-level
gate in the library survives conversion unchanged), scalar tails for n % 4,
load/store-bound in practice (the shuffles hide). They are unfused and simple on
purpose: vocabulary, not sentence.

### 3b. The universal wrapper — the stable contract

`stride_il_t` wraps **any** plan: 1D lane-batched, the fft2d/fft3d/fftnd override
wraps, r2c — because all of them expose the same split `(re, im)` execution surface.
One mechanism covers every fft type; there is no per-family IL code to keep
consistent.

```
stride_il_wrap(plan, take_ownership)  → owns a split cube (2·n doubles)
stride_il_fwd(w, z):  il2sp(z→cube); execute_fwd(cube); sp2il(cube→z)
stride_il_bwd(w, z):  mirrored
```

The API is the frozen contract; the two sweeps are an implementation stage with a
published expiry (§6). Callers written against `stride_il_fwd/bwd` today are exactly
the callers of the fully-fused version later.

### 3c. Free riders — the north star generalized

Wherever the library already copies every element at a boundary, layout is free.
r2c's pack/unpack staging (rows copied to/from the padded K_pad cube) became
`stride_plan_nd_r2c_il` by swapping `memcpy` for `sp2il/il2sp` per row: interleaved
complex output at **zero marginal cost**, spectrum pairs bit-exact vs the split
plan. Design rule extracted: *never move data twice for layout when you're already
moving it once for anything else.* Every future boundary (P1b's tiled pass, P2's
stage-0 loads) is an application of this rule.

## 4. Interaction matrix

| subsystem | interaction | status |
|---|---|---|
| order (scrambled / natural / strided-natural) | conversion is position-preserving ⇒ the wrapped plan's order appears identically in pairs; the strided/natural contract is untouched | bit-gated |
| tails (padded strided, rem-aware in-place, r2c route-arounds) | live entirely inside the wrapped plan; sweeps have their own scalar tails | IL suite runs with strided ON; odd/prime cells pass |
| MT | plan execution MT unchanged (T∈{1,4} bit-identical through the wrapper); sweeps single-threaded today — streaming-bound, MT-sweep is a trivial option if a host ever shows headroom | bit-gated |
| JIT | wrapper calls `stride_execute_*` ⇒ internal JIT resolves apply as usual; JIT proven rem-aware (bit-exact at odd K) so no gating needed | probed |
| conv / spectrum consumers | **stay split by design** — converting for a pointwise consumer is a round trip for nothing. If a user pipeline ever demands IL spectra in conv, the design is: interleaved kernel spectrum + an IL×IL pointwise kernel (4 shuffles + split math per vector). Deferred, documented | deferred |
| natorder maps / gather | operate on split spectra; an IL consumer converts once at the end (or uses r2c IL-out where the gather is the boundary copy) | as-is |

## 5. Measured economics — why fusion is the feature, not the polish

```
64³, container, strided-rows era:
  split transform        2,146,382 cyc   (near the memory floor)
  IL wrapper             4,149,731 cyc
  P1a tax                1.933×
```

Mechanism: two sweeps per direction move 4·(8 MB) ≈ 32 MB of extra traffic — a
whole transform's worth, *because* the strided-row work pushed the split path to the
floor. The historical irony is the argument: pre-strided the same wrapper taxed
~1.4×; our own optimization doubled the relative cost of unfused conversion.
Conclusion frozen into the roadmap: **boundary fusion is not an optimization of IL
support — it is IL support.** P1a is the correctness vehicle, the API freeze, and
the measured ceiling that P1b/P2 are judged against.

## 6. The removal path — exact mechanics

**P1b — fuse the tiled-side boundary (halves the tax).** Pass order is axes
0..r−2 then the tiled row pass (fwd), mirrored (bwd). So the tiled pass is the LAST
writer of user memory going forward and the FIRST reader coming backward — the
natural fusion site:

- fwd: axes run on the split cube as today; the tiled pass reads the cube and
  **il-scatters** directly to user z — the existing scratch→rows transpose gains
  interleaved stores (the same 4-shuffle lattice applied to data already in
  registers). One explicit sweep remains (input side: il2sp into the cube, because
  axis-0 needs split input).
- bwd: **il-gather** (tiled first) + one sp2il sweep out.
- Strided-rows variant: the strided codelet's in-register transpose gains an
  IL-store flavor — or interim, strided writes split and only non-strided cells
  fuse; per-cell calibrator decision.
- Tail rows: the **padded-staging pattern from the tails work is exactly the
  required shape** — stage rem rows split, convert on the copy-back.

**P2 — emitter stage-0 flags (removes the last sweep).** The generator gains
IL-load (fwd stage 0) and IL-store (bwd final stage) modes for the n1 and strided
families: deinterleave fused into loads the codelet performs anyway. The emission
seam already exists — `render_load`/`emit_store` are mode-aware (that machinery is
what carries the masked tails); IL is one more load/store mode through the same
seam, not new infrastructure. After P2: fwd axis-0 reads user z directly; **zero
sweeps, zero cube** for the c2c path.

**P3 — the corner turn.** Transform-major batched 1D: blocked transpose fused with
the first/last stage, Bailey-style. Independent of P1/P2; shares only the
conversion vocabulary.

**P4 — calibrator.** `layout=` joins `s`/`blk`/`rows`/`T` in the wisdom key. Some
cells may keep the through-cube route if a fused variant costs codelet ILP — a
measured per-cell verdict, like everything else here.

## 6a2. Does P2 need new codelets? — the scoping result (measured)

Three-layer answer, each grounded:

**Runtime reuse of existing codelets: impossible.** The vector load
`loadu(re + leg*ios + k)` bakes lane-contiguity into the instruction; re-basing to
`(z, z+1)` with doubled strides would require stride-2 vector loads that do not
exist. Scalar codelets could take a stride-2 parameter — and forfeit SIMD.

**The zero-new-codelets middle: BOUNCE-TILE.** Convert one lane-block into
cache-resident scratch, run the UNCHANGED split codelet as a slice on it, convert
back. DRAM traffic is identical to fused P2 (each user byte read once, written
once); the cost is a cache-tier round-trip. Measured on the axis-0-shaped boundary
pass (N=64, K=4096, DRAM-ish):

```
 split floor            1,526,450 cyc      1.00×
 P1a full sweeps        3,782,793          2.478×
 bounce-tile (C=64)     2,216,072          1.452×   ← zero new codelets
 true P2 (projected)         —            ~1.1×    (stage-0 shuffle μops only,
                                                    §6b micro: 1.12–1.16×)
```

Bounce cuts the sweep tax nearly in half today, is C-insensitive (64/128/256 within
noise), and per-cell may simply be kept forever where the residual doesn't matter —
the calibrator arbitrates, as always.

**UPDATE — P2 MEASURED (derived codelets, whole 64³ zero-sweep).** The derivation
route (tools/il_derive.py, §6a3 below) produced working boundary codelets and the
full interleaved-in → interleaved-out 64³ transform with no conversion sweep
anywhere. Bit-exact fwd vs the (il2sp + split 3-pass + sp2il) reference; roundtrip
1.3–2.0e-15. Same-process container ratios, mono-r64 axis-0 structure both arms:

```
                       AVX-512                AVX2
 split 3-pass floor    3,565,543   1.000×     5,240,304   1.000×
 P1a full sweeps       5,385,984   1.511×     6,661,115   1.271×
 ZERO-SWEEP (P2)       4,360,708   1.223×     5,696,570   1.087×
```

AVX2 lands on the §6b projection almost exactly. The AVX-512 residual is larger
because the per-line derivation DOUBLES the rows-boundary store count (2 masked
stores per original store); folding the interleave into the strided transpose
network — which only the emitter can do — is the remaining delta to ~1.1× there.

## 6a3. The derivation route (shipped)

`tools/il_derive.py` mechanically transforms EMITTED split codelets into IL
variants: every load/store of the target re/im pointers is replaced per-line by a
shuffle lattice against the pair pointer, at all four widths (512/256/128/scalar),
including the avx512 masked tail (runtime kmask pair-doubled via `pdep`, hence
-mbmi2). Per-line means scope-proof — no assumption that re and im accesses share
a basic block (they don't, in either the oop tail or the strided transpose).
Arithmetic lines are untouched, so accuracy and the wisdom corpus carry over
unchanged. Derived set (codelets/il/): r64 oop n1 il_in + il_out_sw (bwd via the
swap trick — fwd math, (im,re)-flipped stores, caller swaps input pointers), and
r64 strided il_out (fwd rows) + il_in (bwd rows), both ISAs; externs in
vfft_il_codelets.h; gates in build_tuned/benches/test_il_codelets.c; the
zero-sweep assembly in bench_zero_sweep_64c.c. The derivation is the bridge: it
proves the lattices and the contracts; the emitter mode (render_load/emit_store
seam) inherits them and adds the folding.

**True P2 emission: yes — but the scope collapses to ONE family.** Under DIT chains
(which exhaustive planning selects essentially everywhere), the forward pass's only
reader of user memory is stage-0 = the **n1** codelet, and the backward pass's only
writer is its mirror, **n1_bwd** scheduled last. Interior stages never touch user
memory. The emission set is therefore `n1_fwd_il` + `n1_bwd_il` per (radix × ISA) —
roughly 30–40 files — expressed as one more load/store mode through the EXISTING
mode-aware `render_load`/`emit_store` seam (the machinery that already carries the
masked tails). DIF plans' boundary stage is a twiddle stage and stays out of scope
initially: the IL-mode planner prefers DIT, and any stubborn DIF cell uses bounce.
The strided family needs no interior change at all — one extra unpack layer in its
existing transpose network (P1b, transpose kernels).

## 6a4. Chained boundaries: the wrong-donor lesson and the shipped adapters

Mono-r64 stage-0 (§6a2's bench shape) measures **1.340× (AVX-512) / 1.615×
(AVX2)** of the exhaustive chain at the axis-0 shape (64, K=4096) — the v1
zero-sweep ratios were divided by a mono-inflated floor. Production IL must
attach to the chain's own stage 0.

**Engine anatomy makes both boundaries n1.** DIT forward runs stage 0 first
(untwiddled in every group); the backward generic executor runs stages in
*reverse*, so stage 0's `n1_bwd` writes user memory last. No t1 derivation is
needed for either direction. `vfft_proto_execute_bwd_generic` was refactored to
`_until(plan, re, im, K, until_stage)` (the reverse-order mirror of the forward
`_from`), with the old name kept as a thin `until=0` wrapper.

**Wrong donor, measured.** The first chained derivation used the UG_UG oop
family. Five-way decomposition at the axis-0 shape (AVX-512): engine full chain
1.60M cyc; engine stage-0 ≈ 1.19M; UG_UG split stage-0 ×4 groups **3.56M (3×)** —
regalloc is not wired on the oop path (render-convention blocker, §36). The
lattice itself was *negative* tax: il_in ran 381K cyc **faster** than its own
split parent (interleaved pair loads are friendlier than two split streams
here). Re-derived from `codelets/inplace/rR_n1_{fwd,bwd}.c` — the emission the
registry's 7-arg wrappers actually call — for R ∈ {2,4,8,16,32,64}, both ISAs
(driver: `tools/il_derive_inplace.py`). The superseded small-radix UG_UG il
files were removed; the r64 pair stays for the §8 gates and the v1 bench.

**Adapters (`src/core/oop/il_execute.h`).** `vfft_proto_execute_fwd_ilin(plan,
z, dst_re, dst_im, K)`: per-group stage-0 il_in (z→split) then
`fwd_generic_from(…, 1)`. `vfft_proto_execute_bwd_ilout(plan, src_re, src_im,
z, K)`: `bwd_generic_until(…, 1)` in place, then per-group stage-0 il_out
(split→z). Eligibility mirrors oop_execute.h (DIT, no override, stage-0
untwiddled, radix in table); −1 falls back to il2sp/sp2il + normal execute.
Output aliasing `z == zo` is legal both directions (axis-0 fully consumes z
before rows writes it; rows fully consumes z before axis-0 writes it) and gated
bit-exact.

**Gates (64³, both ISAs).** fwd and bwd BIT-EXACT vs il2sp + engine + sp2il
(donors are the engine emission; the generic resume is bit-identical to the
baked path). Roundtrip 1.3–1.4e-15. z-aliasing BIT-EXACT.

**Per-pass taxes (isolated, min-of-7 — the transferable numbers):**

| pass | AVX-512 | AVX2 |
|---|---|---|
| axis-0: ilin vs engine chained | +430K, 1.260× | +455K, 1.227× |
| rows: il_out vs strided split | +739K, 1.537× | +781K, 1.456× |
| sum of parts vs floor | 1.281× | 1.218× |

**Whole-transform (64³, container, two consecutive runs):**

| arm | AVX-512 | AVX2 |
|---|---|---|
| chained split floor | 1.000× | 1.000× |
| P1a sweeps | 1.585× / 1.633× | 1.643× / 1.594× |
| zero-sweep fwd (zo separate) | **1.209× / 1.204×** | **1.325× / 1.290×** |
| zero-sweep fwd (z in-place) | 1.281× / 1.302× | 1.456× / 1.391× |
| zero-sweep bwd (zo separate) | 1.233× / 1.248× | 1.264× / 1.199× |
| zero-sweep bwd (z in-place) | **1.178× / 1.192×** | **1.147× / 1.114×** |

One earlier run showed a 1.66× fwd outlier arm — the 1-vCPU container swings
individual arm windows by ~30%; the per-pass isolation table is the
measurement of record. The fwd-prefers-separate / bwd-prefers-in-place split
reproduces across runs but is unexplained; both are bit-exact, so the API
supports either.

**Remaining tax anatomy, ranked:** (1) rows il_out doubled masked stores —
+0.74/0.78M, the single biggest item; folds away when the emitter interleaves
inside the existing back-transpose shuffles (`ls_mode`, §6a3's endpoint).
(2) generic-resume dispatch ~143K/axis — closes by passing the plan's JIT
executor with `start_stage=1` (the documented 5–6% tier-1 gap). (3) axis-0
lattice residual — partially inherent (deinterleave is real work), partially
per-line load doubling the emitter also folds.

## 6a5. The emitter fold (ls_mode) — shipped, measured

The rows il_out store doubling (section 6a4's top-ranked tax) is now folded in
the generator itself. `emit_codelet`'s strided postamble gained an
interleaved-output mode: flags `--strided-il-out` (regular stores) and
`--strided-il-out-nt` (non-temporal), refs `Emit_state.strided_il_out` /
`strided_ilo_nt`, symbols `radix{R}_n1_fwd_{isa}_strided_il_out[_nt]`.
Signature: `(const rio_re, const rio_im, double *out_z, tw, tw, row_stride,
me)`. `--strided-il-in` is parsed but fails at emit time (load-side lattice
lands next).

**Design.** The inverse-transpose postamble runs both sides' stage-1/2 in one
group block, then pairs rows at the store stage. AVX2: per 4-wide group the
interleave costs **+8 unpacks over the native split path** with the *same*
plain-store count (8) and zero masked ops (`_p{k}_lo/hi = unpack{lo,hi}(_u{k}_re,
_u{k}_im)`, rows via `permute2f128` 0x20/0x31). AVX-512: rows materialize as
`_r{i} = shuffle_f64x2(_v·,_v·)` pairs, then two `permutex2var` against
function-scope `_il_idx_e/_il_idx_o` — +16 permutex2var per group, same store
count, one full cache line per store. Port-5 delta ≈ 65K cyc over the whole
64³ rows pass — the measured savings are store-μop and masking removal.

**Gates.** Byte-identical plain-strided regeneration (default flags reproduce
the shipped schedule exactly — regression guard). il_out emitted and NT are
BIT-EXACT vs the derived per-line family at (rs,me) ∈ {(64,64),(96,16),(64,8)},
both ISAs, and BIT-EXACT as whole-transform drop-ins in the 64³ zero-sweep.

**Isolated rows pass (64 slabs, same window):**

| arm | AVX-512 | AVX2 |
|---|---|---|
| split native (in-place) | 740K, 1.000× | 853K, 1.000× |
| il_out derived (per-line) | 1.835× (+618K) | 1.709× (+605K) |
| il_out emitted | 1.613× (+454K) | 1.551× (+470K) |
| il_out emitted NT | 1.559× (+414K) | 1.429× (+366K) |

The residual vs in-place split is dominated by inherent OOP write traffic —
z is a different 4MB buffer; in-place stores hit just-loaded lines. The honest
production comparison is the P1a-equivalent (split rows + sp2il ≈ 1.85M),
which the emitted pass beats by ~600–700K.

**Whole-transform 64³ fwd (v6, all arms in one binary/window):**

| arm | AVX-512 | AVX2 |
|---|---|---|
| chained split floor | 1.000× | 1.000× |
| P1a sweeps | 1.785× | 1.652× |
| zero-sweep, derived il_out | 1.350× | 1.339× |
| zero-sweep, **emitted il_out** | **1.294×** | **1.253×** |
| zero-sweep, emitted NT | 1.351× | 1.282× |

NT wins the isolated pass but loses at whole-transform: repeated reps re-hit
z's M-state lines for free with regular stores, while NT streams 4MB to DRAM
every rep. Default: **regular emitted**. NT contract when opted in: out_z
64-byte aligned, `row_stride % 4 == 0` (AVX-512) / `% 2 == 0` (AVX2);
`_mm_sfence()` is emitted before return for cross-thread visibility.

**Status.** Emitted il_out replaces the derived fwd family for production rows.
The derived bwd il_in stays until the load-side emitter fold (deinterleave in
the load preamble: 2 z-loads + `permutex2var` per row pair on AVX-512 — halves
load count) — that and the JIT `start_stage=1` resume are the remaining ranked
items.

## 6a6. Per-slab fusion and the il_in load fold (shipped)

**Per-slab axis-1→rows fusion.** Loop-order change only: fwd runs `for s {
axis1(slab); rows_ilo(slab); }`, bwd runs `for s { rows_ili(slab);
axis1_bwd(slab); }` — rows consumes each 64KB slab while it is still cache-hot
from axis-1 instead of re-sweeping 4MB after eviction. Slabs are independent,
so the reordering is gated bit-exact both directions. Measured (v7, same
window): fwd 1.451→**1.245×** (AVX-512) / 1.514→**1.311×** (AVX2); bwd
1.332→**1.209×** / 1.218→**1.166×** — roughly −0.2 of floor on three of four
cells, the single largest recovery in the campaign. Lives at the composition
level (bench/fftnd executor), not in any codelet.

**Strided il_in emitter fold** (`--strided-il-in`, symbol
`radix{R}_n1_bwd_{isa}_strided_il_in`, signature `(const in_z, rio_re, rio_im,
tw, tw, row_stride, me)`). The load preamble emits 2 z-loads per row plus a
deinterleave pair — AVX-512 `permutex2var` against function-scope
`_il_idx_de/_il_idx_do`, AVX2 `unpack{lo,hi}` + `permute4x64(0xD8)` — then runs
both transpose networks in one group block. Load count equals the split-native
path (2/row) and halves the derived per-line version's (4/row). Plain-bwd
regeneration stays byte-identical (regression guard). Gates: BIT-EXACT vs
derived at (64,64), (96,16), (64,8), both ISAs, and as a whole-transform
drop-in.

**Isolated bwd rows pass (64 slabs, same window):**

| arm | AVX-512 | AVX2 |
|---|---|---|
| split native (in-place) | 1.000× | 1.000× |
| il_in derived (per-line) | 1.438× (+511K) | 1.381× (+528K) |
| il_in emitted | 1.312× (+364K) | 1.308× (+428K) |

**Production stack after items 1+4** — fwd: ilin → fused(axis1 + il_out
emitted); bwd: fused(il_in emitted + axis1) → ilout. Roundtrip 1.33e-15.
Whole-transform arm ratios remain window-sensitive on the bench container
(±0.15–0.2× of floor across runs); per-pass same-window tables above are the
measurement of record. Remaining ranked items: JIT `start_stage=1` resume
(~143K/axis) and the inplace-family boundary fold (entry loads likely already
CSE-folded by the C compiler — verify — leaving the bwd-exit store fold).

## 6a7. Baked resume and the bwd kernel-tier discovery (shipped)

**Wiring.** `il_execute.h` adapters gained `_ex(..., int use_baked)` forms
(public names = thin `use_baked=1` wrappers). When `_vfft_proto_lookup_{fwd,bwd}`
resolves, the adapter calls the baked executor with `start_stage=1`; every
STAGE macro is gated `if (start_stage <= S)`, so on the *reversed* bwd
executor the same parameter yields until(1) semantics for free. Generic
helpers remain the fallback and the A/B reference.

**Coverage.** The lookup table is (N,K,factors,variants)-exact and had no
entries for the 64³ chain. Three wisdom lines were appended to
`spike_wisdom.txt` (planner-probe-verified per ISA — the factorization is
ISA-dependent): (64,4096) [4,16] v02 for AVX-512, (64,4096) [4,4,4] v022 for
AVX2, (64,64) [8,8] v02 for both. `plan_executors.h` regenerated per the
documented procedure. Notes from the regen: the current emitter reproduces
808/820 shipped executor symbols from the reconstructed wisdom; the 12 missing
are three (N,K) shapes — (128,32)[4,32]-DIF, (128,64)[8,16], (64,64)[4,16]-DIF
— whose factorizations the current planner does not produce (they were
unreachable dead entries; restorable via three wisdom lines if ever needed).
The regenerated header also carries the current emitter's no-variant twin
entries and a brace-style reflow relative to the shipped file.

**The discovery.** fwd baked resume ≡ generic BIT-EXACT (both dispatch t1s)
and saves only the dispatch overhead (small). But the generic *bwd* path
never engaged the fused `t1s_dit_bwd` kernels — it runs plain t1 — while
`STAGE_BWD` calls the fused tier-1 variant. Baked-vs-generic bwd differs at
4.4–5.6e-16 max relative (kernel rounding, tolerance-gated) and runs the
axis-0 bwd pass at **0.625× (−1.30M cyc, AVX-512) / 0.545× (−1.93M, AVX2)**.
This engages in the floor too (`execute_bwd` lookup): the whole-transform bwd
floor dropped 21–29%.

**Whole-transform vs the improved floor (v8c, same window):**

| arm | AVX-512 | AVX2 |
|---|---|---|
| fwd zero-sweep (prod stack) | 1.304× | 1.217× |
| bwd zero-sweep, fused + emitted il_in | **1.120×** | **1.113×** |

Roundtrip through the full production stack: 1.22–1.44e-15.

## 6a8. JIT tier wiring and the regen cleanup (shipped)

**Tier architecture (per project doctrine).** JIT (`jit/jit_runtime.h`) is the
recommended user path; wisdom-baked static executors (`plan_executors.h`) are
the fallback; generic is the floor. The adapters now mirror oop_execute.h's
contract: `vfft_proto_execute_{fwd_ilin,bwd_ilout}_jit(..., vfft_proto_exec_fn
stages1)` take the caller's plan-time-resolved fn (via
`vfft_proto_plan_jit_fwd/bwd`), the public names resolve the baked lookup, and
NULL falls to generic. The JIT emission (`emit_jit.py`) expands the same
`start_stage`-gated STAGE macros as the baked header, so the resume contract
(`start_stage=1` ⇒ stages 1.., both directions) holds on all three tiers.

**Container validation of the JIT pipeline** (emit_jit.py → gcc -shared →
dlopen, persistent cache): forced-JIT executors for (64,4096)[4,4,4] and
(64,64)[8,8] are **bit-identical (0.00e+00)** to baked through the adapters,
both directions.

**Three tiers, AVX2, same window (ratio vs generic):**

| pass | baked | jit |
|---|---|---|
| axis-0 fwd resume | 0.981× | 0.981× |
| axis-0 bwd resume | **0.553×** | **0.551×** |
| P1 (64,64) ×64 fwd | 0.969× | 0.973× |

Baked and JIT are at parity here — expected, since both expand identical
macros over identical codelets and this bench TU is small enough for the
static header to compile cleanly. The JIT tier's structural value is coverage
without rebuild (cold plans never fall to generic) and per-TU -O3 compilation
when the monolithic header lives inside large production TUs. The bwd tier-1
kernel engagement (0.55×) reproduces on both tiers.

**Regen cleanup.** `spike_wisdom.txt` now regenerates `plan_executors.h` as a
**strict superset** of the shipped header: the three legacy specializations —
(128,32)[4,32]-DIF, (128,64)[8,16], (64,64)[4,16]-DIF — were restored as
wisdom lines (0 shipped symbols missing; +160 = new-emitter twins + the 6a7
chain entries), and the stale entry-count header comment was fixed.
`spike_wisdom.txt.pre-il` preserves the pre-campaign file.

## 6a9. Inplace-family boundary fold — item 2 closed (shipped)

**Entry side: already free.** Instruction-count evidence (objdump, r4 AVX-512):
the derived il_in codelet compiles to **32 vmovupd — identical to the original
inplace emission** — plus 8 permutex2var for the deinterleave. gcc CSE folds
the per-line doubled z loads completely (z is const-restrict, no intervening
stores). The fwd entry lattice was at the emitter-fold endpoint all along; no
work needed, closed on evidence.

**Exit side: pair-fusion in the deriver.** The list scheduler's sink-first
pass emits every store site as an adjacent `(rio_re[E], rio_im[E])` pair with
identical index expressions — universal across radices, main loop and masked
tail. `tools/il_derive.py` gained a lookahead pair pass: a fused pair issues
**one full store per z vector** (2 `permutex2var` against
`_vfft_il_pair_lo/hi` on 512; unpack + `permute2f128` on AVX2; unpack pairs at
128-bit) instead of the per-line form's two complementary-masked half-writes
per line — halving store μops and removing the double-write RFO pathology.
Masked-tail pairs expand the column mask once (`_pdep_u32(m,0x5555)*3`) and
store full pair vectors under it. Unpaired lines fall through to the per-line
lattice. r4 AVX-512 il_out: 52→**35** vmovupd, 32→**14** shuffles, main-loop
masking eliminated. All 12 il_out files regenerated; il_in files unchanged
modulo dead constants.

**Gates.** Fused vs per-line BIT-EXACT at (ios,me) covering main loop, masked
tail, and sub-vector widths — r4 and r16, both ISAs — and BIT-EXACT plus
roundtrip 1.22–1.33e-15 as whole-transform drop-ins.

**Measured at the live stage-0 exit geometry (3 windows):**

| arm | AVX-512 (r16, g=4) | AVX2 (r4, g=16) |
|---|---|---|
| engine inplace split | 1.000× | 1.000× |
| il_out per-line (old) | 1.250–1.647× (+244–342K) | 1.764–1.820× (+311–352K) |
| il_out fused-pair | **0.996–1.488×** (−4–258K) | 1.784–2.005× |

AVX-512: fused ≤ old in every window (−84 to −274K) and reaches in-place
parity in two of three. AVX2's r4 exit is bandwidth-bound (tiny compute per
byte at 512KB leg strides): the lattice fold has no measurable effect and the
tax is traffic-inherent; an NT-store variant trends −40K but within noise —
both are host A/B items (36MB L3 + real single-core bandwidth change the
regime entirely).

**Planner note.** The exhaustive planner's (64,4096) AVX-512 choice is
nondeterministic on the bench container ([4,16] vs [16,4] across runs); baked
coverage now spans all three observed factorizations ([4,16], [16,4],
[4,4,4]). The regenerated `plan_executors.h` was verified ABI-identical to the
shipped header (`stride_plan_t`/`stride_stage_t` byte-equal after whitespace
normalization).

**Ranked-list closure.** Items 1 (per-slab fusion), 2 (this section), 3 (baked
/JIT resume + bwd tier-1 kernel engagement), 4 (strided il_in emitter fold)
are shipped and gated; item 5 (hugepages) is a platform-side task. Production
stack whole-transform (window-noisy; per-pass tables are the record): fwd
1.28–1.47×, bwd **1.13–1.14×** of the improved floors, roundtrip ≤1.4e-15.

## 6a10. 1D c2c vs MKL on native interleaved — first contact (measured, container)

**Protocol.** Mirror of v1.0's layout convention inverted: v1.0 forced MKL into
its split-storage compatibility mode; this bench runs BOTH sides on MKL's home
format. Lane-major interleaved for both (plus an MKL-preferred contiguous arm
for transparency). Headline metric: fwd+bwd roundtrip z→z — order-neutral (our
in-place is digit-scrambled by the convolution contract) and fully folded on
our side (il_in entry + il_out exit). Plans: fresh DP MEASURE sweep per cell
(`vfft_proto_plan`, orchestrator); executors resolved baked-or-JIT (all cells
printed `exec=jit/jit`). Single thread. Container caveats apply throughout
(1 shared vCPU, gcc -O2, ±30% arm windows). MKL note: `mkl_set_num_threads()`
segfaults in the pip mkl_rt build — control threading via
`MKL_THREADING_LAYER=SEQUENTIAL MKL_NUM_THREADS=1` env only.

| N | K | DP plan | folded | vs MKL-lane | vs MKL-ctg | IL vs split |
|---|---|---|---|---|---|---|
| 64 | 256 | [4,4,4] | yes | **1.09–1.11×** | 0.50× | +10% |
| 256 | 256 | [4,4,4,4] | yes | 0.68× | 0.55× | **+30%** (r4 boundary) |
| 1024 | 32 | [8,16,8] | yes | 0.87–0.99× | 0.66× | +4% |
| 4096 | 32 | [4,4,4,16,4] | yes | 0.81× | 0.58× | — |
| 320 | 32 | [8,8,5] | yes | 0.82× | 0.80× | +21% |
| 1000 | 32 | [5,8,5,5] | no (r5 gap) | 0.73× | 0.75× | +29% unfolded |

**Findings.** (1) The IL boundary itself is cheap (+4–10%) except at
memory-bound r4 stage-0 shapes (section 6a9's pathology). (2) v1.0's odd-N
blowouts were substantially MKL-split-mode artifacts: on native interleaved,
odd composites sit at parity for the split engine. (3) The pow2 deficit vs
MKL-contiguous tracks the **memory-sweep ratio** (our one-sweep-per-stage vs
MKL's rank-fused kernels), not layout: the DP picked 3–5 thin stages where
2-sweep fat factorizations ([16,16], [64,4], [64] mono) exist in-engine.
Candidate explanations, in test order: sweep-through-generic rank inversion
(doctrine-documented), missing calibrator protocol in the orchestrator sketch,
candidate enumeration not surfacing blocked/BUF fat variants. The
discriminating experiment (bench_fat_plans.c: forced [16,16]{T1S,LOG3} and
[64,4] via WISDOM_ONLY, raced under BOTH generic and specialized tiers) is
staged; run on the 14900KF for a stable verdict.

**Coverage extension.** The il_in/il_out derived family now spans every
inplace radix — {2,3,4,5,6,7,8,10,11,12,13,16,17,19,20,25,32,64} — closing the
r5-class eligibility gap that left (1000,32) unfolded. Adapter switch extended
to match. Sources regenerated with the 6a9 pair-fusion; objects are build
artifacts.

**Strided il coverage completed.** The strided il variants (fwd il_out, fwd
il_out_nt, bwd il_in) now exist for r8/r16/r32 alongside r64 — matching the
strided registry — generated with the 6a3+ emitter flags after per-radix
plain-regression byte-diffs (all identical to shipped). Gated BIT-EXACT
against composed references (plain strided + manual (de)interleave) at tight
and padded row strides, both ISAs, 12/12 cells. Split-layout sources remain
untouched throughout the campaign: the deriver reads inplace/ and writes only
to il/; provenance headers verified original.

**Ops note.** The prior session-container died mid-run (OOM: 4GB ceiling;
concurrent bench+mkl_rt+DP sweep+JIT gcc forks) and restored from a §6a9-epoch
snapshot; post-6a9 deltas were replayed from the conversation record. Keep JIT
cache warming and large TU compiles in separate steps from sweeps.

## 6a11. Emitter-level inplace IL modes — the lattice enters the machinery

**What shipped.** `gen_radix.exe R --ip-il-in | --ip-il-out --bwd` now emit
the inplace boundary codelets directly: the z lattice is rendered inside the
generator rather than derived from emitted C. Design (render-level, the
documented middle step before full DAG-node IR):

- **il_in**: memoized first-touch emission. The first scheduled consumer of
  either side of input index j triggers the shared z pair-loads plus BOTH
  deinterleave permutes, flushed as a prefix of that node's definition — so
  placement follows the scheduler's own lazy ordering. Subsequent touches
  resolve to names. Widths: 8 (permutex2var vs function-scope `_il_de/_il_do`),
  4 (unpack + permute4x64 0xD8), 2 (unpack pair), 1 (direct pair indexing).
  Masked pass: maskz z loads under the pdep-expanded column mask.
- **il_out**: the re-side store defers one statement (stash) and fuses with
  the adjacent im-store (sink-first pairs are structural) into full-width z
  pair stores, through the regalloc-aware value resolution (`name_overrides`
  / `spilled_of_tag`). Masked: expanded-mask pair stores.
- Flags imply `--in-place`; symbols match the derived family exactly
  (`radix{R}_n1_{fwd,bwd}_{isa}_il_{in,out}`) — drop-in, adapters untouched.

**Derived-family tail bug (disclosure).** The deriver only rewrote the main
vector loop: every derived il_in codelet's scalar and masked TAIL passes read
`out_re/out_im` — the uninitialized outputs — instead of deinterleaving z.
Broken for any `me % VW != 0`. Exposure: none shipped — every production
shape (the 64³ stack, all 1D bench cells) used vector-multiple `me`, and all
prior gates did too — but the adapter contract permits arbitrary K, so the
class was live. Caught by the first emitter-vs-derived bit gate at
`me ∈ {65,67}`. The deriver is retired for the inplace family (header note);
all 18 radices × 2 modes × 2 ISAs regenerated from the emitter.

**Gates: 144/144.** Composed reference (original split codelet + manual
(de)interleave), all 18 radices, both ISAs, `me ∈ {64,65,66,67}` — covering
main, masked (512), sse2-pair (avx2), and scalar-straggler paths. Doctrine
note: all vector-width paths are BIT-exact; the scalar straggler lane is
gated at ≤4 ULP because the il signature's const/restrict context changes
gcc's FMA contraction inside the plain-double scalar block (proved: r25 goes
bit-exact under `-ffp-contract=off`; both roundings are valid DFTs). Only
r25's 473-line scalar block was large enough to diverge.

**Spill audit (emitter vs original; derived delta for comparison):**

| codelet | orig | emit | Δemit | Δderived |
|---|---|---|---|---|
| r16 512 il_in | 30 | 66 | +36 | +33 |
| r16 512 il_out | 28 | 47 | +19 | +15 |
| r16 avx2 il_out | 148 | 168 | +20 | +20 |
| r64 avx2 il_in | 1539 | 1261 | **−278** | −242 |
| r64 avx2 il_out | 1531 | 1702 | +171 | +171 |
| r64 512 il_in | 644 | 504 | **−140** | −142 |
| r4 both | 0 | 0 | 0 | 0 |

Honest read: emitter placement does NOT reduce the inflation — the deltas are
liveness-inherent (the im side hoisted to the re side's first touch; store
pairs held together), and gcc allocates the same dataflow identically whether
it CSE'd it from derived text or received it emitted. The il_in reductions
(const/restrict breaking in-place alias chains) persist. Measured cost of the
positive deltas at production shapes is noise (6a9/6a10 whole-transform data).
Escalation path if a profile ever shows a boundary hot: a reload policy
(re-load z at the second side's touch, trading 2 register lifetimes for 2 L1
hits) or full DAG-node lattice IR under the pressure model.

## 6a12. Complete boundary-fold matrix — DIF and twiddled-stage IL (gated)

**Motivation.** 6a11's finding sharpened: the DP planner always searched DIT
and DIF (wisdom carries `use_dif`; the legacy corpus contains DIF variants) —
it was the FOLD layer that only covered DIT's n1 boundaries. Every other
plan-shape boundary fell back to conversion sweeps or was unfoldable.

**Now shipped** (emitter modes compose with every source family; twiddle
loads are `Expr.Twiddle`, structurally untouchable by the `Expr.Input`
lattice guard):

| plan | fwd entry | fwd exit | bwd entry | bwd exit |
|---|---|---|---|---|
| DIT | n1 il_in (6a11) | **t1s_dit il_out** | **t1s_dit il_in** | n1 il_out (6a11) |
| DIF | **t1_dif il_in** | **n1 fwd il_out** | **n1 bwd il_in** | **t1_dif il_out** |

Flags: the n1 combos are pure flag runs (`--ip-il-out` fwd / `--ip-il-in
--bwd`); twiddled combos compose as `--twiddled --in-place [--t1s|--dif]
[--bwd] --ip-il-{in,out}`. The twiddled name builder gained the il suffix.
Coverage: n1 combos at all 18 radices; twiddled at {4,5,8,16,25,32} — the
radix set observed in DP winners — with full-18 a loop extension (sources
exist for all).

**Gates: 432/432.** Composed references (original + manual (de)interleave,
real twiddles: broadcast tw[R] for t1s, per-element tw[R·me] for t1_dif),
both ISAs, `me ∈ {64,65,(66),67}`: 6a11 set 144, n1 combos 144, twiddled 144.
Gate-doctrine refinement: partial-tail lanes accept ≤4 ULP **or**
≤4·ε·max(|a|,|b|,1) — the absolute floor catches catastrophic-cancellation
sites (~1e-17 results flipping sign under different FMA contraction; ULP-
astronomical, numerically nil). Both r25 exemplars (n1 and t1_dif scalar
blocks) proven pure contraction artifacts: bit-exact under
`-ffp-contract=off`, ≤2 ULP measured otherwise. All vector-width paths BIT.

**To take DIF folding live** (adapter/planner layer, not emitter):
il_execute needs fn tables for the three new combo classes and DIF-aware
wrappers respecting reversed stage order; t1_dif carries no tw[0] references
(leg 0 untwiddled in-codelet — no executor fixup interaction, unlike t1s
whose cf0 contract must be preserved by a t1s-boundary adapter); and once
folding is plan-shape-universal, planner boundary-blindness reduces to a
pricing term (stage-0 r4 IL penalty vs free r16/r64 — 6a10 data). The 1D
payoff is DIF-fwd: both boundaries fold with zero conversion sweeps,
eliminating the sp2il pass behind the fwd-only 0.5–0.8× column in the
6a10 MKL table.

## 6a13. DP planner vs IL-truth at (1024, 4), AVX2 (measured, container)

The boundary-blindness question, tested directly: fresh DP MEASURE pick raced
against 12 forced DIT factorizations under the fully-folded IL roundtrip
(fwd_ilin + bwd_ilout, jit tier, K=4 = one AVX2 vector).

**Verdict: the DP found the IL-truth winner — [64,16], stable 5/5 fresh
sweeps, IL-rt 78.7K, 10.6% ahead of #2 [4,16,16].** But not by construction:
under the jit tier the split ranking has [32,32] ≈ [64,16] (73.7K vs 74.5K, a
1% tie), and their IL fates diverge sharply (+21.2% vs +5.6% boundary tax).
The generic-tier sweep separates the pair stably in the IL-favorable
direction at this cell; where that coincidence breaks, the boundary pricing
term (6a10/6a12) remains the principled fix.

New microarchitectural data from the race:
- **r32 avx2 il_out: +94 spills** (orig 524 → 618; audit-gap radix) — at
  me=4, zero amortization makes this the whole +21.2% [32,32] tax. il_in side
  −43. The r32 exit is the worst avx2 boundary measured.
- **Negative IL tax is real**: r8-stage-0 plans ([8,16,8], [8,8,16]) run the
  IL roundtrip **12% faster than split** — the const-z/restrict alias-chain
  effect (6a11's r64 −278 spills) manifesting live where in-place
  conservatism dominates at tiny me.
- Regime inversion vs (256,256)/K=256: at K=4 the thin extreme [4,4,4,4,4]
  is worst (114K, +45% over best) and 2-stage fat wins — matching the wisdom
  corpus's mono picks at K=4. The planner handles both regimes.

**Head-to-head vs MKL at the same cell** (bench_best_plan_vs_mkl.c, [64,16]
forced, jit, MKL native interleaved, env-only threading): VFFT IL-rt 83.5K,
split 77.1K, MKL-lane 50.5K (0.61x), MKL-ctg 47.4K (0.57x); fwd-only 0.52x
(unfolded sp2il exit — the DIF-fwd both-folded case). K=4 mechanism notes:
MKL-lane ~ MKL-ctg (7% apart) because the K=4 complex lane stride is 64B =
one cache line, neutralizing the strided penalty that produces our large-K
wins; simultaneously me=4 = one AVX2 vector per codelet inner iteration, so
per-leg fixed costs amortize over a single vector. The gap is engine-vs-
kernel (split alone is 0.61x of MKL-ctg); the IL layer adds only ~6-8% at
this plan. Container/-O2 caveats apply; the 14900KF/-O3 run prices the build
handicap. (A side excursion to K=32/256 produced numbers inconsistent with
the 6a10 grid for the same cells — unresolved, out of scope per direction,
and NOT to be treated as valid data.)

Bench: build_tuned/benches/bench_dp_vs_il_truth.c (WISDOM_ONLY forced arms +
fresh MEASURE arm, warm/measure invocations separated per the OOM
discipline).

## 6a14. The mode gap — v1.0 and the native-interleaved results reconciled

Measured on the bench container, one window (bench_mode_gap_reconcile.c):
ours-split beats MKL-SPLIT-mode (the v1.0 comparator: DFTI_REAL_REAL,
identical lane-major layout) by **3.33x at (1024,4)** and **4.53x at
(64,256)** — the v1.0 record reproduces here, at gcc -O2 on a shared vCPU.
In the same window, **MKL's split mode runs 5.63x / 8.50x slower than its
own native interleaved kernels**. The identity closes to a few percent:

    v1.0 margin / mode gap = position vs MKL-native
    3.33 / 5.63 = 0.59   (measured 0.57-0.61)
    4.53 / 8.50 = 0.53   (measured 0.50)

Consequences. (1) Nothing is degraded and no build-handicap story is needed:
the engine reproduces v1.0-class margins today. (2) The IL layer is not the
loss — the split column loses the native-MKL cells identically; the boundary
lattices cost +6-8% (sometimes negative). (3) v1.0's comparison universe —
deliberately identical-layout, therefore MKL's split compatibility path —
sat 5.6-8.5x below MKL's actual ceiling. The library's true position against
MKL's best is v1.0-margin-over-mode-gap: decisive wins in the split /
batched / odd / threaded / scrambled universe, 0.5-0.6x at native-interleaved
small-K pow2, with the (64,256)-class large-K IL cells at parity-to-win
because lane-major hurts MKL more than the residual. (4) The section-7
non-goal (no interleaved arithmetic) now has a price tag; the remedies
ladder stands: four-step for K<VW, rank-fused construction for K~VW,
within-transform codelet family as the endgame if native-IL small-K becomes
a first-class product target. The v1.0 results doc should carry this mode-gap
number alongside its margins so the record states its own baseline.

## 6b. Register-pressure analysis for P2 (measured)

The natural objection to fusing deinterleave into stage-0 loads: doesn't the shuffle
lattice push codelets past 16 ymm / 32 zmm? Probed at the worst boundary case —
an AVX2 R=8 stage-0 (8 complex legs = 16 live plane vectors = the entire file)
with the full per-leg lattice (2 loads + unpacklo/hi + 2×permute4x64):

```
                    spills   asm lines   runtime vs split loads
 split loads          0         85              1.00×
 IL-deint loads       0        109           1.12–1.16×
```

**Zero spills even at the full-file boundary.** The mechanism: the lattice's
transients (pair vectors, lo/hi) have 2–4 instruction lifetimes and die INTO the
plane vectors they produce — they never coexist with the butterfly's peak pressure,
and any pressure-aware scheduler (gcc here; the list scheduler's lazy-loads in the
generator) interleaves load sites so transients from different legs don't stack.
Peak = plane vectors + ≤2 transients, transiently.

Persistent-register cost by ISA: **AVX2 zero** — `permute4x64` takes an immediate,
no index vector exists. **AVX-512: +2** loop-invariant `vpermt2pd` index constants
(and the lattice drops to 2 μops per pair instead of 4) — the same class as twiddle
broadcasts, drawn from the documented free pool (R=20 pairing leaves 8 zmm free).
Store-side (bwd / output boundary) is cheaper still: the shuffles consume dying
values. The strided family is the best case of all: its in-register 8×8 transpose
network already exists, and pairs are one additional unpack layer folded into it.

The 1.12–1.16× is shuffle-port μops on stage-0 ONLY — one stage of the chain, priced
against the full conversion sweep it deletes (≈ a whole transform's traffic at 64³).
And the confinement matters: P1b touches **no codelet registers at all** (its fusion
site is the transpose kernels, with their own small working sets); only P2 enters
codelet budgets, only at the boundary stages, and every P2 variant passes through
the emitter's existing pressure gates plus DP racing — a variant that spills loses
its race and the calibrator keeps the sweep route for that cell. Nothing ships on
faith.

## 7. Deliberate non-goals

- **No interleaved arithmetic anywhere** (§2 — forfeits accuracy shape + register
  economics + the calibrated corpus).
- **No in-place il↔sp on the same buffer** (z reinterpreted as [re-half | im-half]):
  the permutation does not tile-decompose — early tiles' im-half writes land on
  unread later rows. The cube costs n·2 doubles and buys total freedom; rejected
  cleverness.
- **No silent transform-major acceptance** (§1).
- **No FFTW-API shim yet**: the shim is a consumer of this layer (it needs P1b/P2
  economics to be honest about `FFTW_MEASURE`-class comparisons), not a part of it.

## 8. Test architecture

`test_il_layout.c` gates, in order of strength: kernel round-trip identity with odd
tails (bit); wrapper output **memcmp-equal** to the manual il2sp→split→sp2il route
on 1D lane-major, rank-2/3/4, and prime cells (conversion exactness makes the
equivalence claim bit-level, not tolerance-level); IL roundtrips ~1e-15; T∈{1,4}
bit-identity under strided rows; r2c IL-out spectrum pairs bit-exact vs the split
plan; and the 64³ tax bench printed for the record. The suite runs with
`VFFT_STRIDED_ROWS` on, so the IL boundary is exercised over the padded-tail and
natural-order machinery simultaneously.
