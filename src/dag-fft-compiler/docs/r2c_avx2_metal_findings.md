# R2C on the i9-14900KF (AVX2) metal box — what we tried, what we found, where it stands

STATUS: investigation conclusion (2026-06-18). R2C forward is **build-integrated and
correct on this box; competitive at low K; loses ~2× at high K**. The high-K gap is
real, its cause is pinned (compulsory plane traffic, not capacity), and the fix is a
**multi-lever stack that is only partial on AVX2** — the headline R2C win lives at
AVX-512 width, which this box does not have. This doc records the platform, every
experiment, the measured numbers, and the honest verdict so the next session (or the
AVX-512 host) starts from fact, not inference.

---

## 1. The platform (why it shapes everything)

- **CPU: Intel i9-14900KF (Raptor Lake).** AVX2 + FMA. **No AVX-512** (consumer
  Raptor Lake fuses it off). L1d 32 KB, L2 2 MB/P-core, L3 33 MB. Up to ~5.7 GHz.
  24 cores (8 P + 16 E), so MKL multithreads aggressively by default.
- **Toolchain: native Windows + mingw gcc 15.2** (`C:\mingw152`), **MKL via oneAPI
  (ILP64, sequential when pinned)**, VTune present (oneAPI) but `uarch-exploration`
  needs an **elevated** shell for hardware EBS (unavailable from the agent shell;
  unelevated VTune gives only time/line hotspots, not the memory-bound % that would
  be the clincher).
- **The OCaml codelet generator only runs under WSL** (the built binaries are ELF;
  no OCaml/dune on the Windows side). WSL Ubuntu is present, runs the ELF generator,
  and sees the repo via `/mnt/c`. So **every R2C codelet is a WSL round-trip**; the
  C is portable and compiles with mingw on the Windows side.

Two consequences that recur below: (a) MKL's default multithreading will dwarf any
single-thread comparison if you forget to pin it; (b) this box's **real cache
hierarchy + prefetcher** make it the metal that the prior R2C work (done on a
container with L3≈DRAM and no prefetcher) explicitly said it *needed* to decide its
open questions.

---

## 2. The goal

R2C (real-input forward FFT) is the highest-value parity feature to bring into the
canonical `src/dag-fft-compiler/` tree (it was developed/validated elsewhere, bundled
under `docs/r2c_bundle/`). The engine headers (`rfft.h`, `c2r.h`, `r2c.h`,
`r2c_dispatch.h`) are **byte-identical to the bundle** — already in dag/core. The
work was integration + validation on this box, then closing the vs-MKL gap.

---

## 3. What we did, in order, with results

### 3a. Build integration — DONE
- `build_tuned/build.py`'s codelet lib globbed a single dir (the 324 c2c codelets).
  Extended `dag_codelet_srcs()` to also glob `codelets/rfft/<isa>` + `codelets/c2r/<isa>`
  → **417 codelets** link cleanly.
- `rfft.h` used C11 `aligned_alloc`, which mingw lacks (and `free()` on
  `_aligned_malloc` memory corrupts the heap). Added matched
  `RFFT_ALIGNED_ALLOC/_FREE` (Windows `_aligned_malloc/_aligned_free`, else
  `aligned_alloc/free`). Same class of fix as the Rader bridge.
- Forward R2C bench: `build_tuned/benches/bench_r2c_fwd_vs_mkl.c` (native `rfft.h`
  path, `rfft_register_all_avx2`, K + Kb args, vs MKL `DFTI_REAL`).

### 3b. The "12× slower" panic — it was MKL multithreading (FIXED)
First run showed mkl/ours = 0.62 / 0.14 / 0.08 at K=8/64/256 — absurd. Cause: the
bundle's minimal bench **never pinned MKL threads**, so MKL spread the K=256 batch
across all 24 cores. Adding `mkl_set_num_threads(1)` gave the fair single-thread race.
**Lesson: always pin MKL to 1 thread for an ST comparison.**

### 3c. Fair forward R2C vs MKL (native rfft.h, AVX2, ST, MKL pinned) — the real numbers
```
FWD N=256 (8,32)  K=8   | mkl/ours = 0.95   (near parity)
                  K=64  | mkl/ours = 0.67
                  K=256 | mkl/ours = 0.57
```
Per-column cycles: **ours climbs 442 → 724 → 914 cyc/col**; **MKL stays ~flat
422 → 430 → 474**. So we're parity at cache-resident K and lose ~2× by K=256 because
our per-column cost grows with batch while MKL's doesn't.

### 3d. The Kb (lane-blocking) decision experiment on metal — capacity hypothesis DEAD
`doc 63` (bundle) pre-registered a Kb sweep to decide *capacity-bound* (fixable by
lane-blocking) vs *compulsory traffic* — and said the **container could not run it**
(its L2=1 MiB never spills the 256 KB working set). This box's L1=32 KB *does* spill
at K≥64, so it *can* host the test. Result (mkl/ours, higher=better):
```
K=64:  Kb=full 0.669 | Kb=32 0.585 | Kb=16 0.522 | Kb=8 0.538
K=256: Kb=full 0.570 | Kb=64 0.502 | Kb=32 0.526 | Kb=16 0.506
```
**Monotone-worse: full-width is best, narrowing Kb only hurts.** Per doc 63's decision
tree this kills the capacity hypothesis on metal too. The large-K gap is **compulsory
plane traffic**, and the named lever is **stage fusion** (delete a plane round-trip),
not lane-blocking. (RANGED — call-granularity — was also tried: inert here, as
predicted, since the per-call overhead is K-invariant.)

### 3e. Fusion investigation (multi-agent + adversarial) — fusion is the right lever, but PARTIAL
Both adversarial verifiers independently:
- **ruled out the alternatives** (twiddle-table growth, per-call overhead, pure
  compute — all K-invariant, so dead as the *climb's* cause), cornering it to
  compulsory traffic + locality;
- concluded **fusion caps at ~0.6–0.72×, not parity** — because the K=8 floor
  (442 cyc/col, register-resident) is already ~93% of MKL's *entire* K=256 budget
  (474). Fusion removes traffic, not that compute/overlap floor.
- **unanimous recommendation: measure on metal before building** — and flagged that
  an equivalent fused path is already built (model-b, below).

### 3f. The decisive reveal — `step2_fusion_design.md`: model-b is BUILT, CORRECT, tax-fixed — but on the *stride* path, and only PARTIAL
The "MKL move" (fuse last DIT stage + the r2c fold, deleting the scratch round-trip)
was **fully implemented and gated correct** in the prior work:
- math builder + `radix256_r2c_term_ls_r8` codelet + `_until` slice executor +
  `_r2c_laststage_fused`, opt-in via `ls_fwd`. **FUSED-vs-DEFAULT = 2.84e-14**
  (numerically correct, every frequency).
- A first attempt **regressed −15.7%** from a monolithic-emission spill storm
  (163 stack movs/iter); **diagnosed and FIXED** via the doc-58 PASS-1/PASS-2 seam
  (laststage phase 90µs→44.5µs, total −15.7%→−1.3% on the container).
- The doc's own ablation decomposition (stride path): **pack 48% (3.2× over floor),
  post 36%, inner 16%.** Model-b → **~1.40× (still a loss)**; the path that actually
  *beats* MKL is **pack-to-floor → ~0.86×**. The doc states it: fusion is
  **"NECESSARY, NOT SUFFICIENT."**
- Every measurement was on the **container** (L3≈DRAM, no prefetcher — blind to the
  exact access-pattern/round-trip effect fusion targets). The journey's last words:
  **"STILL needs metal to confirm the net win."**

### 3g. The path distinction that reframes everything
- **Production path = native `rfft.h`** (`r2c_dispatch.h` routes (8,32) here; my
  benches measured it). Its high-K loss (0.57×) is a leaf→planeA→terminator
  round-trip; its fusion (leaf+terminator) is **UNBUILT**.
- **Model-b lives on the *stride* `r2c.h` path** — the slower fallback. So the built,
  correct, tax-fixed fusion **does not directly move the production path**. It's a
  proxy for "does the fusion approach win on silicon," not the production win.

---

## 4. The verdict

For R2C forward **on this AVX2 box**:

1. **Integrated and correct** (417 codelets link; rfft.h Windows-portable; engine
   headers identical to the validated bundle). *Pending:* the roundtrip correctness
   gate (`gate_c2r_matrix`, r2c→c2r = N·x) re-run on AVX2 to formally lock it in.
2. **Competitive at low/cache-resident K** (parity, 0.95× at K=8).
3. **Loses ~2× at high K** (0.57× at K=256) to **compulsory plane traffic** — proven
   on metal (Kb monotone-worse), not inferred.
4. **Closing it is a multi-lever stack, each lever partial on AVX2:**
   - *stage fusion* (delete a plane round-trip): native-path version UNBUILT; the
     stride-path model-b is built+correct but → ~1.40× alone (necessary, not
     sufficient);
   - *pack-to-floor* (the fused first stage is 3.2× its bandwidth floor): the **bigger
     lever** per the decomposition, → ~0.86× if reached; unbuilt-optimization;
   - *natural-order output* (kill the mirror scatter): cheap, complementary.
5. **The headline R2C win (1.2–1.4× over MKL) was measured at AVX-512 width.** This
   box has no AVX-512. At AVX2 (4-wide vs MKL's tuned K-blocking) the realistic ceiling
   of the full stack is **parity-ish, not a blowout.**
6. **All R2C codegen requires the WSL/OCaml bridge** (feasible; every codelet is a
   round-trip).

**Bottom line:** R2C is real, integrated, and low-K-competitive here. The high-K
parity is a genuine multi-lever, codegen-heavy effort with a partial AVX2 ceiling on
the path/ISA that structurally favors MKL — best realized on AVX-512 hardware, or
pursued deliberately here with eyes open.

---

## 4b. THE METAL VERDICT (2026-06-18): model-b fusion REGRESSES on silicon — no-go

The doc-58 journey ended "needs metal" for the model-b fusion. We ran it here. To do
so we **ported the doc-58 spill seam into this tree's generator** (it had only the
MONOLITHIC emission): added `dft_expand_r2c_term_laststage_spill` (the two DFT-r
columns as PASS-1 spill markers, cluster `(2,r)`), dispatched it from `gen_main`,
and added the `--r2c-term-ls` `recipe_applicable` trigger. Rebuilt (WSL/dune), the
codelet now emits **"BLOCKED two-pass CT 2×8"** with 16 spill slots (matches the
doc's fixed result). A/B driver: `build_tuned/benches/bench_r2c_modelb_ab.c` (stride
r2c, inner=(16,8) — the guard-whitelisted shape with last radix 8, matching the r8
codelet; default `ls_fwd=NULL` vs fused; vs MKL; ST).

Result (N=256, ST, MKL pinned), fused vs default and both vs MKL:
```
K=8    fused CORRECT (7e-15)   FUSED/DEFAULT 0.934x   mkl/def 0.81  mkl/fus 0.76
K=64   fused CORRECT (7e-15)   FUSED/DEFAULT 0.925x   mkl/def 0.61  mkl/fus 0.57
K=256  fused CORRECT (8e-15)   FUSED/DEFAULT 0.783x   mkl/def 0.63  mkl/fus 0.49
```
**Fusion is numerically correct but SLOWER than default on metal, and the regression
GROWS with K (0.93→0.78).** This is the doc's deferred "second-order" concern
confirmed: the fused codelet's **scattered dual-output writes** (Xp at k·K, Xm at
mir·K, slot-reversed) defeat the prefetcher on real silicon, and that penalty
*exceeds* the deleted scratch round-trip — the opposite of the hoped access-pattern
win the container was blind to. The round-trip deletion is real; the scatter it
introduces costs more.

**Decision (per the pre-registered discipline): negative proxy → B, with confidence.**
Porting model-b to the native `rfft.h` path would inherit the same scatter penalty; a
contiguous-output codelet redesign is yet another lever, and even then the AVX2
ceiling is partial. The fusion thread is NOT the path to an R2C win on this box. The
spill-seam port is KEPT (correct, zero-regression, generator now has the doc-58 fix).

## 5. The costed roadmap (if/when the R2C high-K win is pursued)

In dependency order, each gated brute-Hermitian + measured on metal:

| lever | state | expected (per doc / verifiers) |
|---|---|---|
| **0. correctness gate** on AVX2 (r2c→c2r = N·x) | not yet run here | lock in integration |
| **1. model-b metal A/B** (stride path, built) — generate `r2c_term_ls_r8` via WSL, A/B default vs `ls_fwd` | built; needs codelet gen + driver | decides "fusion wins on silicon?" (proxy) |
| **2. native-path fusion** (leaf+terminator into `rfft.h`) | UNBUILT (reuses model-b fold + spill seam) | partial: ~0.57×→~0.65–0.72× |
| **3. pack-to-floor** (why is the fused first stage 3.2× its bw floor?) | unbuilt-optimization | the bigger lever: → ~0.86× (beats MKL) |
| **4. natural-order output** (contiguous mirror streams) | cheap loop reorg / emission | complementary scatter kill |

The decisive cheap next step is **#1** (built code + WSL codegen + an A/B driver): a
metal default-vs-fused number settles whether the whole fusion thread is worth porting
to the native path, using already-correct code.

---

## 6. Recommendation

**Bank the real R2C progress and redirect the parity effort.** Concretely: keep the
build integration + the Windows portability fix + (after the gate) the correctness
proof + this costed roadmap; do **not** open the multi-lever, WSL-codegen, partial-on-
AVX2 push mid-stream. Spend max effort on a parity feature where AVX2 can win cleanly
(**2D, DCT** — machinery already validated in the bundle/prototype), and run the R2C
high-K stack deliberately when on AVX-512 hardware (where the 1.2–1.4× lives) or when
it's the prioritized target. If the metal verdict is wanted regardless, roadmap **#1**
(the model-b A/B) is the sharp, built-code test to run first.

---

## 7. Artifacts / references
- Bench: `build_tuned/benches/bench_r2c_fwd_vs_mkl.c` (native rfft path, K + Kb args).
- Engine: `core/rfft.h` (native), `core/r2c.h` (stride + model-b `ls_fwd`),
  `core/r2c_dispatch.h` (routing), `core/c2r.h`.
- Prior work: `docs/step2_fusion_design.md` (model-b full log), `docs/r2c_bundle/docs/`
  {59,60,61,62,63}, `verdict_r2c_c2r_vs_mkl.md`, `native_rfft_design.md`.
- Generator (WSL/ELF): `generator/_build/default/bin/` (`gen_radix`, `gen_set`),
  flags `--r2c-term-ls --r2c-term-ls-r {radix}`.
- Memory notes: `r2c_metal_kb_dead_stage_fusion.md`, `dag_feature_gap_vs_production.md`.
