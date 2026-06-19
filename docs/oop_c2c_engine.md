# Out-of-Place C2C: the engine, the 2-axis planner, and wisdom-driven dispatch

**Date:** 2026-06-19
**Hardware:** Intel i9-14900KF (Raptor Lake), AVX2 only, single-thread, pinned core 2
**Toolchain:** mingw gcc 15.2, MKL oneAPI (`mkl_set_num_threads(1)`), split (REAL_REAL) layout
**Scope:** out-of-place complex→complex FFT (`src ≠ dst`), the OOP counterpart to the in-place c2c engine

> TL;DR — The OOP c2c path is wired, measured, and wisdom-driven on Raptor Lake AVX2. With a calibrated
> `oop_wisdom.txt`, the runtime does a **pure lookup** (no measurement) and **beats MKL 1.46×–2.64× across
> all 10 calibrated cells**, correctness verified. The planner decides **two coupled axes** —
> *kind* (LEAF/BAILEY2/MODEB) and *factorization* — and the DP planner does the joint search.

---

## 1. Three execution kinds

`vfft_oop_plan_t` is a tagged union over three models (`core/oop_plan.h`):

| kind | what it is | order |
|------|-----------|-------|
| **LEAF** | N≤128 with a single OOP codelet. One call, column layout. | natural |
| **BAILEY2** | fused 4-step two-factor (N=R1·R2). s1 = R1 calls of the R2-point leaf with the transpose **fused into the stores**; s2 = one `t1p_log3(R1)` pass + K-replicated twiddle table. | natural `X[k2+R2·k1]` |
| **MODEB** | general-N via the stride core, out-of-place adapter (`core/oop_execute.h`): stage 0 OOP (reads `src`, writes `dst`), stages 1.. in-place on `dst` — **no extra copy** (the boundary is fused into the leaf, same trick that fixed the r2c pack). | scrambled (digit-perm) |

MODEB inherits the in-place c2c engine (the 238/238-vs-MKL winner) redirected to a separate output buffer.
That's why **MODEB is the workhorse** — its DP-optimal multi-factor decomposition beats the 2-factor BAILEY2
in most cells (in calibration, BAILEY2 won only at 169=13² and 512 K=120).

---

## 2. The two decision axes

A plan must decide **both**, and they are **coupled** (the best factorization depends on the kind, and the
best kind depends on its best factorization):

- **Axis 1 — kind:** LEAF vs BAILEY2 vs MODEB.
- **Axis 2 — factorization:** within BAILEY2 the `R1×R2` pair; within MODEB the multi-factor decomposition
  (LEAF is trivial — N itself).

So the planner is a **2-level joint search**: find each kind's best factorization, then pick the winner
across kinds.

- **Axis-2 within BAILEY2** → the pair tuner (`vfft_oop_tune_pairs`, `core/oop_auto.h`) measures LEAF + all
  unmasked `R1×R2` pairs same-binary. (Pairs whose stride is a multiple of the 32KB L1-set period with more
  streams than 8-way associativity are *aliasing-masked* and skipped — `oop_plan.h:72`.)
- **Axis-2 within MODEB** → the recursive DP planner (`core/dp_planner.h`, FFTW-PATIENT-style measured
  search with sub-problem memoization).
- **Axis 1** → measure the two champions, keep the faster.

This is `vfft_oop_plan_create_dp_best` in **`core/oop_dp.h`** — the joint chooser. It is the offline
*calibration primitive*.

### Why this matters: the aliasing cell

At **N=1024 K=256**, the only *unmasked* BAILEY2 pair is `8×128` (the balanced `32×32` aliases at K=256), and
it runs at **0.74× MKL** — a loss. The joint chooser measures `8×128` against DP-MODEB `4⁵` and keeps MODEB
→ **1.49× MKL**. The loss is resolved by *measurement across both axes*, not a hand rule.

---

## 3. Wisdom: persisting the 2-axis decision

DP measuring at plan time (~150 sub-benchmarks) is calibration-time only — far too expensive per call. So the
verdict is cached in a **separate `oop_wisdom.txt`** (not the c2c `spike_wisdom`, which is MODEB-shaped only
and shared with the in-place path). One line per `(N,K)` encodes **both axes**:

```
N  K  kind  [params...]              ns
# kind 0 = LEAF    : (no params)        64   512  2 3 4 4 4         (MODEB nf=3 4,4,4)
# kind 1 = BAILEY2 : R1 R2              169  512  1 13 13
# kind 2 = MODEB   : nf f0..f(nf-1)     1024 256  2 5 4 4 4 4 4
```

(MODEB variants aren't stored — the DP planner is all-T1S today, so MODEB rebuilds with `variants=NULL`
= T1S. A variant-aware DP would add a column + a version bump.)

**Lifecycle (matches the rest of the library — rfft/c2r/c2c each have their own wisdom):**

```
calibrate_oop  ──(vfft_oop_plan_create_dp_best over a grid)──▶  oop_wisdom.txt
                                                                     │
runtime:  vfft_oop_plan_create_wisdom(N,K,&wis,reg)  ◀── pure lookup, build exact kind+params
          (core/oop_wisdom.h — no measurement, no DP dependency; rule/DP fallback on miss)
```

---

## 4. Results (Raptor Lake AVX2, vs MKL NOT_INPLACE split)

Pure-lookup rebuild from `oop_wisdom.txt` (`oop_wisdom_smoke.c`) — every cell beats MKL, correctness verified
(MODEB bit-exact vs in-place `0.0`; LEAF/BAILEY2 roundtrip `≤1.5e-14`):

| N | K | kind | factorization | vs MKL |
|---|---|------|---------------|--------|
| 64 | 512 | MODEB | 4,4,4 | **2.64×** |
| 128 | 512 | MODEB | 4,4,4,2 | **1.94×** |
| 169 | 512 | BAILEY2 | 13×13 | **1.50×** |
| 256 | 256 | MODEB | 4,4,4,4 | **1.48×** |
| 512 | 120 | BAILEY2 | 32×16 | **1.58×** |
| 1024 | 120 | MODEB | 4,8,4,8 | **1.84×** |
| 1024 | 256 | MODEB | 4⁵ | **1.49×** (was 0.74× rule-BAILEY2) |
| 2048 | 256 | MODEB | 4,4,4,4,2,4 | **1.46×** |
| 4096 | 256 | MODEB | 4⁶ | **1.62×** |
| 2310 | 32 | MODEB | 11,6,7,5 | **2.42×** |

`fails=0` (built kind matches the entry; errors below tolerance).

---

## 5. Files

| file | role |
|------|------|
| `core/oop_plan.h` | the 3-kind plan + LEAF/BAILEY2 constructors + execute dispatch |
| `core/oop_auto.h` | rule-spine + wisdom/hint auto-create + the BAILEY2 pair tuner (made Windows-portable) |
| `core/oop_execute.h` | MODEB out-of-place adapter onto the stride core (boundary-fused, no copy) |
| `core/oop_dp.h` | **DP-backed** constructors: `_dp` (fallback), `_dp_modeb` (force), `_dp_best` (**2-axis joint chooser**) |
| `core/oop_wisdom.h` | `oop_wisdom.txt` format + load + lookup + **`vfft_oop_plan_create_wisdom`** (runtime pure-lookup) |
| `build_tuned/benches/calibrate_oop.c` | offline calibrator (drives `dp_best`, writes the wisdom) |
| `build_tuned/benches/oop_wisdom_smoke.c` | lookup-rebuild + correctness + vs-MKL validation |
| `build_tuned/benches/{bench_oop_dp,bench_oop_vs_mkl}.c` | measurement harnesses |

Build wiring: `build.py`'s codelet lib now globs `codelets/oop/{isa}` (the OOP `n1`/`t1p` symbols).

---

## 6. Gotchas

- **MODEB output is scrambled order.** Its correctness is *bit-exact vs the in-place dataflow* — not vs MKL
  and not via the swap-roundtrip (those work only for natural-order LEAF/BAILEY2). A consumer needing
  natural order from MODEB pays a reorder pass (not yet wired/measured).
- **DP planning measures** (~150 sub-benches) → calibration-time only; the runtime path is a pure lookup.
- **mingw lacks `aligned_alloc`** → the OOP path uses `_aligned_malloc`/`_aligned_free` (must pair; not `free`).
- BAILEY2's `t1p` is hardcoded to the log3 variant; flat-vs-log3 selection by the cost model is a TODO
  (`oop_plan.h`).

---

## 7. Reproduce

```sh
cd build_tuned
python build.py --src benches/calibrate_oop.c   --compile        # offline calibrator
python build.py --src benches/oop_wisdom_smoke.c --mkl --compile  # lookup + validate
python build.py --src benches/bench_oop_dp.c     --mkl --compile  # DP / dp_best A/B
cd benches
PATH="<MKL bin>;C:\mingw152\mingw64\bin;$PATH"
./calibrate_oop.exe        # writes oop_wisdom.txt
./oop_wisdom_smoke.exe     # rebuild via lookup + correctness + vs MKL
```

Related: `docs/performance/high_k_real_fft_architecture_wall.md` (the r2c counterpart), project memory
`oop_c2c_productionized.md`, and the OOP machinery map (`oop_machinery_map.md`).
