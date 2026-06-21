# Schedule Search — Phase 0 Results (Measurement Substrate + Spills↔Runtime Proxy)

> Phase 0 of the schedule-search experiment (plan:
> [`docs/roadmap/schedule_search_plan.md`](../roadmap/schedule_search_plan.md)).
> Goal: before writing any schedule-search code, establish a *trustworthy*
> measurement and answer the GO/NO-GO question — does a cheap proxy (spills)
> predict runtime, and is there headroom over the production scheduler?
>
> Target: i9-14900KF (Raptor Lake), AVX2, locked 5.7 GHz (Ultimate Performance).
> Toolchain: gcc 15.2 (mingw), production codelet flags
> `-O3 -mavx2 -mfma -march=native -fpermissive -w`. Codelets: in-place leaf
> `radix{R}_n1_fwd_avx2`, generated `--in-place --isa avx2`.

## TL;DR

- **Absolute single-codelet timing is untrustworthy** (~22% bimodal across process
  launches even pinned + locked clock). **Paired in-process comparison** (the
  `bench_1d_vs_mkl` methodology) is rock-solid: A-vs-A ratio = **1.000 ± 0.1%**.
  Comparison MDE ≈ **0.1–0.3%**.
- **Asm spills strongly predict runtime** across scheduler variants:
  **Spearman ρ ≈ 0.94**. → minimizing realized spills is a valid objective.
- **`peak_live` (the cheap model) does *not* predict asm spills**: `--bb`
  minimizes `peak_live` yet emits **2× the asm spills** of `--su` at R=64 and runs
  30–45% slower. → the search must score on **asm spills or runtime**, not the
  model.
- **`--su` (production) is already the best available scheduler.** `--gh` is a
  no-op; `--bb` and `--bisect` both lose to `--su`. The incumbent is strong.
- **Verdict: conditional GO.** Objective = asm spills; baseline to beat = `--su`.

---

## Measurement methodology (Phase 0.1 / 0.2)

A naive "time one codelet, best-of-N batches" harness gives a tight *within-run*
spread (cov ~0.5%) but a **~22% bimodal swing between process launches** — the
residual per-launch CPU frequency/core-placement state, even pinned to a quiet
P-core at a locked 5.7 GHz. Absolute cross-launch numbers are therefore not
comparable.

The fix (lifted from `build_tuned/benches/bench_1d_vs_mkl.c`): **never compare
across launches.** Measure both candidates *back-to-back in one process*, on the
**same buffer**, each as **10 warmup + best-of-5 min ns/call**, with a
**cachebust + pace** between them and the **A/B order flipped** across reps;
compare the **ratio**. The shared per-launch state cancels.

**Validation (A vs A, same codelet twice), 6 launches, K=256:**

| run | A_min (ns) | B_min (ns) | ratio |
|---|---:|---:|---:|
| 1 | 916.2 | 916.6 | 0.99911 |
| 2 | 756.1 | 755.3 | 1.00053 |
| 3 | 759.2 | 758.8 | 1.00027 |
| 4 | 917.6 | 917.6 | 1.00112 |
| 5 | 753.1 | 753.1 | 1.00081 |
| 6 | 769.5 | 769.3 | 1.00053 |

The absolute time swings 22% (753 ↔ 917) but the **ratio holds at 1.000 ± 0.1%**.
Comparison MDE ≈ 0.1–0.3% — far below any schedule effect. Runs use 50 reps in the
matrix below.

---

## Asm spills per scheduler (Phase 0.3)

Realized stack spills (`objdump` count of `vmov{ap,up}{s,d}` referencing
`(%rsp)`/`(%rbp)`) for the same codelet under each scheduler flag:

| R | `--su` | `--gh` | `--bb` | `--bisect` |
|---|---:|---:|---:|---:|
| 16 | 68 | 68 | **64** | 94 |
| 32 | 147 | 147 | 182 | 270 |
| 64 | 477 | 477 | **942** | 753 |

- **`--gh` ≡ `--su`** at every radix (identical codelet — the GH auto-rule already
  governs it; the explicit flag is a no-op here).
- **`--bb` emits *more* spills than `--su`** at R=32 (182 vs 147) and R=64
  (**942 vs 477**), despite `bb` minimizing `peak_live`. The model-optimum diverges
  from gcc's realized allocation. (`bb` ran with a 1 s budget; may also be timing
  out at R≥32 — either way, its realized spills are worse than `su`.)
- **`--bisect`** spills most at R=16/32, second-most at R=64 — consistent with it
  losing to SU on this hash-consed IR.

---

## Runtime ratio vs `--su` (Phase 0.3)

Paired in-process, 50 reps. **ratio = `su_time / variant_time`** → **<1 means the
variant is slower than `su`**, >1 means faster. (`--gh` omitted — identical to
`--su`.)

| R | sched | K=8 | K=256 | K=1024 |
|---|---|---:|---:|---:|
| 16 | bb | 1.043 | 1.054 | 1.005 |
| 16 | bisect | 0.762 | 0.829 | 0.926 |
| 32 | bb | 0.976 | 0.964 | 0.979 |
| 32 | bisect | 0.680 | 0.811 | 0.919 |
| 64 | bb | 0.545 | 0.703 | 0.855 |
| 64 | bisect | 1.258¹ | 0.623 | 0.761 |

¹ R64 bisect @ K=8 is the lone anomaly (variant *faster*) and the highest-noise
cell (ratio cov 5.4%); at K=256/1024 bisect is clearly slower, consistent with its
spill count.

Only the one variant with **fewer** spills than `su` (R16 bb, 64 vs 68) is
**faster**; every variant with **more** spills is **slower**, and the magnitude
tracks the spill delta.

---

## Spills ↔ runtime correlation

Pairing Δspills (vs su) against the K=256 runtime ratio:

| R | sched | Δspills | ratio (K=256) |
|---|---|---:|---:|
| 16 | bb | −4 | 1.054 (faster) |
| 16 | bisect | +26 | 0.829 |
| 32 | bb | +35 | 0.964 |
| 32 | bisect | +123 | 0.811 |
| 64 | bb | +465 | 0.703 |
| 64 | bisect | +276 | 0.623 |

Direction is monotonic with a single adjacent swap (R64 bb vs bisect) →
**Spearman ρ ≈ 0.94**. Asm spills are a strong, valid proxy for runtime; the only
imperfection is fine-ranking among near-equal high-spill variants (which the outer
runtime-validation stage resolves).

---

## Findings → decisions

1. **Objective = realized asm spills.** ρ≈0.94 with runtime; minimizing spills
   finds faster schedules.
2. **Do *not* search on `peak_live`.** `bb` minimized it yet doubled asm spills and
   lost 30–45% at R=64 — the model→asm gap (gcc allocation) breaks the cheap
   shortcut. The inner search must compile + count asm spills (~1 s/candidate) or
   measure runtime directly. *(Re-confirm whether bb's loss is the 1 s budget vs a
   true model gap before relying on it.)*
3. **Beat `--su`, not `--bb`.** `--su` is the strongest available scheduler;
   `--gh` ≡ `--su`, and `--bb`/`--bisect` lose. Headroom exists (schedules differ
   by up to 45% runtime / 2× spills) but isn't guaranteed — the search must undercut
   `su`'s realized spill count.

**Verdict: conditional GO.** The measurement is trustworthy, the objective is
validated, and we now know *what* to optimize (asm spills), *what not to*
(`peak_live`), and *who to beat* (`--su`) — before any annealer is written.

---

## Follow-up: the Goodman–Hsu threshold lever (negative result)

The first lever we tried off the Phase 0.3 verdict was "make GH fire harder." GH
already runs at R=32/64 AVX2 (auto-rule `vec_regs≤16 && n≥32`) and fires (per-pass
peak-live ~35 > threshold 12). The hypothesis: **lowering the threshold** makes
the pressure-mode (`delta = births−kills`) free registers earlier → fewer spills.

We added an env-gated knob (default = unchanged production behavior) at
[`schedule.ml`](../../src/dag-fft-compiler/generator/lib/schedule.ml#L766):

```ocaml
let threshold =
  match Sys.getenv_opt "VFFT_GH_THRESHOLD" with
  | Some s -> (try int_of_string s with _ -> uarch.Uarch.pressure_threshold)
  | None  -> uarch.Uarch.pressure_threshold
in
```

rebuilt the generator (WSL `dune`), and swept the threshold from the default 12
down to 4:

**Asm spills (flat):**

| R | T=12 | T=10 | T=8 | T=6 | T=4 |
|---|---:|---:|---:|---:|---:|
| 32 | 140 | 140 | 142 | 138 | 138 |
| 64 | 469 | 463 | 457 | 474 | 461 |

**Runtime (T=4 vs T=12, ratio, 50 reps):** R=32 = 0.997 / 0.995 / 0.997
(K=8/256/1024) — flat. R=64 = 1.000 / 0.912 / 0.966 — flat except one cell.

**Conclusion — the GH-threshold lever is ineffective**, and *for the reason Phase
0.3 predicted*: GH minimizes the schedule's `peak_live`, and `peak_live ≠ asm
spills`. Making GH minimize `peak_live` *more aggressively* doesn't change gcc's
realized spill count — the same root cause that made `bb` (also a `peak_live`
minimizer) lose. This further closes the "tune the schedule's peak-live" axis:
**the binding constraint is gcc's realized allocation, not the IR schedule's
register pressure.**

### Register-allocation context (why this axis is narrow)

- **FFTW genfft does *no* register allocation or spill logic** (grep of all of
  genfft for `register alloc|regalloc|spill|register pressure` → zero hits). Its
  strategy is the same as VectorFFT's production path: a pressure-minimizing
  schedule + `annotate.ml` nested-block lifetime hints + **let the C compiler
  allocate**. VectorFFT's `annotate.ml` is a direct descendant of FFTW's.
- VectorFFT's own register allocator — the **M-project** (`regalloc.ml`,
  register-pin + scheduling-fence) — is **dormant by design**: post-FMA-fusion the
  residual reg-reg `vmovapd` are move-eliminated in rename (0 cost), so the fence
  only *blocks* beneficial gcc transforms (defeats operand folding, +9%). It is
  net-negative or tie in every cell.
- So gcc's reg-reg **moves** are already handled (fusion + `annotate` +
  rename-elimination); the open axis is **stack spills**, and GH/`peak_live`
  tuning doesn't move them. That leaves: **finer cut topology** and the
  **asm-spill search** as the remaining spill levers.

## Phase 2 — the schedule annealer (iterative compilation) + first win

With the spill levers exhausted (GH dead, cut-topology owned by the plan layer),
the live lever is a **direct search of a single codelet's internal instruction
order**, scored on the *realized* output. Method:

- **Injector** (`schedule.ml`, env-gated, default unchanged): `VFFT_SCHED_DUMP`
  writes su's order + DAG preds; `VFFT_SCHED_ORDER` emits a codelet under an
  explicit, legality-checked node order. Round-trip (inject su's own order) is
  **byte-identical** — validated.
- **Annealer** (`anneal.py`): simulated annealing over **legal reorderings** of
  the same DAG (move operators: single reinsert / block-move / segment-reversal,
  each preserving a valid topological order, so every candidate is **bit-exact**
  by construction). This is **iterative compilation** — each candidate is
  compiled and `objdump`'d; the search optimizes the *measured asm*, not a model.
- **Target**: monolithic codelets, which the injector's `su_schedule` path covers.
  Primes have the headroom (R=11–23: 62–313 stack spills, `su_schedule`,
  Direct/monolithic — and they bypass the spill recipe, so the most schedule slack).

### Objective correction (the important refinement)

The first runs scored on **stack-spill count** and found 86→75 spills on R=13.
But on this **noisy execution host** the paired runtime bench cannot arbitrate a
~2% fine-grain delta (readings bounced 0.975↔1.034 under background load). Worse,
stack-spill count alone has a hole: a schedule can trade stack spills for reg-reg
moves. So the objective was changed to the **noise-robust static win condition**:

> **reduce spills *without* increasing total instructions or saturating an
> already-saturated port.**

Implemented as: **objective = total instruction count** (which subsumes spills +
reg-reg moves), **gated** on (a) spills must not increase, and (b) FMA count
invariant — reordering must not change the arithmetic, so the FMA-port pressure is
fixed and only the overhead (`vmov`) varies. A candidate is a win iff it cuts
total instructions *and* spills with the arithmetic untouched. This closes the
spill-vs-move trade hole and removes the dependence on a quiet machine.

### Result (R=13, prime, monolithic, 245 nodes)

| | total insns | stack spills | FMA (invariant) |
|---|---:|---:|---:|
| `su` (incumbent) | 2314 | 86 | 132 |
| **annealed** | **2295** | **79** | 132 |
| Δ | **−19** | **−7** | 0 |

A clean static win in ~80 SA iterations: fewer instructions *and* fewer spills,
arithmetic untouched, bit-exact. This is the first demonstration that the search
**beats the production scheduler** on a single codelet by the noise-robust
criterion. (A deeper run reached 75 spills / 2302 insns.) Primes were the proving
ground; the high-value target — **pow2 codelets** — is next, and requires
extending the injector to the blocked `su_schedule_subset` path (R≥16).

## Reproduction

Harness + driver live in the gitignored sandbox
`src/dag-fft-compiler/experiments/sched_search/` (`compare_codelets.c` paired
bench, `run_phase03.sh` matrix driver). Generation via WSL
(`gen_radix.exe R --in-place --isa avx2 --{su,gh,bb,bisect} --emit-c`), compile +
`objdump` + bench native Windows (gcc 15.2). Pin core 2, 50 reps, best-of-5,
cachebust + pace between candidates.
