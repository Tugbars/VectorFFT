# `transforms/real/` — real-input FFTs (r2c / c2r)

Real-to-complex (`r2c`, forward) and complex-to-real (`c2r`, backward) transforms.
Unnormalized, so `c2r(r2c(x)) = N·x`. This folder holds the **hand-maintained engine
and dispatch headers**; the per-radix codelet *registries* are generated artifacts and
live in `generator/generated/` (pulled in via the build's `-I`, never co-located here).

---

## 0. Design thesis — throughput over single-core latency

VectorFFT's real-FFT path is built around a **split, lane-batched layout**: the complex
spectrum is two separate planes `out_re[]` / `out_im[]`, and a batch of `K` transforms is
laid out lane-interleaved as `data[n*K + lane]`. This is a *deliberate* choice with a
clear trade:

- **It costs us on a single core.** Split layout forces a real→complex **pack** pass
  (de-interleave), and the Hermitian recombine is a second full-plane memory pass. Together
  these "dumb memory passes" are the entire gap to MKL at high K (see §4). On one thread the
  decoupled path lands ~0.81–0.91× MKL.
- **It wins decisively under multithreading.** The same wide-K-batch layout makes the batch
  *embarrassingly parallel* — split the `K` lanes across the pool and every thread runs a
  disjoint, cache-line-aligned, contiguous slice with **no barriers, no transpose, no
  cross-thread movement.** dag scales 2.9–4.8× across 8 P-cores while MKL's batched small-N
  r2c barely threads; net, **dag beats MKL 2.56–4.77× at T8** (see §5).

So the architecture is tuned for the regime that dominates real workloads — **high batch ×
many cores** — and accepts a single-core packing tax as the price of a layout that
parallelizes perfectly. If you only ever run one transform on one core, MKL's interleaved
in-place r2c is the better tool; everywhere else, throughput wins.

Full measured analysis: [`docs/performance/high_k_real_fft_architecture_wall.md`](../../../../docs/performance/high_k_real_fft_architecture_wall.md).

---

## 1. Two engines, two layouts

There are **two independent r2c implementations**, chosen per-plan by a threshold (§2):

| | **rfft** (native) | **decoupled / stride** |
|---|---|---|
| File | `rfft.h` (fwd) + `c2r.h` (bwd) | `r2c.h` (`stride_r2c_plan` / `stride_execute_r2c` / `stride_execute_c2r`) |
| Method | FFTW-style real mixed-radix: r2cf leaf + hc2hc twiddle stages. **No pack stage.** | pack(real→complex N/2) + c2c(N/2) + Hermitian fold |
| Output layout | **packed** halfcomplex (one N×K plane) — or **split** via its natural terminator | **split** only (`out_re` / `out_im`) |
| Wins at | **low K** (≤16): L1-resident, compute-bound | **high K** (≥32): the K-batch parallelizes |
| Covers | even N over the rfft radix set | even N (inner c2c of N/2); also the universal fallback for odd/prime/uncovered N |

The native rfft cascade is ~1.5–1.7× faster than the decoupled path at *low* K, but collapses
to ~0.50× MKL at K=256 — the decoupled method is ~2× better than our own rfft at high K. The
dispatcher exists to pick the right one.

---

## 2. Dispatch & the K-threshold (`r2c_dispatch.h`)

`vfft_r2c_plan_create(N, K, layout, rfft_reg, have, c2c_reg)` returns a unified
`vfft_r2c_plan_t` (exactly one of `{rfft, stride}` is non-NULL; `path` field records which).
Routing, in order:

```
1. HYBRID stride-first:  layout==SPLIT && N even && K >= decouple_min_k && c2c_reg
                         → build STRIDE, take it if it builds
2. rfft (primary):       wisdom-first factorization, else the fewest-stage heuristic
3. stride fallback:      if not PACKED → STRIDE  (odd N, primes, rfft-uncovered)
4. else → NULL
```

The knob is **`_vfft_r2c_decouple_min_k`, default 32** — the measured N=256 crossover, settable
at runtime via `vfft_r2c_dispatch_set_decouple_min_k()`. Below it you get rfft; at/above it the
decoupled stride path. `SIZE_MAX` disables the hybrid (always rfft). A **PACKED** request always
means rfft (stride cannot pack → NULL if no rfft registry).

### The crossover (live, N=256, single-thread, ratio = MKL_ns / ours, >1 beats MKL)

| K | 8 | 16 | **32** | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| **rfft** | 1.39 | 1.05 | 0.60 | 0.62 | 0.58 | 0.50 |
| **stride** | 0.91 | 0.91 | **0.97** | 0.67 | 0.66 | 0.92 |
| dispatcher picks | rfft | rfft | **STRIDE** | STRIDE | STRIDE | STRIDE |

The crossover is sharp at **K=32**: rfft drops to 0.60× while stride holds 0.97×. K=64–128 is the
weakest regime (both ~0.66×). Reproduce with `bench_r2c_dispatch_vs_mkl.c`.
**The threshold is N=256-specific — recalibrate per host/N** (see the doc's §7 TODO).

---

## 3. r2c ↔ c2r layout pairing

There are **two c2r mechanisms, paired by layout** — do not cross them:

- **packed**: `c2r.h` `c2r_execute_packed` inverts the **packed** rfft forward
  (`rfft_execute_fwd_packed`). One contiguous buffer (im at `+N·K`). Dispatched via
  `c2r_dispatch.h` (wisdom-first; mirror of r2c minus the rfft/stride routing).
- **split**: `r2c.h` `stride_execute_c2r(plan, in_re, in_im, real_out)` is the **stride r2c
  plan's backward** — it inverts the split stride forward (`c2r(r2c(x)) = N·x`).

The unified `vfft` API (`vfft_create` with `VFFT_C2R`) uses the **split-stride pair**: it forces
the stride path (`decouple_min_k=0` during create) so the split backward exists, and it round-trips
the split `vfft` r2c for K≥32. Low-K split c2r (where the rfft regime is packed) is the documented gap.

---

## 4. The split-layout pack tax (single core)

K=256 decoupled breakdown — the two memory passes, not the FFT, are the gap:

| stage | cost | % | note |
|---|---|---|---|
| pack | ~11.6µs | **28%** | de-interleave real `x` → split-complex `z` |
| inner c2c FFT | ~17.4µs | 42% | the actual transform — *faster than MKL's* |
| Hermitian recombine | ~12.6µs | 30% | bandwidth-bound (AVX2 == scalar) |
| **total** | ~41.7µs | | 0.81× MKL |

pack + recombine ≈ **58%** of single-core time. The recombine is 100% memory-bandwidth-bound
(~83 GB/s, at the floor) — vectorizing it does nothing.

**OOP buys back the pack.** An out-of-place leaf codelet (`r4_n1_oop_avx2`) has separate in/out
strides: stage-0 reads `x` directly (load stride 2K) and writes packed `z` — no separate pack pass.
Order-neutralized A/B at K=256: **in-place (with pack) 0.79× → OOP (no pack) 0.91×** (+15%, ≈5.8µs).
OOP is the *intended* high-K r2c forward path; generalizing it to all leaf radixes is the open
follow-up (today the OOP win only fires where the inner stage-0 leaf is radix-4). See doc §3a/§6.

---

## 5. Multithreading — the wall reverses

Everything in §4 is single-thread. Under MT the picture flips. The dispatcher threads r2c by
splitting the K-batch: `_vfft_r2c_block_k(K)` picks the largest multiple-of-8 divisor of K that is
≤ K/T (must divide K — a partial block over-reads; must be a mult of 8 for the lane group), giving
~T full blocks. **You must `stride_set_num_threads()` BEFORE `vfft_r2c_plan_create`** — T is
snapshotted for scratch sizing and block choice.

dag (8 P-cores, caller pinned core 0) vs MKL at 8 threads, split r2c:

| N | dag scaling | MKL scaling | **T8 dag vs MKL** |
|---|---|---|---|
| 256 | 2.9× | 0.73× (T8 *slower* than T1) | **3.51×** |
| 1024 | 4.8× | 0.87× | **3.58×** |
| 8192 | 3.4× | 0.96× (DRAM-bound) | **2.56×** |

The dispatcher MT bench (`bench_r2c_dispatch_mt_vs_mkl.c`) shows **dag beats MKL 2.85–4.77× @T8**
across K=256..8192, MT==T1 exact. Two effects compound: our split lane-batch is the cleanest
possible MT decomposition, and MKL's batched small-N r2c threading is weak. Full table: doc §6c.

---

## 6. Wisdom — three distinct tables

The two engines and their inners calibrate separately:

- **rfft wisdom** (`rfft_wisdom.txt`) — per-cell factorization + per-stage variant for the rfft path.
  Set via `vfft_r2c_dispatch_set_wisdom()`. On a hit it pins factors+variant; else the fewest-stage
  heuristic.
- **c2c inner wisdom** (`spike_wisdom.txt`) — for the decoupled path's **inner complex FFT of N/2**.
  Set via `vfft_r2c_dispatch_set_c2c_wisdom()`. Without it the inner falls back to the factorizer
  default (often degenerate, e.g. (64,2)).
- **c2r wisdom** — separate optima from r2c (calibrate independently); `c2r_dispatch.h`.

---

## 7. Gotchas

- **radix-32 is leaf-only** (no `hc2hc[32]`), so it is **excluded from the stage chooser's
  `default_have`** — including it makes the greedy chooser pick 32 as a *stage* and fail. Leaf-32
  plans (e.g. the doc-60 (8,32) winner) come from **wisdom only**, not the heuristic.
- **stride inner is force-DIT** — pack-fusion is a DIT-leaf technique; if c2c wisdom picks a DIF
  inner it is rebuilt as the same factorization in DIT. DIF inners are *correct* (a safety net) but
  slower here: N=256 K=32 DIT+fused 0.99× vs DIF+explicit-pack 0.87×.
- **MT needs sub-K blocks** — the default `block_K = K` is a single block = serial. The dispatcher
  picks `block_K < K` only when `stride_get_num_threads() > 1` at plan-create.
- **packed buffer sizing** — the packed path allocates `2·N·K` (base plane + the N·K fold region).

---

## 8. File map

| file | role |
|---|---|
| `rfft.h` | native real mixed-radix FFT (r2hc), forward, packed output (+ natural-split terminator) |
| `c2r.h` | packed backward real (halfcomplex → real), the inverse of `rfft.h` |
| `r2c.h` | the decoupled/stride engine: `stride_r2c_plan`, `stride_execute_r2c` (fwd, split), `stride_execute_c2r` (bwd, split), in-place variant |
| `r2c_dispatch.h` | top-level r2c entry — rfft-vs-stride routing, the K-threshold, MT block sizing, the rfft factor chooser |
| `c2r_dispatch.h` | wisdom-first c2r entry (packed path) |

Codelet registries (`rfft_registry_{avx2,avx512}.h`, `c2r_registry_{avx2,avx512}.h`) are
**auto-generated** by the dag pipeline (`emit_rfft_registry.ml` / `emit_c2r_registry.ml`) and live in
`generator/generated/`. They are reached via `-I`, not stored in this folder. Do not hand-edit.
