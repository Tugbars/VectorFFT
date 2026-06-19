# `core/primes/` — prime-size FFT (Rader + Bluestein)

The CT engine factors `N` into the radix set; a **prime N** (or any N with no smooth
factorization) has nothing to factor, so `auto_plan` returns NULL. This folder fills that
hole by **reducing a prime-N DFT to a convolution** — which is just two ordinary FFTs — so
prime sizes **ride the entire CT engine** (planner, wisdom, codelets, JIT) for their inner
work. No prime-specific FFT kernels exist; the reduction *is* the trick.

Two algorithms, picked by the shape of `N`:

| algorithm | when | reduces N-DFT to | inner FFT size | cost |
|-----------|------|------------------|:--------------:|------|
| **Rader** | prime N with **N−1 radix-smooth** (19-smooth) | cyclic convolution of length **N−1** | N−1 | ~2× faster — needs **no wisdom** |
| **Bluestein** | any other N (non-smooth N−1) | circular convolution of size **M ≥ 2N−1** | M (free, padded to smooth) | needs **(M,B) wisdom** |

Rader wins when it applies because its convolution is `N−1` (vs Bluestein's `~2N`), so the
two inner FFTs are roughly half the size.

---

## The dispatch (`prime_dispatch.h`)

`vfft_proto_auto_plan_dispatch(N, K, reg, wis)` is the prime-aware front door:

1. Try `vfft_proto_auto_plan` — factorable N → CT / wisdom, return it.
2. Non-prime + unfactorable → NULL.
3. Prime, **N−1 radix-smooth** → **Rader** (`M = N−1` fixed, `B` heuristic).
4. Otherwise → **Bluestein** — `M` from the Bluestein wisdom if the caller set one
   (`vfft_proto_dispatch_set_bluestein_wisdom`), else the `_bluestein_choose_m` heuristic.

**The elegant part:** both Rader and Bluestein build their **inner FFT through
`vfft_proto_auto_plan`** — so the inner (size N−1 or M) *rides the CT wisdom and codelets
like any other transform*. A prime cell is "a CT FFT of size M, wrapped in chirp/gather
passes." That's why the primes path needs no codelet tree of its own.

**Execution** is via the plan-level **override** mechanism (same as DCT/2D/OOP-MODEB): the
Rader/Bluestein plan sets `override_fwd/bwd/destroy`, which `vfft_proto_execute_fwd` and the
compat `stride_execute_fwd` both honor, and `plan_destroy` frees via `override_destroy`.

> **Include-order gotcha:** `prime_dispatch.h` sits *above* `planner.h` and pulls
> `proto_stride_compat.h` (the bridge that supplies the stride API — thread pool,
> `STRIDE_ALIGNED_ALLOC`, `stride_*` names — that `rader.h`/`bluestein.h` are written
> against). That bridge must load *after* `planner.h` and *before* rader/bluestein, which
> is exactly why this dispatch can't live inside `planner.h`.

---

## Rader (`rader.h`)

For prime N with smooth `N−1`, the N-point DFT becomes a length-`(N−1)` cyclic convolution,
indexed by a **primitive root (generator)** of the multiplicative group mod N:

```
1. DC sum                      O(NK)
2. gather by generator         O(NK)   scattered read (permutation by g^k mod N)
3. forward FFT of size N−1     (the CT engine — rides wisdom)
4. pointwise multiply          O(NK)   flat AVX2, by the pre-transformed kernel
5. inverse FFT of size N−1     (the CT engine)
6. scatter + DC add            O(NK)   scattered write
```

**Block-walk for large K**: process K in `B`-lane chunks so the `(N−1)·B` scratch fits L2
(not `(N−1)·K`). AVX2 pointwise multiply; pre-expanded kernel for flat SIMD. Memory:
`2(N−1)` permutation + `4(N−1)B` kernel + `2(N−1)B` scratch. `n_threads` is snapshotted at
plan-create (scratch sized for the pool).

Rader needs **no wisdom**: `M = N−1` is forced, `B` is the cache-blocking heuristic, and the
inner FFT's own factorization/variants come from the CT wisdom for size `N−1`.

---

## Bluestein (`bluestein.h`)

For any N (the general fallback), embed the N-point DFT in a length-`M` circular convolution
(`M ≥ 2N−1`, M chosen smooth so the CT engine can do it):

```
1. chirp modulation            O(NK)   AVX2 complex multiply
2. forward FFT of size M       (the CT engine)
3. pointwise multiply          O(NK)   by the precomputed kernel
4. inverse FFT of size M       (the CT engine)
5. chirp demodulation          O(NK)   AVX2
```

**M is a free parameter** (any smooth composite ≥ 2N−1), and **the choice matters a lot** —
this is the whole reason Bluestein needs wisdom (Rader doesn't). `_bluestein_choose_m`
searches `[2N−1, ~1.12·(2N−1)]` (always including the next power of two) and picks the
**fewest-stage** M — but stage-count is **blind to per-codelet quality**:

> At **N=179**, the heuristic picks `M=361=19²` (2 stages of radix-19), but `M=384=64·6` is
> **4.65× faster** — same 2 stages, but radix-64 vastly outperforms radix-19 (register
> pressure, retiring efficiency). For N=107 the gap is 1.14×. So the heuristic's
> stage-count proxy can be 4–5× off; measured `(M,B)` wisdom fixes it.

**Block-walk** (`_bluestein_block_size`): cap `B` so the `2·M·B·8`-byte scratch fits ~1 MB
of L2. Memory: `2N` chirp + `4MB` kernels + `2MB` scratch, all pre-allocated at plan time
(not per-call like FFTW).

**Performance reality:** profiling N=509 K=256 (M=1024, B=64) shows the **inner FFTs are
77–85% of the time**; chirp/pointwise are 7–9% each. So Bluestein's speed *is* the inner
FFT's speed — it currently runs ~0.68× MKL at N=509 K=256, and the lever is faster/better-M
inner FFTs (e.g. composite M with our strong non-pow2 codelets), not chirp micro-opt.

---

## Bluestein wisdom (`bluestein_wisdom.h`) + calibrator (`bluestein_calibrator.h`)

Because M and B are a *separate* search space from CT factorization, primes get their **own
wisdom file** (the CT calibrator falls through to NULL on prime cells — no smooth factors to
search):

```
@bluestein_version 1
# N    K    M    B    best_ns
47   256  95   64   53234.0
179  256  384  16   209911.0
```

An entry fixes only `(M, B)`; the inner FFT of size M still rides the **CT wisdom**
(`spike_wisdom`) for its factorization + variants. A lookup miss → the heuristic (zero risk
to un-benched cells).

The **calibrator** (`bluestein_calibrator_one`) sweeps factorable `M ∈ [2N−1, 4N]`
(Bluestein) or fixed `M = N−1` (Rader) × `B ∈ {16,32,64,128,256,4,8}`, builds + times each
plan (min-of-N trials), and records the lowest-measured `(M,B)`. It's the prime-N analog of
the CT MEASURE sweep — called on a prime-N MEASURE miss by both the dev tool and the
public-API `_calibrate_one` path (and the orchestrator's `_vfft_proto_sweep_prime`).

---

## JIT (when `VFFT_USE_JIT`)

Prime plans execute via override, but the **heavy inner CT FFT can be JIT-specialized** both
directions — Rader runs its inner forward *and* backward, so `vfft_proto_plan_jit_{fwd,bwd}`
are wired into the inner via `stride_rader_set_inner_jit` / `stride_bluestein_set_inner_jit`
(see `planning/plan_orchestrator.h`). The chirp/gather wrappers stay generic; only the
dominant inner gets the JIT'd direct-call path.

---

## Files

| file | role |
|------|------|
| `prime_dispatch.h` | the prime-aware front door (`auto_plan_dispatch`): CT → Rader → Bluestein routing |
| `rader.h` | Rader's algorithm (smooth N−1 → length-(N−1) cyclic convolution) |
| `bluestein.h` | Bluestein's algorithm (any N → length-M circular convolution) + M/B heuristics |
| `bluestein_wisdom.h` | the separate `(N,K) → (M,B)` wisdom file (load/lookup/save) |
| `bluestein_calibrator.h` | the (M,B) sweep that populates that wisdom on a prime-N miss |

## Gotchas

- **The inner FFT is the cost** (77–85% for Bluestein). Prime speed = CT-engine speed at
  size M/(N−1); chirp/gather overhead is second-order.
- **Bluestein's M is codelet-quality-sensitive** — the fewest-stage heuristic can be 4–5×
  off (N=179: 361 vs 384). Calibrate `(M,B)` for benched cells; heuristic for the rest.
- **Rader needs no wisdom** (M = N−1 fixed); only Bluestein does.
- **Override execution** — primes are `override_fwd/bwd` plans (like DCT/2D); `slice_K` is
  moot, they handle the full transform + any internal block-walk threading themselves.
- **Bridge include order** — `prime_dispatch.h` must pull `proto_stride_compat.h` between
  `planner.h` and `rader.h`/`bluestein.h` (they use the stride API).

See also: `core/planning/plan_orchestrator.h` (the prime sweep-on-miss + JIT wiring),
`core/engine/README.md` (the CT engine the inner FFT rides), `bluestein.h` header (the full
performance analysis + optimization leads).
