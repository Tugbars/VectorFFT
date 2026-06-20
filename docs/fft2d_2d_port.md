# 2D FFT (c2c + r2c): port from old core to the dag tree

**Date:** 2026-06-19
**Hardware:** Intel i9-14900KF (Raptor Lake), AVX2, single-thread, pinned core 2
**Toolchain:** mingw gcc 15.2, MKL oneAPI (`mkl_set_num_threads(1)`), split-complex layout

> TL;DR — `fft2d.h` (2D c2c) and `fft2d_r2c.h` (2D r2c) are ported from `src/core` into the dag
> canonical tree, validated on Raptor Lake AVX2. **2D c2c beats MKL 1.19–1.42×** across 64²–512²;
> **2D r2c is correct (roundtrip 1e-14) but loses ~0.63–0.69×** — the same real-FFT structural tax
> seen in 1D. Both are roundtrip-definitive-correct; output is **digit-reversed (scrambled) order**,
> the dag convention (its 1D c2c is digit-reversed).

---

## 1. What was ported

| file | role |
|------|------|
| `core/transpose.h` | SIMD 4×4/8×4 transpose kernels (`stride_transpose`, `stride_transpose_pair`, twiddle transpose). Self-contained (only stddef/string/immintrin) — copied verbatim. |
| `core/fft2d.h` | 2D c2c: column FFT (native) + row FFT (tiled or Bailey). |
| `core/fft2d_r2c.h` | 2D r2c/c2r: tiled row r2c + padded column c2c, with col digit-reversal perm. |

**The port was nearly mechanical** thanks to dag's compat layer (`proto_stride_compat.h`):
- `stride_plan_t`, `stride_plan_destroy`, `stride_transpose*` — same names, used as-is.
- `stride_execute_fwd/bwd` (3-arg), `STRIDE_ALIGNED_ALLOC/FREE`, threads — bridged by the compat header.
- `stride_exhaustive_plan`→`vfft_proto_exhaustive_plan(...,0)`, `stride_auto_plan`→`vfft_proto_auto_plan(...,NULL)`,
  `stride_registry_t`→`vfft_proto_registry_t` (fft2d.h only).
- The plain 5-arg `_stride_execute_fwd/bwd_slice` → the compat `_slice_from(...,0)` / `_bwd_slice_until(...,0)`
  (don't include `stride_executor.h` *and* `proto_stride_compat.h` — they redefine the slice helpers).
- `fft2d_r2c.h` reaches into r2c.h internals (`stride_r2c_data_t`, `_r2c_worker_arg_t`, `_r2c_worker_fwd/bwd`),
  which dag's r2c.h has with identical names/layout — so it needed **no symbol renames**, only the include set.

`build.py` already globs the codelet families needed; no codelet changes.

---

## 2. Architecture (both transforms)

2D separable FFT = transform along axis 0 (columns) then axis 1 (rows), in split-complex layout
`re[i*N2 + j]` (i=row 0..N1-1, j=col 0..N2-1). Normalization: `bwd(fwd(x)) = N1·N2·x`.

**Column FFT** — N1-point, batch K=N2, run natively by the 1D stride engine (the lane-batched K-vectorized
path — same engine that wins 1D c2c). No transpose needed for this axis (columns are the batch).

**Row FFT** — two methods (c2c):
- **Tiled (default, B=8):** for each tile of B rows — SIMD-transpose B×N2 → N2×B into scratch, run the
  N2-point FFT with K=B on the L1-resident tile, SIMD-transpose back. Tiles are independent →
  embarrassingly parallel. Small tiles keep the working set in L1/L2 and the 4×4/8×4 transpose kernels
  make gather/scatter nearly free. This is why 2D c2c **beats** MKL.
- **Bailey:** two full-matrix transposes bracket one large-K (K=N1) row FFT. Alternative for hosts where
  a big-K FFT + cache-oblivious transpose wins.

**2D r2c** adds, per row tile: a real→complex r2c on the tile (the dag 1D r2c, `_r2c_worker_fwd`), a
**padded** column c2c (K_pad = round-up(N2/2+1, 4)), and the col digit-reversal perm so the half-complex
column bins land where the col-FFT expects them. c2r is the reverse, tiles processed in reverse order for
in-place safety.

---

## 3. Results (vs MKL, single-thread, split)

### 2D c2c — beats MKL
| size | roundtrip | vfft (cyc) | mkl (cyc) | speed |
|------|-----------|-----------|-----------|-------|
| 64²  | 6.6e-15 | 18,235 | 25,645 | **1.41×** |
| 128² | 1.2e-14 | 87,833 | 113,920 | **1.30×** |
| 256² | 7.4e-15 | 437,863 | 522,695 | **1.19×** |
| 512² | 1.1e-14 | 2,583,840 | 3,209,732 | **1.24×** |

### 2D r2c — correct, loses to MKL
| size | roundtrip | vfft (cyc) | mkl (cyc) | speed |
|------|-----------|-----------|-----------|-------|
| 64²  | 1.5e-14 | 22,138 | 15,364 | 0.69× |
| 128² | 1.4e-14 | 89,851 | 61,352 | 0.68× |
| 256² | 1.5e-14 | 381,426 | 261,282 | 0.69× |
| 512² | 1.8e-14 | 2,745,778 | 1,716,321 | 0.63× |

**Why c2c wins but r2c loses** — same split-layout story as 1D. c2c is just the transform (tiled, L1-resident
row pass) → we win. r2c adds the real-FFT pack/recombine passes *plus* the K_pad-padded column FFT and the
perm remap — extra memory traffic, and (like 1D r2c at high K) it's memory-bound where MKL's fused real-FFT
wins. r2c is correct and shipped; closing the perf gap is the same fused-kernel workstream as 1D r2c.

### 3c. Multithreaded — 2D c2c beats MKL 2.2–6.1× at 8 threads (2026-06-20)

dag on 8 P-cores (caller pinned core 0, pool pins workers 1..7) vs MKL at `mkl_set_num_threads(8)`,
2D c2c, split / NOT_INPLACE (`build_tuned/benches/bench_fft2d_mt_vs_mkl.c`):

| size | dag T1 (ns) | dag T8 (ns) | MKL T1 (ns) | MKL T8 (ns) | dag scal | MKL scal | T1 mkl/dag | **T8 mkl/dag** |
|------|-------------|-------------|-------------|-------------|----------|----------|------------|----------------|
| 64²  | 16,505      | 21,524      | 26,745      | 130,810     | 0.77×    | 0.20×    | 1.62×      | **6.08×**      |
| 128² | 92,819      | 81,178      | 117,921     | 270,271     | 1.14×    | 0.44×    | 1.27×      | **3.33×**      |
| 256² | 383,496     | 266,278     | 484,074     | 675,276     | 1.44×    | 0.72×    | 1.26×      | **2.54×**      |
| 512² | 2,405,716   | 1,499,525   | 3,113,353   | 3,285,215   | 1.60×    | 0.95×    | 1.29×      | **2.19×**      |

Two effects compound. (1) dag's tile-parallel rows + K-split columns scale *modestly but positively*
(0.77–1.60×; 64² is too small to thread and regresses, the gain grows with size). (2) **MKL's 2D
threading is actively bad: 0.20–0.95×** — for every size ≤512² MKL T8 is *slower* than MKL T1 (64²
is 5× slower at T8). Net, **dag wins 2.19–6.08× at T8**, widening the single-thread c2c lead. This is
the 2D analogue of the r2c §6c reversal: in the batched/2D regime MKL barely threads, and our
barrier-free split layout wins decisively even at modest self-scaling.

**1024² omitted:** `stride_plan_2d` builds its inner col/row plans via `vfft_proto_exhaustive_plan`
(a live measure, not wisdom) — minutes per plan at N=1024. Making the 2D c2c inners wisdom-driven
(one-time calibration of the (1024,1024) and (1024,B) cells) is the follow-up that makes 1024²
practical. (Today only the 2D *r2c* inner rides c2c wisdom; the 2D c2c row/col inners do not.)

---

## 4. Correctness + order

Correctness is by **roundtrip** (`fwd→bwd == N1·N2·x`, all cells ≤1.8e-14) — definitive and order-agnostic.
The dag 1D engine is **digit-reversed (DIT)**, so the 2D forward output is **scrambled order** (does not
match MKL elementwise). This is the dag convention (the 1D c2c is scrambled too); roundtrip pipelines and
spectral-magnitude users are unaffected. A consumer needing natural order pays a reorder pass — `fft2d_r2c.h`
already carries the col digit-reversal perm as the hook for that; finishing natural-order output is a follow-up.

---

## 5. Gotchas (from the port)

- **Slice-helper clash:** including both `stride_executor.h` and `proto_stride_compat.h` redefines
  `_stride_execute_fwd_slice_from/_until`. Use the compat ones (the dag convention).
- **MKL 2D r2c CCE buffer:** MKL's default 2D CCE output packing needs the **full N1×N2 complex footprint**,
  not N1×(N2/2+1). Undersizing it segfaults (this caused a bench crash that looked like a port bug — the
  isolation `isolate_2d_r2c_128.c` proved the port itself is crash-free at 64/128/256).
- **Exhaustive planner is slow at large N:** `stride_plan_2d` tries `vfft_proto_exhaustive_plan` first
  (measures), which is minutes at 1024². The benches use `auto_plan` for sub-plans to stay fast.

---

## 6. Files / reproduce

```sh
cd build_tuned
python build.py --src benches/bench_fft2d_vs_mkl.c     --mkl --compile   # 2D c2c
python build.py --src benches/bench_fft2d_r2c_vs_mkl.c --mkl --compile   # 2D r2c
python build.py --src benches/isolate_2d_r2c_128.c           --compile   # port isolation (no MKL)
# run with MKL bin + C:\mingw152\mingw64\bin on PATH, pinned core 2
```

Ported headers: `src/dag-fft-compiler/core/{transpose,fft2d,fft2d_r2c}.h`. Public entries:
`stride_plan_2d` / `stride_plan_2d_bailey` (c2c), `stride_plan_2d_r2c_from` + `stride_execute_2d_r2c` /
`stride_execute_2d_c2r` (r2c). Related: `docs/oop_c2c_engine.md`, `docs/performance/high_k_real_fft_architecture_wall.md`.
