# `transforms/fft2d/` — 2D FFTs (c2c / r2c / c2r)

Two-dimensional transforms over an N1×N2 split-complex plane (`re[i*N2 + j]`,
`im[i*N2 + j]`). Unnormalized: `bwd(fwd(x)) = N1·N2·x`. Built entirely on the 1D
engine (`engine/` + `transforms/real/`) plus a SIMD transpose — no new butterfly math.

Full measured analysis: [`docs/fft2d_2d_port.md`](../../../../docs/fft2d_2d_port.md).

---

## 0. Design idea — row pass + column pass, kept in cache by tiling

A 2D FFT is 1D FFTs along each axis. The whole performance question is **how you feed the
row pass**: the rows are strided in memory (stride N2 between elements of a column), so a
naïve row FFT thrashes cache. The two classic answers are *transpose-then-batch* (Bailey)
vs *tile-and-transpose-locally* (tiled). VectorFFT defaults to **tiled B=8**, because small
tiles keep the working set in L1/L2 and our SIMD transpose makes the gather/scatter nearly
free — so the row pass costs little more than the raw transform.

The result: **2D c2c beats MKL 1.19–1.42×** (it's "just the transform," tiled and
L1-resident). 2D r2c is correct but loses ~0.63–0.69× — it inherits the same split-layout
real-FFT pack/recombine tax as 1D r2c (§4).

---

## 1. Architecture — two phases, two row-pass methods

Both transforms share the same **column FFT** (native, batched K=N2 — columns are
contiguous-strided, so no transpose is needed). They differ only in the **row pass**:

**Tiled (default, `FFT2D B=8`)** — for each tile of B rows:
1. gather B rows → scratch via SIMD transpose (B×N2 → N2×B)
2. N2-point FFT with K=B on the scratch (small batch, L1-resident)
3. scatter scratch back via SIMD transpose (N2×B → B×N2)

**Bailey (alternative)** — two full-matrix transposes bracket one large-K row FFT:
1. transpose N1×N2 → N2×N1
2. N2-point FFT with K=N1
3. transpose back

Tiled B=8 beats both Bailey and MKL at every tested size (32²–1024²): the full-matrix
transposes in Bailey touch the whole plane twice, while tiling streams it once in
cache-resident chunks.

---

## 2. The transpose engine (`transpose.h`)

The reason tiling is cheap. A multi-regime, cache-oblivious **SIMD transpose** of split-complex
doubles, auto-selected per ISA:

- **Kernel A — 4×4 AVX2** (compute-bound): classic unpack/`permute2f128`, for L1-resident
  problems and tails.
- **Kernel B — 8×4 AVX2 → 4×8 line-filling**: two stacked 4×4 transposes whose outputs write
  side-by-side so each dest row gets two adjacent 32-byte stores = **one full 64-byte cache
  line**. Peak 8 YMMs live.
- **Kernel C — 8×8 AVX-512 line-filling**: three-stage shuffle pipeline (`unpack_pd` →
  `permutex2var_pd` → `shuffle_f64x2`), one dest row = one 64-byte line.

Both AVX2 and AVX-512 variants beat their matching-ISA `mkl_domatcopy` on power-of-2 sizes ≥128
(single-thread). This is what makes the tiled gather/scatter "nearly free" — and why 2D c2c wins.

---

## 3. 2D c2c — beats MKL

`stride_plan_2d(N1, N2, reg)` → a `stride_plan_t`; in-place `stride_execute_fwd/bwd(plan, re, im)`.
The inner row/col plans are built internally (exhaustive-then-auto, §7).

| size | roundtrip | vfft (cyc) | MKL (cyc) | speed |
|------|-----------|-----------|-----------|-------|
| 64²  | 6.6e-15 | 18,235 | 25,645 | **1.41×** |
| 128² | 1.2e-14 | 87,833 | 113,920 | **1.30×** |
| 256² | 7.4e-15 | 437,863 | 522,695 | **1.19×** |
| 512² | 1.1e-14 | 2,583,840 | 3,209,732 | **1.24×** |

(N=256² single-thread, split. Reproduce: `bench_fft2d_vs_mkl.c`.)

---

## 4. 2D r2c / c2r — correct, loses single-thread

`fft2d_r2c.h`, FFTW convention (reduce along the inner axis):

- **Forward (r2c):** N1·N2 reals → N1×(N2/2+1) complex. Phase 1 = tiled **r2c** row pass
  (transpose → 1D r2c N=N2 K=B → transpose split output). Phase 2 = **padded** column c2c
  (K_pad = round-up(N2/2+1, 4)) plus a **col digit-reversal perm** so the half-complex column
  bins land where the col-FFT expects them.
- **Backward (c2r):** column c2c IFFT, then a tiled c2r row pass processed in **reverse tile
  order** (scatter writes longer rows than gather reads → reverse avoids clobbering future
  tiles' input).
- Builder: `stride_plan_2d_r2c_from(N1, N2, B, K_pad, plan_r2c, plan_col)` (caller supplies the
  inner r2c(N2,B) and c2c(N1,K_pad) plans, which the 2D plan then owns). Execute via
  `stride_execute_2d_r2c` / `stride_execute_2d_c2r`.

| size | roundtrip | speed vs MKL |
|------|-----------|--------------|
| 64²  | 1.5e-14 | 0.69× |
| 256² | 1.5e-14 | 0.69× |
| 512² | 1.8e-14 | 0.63× |

**Why c2c wins but r2c loses** — the same split-layout story as 1D ([real/README §4](../real/README.md)).
c2c is just the transform; r2c adds the pack/recombine memory passes *plus* the K_pad-padded column
FFT and the perm remap → memory-bound where MKL's fused real-FFT wins. Correct and shipped; the
perf gap is the same fused-kernel workstream as 1D r2c.

---

## 5. Multithreading — 2D c2c beats MKL 2.2–6.1× at 8 threads

Both phases are embarrassingly parallel — **no barriers**:
- **Phase 1 (rows):** tile-parallel — tiles distributed across threads, each with its own scratch.
- **Phase 2 (columns):** the C2C executor's built-in K-split (K=N2, or K=N2/2+1 for r2c).

Measured, 2D c2c, dag on 8 P-cores (caller pinned core 0, pool pins workers 1..7) vs MKL at
`mkl_set_num_threads(8)`, split / NOT_INPLACE (`bench_fft2d_mt_vs_mkl.c`):

| size | dag T1 (ns) | dag T8 (ns) | MKL T1 (ns) | MKL T8 (ns) | dag scal | MKL scal | T1 mkl/dag | **T8 mkl/dag** |
|------|-------------|-------------|-------------|-------------|----------|----------|------------|----------------|
| 64²  | 16,505      | 21,524      | 26,745      | 130,810     | 0.77×    | 0.20×    | 1.62×      | **6.08×**      |
| 128² | 92,819      | 81,178      | 117,921     | 270,271     | 1.14×    | 0.44×    | 1.27×      | **3.33×**      |
| 256² | 383,496     | 266,278     | 484,074     | 675,276     | 1.44×    | 0.72×    | 1.26×      | **2.54×**      |
| 512² | 2,405,716   | 1,499,525   | 3,113,353   | 3,285,215   | 1.60×    | 0.95×    | 1.29×      | **2.19×**      |

**The honest read — two effects compound.** (1) Our tile-parallel decomposition scales *modestly*
but positively (0.77–1.60× T8/T1 — 64² is too small to thread and regresses; the gain grows with
size as the row tiles dominate). (2) **MKL's 2D threading is actively bad here: 0.20–0.95×** — for
every size ≤512² MKL's T8 is *slower* than its T1 (threading overhead > benefit; 64² is 5× slower).
Net, **dag wins 2.19–6.08× at T8**, widening the 1.26–1.62× single-thread c2c lead. Same pattern as
1D r2c §6c: in the batched/2D regime MKL barely threads, and our barrier-free layout wins decisively
even at modest self-scaling.

(2D r2c MT is more limited still — the column pass and c2r backward carry more serial structure;
its single-thread number also starts behind, §4.)

> **Sweep stops at 512².** 1024² is omitted because `stride_plan_2d` builds its inner col/row plans
> via `vfft_proto_exhaustive_plan` (a live measure, §7) rather than from wisdom — at N=1024 that is
> minutes per plan. Making the 2D c2c inners wisdom-driven (one-time calibration of the (1024,1024)
> and (1024,B) cells) is the follow-up that makes 1024² practical.

---

## 6. Output order

Correctness is by **roundtrip** (`fwd→bwd == N1·N2·x`, all cells ≤1.8e-14) — definitive and
order-agnostic. The dag 1D engine is **digit-reversed (DIT)**, so 2D forward output is
**scrambled order** (does not match MKL elementwise) — the dag convention. Roundtrip pipelines,
convolution (same-plan pointwise multiply), and spectral-magnitude users are unaffected. A consumer
needing natural order pays a reorder pass; `fft2d_r2c.h` already carries the col digit-reversal perm
as the hook, but finishing natural-order output is a follow-up.

---

## 7. Gotchas

- **Exhaustive-first is slow at large N.** `stride_plan_2d` tries `vfft_proto_exhaustive_plan`
  (measures) before falling back to `auto_plan` — minutes at 1024². Prefer wisdom-driven inners or
  `auto_plan` for the sub-plans when planning large planes.
- **N2 must be even** (inherits the 1D r2c even-N constraint).
- **MKL 2D r2c CCE buffer** needs the **full N1×N2 complex footprint**, not N1×(N2/2+1) —
  undersizing it segfaults (a bench crash that once looked like a port bug; isolation proved the
  port crash-free).
- **Slice-helper clash:** including both `stride_executor.h` and `proto_stride_compat.h` redefines
  `_stride_execute_fwd_slice_from/_until` — use the compat ones (the dag convention).

---

## 8. File map

| file | role |
|------|------|
| `fft2d.h` | 2D c2c — tiled (default B=8) + Bailey row methods, native column FFT, `stride_plan_2d` |
| `fft2d_r2c.h` | 2D r2c / c2r — tiled real row pass + padded column c2c + col digit-reversal perm |
| `transpose.h` | multi-regime SIMD transpose (4×4 / 8×4 AVX2, 8×8 AVX-512), line-filling; beats `mkl_domatcopy` |

Benches: `bench_fft2d_vs_mkl.c` (c2c ST), `bench_fft2d_r2c_vs_mkl.c` (r2c ST),
`bench_fft2d_mt_vs_mkl.c` (c2c dag-T8 vs MKL-T8, §5), `bench_2d_mt.c` (dag MT scaling).
