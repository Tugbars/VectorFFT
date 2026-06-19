# Trig / DSP transforms: multithreading vs FFTW3

**Date:** 2026-06-19
**Hardware:** Intel i9-14900KF (Raptor Lake), AVX2, 8 P-cores (0–7), single CCX
**Toolchain:** mingw gcc 15.2; FFTW3 from vcpkg (**single-threaded build** — no `fftw3_threads`)
**Transforms:** DCT-I/II/III/IV, DST-I/II/III, DHT; batch K=2048, lane-batched split layout

> TL;DR — The full trig/DSP family is now multithreaded on the dag tree and **correct** (matches FFTW to
> 4–5e-15, MT output bit-identical to single-thread). dag beats FFTW **single-threaded** on every transform
> (1.4–3.0×) and **2.7–6.4× at 8 threads**. Seven of eight scale 1.9–3.4× (the inner real-FFT threads +
> pre/post K-split); **DCT-IV** is the lone partial case (1.35×) because its inner is a *complex* FFT routed
> through the serial compat wrapper — full MT for it is a documented follow-up.

---

## 1. Results (K=2048, block_K=256, caller pinned core 0)

Bench: `build_tuned/benches/bench_trig_mt_vs_fftw.c`. Timings are best-of wall-clock (ns/transform-batch).

| transform | inner | err vs FFTW | MT==T1 | dag T1 (ns) | dag T8 (ns) | FFTW (ns) | scaling | FFTW/dagT1 | FFTW/dagT8 |
|-----------|-------|-------------|--------|-------------|-------------|-----------|---------|------------|------------|
| DCT-II    | r2c   | 5e-15 | 0 | 1,327,375 | 512,429   | 3,052,143 | 2.59× | 2.30× | **5.96×** |
| DCT-III   | r2c   | 5e-15 | 0 | 1,333,254 | 492,821   | 3,133,679 | 2.71× | 2.35× | **6.36×** |
| DCT-IV    | c2c   | 5e-15 | 0 | 1,432,282 | 1,063,136 | 2,852,746 | 1.35× | 1.99× | **2.68×** |
| DST-II    | r2c   | 5e-15 | 0 | 2,249,729 | 654,204   | 3,165,454 | 3.44× | 1.41× | **4.84×** |
| DST-III   | r2c   | 5e-15 | 0 | 1,936,525 | 700,304   | 3,311,407 | 2.77× | 1.71× | **4.73×** |
| DHT       | r2c   | 5e-15 | 0 | 1,499,582 | 778,118   | 4,503,143 | 1.93× | 3.00× | **5.79×** |
| DCT-I     | r2c   | 4e-15 | 0 | 2,496,271 | 898,800   | 4,053,057 | 2.78× | 1.62× | **4.51×** |
| DST-I     | r2c   | 4e-15 | 0 | 2,553,207 | 855,554   | 4,014,746 | 2.98× | 1.57× | **4.69×** |

- **err vs FFTW** — relative max-abs diff between dag output and FFTW (REDFT/RODFT/DHT). 4–5e-15 confirms
  both numerical correctness *and* convention match (FFTW-unnormalized).
- **MT==T1** — max-abs diff of the 8-thread output vs the 1-thread output. **0 for all** = threading is
  bit-exact (no races, correct scratch partitioning).
- **scaling** = dag T1 / dag T8 (self-speedup on 8 P-cores).
- **FFTW/dagT1** > 1 ⇒ dag faster single-threaded; **FFTW/dagT8** = dag's 8-core throughput vs the
  (single-threaded) FFTW available here.

## 2. How the MT works

Each transform is a thin wrapper around an inner FFT with pre/post passes (Makhoul-style permute + twiddle
combine). All work is **K-parallel** (the K lane-batches are independent). The dag transforms thread in three
phases on the shared pool:

1. **Pre-process** — K-split across T workers (each owns a lane range `[k0,k1)`).
2. **Inner FFT** — for the r2c-based transforms (DCT-I/II/III, DST-I/II/III, DHT) the inner is a real FFT
   whose executor *threads internally* (block-parallel over K, `block_K<K`; see the r2c MT note). So the
   FFT phase parallelizes with no extra wiring.
3. **Post-process** — K-split across T workers.

The caller must `stride_set_num_threads(T)` **before plan creation** (T is snapshotted for scratch sizing and
for the inner r2c's block choice) and must build the inner r2c with a sub-K block (`block_K=K/8` here) so the
inner FFT actually splits. Pin the calling thread to core 0 (the pool pins workers to cores 1..T-1).

## 3. What changed this session

- **DCT-I / DST-I (`core/dct1.h`) had no MT** — they were the Phase-1 serial shells (the file's own header
  noted "MT K-split as in dct.h is the same follow-up"). Added: `_dct1_mt_threads`, K-split pre/post workers
  for both the even (DCT-I) and odd (DST-I) extensions and the Re/-Im extract, and three-phase dispatch in
  both executes; `n_threads` snapshot at plan-create. Now scale 2.78×/2.98× @T8, correct to 4e-15.
- **DCT-II/III, DCT-IV, DST-II/III, DHT already had the MT machinery** (`_*_mt_threads` + worker functions);
  this session validated them end-to-end at T8 and benched vs FFTW. The win was confirming the inner r2c
  threads when built with `block_K<K` (a `block_K=K` inner is a single block → serial).

## 4. The DCT-IV exception (partial MT — follow-up)

DCT-IV's inner is a **complex** FFT (size N/2 backward), executed via the serial compat `stride_execute_bwd`
on the full K batch in one call. Only its pre/post twiddle passes K-split, so it scales 1.35× (vs 1.9–3.4×
for the real-FFT-inner transforms). It is **correct** and still beats FFTW 2.68× at T8.

Full MT for DCT-IV needs the c2c inner to run per-worker on a lane slice (whole-transform K-split: each
worker does pre-slice → c2c-slice → post-slice). The slice executors in `proto_stride_compat.h` are currently
forward-only (`_stride_execute_fwd_slice_*`); a backward slice path (or routing the c2c through the
internally-threading executor) is the clean fix. Deferred — it's a real restructure best validated
deliberately, and DCT-IV is correct + already ahead of FFTW today.

## 5. Caveats / honest framing

- **FFTW here is single-threaded** (vcpkg build lacks the threads library). The fair single-thread fight is
  `FFTW/dagT1` (dag wins 1.4–3.0× everywhere). `FFTW/dagT8` is dag's 8-core throughput against the FFTW the
  user actually has — a real-world number, not an apples-to-apples MT comparison. A threaded FFTW build would
  narrow the T8 column; the T1 column is the layout/codelet comparison and stands on its own.
- **MKL is not a competitor here** — DFTI has no DCT/DST/DHT; FFTW (REDFT/RODFT/FFTW_DHT) is the reference.
- Sizes were chosen so inners factor over available radixes (DCT/DST/DHT N=256→inner 128; DCT-I N=257→M=512
  inner 256; DST-I N=255→M=512 inner 256). Other N follow the same structure.

## Appendix — reproduce

```sh
cd build_tuned
python build.py --src benches/bench_trig_mt_vs_fftw.c --fftw --compile
# run with vcpkg FFTW bin + mingw bin on PATH (caller pins core 0 internally)
PATH="/c/vcpkg/installed/x64-windows/bin:/c/mingw152/mingw64/bin:$PATH" ./benches/bench_trig_mt_vs_fftw.exe
```

Related: `docs/performance/high_k_real_fft_architecture_wall.md` (§6c — the r2c MT story this builds on),
project memory `trig_transforms_validated_on_dag.md` (single-thread validation) and `mt_2d_and_r2c_dag.md`.
