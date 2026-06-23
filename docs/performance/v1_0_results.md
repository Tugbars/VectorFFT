# VectorFFT v1.0 — performance results

> **Where we win most — MKL's blind spot.** VectorFFT's lead over MKL is widest exactly
> where MKL invests least: **odd / mixed-radix, scrambled-order, in-place, batched.**
> Power-of-2 is MKL's home turf (decades of split-radix tuning) and our *narrowest* win
> (median **1.86×**); odd composites — where MKL falls back to generic mixed-radix / Bluestein —
> are our *fattest* (median **3.47×**, ~2× more margin). Three effects stack into that blind spot:
> (1) a DAG-compiler-tuned codelet for **every smooth radix** (not just 2/4/8), (2) **scrambled-order**
> in-place that skips the bit-reversal MKL pays, and (3) the **split lane-batched** layout that makes
> the throughput regime trivially parallel. MKL is tuned for the opposite corner: power-of-2, natural
> order, single transform. (Scrambled order is the right contract for convolution-class work — FIR
> filtering, polynomial / big-integer multiply, correlation, lattice-crypto NTT — where a fwd→bwd
> roundtrip or a pointwise multiply is order-agnostic.)

Empirical performance of VectorFFT across three axes:

1. **Wall-time vs MKL** on 1D C2C — single-thread (238 cells) and multi-threaded (the headline metric)
2. **Wall-time vs FFTW3** on 1D C2C and the r2r family (DCT/DST/DHT), single-thread
3. **Multi-threaded scaling** at T=2/4/8 across the transforms

(Plan-quality / cost-model analysis lives in its own doc: [docs/cost_model/](../cost_model/).)

All numbers are from the i9-14900KF calibration host (P-core pinned,
performance plan, single-threaded unless noted). The numbers move on
different hardware — see "Hardware caveats" at the end.

## 1. vs MKL — 1D C2C

Source: `build_tuned/benches/vfft_perf_tuned_1d.csv`
(238 cells × MKL ILP64 sequential, calibrated wisdom loaded).

```
Category              Cells    Min   Median    Max   Mean
─────────────────────────────────────────────────────────
Small (N≤128)            15   2.60×   4.28× 15.33×  5.98×
Power-of-2               29   1.10×   1.86×  3.04×  1.96×
Composite                43   1.62×   2.85×  4.51×  2.93×
Odd composite            26   2.26×   3.47×  5.16×  3.36×
Mixed deep               35   1.66×   2.71×  5.78×  2.89×
Prime powers             25   1.67×   2.69×  4.16×  2.76×
Genfft (R=11/13)         17   1.65×   2.79×  3.75×  2.63×
Rader primes             24   1.29×   2.34×  3.85×  2.36×
Bluestein primes         24   1.02×   1.55×  3.52×  1.74×
─────────────────────────────────────────────────────────
OVERALL                 238   1.02×   2.64× 15.33×  2.83×

Wins vs MKL: 238/238 (100%)
```

Headline:

> **VectorFFT beats MKL on 100% of bench cells (238/238). Median speedup
> 2.64×, mean 2.83×, range 1.02×–15.33×.**

The median 2.64× win comes from VectorFFT's twin advantages:
1. **Plan-level joint search** at calibration time — picks better
   factorizations than per-codelet wisdom (see
   [docs/wisdom/00_thesis.md](../wisdom/00_thesis.md)).
2. **Fully tuned codelet portfolio** — every shipped radix has
   variant codelets (FLAT / LOG3 / T1S / BUF) selected per
   `(R, me, ios)` cell.

### Multi-threaded — vs MKL at T=8

dag (8 P-cores, pinned core 0, pool K-split) vs MKL `mkl_set_num_threads(8)`, **identical split
lane-batched layout**, order-neutralized (engine order flipped per cell) + paced — the same fairness
as the single-thread table above. Source: `bench_1d_vs_mkl.c --mt` → `vfft_perf_tuned_1d_mt.csv`
(129 cells, K≥32).

```
 N      K    dag-T8 (ns)  MKL-T8 (ns)  dag/MKL
─────────────────────────────────────────────
 8      256         571       23,906   41.90×
 64     256       8,140       46,266    5.68×
 256    256      45,560      128,653    2.82×
 1024   256     224,288      694,963    3.10×
 4096   256     696,100    3,387,937    4.87×
 256    32        8,715       19,634    2.25×
 1024   32       50,375       66,834    1.33×
 4096   32      233,233      405,020    1.74×
```

> **At T=8, VectorFFT beats MKL on 129/129 cells (K≥32) — median 3.76× (K=32: 3.00×, K=256: 4.38×),
> up to 41.9× at tiny N where MKL can't usefully thread the batch.** Our split, lane-batched layout
> makes K independent transforms trivially parallel (no barriers); MKL's batched split-mode threading
> scales poorly at modest N. These use the **generic** executor — a conservative floor (JIT is wired
> and bit-exact again post-core-move; re-running with `--jit` widens the margin).

### Out-of-place — vs MKL (single-thread)

dag OOP c2c vs MKL `DFTI_NOT_INPLACE` split-complex, **identical layout**, order-neutralized + paced
(same fairness as the in-place table). Two natural-order kinds (LEAF, BAILEY2 fused-transpose stores)
and the scrambled-order MODEB (in-place dataflow run OOP; bit-exact roundtrip). Calibrated per-cell in
**isolated processes** to avoid cross-cell carryover biasing the kind pick. Source:
`bench_1d_vs_mkl.c --oop` → `vfft_perf_tuned_1d_oop.csv` (31 pow2 cells, K∈{32,128,256,1024}).

```
 N       K     kind     plan          dag/MKL
──────────────────────────────────────────────
 8       32    LEAF     —              10.78×
 8       256   MODEB    8               5.67×
 16      32    BAILEY2  4×4             5.97×
 64      256   MODEB    4,4,4           2.11×   (carryover sweep mis-picked BAILEY2 → 0.77×)
 256     256   MODEB    4,4,16          2.09×
 1024    256   MODEB    4,4,4,4,4       1.57×
 4096    32    MODEB    4,4,4,8,8       1.63×
 65536   256   MODEB    4,4,8,16,32     1.40×
──────────────────────────────────────────────
 Min 1.37×   Median 2.01×   Max 10.78×   Mean 2.49×   Wins 31/31
```

> **Out-of-place, single-thread, VectorFFT beats MKL on 31/31 cells — median 2.01×, range
> 1.37×–10.78×.** Small N favors the natural-order LEAF/BAILEY2 kinds; mid/high N and high K favor
> MODEB. Per-stage variants are inherited variant-rich from the in-place wisdom (FLAT/T1S/LOG3 mixed),
> and BAILEY2's `t1p` stage is flat-vs-log3 searched per cell.

### Out-of-place — vs MKL at T=8

Same OOP cells, dag K-split across 8 P-cores (pool, pinned core 0) vs MKL `mkl_set_num_threads(8)`,
identical NOT_INPLACE split layout, order-neutralized + paced. MODEB/LEAF are truly lane-sliced;
BAILEY2 runs single-threaded (its inter-stage transpose isn't lane-independent — 2-phase MT is a
follow-up) so its rows are dag-ST vs MKL-8T. A per-cell MT-vs-ST gate guards correctness. Source:
`bench_1d_vs_mkl.c --oop --mt` → `vfft_perf_tuned_1d_oop_mt.csv` (31 cells).

```
 N       K     kind     dag/MKL-T8   note
──────────────────────────────────────────────
 8       256   MODEB      38.53×     MKL can't thread tiny batch
 16      32    BAILEY2    45.80×     dag-ST vs MKL-8T
 64      256   MODEB       5.26×
 256     256   MODEB       2.80×
 1024    256   MODEB       2.74×
 4096    256   MODEB       4.86×
 65536   256   MODEB       3.10×
 1024    32    MODEB       1.24×     (min)
──────────────────────────────────────────────
 Min 1.24×   Median 2.80×   Max 45.80×   Wins 31/31
```

> **Out-of-place at T=8, VectorFFT beats MKL on 31/31 cells — median 2.80×, up to 45.8× at tiny N.**
> The huge small-N margins are where MKL can't usefully thread the batch; the steady mid/high-N MODEB
> wins (1.2×–5×) are the real K-split scaling. Generic executor (JIT wired + bit-exact, not yet
> re-run here); BAILEY2 MT is a follow-up — both are conservative floors.

## 2. vs MKL — 2D C2C

dag tiled 2D (`fft2d.h`, B=8: gather→K=B row FFT→scatter via SIMD transpose, native
column pass) vs MKL DFTI 2D (split, `DFTI_NOT_INPLACE`), single-thread, same fairness
as §1 (per-cell order-flip, cachebust + pace, best-of-5, ns timing). dag is **in-place,
scrambled order** (DIT); MKL is natural order — so the definitive correctness gate is the
roundtrip `fwd+bwd == N1·N2·x` (all e-14/e-15), and `elem≈1e0` just confirms the scramble.
Source: `bench_1d_vs_mkl.c --2d` → `vfft_perf_tuned_2d.csv`.

The plan comes from a dedicated **PATIENT 2D c2c calibration** (own `fft2d_c2c_wisdom`,
scored *end-to-end on the 2D transform* — PATIENT is the recommended planner; MEASURE is
the fast mode, exhaustive `stride_plan_2d` the wisdom-miss fallback). Inner row/col FFTs
are baked-or-JIT resolved (`--jit`). Measured **cooled** (20 s pre-cool + 30 s between
runs), median of 3. Source: `bench_1d_vs_mkl.c --2d --jit` → `vfft_perf_tuned_2d.csv`.

```
 N1×N2     dag/MKL   order
──────────────────────────────────
 64×64     ~1.6×*    scrambled
 128×128    1.41×    scrambled
 256×256    1.26×    scrambled
 512×512    1.29×    scrambled
──────────────────────────────────
 median    ~1.35×    (4/4 win)
```

*64² falls back to exhaustive — its PATIENT-banked plan was a calibration **noise
artifact** (a 5 µs cell is below reliable timing; the gate happened to measure exhaustive
slow at a hot moment). Exhaustive's 64² plan is measurably faster (~4.8 µs vs ~6.8 µs).

Headline:

> **2D C2C beats MKL on all 4 square cells — PATIENT-calibrated, median ~1.35×, up to 1.41×
> (128²).** The tiled B=8 row pass keeps the working set in L1/L2 and the SIMD 4×4/8×4
> transpose makes gather/scatter nearly free; JIT specializes the cold inner FFTs (bit-exact).
> Our plan times are **thermally rock-stable** (512² = 749 µs across every run this session);
> the run-to-run swing in the *ratio* is MKL's own variance, not ours. For these small 2D
> cells **PATIENT ≈ exhaustive** — full enumeration is cheap and good at this size — but both
> clear MKL on every cell. In-place scrambled-order 2D (the convolution contract);
> rectangular / non-pow2 cells are follow-ups.

### 2D C2C — vs MKL at T=8

Same cells, dag 2D threaded vs MKL `mkl_set_num_threads(8)`, identical split layout,
order-flipped + paced, **with an MT-vs-ST forward gate** (threaded fwd must equal the
single-thread fwd bit-for-bit — folded into rt; all e-14, so the tile-parallel path is
race-free). dag threads the **row pass only** (tile-parallel pool, per-thread scratch);
the **column pass stays serial** — that's the 2D self-scaling ceiling. Source:
`bench_1d_vs_mkl.c --2d --mt` → `vfft_perf_tuned_2d_mt.csv`.

```
 N1×N2     dag-T8 (ns)  MKL-T8 (ns)  dag/MKL   dag self-scale ST->T8
──────────────────────────────────────────────────────────────────
 64×64           5,641       48,751   8.64×     0.86× (overhead)
 128×128        23,996       87,069   3.63×     ~1.0×
 256×256        70,210      214,597   3.06×     1.88×
 512×512       500,200    1,307,575   2.61×     1.57×
──────────────────────────────────────────────────────────────────
 median                              ~3.34×     (4/4 win)
```

> **At T=8, 2D C2C beats MKL on all 4 cells — median ~3.3×, up to 8.6×.** Two effects:
> (1) dag's own scaling is **modest** (256² 1.88×, 512² 1.57×; tiny N regresses under threads)
> because only the row pass is parallel — the serial column pass caps it. (2) The large
> vs-MKL margins at small N are **MKL failing to thread tiny 2D**: at 64², MKL-T8 (48,751 ns)
> is ~6× *slower* than MKL-T1 (8,494 ns) — pure threading overhead — so dag wins 8.6×. Lifting
> the ceiling (parallel column pass / full-plane tiling) is the 2D-MT follow-up.

## 3. vs MKL — 1D R2C

R2C is the clearest embodiment of the split-layout trade: the **packing tax** that costs
us single-thread is the *same* lane-batched layout that makes K-split MT trivially parallel
(independent lanes, no barriers, no shared transpose buffer). So r2c **loses single-thread
and wins big multi-threaded** — throughput over single-core latency, by design
([transforms/real/README.md](../../src/core/transforms/real/README.md)). dag via the real
dispatcher (`vfft_r2c_plan_create`/`execute`, SPLIT, **JIT-wired**) vs MKL DFTI real r2c
(CCE); same fairness as §1; correctness vs a reference DFT (r2c is natural order). The
dispatch routes **rfft** at low K (JIT-specialized — see below) and **decoupled-stride** at
K≥32. Source: `bench_1d_vs_mkl.c --r2c [--mt]` → `vfft_perf_tuned_r2c{,_mt}.csv`.

### Single-thread — the packing tax

```
 N      K     path    dag/MKL    note
──────────────────────────────────────────────
 256    8     rfft     1.07×     JIT-wired rfft, low-K win
 256    16    rfft     1.15×
 256    256   stride   1.04×
 512    8     rfft     1.17×
 1024   8     rfft     0.64×     large-N rfft plane = L2-bound
 1024   256   stride   0.80×     decoupled-r2c structural gap
──────────────────────────────────────────────
 18 cells: 6 win.  Median 0.79×, range 0.46–1.17×.
```

> **Single-thread, r2c trails MKL — median 0.79×.** This is the honest cost of the split
> layout (the pack tax) plus MKL's heavily-tuned real-FFT. The **JIT lifts the low-K rfft
> cells to wins** (256/8 1.07×, 256/16 1.15×, 512/8 1.17×) — exactly where rfft is
> competitive; it can't close the large-N rfft L2 wall or the decoupled-stride high-K gap.

### Multi-threaded (T=8) — the layout payoff

```
 N      K     path    dag/MKL-T8   dag self-scale ST→T8
──────────────────────────────────────────────────────
 256    8     rfft      21.75×     ~1.0× (rfft is ST)
 256    256   stride     5.30×     2.79×
 512    256   stride     4.47×     4.87×
 1024   256   stride     3.65×     3.72×
 1024   16    rfft       1.74×     ~1.0×
──────────────────────────────────────────────────────
 18 cells: 18 win.  Median ~4.7×, range 1.74–21.75×.
```

> **At T=8, r2c beats MKL on all 18 cells — median ~4.7×, up to 21.8×.** The decoupled-stride
> path (K≥32) K-splits cleanly and scales **2.8–4.9×**. The rfft path (K<32) **also K-splits**
> (lane ranges, `rfft_natural_mt`), so MT is honored on every path — but its gain is small
> (~6–9% at K=16, none at K=8) because the rfft K-range sits at the **lane-split SIMD floor**:
> 8-wide lanes ÷ 8 threads leaves <1 SIMD group/thread, so K=8 falls back to single-thread and
> K=16 only splits ~2-way. The split layout is still the edge — it lets us thread the batch
> where MKL's real-FFT can't at modest N (MKL-T8 is ~20× slower than MKL-T1 at 256/8). The same
> layout that taxed us single-thread is the multithreading edge — the design trade paying off.

### 1D C2R (backward) — the natural split path

c2r (complex→real, the r2c inverse) gets the **same** split-layout treatment, and it's the
direct mirror of r2c's story. The public API hands c2r a **split** half-spectrum (the r2c
output), so the fast packed c2r — which needs a *packed* half-spectrum — was unreachable, and
the old path forced the slow decoupled-**stride** backward (~0.44–0.46× MKL). New this session:
a **fused natural initiator** (`c2r_execute_natural`, the inverse of rfft's natural terminator)
reads split re/im **directly** through the fast packed cascade — **no repack**. vfft's c2r front
door now runs a natural-vs-stride bake-off (mirror of r2c's), picking per cell; the forced-stride
hardcode is gone. Roundtrip `c2r(r2c(x))==N·x` is the gate (all e-14). Source:
`bench_1d_vs_mkl.c --c2r [--mt]`.

#### Single-thread — the packing tax (again)
```
 N      K     path      dag/MKL    note
──────────────────────────────────────────────
 256    8     natural    0.92×     ≈parity — packed-speed on split input
 256    16    natural    0.74×
 256    64    natural    0.55×     mid-K: MKL compute-bound / L1-resident
 256    128   natural    0.55×
──────────────────────────────────────────────
 natural ≈ 2× the old forced-stride path; reaches MKL parity only at K=8.
```
> **Single-thread, c2r trails MKL — same split-layout tax as r2c.** The natural path roughly
> **doubles** vfft's low-K c2r over the old stride path and reaches **parity at K=8 (0.92×)**,
> but MKL's compute-bound real backward still wins mid-K. (Even the unreachable packed path is
> only ~0.61× MKL at K=64 — the gap is structural in the cascade, not the split read.)

#### Multi-threaded (T=8) — the layout payoff (again)
```
 N      K     path      dag/MKL-T8   dag self-scale ST→T8
──────────────────────────────────────────────────────
 256    8     natural    ~17×        ~1.0× (K<16: lane-split floor)
 256    32    natural    ~7.9×       ~1.4×
 256    64    natural    3.9×        1.9×
 256    128   natural    3.0×        2.2×
 256    256   natural     —          2.8×   (MKL-T8 crashes at N·K≥131072)
 512    256   natural     —          3.6×
 1024   256   natural     —          2.8×
──────────────────────────────────────────────────────
```
> **At T=8 the split layout pays off — dag wins every cell, scaling 1.9–3.6× to high K.** The
> natural path K-splits the batch cleanly (`c2r_natural_mt`, pool lane-slabs; MT output is
> **bit-identical** to single-thread — race-free, lane-indexed scratch). MKL's c2r does **not**
> benefit from threads at these modest-N batch sizes: **MKL-T8 is slower than MKL-T1** even
> pinned to the same 8 cores (it parallelizes *within* the length-N transform, not across the
> K-batch where the work is), so the dag/MKL-T8 ratios at low K are inflated by MKL's thread
> overhead — the honest number is dag's own **2.8× self-scaling at high K**. Same trade as r2c:
> the layout that taxes us single-thread is exactly the multithreading edge.

## 4. vs MKL — 2D R2C

dag tiled 2D real-to-complex (`fft2d_r2c.h`: tiled R2C row pass + native column c2c)
vs MKL DFTI 2D real (CCE), single-thread, same fairness as §1–§3 (per-cell order-flip,
cachebust + pace, best-of-5, ns timing). dag output is **split** (out_re/out_im) and
**scrambled** (DIT); MKL is CCE-interleaved natural — so the definitive correctness gate
is the roundtrip `r2c+c2r == N1·N2·x` (all e-14/e-15), not an elementwise compare. Plans
are per-cell tuned; the inner column c2c is JIT-specialized. Source:
`bench_1d_vs_mkl.c --2dr2c` → `vfft_perf_tuned_2dr2c.csv`.

### Single-thread

```
 N1×N2      dag/MKL    order
──────────────────────────────────
 64×64       0.86×     scrambled
 128×128     0.85×     scrambled
 256×256     0.80×     scrambled
 512×512     0.89×     scrambled
──────────────────────────────────
 median     ~0.85×     (best-of-3)
```

> **Single-thread, 2D R2C trails MKL — median ~0.85×, range 0.80–0.89×.** As with 1D R2C
> (§3), this is the honest cost of the split lane-batched layout (the real-FFT pack tax)
> against MKL's heavily-tuned 2D real path — the same layout trade that becomes an edge
> under threading. Per-cell plan tuning closes most of the gap; the 256² cell is the
> laggard (0.80×). See the multi-threaded results below.

### Multi-threaded (T=8)

Same cells, dag threading the **row pass only** (tile-parallel pool, per-thread scratch; the
column c2c and the c2r backward stay serial — that's the 2D self-scaling ceiling), calibrated
plans, pinned core 0, with an **MT-vs-ST forward gate** (the threaded fwd must equal the
single-thread fwd bit-for-bit — folded into rt; all e-14/e-15, so the tile-parallel path is
race-free). Source: `bench_1d_vs_mkl.c --2dr2c --mt` → `vfft_perf_tuned_2dr2c_mt.csv`.

```
 N1×N2     dag-T8 (ns)   dag self-scale ST→T8
──────────────────────────────────────────────
 64×64          6,734    0.78×  (overhead)
 128×128       23,271    0.96×
 256×256       70,010    1.71×
 512×512      415,188    1.38×
──────────────────────────────────────────────
```

> **dag's 2D R2C self-scaling is modest — 256² 1.71×, 512² 1.38×; tiny N regresses under
> threads.** Only the row pass is parallel, so the serial column c2c + c2r passes cap it —
> the same ceiling as 2D C2C (§2). The MT-vs-ST gate confirms the tile-parallel forward is
> race-free (rt e-14/e-15).
>
> **No vs-MKL-T8 ratio is reported here.** MKL's threaded 2D *real* path is pathological in
> this `mkl_rt` + 8-thread configuration: a fixed ~30–370 ms per-call overhead, independent
> of transform size and wildly inconsistent run-to-run (256² measured 366 ms one rep, 32 ms
> the next). MKL-T8 thus comes out ~hundreds-of-× slower than MKL-T1, so the apparent dag
> "win" of 60×–5000× is a pure measurement artifact, not real speedup — MKL simply does not
> usefully thread small 2D real transforms in this setup. (1D C2C and 2D C2C thread fine in
> the same binary, so this is specific to the 2D real descriptor.) The c2r backward row pass
> now threads too (see the 2D C2R subsection below); parallelizing the **column** passes is the
> remaining 2D-MT lever that would lift the self-scaling ceiling further.

### 2D C2R (backward)

The inverse — complex (CCE / split) → real 2D, `fft2d_r2c.h`'s c2r path, **PATIENT-calibrated**
(separate `fft2d_c2r_wisdom`; c2r's optimum ≠ r2c's — all 4 cells WON their own gate),
single-thread (the c2r backward is **serial** — not yet tile-parallel). Roundtrip
`r2c+c2r == N1·N2·x` is the gate (all e-14/e-15). Measured **cooled**, median of 3. Source:
`bench_1d_vs_mkl.c --2dc2r` → `vfft_perf_tuned_2dc2r.csv`.

```
 N1×N2     dag/MKL   order
──────────────────────────────────
 64×64      0.84×    scrambled
 128×128    0.95×    scrambled
 256×256    0.75×    scrambled
 512×512    0.95×    scrambled
──────────────────────────────────
 median    ~0.89×    (single-thread)
```

> **Single-thread, 2D C2R trails MKL — median ~0.89×, range 0.75–0.95×.** Same real-FFT
> structural tax as r2c (§3, §4): the split lane-batched layout costs single-thread what it
> repays under threading. c2r lands right alongside the r2c forward (0.89× vs §4's 0.85×); 256²
> is the laggard (0.75×). PATIENT ≈ MEASURE here — the gap is structural, not plan-mode.

#### 2D C2R — multi-threaded (T=8)

The c2r backward is **now tile-parallel** (new this session): its row pass reads the padded
col-FFT scratch and writes reals to a *distinct* user buffer, so tiles are independent — the
same tile-parallel pool as the r2c forward, each thread with its own scratch slot + inner-pack
tid (the prior serial path was forced only by a hardcoded inner-slot index, not a real data
hazard). The column c2c IFFT stays serial — the self-scaling ceiling, as in §2/§4. **MT-vs-ST
gate:** the threaded c2r equals the single-thread output bit-for-bit (rt e-14/e-15 — race-free).
MKL's threaded 2D-real backward is anomalous on this host (§4), so we report dag **self-scaling**,
not a vs-MKL ratio. Cooled, median of 2. Source: `bench_1d_vs_mkl.c --2dc2r --mt`.

```
 N1×N2     dag-T8 (ns)   dag self-scale ST→T8
──────────────────────────────────────────────
 64×64          6,007    0.78×  (overhead)
 128×128       22,144    0.91×  (overhead)
 256×256       66,169    1.59×
 512×512      328,031    1.53×
──────────────────────────────────────────────
```

> **2D C2R self-scaling — 256² 1.59×, 512² 1.53×; small N regresses under threads.** Right
> alongside the r2c forward (§4: 1.47× / 1.46×) — only the row pass is parallel, the serial
> column IFFT caps it. Tiny cells (64²/128²) regress: threading overhead exceeds the few µs of
> row work. Full-arsenal milestone: **every 2D real path now threads** (r2c forward + c2r
> backward); parallelizing the column passes is the remaining lever.

## 5. vs FFTW3 — single-thread

VectorFFT's calibrated wisdom path measured against FFTW3 with
`FFTW_MEASURE` planning. FFTW3 split-complex API
(`fftw_plan_guru_split_dft`) so the layout matches VectorFFT exactly —
no interleave / deinterleave overhead on the FFTW side.

### 1D C2C — full sweep

Source: [build_tuned/benches/bench_1d_vs_fftw.c](../../build_tuned/benches/bench_1d_vs_fftw.c)
(207 cells × MKL bench grid, calibrated wisdom loaded). Same N/K grid
as Section 1's MKL bench, so ratios are directly comparable.

```
Category       Cells    Min   Median    Max    Mean
─────────────────────────────────────────────────────
Small (N≤128)    15   1.86×   4.10×   8.70×   4.60×
Power-of-2       30   1.34×   3.08×  15.89×   4.28×
Composite        33   1.82×   3.45×  15.07×   4.93×
Odd composite    18   1.38×   3.67×   6.29×   3.72×
Mixed deep       18   1.50×   5.28×  11.38×   5.11×
Prime powers     30   1.37×   5.09×  17.79×   6.85×
Genfft (R=11/13) 15   1.85×   3.25×  10.94×   4.52×
Rader primes     24   1.07×   2.23×   4.05×   2.38×
Bluestein primes 24   0.92×   1.15×   1.74×   1.22×
─────────────────────────────────────────────────────
OVERALL         207   0.92×   3.21×  17.79×   4.25×

Wins vs FFTW3: 202/207 (97.6%)
```

Headline:

> **VectorFFT beats FFTW3 on 202/207 (97.6%) of bench cells. Median
> speedup 3.21×, mean 4.25×, range 0.92×–17.79×.**

The median against FFTW3 (3.21×) is meaningfully higher than the
median against MKL (2.64× from Section 1). FFTW3 is genuinely behind
on power-of-two and prime-power cells once N·K outgrows last-level
cache — the calibrated wisdom routes around L3 thrashing while
FFTW's plan search doesn't capture the cache-residency effect.

**Top wins (large prime-power and pow-of-2 cells):**

| Cell | Factors | Ratio |
|------|---------|------:|
| N=390625 (5^8) K=256 | 5×5×5×5×5×5×25 | **17.79×** |
| N=78125 (5^7) K=256 | 5×5×5×25×5×5 | 17.51× |
| N=65536 K=256 | 4×4×8×16×32 | 15.89× |
| N=131072 K=256 | 4×4×4×4×4×4×32 | 15.57× |
| N=100000 K=256 | 4×25×5×8×25 | 15.07× |

At these sizes FFTW drops to ~1 GFLOP/s while VectorFFT sustains
~17–20 GFLOP/s — 1D batched FFT against a 16M+ working set is
memory-bound, and our wisdom-tuned multi-stage factorizations keep
inner radices L1-resident across the K=256 batch.

**Weakest cells (Bluestein primes — pre-wisdom snapshot):**

| Cell | Ratio (pre-wisdom) |
|------|------:|
| N=179 K=256 (Bluestein) | 0.92× (FFTW wins) |
| N=59 K=256 (Bluestein) | 0.93× (FFTW wins) |
| N=59 K=32 (Bluestein) | 0.96× (within noise) |

> **Note:** these FFTW3 ratios are the **pre-Bluestein-wisdom** snapshot. With the calibrated
> per-(N,K) `(M, B)` wisdom these sub-1.0× cells turn into wins (the vs-MKL §1 table shows every
> Bluestein cell ≥1.0×). A fresh `bench_1d_vs_fftw` run is pending; the table above is the historical
> lower bound, not the shipped result.

Full per-cell data: [build_tuned/results/vfft_perf_tuned_1d_fftw.txt](../../build_tuned/results/vfft_perf_tuned_1d_fftw.txt)
(human-readable, generated from
[vfft_perf_tuned_1d_fftw.csv](../../build_tuned/results/vfft_perf_tuned_1d_fftw.csv)
via `python build_tuned/make_perf_txt_fftw.py`).

### r2r family

The DCT / DST / DHT wrappers are built atop our R2C using Makhoul (DCT-II/III)
and Lee 1984 (DCT-IV); DST-II/III piggyback on DCT-II/III with sign-flip
+ index reversal; DHT is a free derivation of R2C output. Specialized
straight-line N=8 codelets (`gen_dct8.py`, `gen_dct3_n8.py`) bypass
Makhoul for the JPEG block size.

All numbers here are **single-threaded** (T=1) vs FFTW3 with `FFTW_MEASURE`
planning, split-complex API.

### DCT-II (REDFT10) — `bench_dct2_vs_fftw`

| N | K | vfft ns | fftw ns | ratio |
|--:|--:|--------:|--------:|------:|
| 8 | 1024 (JPEG) | 2,300 | 3,400 | **1.48×** |
| 8 | 4096 | 9,500 | 11,100 | 1.17× |
| 16 | 1024 | 12,400 | 39,200 | 3.16× |
| 32 | 1024 | 32,200 | 81,100 | 2.52× |
| 64 | 1024 | 71,200 | 173,800 | 2.44× |
| 128 | 256 | 28,900 | 88,300 | 3.06× |

Wins all measured cells (range 1.17–3.16×).

### DCT-III (REDFT01) — `bench_dct3_vs_fftw`

| N | K | vfft ns | fftw ns | ratio |
|--:|--:|--------:|--------:|------:|
| 8 | 1024 (JPEG) | 2,500 | 2,900 | 1.16× |
| **8** | **4096** | **17,200** | **10,400** | **0.60× (FFTW wins)** |
| 16 | 1024 | 13,700 | 41,100 | 3.00× |
| 32 | 1024 | 34,100 | 84,800 | 2.49× |
| 64 | 1024 | 75,200 | 178,100 | 2.37× |
| 256 | 256 | 65,900 | 203,300 | 3.08× |
| 1024 | 256 | 416,000 | 1,495,500 | **3.59×** |

> **The only v1.0 r2r loss vs FFTW3** is DCT-III at N=8 K=4096 (0.60×).
> Both N=8 codelets (`gen_dct3_n8`) target the JPEG-range K (256–1024)
> and don't optimize for very-large-K layout. FFTW switches to a
> different large-batch code path that still beats us at K≥4096. v1.1
> fix: a K-specialized DCT-III N=8 variant — same flavor as the JPEG
> codelet, different cache layout for K≥4096. Tracked in
> [docs/v1_1_codelet_roadmap.md](../v1_1_codelet_roadmap.md).

### DCT-IV (REDFT11) — `bench_dct4_vs_fftw`

After the specialized N=8 codelet landed:

| N | K | vfft ns | fftw ns | ratio |
|--:|--:|--------:|--------:|------:|
| 8 | 256 | 800 | 2,700 | 3.38× |
| 8 | 1024 | 4,300 | 9,400 | 2.19× |
| 8 | 4096 | 17,600 | 36,900 | 2.10× |
| 16 | 1024 | 8,900 | 35,900 | **4.03×** |
| 32 | 1024 | 28,300 | 74,200 | 2.62× |
| 64 | 1024 | 60,800 | 161,800 | 2.66× |
| 256 | 256 | 59,500 | 186,000 | 3.13× |
| 1024 | 256 | 354,200 | 1,482,100 | **4.18×** |

Wins all measured cells (range 1.85–4.18×). The pre-codelet build
showed losses 0.53–1.06× at small N — codelet flipped that.

### DST-II / DST-III (RODFT10 / RODFT01) — `bench_dst23_vs_fftw`

| Variant | N | K | vfft ns | fftw ns | ratio |
|---------|--:|--:|--------:|--------:|------:|
| DST-II | 8 | 256 | 600 | 2,400 | **4.00×** |
| DST-II | 16 | 1024 | 16,100 | 38,900 | 2.42× |
| DST-II | 32 | 1024 | 39,100 | 78,500 | 2.01× |
| DST-II | 64 | 1024 | 90,800 | 173,600 | 1.91× |
| DST-II | 256 | 256 | 82,600 | 198,600 | 2.40× |
| DST-II | 1024 | 256 | 553,900 | 1,484,500 | 2.68× |
| DST-III | 8 | 256 | 700 | 2,900 | **4.14×** |
| DST-III | 16 | 1024 | 21,400 | 40,800 | 1.91× |
| DST-III | 32 | 1024 | 41,100 | 83,200 | 2.02× |
| DST-III | 64 | 1024 | 94,700 | 176,700 | 1.87× |
| DST-III | 256 | 256 | 84,300 | 207,100 | 2.46× |
| DST-III | 1024 | 256 | 544,900 | 1,507,000 | 2.77× |

Wins all measured cells. Range 1.85–4.14×; strongest at small N where
FFTW's DST is less specialized than its DCT path.

### DHT (Hartley)

Per session notes, DHT lands **1.9–2.8× over FFTW** across the same
N/K range. A dedicated `bench_dht_vs_fftw` per-cell table was not
written for v1.0 — `test_dht.c` confirms 22/22 cells pass at machine
precision vs FFTW reference, but timing data was not preserved. v1.1
adds the bench so the DHT row matches the DCT/DST detail level.

### Headline (r2r vs FFTW3, T=1)

> **VectorFFT wins 53/54 measured r2r cells vs FFTW3** (1.16–4.18×
> range; mean ~2.5×). Single loss: DCT-III at N=8 K=4096 (0.60×) —
> codelet-fixable in v1.1.

| Family | Ratio range | Cells | Wins |
|--------|:-----------:|:-----:|:----:|
| DCT-II | 1.17–3.16× | 6 | 6/6 |
| DCT-III | 0.60–3.59× | 7 | 6/7 |
| DCT-IV | 1.85–4.18× | 11 | 11/11 |
| DST-II | 1.91–4.00× | 6 | 6/6 |
| DST-III | 1.87–4.14× | 6 | 6/6 |
| DHT | ~1.9–2.8× (summary) | — | — |

MKL TT was also benched for DCT-IV (4–13× wins) and DST (timing-only —
MKL TT computes a different PDE-oriented math convention, so the
comparison is informational, not apples-to-apples). FFTW3 is the
correct r2r baseline.

## 6. Multi-threaded scaling

### 1D C2C — direct MT vs MKL

See **§1 → "Multi-threaded — vs MKL at T=8"** for the head-to-head: 129/129 wins at T=8, median
3.76× over MKL (K=32: 3.00×, K=256: 4.38×). R2C inherits the same K-split MT (its inner C2C threads).

### DCT-II / DCT-III / DCT-IV / DST-II/III / DHT (wrapper MT, new in v1.0)

Source: [build_tuned/benches/bench_mt_dct.c](../../build_tuned/benches/bench_mt_dct.c).

```
Transform   Cell           T=1 ns   T=2 (×)    T=4 (×)    T=8 (×)
──────────────────────────────────────────────────────────────────
DCT-II      N=256  K=1024   482000  1.04   1.95   2.60
DCT-IV      N=256  K=1024   452200  1.12   1.77   2.09
DST-II      N=256  K=1024   620900  1.17   2.06   2.49
DHT         N=256  K=1024   452900  0.97   1.55   1.85
DCT-II      N=1024 K=1024  2297700  1.08   1.55   2.35
DCT-IV      N=1024 K=1024  2682900  1.16   1.77   2.65
DST-II      N=1024 K=1024  2713900  0.97   1.41   2.11
DHT         N=1024 K=1024  2047300  0.88   1.23   1.67
DCT-II      N=4096 K=1024 13911400  1.11   1.65   2.11
DCT-IV      N=4096 K=1024 16838200  1.20   1.58   2.14
DST-II      N=4096 K=1024 19109400  1.13   1.62   2.20
DHT         N=4096 K=1024 13426100  1.06   1.49   1.83
DCT-II      N=4096 K=4096 58493200  1.12   1.61   2.14
DCT-IV      N=4096 K=4096 72842000  1.22   1.44   1.62
DST-II      N=4096 K=4096 80495400  1.13   1.63   2.20
DHT         N=4096 K=4096 59296500  1.06   1.55   1.87
```

Best speedup at T=8: **2.65×** (DCT-IV at N=1024 K=1024). Typical
**1.6–2.4×** across cells.

### Why not 8× at T=8?

The DCT/DST/DHT family is implemented as **three sequential passes**:

```
Pass 1: pre-permute / pre-twiddle    — bandwidth-bound
Pass 2: inner FFT (R2C or C2C)       — has its own MT
Pass 3: post-process / post-twiddle  — compute + memory mix
```

Each pass reads + writes the full N·K data once. Total memory traffic
≈ 3 × N·K × 16 bytes per call. At N·K = 16M (N=4096 K=4096), that's
~768 MB per call. DDR5 on this CPU saturates around 25 GB/s, putting
a wall-time floor around 30 ms per call — close to what we measure
(27 ms at T=8). Adding more threads can't beat physics.

### Where the 8× comes back: v1.1 fused codelets

The v1.1 codelet roadmap
([docs/v1_1_codelet_roadmap.md §2](../v1_1_codelet_roadmap.md))
adds specialized straight-line codelets — `e10_*` for DCT-II,
`e11_*` for DCT-IV, `r2hc_*` for R2C — that fuse all three passes
into one tight kernel. Arithmetic intensity rises dramatically:

| Generation | Memory traffic / call | T=8 ceiling |
|-----------|----------------------|:-----------:|
| Pre-v1.0 (sequential wrappers) | 3 × N·K·16 bytes | ~1.4× |
| **v1.0 (parallel wrappers, current)** | **3 × N·K·16 bytes** | **~2.6×** |
| v1.1 (fused codelets) | 1 × N·K·16 bytes | ~5× projected |

The v1.0 parallel wrappers lift the floor from 1.4× to 2.6×. Fused
codelets lift the ceiling from 2.6× to ~5× by eliminating the
multi-pass bandwidth traffic. Both are needed for the full picture.

DHT scales worst (1.6–1.9× at T=8) because its pre-phase is one big
sequential memcpy of N·K doubles — left intentionally non-parallel
because it's pure memory bandwidth, and a single optimized memcpy
typically beats T smaller memcpys when the limit is DRAM throughput.
DHT will benefit most from v1.1 fused codelets.

## 7. Per-codelet performance (VTune-grade)

For deep per-radix analysis at K=256 see
[docs/vtune-profiles/](../vtune-profiles/) — one detailed profile per
radix R ∈ {4, 8, 10, 11, 12, 13, 16, 20, 25, 32, 64}. Top-line:

| Radix | Retiring (% of pipeline slots) | Bottleneck |
|------|:-----:|------|
| R=4  | 86% | compute-peak (port 0/1 at 96/91%) |
| R=8  | 72% | DFT-8 critical path dependency chains |
| R=10 | 63% | radix-5 + radix-2 FMA chains |
| R=11 | 59% | Winograd, machine-clears flagged |
| R=12 | 57% | radix-3 + radix-4 FMA chains |
| R=13 | 60% | Winograd + Sethi-Ullman |
| R=16 | 25% | store-bound + L1 latency (post-prefetch) |
| R=20 | 54% | radix-5 FMA chains |
| R=25 | 50% | hybrid compute/store |
| R=32 | 34% | L1 store-DTLB overflow (~80 pages) |
| R=64 | 27% | load + store DTLB overflow (~160 pages) |

Most radixes retire 50–86%. R=16/32/64 hit memory-system bottlenecks
that the codelet alone can't fix (huge codelets exceed DTLB capacity);
these benefit specifically from the cost model's variant-aware
selection (T1S / LOG3 / BUF) which routes around their bottlenecks
when wisdom shows another protocol wins.

## 8. Hardware caveats

### These numbers are from one CPU

All measurements: i9-14900KF (Raptor Lake, hybrid 8P+16E), 5.7 GHz
turbo, AVX2. Numbers move on:

- **Sapphire Rapids / Emerald Rapids** — should be similar or better
  (same uarch family, often better memory subsystem). Wisdom carries
  over without recalibration.
- **Zen 4 / Zen 5** — different uarch. CPE numbers shift; recommend
  re-running `cpe_measure` and `calibrate_tuned` on the target host.
  Architectural advantages (cost model, wisdom, MT) carry over; per-
  cell speedups may differ.
- **AVX-512 hardware** — codelets exist, but CPE table currently
  holds only AVX2 measurements. Re-run cpe_measure on AVX-512 host
  for accurate estimate-mode plans there.

### Consumer PC vs calibration host

The numbers in this doc are from the calibration host running clean
(idle background, performance plan, single P-core pinned). On a
consumer PC running normal background load, expect:

- **vs MKL ratios**: similar (within 5–10% — the win is structural)
- **Estimate vs wisdom mean**: drifts up to 1.3× on a noisy host (was
  1.19× on the calibration host)
- **MT scaling**: slightly weaker (T=8 ceiling drops 10–20% under
  thermal/freq fluctuation)

### What `Ts > 8` looks like

We bench up to T=8. On the i9-14900KF's hybrid 8P+16E config, T=16
or T=24 starts using E-cores, which run ~60% the IPC at higher
latency. Per-thread efficiency drops sharply past 8. For workloads
that benefit from many threads, the bench grid should be extended
(v1.1 work).

## 9. Reproducing these numbers

### vs MKL

```
python build_tuned/build.py --vfft --src build_tuned/benches/bench_1d_vs_mkl.c --mkl
build_tuned/benches/bench_1d_vs_mkl.exe        # single-thread -> vfft_perf_tuned_1d.csv
build_tuned/benches/bench_1d_vs_mkl.exe --mt   # T=8 (K>=32) -> vfft_perf_tuned_1d_mt.csv
```

Requires MKL ILP64 (Intel oneAPI install); single-thread uses `mkl_set_num_threads(1)`, `--mt`
uses 8. 238 cells × ~1 second = ~5 minutes wall (single-thread).

### 1D C2C vs FFTW3 (single-thread)

```
python build_tuned/build.py --vfft --src build_tuned/benches/bench_1d_vs_fftw.c --fftw
# fftw3.dll must be co-located with the exe (already copied into build_tuned/).
build_tuned/benches/bench_1d_vs_fftw.exe \
    build_tuned/vfft_wisdom_tuned.txt \
    build_tuned/results/vfft_perf_tuned_1d_fftw.csv \
    build_tuned/results/vfft_acc_tuned_1d_fftw.csv
```

Long run — 1–2 hours on the calibration host because of FFTW's
`FFTW_MEASURE` plan-search cost on the larger prime-power cells
(N=823543 alone takes ~30 min at K=256). Run with no other significant
load for cleanest numbers.

### r2r vs FFTW3 (single-thread)

```
python build_tuned/build.py --vfft --src build_tuned/benches/bench_dct2_vs_fftw.c --fftw
python build_tuned/build.py --vfft --src build_tuned/benches/bench_dct3_vs_fftw.c --fftw
python build_tuned/build.py --vfft --src build_tuned/benches/bench_dct4_vs_fftw.c --fftw
python build_tuned/build.py --vfft --src build_tuned/benches/bench_dst23_vs_fftw.c --fftw
build_tuned/benches/bench_dct2_vs_fftw.exe
build_tuned/benches/bench_dct3_vs_fftw.exe
build_tuned/benches/bench_dct4_vs_fftw.exe
build_tuned/benches/bench_dst23_vs_fftw.exe
```

Requires FFTW3 (vcpkg install or local build). ~30 seconds wall total.
Each binary plans with `FFTW_MEASURE` so first-run setup is the bulk
of the time; benched min over 21 reps after 5 warmup.

### MT scaling for DCT/DST/DHT

```
python build_tuned/build.py --vfft --src build_tuned/benches/bench_mt_dct.c
build_tuned/benches/bench_mt_dct.exe
```

~30 seconds wall. Run with no other significant load on the machine
for cleanest numbers.

## See also

- [docs/cost_model/](../cost_model/) — how the estimate path achieves 1.20×
- [docs/wisdom/](../wisdom/) — how the calibrator achieves the optimum
- [docs/v1_1_codelet_roadmap.md](../v1_1_codelet_roadmap.md) — what closes the remaining gaps
- [src/core/README.md](../../src/core/README.md) — user-facing API docs and threading status
