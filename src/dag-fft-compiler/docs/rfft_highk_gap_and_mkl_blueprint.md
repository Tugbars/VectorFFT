# The high-K real-FFT gap vs MKL: diagnosis, ruled-out levers, and MKL's blueprint

**Date:** 2026-06-18
**Platform:** i9-14900KF (Raptor Lake), AVX2-only, single P-core pinned, 5.69 GHz, MKL pinned to 1 thread (`mkl_set_num_threads(1)`), mingw gcc 15.2.
**Scope:** the dag-fft-compiler native real-FFT (`rfft`) forward, packed-halfcomplex output, K-batch split-complex.

---

## 0. TL;DR

- Our rfft is at **parity with MKL r2c at K=8** but loses **~2× by K=256** (a real-FFT-specific loss — our **1D C2C wins 238/238 vs MKL at all K**).
- VTune says we're **store-TLB / L2-latency bound** (74% memory-bound, 50.8% store-bound, 41.3% DTLB-store, **19.7% retiring**); MKL is **compute/L1-bound** (51.5% retiring, 5% memory-bound).
- **Two cheap levers were tested and decisively ruled out:** buffered-tiling (copy-to-contiguous) and 2 MB huge pages. Neither helps. The 41% "DTLB-store" was a VTune **overlap artifact** — fixing the TLB (huge pages, engaged) gave **zero** speedup.
- **Disassembling MKL** revealed the actual mechanism: MKL **decouples** the transform into a clean compute pass + a **fused Hermitian-terminator/twiddle/transpose pass that is L1-blocked and run twice** (per conjugate-symmetric half). We **fuse** butterflies with a page-scattered packed-store, so the store traffic pollutes the compute.
- **The fix is a real but de-risked redesign:** decouple our rfft the same way. Every piece already exists and is measured competitive — `transpose.h` already beats MKL's `domatcopy`; `stride_twiddle_transpose` is the fuse template; MKL's inner loop is a line-by-line reference for the conjugate combine.
- **Verdict: upgraded from "structural, accept it" → "concrete de-risked redesign worth a prototype."**

---

## 1. The gap, located and quantified

`build_tuned/benches/bench_r2c_fwd_vs_mkl.c` (N=256, factorization (8,32)), swept over K, MKL pinned to 1 thread. Ratio = `mkl_ns / ours` (<1 ⇒ MKL faster):

| K | mkl/ours | our position | working set |
|---|---|---|---|
| 8 | 0.995 | parity | L1/L2 |
| 32 | 0.690 | lose 1.45× | L2 |
| 64 | 0.544 | lose 1.84× | L2 |
| 128 | 0.620 | lose 1.61× | L2 |
| 256 | **0.516** | **lose ~1.94×** | ~1–2 MB (L2 edge) |

The loss grows with K and saturates near 2× once the plane is L2-resident. Confirmed isolated: `vtune_rfft.exe` measures **ours 80.6 µs vs MKL 39.5 µs** at N=256 K=256.

**It is real-FFT-specific.** Our 1D C2C wins MKL at K=256 (median 2.48×, 238/238 sweep). So the loss is not "high K is hard" — it is something about our *real-FFT* path specifically.

---

## 2. VTune diagnosis (uarch-exploration, N=256 K=256)

Target `build_tuned/benches/vtune_rfft.c`, runner `build_tuned/benches/run_vtune_rfft.ps1` (needs an **elevated** shell for hardware EBS).

| Top-Down | **ours (rfft)** | **MKL (r2c)** |
|---|---|---|
| Retiring | **19.7%** | **51.5%** |
| CPI | 0.833 | 0.313 |
| Back-End Bound | 77.9% | 45.5% |
| Memory Bound | **74.0%** | 5.2% |
| → Store Bound | **50.8%** | 0.0% |
| → → Store Latency | 67.5% | 16.1% |
| → → **DTLB-store** | **41.3%** | 0.9% |
| → Load DTLB | 25.9% | 0.9% |
| → DRAM Bound | 0.2% | 0.4% |
| Core Bound | 3.9% | **40.3%** |
| → Port 1 (FMA) | 22.0% | **62.9%** |

**Read:** MKL runs the FMAs near peak and barely touches memory. We spend 74% of cycles waiting on memory, half of it store-bound, dominated by store-TLB walks. **Neither is DRAM-bound** (0.2%) — so it is *not* bandwidth and *not* more flops. Our packed plane spans 128–384 × 4 KB pages and our output stride (`os = Q·m·K`) lands each store on a different page, thrashing the ~96-entry L1 DTLB.

---

## 3. Two cheap levers — ruled out

### 3a. Buffered-tiling (copy strided → contiguous scratch → unit-stride codelet → copy back)
Proxies `build_tuned/benches/buf_tiling_proxy.c` and `buf_tiling_amortized.c` (radix-8 stage). Even the amortized, L1-tiled version (gather an `[N×Kb]` slab once, S cascade passes, scatter once) **loses at K=256 in every config** (Kb ∈ {256,128,64,32}, S ∈ {1..4}; B/A = 1.28–3.79, never < 1.0).
Reason: a DIT butterfly's R legs sit Q rows apart, **spread across all N rows**, so any tile holding a stage's working set spans ~N rows = ~256 KB = L2, not L1. The gathered slab is L2-resident → the codelet is L2-bound just like strided → the copy is pure overhead. (The tempting "unit-stride is 4× faster" came from re-hammering a single 8 KB L1-hot buffer — not realizable in a real cascade.)

### 3b. 2 MB huge pages (env-gated `VFFT_RFFT_HUGE`, allocator in `core/rfft.h`)
Large pages **engaged** (plane + output + input all large-page backed) — but the timing was **unchanged**: 80/75 µs (off) vs 80/77 µs (on). The 41% DTLB-store metric was a **VTune overlap artifact**: those cycles are charged to DTLB-store *and* to the deeper store/L2-latency stall sitting behind them. Remove the TLB miss and the store still waits on L2, so the critical path does not move. (The allocator stays in `rfft.h`, default-off, byte-identical, opt-in for future use.)

**Conclusion:** the gap is not a knob. It is structural — *where* the work happens in the cache hierarchy.

---

## 4. MKL's blueprint, reverse-engineered from `mkl_avx2.2.dll`

VTune attributes MKL's time to two static AVX2 kernels (not JIT'd — they live in `mkl_avx2.2.dll`). VTune normalizes addresses to ImageBase 0x180000000, so `objdump --start-address=<VA>` hits them directly.

**Per-function memory breakdown (programmatic parse of the VTune CSV):**

| | `func@0x1825c3800` (COMPUTE) | `func@0x182549180` (PACK) |
|---|---|---|
| CPU time | 4.88 s | 3.25 s |
| Retiring | 60.1% | 44.7% |
| Memory Bound | 2.6% | **1.9%** |
| Store / L2 / DRAM Bound | 0.0 / 0.9 / 0.4% | **0.0 / 0.0 / 0.0%** |
| DTLB-store | 1.3% | **0.1%** |
| instruction mix | 228 add/sub/mul/fma vs 37 shuffles | 75 shuffles (vunpck/vperm2f128/vinsertf128) + combine |

**Both passes are L1-resident.** `func1` is clean butterflies; `func2` does all its data movement inside L1 — a naive 512 KB-plane transpose would be L2-bound; it isn't.

### 4a. `func2` is an L1-blocked transpose, done twice (read from the loop nest)
`func2` contains **two near-identical blocked-transpose routines** (`0x182549180→ret 0x182549480`, `0x182549494→ret 0x1825497a1`). Each is a two-level loop:
- **OUTER** stride `0x2000` = 1024 doubles (a row-tile), counter `+= 0x400`.
- **INNER** step `-0x40` = **64 B = one cache line** (8 doubles), bound `0x3f8` (1016 → ~127 iters), with the 4×4 register transpose inside.
- entry `cmp …,0xffff` = small/large tile dispatch (same idea as `transpose.h`'s L1/L2/L3 split).

So the transpose is **L1-blocked** (cache-line inner, row-tile outer) and **done twice** (the negative inner stride = the `N−k` conjugate partner walking down from the high end). This confirms the "done 2× L1-friendly" hypothesis *from the code*.

### 4b. The Hermitian fold is FUSED into the transpose (read from the inner loop)
`func2`'s inner loop is the **r2c Hermitian terminator + per-k twiddle + transpose, all fused**, walking conjugate pairs `(k, N−k)` from both ends:

```
load X[k] (fwd, rcx+r11*8) + X[N−k] (bwd, rsi)  → de-interleave to re/im
× 0.5
Hermitian split:  Xe = ½(X[k] + conj X[N−k]),  Xo = ½(X[k] − conj X[N−k])   (vadd/vsub)
twiddle:          W^k · Xo            (vmul / vfmadd231pd / vfmsub231pd; table at [r9+r11*8])
recombine:        out[k]   = Xe + W·Xo,   out[N−k] = Xe − W·Xo
transpose (vunpck/vperm2f128/vinsertf128) → store BOTH ends (rcx and rsi)
add r11,0x8 ; add rsi,-0x40 ; cmp edi,0x3f8 ; jb
```

### 4c. The full MKL r2c structure
```
func1: complex half-spectrum FFT butterflies  → contiguous intermediate   (L1, compute-bound, 60% retiring)
func2: Hermitian split + W^k twiddle + TRANSPOSE, FUSED, L1-blocked,
       over conjugate pairs (k, N−k), run twice                            (L1, no store/L2 penalty)
```

### What we do instead
Our `hc2hc` codelet **fuses butterflies with the packed Hermitian store**, and that store is **page-scattered** (`os = Q·m·K`). So the compute and the scatter share one loop, the store-TLB/L2 stalls starve the FMAs, and we retire 20% instead of 60%.

---

## 5. The fix: decouple (de-risked)

Reproduce MKL's structure:

1. **Compute pass** — run our butterflies writing a **contiguous** unit-stride intermediate (not the scattered `os = Q·m·K` packed layout). Compute-bound, L1-friendly stores. *(We have the butterfly codelets.)*
2. **Pack pass** — **Hermitian split + twiddle + transpose, fused, L1-blocked**, conjugate-pairs `(k, N−k)`, producing packed-halfcomplex. *(We have `core/transpose.h` — cache-oblivious recursive, line-filling AVX2 `_t8x4`, split-complex `stride_transpose_pair`, **already beats `mkl_domatcopy`** on pow2 ≥128. And `stride_twiddle_transpose` already fuses arithmetic into the transpose — the template for the combine. MKL's `func2` inner loop is a line-by-line reference for the `½(X[k]±conj X[N−k])` part.)*

**Why this is not the buffered-tiling that lost:** that copied the *input* to *feed* compute (naive memcpy, never amortized). This isolates the *output* pack *after* compute, using a tuned line-filling register transpose. Different structure, MKL-validated.

---

## 6. Next step (go/no-go)

A scoped **N=256 K=256 prototype**: restructure the cascade's final stage to emit a contiguous intermediate, run a `transpose.h`-based pack with the Hermitian/twiddle combine fused, and A/B against the fused packed path **and** MKL. If it lands near or under MKL → build out across the rfft radix set; if not → accept the gap with full confidence.

---

## 7. Reproduction

- Gap sweep: `build.py --src benches/bench_r2c_fwd_vs_mkl.c --mkl --compile`, run `bench_r2c_fwd_vs_mkl.exe <K>`.
- VTune (elevated): `build.py --src benches/vtune_rfft.c --mkl --compile`, then `run_vtune_rfft.ps1` → `summary_{ours,mkl}.txt` + `vt_rfft_{ours,mkl}/`.
- Buffered-tiling proxies: `benches/buf_tiling_proxy.c`, `benches/buf_tiling_amortized.c`.
- Huge pages: set `VFFT_RFFT_HUGE=1` (needs the "Lock pages in memory" privilege).
- MKL disasm: `vtune_rfft.exe` prints `mkl_avx2.2.dll` base; `func@0x1825c3800` (compute) / `func@0x182549180` (pack) disassemble directly via `objdump -d -M intel --start-address=… mkl_avx2.2.dll` (VTune VAs == ImageBase + RVA).
  - **Saved disassembly** (this analysis): `build_tuned/benches/mkl_r2c_func2_hermitian_transpose.asm` (the pack pass — two L1-blocked transpose routines + the fused Hermitian/twiddle inner loop) and `build_tuned/benches/mkl_r2c_func1_compute.asm` (the butterfly compute pass).
- Per-function metric CSV: `vtune -report hotspots -result-dir vt_rfft_mkl -group-by function -format csv`.

## 8. Cross-references
- `r2c_avx2_metal_findings.md` — earlier R2C-on-AVX2 characterization (Kb-blocking dead, model-b fusion regresses).
- `core/transpose.h` — the L1-blocked transpose (used by `core/fft2d.h`, which beats MKL 2D 9/9).
- 1D C2C vs MKL: 238/238 wins (separate result) — the contrast that proves this is real-FFT-specific.
